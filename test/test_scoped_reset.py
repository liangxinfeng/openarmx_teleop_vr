"""No node construction: scoped reset target and mapper ownership checks."""
import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openarmx_teleop_vr_306_v4.controller_node import IndependentArmController as Controller
from openarmx_teleop_vr_306_v4.vr_mapper_node import IndependentVrMapper as Mapper
from openarmx_teleop_vr_306_v4.reset_state import QuickResetState


def subject():
    params = {'quick_reset_left_arm_joints':[40,0,0,130,0,0,0],
              'quick_reset_right_arm_joints':[-40,0,0,-130,0,0,0],
              'quick_reset_neck_joints':[0,-30,0], 'quick_reset_open_grippers':True,
              'quick_reset_gripper_open_position':10,'gripper_min_position':10,'gripper_max_position':360}
    measured = {'left_arm':np.arange(7,dtype=float), 'right_arm':np.arange(7,dtype=float)+10,
                'neck':np.array([4.,-15.,3.]), 'leg_waist':np.array([0.,10.,20.,5.]),
                'left_gripper':np.array([80.]), 'right_gripper':np.array([90.])}
    calls = []
    s = NS(get_parameter=lambda n:NS(value=params[n]),
        _configuration_guard_error=lambda *a,**k:('',None),
        _kinematics=NS(groups_from_q_deg=lambda q,g:g),
        _reset_required_vector=lambda g:np.r_[g['left_arm'],g['right_arm'],g['neck']],
        _build_motion_controller_state=lambda g:{},
        _prepare_quick_reset_path=lambda a,b:('',{'start':np.zeros(21),'goal':np.ones(21),'duration_sec':2}),
        _latest_target_mailbox=NS(reset=lambda:calls.append('commit')),
        _commit_motion_controller_state=lambda state:None,
        _lock_head_follow_for_control_boundary_locked=lambda:None,
        _deactivate_body_height_locked=lambda:None, _velocity_active_sides=set(),
        _waist_follow_enabled=False, _gripper_targets={'left':270.,'right':250.})
    return s, measured, calls


@pytest.mark.parametrize('scope', ['left','right','both'])
def test_only_selected_arms_change_and_grippers_body_head_hold(scope):
    s, measured, calls = subject()
    assert Controller._start_quick_reset_locked(s,1.,NS(as_dict=lambda:measured),
        preserve_grippers=True,arm_scope=scope)==''
    assert s._gripper_targets == {'left':270.,'right':250.}
    assert s._gripper_dirty == {'left':False,'right':False}
    assert not s._reset_open_grippers
    for group in ['leg_waist','neck']:
        np.testing.assert_array_equal(s._target_groups[group], measured[group])
    for side in ['left','right']:
        expected = s.get_parameter('quick_reset_'+side+'_arm_joints').value if scope in [side,'both'] else measured[side+'_arm']
        np.testing.assert_array_equal(s._target_groups[side+'_arm'], expected)
    assert calls==['commit']


def test_scoped_reset_rejects_clamp_that_moves_the_held_arm_without_commit():
    s, measured, calls = subject()
    def clamp(_q, groups):
        groups['left_arm']=groups['left_arm']+1
        return groups
    s._kinematics.groups_from_q_deg=clamp
    assert 'held arm' in Controller._start_quick_reset_locked(s,1.,NS(as_dict=lambda:measured),
        preserve_grippers=True,arm_scope='right')
    assert calls==[]


def test_collision_rejection_does_not_change_gripper_or_commit():
    s, measured, calls = subject()
    s._prepare_quick_reset_path=lambda *_:('collision detected',None)
    assert Controller._start_quick_reset_locked(s,1.,NS(as_dict=lambda:measured),
        preserve_grippers=True,arm_scope='both')=='collision detected'
    assert calls==[] and s._gripper_targets=={'left':270.,'right':250.}


def test_normal_xa_reset_still_opens_grippers_and_resets_neck():
    s, measured, _calls = subject()
    assert Controller._start_quick_reset_locked(s,1.,NS(as_dict=lambda:measured))==''
    assert s._gripper_targets=={'left':10.,'right':10.}
    assert s._reset_open_grippers
    np.testing.assert_array_equal(s._target_groups['neck'],[0,-30,0])


def test_navigation_yb_never_calls_legacy_hold_reset_client():
    calls=[]
    s=NS(_reset_menu_active=False,_yb_reset_menu_enabled=True,
        _hold_reset_state=QuickResetState(button_keys=('left_y','right_b')),
        _quick_state=QuickResetState(),_reset_client='xa',_hold_reset_client='yb',
        _check_reset_gesture=lambda now,state,client:calls.append(client))
    s._hold_reset_state.update_full_snapshot(True,True,1)
    Mapper._check_quick_reset(s,3)
    assert calls==['xa']


def test_menu_ownership_expires_and_releases_on_missing_input():
    calls=[]
    s=NS(_reset_menu_active=True,_reset_menu_grips_released=True,_last_vr_time=1,
        _yb_reset_menu_enabled=True,_hold_reset_state=QuickResetState(),
        _quick_state=QuickResetState(),_reset_client='xa',
        _release_all=lambda **kw:calls.append('release'),
        _check_reset_gesture=lambda *_:False)
    assert Mapper._check_quick_reset(s,1.1) is True
    assert not Mapper._check_quick_reset(s,2)
    assert not s._reset_menu_active and not s._reset_menu_grips_released
    assert calls==['release']


def test_menu_pose_packet_masks_clutch_and_tracks_real_grip_release():
    from openarmx_teleop_vr_306_v4.reset_state import GripRearmState
    releases=[]
    s=NS(_now=lambda:10.,_yb_reset_menu_enabled=True,_lock=threading.RLock(),
        _reset_menu_active=False,_reset_menu_grips_released=False,
        _release_all=lambda **kw:releases.append(kw),_samples={},_sample_times={},
        _sample_versions={'left':0,'right':0},_grip_false_since={'left':None,'right':None},
        _grip_rearm=GripRearmState(),get_parameter=lambda name:NS(value=.04),
        _quick_state=QuickResetState(),_hold_reset_state=QuickResetState(button_keys=('left_y','right_b')),
        _desktop_gate=NS(enabled=False))
    controller={'position':{'x':0,'y':0,'z':0},'quaternion':{'x':0,'y':0,'z':0,'w':1},
                'rotation':{'x':0,'y':0,'z':0},'gripActive':True,'gripValue':1,'trigger':1,
                'xButton':True,'yButton':True,'aButton':True,'bButton':True}
    packet={'leftController':dict(controller),'rightController':dict(controller),
            'resetMenuActive':True,'resetMenuGripsReleased':False}
    Mapper._on_vr_input(s,NS(data=json.dumps(packet)))
    assert s._reset_menu_active and not s._reset_menu_grips_released
    assert all(not sample.grip_active and sample.trigger==0 for sample in s._samples.values())
    assert not s._quick_state.both_pressed and not s._hold_reset_state.both_pressed
    assert len(releases)==1
    packet['resetMenuGripsReleased']=True
    Mapper._on_vr_input(s,NS(data=json.dumps(packet)))
    assert s._reset_menu_grips_released
    assert len(releases)==1
