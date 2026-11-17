"""No robot/node/publisher construction. Exercise actual selection and height methods."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from openarmx_teleop_vr_306_v5.manual_squat import ManualSquatGuard
sys.path.insert(0, str(Path(__file__).parent))
from test_height_watchdog_regression import Harness, namespace, controller, SOURCE

scope=dict(namespace,ManualSquatGuard=ManualSquatGuard)
names={'_manual_squat_snapshot','_select_manual_squat_mode'}
exec(compile(ast.Module(body=[n for n in controller.body if isinstance(n,ast.FunctionDef)
    and n.name in names],type_ignores=[]),str(SOURCE),'exec'),scope)
for name in names:setattr(Harness,name,scope[name])

def standing():
    c=Harness();c.measured[:]=0;c._body_height_active=False
    c._body_height_last_direction=0;c._body_height_reverse_mode=False
    c._waist_follow_neutral_pitch_deg=c._waist_follow_pitch_target_deg=0
    for stamp in (10.,10.2,10.4):
        c._feedback_time=stamp;c._manual_squat_snapshot(stamp)
    return c

def request(c,mode='reverse',**extras):
    p=dict(select_squat_mode=mode,request_id='request-1',enabled=False,direction=0,
        expires_monotonic=11.)
    p.update(extras)
    with patch.object(scope['time'],'monotonic',return_value=10.4):
        c._on_body_height_command(NS(data=json.dumps(p)))

def test_selection_changes_no_joint_target_or_enable():
    c=standing();before=c._target_groups['leg_waist'].copy();target=c._body_height_joint_targets.copy()
    request(c)
    assert c._manual_squat_ack['accepted'] and c._body_height_reverse_mode
    np.testing.assert_array_equal(c._target_groups['leg_waist'],before)
    np.testing.assert_array_equal(c._body_height_joint_targets,target)
    assert c._hardware_enabled and not c._body_height_active

@pytest.mark.parametrize('fault',['crouched','stale','moving','reset','disabled','absolute','expired','with_motion'])
def test_invalid_choice_never_alters_pose_or_mode(fault):
    c=standing();extras={}
    if fault=='crouched':c.measured[:3]=[10,20,10]
    elif fault=='stale':c._feedback_time=8.
    elif fault=='moving':c._body_height_active=True;c._body_height_last_direction=1
    elif fault=='reset':c._reset_active=True
    elif fault=='disabled':c._hardware_enabled=False
    elif fault=='absolute':c._body_height_active=True;c._body_height_input_absolute=True
    elif fault=='expired':extras['expires_monotonic']=9.
    elif fault=='with_motion':extras.update(enabled=True,direction=1)
    before=c._target_groups['leg_waist'].copy();request(c,**extras)
    assert not c._manual_squat_ack['accepted'] and not c._body_height_reverse_mode
    np.testing.assert_array_equal(c._target_groups['leg_waist'],before)

def test_repeated_cached_feedback_cannot_fake_stability():
    g=ManualSquatGuard()
    for now in (10.,10.1,10.2,10.3,10.4):
        assert not g.observe([0,0,0],10.,now,.5)['can_switch']
    assert not g.observe([0,0,0],10.,10.6,.5)['can_switch']

def test_startup_uses_measured_branch_waist_lean_ignored():
    g=ManualSquatGuard()
    assert g.observe([-40,-80,12],10,10,.35)['mode']=='reverse'
    assert g.observe([40,80,-12],10.1,10.1,.35,True)['mode']=='normal'
    for stamp in (11,11.2,11.4):s=g.observe([0,0,12],stamp,stamp,.35,True)
    assert s['can_switch'] and s['mode']=='reverse'

@pytest.mark.parametrize('mode,sign',[('normal',1),('reverse',-1)])
def test_x_lowers_y_raises_on_both_branches_without_crossing(mode,sign):
    c=standing();request(c,mode)
    for t in (10.4,10.5,10.6):
        c._feedback_time=t
        with patch.object(scope['time'],'monotonic',return_value=t):
            c._on_body_height_command(NS(data='{"enabled":true,"direction":1}'))
        c.measured[:3]=c._target_groups['leg_waist'][:3]
    assert sign*c.measured[1]>0
    for i in range(15):
        t=10.7+i*.1;c._feedback_time=t
        with patch.object(scope['time'],'monotonic',return_value=t):
            c._on_body_height_command(NS(data='{"enabled":true,"direction":-1}'))
        c.measured[:3]=c._target_groups['leg_waist'][:3]
        assert sign*c.measured[1]>=-1.e-6
    assert abs(c.measured[1])<.01
