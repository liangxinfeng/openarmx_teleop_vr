"""Motor-coordinate squat FK and arm IK regressions; no robot transport."""
from pathlib import Path
import numpy as np
import pinocchio as pin
import pytest
from openarmx_teleop_vr_306_v5.kinematics import IndependentArmKinematics, LEG_WAIST_JOINTS
from openarmx_teleop_vr_306_v5.mink_kinematics import MinkKinematics
MODEL=Path(__file__).resolve().parents[1]/'urdf/robot_v3_1_simplified.urdf'
def groups(h):
 return dict(leg_waist=np.array([76.,155.,81.,0.])*h/.56,
             left_arm=np.array([0.,20.,0.,-50.,0.,0.,0.]),
             right_arm=np.array([0.,-20.,0.,-50.,0.,0.,0.]),neck=np.zeros(3))
@pytest.mark.parametrize('h',[-.5,-.25,0.,.25,.5])
def test_squat_torso_stays_upright_and_motor_values_roundtrip(h):
 ik=IndependentArmKinematics(MODEL,collision_check=False)
 g=groups(h);q=ik.q_from_feedback(g,clip=False)
 data=ik.model.createData();pin.forwardKinematics(ik.model,data,q);pin.updateFramePlacements(ik.model,data)
 t=data.oMf[ik.model.getFrameId('Link_Waist_Pitch_to_Waist_Yaw')]
 a,k,w=np.deg2rad(g['leg_waist'][:3]);pitch=a-k+w
 assert np.allclose(t.rotation[:,2],[np.sin(pitch),0,np.cos(pitch)],atol=1e-8)
 expected=.28+.3395*np.cos(a)+.3675*np.cos(a-k)
 assert abs(t.translation[2]-expected)<1e-8
 assert t.rotation[2,2]>.998
 assert np.allclose(ik.groups_from_q_deg(q,g)['leg_waist'],g['leg_waist'])
@pytest.mark.parametrize('h',[-.25,0.,.25])
@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('axis',[0,1,2])
def test_small_arm_translation_keeps_direction_after_squat(h,side,axis):
 ik=MinkKinematics(MODEL,collision_check=False);q=ik.q_from_feedback(groups(h));original=q.copy()
 pose=ik.eef_poses(q)[side];start=np.array(pose['position']);target=start.copy();target[axis]+=.015
 for _ in range(90):
  result=ik.solve({side:(target,pose['orientation'])},q,feedback_q=q)
  assert result.success,result.reason
  q=result.q
 final=np.array(ik.eef_poses(q)[side]['position'])
 assert final[axis]-start[axis]>.009,(h,side,axis,final-start)
 assert np.linalg.norm(final-target)<.006
 for n in LEG_WAIST_JOINTS:assert q[ik.joint_q[n]]==original[ik.joint_q[n]]
