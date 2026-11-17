import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from openarmx_teleop_vr_306_v5.operator_heading import StableOperatorHeading
from openarmx_teleop_vr_306_v5.teleop_core import webxr_yaw_deg_from_quaternion
def pose(yaw,pitch):
 return (Rotation.from_euler('y',yaw,degrees=True)*Rotation.from_euler('x',pitch,degrees=True)).as_quat()
@pytest.mark.parametrize('yaw',[-150.,-45.,0.,50.,160.])
def test_looking_down_across_vertical_cannot_flip_clutch_heading(yaw):
 h=StableOperatorHeading();h.update(pose(yaw,0));initial=h.value()
 for pitch in [-40.,-70.,-89.,-90.,-91.,-110.,-80.,-40.,0.]:
  h.update(pose(yaw,pitch));assert abs(h.value()-initial)<1e-6
 # Reproduce the old geometric failure rather than only checking new code.
 a=webxr_yaw_deg_from_quaternion(pose(yaw,-89));b=webxr_yaw_deg_from_quaternion(pose(yaw,-91))
 assert abs(abs((b-a+180)%360-180)-180)<1e-6
def test_no_initial_horizontal_reference_rejects_instead_of_guessing():
 h=StableOperatorHeading();h.update(pose(90,-90))
 with pytest.raises(ValueError):h.value()
 h.update(pose(90,-20));assert h.value()==pytest.approx(90.)
def test_upright_turn_can_establish_a_new_clutch_direction():
 h=StableOperatorHeading()
 for yaw in [0,30,80,150,-170]:
  h.update(pose(yaw,20));assert h.value()==pytest.approx(yaw)
