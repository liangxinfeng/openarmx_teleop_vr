"""Keep crouch/look-down pitch from reversing the next clutch's XY basis."""
import math
import numpy as np
from .teleop_core import quaternion_to_matrix

class StableOperatorHeading:
    def __init__(self):
        self.yaw = None
        self.held_for_tilt = False

    def update(self, quaternion):
        rotation = quaternion_to_matrix(quaternion)
        forward = rotation @ np.array([0., 0., -1.])
        # Near vertical, projecting gaze onto the floor is ill-conditioned;
        # crossing vertical flips it by 180 degrees. Keep the last reliable
        # facing direction instead. Fresh HMD samples remain required by caller.
        reliable = math.hypot(float(forward[0]), float(forward[2])) >= .5 and rotation[1, 1] >= .5
        self.held_for_tilt = not reliable
        if reliable:
            self.yaw = math.degrees(math.atan2(-float(forward[0]), -float(forward[2])))

    def value(self):
        if self.yaw is None:
            raise ValueError('请先抬头平视建立方向参考，再按 Grip 接管')
        return self.yaw
