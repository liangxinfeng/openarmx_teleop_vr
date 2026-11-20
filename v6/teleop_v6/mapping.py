"""One shared heading, independent measured clutch origins and reversible workspace limits."""
import numpy as np
from scipy.spatial.transform import Rotation

from openarmx_teleop_vr_306_v5.teleop_core import ArmMapper, VR_TO_ROBOT_ROT
from openarmx_teleop_vr_306_v5.operator_heading import StableOperatorHeading


class SessionHeading(StableOperatorHeading):
    def __init__(self):
        super().__init__()
        self.fixed = None

    def value(self):
        if self.fixed is None:
            self.fixed = super().value()
        return self.fixed


def remaining_offset(position, anchor, home, error_length):
    """Linearly repay an anchor error along the return stroke, at most 1:1 with hand travel."""
    stroke = anchor - home
    length = np.linalg.norm(stroke)
    if length < 1e-8:
        return 1.
    progress = (anchor-position) @ stroke / (length*max(length, error_length))
    return 1. - float(np.clip(progress, 0., 1.))


class IncrementalArm(ArmMapper):
    """Keep operator intent for the session; rebase only the measured output on Grip."""
    def __init__(self, return_frame="clutch", **kwargs):
        super().__init__(**kwargs)
        if return_frame not in ("clutch", "room"):
            raise ValueError("return_frame must be clutch or room")
        self.return_frame = return_frame
        self.reference = None
        self.human_length = None
        self.retarget_status = {"mode": "v6_incremental", "calibrated": False}

    def set_geometry(self, geometry):
        # The adapter reports the native shoulder model; the robot IK owns its limits.
        self.geometry = geometry

    def reset_reference(self):
        self.release()
        self.reference = None

    def _hand_rotation(self, orientation, origin):
        relative = Rotation.from_quat(orientation) * Rotation.from_quat(origin).inv()
        rotvec = relative.as_rotvec()
        gain = 1 + (self.wrist_orientation_gain-1)*np.sinc(np.linalg.norm(rotvec)/np.pi)
        return Rotation.from_rotvec(gain * (VR_TO_ROBOT_ROT @ rotvec))

    def _intent(self, sample):
        hand, quat, position, rotation = self.input_anchor
        delta = VR_TO_ROBOT_ROT @ (sample.position-hand)
        delta *= [self.forward_position_scale, self.position_scale, self.position_scale]
        return position + delta, self._hand_rotation(sample.orientation, quat) * rotation

    def latch(self, sample, current_pose, body_position=None):
        super().latch(sample, current_pose)
        measured = self.robot_origin_position
        measured_rotation = Rotation.from_quat(self.robot_origin_orientation)
        if self.reference is None:
            self.reference = (sample.position.copy(), sample.orientation.copy(), measured.copy(), measured_rotation)
            self.intent_position, self.intent_rotation = self.reference[2:]
        self.input_anchor = self.reference if self.return_frame == "room" else (
            sample.position.copy(), sample.orientation.copy(), self.intent_position, self.intent_rotation)
        self.intent_position, self.intent_rotation = self._intent(sample)
        self.grip_position = self.intent_position
        self.position_anchor = self.intent_position
        self.position_offset = measured - self.intent_position
        self.rotation_anchor = (self.intent_rotation * self.reference[3].inv()).magnitude()
        self.rotation_offset = (measured_rotation * self.intent_rotation.inv()).as_rotvec()

    def _project(self, position):
        relative = position - self.robot_origin_position
        distance = np.linalg.norm(relative)
        if 0 < self.maximum_anchor_displacement < distance:
            relative *= self.maximum_anchor_displacement/distance
        if self.maximum_forward_displacement > 0:
            relative[0] = min(relative[0], self.maximum_forward_displacement)
        return self._constrain_position(self.robot_origin_position+relative)

    def map(self, sample, body_position=None):
        if not self.latched or not np.isfinite(sample.position).all():
            return None, "Grip requires a fresh measured anchor"
        position, rotation = self._intent(sample)
        moved = np.any(position != self.intent_position)
        self.intent_position, self.intent_rotation = position, rotation
        home, home_rotation = self.reference[2:]
        reason = ""
        if moved:
            fraction = remaining_offset(position,self.position_anchor,home,np.linalg.norm(self.position_offset))
            desired = position + fraction*self.position_offset
            self.filtered_position, reason = self._project(desired)
            if np.max(np.abs(self.filtered_position-desired)) > 1e-9:
                self.position_anchor = position
                self.position_offset = self.filtered_position-position
        angle = (rotation * home_rotation.inv()).magnitude()
        if self.rotation_anchor > 1e-8:
            angular_span = max(self.rotation_anchor, np.linalg.norm(self.rotation_offset))
            weight = 1. - np.clip((self.rotation_anchor-angle)/angular_span, 0., 1.)
        else:
            weight = remaining_offset(position,self.grip_position,home,0.)
        orientation = Rotation.from_rotvec(weight*self.rotation_offset) * rotation
        return {"position": self.filtered_position.copy(), "orientation": orientation.as_quat()}, reason
