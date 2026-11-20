"""ROS adapter. The vendor controller remains the sole joint and reset writer."""
from std_srvs.srv import SetBool
import os
import time
from pathlib import Path
from .compat.controller_node import IndependentArmController
from .lifecycle import ReleaseGate


class Controller(IndependentArmController):
    def __init__(self):
        self.exit_gate = ReleaseGate()
        self._v6_acquisition = None
        self.closing = False
        self.drain_reset = False
        self._verified_reset_sync = False
        self.owner_marker = Path(os.environ["V6_OWNER_MARKER"]) if "V6_OWNER_MARKER" in os.environ else None
        super().__init__()
        self.create_timer(.025, self._complete_exit)

    def _request_sync_session_exit(self):
        acquisition = self._v6_acquisition or self._sync_session_future
        if self._sync_session_acquired or acquisition is not None:
            self.exit_gate.begin(acquisition, float(self.get_parameter("sync_session_request_timeout_sec").value))
        self._v6_acquisition = None
        self._sync_session_acquired = False

    def _on_hardware_enabled(self, request, response):
        gate = self.exit_gate
        if request.data and not self._enable_pending and not self._hardware_enabled:
            self._verified_reset_sync = False
        if self._enable_pending and self._enable_stage == "wait_full_reset" and not self._estop_latched:
            # A published vendor reset cannot be recalled. Finish observing that
            # same reset, then either stay disarmed or continue the requested entry.
            self.drain_reset = self.closing or not request.data
            response.success = not (request.data and self.closing)
            response.message = "Waiting for the current full reset to settle"
            return response
        if not request.data:
            gate.reenter = False
            gate.retry(float(self.get_parameter("sync_session_request_timeout_sec").value))
        elif self.closing:
            response.success = False
            response.message = "Controller is exiting"
            return response
        elif gate.pending and not self._estop_latched:
            gate.reenter = not bool(gate.error)
            response.success = gate.reenter
            response.message = gate.error or "Re-enable queued until previous SYNC exit acknowledged"
            return response
        result = super()._on_hardware_enabled(request, response)
        if self.owner_marker and request.data and result.success:
            self.owner_marker.touch()
        return result

    def _on_emergency_stop(self, request, response):
        self.exit_gate.reenter = False
        self.drain_reset = False
        return super()._on_emergency_stop(request, response)

    def _run_full_reset_enable(self, now, feedback):
        stage = self._enable_stage
        handled = super()._run_full_reset_enable(now, feedback)
        if stage == "wait_full_reset_quiet" and self._enable_stage == "hold_before_enable":
            self._verified_reset_sync = True
        if self.drain_reset and self._enable_stage != "wait_full_reset":
            self.drain_reset = False
            if self._enable_pending:
                self._on_hardware_enabled(SetBool.Request(data=False), SetBool.Response())
            return True
        return handled

    def _stationary_feedback(self, feedback):
        if self._feedback_motion is None:
            return False
        try:
            error,speed=self._feedback_motion.maximum_error_and_speed(feedback.as_dict())
        except (ValueError,TypeError,KeyError):
            return False
        return (error <= self.get_parameter("full_reset_position_tolerance_deg").value and
                speed <= self.get_parameter("full_reset_speed_tolerance_deg_sec").value)

    def _complete_exit(self):
        resume = self.exit_gate.tick(lambda: self._sync_session_client.call_async(SetBool.Request(data=False)))
        if self.owner_marker and not (self.exit_gate.pending or self._hardware_enabled or self._enable_pending):
            self.owner_marker.unlink(missing_ok=True)
        if resume:
            response = self._on_hardware_enabled(SetBool.Request(data=True), SetBool.Response())
            if not response.success:
                self.get_logger().warning(response.message)

    def _describe_exit(self, payload):
        payload["release_pending"] = self.exit_gate.pending
        if self.drain_reset and self._enable_pending:
            payload.update(state="EXITING",hardware_enable_pending=False,reason="Waiting for the issued full reset to finish")
        if self.exit_gate.pending:
            payload["hardware_enable_pending"] |= self.exit_gate.reenter
            if payload["state"] not in ("E_STOP", "FAULT"):
                payload["state"] = "FAULT" if self.exit_gate.failed else "EXITING"
                payload["reason"] = self.exit_gate.error or "waiting for vendor SYNC exit acknowledgement"
                payload["detail"] = payload["reason"]

    def finish(self):
        import rclpy
        self.closing = True
        self._on_hardware_enabled(SetBool.Request(data=False), SetBool.Response())
        deadline = time.monotonic() + (float(self.get_parameter("arm_enable_completion_timeout_seconds").value) if self.drain_reset else 3.0)
        while (self.exit_gate.pending or self.drain_reset) and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.025)
        self._complete_exit()
