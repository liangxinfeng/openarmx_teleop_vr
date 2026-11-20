"""Assemble three V6 compatibility adapters from the installed, fingerprinted vendor stack.
Only these adapters are version-specific. V5 source, entry points and configs stay in place.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
BASE = "openarmx_teleop_vr_306_v5"
PINS = {
    "controller_node": "33aea8bfad40587703bdd5e3ba0fca863f3de8551b99435b7d37dd39c3f15784",
    "vr_mapper_node": "74f56ab16d0463e8282354ddc8d44cd7b3b8770edf4f2d55cf936b584fbacce0",
    "mink_kinematics": "23a2f0ed2443505d77bdbbb32d1c8d6dd7b9d9261194b0e77d49899fc2e13b5d",
}


def change(source, old, new):
    if source.count(old) != 1:
        raise ValueError("V6 adapter mismatch: "+old)
    return source.replace(old, new)


def adapt(name, src):
    src = re.sub(r"from \.(\w+) import", rf"from {BASE}.\1 import", src)
    if name == "vr_mapper_node":
        src = change(src, "            'position_scale': 0.70,",
                     "            'return_frame': 'clutch',\n            'position_scale': 0.70,")
        src = change(src, "        mapper_args = {", "        mapper_args = {\n"
                     "            'return_frame': self.get_parameter('return_frame').value,")
        src = change(src, "\n        self._mappers[side].release()", "\n        if require_release:\n"
                     "            self._mappers[side].reset_reference()\n        else:\n"
                     "            self._mappers[side].release()")
        src = change(src, "from "+BASE+".operator_heading import StableOperatorHeading",
                     "from teleop_v6.arm_shape import body_observation\nfrom "+BASE+".operator_heading import StableOperatorHeading")
        src = change(src, "            self._reach_frame_sequence += 1", """            landmarks = packet.get('armLandmarks')
            self._body_frame = (now, landmarks if isinstance(landmarks, dict) else {})
            self._reach_frame_sequence += 1""")
        src = change(src, "                targets[side] = target\n                active.append(side)", """                stamp, arms = getattr(self, '_body_frame', (0., {}))
                basis = (self._desktop_position_rotation if self._desktop_gate.enabled
                         else self._wearable_position_rotation[side])
                target = dict(target)
                observation = body_observation(arms.get(side), basis) if now-stamp < .15 else None
                if observation is not None:
                    target['arm_shape'] = observation
                targets[side] = target
                active.append(side)""")
        src = change(src, "        payload[f'pos_{side}_in_robot']", """        if 'arm_shape' in pose:
            payload.setdefault('arm_shape', {})[side] = pose['arm_shape']
        payload[f'pos_{side}_in_robot']""")
        src = change(src, f"from {BASE}.operator_heading import StableOperatorHeading",
                     "from teleop_v6.mapping import SessionHeading as StableOperatorHeading")
        src = change(src, f"from {BASE}.shoulder_retarget import ShoulderReachMapper",
                     "from teleop_v6.mapping import IncrementalArm as ShoulderReachMapper")
        # V6 uses one incremental frame for every operator. Disable the legacy
        # anthropometric calibration rather than advertising an inactive scale.
        start = src.index("        fresh =", src.index("    def _update_reach_calibration"))
        end = src.index("\n    def ", start)
        src = src[:start]+"        return\n"+src[end:]
        start = src.index("\n", src.index("    def _on_calibrate_reach"))
        end = src.index("\n    def ",start)
        src = src[:start]+"\n        response.success = False\n        response.message = 'V6 uses clutch-relative incremental control'\n        return response\n"+src[end:]
        start = src.index("        try:\n            stored=json.loads(self._reach_path")
        end = src.index("        self._reach_service", start)
        src = src[:start]+"        self._reach_state.update(state='v6_incremental', message='V6 shared-frame incremental control')\n"+src[end:]
    elif name == "mink_kinematics":
        src = change(src, "algorithm_revision = 'clutch-frame-tracking-20261030'",
                     "algorithm_revision = 'v6-elbow-tracking'")
        src = change(src, "from "+BASE+".elbow_plane import ElbowPlaneTask",
                     "from "+BASE+".elbow_plane import ElbowPlaneTask\nfrom teleop_v6.arm_shape import set_arm_shape")
        src = change(src, "'_session_references', '_target_history', '_return_shape_history', '_straight_history',",
                     "'_target_history', '_swivel_history',")
        for obsolete in ("                self._session_references[side] = reference[idx].copy()\n",
                         "                self._return_shape_history.pop(side, None)\n                self._straight_history.pop(side, None)\n"):
            src = change(src, obsolete, "")
        start = src.index("        self.straight_posture_tasks =")
        src = src[:start] + src[src.index("        self.elbow_plane_tasks =",start):]
        src = change(src, "        return_start_m=.45,", "        arm_shape=None, return_start_m=.45,")
        start = src.index("            # Retain the frozen Grip reference")
        end = src.index("                'reach_projection_m':",start)
        src = src[:start]+"""            shape_status = set_arm_shape(self, side, names, idx, seed_mj, reference, home, transform,
                                         arm_shape, dt, epoch, task_stack)
            self.last_retarget_status[side] = {
                **shape_status, 'solver': 'mink/quadprog', 'algorithm_revision': self.algorithm_revision,
                'orientation_cost': orientation_cost, 'extension_blend': extension,
"""+src[end:]
        src = change(src,"orientation_cost = self._orientation_weight*(1.-0.75*extension)",
                     "orientation_cost = orientation_weight_for_motion(self._orientation_weight, extension, reach_projection,\n"
                     "                np.linalg.norm(position-self.configuration.data.site_xpos[self._site_ids[side]]))")
        src = change(src,f"from {BASE}.return_shape import return_shape_step",
                     "from teleop_v6.dexterity import orientation_cost as orientation_weight_for_motion")
    else:
        src = change(src, "                posture_reference=ticket.reference,", "                arm_shape=payload.get('arm_shape'),\n                posture_reference=ticket.reference,")
        # Accumulate the stability windows during the mandatory observation
        # intervals. Both the minimum duration and measured stability must pass.
        start = src.index("            if elapsed < minimum_wait:")
        end = src.index("            if self._feedback_motion is None:", start)
        src = src[:start]+"""            if self._feedback_time <= self._full_reset_published_at:
                return True
"""+src[end:]
        src = change(src, "            self._full_reset_published_at = now", "            self._reset_motion_seen = False\n            self._full_reset_published_at = now")
        src = change(src, "            if maximum_error > position_tolerance or maximum_speed > speed_tolerance:",
                     "            if maximum_error > position_tolerance or maximum_speed > speed_tolerance:\n                self._reset_motion_seen |= maximum_speed > speed_tolerance")
        src = change(src, "            if self._full_reset_stable_since is None:", """            if elapsed < minimum_wait and not self._reset_motion_seen:
                return True
            if self._full_reset_stable_since is None:""")
        src = change(src, "            if stable_elapsed < stable_seconds:",
                     "            if elapsed < minimum_wait or stable_elapsed < stable_seconds:")
        start = src.index("        if now - self._enable_stage_started < settle_seconds:")
        end = src.index("        current = feedback.as_dict()",start)
        src = src[:start]+src[end:]
        src = change(src, "        if now - self._enable_stable_since < stable_seconds:",
                     "        if now-self._enable_stage_started < settle_seconds or now-self._enable_stable_since < stable_seconds:")
        src = change(src,"            if now - self._enable_stage_started < hold_seconds:", """            if self._verified_reset_sync:
                if not self._stationary_feedback(feedback):
                    return True
            elif now - self._enable_stage_started < hold_seconds:""")
        src = change(src,"        if frame_delta > stable_delta:", """        if self._verified_reset_sync and self._sync_session_acquired:
            if self._feedback_time <= self._enable_stage_started or not self._stationary_feedback(feedback):
                self._reason = 'SYNC acquired; waiting for fresh stationary feedback'
                return True
            if hold_error <= hold_tolerance and frame_delta <= stable_delta:
                settle_seconds = stable_seconds = 0.
                self._enable_stable_since = now
        if frame_delta > stable_delta:""")
        src = change(src, "self._enable_stage = 'waiting_sync_session'",
                     "self._v6_acquisition = self._sync_session_future\n            self._enable_stage = 'waiting_sync_session'")
        src = change(src, "            'ik_session_v5': {",
                     "            'teleop_revision': 'V6',\n            'ik_session_v5': {")
        src = change(src, "        self._status_pub.publish(String(data=json.dumps(payload",
                     "        self._describe_exit(payload)\n        self._status_pub.publish(String(data=json.dumps(payload")
        src = change(src,f"from {BASE}.latest_sample import LatestSampleMailbox",
                     "from teleop_v6.session import SessionMailbox as LatestSampleMailbox")
        src = change(src,f"from {BASE}.mink_kinematics import MinkKinematics","from .mink_kinematics import MinkKinematics")
        src = change(src, "        self._clutch_sequence = sequence\n", """        signature = (session, state['left'], state['right'])
        if signature != getattr(self, '_grip_signature', None):
            self._latest_target_mailbox.reset()
            self._grip_signature = signature
        self._clutch_sequence = sequence
""")
        src = change(src,"or payload.get('clutch_session') != self._clutch_session\n                        or payload.get('clutch_sequence') != self._clutch_sequence):",
                     "or payload.get('clutch_session') != self._clutch_session):")
        src = change(src,"            self._last_ik = result\n            if result.success:",
                     """            if time.monotonic() - self._feedback_time > float(self.get_parameter('feedback_timeout_sec').value):
                return
            self._last_ik = result
            if result.success:""")
        src = change(src, "if feedback is not None and joint_pub is not None:", "if self._hardware_enabled and feedback is not None and joint_pub is not None:")
        for marker in ("def _set_hardware_fault(self, reason):","def _start_quick_reset_locked(self, now, feedback, *, source='quick_reset', preserve_grippers=False):"):
            # Insert after the function docstring when present.
            lines=src.splitlines(True)
            fn=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==marker.split('(')[0][4:])
            first=fn.body[0]
            offset=first.end_lineno if ast.get_docstring(fn) else first.lineno-1
            lines.insert(offset,"        self._latest_target_mailbox.reset()\n")
            src="".join(lines)
    compile(src, name, "exec")
    return src


def main():
    from ament_index_python.packages import get_package_share_directory
    upstream = Path(importlib.util.find_spec(BASE).origin).parent
    out = ROOT/"teleop_v6"/"compat"
    sources = {}
    for name, expected in PINS.items():
        data = (upstream/(name+".py")).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Upstream version changed: "+name)
        sources[name] = data
    if sys.argv[1:] == ["--check"]:
        return
    out.mkdir(exist_ok=True)
    (out/"__init__.py").touch()
    for name, data in sources.items():
        (out/(name+".py")).write_text(adapt(name, data.decode()),encoding="utf-8")
    web = ROOT / "teleop_v6" / "web"
    web.mkdir(exist_ok=True)
    for asset in (Path(get_package_share_directory(BASE)) / "web").iterdir():
        dest = web / asset.name
        if asset.name in ("index.html", "vr_app.js"):
            content = asset.read_text(encoding="utf-8").replace("306 V5", "V6").replace("V5 双臂", "V6 双臂").replace("V5 连续", "V6 连续")
            if asset.name == "index.html":
                content = change(content, "optionalFeatures: local-floor, bounded-floor", "optionalFeatures: local-floor, bounded-floor, body-tracking")
                content = content.replace("</head>", "<style>#reachCalibrationButton,#reachCalibrationStatus{display:none!important}</style></head>")
            else:
                content = change(content, "${labels[info.state] || '求解中'}", "${labels[info.state] || '求解中'} · ${({body:'人体前臂',last_body:'保留最近肘向',raised_wrist:'抬腕姿态估计'})[info.shape_source] || '中立臂形'}")
                content = (ROOT/"teleop_v6"/"body_input.js").read_text(encoding="utf-8")+"\n"+change(content,
                    "      head: this.headData(leftController, rightController),",
                    "      armLandmarks: readArmLandmarks(this.el.renderer.xr),\n      head: this.headData(leftController, rightController),")
            dest.write_text(content,encoding="utf-8")
        elif not dest.exists():
            dest.symlink_to(asset.resolve())


if __name__ == "__main__":
    main()
