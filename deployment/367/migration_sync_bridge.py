"""367迁移保持会话。复用已验证会话逻辑，只在原厂现有进程中注册。"""
import threading, time, json
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from autolife_robot_arm import ArmVrCmdHandler, PROGRAM_SETTINGS
from autolife_robot_arm.utils import ControlState
from autolife_robot_sdk.interface import ControlMode
from std_msgs.msg import String
from std_srvs.srv import SetBool

def install(ros_state, shared_state_manager, config, TOPIC_NODE_ID):
    # Blocking SYNC entry must never starve the vendor feedback timers.
    # Serialize lease transitions/watchdog together, separately from the
    # default feedback/heartbeat callbacks; preserve both timeout guards.
    sync_transition_group = MutuallyExclusiveCallbackGroup()
    independent_sync_active = threading.Event()
    independent_sync_lock = threading.RLock()
    independent_sync_last_heartbeat = {'time': 0.0}
    official_arm_input_topics = {
        f"/control_topic_{TOPIC_NODE_ID}",
        f"/control_reset_{TOPIC_NODE_ID}",
        f"/control_safe_recover_{TOPIC_NODE_ID}",
        f"/topic_arm_move_eef_pose_in_robot_frame_{TOPIC_NODE_ID}",
        f"/topic_arm_move_eef_pose_in_vr_frame_{TOPIC_NODE_ID}",
        f"/topic_arm_move_joints_trajectory_{TOPIC_NODE_ID}",
        f"/topic_arm_move_up_down_z_{TOPIC_NODE_ID}",
        f"/topic_arm_move_whole_body_joints_by_planning_{TOPIC_NODE_ID}",
        f"/topic_arm_set_dexterous_hand_vr_cmd_{TOPIC_NODE_ID}",
        f"/topic_arm_set_gripper_vr_cmd_{TOPIC_NODE_ID}",
    }
    official_arm_input_topics.update({
        topic.rsplit(f"_{TOPIC_NODE_ID}", 1)[0]
        for topic in tuple(official_arm_input_topics)
    })

    class IndependentVrGateNode(Node):
        """Suppress official arm-VR callbacks during the independent lease."""

        def create_subscription(
            self, msg_type, topic, callback, qos_profile, *args, **kwargs
        ):
            if topic in official_arm_input_topics:
                original_callback = callback

                def callback(message, _original=original_callback):
                    if independent_sync_active.is_set():
                        return
                    return _original(message)

            return super().create_subscription(
                msg_type, topic, callback, qos_profile, *args, **kwargs
            )

    ros_state['node'] = IndependentVrGateNode(
        "node_arm_vr_control_service_" + TOPIC_NODE_ID
    )
    ros_state['arm_vr_service'] = ArmVrCmdHandler(
        ros_state['node'], TOPIC_NODE_ID, PROGRAM_SETTINGS, config,
        shared_state_manager
    )

    arm_joint_names = [
        "Joint_Left_Shoulder_Inner",
        "Joint_Left_Shoulder_Outer",
        "Joint_Left_UpperArm",
        "Joint_Left_Elbow",
        "Joint_Left_Forearm",
        "Joint_Left_Wrist_Upper",
        "Joint_Left_Wrist_Lower",
        "Joint_Right_Shoulder_Inner",
        "Joint_Right_Shoulder_Outer",
        "Joint_Right_UpperArm",
        "Joint_Right_Elbow",
        "Joint_Right_Forearm",
        "Joint_Right_Wrist_Upper",
        "Joint_Right_Wrist_Lower",
        "Joint_Left_Gripper",
        "Joint_Right_Gripper",
        "Joint_Waist_Pitch",
        "Joint_Waist_Yaw",
        "Joint_Neck_Roll",
        "Joint_Neck_Pitch",
        "Joint_Neck_Yaw",
    ]
    all_joint_states = (
        shared_state_manager.neck_joint_state,
        shared_state_manager.leg_waist_joint_state,
        shared_state_manager.left_arm_joint_state,
        shared_state_manager.right_arm_joint_state,
        shared_state_manager.left_gripper_state,
        shared_state_manager.right_gripper_state,
    )
    sync_stability_groups = (
        (shared_state_manager.leg_waist_joint_state, (2, 3)),
        (shared_state_manager.left_arm_joint_state, tuple(range(7))),
        (shared_state_manager.right_arm_joint_state, tuple(range(7))),
    )

    # 仅持物观察/放置会话启用。进入时保留原厂当前指令，不从历史文件重发夹紧量。
    loaded_gripper_session = {'active': False}

    def capture_gripper_targets():
        import math
        result = []
        for state in all_joint_states[-2:]:
            target = tuple(float(v) for v in state.get_target_joint_positions())
            if len(target) != 1 or not all(math.isfinite(v) and 10. <= v <= 360. for v in target):
                raise ValueError('invalid live gripper target; loaded handoff not started')
            result.append(target)
        return tuple(result)

    def with_gripper_targets(snapshot, targets):
        return tuple(snapshot[:-2]) + tuple(targets)

    def capture_measured_pose(states=all_joint_states):
        return tuple(
            tuple(float(value) for value in state.get_current_joint_positions())
            for state in states
        )

    def hold_pose(snapshot, states=all_joint_states):
        if len(snapshot) != len(states):
            raise ValueError("measured-pose snapshot size mismatch")
        for state, positions in zip(states, snapshot):
            state.set_control_mode(ControlMode.POSITION)
            state.set_target_joint_positions(list(positions))

    def capture_stability_pose():
        return tuple(
            tuple(
                float(state.get_current_joint_positions()[index])
                for index in indices
            )
            for state, indices in sync_stability_groups
        )

    def measured_stability(snapshot):
        maximum_error = 0.0
        maximum_speed = 0.0
        for (state, indices), reference in zip(
            sync_stability_groups, snapshot
        ):
            current = state.get_current_joint_positions()
            speeds = state.get_current_joint_velocities()
            if any(index >= len(current) or index >= len(speeds) for index in indices):
                raise RuntimeError("joint feedback size changed during SYNC entry")
            maximum_error = max(
                maximum_error,
                max((
                    abs(float(current[index]) - float(expected))
                    for index, expected in zip(indices, reference)
                ), default=0.0),
            )
            maximum_speed = max(
                maximum_speed,
                max((abs(float(speeds[index])) for index in indices), default=0.0),
            )
        return maximum_error, maximum_speed

    def exit_independent_sync_session(reason="independent teleop exited", *, preserve_closing_grippers=False):
        with independent_sync_lock:
            if not independent_sync_active.is_set():
                return
            snapshot = capture_measured_pose()
            if loaded_gripper_session['active']:
                # 正常结束、取消和看门狗退出都保留此刻目标；放置已松爪时保留松爪目标，不能重新夹紧。
                snapshot = with_gripper_targets(snapshot, capture_gripper_targets())
            elif preserve_closing_grippers:
                # 抓取正常结束：只保留已下发的闭合目标，不把负载下的实际开口当作新目标。
                # 腰腿/双臂/头部仍锁存实测位置；普通VR、故障和看门狗退出行为不变。
                import math
                stable=list(snapshot)
                for i,state in ((4,shared_state_manager.left_gripper_state),(5,shared_state_manager.right_gripper_state)):
                    target=tuple(float(v) for v in state.get_target_joint_positions())
                    actual=snapshot[i]
                    if len(target)!=len(actual) or any(not math.isfinite(v) for v in target+actual):
                        raise RuntimeError('invalid grasp jaw handoff feedback/target')
                    stable[i]=tuple(max(a,t) if 10.<=t<=360. else a for a,t in zip(actual,target))
                snapshot=tuple(stable)
            hold_pose(snapshot)
            time.sleep(0.10)
            state = shared_state_manager.get_control_state()
            if state in (ControlState.SYNC, ControlState.HOME):
                shared_state_manager.set_control_state(ControlState.ASYNC)
            elif state != ControlState.ASYNC:
                raise RuntimeError(f"unexpected control state on exit: {state}")
            independent_sync_active.clear()
            loaded_gripper_session['active'] = False
            independent_sync_last_heartbeat['time'] = 0.0
            print(
                f"[{TOPIC_NODE_ID}] Independent SYNC session exited safely: "
                f"measured pose held, state=ASYNC, reason={reason}"
            )

    def grasp_finish_callback(request,response):
        if request.data:
            response.success=False;response.message="finish-only service does not enable motion";return response
        try:
            if not independent_sync_active.is_set():
                raise RuntimeError("no active SYNC grasp session")
            exit_independent_sync_session("grasp complete; retain closing jaw target",preserve_closing_grippers=True)
            response.success=True;response.message="body measured hold; closing gripper targets retained; ASYNC"
        except Exception as exc:
            response.success=False;response.message=str(exc)
        return response

    ros_state['node'].create_service(
        SetBool,f"/control_independent_sync_grasp_finish_{TOPIC_NODE_ID}",
        grasp_finish_callback,callback_group=sync_transition_group)

    ros_state['sync_exit'] = exit_independent_sync_session

    def independent_sync_session_callback(request, response, *, clear_errors=True, preserve_gripper_targets=False):
        if not request.data:
            try:
                exit_independent_sync_session("explicit service request")
                response.success = True
                response.message = "measured pose held; vendor state is ASYNC"
            except Exception as exc:
                response.success = False
                response.message = f"failed to exit independent SYNC session: {exc}"
            return response

        with independent_sync_lock:
            if independent_sync_active.is_set():
                actual_state = shared_state_manager.get_control_state()
                if actual_state == ControlState.SYNC:
                    if preserve_gripper_targets != loaded_gripper_session['active']:
                        response.success = False
                        response.message = 'active SYNC session belongs to a different handoff mode'
                        return response
                    response.success = True
                    response.message = (
                        "independent SYNC session is active and vendor "
                        "control state is SYNC"
                    )
                    return response
                independent_sync_active.clear()
                independent_sync_last_heartbeat['time'] = 0.0

            # 必须在任何目标写入之前锁存并核验；无效时不进入会话、不覆盖原目标。
            try:
                retained = capture_gripper_targets() if preserve_gripper_targets else None
            except Exception as exc:
                response.success = False
                response.message = str(exc)
                return response
            def entry_snapshot():
                measured = capture_measured_pose()
                return with_gripper_targets(measured, retained) if retained is not None else measured
            loaded_gripper_session['active'] = preserve_gripper_targets
            independent_sync_active.set()
            try:
                if shared_state_manager.is_protected():
                    raise RuntimeError('vendor heartbeat protection is active')
                arm_service = ros_state['arm_vr_service']
                arm_service.keep_disabling_joints = False
                arm_service.arm_task_control.sync_waist_joints_follow_dual_arm = True
                arm_service.previous_timestamp = arm_service.realtime_clock.now()
                arm_service.last_received_vr_command_time = 0
                disable_thread = getattr(arm_service, "disable_joints_thread", None)
                if disable_thread is not None and disable_thread.is_alive():
                    disable_thread.join(timeout=2.0)
                    if disable_thread.is_alive():
                        raise RuntimeError("startup joint-disable loop did not stop")

                state = shared_state_manager.get_control_state()
                if state == ControlState.SYNC:
                    snapshot = entry_snapshot()
                    hold_pose(snapshot)
                    shared_state_manager.set_control_state(ControlState.ASYNC)
                    state = ControlState.ASYNC
                elif state == ControlState.HOME:
                    shared_state_manager.set_control_state(ControlState.ASYNC)
                    state = ControlState.ASYNC
                if state != ControlState.ASYNC:
                    raise RuntimeError(f"SYNC entry requires ASYNC, got {state}")

                snapshot_all = entry_snapshot()
                hold_pose(snapshot_all)
                stability_snapshot = capture_stability_pose()
                # Legacy callers retain their existing recovery behavior.
                # The hold-only entry never clears drivers: clearing an
                # already holding motor can briefly remove holding torque.
                if clear_errors:
                    arm_service.pub_arm_joints_clear_error.publish(
                        String(data=json.dumps(arm_joint_names))
                    )
                    time.sleep(0.50)
                hold_pose(snapshot_all)
                shared_state_manager.set_arm_joints_pid_loop_state(True)
                arm_service.pub_arm_joints_set_enable_state.publish(
                    String(data=json.dumps({
                        joint_name: True for joint_name in arm_joint_names
                    }))
                )
                shared_state_manager.set_control_state(ControlState.HOME)

                stable_since = None
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if shared_state_manager.is_protected():
                        raise RuntimeError('vendor heartbeat protection became active')
                    arm_service.previous_timestamp = arm_service.realtime_clock.now()
                    hold_pose(snapshot_all)
                    error, speed = measured_stability(stability_snapshot)
                    if error <= 3.0 and speed <= 2.0:
                        if stable_since is None:
                            stable_since = time.monotonic()
                        elif time.monotonic() - stable_since >= 0.60:
                            break
                    else:
                        stable_since = None
                    time.sleep(0.05)
                else:
                    raise RuntimeError(
                        "joint pose did not stabilize in HOME "
                        f"(error={error:.2f} deg, speed={speed:.2f} deg/s)"
                    )

                if shared_state_manager.get_control_state() != ControlState.HOME:
                    raise RuntimeError("control state changed while waiting in HOME")
                if shared_state_manager.is_protected():
                    raise RuntimeError('vendor heartbeat protection became active')
                hold_pose(entry_snapshot())
                shared_state_manager.set_control_state(ControlState.SYNC)
                independent_sync_last_heartbeat['time'] = time.monotonic()
                response.success = True
                response.message = (
                    "measured pose stable; legal ASYNC-HOME-SYNC transition complete; "
                    "official VR arm inputs are gated"
                )
                print(f"[{TOPIC_NODE_ID}] {response.message}")
                return response
            except Exception as exc:
                try:
                    hold_pose(entry_snapshot())
                    state = shared_state_manager.get_control_state()
                    if state in (ControlState.SYNC, ControlState.HOME):
                        shared_state_manager.set_control_state(ControlState.ASYNC)
                except Exception as rollback_exc:
                    print(f"[{TOPIC_NODE_ID}] SYNC rollback failed: {rollback_exc}")
                independent_sync_active.clear()
                loaded_gripper_session['active'] = False
                independent_sync_last_heartbeat['time'] = 0.0
                response.success = False
                response.message = f"guarded SYNC entry failed: {exc}"
                print(f"[{TOPIC_NODE_ID}] {response.message}")
                return response

    ros_state['node'].create_service(
        SetBool,
        f"/control_independent_sync_session_{TOPIC_NODE_ID}",
        independent_sync_session_callback,
        callback_group=sync_transition_group,
    )
    ros_state['node'].create_service(
        SetBool,
        f"/control_independent_sync_hold_session_{TOPIC_NODE_ID}",
        lambda request, response: independent_sync_session_callback(
            request, response, clear_errors=False
        ),
        callback_group=sync_transition_group,
    )
    ros_state['node'].create_service(
        SetBool,
        f"/control_independent_sync_loaded_session_{TOPIC_NODE_ID}",
        lambda request, response: independent_sync_session_callback(
            request, response, clear_errors=False, preserve_gripper_targets=True
        ),
        callback_group=sync_transition_group,
    )
    print(
        f"[{TOPIC_NODE_ID}] Independent SYNC session services registered",
        flush=True,
    )

    def independent_sync_heartbeat_callback(msg):
        if not independent_sync_active.is_set():
            return
        try:
            payload = json.loads(msg.data)
            if payload.get("active") is True:
                independent_sync_last_heartbeat['time'] = time.monotonic()
                arm_service = ros_state.get('arm_vr_service')
                if arm_service is not None:
                    arm_service.previous_timestamp = arm_service.realtime_clock.now()
                    arm_service.last_received_vr_command_time = 0
        except Exception as exc:
            print(f"[{TOPIC_NODE_ID}] Invalid independent heartbeat: {exc}")

    ros_state['node'].create_subscription(
        String,
        f"/control_independent_sync_heartbeat_{TOPIC_NODE_ID}",
        independent_sync_heartbeat_callback,
        10,
    )

    def independent_sync_command_callback(msg):
        try:
            payload = json.loads(msg.data)
            if payload.get("enable") is False:
                exit_independent_sync_session(
                    str(payload.get("reason", "controller exit command"))
                )
        except Exception as exc:
            print(f"[{TOPIC_NODE_ID}] Invalid independent session command: {exc}")

    ros_state['node'].create_subscription(
        String,
        f"/control_independent_sync_session_command_{TOPIC_NODE_ID}",
        independent_sync_command_callback,
        10,
        callback_group=sync_transition_group,
    )

    def independent_sync_watchdog():
        if not independent_sync_active.is_set():
            return
        state = shared_state_manager.get_control_state()
        if state != ControlState.SYNC:
            try:
                exit_independent_sync_session(
                    f"vendor control state left SYNC unexpectedly: {state}"
                )
            except Exception as exc:
                print(f"[{TOPIC_NODE_ID}] SYNC state-coherence fail-safe failed: {exc}")
            return
        age = time.monotonic() - independent_sync_last_heartbeat['time']
        if age > 1.5:
            try:
                exit_independent_sync_session(
                    f"controller heartbeat expired after {age:.2f}s"
                )
            except Exception as exc:
                print(f"[{TOPIC_NODE_ID}] SYNC heartbeat fail-safe failed: {exc}")

    ros_state['node'].create_timer(
        0.10, independent_sync_watchdog,
        callback_group=sync_transition_group,
    )

    return ros_state
