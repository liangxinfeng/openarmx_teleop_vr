"""V6 desktop stack with the shared WebXR transport and vendor hardware protocol."""
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    from control_center_module_lease import controller_lease
    share = Path(get_package_share_directory("openarmx_teleop_vr_306_v5"))
    simulation = os.environ.get("V6_LOOPBACK_TEST") == "1"
    if simulation and (os.environ.get("ROS_DOMAIN_ID") != "191" or os.environ.get("ROS_LOCALHOST_ONLY") != "1"):
        raise RuntimeError("Loopback tests require an isolated ROS domain")
    suffix = "v6_sim" if simulation else os.environ["V6_TOPIC_SUFFIX"]
    root = str(Path(__file__).resolve().parents[1])
    env = {"PYTHONPATH": root+os.pathsep+os.environ.get("PYTHONPATH","")}
    interfaces = """body_height_command calibrate_reach clear_emergency_stop eef_feedback
        eef_target emergency_stop gripper_target head_target joint_target joint_target_preview
        quick_reset quick_reset_hold_grippers release_hold set_desktop_mode set_hardware_enabled
        set_head_follow_enabled set_waist_follow_enabled status task_head_target teleop_heartbeat
        teleop_status vr_input""".split()
    remaps = [item for topic in interfaces for item in
              ("-r", f"/openarmx_teleop_vr_306_v5/{topic}:=/teleop_v6/{topic}")]
    def process(role, args):
        cmd = [sys.executable, "-m", "teleop_v6.run_node", role, "--ros-args",
               *remaps, *args]
        return ExecuteProcess(cmd=cmd, additional_env=env, output="screen")
    def args(config, node, **overrides):
        result = ["-r", "__node:="+node, "--params-file", str(share/"config"/config)]
        for key, value in overrides.items():
            result += ["-p", key+":="+(str(value).lower() if isinstance(value,bool) else str(value))]
        return result
    controller = process("controller", args("controller.yaml","independent_arm_controller_306_v5",
        topic_suffix=repr(suffix), dry_run=False, require_teleop_heartbeat=True,
        enable_self_collision_check=True, urdf_path=share/"urdf/robot_v3_1_simplified.urdf",
        srdf_path=share/"urdf/robot_v3_1.srdf",
        **({"vendor_pid_loop_shm_glob": "/nonexistent-v6-sim/pid_loop_*"} if simulation else {})))
    mapper = process("mapper", args("teleop.yaml","independent_vr_mapper_306_v5",
        topic_suffix=repr(suffix), dry_run=False, operator_translation_compensation_enabled=False,
        wrist_orientation_gain=1.0, return_frame=os.environ.get("V6_RETURN_FRAME","clutch")))
    web = process("web", ["-p","host:="+("127.0.0.1" if simulation else "0.0.0.0"),
                          "-p","https_port:="+("18448" if simulation else "8448"),
                          "-p","rgbd_camera_enabled:="+("false" if simulation else "true")])
    nodes = [controller, mapper, web]
    # Domain isolation is also exercised with the same hardware/SYNC code in the test.
    actions = [] if simulation else [controller_lease("vr_arm")]
    actions += [RegisterEventHandler(OnProcessExit(target_action=n, on_exit=[
        Shutdown(reason="V6 component exited")])) for n in nodes]
    return LaunchDescription(actions + nodes)
