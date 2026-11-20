set -e
root=$(cd "$(dirname "$0")" && pwd -P)
cd "${V6_PLATFORM_ROOT:-$HOME/migration367_20261118}"
. releases/control-center-current/environment.sh
cd "${V6_TEST_OUTPUT:-$root}"
export V6_SOURCE_ROOT="$root" PYTHONPATH="$root:$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 ROS_DOMAIN_ID=191 ROS_LOCALHOST_ONLY=1 V6_LOOPBACK_TEST=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# The isolated domain, fake robot suffix and loopback-only DDS prevent a route to hardware.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers><Peer address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
python - <<'PY'
import rclpy,time
rclpy.init();n=rclpy.create_node('v6_isolation_preflight');time.sleep(.9)
assert set(n.get_node_names())=={n.get_name()},n.get_node_names()
n.destroy_node();rclpy.shutdown()
PY
python - <<'PY'
import os, signal, subprocess as sp, sys
from pathlib import Path
with open('loop_nodes.log','w') as nodes, open('loop_result.log','w') as report:
    server=sp.Popen([sys.executable,'/opt/ros/jazzy/bin/ros2','launch',
                     str(Path(os.environ['V6_SOURCE_ROOT'])/'teleop_v6/bringup.launch.py')],
                    start_new_session=True,stdout=nodes,stderr=sp.STDOUT)
    rc=1
    try:
        rc=sp.run([sys.executable,str(Path(os.environ['V6_SOURCE_ROOT'])/'test_ros_loop.py')],stdout=report,stderr=sp.STDOUT,
                  env={**os.environ,"V6_TEST_LAUNCH_PID":str(server.pid)}).returncode
    finally:
        if rc:
            server.send_signal(signal.SIGINT)
        try: server.wait(timeout=10)
        except sp.TimeoutExpired:
            os.killpg(server.pid,signal.SIGTERM)
            server.wait(timeout=6)
print(Path('loop_result.log').read_text())
if rc:
    print('\n'.join(line[:320] for line in Path('loop_nodes.log').read_text().splitlines()[-16:]))
sys.exit(rc)
PY
