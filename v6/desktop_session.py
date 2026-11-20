"""Desktop ownership handoff around the shared ROS launch lease."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from time import monotonic, sleep

UNIT = "flow-service.service"


def systemctl(action):
    return subprocess.run(["systemctl","--user",action,UNIT], capture_output=True, text=True)


def flow_owns_joints(find_owner, domain):
    while True:
        owner = find_owner("joints", domain)
        if owner is None:
            return False
        try:
            groups = Path(f"/proc/{owner}/cgroup").read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(group.rstrip("/").endswith("/"+UNIT) for group in groups):
            return True
        raise RuntimeError(f"关节正在由其他控制程序使用（PID {owner}），请从原程序退出。")


def wait_for_handoff(initial_state, find_owner, domain):
    if initial_state in ("active", "activating", "reloading", "deactivating"):
        print("正在等待原厂流程释放关节…", flush=True)
        result = systemctl("stop")
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "原厂流程停止失败")
    deadline = monotonic()+15
    while flow_owns_joints(find_owner, domain):
        if monotonic() >= deadline:
            raise RuntimeError("原厂流程仍未释放关节，请查看 flow-service.service 状态。")
        sleep(.1)


def main():
    folder = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/"+str(os.getuid())))
    with (folder/"teleop-v6-desktop.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("V6 is already open")
        # Leave existing manual control owners intact; stop only the original flow service.
        from control_center_module_lease import _legacy_owner
        domain = os.environ.get("ROS_DOMAIN_ID","0")
        flow_owns_joints(_legacy_owner, domain)
        initial_state = systemctl("is-active").stdout.strip()
        restore = initial_state in ("active", "activating", "reloading")
        child = None
        session = tempfile.TemporaryDirectory(prefix="teleop-v6-",dir=folder)
        marker = Path(session.name)/"unreleased"
        closing = False
        def close(signum, frame):
            nonlocal closing
            closing = True
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)
        for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
            signal.signal(sig,close)
        try:
            wait_for_handoff(initial_state, _legacy_owner, domain)
            if closing:
                return 0
            launch = Path(__file__).parent/"teleop_v6/bringup.launch.py"
            child = subprocess.Popen([sys.executable,"/opt/ros/jazzy/bin/ros2","launch",str(launch)],start_new_session=True,
                                     env={**os.environ,"V6_OWNER_MARKER":str(marker)})
            if closing:
                child.send_signal(signal.SIGINT)
            return child.wait()
        finally:
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)
                child.wait()
            unconfirmed = marker.exists()
            session.cleanup()
            if restore and unconfirmed:
                raise RuntimeError("Vendor release was not confirmed. Flow service remains paused.")
            if restore:
                # Only restore after the shared joint lease and vendor owner are free.
                path=Path.home()/".local/state/autolife-control-center/leases/joints.lock"
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open("a+") as shared:
                    deadline=monotonic()+10
                    while True:
                        try:
                            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if monotonic()>=deadline:
                                raise RuntimeError("Joint lease is still owned; flow remains paused")
                            sleep(.1)
                    if _legacy_owner("joints",domain) is None:
                        result=systemctl("start")
                        if result.returncode:
                            raise RuntimeError(result.stderr.strip())


if __name__=="__main__":
    sys.exit(main())
