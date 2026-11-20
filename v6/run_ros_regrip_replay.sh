set -e
rel=${V6_TEST_RELEASE:-$(cd "$(dirname "$0")" && pwd -P)}
test -f "$rel/test_ros_loop.py"
sim=$(mktemp -d /tmp/v6-regrip-XXXXXXXX)
ln -s "$rel/teleop_v6" "$sim/teleop_v6"
ln -s "$rel/test_straight_return.py" "$rel/test_regrip_return.py" "$sim/"
cp "$rel/loop.sh" "$sim/loop.sh"
sed '/^pool=ThreadPoolExecutor/,$d' "$rel/test_ros_loop.py" > "$sim/test_ros_loop.py"
cat >> "$sim/test_ros_loop.py" <<'PY'
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
from openarmx_teleop_vr_306_v5.teleop_core import VR_TO_ROBOT_ROT as A
from teleop_v6.compat.mink_kinematics import ARM_JOINTS, MinkKinematics
from test_straight_return import landmarks
straight=os.environ.get("V6_TEST_STRAIGHT")=="1"
share=Path(get_package_share_directory("openarmx_teleop_vr_306_v5"))
model=MinkKinematics(str(share/"urdf/robot_v3_1_simplified.urdf"),str(share/"urdf/robot_v3_1.srdf"),collision_check=True)
nominal=model.q_from_feedback(home)
peak=nominal.copy()
for side,angles in (("right",[-17.051,8.745,15.941,-2.01,-10.524,-19.026,-71.177]),("left",[-25,15,0,-2.02,0,0,60])):
    peak[[model.joint_q[j] for j in ARM_JOINTS[side]]]=np.deg2rad(angles)
reference=model.forward_poses(nominal)
reach_mm=float(os.environ.get("V6_TEST_REACH_MM","0"))
radial={}
model._reference_configuration.update(model._mujoco_from_guard(peak))
for side in ARM_JOINTS:
    shoulder,_,wrist=model.elbow_plane_tasks[side]._positions(model._reference_configuration.data)
    radial[side]=(wrist-shoulder)/np.linalg.norm(wrist-shoulder)
frames=[]
for j in range(151):
    f=j/150
    f=f**3*(10+f*(-15+6*f))
    poses=model.forward_poses(nominal+f*(peak-nominal))
    frame={}
    for s,sign in (("left",-1),("right",1)):
        p,basepose=poses[s],reference[s]
        pos=A.T@(p.translation-basepose.translation+f*radial[s]*reach_mm/1000)+np.array([sign*.22,.9,-.35])
        quat=R.from_matrix(A.T@(p.rotation@basepose.rotation.T)@A).as_quat()
        if straight:
            pos=np.array([sign*.22,.9,-.35-.24*f])
            quat=np.array([0.,0.,0.,1.])
        frame[s]=dict(position=dict(zip("xyz",pos.tolist())),quaternion=dict(zip("xyzw",quat.tolist())))
    frames.append(frame)
index=0
free_move=0.
def packet():
    snap=(index,free_move,grip)
    hands={}
    arms={}
    for s in ("left","right"):
        entry=frames[snap[0]][s]
        position=entry["position"].copy()
        if straight:
            sign=1 if s=="left" else -1
            position["x"]+=sign*snap[1]
            reach=-position["z"]-.35
            pts=landmarks(.24+reach,-sign*(.035+snap[1]),-.21,0.)
            arms[s]=(pts+np.array([-sign*.2,1.3,0.])).tolist()
        elif os.environ.get("V6_RETURN_FRAME")=="room" and edges_completed and snap[2]:
            position["z"]-=snap[1]*snap[0]/150
        else:
            position["z"]-=snap[1]
        hands[s+"Controller"]=dict(position=position,quaternion=entry["quaternion"],gripActive=snap[2],trigger=0.)
    return dict(head=dict(position=dict(x=0.,y=1.6,z=0.),rotation=dict(x=0.,y=0.,z=0.),quaternion=dict(x=0.,y=0.,z=0.,w=1.)),armLandmarks=arms,**hands)
def slide(back=False):
    global index
    start=time.monotonic()
    while True:
        n=min(150,int((time.monotonic()-start)*100))
        index=150-n if back else n
        if n==150:return
        rclpy.spin_once(node,timeout_sec=.006)
def active(sides):
    return status.get("ik_session_v5",{}).get("active")==sides
def measure():
    now=model.q_from_feedback(q)
    poses=model.forward_poses(now)
    result={}
    for side in ARM_JOINTS:
        idx=[model.joint_q[n] for n in ARM_JOINTS[side]]
        p=poses[side]
        base=origin_pose[side]
        def elbow(conf):
            model._reference_configuration.update(model._mujoco_from_guard(conf))
            return model.elbow_plane_tasks[side]._positions(model._reference_configuration.data)[1].copy()
        result[side]=dict(elbow_deg=round(float(np.rad2deg(now[idx[3]])),2),
            outward_mm=round(float(1000*(1 if side=="left" else -1)*(elbow(now)-elbow(origin))[1]),3),
            joints_deg=np.round(np.rad2deg(now[idx]-origin[idx]),3).tolist(),
            tcp_mm=round(float(1000*np.linalg.norm(p.translation-base.translation)),3),
            rot_deg=round(float(np.rad2deg(R.from_matrix(p.rotation@base.rotation.T).magnitude())),3),
            elbow_mm=round(float(1000*np.linalg.norm(elbow(now)-elbow(origin))),3))
    return result
pool=ThreadPoolExecutor(max_workers=2)
edges_completed=False
try:
    wait(lambda:enabled.service_is_ready() and status.get("body_height",{}).get("feedback_fresh"),20,"virtual feedback")
    stream=pool.submit(asyncio.run,headset())
    pause(1.15)
    f=pool.submit(web_enable)
    wait(f.done,30,"simulated enable")
    f.result()
    wait(lambda:synced and status.get("hardware_enabled"),25,"simulation enabled")
    pause(.5)
    origin=model.q_from_feedback(q)
    origin_pose=model.forward_poses(origin)
    grip=True
    wait(lambda:active(["left","right"]),5,"initial grip")
    slide()
    pause(1)
    extended=measure()
    excursion=min(extended[s]["elbow_deg"]-np.rad2deg(origin[model.joint_q[ARM_JOINTS[s][3]]]) for s in ARM_JOINTS)
    assert excursion>(30 if straight else 55),extended
    edges=0
    for k in range(5):
        grip=False
        wait(lambda:active([]),3,"release")
        free_move+=.006
        pause(.22)
        grip=True
        wait(lambda:active(["left","right"]),3,"reacquire")
        edges+=1
        pause(.3)
    edges_completed=True
    slide(True)
    at_stop=measure()
    if straight:
        assert all(status["ik_session_v5"]["retargeting"][s]["shape_source"]=="body" for s in ARM_JOINTS)
        grip=False
        wait(lambda:active([]),3,"release at end of stroke")
    pause(1.5)
    final=measure()
    assert events.count("full_reset")==1 and status.get("hardware_enabled"),events
    print(json.dumps(dict(straight=straight,return_frame=os.environ.get("V6_RETURN_FRAME"),reach_extra_mm=reach_mm,extended=extended,confirmed_regrips=edges,at_stop=at_stop,final=final,
        full_resets_since_enable=events.count("full_reset"),free_hand_translation_mm=30)),flush=True)
    for s,result in final.items():
        assert result["tcp_mm"]<=4 and result["rot_deg"]<=2 and result["elbow_mm"]<=10, (s,result)
        assert max(map(abs,result["joints_deg"]))<=3, (s,result)
    call(enabled,SetBool.Request(data=False))
    wait(lambda:not synced,8,"fake vendor released")
    os.kill(int(os.environ["V6_TEST_LAUNCH_PID"]),signal.SIGINT)
finally:
    stop_stream.set()
    pool.shutdown(wait=True)
    node.destroy_node()
    rclpy.shutdown()

PY
for scenario in clutch:0 clutch:28 room:28; do
    V6_RETURN_FRAME="${scenario%:*}" V6_TEST_REACH_MM="${scenario#*:}" bash "$sim/loop.sh"
done
V6_TEST_STRAIGHT=1 V6_RETURN_FRAME=clutch V6_TEST_REACH_MM=0 bash "$sim/loop.sh"
