"""Replay the held pose captured 2026-11-19. Only model FK/IK and virtual joints."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from test_return import REST, ARM_JOINTS, MinkKinematics, get_package_share_directory
from teleop_v6.arm_shape import body_observation
from openarmx_teleop_vr_306_v5.teleop_core import VR_TO_ROBOT_ROT

HELD={"leg_waist":[0.024839401245117188,-0.026899337768554688,-0.005175377707928419,-0.0014527376042678952],"neck":[-0.18578115105628967,-0.03278490900993347,-0.18578115105628967],"left_arm":[-10.887619018554688,14.914169311523438,-27.98175621032715,-146.5445556640625,-0.8851925134658813,27.659971237182617,21.78043556213379],"right_arm":[-16.9576416015625,-11.94781494140625,15.303722381591797,-147.99473571777344,8.994102478027344,-10.589689254760742,31.594215393066406],"left_gripper":[10.285774230957031],"right_gripper":[10]}

def arm_metrics(model,q,side,pose):
    model._reference_configuration.update(model._mujoco_from_guard(q))
    _,e,w=model.elbow_plane_tasks[side]._positions(model._reference_configuration.data)
    v=w-e
    actual=model.forward_poses(q)[side]
    return dict(side_mm=round(float(v[1]*1000),2),vertical_deg=round(float(np.rad2deg(np.arccos(v[2]/np.linalg.norm(v)))),2),
        tcp_mm=round(float(1000*np.linalg.norm(actual.translation-pose.translation)),3),
        wrist_deg=round(float(np.rad2deg(R.from_matrix(actual.rotation@pose.rotation.T).magnitude())),3))

def replay(mode):
    share=Path(get_package_share_directory("openarmx_teleop_vr_306_v5"))
    m=MinkKinematics(str(share/"urdf/robot_v3_1_simplified.urdf"),str(share/"urdf/robot_v3_1.srdf"),collision_check=True)
    home=m.q_from_feedback(dict(HELD,left_arm=REST,right_arm=REST))
    q=m.q_from_feedback(HELD)
    seed=q.copy()
    poses=m.forward_poses(q)
    goals={s:(poses[s].translation.copy(),R.from_matrix(poses[s].rotation).as_quat()) for s in ARM_JOINTS}
    start={s:arm_metrics(m,q,s,poses[s]) for s in goals}
    obs={}
    # Operator faces forward, upper arm forward/down; forearm straight up.
    # The robot and operator shoulder-to-wrist axes intentionally differ.
    for s in goals:
        sign=1 if s=="left" else -1
        e=np.array([.13,sign*.08,-.22])
        w=e+np.array([0.,0.,.23])
        obs[s]=body_observation(np.array([np.zeros(3),e,w])@VR_TO_ROBOT_ROT,np.eye(3))
    sources=[]
    for k in range(200):
        with_body=mode=="body" or (mode=="dropout" and k<90)
        r=m.solve(goals,seed,feedback_q=q,posture_reference=q,reset_reference=home,
            session_epoch={s:(1,1) for s in goals},integration_dt=.01,transactional=True,
            arm_shape=obs if with_body else {})
        assert r.success,(k,r.reason)
        m.commit_retarget_state()
        seed=r.q.copy()
        q+=(1-np.exp(-.01/.055))*(seed-q)
        if k in (89,92,130,199):sources.append({s:m.last_retarget_status[s]["shape_source"] for s in goals})
    return dict(mode=mode,start=start,end={s:arm_metrics(m,q,s,poses[s]) for s in goals},sources=sources)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--verify",action="store_true")
    args=p.parse_args()
    for mode in ("body","fallback","dropout"):
        row=replay(mode)
        print(json.dumps(row),flush=True)
        if args.verify:
            for s,m in row["end"].items():
                assert abs(m["side_mm"])<5,(mode,s,m)
                assert m["tcp_mm"]<4 and m["wrist_deg"]<2,(mode,s,m)
