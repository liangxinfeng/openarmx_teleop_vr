"""Native seven-axis, bilateral tracking with grip edges and the same Cartesian input."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from openarmx_teleop_vr_306_v5.grip_session_ik import GripSessionIK
from teleop_v6.compat.mink_kinematics import ARM_JOINTS, MinkKinematics


def landmarks(model, q, side):
    model._reference_configuration.update(model._mujoco_from_guard(q))
    task = model.elbow_plane_tasks[side]
    return task._positions(model._reference_configuration.data)


def axis_bend(points):
    s,e,w=points
    axis=w-s
    axis/=np.linalg.norm(axis)
    bend=e-s
    bend-=axis*np.dot(bend,axis)
    return axis,bend/np.linalg.norm(bend)


def run(model_class, body, previous=False):
    share=Path(get_package_share_directory("openarmx_teleop_vr_306_v5"))
    model=model_class(str(share/"urdf/robot_v3_1_simplified.urdf"),str(share/"urdf/robot_v3_1.srdf"),collision_check=True)
    groups=dict(leg_waist=[0]*4,neck=[0]*3,left_gripper=[10],right_gripper=[10])
    for side in ARM_JOINTS:
        groups[side+"_arm"]=[20,0,0,-110,0,0,0]
    home=model.q_from_feedback(groups)
    q=home.copy()
    command=q.copy()
    session=GripSessionIK()
    indices={s:[model.joint_q[n] for n in names] for s,names in ARM_JOINTS.items()}
    peak=home.copy()
    for side in indices:
        signs=np.ones(7) if side=="left" else np.array([1,-1,-1,1,-1,1,-1])
        peak[indices[side]]+=np.deg2rad([-23,8,15,38,-12,8,5])*signs
    flips=0
    cart=[]
    elbow_errors=[]
    final={}
    last_return={}
    for i in range(720):
        # Four round trips. Regrip at both extremes and while returning.
        phase=i%180
        down={s:(phase not in ([48,49,104,105,144,145] if s=="left" else [62,63,118,119])) for s in ARM_JOINTS}
        f=np.clip((phase-8)/39 if phase<82 else 1-(phase-82)/61,0,1)
        f=f**3*(10-15*f+6*f*f)
        human=home+f*(peak-home)
        poses=model.forward_poses(human)
        goals={s:(poses[s].translation.copy(),Rotation.from_matrix(poses[s].rotation).as_quat()) for s in ARM_JOINTS if down[s]}
        obs={}
        for s in goals:
            a,b=axis_bend(landmarks(model,human,s))
            obs[s]=[a.tolist(),b.tolist(),1.]
        if not body:obs={}
        stamp=(i+1)*.01
        session.observe("sim",down,q,stamp)
        for s in ARM_JOINTS:
            if not down[s]:
                command[indices[s]]=q[indices[s]]
        ticket=session.prepare(goals,q,model.joint_q,stamp)
        result=model.solve(goals,ticket.seed,feedback_q=q,posture_reference=ticket.reference,
            reset_reference=home,session_epoch=ticket.solver_epochs,integration_dt=ticket.dt,
            transactional=True,**({} if previous else {"arm_shape":obs}))
        assert result.success,(i,result.reason)
        assert session.commit(ticket,result.q)
        model.commit_retarget_state()
        for s in goals:command[indices[s]]=result.q[indices[s]]
        for s,tau in zip(ARM_JOINTS,(.045,.065)):
            q[indices[s]]+=(1-np.exp(-.01/tau))*(command[indices[s]]-q[indices[s]])
        for s in goals:
            actual=landmarks(model,q,s)
            desired=landmarks(model,human,s)
            elbow_errors.append(1000*np.linalg.norm(actual[1]-desired[1]))
            cart.append(1000*np.linalg.norm(model.forward_poses(q)[s].translation-poses[s].translation))
            if phase>=143:
                neutral=landmarks(model,home,s)
                lateral=(1 if s=="left" else -1)*(actual[1,1]-neutral[1,1])*1000
                last_return.setdefault(s,[]).append(float(lateral))
            if phase==179:
                final[s]=max(final.get(s,0.),float(np.max(np.abs(np.rad2deg(q[indices[s]]-home[indices[s]])))))
        if phase in (48,62,104,118,144):
            flips+=1
    # Same hand pose, commanded human elbow alone turns about the shoulder-wrist axis.
    poses=model.forward_poses(home)
    goals={s:(p.translation.copy(),Rotation.from_matrix(p.rotation).as_quat()) for s,p in poses.items() if s in ARM_JOINTS}
    angles={}
    for i in range(130):
        obs={}
        for s in goals:
            a,b=axis_bend(landmarks(model,home,s))
            g=np.deg2rad(18 if s=="left" else -18)
            bend=Rotation.from_rotvec(g*a).apply(b)
            obs[s]=[a.tolist(),bend.tolist(),1.]
        stamp+=.01
        session.observe("sim",{s:True for s in goals},q,stamp)
        t=session.prepare(goals,q,model.joint_q,stamp)
        r=model.solve(goals,t.seed,feedback_q=q,posture_reference=t.reference,reset_reference=home,
                      integration_dt=t.dt,transactional=True,session_epoch=t.solver_epochs,
                      **({} if previous else {"arm_shape":obs if body else {}}))
        assert r.success,r.reason
        assert session.commit(t,r.q)
        model.commit_retarget_state()
        q+=.2*(r.q-q)
    for s in goals:
        a,b=axis_bend(landmarks(model,home,s))
        a1,b1=axis_bend(landmarks(model,q,s))
        plane=model.elbow_plane_tasks[s]
        plane._reference_axis,plane._reference_bend=a,b
        neutral=plane._transport_bend(a1)
        angles[s]=float(np.rad2deg(np.arctan2(np.dot(a1,np.cross(neutral,b1)),np.dot(neutral,b1))))
    return dict(regrips=flips,tracking_rms_mm=float(np.sqrt(np.mean(np.square(cart)))),
                elbow_rms_mm=float(np.sqrt(np.mean(np.square(elbow_errors)))),max_home_joint_error_deg=final,
                peak_outward_after_return_mm={s:max(0,max(v)) for s,v in last_return.items()},
                fixed_hand_swivel_deg=angles,
                fixed_hand_error_mm={s:1000*np.linalg.norm(model.forward_poses(q)[s].translation-poses[s].translation) for s in goals})


def main():
    data={}
    baseline=os.environ.get("V6_SHAPE_BASELINE")
    if baseline:
        spec=importlib.util.spec_from_file_location("prior_shape_solver",baseline)
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        data["prior"]=run(old.MinkKinematics,False,True)
    for mode in ("neutral","body"):
        data[mode]=run(MinkKinematics,mode=="body")
    Path("arm_shape.json").write_text(json.dumps(data,indent=2))
    print(json.dumps(data))
    for mode in ("neutral","body"):
        assert max(data[mode]["max_home_joint_error_deg"].values())<2,data[mode]
        assert max(data[mode]["fixed_hand_error_mm"].values())<3,data[mode]
    for s,goal in (("left",18),("right",-18)):
        assert abs(data["body"]["fixed_hand_swivel_deg"][s]-goal)<3,data["body"]


if __name__=="__main__":
    main()
