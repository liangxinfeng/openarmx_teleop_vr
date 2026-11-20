"""Offline clutch replay. No ROS node, publisher, service, or hardware interface."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
from openarmx_teleop_vr_306_v5.grip_session_ik import GripSessionIK
from openarmx_teleop_vr_306_v5.teleop_core import VR_TO_ROBOT_ROT as A
from teleop_v6.mapping import IncrementalArm
from teleop_v6.compat.mink_kinematics import ARM_JOINTS, MinkKinematics

REST = [20, 0, 0, -110, 0, 0, 0]
PEAK = {"right": [-17.0510,8.7453,15.9410,-2.01048,-10.5241,-19.0264,-71.1771],
        "left": [-25,15,0,-2.02,0,0,60]}
DT = .01
SHARE = Path(get_package_share_directory("openarmx_teleop_vr_306_v5"))
PARAMS = yaml.safe_load((SHARE/"config/teleop.yaml").read_text())["independent_vr_mapper_306_v5"]["ros__parameters"]

def make_mapper(side, frame="clutch"):
    keys = "workspace_minimum workspace_maximum left_y_minimum right_y_maximum forbidden_box_minimum forbidden_box_maximum position_scale forward_position_scale filter_alpha".split()
    return IncrementalArm(side=side, return_frame=frame, **{k: PARAMS[k] for k in keys},
        maximum_position_step=PARAMS["max_position_step"],
        maximum_orientation_step=np.deg2rad(PARAMS["max_orientation_step_deg"]),
        maximum_anchor_displacement=PARAMS["max_anchor_displacement"],
        maximum_forward_displacement=PARAMS["max_forward_displacement"],wrist_orientation_gain=1.)

def smooth(s):
    return s**3*(10+s*(-15+6*s))

def metrics(model, q, home, side, idx):
    now, ref = (model.forward_poses(x)[side] for x in (q,home))
    diff = np.rad2deg(q[idx]-home[idx])
    station = lambda x: elbow_xyz(model, x, side)
    return dict(tcp_mm=round(float(np.linalg.norm(now.translation-ref.translation)*1000),3),
        rot_deg=round(float(np.rad2deg(R.from_matrix(now.rotation@ref.rotation.T).magnitude())),3),
        elbow_mm=round(float(np.linalg.norm(station(q)-station(home))*1000),3),
        joint_deg=np.round(diff,3).tolist())

def elbow_xyz(model,q,side):
    model._reference_configuration.update(model._mujoco_from_guard(q))
    return model.elbow_plane_tasks[side]._positions(model._reference_configuration.data)[1].copy()

def passes(m):
    return m["tcp_mm"]<=4 and m["rot_deg"]<=2 and max(map(abs,m["joint_deg"]))<=3 and m["elbow_mm"]<=10

def exercise(side, cycles, mode, captured=False, early=False, duration=1.5, body=False, reach_mm=0):
    model=MinkKinematics(str(SHARE/"urdf/robot_v3_1_simplified.urdf"), str(SHARE/"urdf/robot_v3_1.srdf"),collision_check=True)
    home=model.q_from_feedback(dict(leg_waist=[.01044,-.01866,-.00454,-.00372],neck=[.2295,-.2295,.01093],
        left_gripper=[10],right_gripper=[10],left_arm=REST,right_arm=REST))
    idx=[model.joint_q[n] for n in ARM_JOINTS[side]]
    peak=home.copy()
    peak[idx]=np.deg2rad(PEAK[side])
    q=peak.copy() if captured else home.copy()
    assert not model.collision_pairs(q)
    command=q.copy()
    origin=model.forward_poses(home)[side]
    mapper=make_mapper(side,"room" if mode=="room" else "clutch")
    session=GripSessionIK()
    offset=np.zeros(3)
    count=0
    model._reference_configuration.update(model._mujoco_from_guard(peak))
    shoulder,_,wrist=model.elbow_plane_tasks[side]._positions(model._reference_configuration.data)
    radial=wrist-shoulder
    radial/=np.linalg.norm(radial)
    previous_goal=None
    relatch_shifts=[]
    epochs=[]
    clips={}
    freeze_drift=0.
    first_target_jump=0.
    beta=1-np.exp(-DT/.055)
    def tick(f,down,vr_offset):
        nonlocal count,command,first_target_jump,previous_goal
        count+=1
        t=count*DT
        ref=home+f*(peak-home)
        p=model.forward_poses(ref)[side]
        hand=SimpleNamespace(position=A.T@(p.translation-origin.translation+radial*f*reach_mm/1000)+vr_offset,
            orientation=R.from_matrix(A.T@(p.rotation@origin.rotation.T)@A).as_quat())
        session.observe("sequence",{s:down and s==side for s in ARM_JOINTS},q,t)
        if not down:
            mapper.release()
            command=q.copy()
            return
        edge=not mapper.latched
        current=model.forward_poses(q)[side]
        if edge:
            mapper.latch(hand,dict(position=current.translation.copy(),orientation=R.from_matrix(current.rotation).as_quat()))
        target,reason=mapper.map(hand)
        if reason: clips[reason]=clips.get(reason,0)+1
        if edge:
            first_target_jump=max(first_target_jump,1000*np.linalg.norm(target["position"]-current.translation))
        if edge and previous_goal is not None:
            relatch_shifts.append(np.round(1000*(target["position"]-previous_goal),4).tolist())
        previous_goal=target["position"].copy()
        goals={side:(target["position"],target["orientation"])}
        ticket=session.prepare(goals,q,model.joint_q,t)
        if edge:epochs.append(list(ticket.solver_epochs[side]))
        obs={}
        if body:
            model._reference_configuration.update(model._mujoco_from_guard(ref))
            s,e,w=model.elbow_plane_tasks[side]._positions(model._reference_configuration.data)
            axis=w-s
            axis/=np.linalg.norm(axis)
            bend=e-s
            bend-=axis*np.dot(axis,bend)
            obs[side]=[axis.tolist(),(bend/np.linalg.norm(bend)).tolist(),1.]
        result=model.solve(goals,ticket.seed,feedback_q=q,posture_reference=ticket.reference,
            reset_reference=home,session_epoch=ticket.solver_epochs,integration_dt=ticket.dt,transactional=True,arm_shape=obs)
        if not result.success:raise ValueError(f"frame {count}: {result.reason}")
        assert session.commit(ticket,result.q)
        model.commit_retarget_state()
        command[idx]=result.q[idx]
        q[idx]+=beta*(command[idx]-q[idx])
    if not captured:
        for j in range(151):tick(smooth(j/150),True,offset)
    else:tick(1,True,offset)
    for _ in range(0 if early or captured else 100):tick(1,True,offset)
    extended=round(float(np.rad2deg(q[idx[3]])),2)
    for k in range(cycles):
        old=q.copy()
        move=np.zeros(3) if mode=="still" else A.T@np.array([.03/max(1,cycles),0.,0.])
        for j in range(25):tick(1,False,offset+move*smooth((j+1)/25))
        freeze_drift=max(freeze_drift,np.rad2deg(np.max(np.abs(q-old))))
        offset+=move
        for _ in range(30):tick(1,True,offset)
    assert len(set(map(tuple,epochs)))==cycles+1,(epochs,cycles)
    length=round(duration/DT)
    start=None
    after=[]
    for j in range(length+1+150):
        frac=smooth(min(1,j/length))
        tick(1-frac,True,offset*(1-frac) if mode in ("world","room") else offset)
        if j>=length:
            m=metrics(model,q,home,side,idx)
            after.append(m)
            if j==length:start=m
    # Require tolerance until the last sample, at least 300 ms.
    def stable(predicate):
        last_bad=max((j for j,m in enumerate(after) if not predicate(m)),default=-1)
        return (last_bad+1)*10 if len(after)-last_bad-1>=31 else None
    return dict(side=side,regrips=cycles,mode=mode,seed="extended" if captured else "out_and_back",
        release_without_settling=early,return_s=duration,body=body,extended_elbow_deg=extended,
        reach_extra_mm=reach_mm,epochs=epochs,relatch_shifts_mm=relatch_shifts,
        final_target_mm=round(float(1000*np.linalg.norm(previous_goal-origin.translation)),3),clamped_frames=clips,release_drift_deg=float(freeze_drift),
        latch_jump_mm=float(first_target_jump),at_hand_stop=start,final=after[-1],
        wrist_settle_ms=stable(lambda m:m["tcp_mm"]<=4 and m["rot_deg"]<=2),
        elbow_settle_ms=stable(lambda m:abs(m["joint_deg"][3])<=3 and m["elbow_mm"]<=10),
        all_settle_ms=stable(passes),passed=stable(passes) is not None)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--side",choices=list(PEAK)+["both"],default="both")
    parser.add_argument("--control-only",action="store_true")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    variants=[(0,"still"),(1,"still"),(5,"still"),(10,"relative"),(10,"world"),(10,"room"),
              (5,"relative",True),(5,"relative",False,True,.75),(5,"relative",False,False,1.5,True)]
    radial_cases=[(5,"relative",False,False,1.5,False,extra) for extra in (12,28,45)]
    controls=[(0,"still",False,False,1.5,False,28),(5,"still",False,False,1.5,False,28)]
    variants=controls if args.control_only else variants+radial_cases+controls
    failures=[]
    for side in PEAK if args.side=="both" else [args.side]:
        for v in variants:
            try: row=exercise(side,*v)
            except ValueError as err: row=dict(side=side,case=v,error=str(err))
            print(json.dumps(row,allow_nan=False),flush=True)
            if "error" in row or row["latch_jump_mm"]>1e-6 or row["release_drift_deg"]>1e-6:
                failures.append((side,v))
            elif v[1]=="world":
                # Clutch mode intentionally excludes the 30 mm of free-hand motion.
                if abs(row["final"]["tcp_mm"]-30)>4:
                    failures.append((side,v))
            elif not row["passed"]:
                failures.append((side,v))
    if args.verify and failures:
        raise SystemExit(f"Failed return checks: {failures}")

if __name__=="__main__":
    main()
