"""Retraction from the recorded 367 elbow limit, matched synthetic targets, 100 Hz."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from teleop_v6.compat.mink_kinematics import ARM_JOINTS, MinkKinematics

# Read from the physical arm after grip release, in the vendor's seven-axis order.
CAPTURE = np.array([-17.0510,8.7453,15.9410,-2.01048,-10.5241,-19.0264,-71.1771])
REST = np.array([20.,0.,0.,-110.,0.,0.,0.])
MIRROR = np.array([1,-1,-1,1,-1,1,-1])


def exercise(side,mode,reverse=False,duration=1.5):
    share=Path(get_package_share_directory("openarmx_teleop_vr_306_v5"))
    m=MinkKinematics(str(share/"urdf/robot_v3_1_simplified.urdf"),str(share/"urdf/robot_v3_1.srdf"),collision_check=True)
    groups=dict(leg_waist=[.01044,-.01866,-.00454,-.00372],neck=[.2295,-.2295,.01093],
                left_gripper=[10],right_gripper=[10],left_arm=REST,right_arm=REST)
    home=m.q_from_feedback(groups)
    idx=[m.joint_q[n] for n in ARM_JOINTS[side]]
    start=home.copy()
    start[idx]=np.deg2rad(CAPTURE if side=="right" else [-25,15,0,-2.02,0,0,60])
    if mode=="paired":
        start[idx[3]]=np.deg2rad(-35.)
    finish=home.copy()
    if mode=="near":
        finish[idx]=home[idx]+.22*(start[idx]-home[idx])
    if mode in ("near_outside","near_centre"):
        # Both reachable alternatives match the two distances and orientation
        # residual seen in the last live status. The original target was lost.
        inferred=([10.66,-9.77,-10.66,-99.62,-2.37,10.25,1.11] if mode=="near_outside"
                  else [6.16,.58,13.16,-108.57,13.28,-6.21,10.7])
        finish[idx]=np.deg2rad(np.array(inferred)*(MIRROR if side=="left" else 1))
    if reverse:start,finish=finish,start
    contacts=m.collision_pairs(start) or m.collision_pairs(finish)
    if contacts:
        return dict(side=side,mode=mode,direction="extend" if reverse else "return",
                    rejected="fixture in collision",pairs=contacts,settle_after_input_ms=None)
    q=start.copy();command=q.copy()
    initial=m.forward_poses(start)[side]
    goal=m.forward_poses(finish)[side]
    first_bend=None;settled=None;wrist_at=None;elbow_at=None;errs=[];late=[];elbow=[];gap=[]
    steps=round(duration/.01)
    for i in range(steps+155):
        # Both directions use the identical FK path, speed profile and plant.
        f=min(i/steps,1.)
        f=f**3*(10+f*(-15+6*f))
        reference=start+f*(finish-start)
        fk=m.forward_poses(reference)[side]
        r=m.solve({side:(fk.translation,Rotation.from_matrix(fk.rotation).as_quat())},command,
            feedback_q=q,posture_reference=start,reset_reference=home,session_epoch={side:(1,1)},
            integration_dt=.01,transactional=True)
        if not r.success:
            return dict(side=side,mode=mode,direction="extend" if reverse else "return",
                        rejected=r.reason,step=i,pairs=r.collision_pairs,settle_after_input_ms=None)
        command=r.q.copy();m.commit_retarget_state()
        q[idx]+=(1-np.exp(-.01/.055))*(command[idx]-q[idx])
        actual=m.forward_poses(q)[side]
        error=float(1000*np.linalg.norm(actual.translation-fk.translation))
        angle=float(np.rad2deg((Rotation.from_matrix(actual.rotation).inv()*Rotation.from_matrix(fk.rotation)).magnitude()))
        bend=float(np.rad2deg(q[idx[3]]))
        errs.append(error);elbow.append(bend)
        if i>=steps:late.append(error)
        gap.append(abs(bend-np.rad2deg(reference[idx[3]])))
        if first_bend is None and abs(bend-np.rad2deg(start[idx[3]]))>3:first_bend=i*10
        if i>=steps:
            wrist_at=(i if wrist_at is None else wrist_at) if error<=4 and angle<=2 else None
            elbow_at=(i if elbow_at is None else elbow_at) if abs(bend-np.rad2deg(finish[idx[3]]))<=3 else None
    if wrist_at is not None and elbow_at is not None:settled=(max(wrist_at,elbow_at)-steps)*10
    start_angle,end_angle=np.rad2deg([start[idx[3]],finish[idx[3]]])
    middle=next((k for k,b in enumerate(elbow) if (b-start_angle)/(end_angle-start_angle)>=.5),None)
    return dict(side=side,mode=mode,direction="extend" if reverse else "return",first_3deg_ms=first_bend,duration=duration,
        elbow_midpoint_lag_ms=round((middle-steps/2)*10) if middle is not None else None,
        wrist_settle_ms=(wrist_at-steps)*10 if wrist_at is not None else None,
        elbow_settle_ms=(elbow_at-steps)*10 if elbow_at is not None else None,
        settle_after_input_ms=settled,elbow_at_input_end_deg=round(elbow[steps],2),end_elbow_deg=round(elbow[-1],2),
        peak_elbow_lag_deg=round(max(gap),2),rms_position_mm=round(float(np.sqrt(np.mean(np.square(errs)))),2),
        end_mm=round(late[-1],3),end_orientation_deg=round(angle,3),
        eef_excursion_mm=round(1000*float(np.linalg.norm(goal.translation-initial.translation)),1))


def run():
    rows=[exercise(s,mode,rev) for s in ("right","left") for mode in ("home","paired") for rev in (False,True)]
    rows+= [exercise("right",mode,duration=.75) for mode in ("home","near","near_outside","near_centre")]
    Path("return_replay.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows))
    return rows


if __name__=="__main__":
    rows=run()
    if not os.environ.get("V6_MEASURE_ONLY"):
        for row in rows:
            assert row.get("wrist_settle_ms") is not None,row
            if row["direction"]=="return" or row["mode"]=="paired":
                assert row["settle_after_input_ms"] is not None,row
