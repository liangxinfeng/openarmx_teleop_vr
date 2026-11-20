"""Same 367 URDF/SRDF, native CPU Mink QP and two lagged 7-axis joint plants."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory as share
from openarmx_teleop_vr_306_v5.kinematics import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from openarmx_teleop_vr_306_v5.mink_kinematics import MinkKinematics as V5
from teleop_v6.compat.mink_kinematics import MinkKinematics as V6
from openarmx_teleop_vr_306_v5.latest_sample import LatestSampleMailbox
from teleop_v6.session import SessionMailbox
import yaml

SIDES = ("left","right")
STAGES = [
    ("front", [8, -6, 2, 13, 0, 0, 0]), ("lateral", [-9, 7, -6, 3, 4, 0, 0]),
    ("raised", [12, 0, -4, -7, 0, 3, 0]), ("wrist", [5, -2, 1, 4, 9, -10, 8]),
    ("home", [0]*7)]
DT=.01


def pose_error(a,b):
    return [1e3*np.linalg.norm(a.translation-b.translation),
            np.rad2deg((R.from_matrix(a.rotation).inv()*R.from_matrix(b.rotation)).magnitude())]


def simulate(version, latency_ms, stages=STAGES, expect_collision=False):
    folder=Path(share("openarmx_teleop_vr_306_v5"))
    cfg=yaml.safe_load((folder/"config/controller.yaml").read_text())["independent_arm_controller_306_v5"]["ros__parameters"]
    model=(V6 if version==6 else V5)(str(folder/"urdf/robot_v3_1_simplified.urdf"),str(folder/"urdf/robot_v3_1.srdf"),
                collision_check=True,dt=DT)
    groups={"leg_waist":np.zeros(4),"neck":np.zeros(3),"left_gripper":[10.],"right_gripper":[10.]}
    for side in SIDES:
        groups[side+"_arm"]=cfg["quick_reset_"+side+"_arm_joints"]
    home=model.q_from_feedback(groups)
    q=home.copy()
    cmd=q.copy()
    seed=q.copy()
    ix={s:[model.joint_q[n] for n in names] for s,names in zip(SIDES,(LEFT_ARM_JOINTS,RIGHT_ARM_JOINTS))}
    previous=home.copy()
    mailbox=(SessionMailbox if version==6 else LatestSampleMailbox)()
    accepted=0; inputs=0; inflight=None; pending=False
    times=[]; jerk=[]; old_velocity=np.zeros_like(q); old_acceleration=np.zeros_like(q)
    tracking=[]; output=[]
    # One-ms event clock, 90 Hz input, a 100 Hz feedback plant; injected delay is deterministic.
    elapsed=0; next_input=0.; native_us=[]
    for name,offset in stages:
        dest=home.copy()
        dest[ix["left"]]+=np.deg2rad(offset)
        dest[ix["right"]]+=np.deg2rad(np.array(offset)*[1,-1,-1,1,-1,1,-1])
        goal=model.forward_poses(dest)
        arrivals={s:None for s in SIDES}
        for ms in range(1800):
            elapsed+=1
            if elapsed>=next_input:
                next_input+=1000/90
                f=min(ms/800,1.)
                blend=f*f*f*(10-15*f+6*f*f)
                ref=previous+blend*(dest-previous)
                target=model.forward_poses(ref)
                targets={s:(target[s].translation.copy(),R.from_matrix(target[s].rotation).as_quat()) for s in SIDES}
                mailbox.put(targets);inputs+=1;pending=True
            if inflight and elapsed>=inflight[0]:
                _,ticket,result=inflight;inflight=None
                if mailbox.is_latest(ticket):
                    if expect_collision and result.reason=="self collision":
                        assert not result.success and not model.collision_pairs(cmd)
                        return dict(point=name,elapsed_ms=ms,collision_pairs=model.collision_pairs(result.q),
                                    last_command_collision_free=True)
                    assert result.success,(name,ms,result.reason)
                    cmd=result.q.copy();seed=cmd.copy()
                    model.commit_retarget_state()
                    accepted+=1;times.append(elapsed)
                else:
                    mailbox.discard_inflight()
            if inflight is None and pending:
                ticket,targets=mailbox.take();pending=False
                cost = [6,8,15,14,9,13,7][inputs%7] if latency_ms=="jitter" else latency_ms
                result=model.solve(targets,seed,feedback_q=q,posture_reference=home,reset_reference=home,
                           integration_dt=max(.01,cost/1000),transactional=True,session_epoch={s:(1,1) for s in SIDES},
                           maximum_arm_lead_rad=np.deg2rad(12))
                native_us.append(result.solve_time_ms)
                inflight=elapsed+cost,ticket,result
            if elapsed%10==0:
                # Both sides use the same clock, different joint response times.
                old=q.copy()
                for s,tau in zip(SIDES,(.055,.070)):
                    q[ix[s]]+=(1-np.exp(-DT/tau))*(cmd[ix[s]]-q[ix[s]])
                vel=(q-old)/DT;acc=(vel-old_velocity)/DT
                jerk.append(float(np.linalg.norm((acc-old_acceleration)/DT)))
                old_velocity,old_acceleration=vel,acc
                actual=model.forward_poses(q)
                for s in SIDES:
                    tracking.append(pose_error(actual[s],target[s])[0])
                    mm,deg=pose_error(actual[s],goal[s])
                    if mm<=4 and deg<=2:
                        if arrivals[s] is None:arrivals[s]=ms
                    else:arrivals[s]=None
        end={s:pose_error(actual[s],goal[s]) for s in SIDES}
        output.append(dict(point=name,end_mm_deg=end,stable_arrival_ms=arrivals,
                  skew_ms=abs(arrivals['left']-arrivals['right']) if None not in arrivals.values() else None))
        previous=dest.copy()
    return dict(version=version,delay_ms=latency_ms,accepted=accepted,inputs=inputs,points=output,
                discarded=mailbox.stats()["discarded_inflight"],rms_mm=float(np.sqrt(np.mean(np.square(tracking)))),
                peak_gap_ms=max(np.diff(times),default=None).item() if len(times)>1 else None,
                joint_jerk_rms_rad_s3=float(np.sqrt(np.mean(np.square(jerk)))),
                native_qp_p95_ms=float(np.percentile(native_us,95)),
                home_joint_max_deg=float(np.rad2deg(np.max(abs(q-home)))))


if __name__=="__main__":
    data=[]
    for delay in (6,12,"jitter"):
        for v in (5,6):
            row=simulate(v,delay);data.append(row);print(json.dumps(row),flush=True)
    Path("tracking.json").write_text(json.dumps(data,indent=2))
    blocked=simulate(6,"jitter",[("near_body",[8,-4,1,35,0,0,0]),("across_body",[10,-4,1,72,0,0,0])],expect_collision=True)
    assert blocked.get("collision_pairs"),blocked
    Path("collision_case.json").write_text(json.dumps(blocked,indent=2))
    edge_stages=[("reach_mid",[-22,8,0,40,0,0,0]),("reach_far",[-45,15,0,80,0,0,0]),
                 ("reach_wrist",[-45,15,0,80,9,-10,8]),("retreat",[-22,8,0,40,0,0,0]),STAGES[-1]]
    edge=simulate(6,"jitter",edge_stages)
    Path("extended_tracking.json").write_text(json.dumps(edge,indent=2))
    for row in [*data,edge]:
        if row["version"]==6:
            assert row["accepted"]>500
            for point in row["points"]:
                assert point["skew_ms"] is not None and point["skew_ms"]<=200,point
                assert all(mm<=4 and deg<=2 for mm,deg in point["end_mm_deg"].values()),point
