"""Straight robot-X reach, clutch and return with independent operator shoulder geometry."""
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
from test_regrip_return import ARM_JOINTS, DT, REST, SHARE, A, MinkKinematics, GripSessionIK, make_mapper, smooth
from teleop_v6.arm_shape import body_observation

# End of the user's 2026-11-19 stroke, not an operator skeleton recording.
HELD={"leg_waist":[0.03920745849609375,-0.03939628601074219,-0.012711454182863235,-0.0009987570811063051],"neck":[0.054641515016555786,-0.010928303003311155,0.010928303003311155],"left_arm":[21.8851318359375,15.490951538085938,-13.106477737426758,-117.32134246826172,10.83011245727539,17.8680477142334,-1.2348982095718384],"right_arm":[20.01739501953125,-0.3131103515625,-0.03294883295893669,-110.0045166015625,0.010928303003311155,0.0764981210231781,-0.054641515016555786],"left_gripper":[10],"right_gripper":[10]}

def landmarks(x,wy,wz,ey):
    # Intersect the upper-arm circle in the sagittal plane at lateral offset ey
    # with the wrist sphere. Choose the elbow-below-wrist solution.
    axis=np.array([x,wz])
    length=np.linalg.norm(axis)
    axis/=length
    upper_sq=.32**2-ey**2
    fore_sq=.26**2-(wy-ey)**2
    along=(length**2+upper_sq-fore_sq)/(2*length)
    radius=np.sqrt(max(0,upper_sq-along**2))
    e=along*axis+radius*np.array([axis[1],-axis[0]])
    return np.array([[0.,0.,0.],[e[0],ey,e[1]],[x,wy,wz]])@A

def run(side,case):
    m=MinkKinematics(str(SHARE/"urdf/robot_v3_1_simplified.urdf"),str(SHARE/"urdf/robot_v3_1.srdf"),collision_check=True)
    si=1 if side=="left" else -1
    home=m.q_from_feedback(dict(HELD,left_arm=REST,right_arm=REST))
    idx=[m.joint_q[n] for n in ARM_JOINTS[side]]
    q=home.copy()
    recorded=m.q_from_feedback(HELD)
    home_pose=m.forward_poses(home)[side]
    pose_shift=m.forward_poses(recorded)[side].translation.copy()-home_pose.translation
    origin=home_pose.translation.copy()
    if case=="captured":
        q=recorded.copy()
        origin=m.forward_poses(q)[side].translation.copy()
    mapper=make_mapper(side)
    session=GripSessionIK()
    command=q.copy()
    def elbow(conf):
        m._reference_configuration.update(m._mujoco_from_guard(conf))
        return m.elbow_plane_tasks[side]._positions(m._reference_configuration.data)[1].copy()
    elbow_home=elbow(home)
    def metrics(target):
        return dict(out_mm=round(float(1000*si*(elbow(q)-elbow_home)[1]),3),
            tcp_error_mm=round(float(1000*np.linalg.norm(m.forward_poses(q)[side].translation-target)),3),
            home_tcp_mm=round(float(1000*np.linalg.norm(m.forward_poses(q)[side].translation-home_pose.translation)),3))
    tick_count=0
    latch_jump=0
    def tick(f,free,down,body,dy=0.,end_shift=0.):
        nonlocal tick_count,latch_jump
        tick_count+=1
        offset=np.array([.24*f,free,0.])+end_shift*pose_shift
        sample=SimpleNamespace(position=A.T@offset,orientation=np.array([0.,0.,0.,1.]))
        fk=m.forward_poses(q)[side]
        state={s:s==side and down for s in ARM_JOINTS}
        stamp=tick_count*DT
        session.observe("demo",state,q,stamp)
        if not down:
            mapper.release()
            command[:]=q
            return
        edge=not mapper.latched
        if edge:
            mapper.latch(sample,dict(position=fk.translation,orientation=R.from_matrix(fk.rotation).as_quat()))
        target,_=mapper.map(sample)
        if edge:latch_jump=max(latch_jump,1000*np.linalg.norm(target["position"]-fk.translation))
        goals={side:(target["position"],target["orientation"])}
        ticket=session.prepare(goals,q,m.joint_q,stamp)
        # Hands can be separated differently from the robot shoulders.
        points=landmarks(.24+offset[0],-.035*si+offset[1],-.21+offset[2],dy*si)
        obs=body_observation(points,np.eye(3)) if body else None
        result=m.solve(goals,ticket.seed,feedback_q=q,posture_reference=ticket.reference,
            reset_reference=home,session_epoch=ticket.solver_epochs,integration_dt=ticket.dt,
            transactional=True,arm_shape={side:obs} if obs else {})
        assert result.success,(side,case,tick_count,result.reason)
        assert session.commit(ticket,result.q)
        m.commit_retarget_state()
        command[idx]=result.q[idx]
        q[idx]+=(1-np.exp(-DT/.055))*(command[idx]-q[idx])
        return target["position"]
    free=0.
    if case=="captured":
        for _ in range(130):target=tick(0,0,True,True)
    else:
        for i in range(151):target=tick(smooth(i/150),0,True,True)
        for _ in range(35):target=tick(1,0,True,True)
        for _ in range(5):
            for _ in range(12):tick(1,free,False,False)
            if case in ("shift","near_dropout","intentional"):free-=si*.009
            for _ in range(18):target=tick(1,free,True,True)
        for i in range(151):
            progress=smooth(i/150)
            target=tick(1-progress,free,True,case!="near_dropout" or i<135,
                        end_shift=progress if case=="near_dropout" else 0.)
    stop=metrics(target)
    if case=="intentional":
        for i in range(100):target=tick(0,free,True,True,.065*smooth(min(1,i/60)))
    else:
        # Simulated operator releases immediately, so there is no settling help.
        for _ in range(30):tick(0,free,False,False)
    return dict(side=side,case=case,stop=stop,end=metrics(target),latch_jump_mm=latch_jump)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--verify",action="store_true")
    args=p.parse_args()
    for side in ("left","right"):
        for case in ("still","shift","near_dropout","intentional","captured"):
            if case=="captured" and side=="right":continue
            row=run(side,case)
            print(json.dumps(row),flush=True)
            if args.verify:
                assert row["latch_jump_mm"]<1e-6,row
                assert abs(row["stop"]["out_mm"])<16 and row["stop"]["tcp_error_mm"]<4,row
                if case=="intentional":assert row["end"]["out_mm"]>35 and row["end"]["tcp_error_mm"]<4,row
