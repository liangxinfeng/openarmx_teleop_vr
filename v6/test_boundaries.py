from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from teleop_v6.mapping import IncrementalArm, SessionHeading
from teleop_v6.session import SessionMailbox
from teleop_v6.lifecycle import ReleaseGate


def test_epoch_rejects_previous_grip():
    m=SessionMailbox()
    m.put(1)
    ticket,_=m.take()
    m.put(2)
    m.put(3)
    assert m.is_latest(ticket) and m.take()[1]==3
    m.reset()
    assert not m.is_latest(ticket)
    m.put(4)
    m.close()
    assert m.take() is None


def test_exit_waits_for_acquisition_and_ack():
    entry,exit_ack=Future(),Future()
    gate=ReleaseGate()
    gate.begin(entry,10,now=0)
    gate.reenter=True
    def release():
        assert entry.done()
        return exit_ack
    assert not gate.tick(release,1)
    entry.set_result(True)
    assert not gate.tick(release,2)
    exit_ack.set_result(SimpleNamespace(success=True))
    assert gate.tick(release,3) and not gate.pending
    assert not gate.tick(release,4)


def test_failed_ack_requires_retry_and_does_not_resume():
    gate=ReleaseGate()
    gate.begin(None,1,now=0)
    gate.reenter=True
    failed=Future()
    failed.set_exception(RuntimeError("fake vendor unavailable"))
    gate.tick(lambda:failed,2)
    gate.tick(lambda:failed,3)
    assert gate.failed and gate.pending and not gate.reenter
    gate.retry(10)
    ok=Future()
    ok.set_result(SimpleNamespace(success=True))
    gate.tick(lambda:ok)
    assert not gate.tick(lambda:ok) and not gate.pending


def test_mapper_boundary_retreat_reclutch_and_wrist():
    a=IncrementalArm(side="left",position_scale=1.,filter_alpha=1.,maximum_position_step=.03,
                    maximum_orientation_step=.2,maximum_anchor_displacement=.5,
                    workspace_minimum=[0,-1,0],workspace_maximum=[.6,1,1],
                    left_y_minimum=-1,right_y_maximum=1,forbidden_box_minimum=[2,2,2],
                    forbidden_box_maximum=[3,3,3],wrist_orientation_gain=1.18)
    def hand(z=0,angle=0):
        return SimpleNamespace(position=np.array([0.,0.,z]),orientation=Rotation.from_euler("x",angle,degrees=True).as_quat())
    anchor=dict(position=[.5,.2,.5],orientation=[0.,0.,0.,1.])
    a.latch(hand(),anchor)
    np.testing.assert_allclose(a.map(hand(),[1,1,1])[0]["position"],anchor["position"])
    assert abs(a.map(hand(-.3))[0]["position"][0]-.6)<1e-9
    # A 30 cm input stroke has 10 cm of available travel. Retreat starts
    # on the very first sample and still ends exactly at the measured origin.
    np.testing.assert_allclose(a.map(hand(-.29))[0]["position"][0], .5+.29/3)
    np.testing.assert_allclose(a.map(hand())[0]["position"],anchor["position"])
    a.latch(hand(-.29),anchor)
    np.testing.assert_allclose(a.map(hand(-.29))[0]["position"],anchor["position"])
    turns=[Rotation.from_quat(a.map(hand(-.29,deg))[0]["orientation"]) for deg in [179.9,180.1,0]]
    assert (turns[0].inv()*turns[1]).magnitude()<.01
    assert turns[2].magnitude()<1e-10


def test_later_hand_uses_same_heading_until_next_session():
    heading=SessionHeading()
    heading.update([0.,0.,0.,1.])
    first=heading.value()
    heading.update(Rotation.from_euler("y",75,degrees=True).as_quat())
    assert first==heading.value() and abs(heading.yaw-first)>70


def test_repeated_takeover_during_return_and_session_reset():
    from test_regrip_return import make_mapper
    home = dict(position=np.array([.35,-.28,.9]),orientation=Rotation.identity().as_quat())
    travel = np.array([.03,.05,-.16])
    free_shift = np.array([.03,-.01,.01])
    wrist = Rotation.from_rotvec([.22,-.16,.31])
    def hand(f, offset, twist):
        return SimpleNamespace(position=f*travel+offset,
            orientation=(Rotation.from_rotvec(f*wrist.as_rotvec())*twist).as_quat())
    def same_pose(a,b):
        np.testing.assert_allclose(a["position"],b["position"],atol=1e-9)
        assert (Rotation.from_quat(a["orientation"])*Rotation.from_quat(b["orientation"]).inv()).magnitude()<1e-8
    for frame in ("room","clutch"):
        arm=make_mapper("right",frame)
        shift=np.zeros(3)
        twist=Rotation.identity()
        arm.latch(hand(0,shift,twist),home)
        for f in (1.,.75,.5,.25):
            target,_=arm.map(hand(f,shift,twist))
            measured=dict(position=target["position"]+[.003,-.002,.001],
                          orientation=(Rotation.from_rotvec([.02,-.01,.01])*Rotation.from_quat(target["orientation"])).as_quat())
            arm.release()
            shift=shift+free_shift/4
            twist=Rotation.from_rotvec([.012,0.,0.])*twist
            sample=hand(f,shift,twist)
            arm.latch(sample,measured)
            for _ in range(3):
                same_pose(arm.map(sample)[0],measured)
        end=hand(0,np.zeros(3),Rotation.identity()) if frame=="room" else hand(0,shift,twist)
        same_pose(arm.map(end)[0],home)
        arm.reset_reference()
        other=dict(position=np.array([.52,-.32,.75]),orientation=wrist.as_quat())
        arm.latch(end,other)
        same_pose(arm.map(end)[0],other)
        arm.map(hand(1,shift,twist))
        same_pose(arm.map(end)[0],other)


def test_short_stroke_does_not_amplify_a_tracking_error():
    from teleop_v6.mapping import remaining_offset
    home=np.zeros(3)
    anchor=np.array([.0001,0,0])
    weight=remaining_offset(home,anchor,home,.025)
    assert np.isclose((1-weight)*.025,.0001)


def test_forearm_reproduces_reachable_upward_and_sideways_directions():
    from teleop_v6.arm_shape import direction_swivel
    axis=np.array([1.,.12,-.6])
    axis/=np.linalg.norm(axis)
    base=np.cross([0.,1.,0.],axis)
    base/=np.linalg.norm(base)
    plane=SimpleNamespace(_reference_axis=axis,_transport_bend=lambda a:base)
    for turn in (-.6,0.,.6,1.4):
        bend=Rotation.from_rotvec(turn*axis).apply(base)
        desired=.35*axis-np.sqrt(1.-.35**2)*bend
        angle=direction_swivel(plane,axis,desired,desired)
        np.testing.assert_allclose(Rotation.from_rotvec(angle*axis).apply(base),bend,atol=1e-7)


def test_body_shape_requires_a_bend_and_uses_the_same_operator_basis():
    from teleop_v6.arm_shape import body_observation
    from openarmx_teleop_vr_306_v5.teleop_core import VR_TO_ROBOT_ROT
    basis=Rotation.from_euler("y",45,degrees=True).as_matrix()
    points=np.array([[.2,1.3,0],[.2,1.1,0],[.2,1.1,-.25]])
    observed=body_observation(points,basis)
    expected=VR_TO_ROBOT_ROT @ basis @ (points[2]-points[1])
    np.testing.assert_allclose(observed["forearm"],expected/np.linalg.norm(expected))
    assert observed["confidence"]==1
    straight=[points[0],(points[0]+points[2])/2,points[2]]
    for bad in (None,{},[1,2],np.full((3,3),np.nan),np.zeros((3,3)),straight):
        assert body_observation(bad,basis) is None


def test_offset_elbow_regrip_and_radial_retreat():
    from teleop_v6.elbow_reach import bend_for_radius
    shoulder=np.zeros(3)
    elbow=np.array([.30,0.,0.])
    axis=np.array([0.,0.,1.])
    wrist=elbow+np.array([.2,.025,0.])
    current,neutral=np.deg2rad([-2.,-110.])
    limits=np.deg2rad([-148.,-2.])
    reaches=lambda radius:bend_for_radius(shoulder,elbow,wrist,axis,current,radius,limits,neutral)
    assert abs(reaches(np.linalg.norm(wrist))-current)<1e-8
    radius=np.linalg.norm(wrist)-.04
    goal=reaches(radius)
    predicted=elbow+Rotation.from_rotvec(axis*(goal-current)).apply(wrist-elbow)
    assert abs(np.linalg.norm(predicted)-radius)<1e-8 and goal<current-.2
