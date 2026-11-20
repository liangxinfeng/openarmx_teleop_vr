"""A shared reset posture and one observable elbow-swivel coordinate per arm."""
import math
import numpy as np

from openarmx_teleop_vr_306_v5.teleop_core import VR_TO_ROBOT_ROT
from .elbow_reach import elbow_reference


def body_observation(points, operator_basis):
    """Three WebXR joint centres -> directed shoulder/wrist axis and elbow tangent."""
    try:
        points = np.asarray(points, dtype=float)
    except (ValueError, TypeError, OverflowError):
        return None
    if points.shape != (3, 3) or not np.isfinite(points).all():
        return None
    shoulder, elbow, wrist = points @ (VR_TO_ROBOT_ROT @ operator_basis).T
    axis = wrist - shoulder
    length = np.linalg.norm(axis)
    if not .05 < length < 1.4:
        return None
    axis /= length
    bend = elbow - shoulder
    bend -= np.dot(bend, axis) * axis
    radius = np.linalg.norm(bend)
    if radius < .015:
        return None
    upper = elbow-shoulder
    upper_length = np.linalg.norm(upper)
    forearm = wrist-elbow
    forearm_length = np.linalg.norm(forearm)
    if not (.05 < forearm_length < .65 and .05 < upper_length < .65):
        return None
    return dict(forearm=(forearm/forearm_length).tolist(), upper_lateral=float(upper[1]/upper_length),
                confidence=float(np.clip((radius-.015)/.055, 0., 1.)))


def direction_swivel(plane, axis, forearm, current=None, lateral_goal=None):
    """Choose a bend on the native elbow circle, preserving lateral alignment during a lift."""
    # At a fixed wrist the elbow runs on a circle. Its tangent component
    # points opposite the desired forearm, irrespective of the two arm lengths.
    bend = axis*np.dot(axis,forearm)-forearm
    neutral = plane._transport_bend(axis) if plane._reference_axis is not None else None
    if neutral is None or np.linalg.norm(bend)<1e-5:
        return None
    if current is not None and (lateral_goal is not None or forearm[2]>0):
        # v = axial*axis + radius*t, with t on a unit circle normal to axis.
        # Match the operator's lateral component on that circle; of its two
        # solutions choose the one nearest the measured human forearm.
        lateral=np.array([0.,1.,0.])-axis*axis[1]
        lateral_length=np.linalg.norm(lateral)
        axial=np.dot(axis,current)
        radius=math.sqrt(max(0.,1.-axial*axial))
        if radius*lateral_length>1e-5:
            tangent=lateral/lateral_length
            side_component=forearm[1] if lateral_goal is None else lateral_goal
            along=np.clip((side_component-axial*axis[1])/(radius*lateral_length),-1.,1.)
            across=np.cross(axis,tangent)
            other=math.copysign(math.sqrt(max(0.,1.-along*along)),across@forearm)
            upright=-(along*tangent+other*across)
            weight=max(0.,forearm[2])**2 if lateral_goal is None else 1.
            bend=(1-weight)*bend/np.linalg.norm(bend)+weight*upright
    return math.atan2(axis@np.cross(neutral,bend),neutral@bend)


def forearm_goal(plane, data, reference_data, observation, lateral_bias=0.):
    shoulder,elbow,wrist = plane._positions(data)
    axis = wrist-shoulder
    length = np.linalg.norm(axis)
    if length < .02:
        return 0., "neutral"
    axis /= length
    current=(wrist-elbow)/np.linalg.norm(wrist-elbow)
    if isinstance(observation,dict):
        try:
            direction=np.asarray(observation["forearm"],dtype=float)
            quality=float(observation["confidence"])
            if direction.shape==(3,) and abs(np.linalg.norm(direction)-1)<.01 and 0<quality<=1:
                lateral_goal=None
                upper_y=observation.get("upper_lateral")
                if isinstance(upper_y,(float,int)) and abs(upper_y)<=1:
                    # Forward strokes preserve elbow abduction, measured from
                    # the operator's shoulder independently of controller width.
                    # On an upward lift, preserve the forearm inclination.
                    upper_length=np.linalg.norm(elbow-shoulder)
                    elbow_y=shoulder[1]+upper_length*np.clip(upper_y+lateral_bias,-1.,1.)
                    elbow_constraint=(wrist[1]-elbow_y)/np.linalg.norm(wrist-elbow)
                    # Only a predominantly upright forearm transfers priority
                    # to wrist inclination. A forward stroke stays elbow-led.
                    lift=max(0.,direction[2]*abs(direction[2])-direction[:2]@direction[:2])
                    lateral_goal=(1.-lift)*elbow_constraint+lift*direction[1]
                angle=direction_swivel(plane,axis,direction,current,lateral_goal)
                if angle is not None:
                    return angle*quality,"body"
        except (ValueError,KeyError,TypeError,OverflowError):
            pass
    elif observation is not None:
        angle,valid=swivel_goal(plane,observation)
        if valid:
            return angle,"body"
    # With controllers alone elbow swivel is unobservable. Bias toward a
    # vertical forearm as the wrist rises, with zero bias at the reset pose.
    _,rest_elbow,rest_wrist=plane._positions(reference_data)
    rise=np.clip((wrist[2]-rest_wrist[2])/np.linalg.norm(rest_wrist-rest_elbow),0.,1.)
    up=direction_swivel(plane,axis,np.array([0.,0.,1.]),current)
    if up is None or rise <= 0:
        return 0.,"neutral"
    return up*rise**2*(3-2*rise),"raised_wrist"


def swivel_goal(plane, observation):
    if not isinstance(observation, (list, tuple)) or len(observation) != 3:
        return 0., False
    try:
        axis, bend = np.asarray(observation[:2], dtype=float)
        quality = float(observation[2])
    except (TypeError, ValueError, OverflowError):
        return 0., False
    if axis.shape != (3,) or not np.isfinite([axis, bend]).all() or not 0 < quality <= 1:
        return 0., False
    if abs(np.linalg.norm(axis)-1)>.01 or abs(np.linalg.norm(bend)-1)>.01 or abs(np.dot(axis,bend))>.02:
        return 0., False
    if plane._reference_axis is None or np.dot(plane._reference_axis, axis)<-.98:
        return 0., False
    neutral = plane._transport_bend(axis)
    if neutral is None:
        return 0., False
    angle = math.atan2(np.dot(axis,np.cross(neutral,bend)), np.dot(neutral,bend))
    return float(angle*quality), True


def set_arm_shape(solver, side, names, idx, seed_mj, reference, home, transform,
                  observations, dt, epoch, task_stack):
    """Place both tasks in the existing bounded QP. No second joint writer."""
    joints = [solver._mj_q[n] for n in names]
    dofs = [solver._mj_v[n] for n in names]
    posture = seed_mj.copy()
    posture[joints] = (home if home is not None else reference)[idx]
    solver._reference_configuration.update(posture)
    data = solver._reference_configuration.data
    plane = solver.elbow_plane_tasks[side]
    plane.set_reference(data)
    observation = observations.get(side) if isinstance(observations,dict) else None
    # Grip edges reset the servo seed, not the anatomical reference or its filter.
    session = epoch[0] if isinstance(epoch,tuple) else epoch
    history = solver._swivel_history.get(side)
    if history is None or history[0] != session:
        shoulder,elbow,wrist=plane._positions(solver.configuration.data)
        axis=wrist-shoulder
        axis/=max(np.linalg.norm(axis),1e-8)
        forearm=wrist-elbow
        # At near-full reach a millimetre of transverse motion can reverse the
        # observed elbow plane. Seed it only to the extent the bend is visible.
        lever=np.linalg.norm(forearm-axis*(axis@forearm))
        confidence=float(np.clip((lever-.015)/.055,0.,1.))
        angle=confidence*(direction_swivel(plane,axis,forearm) or 0.)
        missing=.25
        # Align the human and robot upper-arm origins once per session.
        # A Grip edge never recalibrates the elbow from the extended pose.
        lateral_bias=None
        last_target=transform[:3,3].copy()
        last_epoch=epoch
        last_observation=None
    else:
        _,angle,missing,lateral_bias,last_target,last_epoch,last_observation=history
    if lateral_bias is None:
        human_y=observation.get("upper_lateral") if isinstance(observation,dict) else None
        shoulder,elbow,_=plane._positions(data)
        upper=elbow-shoulder
        rest_lateral=upper[1]/np.linalg.norm(upper)
        if np.max(np.abs(seed_mj[joints]-posture[joints]))>math.radians(3):
            # First reliable landmarks arrived away from home: use the native
            # neutral geometry; never learn the error of an already flared arm.
            lateral_bias=rest_lateral
        elif isinstance(human_y,(float,int)) and abs(human_y)<=1:
            lateral_bias=rest_lateral-human_y
    bias=0. if lateral_bias is None else lateral_bias
    goal,source=forearm_goal(plane,solver.configuration.data,data,observation,bias)
    tracked=source=="body"
    missing = 0. if tracked else missing + dt
    if tracked:
        last_target=transform[:3,3].copy()
        last_epoch=epoch
        last_observation=observation
    elif last_epoch==epoch and last_observation is not None and missing < .25:
        # Reproject the last anatomical observation on the current elbow circle.
        # An angle solved at an older wrist pose belongs to a different circle.
        travel=np.linalg.norm(transform[:3,3]-last_target)
        confidence=max(0.,1.-travel/.04)
        held,_=forearm_goal(plane,solver.configuration.data,data,last_observation,bias)
        goal+=confidence*math.remainder(held-goal,2*math.pi)
        if confidence>0:
            source = "last_body"
    delta = math.remainder(goal-angle, 2*math.pi)
    angle += float(np.clip(-math.expm1(-dt/.085)*delta, -2.8*dt, 2.8*dt))
    angle = math.remainder(angle,2*math.pi)
    solver._swivel_history[side] = (session,angle,missing,lateral_bias,last_target,last_epoch,last_observation)
    if plane._reference_axis is not None:
        canonical = plane._reference_bend
        plane._reference_bend = math.cos(angle)*canonical + math.sin(angle)*np.cross(plane._reference_axis,canonical)
    task_stack.append(plane)
    site = solver._site_ids[side]
    distance = float(np.linalg.norm(transform[:3,3]-data.site_xpos[site]))
    rotation = data.site_xmat[site].reshape(3,3).T @ transform[:3,:3]
    theta = math.acos(float(np.clip((np.trace(rotation)-1)/2,-1.,1.)))
    # At the same hand position a turned wrist or deliberately raised elbow
    # should still be followed. All seven axes share the neutral preference.
    proximity = math.exp(-(distance/.10)**2-(theta/.55)**2-(angle/.50)**2)
    weight = solver._posture_weight*(1+3*proximity)
    costs = np.zeros(solver.mujoco_model.nv)
    costs[dofs] = weight*np.array([1.,1.,2.,1.,2.,1.,1.])
    bend, radial_error = elbow_reference(solver,side,idx[3],seed_mj[joints[3]],
                                         transform,posture[joints[3]])
    # The Cartesian Jacobian has almost no radial leverage at a straight
    # elbow. Supply the finite bend in the same bounded QP in both directions.
    posture[joints[3]] = bend
    costs[dofs[3]] = .12
    task = solver.posture_tasks[side]
    task.set_target(posture)
    task.set_cost(costs)
    task_stack.append(task)
    return dict(shape_source=source, swivel_deg=math.degrees(angle),
                requested_swivel_deg=math.degrees(goal),return_distance_m=distance,
                return_blend=proximity,posture_cost=weight,elbow_plane_cost=float(plane.cost[0]),
                elbow_goal_deg=math.degrees(bend),wrist_radius_error_m=radial_error,
                target_position_m=transform[:3,3].tolist())
