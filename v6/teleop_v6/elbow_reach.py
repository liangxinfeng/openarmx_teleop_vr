"""Invert the native elbow hinge geometry even at full extension."""
import math

import numpy as np


def bend_for_radius(shoulder, elbow, wrist, hinge, current, desired_radius, limits, neutral):
    # Rotating only this hinge sweeps the wrist around a circle. Its squared
    # distance from the shoulder is constant + B*cos(delta) + C*sin(delta).
    arm = wrist-elbow
    axial = hinge*np.dot(hinge,arm)
    spoke = arm-axial
    centre = elbow-shoulder+axial
    b = 2*np.dot(centre,spoke)
    c = 2*np.dot(centre,np.cross(hinge,spoke))
    amplitude = math.hypot(b,c)
    if amplitude < 1.e-8:
        return current
    constant = np.dot(centre,centre)+np.dot(spoke,spoke)
    opening = math.acos(float(np.clip((desired_radius**2-constant)/amplitude,-1.,1.)))
    straight = current+math.atan2(c,b)
    straight = neutral+math.remainder(straight-neutral,2*math.pi)
    # The real offset hinge has two solutions near full reach. Preserve the
    # current branch at a stationary takeover. An inward request progressively
    # selects the flexing branch even when the local Jacobian points at a stop.
    natural = straight+math.copysign(opening,neutral-straight)
    same = straight+math.copysign(opening,current-straight)
    inward = float(np.linalg.norm(wrist-shoulder))-desired_radius
    blend = float(np.clip(inward/.008,0.,1.))
    target = same + blend*(natural-same)
    return float(np.clip(target,*limits))


def elbow_reference(solver, side, idx, current, target, neutral):
    data=solver.configuration.data
    plane=solver.elbow_plane_tasks[side]
    shoulder,elbow,wrist=plane._positions(data)
    site=solver._site_ids[side]
    # Remove tool-to-wrist offset with the requested wrist orientation.
    local_offset=data.site_xmat[site].reshape(3,3).T @ (wrist-data.site_xpos[site])
    target_wrist=target[:3,3]+target[:3,:3] @ local_offset
    target_radius=float(np.linalg.norm(target_wrist-shoulder))
    radius_error=target_radius-float(np.linalg.norm(wrist-shoulder))
    hinge=data.xaxis[plane._joints["Elbow"]]
    bend=bend_for_radius(shoulder,elbow,wrist,hinge,current,target_radius,
                         (solver.lower[idx],solver.upper[idx]),neutral)
    return bend,radius_error
