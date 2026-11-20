"""Preserve rotational authority for pure wrist turns and inward retreat."""
import numpy as np


def orientation_cost(nominal, extension, outward_projection, requested_distance):
    conflict = np.clip(outward_projection / max(requested_distance, .001), 0., 1.)
    return nominal * (1. - .75 * extension * conflict)
