import numpy as np

from . import math


def fix_collisions(vc, vb, fb, eps=0.002):
    """
    Fix the collisions between the clothing and the body by projecting
    the clothing's vertices outside the body's surface
    """

    # Compute body normals
    nb = math.compute_vertex_normals(vb, fb)

    # For each vertex of the cloth, find the closest vertices in the body's surface
    closest_vertices = math.find_nearest_neighbour(vc, vb)
    vb = vb[closest_vertices] 
    nb = nb[closest_vertices] 

    # Test penetrations
    penetrations = np.sum(nb*(vc - vb), axis=1) - eps
    penetrations = np.minimum(penetrations, 0)

    # Fix the clothing
    corrective_offset = -np.multiply(nb, penetrations[:,np.newaxis])
    vc_fixed = vc + corrective_offset

    return vc_fixed
