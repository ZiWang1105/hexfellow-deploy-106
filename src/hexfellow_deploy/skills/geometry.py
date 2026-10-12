"""Conservative pad-envelope checks using this robot's configured assembly.

This is the approximate GP80 contact-pad model, not a whole-hand/arm collision
checker or a calibrated physical clearance guarantee. Scene heights are supplied
by the policy; no scene or object state is read.
"""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation


def pad_min_z(position, rotation, gripper):
    half = np.asarray(gripper['pad_half_size_m'], float)
    # Cover both jaws throughout their full travel; no inferred measured aperture.
    tool_tcp = np.asarray(gripper['tcp_in_tool_m'], float)
    if half.shape != (3,) or tool_tcp.shape != (3,) or not np.isfinite([*half,*tool_tcp]).all():
        raise ValueError('Invalid GP80 pad geometry')
    corners=[]
    for sign, opening, vertex in product((-1,1), (0., float(gripper['opening_m'])/2), product((-1,1),repeat=3)):
        center=np.array([0.,sign*(half[1]+opening),.130])-tool_tcp
        corners.append(center+half*np.asarray(vertex))
    return float(np.min((np.asarray(rotation) @ np.asarray(corners).T)[2])+position[2])


def check_pad_floor(pair, arms, floor, gripper):
    for side in arms:
        pose=pair[side]
        if pad_min_z(pose['position_m'], Rotation.from_quat(pose['quaternion_xyzw']).as_matrix(), gripper) < floor:
            raise ValueError(f'{side} GP80 pad envelope falls below requested rim clearance')
