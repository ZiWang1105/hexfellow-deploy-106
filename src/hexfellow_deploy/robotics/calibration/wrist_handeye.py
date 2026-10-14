"""Eye-in-hand calibration with explicit source-to-destination transforms.

G_i = T_base_from_wrist_link(q_measured_i)
C_i = T_camera_color_optical_from_board from image PnP
X   = T_wrist_link_from_camera_color_optical, fixed on the wrist
Y   = T_base_from_board, fixed throughout acquisition

The physical closure is G_i @ X @ C_i = Y.  The existing eye-to-hand
optimizer solves X @ C_i = g_i @ Y, so this module passes g_i=inverse(G_i).
Inverting G here also makes its Park initializer receive the original
gripper-to-base poses, as required by OpenCV's eye-in-hand convention.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .handeye import inverse, residual, solve as _solve, validate

DEFAULT_PARENT_FRAME = 'maver_x4_asm_mount2'
DEFAULT_BASE_FRAME = 'maver_x4_asm_maver_x4_base_link'


def _pairs(g, c):
    g, c = validate(g), validate(c)
    if len(g) != len(c) or not len(g):
        raise ValueError('Expected nonempty paired wrist FK and board PnP transforms')
    return g, c


def summarize(g, c, x, y):
    """Report G_i X C_i versus Y in base axes, in mm and degrees."""
    g, c = _pairs(g, c)
    x, y = validate([x, y])
    predicted = g @ x @ c
    position = np.linalg.norm(predicted[:, :3, 3] - y[:3, 3], axis=1) * 1000
    angle = np.rad2deg(Rotation.from_matrix(
        y[:3, :3].T @ predicted[:, :3, :3]).magnitude())
    return {
        'count': len(g),
        'closure': 'T_base_from_wrist @ T_wrist_from_camera @ T_camera_from_board = T_base_from_board',
        'position_mm': position.tolist(),
        'angle_deg': angle.tolist(),
        'position_rms_mm': float(np.sqrt(np.mean(position ** 2))),
        'position_max_mm': float(position.max()),
        'angle_rms_deg': float(np.sqrt(np.mean(angle ** 2))),
        'angle_max_deg': float(angle.max()),
    }


def camera_from_board(g, x, y):
    """Predict board PnP transform from FK and the two fitted constants."""
    g = validate(g)
    x, y = validate([x, y])
    return np.asarray([inverse(x) @ inverse(t) @ y for t in g])


def solve(g, c, *, parent_frame=DEFAULT_PARENT_FRAME, camera_serial='243222074562',
          base_frame=DEFAULT_BASE_FRAME, camera_frame=None,
          validation_g=None, validation_c=None, **solver_options):
    """Park seed and robust joint gradient descent for wrist X and board Y.

    Validation observations are scored after fitting, never used by the seed
    or refinement.  This fits image/encoder evidence; it does not establish
    that the robot geometry, board scale, timing, or physical mounting is right.
    The caller must retain those checks and independently verify the candidate.
    """
    if not isinstance(parent_frame, str) or not parent_frame.strip():
        raise ValueError('An explicit mounting wrist URDF parent frame is required')
    if not isinstance(base_frame, str) or not base_frame.strip():
        raise ValueError('An explicit robot URDF base frame is required')
    g, c = _pairs(g, c)
    inverse_g = np.asarray([inverse(t) for t in g])
    result = _solve(inverse_g, c, **solver_options)
    # These old names identify the generic equation, not physical wrist frames.
    # Remove them to avoid silently consuming a head-camera transform.
    x = np.asarray(result.pop('T_mount_from_camera'))
    y = np.asarray(result.pop('T_hand_from_board'))
    weighted = residual(inverse_g, c, x, y,
                        result['position_sigma_m'], result['angle_sigma_rad']).reshape(-1, 6)
    robust_weights = 1 / np.sqrt(1 + (weighted / 2) ** 2)
    result.update({
        'calibration_type': 'eye_in_hand',
        'parent_frame': parent_frame,
        'wrist_frame': parent_frame,
        'base_frame': base_frame,
        'camera_frame': camera_frame or f'realsense_{camera_serial}_color_optical',
        'camera_serial': str(camera_serial),
        'optical_axes': 'x right, y down, z forward',
        'translation_units': 'metres',
        'transform_convention': 'T_destination_from_source maps homogeneous source points into destination coordinates',
        'equation': 'G_i @ X @ C_i = Y',
        'variables': {
            'G_i': 'T_base_from_wrist from synchronized measured-joint FK',
            'C_i': 'T_camera_from_board from the same camera color image PnP',
            'X': 'T_wrist_from_camera (constant)',
            'Y': 'T_base_from_board (constant during acquisition)',
        },
        'optimizer_substitution': 'existing X @ C_i = g_i @ Y solver receives g_i = inverse(G_i)',
        'T_parent_from_optical': x.tolist(),
        'T_wrist_from_camera': x.tolist(),
        'T_base_from_board': y.tolist(),
        'runtime_camera_base': 'T_base_from_camera(q_measured) = FK(base_from_parent, q_measured) @ T_parent_from_optical',
        'training': summarize(g, c, x, y),
        'final_robust_weights': {
            'formula': '1 / sqrt(1 + (scaled_residual / 2)^2)',
            'order': ['translation_x', 'translation_y', 'translation_z', 'rotation_x', 'rotation_y', 'rotation_z'],
            'values': robust_weights.tolist(),
            'minimum': float(robust_weights.min()),
        },
        'head_extrinsics_used_for_fit': False,
        'evidence': 'paired measured wrist FK and camera board PnP; camera intrinsics are supplied separately',
        'calibration_status': 'candidate_requires_independent_physical_validation',
        'independent_physical_validation_performed': False,
    })
    if (validation_g is None) != (validation_c is None):
        raise ValueError('Both held-out wrist FK and board PnP arrays are required')
    if validation_g is not None:
        result['validation'] = summarize(validation_g, validation_c, x, y)
    return result


# Descriptive aliases are convenient for modules importing both calibrations.
solve_wrist = solve
summarize_wrist = summarize
