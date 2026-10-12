"""Allowlisted robot calibration and time-matched encoder geometry for wrist RGB."""
import re
import numpy as np
from scipy.spatial.transform import Rotation
from ..robot.joints import JOINTS

CAMERAS = {
    'left': ('230422271628', 'maver_x4_asm_mount1'),
    'right': ('243222074562', 'maver_x4_asm_mount2'),
}
# Historical right-camera constants remain available for recorded observations.
SERIAL = '243222074562'
PARENT = 'maver_x4_asm_mount2'
BASE = 'maver_x4_asm_maver_x4_base_link'
FRAME = f'realsense_{SERIAL}_color_optical'
FIELDS = {'arm', 'camera_serial', 'camera_frame', 'parent_frame', 'base_frame',
          'calibration_status', 'calibration_sha256', 'T_parent_from_optical',
          'fixed_camera_intrinsics', 'camera_T_base_from_optical', 'encoder_timestamp_s',
          'joint_positions', 'tcp_poses', 'encoder_max_skew_s',
          'heldout_reprojection_rms_px', 'calibration_image_size', 'distortion_model'}


def rigid_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Invalid rigid camera transform')
    return matrix


def validate_wrist_geometry(geometry, received_s):
    if not isinstance(geometry, dict) or geometry.get('arm') not in CAMERAS:
        raise ValueError('Wrist geometry camera identity mismatch')
    side = geometry['arm']
    tcp_key = f'{side}_tcp_T_from_wrist_optical'
    optional = {tcp_key, 'image_processing'}
    if not FIELDS <= set(geometry) or set(geometry) - FIELDS - optional:
        raise ValueError('Wrist geometry violates the sensor allowlist')
    serial, parent = CAMERAS[side]
    if geometry.get('image_processing', 'native_pinhole') not in ('native_pinhole', 'sdk_rectified_pinhole'):
        raise ValueError('Invalid wrist image processing')
    identities = dict(arm=side, camera_serial=serial, parent_frame=parent,
                      base_frame=BASE, camera_frame=f'realsense_{serial}_color_optical', distortion_model='none',
                      calibration_status='candidate_requires_physical_validation')
    if any(geometry[k] != v for k, v in identities.items()):
        raise ValueError('Wrist geometry camera identity or calibration status mismatch')
    if not re.fullmatch('[0-9a-f]{64}', geometry['calibration_sha256']):
        raise ValueError('Invalid wrist calibration hash')
    for key in ('T_parent_from_optical', 'camera_T_base_from_optical'):
        rigid_transform(geometry[key])
    k = np.asarray(geometry['fixed_camera_intrinsics'], dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('Invalid wrist camera intrinsics')
    stamp, bound = geometry['encoder_timestamp_s'], geometry['encoder_max_skew_s']
    if (not np.isfinite([stamp, bound, received_s]).all() or not 0 < bound <= .08
            or abs(stamp-received_s) > bound):
        raise ValueError('Wrist camera encoder skew exceeded')
    joints = geometry['joint_positions']
    if set(joints) != set(JOINTS) or not np.isfinite(list(joints.values())).all():
        raise ValueError('Wrist geometry requires physical joint encoders only')
    poses = geometry['tcp_poses']
    if set(poses) != {'left', 'right'}:
        raise ValueError('Wrist geometry requires both TCP poses')
    for pose in poses.values():
        if set(pose) != {'position_m', 'quaternion_xyzw'}:
            raise ValueError('Invalid wrist TCP pose fields')
        p, q = np.asarray(pose['position_m']), np.asarray(pose['quaternion_xyzw'])
        if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all()
                or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-6):
            raise ValueError('Invalid wrist TCP pose')
    # Older recorded wrist observations omit the fixed TCP transform.
    if tcp_key in geometry:
        tcp_from_camera = rigid_transform(geometry[tcp_key])
        base_from_tcp = np.eye(4)
        base_from_tcp[:3, :3] = Rotation.from_quat(poses[side]['quaternion_xyzw']).as_matrix()
        base_from_tcp[:3, 3] = poses[side]['position_m']
        if not np.allclose(base_from_tcp @ tcp_from_camera, geometry['camera_T_base_from_optical'],
                           atol=1e-8, rtol=0):
            raise ValueError('Wrist camera transform differs from measured FK composition through corresponding TCP')
    rms = geometry['heldout_reprojection_rms_px']
    size = geometry['calibration_image_size']
    if (not isinstance(rms, (float, int)) or not np.isfinite(rms) or rms < 0
            or not isinstance(size, list) or len(size) != 2
            or any(type(x) is not int or x <= 0 for x in size)):
        raise ValueError('Invalid wrist calibration quality metadata')
