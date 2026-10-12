"""Compose the physical wrist RGB optical frame from measured robot joints."""
import hashlib
import json
from pathlib import Path
import numpy as np

from ..interfaces.contract import validate_sensor_packet, validate_rgb_camera
from ..interfaces.wrist_geometry import CAMERAS, BASE, rigid_transform
from ..robot.assembly import load_assembly
from .observations import HardwareKinematics


class WristGeometry:
    def __init__(self, path, serial, urdf, assembly, max_skew_s=.08):
        raw = Path(path).read_bytes()
        self.calibration_bytes = raw
        calibration = json.loads(raw)
        side = next((side for side, identity in CAMERAS.items() if identity[0] == str(serial)), None)
        if side is None:
            raise ValueError('Wrist calibration identity mismatch: unknown serial')
        serial, parent = CAMERAS[side]
        expected = dict(camera_serial=serial, parent_frame=parent, wrist_frame=parent,
                        base_frame=BASE, camera_frame=f'realsense_{serial}_color_optical', calibration_type='eye_in_hand', translation_units='metres',
                        calibration_status='candidate_requires_physical_validation',
                        simulation_only=False, independent_physical_validation_performed=False)
        if any(calibration.get(k) != v for k, v in expected.items()):
            raise ValueError('Wrist calibration identity/status mismatch')
        matrix = rigid_transform(calibration['T_parent_from_optical'])
        if not np.allclose(matrix, calibration['T_wrist_from_camera'], atol=1e-10, rtol=0):
            raise ValueError('Wrist calibration transform aliases disagree')
        if not np.isfinite(max_skew_s) or not 0 < max_skew_s <= .08:
            raise ValueError('Invalid wrist encoder matching bound')
        self.max_skew_s = max_skew_s
        self.kin = HardwareKinematics(urdf, load_assembly(assembly), {'wrist': calibration}, 'wrist')
        mount = self.kin.model.frames[self.kin.parents['wrist']]
        tcp = self.kin.model.frames[self.kin.tcp_frames[side]]
        if mount.parent != tcp.parent:
            raise ValueError('Wrist camera and corresponding TCP must share a rigid parent joint')
        tcp_from_camera = np.linalg.solve(tcp.placement.homogeneous,
                                          mount.placement.homogeneous @ matrix)
        # Export only camera calibration, never board placement or dataset landmarks.
        self.metadata = dict(arm=side, camera_serial=serial, camera_frame=f'realsense_{serial}_color_optical',
            parent_frame=parent, base_frame=BASE, calibration_status=calibration['calibration_status'],
            calibration_sha256=hashlib.sha256(raw).hexdigest(), T_parent_from_optical=matrix.tolist(),
            encoder_max_skew_s=max_skew_s, distortion_model='none',
            heldout_reprojection_rms_px=calibration['validation_reprojection']['rms_px'],
            calibration_image_size=[calibration['camera']['width'], calibration['camera']['height']])

        self.metadata[f'{side}_tcp_T_from_wrist_optical'] = rigid_transform(tcp_from_camera).tolist()

    def observation(self, rgb, received_s, measured, intrinsics, image_processing=None):
        validate_sensor_packet(measured)
        self.kin.update(measured['joint_positions'])
        geometry = dict(self.metadata, fixed_camera_intrinsics=intrinsics,
                        encoder_timestamp_s=measured['timestamp_s'],
                        joint_positions=dict(measured['joint_positions']), **self.kin.packet())
        if image_processing is not None:
            geometry['image_processing'] = image_processing
        value = dict(rgb=rgb, received_s=received_s, geometry=geometry)
        validate_rgb_camera(value)
        return value


def color_intrinsics(profile):
    """Use the active native RGB profile; never scale calibration-profile K."""
    intr = profile.as_video_stream_profile().get_intrinsics()
    if not np.isfinite(intr.coeffs).all() or np.any(np.asarray(intr.coeffs) != 0):
        raise ValueError('Wrist RGB requires zero distortion; nonzero native distortion needs rectification')
    return [[float(intr.fx), 0., float(intr.ppx)],
            [0., float(intr.fy), float(intr.ppy)], [0., 0., 1.]]
