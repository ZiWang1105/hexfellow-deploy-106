"""Measured RGB-D camera views and encoder-derived extrinsics only."""
from dataclasses import replace
import time

import numpy as np
import pinocchio as pin

from ..robot.kinematics import Kinematics


class HardwareKinematics(Kinematics):
    def __init__(self, urdf, assembly, calibrations, primary):
        super().__init__(urdf, assembly)
        self.calibrations, self.primary = calibrations, primary
        self.parents = {}
        for name, calibration in calibrations.items():
            frame = self.model.getFrameId(calibration['parent_frame'])
            if frame >= self.model.nframes:
                raise ValueError(f'Unknown calibration parent: {calibration["parent_frame"]}')
            self.parents[name] = frame

    def camera_transform(self, name):
        if name not in self.calibrations:
            return super().camera_pose()  # explicit synthetic/mock path only
        return self.data.oMf[self.parents[name]] * pin.SE3(
            np.asarray(self.calibrations[name]['T_parent_from_optical'], dtype=float))

    def camera_pose(self):
        return self.camera_transform(self.primary)


def camera_intrinsics(frame):
    k = frame.intrinsics
    if 'fx' not in k:
        if k.get('frame') != 'synthetic':
            raise ValueError('Missing physical camera intrinsics')
        return [[160., 0., 80.], [0., 160., 60.], [0., 0., 1.]]
    return [[k['fx'], 0., k['cx']], [0., k['fy'], k['cy']], [0., 0., 1.]]


def observe_cameras(runtime, streams, config):
    with runtime.lock:
        observation = runtime.observe()
        kin = runtime.session.executor.kinematics
        kin.update(observation.proprioception['joint_positions'])
        additional = {}
        for name, stream in streams.items():
            if name == config['policy_camera']:
                continue
            status, frame = stream.snapshot()
            if frame is None or time.monotonic() - frame.received_s > config['camera_max_age_s']:
                raise RuntimeError(f'{name} camera unavailable/stale')
            if abs(frame.received_s - observation.timestamp_s) > config['camera_encoder_max_skew_s']:
                raise RuntimeError(f'{name}: camera/encoder receipt-time skew exceeded')
            additional[name] = dict(rgb=frame.rgb, depth_m=frame.depth_m,
                intrinsics=camera_intrinsics(frame),
                T_base_from_optical=kin.camera_transform(name).homogeneous.tolist())
        return replace(observation, cameras=additional)
