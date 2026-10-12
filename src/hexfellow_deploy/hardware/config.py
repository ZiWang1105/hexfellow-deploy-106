"""Validate deployment without opening devices; SDK imports use camera interpreters."""
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import yaml

from .device import BinaryMotorTargets, gp80_encoder_tolerance
from .camera_process import check_camera_sdk, resolve_camera_config
from ..control.velocity_limits import arm_velocity_bounds
from ..robot.upstream import load_umi


def load_camera_calibration(path):
    """Read native UMI camera results without copying or promoting their status."""
    path = Path(path)
    calibration = json.loads(path.read_text())
    if 'T_mount_from_camera' in calibration:
        if calibration.get('camera_frame') != 'rectified LEFT optical; x right, y down, z forward':
            raise ValueError(f'{path}: unsupported UMI camera frame')
        calibration = dict(calibration,
            parent_frame=calibration['mount_frame'],
            T_parent_from_optical=calibration['T_mount_from_camera'])
    # Acceptance annotations are evidence metadata, not runtime interlocks.
    # Selecting this file explicitly selects its matrix without promoting status.
    matrix = np.asarray(calibration['T_parent_from_optical'], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f'{path}: invalid rigid camera transform')
    if not isinstance(calibration.get('parent_frame'), str) or not calibration['parent_frame']:
        raise ValueError(f'{path}: missing camera parent frame')
    return calibration


def load_deployment(path, *, execute=False, mock=False, research=False):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Deployment schema_version must be 1')
    def resolve(value):
        return str((path.parent / value).resolve())
    for name in ('controller', 'base_motion', 'cameras'):
        data[name] = resolve(data[name])
    for name in ('urdf', 'assembly', 'mapping'):
        data['robot'][name] = resolve(data['robot'][name])
    for name in ('research_release', 'verification_report'):
        if data['robot'].get(name):
            data['robot'][name] = resolve(data['robot'][name])
    data['robot']['controller'] = data['controller']
    # This option comes from the explicit operator CLI, never a persisted flag.
    data['robot']['umi_research'] = bool(research)
    cameras = yaml.safe_load(Path(data['cameras']).read_text())['cameras']
    data['camera_configs'] = {n: resolve_camera_config(c, Path(data['cameras']).parent)
                              for n, c in cameras.items() if c.get('enabled', True)}
    if data['robot']['policy_camera'] not in data['camera_configs']:
        raise ValueError('Policy camera must be enabled')
    for name, camera in data['camera_configs'].items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Camera names must be alphanumeric/underscore')
        if camera.get('type') not in ('zed', 'realsense'):
            raise ValueError('Camera must be zed or realsense')
    required_limits = {'arm_velocity_rad_s', 'lift_velocity_m_s', 'arm_following_error_rad',
                       'lift_following_error_m', 'base_linear_m_s', 'base_angular_rad_s',
                       'base_acceleration_m_s2', 'base_angular_acceleration_rad_s2'}
    absent = required_limits - data['robot']['limits'].keys()
    if absent:
        raise ValueError('Missing hardware limits: ' + ', '.join(sorted(absent)))
    arm_setting = data['robot']['limits']['arm_velocity_rad_s']
    for name, value in data['robot']['limits'].items():
        if name == 'arm_velocity_rad_s':
            continue  # Resolved against the controller's per-joint profile below.
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid hardware limit: {name}')
    controller = yaml.safe_load(Path(data['controller']).read_text())
    load_umi()
    from policy_sim.control.motion_profile import load_motion_profile
    profile = load_motion_profile(controller['motion_profile'])
    hardware_bounds = arm_velocity_bounds(arm_setting, profile)
    controller_bounds = arm_velocity_bounds(controller['arm_velocity_limit_rad_s'], profile)
    if (any(controller_bounds[n] > hardware_bounds[n] for n in hardware_bounds) or
            controller['lift_velocity_limit_m_s'] > data['robot']['limits']['lift_velocity_m_s']):
        raise ValueError('UMI velocity limits exceed hardware configuration')
    data['robot']['arm_velocity_bounds_rad_s'] = hardware_bounds
    data['robot']['arm_velocity_limit_source'] = arm_setting
    # Scalar summary remains usable by the standalone keyboard CLI; command
    # admission uses the named per-joint bounds, never this maximum for all joints.
    data['robot']['limits']['arm_velocity_rad_s'] = max(hardware_bounds.values())
    for field in ('feedback_max_age_s', 'feedback_max_skew_s', 'command_ttl_s',
                  'startup_command_timeout_s', 'camera_encoder_max_skew_s', 'camera_max_age_s'):
        value = data['robot'][field]
        if not np.isfinite(value) or not 0 < value <= 5:
            raise ValueError(f'Invalid {field}')
    if data['robot']['command_ttl_s'] > .1:
        raise ValueError('Hardware command TTL must not exceed 100 ms')
    if not mock:
        missing = []
        if execute:
            from hexfellow_deploy.gripper_sdk import require_closure_enabled
            for side in ('left', 'right'):
                try:
                    require_closure_enabled(data['robot']['grippers'][side], 0, side=side)
                except RuntimeError as error:
                    missing.append(str(error))
        try:
            if version('hex-device') != '1.4.10':
                missing.append('SDK: native UMI requires hex-device==1.4.10')
        except PackageNotFoundError:
            missing.append('SDK: install hex-device==1.4.10 in this interpreter')
        for name, camera in data['camera_configs'].items():
            try:
                check_camera_sdk(camera)
            except (OSError, ValueError, RuntimeError) as error:
                missing.append(f'{name} camera SDK: {error}')
        for side in ('left', 'right'):
            try:
                BinaryMotorTargets(data['robot']['grippers'][side])
                gp80_encoder_tolerance(data['robot']['grippers'][side])
            except (ValueError, TypeError) as error:
                missing.append(f'robot.grippers.{side}: {error}')
        data['calibrations'] = {}
        for name in data['camera_configs']:
            calibration = data['camera_calibration'].get(name)
            if not calibration:
                missing.append(f'camera_calibration.{name}: missing measured camera transform JSON')
                continue
            try:
                calibration = load_camera_calibration(resolve(calibration))
            except (OSError, ValueError, KeyError, TypeError) as error:
                missing.append(f'camera_calibration.{name}: {error}')
                continue
            data['calibrations'][name] = calibration
        if execute and not research and data.get('commissioned') is not True:
            missing.append('commissioned: hardware configuration is not commissioned; '
                           'record completed hardware validation before execution')
        try:
            from .umi_device import native_configuration
            native_configuration(data['robot'], execute=execute)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            missing.append(f'UMI configuration: {error}')
        if missing:
            raise ValueError('Hardware setup incomplete:\n  - ' + '\n  - '.join(missing) +
                             '\nNo devices opened. Changing commissioned alone does not supply missing configuration or hardware reviews.')
    else:
        data['calibrations'] = {}
    data['config_path'] = str(path)
    return data
