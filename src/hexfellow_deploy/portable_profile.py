"""Trusted wire guard metadata without importing the native controller runtime.

Isaac and the controller may use different Python ABIs. This reader reproduces
only configuration binding and limit intersection, never a control law. The
bound profile must preserve every setting from the pinned source profile.
"""
import ast
import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .contracts import JOINTS
from .source import UPSTREAM_ROOT, verify_source


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_default(name, *, constant=False):
    """Read a pinned dataclass default without importing its compiled imports."""
    tree = ast.parse((verify_source() / 'policy_sim/control/ik_core.py').read_text())
    if constant:
        field = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == name for target in n.targets))
        return ast.literal_eval(field.value)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PinkControllerConfig')
    field = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == name)
    return ast.literal_eval(field.value)


def actuation_metadata(urdf_path, profile_path=None, *, profile='x7-gp80-highspeed-fixed-base-v1',
                       runtime_position_bounds=None, runtime_velocity_bounds=None,
                       runtime_effort_limits_nm=None, base_limits=None):
    """Bind StreamContract limits to the configured URDF and original profile.

    Runtime bounds may narrow the source domain only. ``base_limits`` uses
    base_linear/base_angular/base_acceleration/base_angular_acceleration keys;
    absent values retain the source profile. The caller must use the same
    configured runtime bounds for the controller and this plant-side reader.
    """
    verify_source()
    if Path(profile).name != profile or Path(profile).suffix:
        raise ValueError('Expected a shipped motion profile name')
    shipped = UPSTREAM_ROOT / f'config/control/motion_profiles/{profile}.yaml'
    original = yaml.safe_load(shipped.read_text())
    path = shipped if profile_path is None else Path(profile_path)
    document = yaml.safe_load(path.read_text())
    def tuning(value):
        value = copy.deepcopy(value)
        value.pop('robot_model', None)
        value.pop('provenance', None)
        value['mit'].pop('urdf_sha256', None)
        return value
    if tuning(document) != tuning(original):
        raise ValueError('Bound profile changed original controller tuning or limits')
    urdf = Path(urdf_path).resolve()
    if profile_path is not None and document['mit']['urdf_sha256'] != _sha(urdf):
        raise ValueError('Configured URDF does not match bound profile')
    root = ET.parse(urdf).getroot()
    joints = {joint.attrib['name']: joint for joint in root.findall('joint')}
    bounds = np.array([[float(joints[n].find('limit').attrib[key]) for key in ('lower', 'upper')]
                       for n in JOINTS])
    vendor = document['vendor_hard_limits']
    if vendor.get('sdk_position_lower_rad') is not None:
        bounds[1:, 0] = np.maximum(bounds[1:, 0], np.maximum(vendor['urdf_position_lower_rad'], vendor['sdk_position_lower_rad']))
        bounds[1:, 1] = np.minimum(bounds[1:, 1], np.minimum(vendor['urdf_position_upper_rad'], vendor['sdk_position_upper_rad']))
    for name, value in (runtime_position_bounds or {}).items():
        index = JOINTS.index(name)
        value = np.asarray(value, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all() or value[0] >= value[1]:
            raise ValueError(f'Invalid runtime position bound: {name}')
        bounds[index] = [max(bounds[index, 0], value[0]), min(bounds[index, 1], value[1])]
    margins = np.r_[_source_default('lift_position_margin'), np.repeat(document['arm_limits']['position_margin_rad'], 14)]
    lower, upper = bounds[:, 0]+margins, bounds[:, 1]-margins
    upper[0] -= _source_default('LIFT_REFERENCE_TOP_RESERVE_M', constant=True)
    if not np.isfinite(bounds).all() or np.any(lower >= upper):
        raise ValueError('Configured physical position bounds have no planning interval')
    body, arms, mit = document['body_limits'], document['arm_limits'], document['mit']
    velocity = np.r_[body['lift_velocity_m_s'], arms['velocity_rad_s']]
    for name, value in (runtime_velocity_bounds or {}).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid runtime velocity bound: {name}')
        if name in JOINTS:
            index = JOINTS.index(name)
            velocity[index] = min(velocity[index], value)
        elif name not in ('mobile_base_x', 'mobile_base_y', 'mobile_base_yaw'):
            raise ValueError(f'Unknown runtime velocity coordinate: {name}')
    cap_ref = document['hardware_capabilities']
    cap_path = UPSTREAM_ROOT / cap_ref['path']
    if _sha(cap_path) != cap_ref['sha256']:
        raise ValueError('Hardware capability hash mismatch')
    caps = yaml.safe_load(cap_path.read_text())
    effort = np.asarray(mit['torque_limit_nm'], dtype=float)
    if runtime_effort_limits_nm is not None:
        value = np.asarray(runtime_effort_limits_nm, dtype=float)
        if value.shape != (14,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError('Runtime MIT effort requires fourteen positive finite bounds')
        effort = np.minimum(effort, value)
    base = dict(base_linear=body['base_linear_velocity_m_s'], base_angular=body['base_angular_velocity_rad_s'],
                base_acceleration=body['base_linear_acceleration_m_s2'],
                base_angular_acceleration=body['base_angular_acceleration_rad_s2'])
    for key, value in (base_limits or {}).items():
        if key not in base or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid base limit: {key}')
        base[key] = min(base[key], value)
    limits = dict(dt_s=document['control_dt_s'], kp=mit['kp'], kd=mit['kd'],
                  persistent_lift_reference=document.get('lift_reference_max_error_m') is not None,
                  position_lower=lower.tolist(), position_upper=upper.tolist(), velocity=velocity.tolist(),
                  acceleration=[body['lift_acceleration_m_s2'], *arms['acceleration_rad_s2']],
                  jerk=[body['lift_jerk_m_s3'], *arms['jerk_rad_s3']],
                  measured_velocity=[caps['lift_velocity_m_s'], *(caps['velocity_rad_s']*2)],
                  effort=effort.tolist(), **base)
    return dict(profile=profile, profile_sha256=_sha(path), urdf_sha256=_sha(urdf),
                actuation_limits=limits)
