"""Shared wire representation and stream validation for simulation and hardware."""
from dataclasses import fields
import math
import time
import numpy as np


def encode_mit(command):
    return {field.name: (getattr(command, field.name).tolist()
                        if isinstance(getattr(command, field.name), np.ndarray)
                        else getattr(command, field.name)) for field in fields(command)}


def decode_mit(value):
    from .contracts import MitFrame
    return MitFrame(**value)


def base_reference_next(requested, previous, limits):
    """One original deployment vector/yaw slew step under trusted limits.

    This is the translation-vector and independent yaw interpolation used by
    PoseExecutor.tick. Acceleration and cadence come from the configured guard
    metadata. The caller retains ``previous`` only on an actual application
    receipt and must never call again while that reference remains pending.
    A zero request brakes using the same rule; no state, clock, or gain changes.
    """
    vectors = []
    for name, value in (('requested', requested), ('previous', previous)):
        raw = np.asarray(value, dtype=object)
        if raw.shape != (3,) or any(isinstance(v, (bool, np.bool_)) for v in raw):
            raise ValueError(f'{name} base reference requires three finite numbers')
        vector = np.asarray(value, dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError(f'{name} base reference requires three finite numbers')
        vectors.append(vector.copy())
    keys = ('dt_s', 'base_linear', 'base_angular', 'base_acceleration', 'base_angular_acceleration')
    for key in keys:
        value = limits[key]
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid trusted base reference limit: {key}')
    for vector in vectors:
        if (math.hypot(*vector[:2]) > limits['base_linear']+1e-8
                or abs(vector[2]) > limits['base_angular']+1e-8):
            raise ValueError('Base reference exceeds configured deployment envelope')
    requested, previous = vectors
    change = requested-previous
    linear_step = limits['base_acceleration']*limits['dt_s']
    norm = math.hypot(*change[:2])
    if norm > linear_step:
        change[:2] *= linear_step/norm
    angular_step = limits['base_angular_acceleration']*limits['dt_s']
    change[2] = np.clip(change[2], -angular_step, angular_step)
    return (previous+change).tolist()


class StreamContract:
    """Command liveness is separate from the clock of feedback used by IK."""
    def __init__(self, timeout_s, feedback_timeout_s, *, metadata=None, clock=time.monotonic):
        if not all(math.isfinite(v) and v > 0 for v in (timeout_s, feedback_timeout_s)):
            raise ValueError('Stream timeouts must be finite and positive')
        self.timeout, self.feedback_timeout, self.clock = timeout_s, feedback_timeout_s, clock
        self.latest = None
        self.fault = None
        self.limits = None if metadata is None else metadata['actuation_limits']
        self.previous_position = None
        self.previous_velocity = None
        self.previous_acceleration = None
        self.previous_base = np.zeros(3)

    def seed(self, measured):
        """Bind initial reference history to the controller's exact startup sample."""
        from .contracts import JOINTS, validate_sensor_packet, stationary_reference_velocity
        if self.latest is not None or self.previous_position is not None:
            raise RuntimeError('Actuator reference history is already initialized')
        validate_sensor_packet(measured)
        velocity = np.array([measured['joint_velocities'][n] for n in JOINTS])
        self.previous_velocity = stationary_reference_velocity(velocity)
        self.previous_position = np.array([measured['joint_positions'][n] for n in JOINTS])
        self.previous_acceleration = np.zeros(len(JOINTS))

    def _validate_profile(self, frame, measured, mit, position, base):
        from .contracts import JOINTS
        limits = self.limits
        if limits is None:
            return None, None
        if self.previous_position is None:
            raise RuntimeError('Actuator reference history was not initialized')
        if abs(mit.timestamp_s-frame['created_s']) > 1e-9:
            raise ValueError('MIT timestamp does not match command creation')
        if not np.array_equal(mit.kp, limits['kp']) or not np.array_equal(mit.kd, limits['kd']):
            raise ValueError('MIT gains differ from the configured profile')
        q = np.array([position[n] for n in JOINTS])
        if (np.any(q < np.asarray(limits['position_lower'])-1e-7)
                or np.any(q > np.asarray(limits['position_upper'])+1e-7)):
            raise ValueError('Reference exceeds profile position bounds')
        dt = limits['dt_s']
        reference_velocity = frame['reference_velocity']
        if set(reference_velocity) != set(JOINTS):
            raise ValueError('Complete physical reference velocity is required')
        velocity = np.array([reference_velocity[n] for n in JOINTS], dtype=float)
        if not np.isfinite(velocity).all() or not np.allclose(velocity[1:], mit.velocity, atol=1e-9, rtol=0):
            raise ValueError('MIT and physical reference velocity disagree')
        if np.any(np.abs(velocity) > np.asarray(limits['velocity'])+1e-7):
            raise ValueError('Reference exceeds configured velocity bounds')
        if not np.allclose((q[1:]-self.previous_position[1:])/dt, mit.velocity, atol=1e-6, rtol=0):
            raise ValueError('Discontinuous MIT position reference')
        # This profile integrates lift velocity from the measured prismatic
        # state, while the arm positions retain persistent reference history.
        lift_origin = (self.previous_position[0] if limits['persistent_lift_reference']
                       else frame['measured']['joint_positions'][JOINTS[0]])
        if not limits['persistent_lift_reference']:
            # Pink projects its computational state inside planning margins;
            # the source measurement itself remains unchanged on the wire.
            lift_origin = np.clip(lift_origin, limits['position_lower'][0]+1e-9,
                                  limits['position_upper'][0]-1e-9)
        if not np.isclose((q[0]-lift_origin)/dt, velocity[0], atol=1e-6, rtol=0):
            raise ValueError('Lift position does not match its source and solved velocity')
        acceleration = (velocity-self.previous_velocity)/dt
        jerk = (acceleration-self.previous_acceleration)/dt
        if (np.any(np.abs(acceleration) > np.asarray(limits['acceleration'])+1e-5)
                or np.any(np.abs(jerk) > np.asarray(limits['jerk'])+1e-3)):
            raise ValueError('Reference exceeds profile acceleration or jerk bounds')
        actual_q = np.array([measured['joint_positions'][n] for n in JOINTS])
        actual_v = np.array([measured['joint_velocities'][n] for n in JOINTS])
        if np.any(np.abs(actual_v) > np.asarray(limits['measured_velocity'])+1e-7):
            raise ValueError('Measured joint speed exceeds physical bounds')
        effort = mit.kp*(mit.position-actual_q[1:])+mit.kd*(mit.velocity-actual_v[1:])+mit.torque_ff
        bounds = np.asarray(limits['effort'])
        if np.any(np.abs(effort) > bounds+1e-7) or np.any(np.abs(mit.torque_ff) > bounds+1e-7):
            raise ValueError('Predicted MIT effort exceeds configured bounds')
        if (np.linalg.norm(base[:2]) > limits['base_linear']+1e-8
                or abs(base[2]) > limits['base_angular']+1e-8):
            raise ValueError('Base command exceeds configured deployment envelope')
        change = base-self.previous_base
        if (np.linalg.norm(change[:2]) > limits['base_acceleration']*dt+1e-8
                or abs(change[2]) > limits['base_angular_acceleration']*dt+1e-8):
            raise ValueError('Base reference exceeds configured acceleration bounds')
        return velocity, acceleration

    def accept(self, frame, measured):
        from .contracts import JOINTS, LEFT, RIGHT, LIFT, validate_sensor_packet
        validate_sensor_packet(measured)
        now = self.clock()
        if self.fault:
            raise RuntimeError(self.fault)
        expected = 1 if self.latest is None else self.latest['sequence']+1
        if type(frame['sequence']) is not int or frame['sequence'] != expected:
            raise ValueError('Reference sequence was replayed or skipped')
        if not 0 <= now-frame['created_s'] < self.timeout:
            raise RuntimeError('New reference is stale or has a future timestamp')
        if not 0 <= now-measured['timestamp_s'] < self.feedback_timeout:
            raise RuntimeError('Actual feedback is stale')
        command = frame['command']
        q = command['joint_positions']
        lift_lower, lift_upper = ((0., .4) if self.limits is None else
                                 (self.limits['position_lower'][0], self.limits['position_upper'][0]))
        if (set(q) != set(JOINTS) or not np.isfinite(list(q.values())).all()
                or not lift_lower-1e-7 <= q[LIFT] <= lift_upper+1e-7):
            raise ValueError('Invalid physical joint reference')
        mit = decode_mit(frame['mit'])
        if mit.joint_names != LEFT+RIGHT:
            raise ValueError('MIT frame must contain both arms in configured order')
        if any(abs(q[n]-mit.position[i]) > 1e-8 for i,n in enumerate(mit.joint_names)):
            raise ValueError('MIT and position reference disagree')
        base = np.asarray(command['base_velocity_body'], dtype=float)
        base_linear = .4 if self.limits is None else self.limits['base_linear']
        base_angular = .25 if self.limits is None else self.limits['base_angular']
        if (base.shape != (3,) or not np.isfinite(base).all()
                or np.linalg.norm(base[:2]) > base_linear+1e-8 or abs(base[2]) > base_angular+1e-8):
            raise ValueError('Base command exceeds configured deployment envelope')
        grips = command['gripper_open']
        if set(grips) != {'left','right'} or any(v not in (None,0,1) for v in grips.values()):
            raise ValueError('Invalid binary gripper command')
        if self.limits is not None:
            source = frame['measured']
            validate_sensor_packet(source)
            clocks = frame['received']
            if (set(clocks) != {'left','right','theta','base'}
                    or any(not math.isfinite(v) or not 0 <= now-v < self.feedback_timeout for v in clocks.values())
                    or not 0 <= now-source['timestamp_s'] < self.feedback_timeout):
                raise ValueError('Command source feedback is stale, future or incomplete')
        velocity, acceleration = self._validate_profile(frame, measured, mit, q, base)
        self.latest = frame
        if self.limits is not None:
            self.previous_position = np.array([q[n] for n in JOINTS])
            self.previous_velocity, self.previous_acceleration = velocity, acceleration
            self.previous_base = base.copy()
        return mit

    def check(self, measured):
        now = self.clock()
        if not 0 <= now-measured['timestamp_s'] < self.feedback_timeout:
            self.fault = 'Actual feedback is stale'
        if self.latest and now-self.latest['created_s'] >= self.timeout:
            self.fault = self.fault or 'Control source stopped updating'
        if self.fault:
            raise RuntimeError(self.fault)


def deployment(workspace, path, *, execute=False):
    """The simulator uses the same robot profile and configured device ceilings."""
    from mobile_manipulator.hardware.config import load_deployment
    from mobile_manipulator.hardware.umi_device import native_configuration
    config = load_deployment(workspace/path, execute=execute, mock=not execute, research=execute)
    mapping, profile, effort = native_configuration(config['robot'], execute=execute)
    limits = mapping.runtime_limits(profile)
    from policy_sim.control.hardware_capabilities import load_hardware_capabilities
    from policy_sim.control.motion_profile import controller_config_for_profile
    from mobile_manipulator.robot.joints import JOINTS
    from mobile_manipulator.control.lift_limits import LIFT_MAX_M
    controller = controller_config_for_profile(profile)
    capabilities = load_hardware_capabilities(mapping.data.get('hardware_capabilities'))
    margin = profile.arm_position_margin_rad
    physical_lower = np.maximum(profile.urdf_position_lower_rad, profile.sdk_position_lower_rad)
    physical_upper = np.minimum(profile.urdf_position_upper_rad, profile.sdk_position_upper_rad)
    base = config['robot']['limits']
    metadata = dict(controller=config['controller'], base_motion=config['base_motion'], robot=config['robot'],
                    effort_limits_nm=np.asarray(effort).tolist(),
                    velocity_bounds={n:limits['runtime_velocity_bounds'][n] for n in JOINTS},
                    profile=profile.profile_id, profile_sha256=profile.sha256)
    metadata['actuation_limits'] = dict(
        dt_s=profile.control_dt_s, kp=list(profile.mit_servo.kp), kd=list(profile.mit_servo.kd),
        persistent_lift_reference=profile.lift_reference_max_error_m is not None,
        position_lower=np.r_[controller.lift_position_margin, physical_lower+margin].tolist(),
        position_upper=np.r_[LIFT_MAX_M-controller.lift_position_margin, physical_upper-margin].tolist(),
        velocity=[metadata['velocity_bounds'][n] for n in JOINTS],
        acceleration=np.r_[controller.lift_acceleration_limit, controller.resolved_arm_limits('acceleration')].tolist(),
        jerk=np.r_[controller.lift_jerk_limit, controller.resolved_arm_limits('jerk')].tolist(),
        measured_velocity=np.r_[capabilities.lift_velocity_m_s, capabilities.velocity_rad_s].tolist(),
        effort=np.minimum(effort, profile.mit_servo.torque_limit_nm).tolist(),
        base_linear=min(base['base_linear_m_s'], mapping.base_linear_limit()),
        base_angular=min(base['base_angular_rad_s'], mapping.data['hardware_base_velocity_limits'][2]),
        base_acceleration=base['base_acceleration_m_s2'],
        base_angular_acceleration=base['base_angular_acceleration_rad_s2'])
    return config, mapping, profile, metadata
