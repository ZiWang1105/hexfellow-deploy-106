"""Portable physical encoder and MIT wire schema, with no simulator or IK imports."""
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
import math
import numpy as np

LIFT = 'maver_x4_asm_URDF_joint_1'
LEFT = tuple(f'maver_x4_asm_7_urdf_SLDASM_joint_{i}' for i in range(1, 8))
RIGHT = tuple(f'maver_x4_asm_7_urdf_SLDASM2_joint_{i}' for i in range(1, 8))
JOINTS = (LIFT, *LEFT, *RIGHT)
SIDES = ('left', 'right')
MEASURED_FIELDS = {'timestamp_s', 'joint_positions', 'joint_velocities', 'gripper_state'}
SHORT_JOINTS = ('lift_joint', *(f'{side}_joint_{i}' for side in SIDES for i in range(1, 8)))


def physical_measurement(q, dq, timestamp_s, *, gripper_state):
    """Map actual encoder aliases, requiring an explicit physical gripper state.

    A fixed tool without gripper actuation may supply its fixed closed state;
    movable grippers must use measured state. No coordinates are synthesized.
    """
    def coordinates(values):
        if not isinstance(values, Mapping):
            raise ValueError('Physical encoders must be joint-name mappings')
        result = {}
        for short, native in zip(SHORT_JOINTS, JOINTS):
            found = [values[n] for n in (short, native) if n in values]
            if not found or any(isinstance(v, (bool, np.bool_)) for v in found):
                raise ValueError(f'Missing or invalid physical encoder: {native}')
            if len(found) == 2 and found[0] != found[1]:
                raise ValueError(f'Physical encoder aliases disagree: {native}')
            result[native] = found[0]
        return result
    packet = dict(timestamp_s=timestamp_s, joint_positions=coordinates(q),
                  joint_velocities=coordinates(dq), gripper_state=dict(gripper_state))
    validate_sensor_packet(packet)
    return packet


def validate_sensor_packet(packet):
    if not isinstance(packet, Mapping) or set(packet) != MEASURED_FIELDS:
        raise ValueError('Observation violates the physical proprioception allowlist')
    for field in ('joint_positions', 'joint_velocities'):
        if not isinstance(packet[field], Mapping) or set(packet[field]) != set(JOINTS):
            raise ValueError('Joint feedback must contain exactly 15 physical lift/arm encoders')
    hands = packet['gripper_state']
    if (not isinstance(hands, Mapping) or set(hands) != set(SIDES)
            or any(state not in ('open', 'closed', 'between') for state in hands.values())):
        raise ValueError('Invalid measured gripper state')
    values = [packet['timestamp_s'], *packet['joint_positions'].values(),
              *packet['joint_velocities'].values()]
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           or not math.isfinite(v) for v in values):
        raise ValueError('Invalid finite physical proprioception')


def stationary_reference_velocity(measured_velocity):
    """Exact source acquisition policy for a new stationary 15-encoder stream.

    This seeds new command history only. Actual measured velocity remains in
    every sensor packet and guard; never call to rebase a live reference.
    """
    velocity = np.asarray(measured_velocity, dtype=float)
    if velocity.shape != (15,) or not np.isfinite(velocity).all():
        raise ValueError('Invalid measured velocity for stationary reference initialization')
    if abs(velocity[0]) > .002 or np.any(abs(velocity[1:]) > .01):
        raise ValueError('Stationary reference initialization requires a stationary robot')
    return np.zeros_like(velocity)


def _vector(value, count, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f'MIT {label} requires {count} finite values')
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MitFrame:
    """Wire value only; never runs IK or imports the compiled controller.

    Field semantics match the pinned source MitCommand, permitting Isaac's
    separate Python runtime to validate/apply the same generated MIT values.
    """
    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    torque_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    timestamp_s: float

    def __post_init__(self):
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('MIT joint names must be nonempty and unique')
        object.__setattr__(self, 'joint_names', names)
        for field in ('position', 'velocity', 'torque_ff', 'kp', 'kd'):
            object.__setattr__(self, field, _vector(getattr(self, field), len(names), field))
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError('MIT gains must be nonnegative')
        if (isinstance(self.timestamp_s, (bool, np.bool_)) or not isinstance(self.timestamp_s, Real)
                or not math.isfinite(self.timestamp_s) or self.timestamp_s < 0):
            raise ValueError('MIT timestamp must be finite and nonnegative')

    def select(self, names):
        names = tuple(names)
        indices = [self.joint_names.index(name) for name in names]
        return MitFrame(names, *(getattr(self, field)[indices] for field in
                                 ('position', 'velocity', 'torque_ff', 'kp', 'kd')), self.timestamp_s)

    def motor_input(self):
        with np.errstate(over='raise', invalid='raise'):
            try:
                result = self.kp*self.position + self.kd*self.velocity + self.torque_ff
            except FloatingPointError as error:
                raise ValueError('MIT motor input overflowed') from error
        if not np.isfinite(result).all():
            raise ValueError('MIT motor input must be finite')
        return result
