"""Fixed numeric DA control messages; policy requests and diagnostics use JSON.

All fields use float64 to preserve original monotonic timestamps and encoder
precision. Version and exact integer checks reject incompatible layouts.
"""
import time
import numpy as np
from deploy_anything.components.comms.wrappers import ArrayCommWrapper
from mobile_manipulator.robot.joints import JOINTS, LEFT, RIGHT
from mobile_manipulator.interfaces.contract import validate_sensor_packet

GROUPS = ('left', 'right', 'theta', 'base')
SIDES = ('left', 'right')
STATES = ('closed', 'open', 'between')
MIT_FIELDS = ('position', 'velocity', 'torque_ff', 'kp', 'kd')


def integer(value):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value != int(value) or abs(value) > 2**53:
        raise ValueError('Invalid numeric message integer')
    return int(value)


def joints(value):
    if set(value) != set(JOINTS):
        raise ValueError('Numeric message requires exactly the physical joint channels')
    return [value[n] for n in JOINTS]


def encode_sensor(value):
    validate_sensor_packet(value)
    return [value['timestamp_s'], *joints(value['joint_positions']), *joints(value['joint_velocities']),
            *[STATES.index(value['gripper_state'][s]) for s in SIDES]]


def decode_sensor(values):
    codes = [integer(x) for x in values[31:33]]
    if any(x not in range(3) for x in codes):
        raise ValueError('Invalid measured gripper state')
    return dict(timestamp_s=float(values[0]), joint_positions=dict(zip(JOINTS, values[1:16].tolist())),
                joint_velocities=dict(zip(JOINTS, values[16:31].tolist())),
                gripper_state=dict(zip(SIDES, [STATES[x] for x in codes])))


def received(value):
    if set(value) != set(GROUPS):
        raise ValueError('Incomplete device receipt timestamps')
    return [value[n] for n in GROUPS]


class NumericTopic(ArrayCommWrapper):
    LENGTH = 0

    def __init__(self, name, prefix='', *, create=True):
        super().__init__(prefix+name, shape=(self.LENGTH,), dtype='float64', create=create)

    def put(self, value, *, timeout=.001):
        array = np.asarray([1., *self.encode(value)], dtype=np.float64)
        if array.shape != self.shape or not np.isfinite(array).all():
            raise ValueError('Invalid numeric control message shape or values')
        return self.send(self.ArrayData(time.monotonic(), array), timeout=timeout)

    def get(self, *, timeout=.001):
        row = self.receive(timeout=timeout)
        if row.timestamp <= 0:
            return None
        if row.array[0] != 1 or not np.isfinite(row.array).all():
            raise ValueError('Invalid numeric control message version or values')
        return self.decode(row.array[1:])


class FeedbackTopic(NumericTopic):
    LENGTH = 40

    def encode(self, value):
        return [*encode_sensor(value['measured']), *received(value['received']),
                integer(value.get('publication_sequence', -1)), value.get('published_s', 0.)]

    def decode(self, a):
        value = dict(measured=decode_sensor(a[:33]), received=dict(zip(GROUPS, a[33:37].tolist())))
        sequence = integer(a[37])
        if sequence >= 0:
            value.update(publication_sequence=sequence, published_s=float(a[38]))
        return value


class MeasuredTopic(NumericTopic):
    LENGTH = 34

    def encode(self, value):
        return encode_sensor(value)

    def decode(self, a):
        return decode_sensor(a)


class ReferenceTopic(NumericTopic):
    LENGTH = 146

    def encode(self, value):
        cmd, mit = value['command'], value['mit']
        if tuple(mit['joint_names']) != LEFT+RIGHT:
            raise ValueError('MIT joint order differs from numeric schema')
        if set(cmd['gripper_open']) != set(SIDES):
            raise ValueError('Incomplete binary gripper reference')
        hands = [cmd['gripper_open'][s] for s in SIDES]
        if any(x is not None and (type(x) is not int or x not in (0, 1)) for x in hands):
            raise ValueError('Gripper reference must be binary or inactive')
        if np.shape(cmd['base_velocity_body']) != (3,) or any(np.shape(mit[k]) != (14,) for k in MIT_FIELDS):
            raise ValueError('Invalid base or MIT vector shape')
        return [integer(value['sequence']), value['created_s'], *encode_sensor(value['measured']),
                *received(value['received']), *joints(cmd['joint_positions']), *cmd['base_velocity_body'],
                *[-1 if x is None else x for x in hands], *joints(value['reference_velocity']),
                mit['timestamp_s'], *[x for k in MIT_FIELDS for x in mit[k]]]

    def decode(self, a):
        codes = [integer(x) for x in a[57:59]]
        if any(x not in (-1, 0, 1) for x in codes):
            raise ValueError('Invalid binary gripper reference')
        return dict(sequence=integer(a[0]), created_s=float(a[1]), measured=decode_sensor(a[2:35]),
            received=dict(zip(GROUPS, a[35:39].tolist())),
            command=dict(joint_positions=dict(zip(JOINTS, a[39:54].tolist())),
                         base_velocity_body=a[54:57].tolist(),
                         gripper_open=dict(zip(SIDES, [None if x == -1 else x for x in codes]))),
            reference_velocity=dict(zip(JOINTS, a[59:74].tolist())),
            mit=dict(joint_names=list(LEFT+RIGHT), timestamp_s=float(a[74]),
                     **{k:a[75+i*14:89+i*14].tolist() for i,k in enumerate(MIT_FIELDS)}))


class ReceiptTopic(NumericTopic):
    LENGTH = 7
    SCOPES = ('simulator_applied', 'sdk_accepted', 'udp_sent')

    def encode(self, value):
        receipt = value.get('receipt')
        return [integer(receipt['sequence']) if receipt else -1,
                receipt['accepted_at'] if receipt else 0., self.SCOPES.index(receipt['scope']) if receipt else -1,
                integer(value['pending']) if value.get('pending') is not None else -1,
                integer(value.get('latest_sequence', -1)), int(bool(value.get('fault')))]

    def decode(self, a):
        seq, scope, pending, latest, fault = (integer(a[i]) for i in (0, 2, 3, 4, 5))
        if fault not in (0, 1) or (seq >= 0 and scope not in range(len(self.SCOPES))):
            raise ValueError('Invalid actuator receipt')
        return dict(receipt=None if seq < 0 else dict(sequence=seq, accepted_at=float(a[1]), scope=self.SCOPES[scope]),
                    pending=None if pending < 0 else pending, latest_sequence=latest, fault=bool(fault))
