"""Binary GP80 motor targets and measured telemetry helpers; no device access on import."""
import copy
import threading
import time

import numpy as np


from hexfellow_deploy.gripper_sdk import (
    GP80_ENDPOINT_TOLERANCE_RAD, GP80_MAX_ENCODER_TOLERANCE_RAD,
    gp80_encoder_tolerance, bounded_gp80_position,
)


def find_gp80(api, side, device_id, *, timeout_s=10.):
    """Find a configured wire ID and verify its type, without acquiring control.

    SDK type enums/default-ID tables are not the IDs advertised by the robot.
    Never fall back to a different device when the configured ID is missing.
    """
    from hex_device.generated.public_api_types_pb2 import SecondaryDeviceType
    if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
        raise ValueError(f'{side}: GP80 device_id must be a nonnegative integer')
    deadline = time.monotonic() + timeout_s
    while True:
        hand = api.find_optional_device_by_id(device_id)
        if hand is not None:
            if (getattr(hand, 'motor_count', None) == 1
                    and callable(getattr(hand, 'get_hand_type', None))
                    and hand.get_hand_type() == SecondaryDeviceType.SdtHandGp80G1):
                return hand
            reason = 'is not a single-motor GP80'
            break
        if time.monotonic() >= deadline:
            reason = 'was not reported'
            break
        time.sleep(.01)
    inventory = []
    for device in api.optional_device_list:
        try:
            kind = SecondaryDeviceType.Name(device.device_type)
        except ValueError:
            kind = str(device.device_type)
        inventory.append(f'ID {device.device_id}: {kind}')
    raise RuntimeError(f'{side}: configured GP80 ID {device_id} {reason}. '
                       f'Reported devices: {", ".join(inventory) or "none"}. '
                       f'Set robot.grippers.{side}.device_id to the reported GP80 wire ID; '
                       'the SDK device-type number is not its wire ID.')


class BinaryMotorTargets:
    """Two motor endpoints only; no conversion to jaw opening in meters."""
    def __init__(self, config):
        from ..control.gripper import gripper_state, validate_gripper
        self.classify, self.validate = gripper_state, validate_gripper
        self.closed, self.opened = config.get('closed_motor_rad'), config.get('open_motor_rad')
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x)
                for x in (self.closed, self.opened)) or self.closed == self.opened):
            raise ValueError('Set distinct closed_motor_rad and open_motor_rad motor targets')

    def measured(self, motor):
        return self.classify(motor, self.closed, self.opened)

    def command(self, opened):
        return self.opened if self.validate(opened) else self.closed


class HandTap:
    """Timestamp new GP80 reports rather than polling cached encoder getters."""
    def __init__(self, hand, *, record_firmware_target=False):
        self.lock, self.value, self.error = threading.Lock(), None, None
        self.original = hand._update_optional_data
        def update(kind, message, timestamp):
            result = self.original(kind, message, timestamp)
            if result:
                status = hand.get_simple_motor_status(pop=False)
                with self.lock:
                    stamp = timestamp.to_dict()
                    if self.value is not None:
                        old = self.value['stamp']
                        if (stamp['s'], stamp['ns']) <= (old['s'], old['ns']):
                            self.error = "GP80 timestamp did not advance"
                            return result
                    self.value = dict(received=time.monotonic(), stamp=stamp,
                                      pos=float(status['pos'][0]), vel=float(status['vel'][0]),
                                      effort_nm=float(status['eff'][0]) if status.get('eff') is not None else None,
                                      states=hand.get_motor_states())
                    if record_firmware_target:
                        # Read the incoming protobuf, not _target_command (the
                        # host request) or an independently sampled SDK cache.
                        self.value['firmware_target'] = reported_hand_target(hand, message)
            return result
        hand._update_optional_data = update

    def read(self):
        with self.lock:
            if self.error or self.value is None:
                raise RuntimeError(self.error or "Waiting for GP80 feedback")
            return copy.deepcopy(self.value)


def reported_hand_target(hand, message):
    """Optional bench evidence. An echoed target is not proof of actuation."""
    from google.protobuf.json_format import MessageToDict
    try:
        motors = message.hand_status.motor_status
        if len(motors) != 1:
            return dict(available=False, reason='Expected one reported motor')
        motor = motors[0]
        if not motor.HasField('current_target'):
            return dict(available=False, reason='Firmware omitted current_target')
        target = motor.current_target
        mode = target.WhichOneof('target')
        if mode is None:
            return dict(available=False, reason='Firmware reported empty current_target')
        data = dict(available=True, mode=mode,
                    raw=MessageToDict(target, preserving_proto_field_name=True),
                    measured_position_pulses=int(motor.position),
                    pulse_per_rotation=int(motor.pulse_per_rotation))
        if mode == 'position' and motor.pulse_per_rotation > 0:
            data['position_rad'] = float(hand.convert_positions_to_rad(
                np.array([target.position]), np.array([motor.pulse_per_rotation]))[0])
        return data
    except (AttributeError, ValueError, TypeError, IndexError) as error:
        # Diagnostics must not replace measured feedback or manufacture a target.
        return dict(available=False, reason=f'{type(error).__name__}: {error}')
