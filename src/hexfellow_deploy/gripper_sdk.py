"""GP80 packet adapter and shared SDK feedback publisher.

The packet encoder serializes our controller's exact reference, bypassing Hands' second
position accumulator. The publisher runs on the existing SDK thread and retains command leases.
"""
import math
import numpy as np

from .gripper_effort import EffortGripperController, GripperFeedback, closing_effort_settings
from .gripper_contract import validate_gripper

# Encoder readings can differ slightly from the SDK's command endpoint. This
# tolerance admits readings only; every generated motor target stays in bounds.
GP80_ENDPOINT_TOLERANCE_RAD = .02
GP80_MAX_ENCODER_TOLERANCE_RAD = .1


def gp80_encoder_tolerance(config):
    """A per-device feedback allowance; it never expands command limits."""
    value = config.get('encoder_boundary_tolerance_rad', GP80_ENDPOINT_TOLERANCE_RAD)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0 < value <= GP80_MAX_ENCODER_TOLERANCE_RAD):
        raise ValueError('GP80 encoder_boundary_tolerance_rad must be finite and within (0, 0.1]')
    return float(value)


def bounded_gp80_position(position, limits, *, context='GP80', tolerance_rad=GP80_ENDPOINT_TOLERANCE_RAD):
    """Convert a near-boundary encoder reading to a legal command reference."""
    lo, hi = limits[:2]
    if (isinstance(position, bool) or not isinstance(position, (int, float))
            or not np.isfinite([position, lo, hi]).all() or not lo < hi):
        raise ValueError(f'{context}: invalid encoder position or SDK bounds')
    tolerance_rad = gp80_encoder_tolerance({'encoder_boundary_tolerance_rad': tolerance_rad})
    if not lo - tolerance_rad <= position <= hi + tolerance_rad:
        raise ValueError(f'{context}: measured motor angle {position:.9f} rad is outside '
                         f'SDK command bounds [{lo:.9f}, {hi:.9f}] rad by more than '
                         f'{tolerance_rad} rad; no target saved or sent')
    return float(np.clip(position, lo, hi))


class GP80EffortPacketBuilder:
    def __init__(self, hand, config):
        if hand.motor_count != 1:
            raise ValueError('Effort controller requires a single-motor gripper')
        low, high = hand.get_joint_limits()[:2]
        if not all(low <= v <= high for v in (config.closed_rad, config.open_rad)):
            raise ValueError('Controller endpoints exceed SDK command bounds')
        self.hand = hand
        self.controller = EffortGripperController(config)
        self.last_output = None

    def build(self, request, row, now_s, *, command_expires_s):
        from hex_device.motor_base import MotorCommand
        from hex_device.generated.public_api_down_pb2 import APIDown
        # Missing effort is invalid; never silently substitute zero torque.
        feedback = GripperFeedback(row.get('pos'), row.get('vel'), row.get('effort_nm'),
                                   row.get('received'), bool(row.get('states')) and
                                   all(v == 'normal' for v in row['states']))
        if (isinstance(command_expires_s, bool) or not isinstance(command_expires_s, (int, float))
                or not math.isfinite(command_expires_s) or not now_s < command_expires_s <= now_s + .1):
            output = self.controller.stop('Gripper command lease expired or invalid')
        else:
            output = self.controller.step(request, feedback, now_s)
        self.last_output = output
        message = APIDown()
        secondary = message.secondary_device_command
        secondary.device_id = self.hand.device_id
        if output.brake:
            secondary.hand_command.motor_targets.targets.add(brake=True)
        else:
            command = MotorCommand.create_position_command([output.position_rad])
            try:
                pulses = self.hand.get_motor_pulse_per_rotations()
                if pulses is None or np.shape(pulses) != (1,) or not np.isfinite(pulses).all() or pulses[0] <= 0:
                    raise ValueError('GP80 pulse conversion unavailable')
                targets = self.hand._construct_target_motor_msg(pulses, command)
                secondary.hand_command.motor_targets.CopyFrom(targets)
            except (ValueError, TypeError, RuntimeError, OverflowError) as error:
                self.last_output = self.controller.stop(f'GP80 packet encoding failed: {error}')
                secondary.hand_command.motor_targets.Clear()
                secondary.hand_command.motor_targets.targets.add(brake=True)
        return message


def require_closure_enabled(config, request, *, side):
    """Enforce the incident lockout before any motion in the command group.

    This operator setting is a lockout, not proof of force qualification. The
    setting is not a force rating; explicit per-device closure disable is honored.
    """
    request = validate_gripper(request, allow_hold=True)
    if request == 0 and config.get('closure_enabled', False) is not True:
        raise RuntimeError(f'{side}: GP80 closure disabled pending repaired fingers and '
                           'operator enablement in robot.grippers configuration')


class GP80FeedbackPublisher:
    """One controller per hand, evaluated by that hand's 250-Hz SDK publisher.

    The 50-Hz owner renews a finite command lease; duplicate close requests do
    not restart approach. Opening uses the original SDK interpolation. No bench
    duration, automatic release, or contact-truth assertion belongs here.
    """
    def __init__(self, hand, tap, config, *, clock=None, gate=None):
        from collections import deque
        from dataclasses import replace
        import threading
        import time
        self.hand, self.tap, self.config = hand, tap, dict(config)
        self.settings = replace(closing_effort_settings(config['closed_motor_rad'],
            config['open_motor_rad'], effort_trip_nm=config.get('effort_trip_nm')),
            feedback_tolerance_rad=gp80_encoder_tolerance(config))
        # Validate before installing a live SDK hook, not during group dispatch.
        GP80EffortPacketBuilder(hand, self.settings)
        self.opening_torque = config.get('opening_max_torque_nm', config['max_torque_nm'])
        if (isinstance(self.opening_torque, bool) or not isinstance(self.opening_torque, (int, float))
                or not math.isfinite(self.opening_torque) or not 0 < self.opening_torque <= 3):
            raise ValueError('GP80 opening torque threshold must be within (0, 3] Nm')
        self.clock = clock or time.monotonic
        self.gate = gate
        self.lock = threading.RLock()
        self.requested = None
        self.expires = 0.
        self.builder = None
        self.fault = None
        self.closed = False
        self.original = hand._construct_joint_command_msg
        self.trace = deque(maxlen=4096)
        self.allowed = ()
        self.sequence = 0
        hand._construct_joint_command_msg = self.packet
        if gate is not None:
            gate.hand_publisher = self

    def _brake(self):
        from hex_device.generated.public_api_down_pb2 import APIDown
        packet = APIDown()
        packet.secondary_device_command.device_id = self.hand.device_id
        packet.secondary_device_command.hand_command.motor_targets.targets.add(brake=True)
        return packet

    def check(self):
        if self.fault:
            raise RuntimeError(self.fault)

    def request(self, opened, *, expires, opening_target=None):
        from dataclasses import replace
        from hex_device.motor_base import CommandType, MotorCommand
        opened = validate_gripper(opened, allow_hold=True)
        require_closure_enabled(self.config, opened, side='GP80')
        with self.lock:
            self.check()
            if self.closed:
                raise RuntimeError('GP80 publisher is closed')
            now = self.clock()
            if (isinstance(expires, bool) or not isinstance(expires, (int, float))
                    or not math.isfinite(expires) or not now < expires <= now+.1+1e-9):
                raise RuntimeError('Invalid GP80 command lease')
            if opened == 1:
                if (isinstance(opening_target, bool) or not isinstance(opening_target, (int, float))
                        or not math.isfinite(opening_target)):
                    raise ValueError('Opening reference unavailable')
                lo, hi = self.hand.get_joint_limits()[:2]
                if not lo <= opening_target <= hi:
                    raise ValueError('Opening reference outside SDK command bounds')
            if opened != self.requested:
                self.allowed = ()
                row = self.tap.read()
                seed = bounded_gp80_position(row['pos'], self.hand.get_joint_limits(),
                                             tolerance_rad=gp80_encoder_tolerance(self.config))
                if opened == 0:
                    # SDK endpoint discrepancy is admitted without editing raw feedback.
                    opened_bound = (min(self.settings.open_rad, seed)
                        if self.settings.closed_rad > self.settings.open_rad
                        else max(self.settings.open_rad, seed))
                    settings = replace(self.settings, open_rad=opened_bound)
                    self.builder = GP80EffortPacketBuilder(self.hand, settings)
                elif opened == 1:
                    self.builder = None
                    self.hand._last_command_send = MotorCommand.create_position_command([seed])
                    self.hand.set_pos_torque(self.opening_torque)
                else:
                    self.builder = None
            self.requested, self.expires = opened, expires
            # Refresh the SDK command watchdog. Closing references are generated
            # exclusively by packet(), never by the SDK's position accumulator.
            if opened == 1:
                self.hand.motor_command(CommandType.POSITION, [opening_target])
            else:
                self.hand.motor_command(CommandType.BRAKE, [True])

    def packet_allowed(self, packet, now):
        # Called under the wire gate's lock: no publisher lock (lock-order cycle).
        payload = packet.secondary_device_command.SerializeToString()
        return (not self.closed and not self.fault and self.requested is not None and
                now < self.expires and any(p == payload and now < expiry for p, expiry in self.allowed))

    def packet(self):
        from dataclasses import asdict
        with self.lock:
            now = self.clock()
            if self.closed or self.fault or self.requested is None:
                return self._brake()
            row = None
            try:
                if now >= self.expires:
                    raise RuntimeError('GP80 command lease expired')
                row = self.tap.read()
                if not 0 <= now-row['received'] <= .05:
                    raise RuntimeError('GP80 feedback expired')
                if not row['states'] or any(s != 'normal' for s in row['states']):
                    raise RuntimeError('GP80 motor health unavailable/abnormal')
                if not math.isfinite(row['vel']) or abs(row['vel']) > 6.:
                    raise RuntimeError('GP80 measured speed outside bounds')
                if not isinstance(row.get('effort_nm'), (int, float)) or not math.isfinite(row['effort_nm']):
                    raise RuntimeError('GP80 measured motor effort unavailable')
                bounded_gp80_position(row['pos'], self.hand.get_joint_limits(),
                    tolerance_rad=gp80_encoder_tolerance(self.config))
                if self.requested == 0:
                    packet = self.builder.build(0, row, now, command_expires_s=self.expires)
                    output = asdict(self.builder.last_output)
                    if output['fault']:
                        raise RuntimeError(output['fault'])
                else:
                    packet = self.original()
                    output = dict(state='opening', position_rad=float(self.hand._last_command_send.position_command[0]))
                self.allowed = tuple((p,e) for p,e in self.allowed if now < e)[-31:] + (
                    (packet.secondary_device_command.SerializeToString(), min(self.expires, row['received']+.05)),)
                if self.gate is not None:
                    pulses = [t.position for t in packet.secondary_device_command.hand_command.motor_targets.targets
                              if t.WhichOneof('target') == 'position']
                    if pulses:
                        self.gate.authorize_hand(self.hand.device_id, pulses, self.expires)
                self.sequence += 1
                self.trace.append(dict(sequence=self.sequence, time_s=now, measured=dict(row),
                                       request=self.requested, output=output))
                return packet
            except Exception as error:
                self.fault = f'{type(error).__name__}: {error}'
                self.allowed = ()
                self.trace.append(dict(time_s=now, measured=row, fault=self.fault))
                return self._brake()

    def stop(self):
        from hex_device.motor_base import CommandType
        with self.lock:
            self.requested, self.expires, self.allowed = None, 0., ()
            self.hand.motor_command(CommandType.BRAKE, [True])

    def close(self):
        with self.lock:
            self.stop()
            self.closed = True
            self.hand._construct_joint_command_msg = self.original

    def report(self):
        from dataclasses import asdict
        if not self.lock.acquire(blocking=False):
            return dict(controller='position_effort_two_phase', fault=self.fault,
                        error='Publisher lock busy at diagnostic capture')
        try:
            return dict(controller='position_effort_two_phase', request=self.requested,
                        fault=self.fault, publication_sequence=self.sequence,
                        latest=asdict(self.builder.last_output) if self.builder and self.builder.last_output else None,
                        trace=list(self.trace))
        finally:
            self.lock.release()
