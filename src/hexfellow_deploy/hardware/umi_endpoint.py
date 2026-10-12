"""Native UMI endpoint with explicitly separate body-pulse and GP80 channels.

Arm/lift acquisition, MIT packets, checked KCP, health and stop confirmation are
upstream. Chassis poses stay inside this publisher and never reach the policy/IK.
"""
import copy
import time

import numpy as np

from ..robot.upstream import load_umi
from ..robot.joints import SIDES
from .device import BinaryMotorTargets, HandTap, find_gp80, bounded_gp80_position, gp80_encoder_tolerance
from ..control.lift_limits import LIFT_MAX_M, validate_lift_target
from ..control.gripper import BinaryGripperServo, validate_gripper
from hexfellow_deploy.gripper_sdk import require_closure_enabled, GP80FeedbackPublisher

load_umi()
from policy_sim.hardware.hexfellow_endpoint import HexWholeBodyEndpoint
from policy_sim.hardware.mit_wire import MitWireGate
from policy_sim.hardware.x7 import HardwareSafetyError


def validate_startup_lift(state, mapping):
    """Check the existing native POSITION window before acquiring any group.

    The encoder-only fixed-base core does not implement upstream's planar
    lift-retreat preparation. Never send an invalid endpoint HOLD or silently
    move the lift as an initialization workaround.
    """
    low, high = state.health['theta']['range']
    model_height = float(state.q[3])
    if model_height > LIFT_MAX_M:
        raise HardwareSafetyError(
            f'MIT startup lift {model_height:.6f} m exceeds the {LIFT_MAX_M:.3f} m model ceiling. '
            'Use keyboard F to lower it below 0.400 m before restarting. No startup motion was sent.')
    sdk_height = mapping.lift_command(model_height, (low, high))
    if not low + .002 <= sdk_height <= high - .002:
        raise HardwareSafetyError(
            f'MIT startup lift outside native operating range: model {model_height:.9f} m, '
            f'SDK {sdk_height:.9f} m, permitted SDK [{low + .002:.9f}, {high - .002:.9f}] m. '
            'Use keyboard R/F or K to put the lift inside this range before restarting. '
            'No startup motion was sent.')


class GP80WireGate(MitWireGate):
    """Leave UMI MIT filtering intact; separately lease optional GP80 positions."""
    def __init__(self, name, clock=time.monotonic):
        super().__init__(name, clock)
        self.hand_id = None
        self.hand_range = None
        self.hand_deadline = 0.
        self.hand_rejection = None

    def authorize_hand(self, device_id, positions, expires):
        with self.lock:
            if self.phase != 'active' or self.fault or self.clock() >= expires:
                raise HardwareSafetyError('GP80 authorization outside active lease')
            values = np.asarray(positions, dtype=float)
            if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                raise HardwareSafetyError('Invalid GP80 encoder pulse authorization')
            # SDK MotorBase serializes POSITION with int(target), truncating
            # toward zero. Compare in that same wire domain, not fractional
            # pulses that would reject valid endpoint packets after conversion.
            pulses = [int(value) for value in values]
            self.hand_id = device_id
            self.hand_range = (min(pulses), max(pulses))
            self.hand_deadline = expires

    def filter(self, data):
        with self.lock:
            before = self.fault
            result = self._filter(data)
            if self.fault and before is None and self.hand_rejection is None:
                self.hand_rejection = dict(
                    packet_kind=data.WhichOneof('down'), packet_text=str(data)[:2048],
                    observed_s=self.clock(), device_id=self.hand_id,
                    allowed_pulse_range=self.hand_range, hand_deadline=self.hand_deadline,
                    arm_deadline=self.deadline, reason=self.fault)
            return result

    def _filter(self, data):
        if data.WhichOneof('down') != 'secondary_device_command':
            return super().filter(data)
        with self.lock:
            # SDK optional-device loops may start before control acquisition.
            if self.phase in ('readonly', 'initializing') or self.hand_id is None:
                return None
            secondary = data.secondary_device_command
            if secondary.device_id != self.hand_id:
                return self.reject()
            if self.phase != 'active' or self.fault or self.clock() >= min(self.deadline, self.hand_deadline):
                result = copy.deepcopy(data)
                result.secondary_device_command.Clear()
                result.secondary_device_command.device_id = self.hand_id
                result.secondary_device_command.hand_command.motor_targets.targets.add(brake=True)
                return result
            targets = secondary.hand_command.motor_targets.targets
            if len(targets) == 1 and targets[0].WhichOneof('target') == 'brake' and targets[0].brake:
                return data
            publisher = getattr(self, 'hand_publisher', None)
            if publisher is not None and len(targets) == 1 and targets[0].WhichOneof('target') == 'position':
                if publisher.packet_allowed(data, self.clock()):
                    return data
                return publisher._brake()
            if (len(targets) != 1 or targets[0].WhichOneof('target') != 'position'
                    or not self.hand_range[0] <= targets[0].position <= self.hand_range[1]):
                result = self.reject()
                self.fault = 'GP80 wire target outside authorized integer pulse range or unsupported mode'
                return result
            return data


class MobileEndpoint(HexWholeBodyEndpoint):
    def __init__(self, mapping, config, *, execute, downlink_gate_factory=None):
        self.mobile_config = config
        self.hands, self.hand_taps = {}, {}
        self.grippers = {s: BinaryMotorTargets(config['grippers'][s]) for s in SIDES}
        self.extras = None
        self.last_base = np.zeros(3)
        self.gripper_servos = {}
        self.gripper_publishers = {}
        super().__init__(mapping, read_only=not execute,
                         downlink_gate_factory=(downlink_gate_factory or GP80WireGate) if execute else None)
        try:
            for spec, api in zip(self.specs, self.apis):
                if spec.name not in SIDES:
                    continue
                side = spec.name
                try:
                    hand = find_gp80(api, side, config['grippers'][side]['device_id'],
                                     timeout_s=self.timeout)
                except (RuntimeError, ValueError) as error:
                    raise HardwareSafetyError(str(error)) from error
                lo, hi = hand.get_joint_limits()[:2]
                if not all(lo <= x <= hi for x in (self.grippers[side].closed, self.grippers[side].opened)):
                    raise HardwareSafetyError(f'{side}: binary GP80 endpoints exceed device limits')
                self.hands[side] = hand
                self.hand_taps[side] = HandTap(hand, record_firmware_target=True)
                if execute:
                    # Cache only; readonly/initializing gate sends no hand commands.
                    hand.command_timeout_check(True)
                    hand.motor_command(self.command_type.BRAKE, [True])
                    hand.set_pos_torque(config['grippers'][side]['max_torque_nm'])
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    self.hand_feedback()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.005)
        except BaseException:
            self.close()
            raise

    def hand_feedback(self):
        rows = {s: self.hand_taps[s].read() for s in SIDES}
        for s, row in rows.items():
            if time.monotonic() - row['received'] > self.mobile_config['feedback_max_age_s']:
                raise HardwareSafetyError(f'{s}: GP80 feedback expired')
            if not row['states'] or any(v != 'normal' for v in row['states']):
                raise HardwareSafetyError(f'{s}: GP80 motor health unavailable/abnormal')
            try:
                bounded_gp80_position(row['pos'], self.hands[s].get_joint_limits(), context=f'{s} feedback',
                    tolerance_rad=gp80_encoder_tolerance(self.mobile_config['grippers'][s]))
            except ValueError as error:
                raise HardwareSafetyError(str(error)) from error
        return rows

    def acquire(self, stage, profile):
        self.hand_feedback()
        self.validate_lift_start(self.state())
        state = super().acquire(stage, profile)
        self.gripper_servos = {s: BinaryGripperServo(self.grippers[s].closed, self.grippers[s].opened,
            bounded_gp80_position(r['pos'], self.hands[s].get_joint_limits(), context=f'{s} startup',
                tolerance_rad=gp80_encoder_tolerance(self.mobile_config['grippers'][s])),
            self.mobile_config['gripper_travel_time_s']) for s, r in self.hand_feedback().items()}
        for side in SIDES:
            previous = self.gripper_publishers.get(side)
            if previous is not None:
                previous.close()
            self.gripper_publishers[side] = GP80FeedbackPublisher(
                self.hands[side], self.hand_taps[side], self.mobile_config['grippers'][side],
                gate=self.direct_gates[side])
        return state

    def validate_lift_start(self, state):
        validate_startup_lift(state, self.mapping)

    def validate_lift_command(self, command):
        validate_lift_target(float(command.positions[3]), 'Native lift command')

    def validate_extras(self, command):
        self.validate_lift_command(command)
        if self.extras is None or self.extras['sequence'] != command.sequence:
            raise HardwareSafetyError('Missing matching mobile/gripper command')
        # Validate both hands before constructing any arm/base dispatch callback.
        for side in SIDES:
            self.gripper_publishers[side].check()
            require_closure_enabled(self.mobile_config['grippers'][side],
                                    self.extras['grippers'][side], side=side)
        base = np.asarray(self.extras['base'], dtype=float)
        limits = self.mobile_config['limits']
        if (base.shape != (3,) or not np.isfinite(base).all()
                or np.linalg.norm(base[:2]) > limits['base_linear_m_s'] + 1e-8
                or abs(base[2]) > limits['base_angular_rad_s']):
            raise HardwareSafetyError('Invalid mobile velocity')
        sdk_body = np.r_[self.mapping.rotation().T @ base[:2], base[2]]
        reviewed = np.asarray(self.mapping.data['hardware_base_velocity_limits'], dtype=float)
        if (np.any(np.abs(sdk_body) > reviewed + 1e-8)
                or np.linalg.norm(base[:2]) > self.mapping.base_linear_limit() + 1e-8):
            raise HardwareSafetyError('Mobile command exceeds reviewed hardware speed')
        if np.linalg.norm(base[:2] - self.last_base[:2]) > limits['base_acceleration_m_s2'] * .02 + 1e-8:
            raise HardwareSafetyError('Mobile acceleration exceeded')
        if abs(base[2] - self.last_base[2]) > limits['base_angular_acceleration_rad_s2'] * .02 + 1e-8:
            raise HardwareSafetyError('Mobile angular acceleration exceeded')
        rows = self.hand_feedback()
        targets = {}
        for side in SIDES:
            requested = validate_gripper(self.extras['grippers'][side], allow_hold=True)
            servo = self.gripper_servos[side]
            # Preview the device reference without changing it until validation succeeds.
            planned = copy.copy(servo)
            target = (bounded_gp80_position(rows[side]['pos'], self.hands[side].get_joint_limits(),
                          tolerance_rad=gp80_encoder_tolerance(self.mobile_config['grippers'][side]))
                      if requested == 0 else planned.step(requested, .02))
            lo, hi = self.hands[side].get_joint_limits()[:2]
            if not lo <= target <= hi:
                raise HardwareSafetyError('GP80 target exceeds device limits')
            targets[side] = target
        return base, targets, rows

    def _prepare_dispatch(self, command, stage, state):
        base, hands, rows = self.validate_extras(command)
        limits = self.mobile_config['limits']
        if (np.linalg.norm(state.v[:2]) > limits['base_linear_m_s'] + 1e-5
                or abs(state.v[2]) > limits['base_angular_rad_s'] + 1e-5):
            raise HardwareSafetyError('Measured chassis speed exceeds deployment bounds')
        # Fixed-base UMI guard/solver receives zero base references. The separate
        # bounded body pulse is admitted here, with the SAME source deadline.
        dispatch = super()._prepare_dispatch(command, 'arms-lift', state)
        from hex_device.generated.public_api_down_pb2 import APIDown
        from policy_sim.hardware.control_timing import feedback_deadline
        from policy_sim.control.feedback_policy import received_deadline
        expected = APIDown()
        xy = self.mapping.rotation().T @ base[:2]
        speed = expected.base_command.simple_move_command.xyz_speed
        speed.speed_x, speed.speed_y, speed.speed_z = float(xy[0]), float(xy[1]), float(base[2])
        deadline = min(command.expires, feedback_deadline(state),
                       received_deadline(command.state_received, self.mapping.feedback_policy))
        hand_pulses = {}
        for side, target in hands.items():
            hand = self.hands[side]
            # Include the last SDK reference and measured position so its torque
            # limited interpolation is admitted, never arbitrary hand commands.
            previous = getattr(hand, '_last_command_send', None)
            prior = rows[side]['pos'] if previous is None else previous.position_command[0]
            pulses = hand.convert_rad_to_positions(np.array([prior, rows[side]['pos'], target]),
                                                   hand.get_motor_pulse_per_rotations())
            if pulses.ndim != 1 or not pulses.size or not np.isfinite(pulses).all():
                raise HardwareSafetyError('Invalid GP80 encoder pulse authorization')
            hand_pulses[side] = pulses

        def write_mobile():
            self.devices['base'].set_vehicle_speed(float(xy[0]), float(xy[1]), float(base[2]))
            for side, target in hands.items():
                hand = self.hands[side]
                self.direct_gates[side].authorize_hand(self.mobile_config['grippers'][side]['device_id'],
                                                       hand_pulses[side], deadline)
                self.gripper_publishers[side].request(self.extras['grippers'][side],
                    expires=deadline, opening_target=target)
                wake = getattr(hand, '_policy_wake', None)
                if wake:
                    wake()
            self.last_base = base.copy()
            for side, target in hands.items():
                self.gripper_servos[side].reference = target

        # Base authorization joins both arms and lift in the upstream batch;
        # no SDK setter runs while only part of the new group is authorized.
        dispatch['base'] = (expected, write_mobile)
        return dispatch

    def stop(self):
        owned = bool(self.owned)
        requested = time.monotonic()
        failures = []
        for publisher in getattr(self, 'gripper_publishers', {}).values():
            try:
                publisher.stop()
            except Exception as error:
                failures.append(f'GP80 publisher stop: {error}')
        if owned:
            for side, hand in self.hands.items():
                try:
                    hand.motor_command(self.command_type.BRAKE, [True])
                except Exception as error:
                    failures.append(f'{side} GP80 brake: {error}')
        # Native stop records its request BEFORE transitioning owned wire gates.
        # Pre-transitioning here creates an older immutable stop_token, causing
        # CheckedUDP to reject even the corresponding acknowledged brake.
        # It also transitions the arm gates used by GP80; do not duplicate it.
        # Auxiliary failures must never prevent the native arm/lift/base stop.
        failures.extend(super().stop())
        if owned and self.hand_taps:
            stable, previous = {}, {}
            deadline = time.monotonic() + 1.
            while time.monotonic() < deadline:
                try:
                    rows = self.hand_feedback()
                    for side, row in rows.items():
                        stamp = row['received']
                        if stamp <= max(requested, previous.get(side, requested)):
                            continue
                        if abs(row['vel']) > .01 or stamp - previous.get(side, stamp) > .1:
                            stable.pop(side, None)
                        else:
                            stable.setdefault(side, stamp)
                        previous[side] = stamp
                    if all(s in stable and previous[s] - stable[s] >= .1 for s in SIDES):
                        break
                except RuntimeError:
                    stable.clear()
                time.sleep(.005)
            else:
                failures.append('GP80 stop not confirmed by fresh stationary feedback; use physical E-stop')
        return failures
