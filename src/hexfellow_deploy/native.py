"""Native leaf adapter: calibrated SDK IO without the former gateway runtime."""
from contextlib import ExitStack
import math
import time
import numpy as np

from mobile_manipulator.hardware.umi_endpoint import MobileEndpoint, GP80WireGate
from mobile_manipulator.hardware.umi_device import sensor_packet, native_packet
from mobile_manipulator.robot.joints import JOINTS
from policy_sim.hardware.mit_wire import authorize_group
from policy_sim.hardware.hexfellow_endpoint import HexWholeBodyEndpoint
from policy_sim.hardware.x7 import HardwareSafetyError
from .reference import Receipt


class ContinuousGate(GP80WireGate):
    """Finite command-stream watchdog independent of per-device feedback age.

    Packet matching, checked KCP and stop filtering remain upstream. Only the
    previous fixed admission/source lease is replaced by the shared DA timeout.
    """
    def __init__(self, name, clock=time.monotonic, timeout_s=.1):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Command timeout must be finite and positive")
        super().__init__(name, clock)
        self.stream_timeout = timeout_s

    def transition(self, phase):
        with self.lock:
            super().transition(phase)
            if phase == 'active':
                self.deadline = self.clock()+1.

    def _validate_authorization(self, sequence, created, expires, payload, source_expires, now):
        if self.fault:
            raise HardwareSafetyError(f'{self.name}: {self.fault}')
        if self.sequence >= 0 and self.sequence not in self._recorded_dispatches:
            raise HardwareSafetyError('Previous reference has no confirmed SDK dispatch')
        if (self.phase != 'active' or type(sequence) is not int or sequence <= self.sequence
                or payload is None or not all(math.isfinite(x) for x in (created,expires,now))
                or not created <= now < expires <= created+self.stream_timeout+1e-9):
            raise HardwareSafetyError('Invalid DA command stream authorization')

    def _commit_authorization(self, sequence, created, expires, payload, now,
                              cycle_started, source_received, source_expires):
        self.authorization_timing.append((sequence,now))
        self.source_timing[sequence] = (cycle_started,source_received)
        self.source_timing = {s:v for s,v in self.source_timing.items() if s >= sequence-1}
        self.sequence = sequence
        self.startup_timing.setdefault('first_authorized',now)
        self.deadline = expires
        self.sources.append((sequence,expires,payload,expires,created))


class StreamEndpoint(MobileEndpoint):
    def _prepare_dispatch(self, command, stage, state):
        # Reuse calibrated arm/lift encoding and device joint bounds.
        base,hands,rows = self.validate_extras(command)
        limits = self.mobile_config['limits']
        if np.linalg.norm(state.v[:2]) > limits['base_linear_m_s']+1e-5 or abs(state.v[2]) > limits['base_angular_rad_s']+1e-5:
            raise HardwareSafetyError('Measured base speed exceeds deployment bounds')
        dispatch = HexWholeBodyEndpoint._prepare_dispatch(self,command,'arms-lift',state)
        from hex_device.generated.public_api_down_pb2 import APIDown
        expected = APIDown()
        xy = self.mapping.rotation().T @ base[:2]
        speed = expected.base_command.simple_move_command.xyz_speed
        speed.speed_x,speed.speed_y,speed.speed_z = float(xy[0]),float(xy[1]),float(base[2])
        pulses = {}
        for side,target in hands.items():
            hand = self.hands[side]
            previous = getattr(hand,'_last_command_send',None)
            prior = rows[side]['pos'] if previous is None else previous.position_command[0]
            pulses[side] = hand.convert_rad_to_positions(np.array([prior,rows[side]['pos'],target]),
                                                         hand.get_motor_pulse_per_rotations())
            if pulses[side].ndim != 1 or not pulses[side].size or not np.isfinite(pulses[side]).all():
                raise HardwareSafetyError('Invalid GP80 encoder pulse authorization')
        def write_mobile():
            self.devices['base'].set_vehicle_speed(float(xy[0]),float(xy[1]),float(base[2]))
            for side,target in hands.items():
                self.direct_gates[side].authorize_hand(self.mobile_config['grippers'][side]['device_id'],
                                                       pulses[side],command.expires)
                hand = self.hands[side]
                if self.extras['grippers'][side] is None:
                    hand.motor_command(self.command_type.BRAKE,[True])
                else:
                    hand.motor_command(self.command_type.POSITION,[target])
                wake = getattr(hand,'_policy_wake',None)
                if wake:
                    wake()
                self.gripper_servos[side].reference = target
            self.last_base = base.copy()
        dispatch['base'] = expected,write_mobile
        return dispatch

    def publish_frame(self, command, state):
        self.check_transport()
        self.check_health(state,'whole-body',check_timing=False)
        dispatch = self._prepare_dispatch(command,'whole-body',state)
        authorize_group([(self.direct_gates[n],p) for n,(p,_) in dispatch.items()],
                        command.sequence,command.created,command.expires,
                        cycle_started=command.created,source_received=min(command.state_received.values()),
                        source_expires=command.expires)
        for name,(_,write) in dispatch.items():
            if time.monotonic() >= command.expires:
                raise HardwareSafetyError('DA command expired during SDK dispatch')
            write()
            self.write_count += 1
        for name in dispatch:
            wake = getattr(self.devices[name],'_policy_wake',None)
            if wake:
                wake()


class NativeDriver:
    def __init__(self,mapping,profile,config,*,timeout_s):
        from functools import partial
        self.profile,self.timeout = profile,timeout_s
        robot = dict(config['robot'],gripper_travel_time_s=1.)
        self.endpoint = StreamEndpoint(mapping,robot,execute=True,
            downlink_gate_factory=partial(ContinuousGate,timeout_s=timeout_s))
        self.acquired = False
        self.pending = None

    @property
    def snapshot(self):
        state = self.endpoint.state()
        hands = self.endpoint.hand_feedback()
        self.state = state
        return dict(measured=sensor_packet(state,hands,self.endpoint.grippers),received=dict(state.received))

    def check(self):
        self.endpoint.check_transport()
        if self.acquired:
            self.endpoint.check_health(self.state,'whole-body',check_timing=False)

    def acquire(self):
        state = self.endpoint.acquire('whole-body',self.profile)
        self.acquired = True
        self.endpoint.start_feedback_worker(state)

    def submit(self,sequence,command,mit,*,created_s,received,measured,reference_velocity):
        state = self.state
        bundle = dict(sequence=sequence,created=created_s,expires=created_s+self.timeout,
                      received=received,source_q=[measured['joint_positions'][n] for n in JOINTS],
                      positions=[command['joint_positions'][n] for n in JOINTS],
                      velocity=[reference_velocity[n] for n in JOINTS],mit=mit)
        self.endpoint.extras = dict(sequence=sequence,base=command['base_velocity_body'],grippers=command['gripper_open'])
        self.endpoint.publish_frame(native_packet(bundle,state),state)
        self.pending = sequence

    def acknowledgement(self,sequence):
        gates = self.endpoint.direct_gates
        names = ('left', 'right', 'theta', 'base')
        # Each receipt refers to this frame's first SDK dispatch, not a newer
        # periodic resend or a merely installed gate authorization.
        with ExitStack() as locks:
            for name in sorted(names):
                locks.enter_context(gates[name].lock)
            rows = [gates[name].first_dispatch for name in names]
            if any(row is None or row['sequence'] != sequence for row in rows):
                return None
            return Receipt(sequence, max(row['sdk_send_returned'] for row in rows), 'sdk_accepted')

    def close(self):
        failures = self.endpoint.close()
        if failures:
            raise RuntimeError('Native close: '+str(failures))
