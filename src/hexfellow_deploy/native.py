"""Native leaf adapter: calibrated SDK IO without the former gateway runtime."""
from contextlib import ExitStack
import math
import time
import numpy as np

from hexfellow_deploy.hardware.umi_endpoint import MobileEndpoint, GP80WireGate
from hexfellow_deploy.hardware.umi_device import sensor_packet, native_packet
from hexfellow_deploy.robot.joints import JOINTS
from policy_sim.hardware.mit_wire import authorize_group, motion_payload
from policy_sim.hardware.hexfellow_endpoint import HexWholeBodyEndpoint
from policy_sim.hardware.x7 import HardwareSafetyError
from .reference import Receipt


class ContinuousGate(GP80WireGate):
    """Finite command-stream watchdog independent of per-device feedback age.

    Packet matching, checked KCP and stop filtering remain upstream. Only the
    previous fixed admission/source lease is replaced by the shared DA timeout.
    """
    def __init__(self, name, clock=time.monotonic, timeout_s=.1, defer_activation=False,
                 startup_lift=False):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Command timeout must be finite and positive")
        super().__init__(name, clock)
        self.stream_timeout = timeout_s
        self.defer_activation = defer_activation
        self.fixed_lift = defer_activation and name == 'theta'
        self.startup_lift = startup_lift
        self.startup_lift_payload = None

    def transition(self, phase):
        with self.lock:
            if phase == 'active' and self.fixed_lift and not self.startup_lift:
                if self.phase != 'initializing':
                    raise HardwareSafetyError('Fixed lift cannot resume a motion stream')
                # Use the existing checked-KCP hold receipt, without enabling
                # motion or consuming the terminal shutdown stop token.
                self.hold_resume = True
                self.recovery_epoch += 1
                super().transition('holding')
                return
            if phase == 'active' and self.defer_activation:
                super().transition('ready')
                if self.startup_lift:
                    self.deadline = self.clock()+30.
                return  # Keep the finite initialization deadline during setup.
            super().transition(phase)
            if phase == 'active':
                self.deadline = self.clock()+1.

    def begin_lift_startup(self):
        with self.lock:
            if not self.fixed_lift or not self.startup_lift or self.phase != 'ready' or self.clock() >= self.deadline:
                raise HardwareSafetyError('Lift startup is not authorized in this phase')
            super().transition('active')
            self.deadline = self.clock()+self.stream_timeout

    def finish_lift_startup(self):
        with self.lock:
            if (not self.fixed_lift or not self.startup_lift or self.phase != 'active'
                    or self.fault or self.clock() >= self.deadline):
                raise HardwareSafetyError('Lift startup cannot enter fixed hold')
            self.startup_lift_payload = self.sources[-1][2] if self.sources else None
            self.startup_lift = False
            self.hold_resume = True
            self.recovery_epoch += 1
            self.sources.clear()
            super().transition('holding')

    def finish_startup(self):
        with self.lock:
            if self.fixed_lift:
                raise HardwareSafetyError('Fixed lift must remain in brake-only hold')
            if self.phase != 'ready' or self.clock() >= self.deadline:
                raise HardwareSafetyError('Native startup did not complete before its deadline')
            self.defer_activation = False
            self.transition('active')

    def resume_delay(self, epoch):
        if self.fixed_lift:
            raise HardwareSafetyError('Fixed lift must remain in brake-only hold')
        return super().resume_delay(epoch)

    def filter(self, data):
        with self.lock:
            if self.phase == 'ready' and self.clock() >= self.deadline:
                self.fault = self.fault or 'Native startup deadline expired before stream activation'
                self.transition('stopping')
            if self.fixed_lift and self.phase == 'holding' and motion_payload(self.name, data) is not None:
                if motion_payload(self.name, data) != self.startup_lift_payload:
                    return self.reject()
                # The SDK may have copied the final startup target before BRAKE.
                # Upstream replaces that known in-flight target with BRAKE.
            return super().filter(data)

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
    """Agent arm/base stream with the fixed lift physically braked."""

    def validate_lift_start(self, state):
        from hexfellow_deploy.control.lift_limits import FIXED_LIFT_MAX_M, FIXED_LIFT_RANGE_ROUNDING_MAX_M
        low, high = state.health['theta']['range']
        height = float(state.q[3])
        sdk_height = self.mapping.lift_command(height, (low, high))
        pulses = state.health['theta'].get('pulse_per_meter')
        rounding = (min(.5 / pulses, FIXED_LIFT_RANGE_ROUNDING_MAX_M)
                    if isinstance(pulses, (int, float)) and math.isfinite(pulses) and pulses > 0 else 0.)
        if (not np.isfinite([low, high, height, sdk_height]).all() or high <= low
                or not -rounding <= height <= FIXED_LIFT_MAX_M + rounding
                or not low <= sdk_height <= high):
            self.fixed_lift_range_failure = dict(height_m=height, sdk_height_m=sdk_height,
                sdk_range_m=[low, high], model_range_m=[0., FIXED_LIFT_MAX_M],
                nominal_rounding_tolerance_m=rounding, pulse_per_meter=pulses)
            raise HardwareSafetyError('Fixed lift feedback outside physical device/model range: '
                                      f'{self.fixed_lift_range_failure}')

    def acquire(self, stage, profile):
        from policy_sim.hardware.shutdown import ThetaObservedStop
        state = super().acquire(stage, profile)
        if getattr(self, 'startup_lift_m', None) is not None:
            state = self.position_startup_lift(state, stage)
        # BRAKE is a stop command, never an endpoint POSITION or auto-retreat.
        brake_requested = time.monotonic()
        self.devices['theta'].motor_command(self.command_type.BRAKE, True)
        self.fixed_lift_reference = float(state.q[3])
        observer = ThetaObservedStop()
        deadline = time.monotonic()+self.timeout
        while True:
            state = self.state()
            self.check_transport()
            self.check_health(state, stage)
            self.validate_lift_start(state)
            progress = getattr(self, 'startup_progress', None)
            if progress is not None:
                progress(dict(getattr(self, 'startup_lift_report', {}), phase='confirming_brake'))
            if abs(state.q[3]-self.fixed_lift_reference) > .002:
                raise HardwareSafetyError('Fixed lift moved during brake acquisition')
            evidence = self.fixed_lift_brake_evidence()
            if observer.update(state, time.monotonic(), brake_requested, evidence, healthy=True):
                self.fixed_lift_confirmation = dict(observer.details)
                if getattr(self, 'startup_lift_m', None) is not None:
                    if abs(state.q[3]-self.startup_lift_m) > .001:
                        raise HardwareSafetyError('Startup lift left its target tolerance during braking')
                    self.startup_lift_report.update(phase='complete', locked_height_m=float(state.q[3]))
                for name in ('left', 'right', 'base'):
                    self.direct_gates[name].finish_startup()
                return state
            if getattr(self, 'cancel_requested', lambda: False)() or time.monotonic() >= deadline:
                raise HardwareSafetyError('Fixed lift brake was not confirmed')
            time.sleep(.004)

    def position_startup_lift(self, initial, stage):
        from .startup_lift import LiftArrival
        from hex_device.generated.public_api_down_pb2 import APIDown
        target = self.startup_lift_m
        arrival = LiftArrival(target, time.monotonic())
        self.startup_lift_report = arrival.report
        self.startup_lift_report['initial_height_m'] = float(initial.q[3])
        low, high = initial.health['theta']['range']
        sdk_target = float(self.mapping.lift_command(target, (low, high)))
        if not low+.002 <= sdk_target <= high-.002:
            raise HardwareSafetyError('Startup lift target outside calibrated device interior')
        device = self.devices['theta']
        expected = APIDown()
        pulses = device.convert_rad_to_positions(np.array([sdk_target]), device._pulse_per_rotation)
        expected.linear_lift_command.target_pos = int(np.sign(device._max_pos)*pulses[0])
        gate = self.direct_gates['theta']
        gate.begin_lift_startup()
        sequence = 0
        next_send = next_progress = 0.
        while True:
            state = self.state()
            now = time.monotonic()
            self.check_transport()
            self.check_health(state, stage)
            if now >= gate.deadline:
                raise HardwareSafetyError('Startup lift command lease expired')
            self.validate_lift_start(state)
            if abs(state.v[3]) > getattr(self, 'feedback_lift_speed', .025)+.002:
                raise HardwareSafetyError('Startup lift exceeds configured measured speed')
            if (np.max(np.abs(state.q[4:]-initial.q[4:])) > .01
                    or np.max(np.abs(state.v[4:])) > .01
                    or np.max(np.abs(state.v[:3])) > .002):
                raise HardwareSafetyError('Arms or base moved during startup lift positioning')
            done = arrival.update(float(state.q[3]), float(state.v[3]), state.received['theta'], now,
                                  stopped=getattr(self, 'cancel_requested', lambda: False)())
            if done:
                gate.finish_lift_startup()
                # Clear an unsent SDK target; any already copied target is
                # replaced with BRAKE by the permanent hold filter.
                with device._status_lock:
                    device._target_pos = None
                return state
            if now >= next_send:
                with gate.lock:
                    dispatched = gate.sequence < 0 or gate.sequence in gate._recorded_dispatches
                if dispatched:
                    gate.authorize(sequence, now, now+gate.stream_timeout, expected,
                                   source_received=state.received['theta'])
                    device.motor_command(self.command_type.POSITION, sdk_target)
                    wake = getattr(device, '_policy_wake', None)
                    if wake:
                        wake()
                    sequence += 1
                    next_send = now+.02
                elif now >= gate.deadline:
                    raise HardwareSafetyError('Startup lift SDK dispatch deadline expired')
            if now >= next_progress:
                self.startup_progress(dict(arrival.report))
                next_progress = now+.02
            time.sleep(.004)

    def fixed_lift_brake_evidence(self):
        """Transport receipt plus a drained stream; not a firmware brake-state ACK."""
        gate = self.direct_gates['theta']
        for api in self.apis:
            client = getattr(api, '_HexDeviceApi__kcp_client', None)
            transport = getattr(client, '_policy_udp', None)
            if transport is None or transport.gate is not gate:
                continue
            valid = (gate.phase == 'holding' and not gate.fault and not transport.fault
                     and transport.identity == (client.server_address, client.server_port,
                                                client._sock, getattr(client, '_kcp', None))
                     and transport.hold_confirmed(gate.recovery_epoch) and transport.motion_drained())
            if valid and not hasattr(self, 'fixed_lift_receipt_observed_at'):
                self.fixed_lift_receipt_observed_at = time.monotonic()
            return dict(brake_received=bool(valid), no_pending_motion=bool(valid),
                        acknowledged_at=getattr(self, 'fixed_lift_receipt_observed_at', float('inf')))
        return dict(brake_received=False, no_pending_motion=False)

    def check_fixed_lift(self, state):
        self.validate_lift_start(state)
        now = time.monotonic()
        if (not getattr(self, 'fixed_lift_confirmation', None)
                or not self.fixed_lift_brake_evidence()['brake_received']
                or not 0 <= now-state.received['theta'] <= .04
                or state.health['theta']['state'] not in {'LsBrake', 'LsAlgorithmControl'}
                or abs(state.v[3]) > .002
                or abs(state.q[3]-self.fixed_lift_reference) > .002):
            raise HardwareSafetyError('Fixed lift brake lost or measured lift moved')

    def validate_lift_command(self, command):
        if (not np.isfinite([command.positions[3], command.velocity[3]]).all()
                or abs(command.positions[3]-self.fixed_lift_reference) > .002
                or abs(command.velocity[3]) > 1e-9):
            raise HardwareSafetyError('Agent requested motion of the braked fixed lift')

    def _prepare_dispatch(self, command, stage, state):
        # Reuse calibrated arm/lift encoding and device joint bounds.
        self.check_fixed_lift(state)
        base,hands,rows = self.validate_extras(command)
        limits = self.mobile_config['limits']
        if np.linalg.norm(state.v[:2]) > limits['base_linear_m_s']+1e-5 or abs(state.v[2]) > limits['base_angular_rad_s']+1e-5:
            raise HardwareSafetyError('Measured base speed exceeds deployment bounds')
        dispatch = HexWholeBodyEndpoint._prepare_dispatch(self,command,'dual-arm',state)
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
                self.gripper_publishers[side].request(self.extras['grippers'][side],
                    expires=command.expires, opening_target=target)
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
    def __init__(self,mapping,profile,config,*,timeout_s,startup_lift_m=None):
        from functools import partial
        self.profile,self.timeout = profile,timeout_s
        robot = dict(config['robot'],gripper_travel_time_s=1.)
        self.endpoint = StreamEndpoint(mapping,robot,execute=True,
            downlink_gate_factory=partial(ContinuousGate,timeout_s=timeout_s,defer_activation=True,
                                         startup_lift=startup_lift_m is not None))
        from .startup_lift import validate_target
        self.endpoint.startup_lift_m = validate_target(startup_lift_m)
        self.acquired = False
        self.pending = None

    @property
    def snapshot(self):
        state = self.endpoint.state()
        hands = self.endpoint.hand_feedback()
        self.state = state
        return dict(measured=sensor_packet(state,hands,self.endpoint.grippers),
                    received=dict(state.received), timing=state.timing)

    def check(self):
        for name, gate in self.endpoint.direct_gates.items():
            if gate.fault and gate.phase != 'stopping':
                raise HardwareSafetyError(f'{name} downlink: {gate.fault}')
        self.endpoint.check_transport()
        if self.acquired:
            self.endpoint.check_health(self.state,'whole-body',check_timing=False)
            self.endpoint.check_fixed_lift(self.state)

    def acquire(self, *, startup_lift_m=None, progress=lambda report: None, stopped=lambda: False):
        if startup_lift_m != self.endpoint.startup_lift_m:
            raise ValueError('Startup lift target must match the preconfigured wire gates')
        self.endpoint.startup_progress = progress
        self.endpoint.cancel_requested = stopped
        state = self.endpoint.acquire('whole-body',self.profile)
        self.startup_lift_report = getattr(self.endpoint, 'startup_lift_report', None)
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
        # Theta receives no POSITION frames. Its independently fresh feedback
        # must confirm the physical brake; receipts cover actual motion sends.
        self.endpoint.check_fixed_lift(self.state)
        names = ('left', 'right', 'base')
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

    def diagnostics(self):
        """Freeze bounded evidence before close clears SDK clients and stop buffers.

        Never wait for a sender lock or serialize to disk in this path. An
        unavailable section is explicit; collecting evidence must not block stop.
        """
        from .scheduling import runtime_state
        started = time.monotonic()
        result = dict(observed_s=started, runtime=runtime_state(), devices={})
        endpoint = self.endpoint
        worker = getattr(endpoint, 'feedback_worker', None)
        result['feedback_worker'] = worker.report() if worker is not None else None
        result['fixed_lift_confirmation'] = getattr(endpoint, 'fixed_lift_confirmation', None)
        result['startup_lift'] = getattr(endpoint, 'startup_lift_report', None)
        result['fixed_lift_range_failure'] = getattr(endpoint, 'fixed_lift_range_failure', None)
        result['gripper_controllers'] = {s:p.report() for s,p in getattr(endpoint, 'gripper_publishers', {}).items()}
        result['grippers'] = {}
        for side, tap in getattr(endpoint, 'hand_taps', {}).items():
            if tap.lock.acquire(blocking=False):
                try:
                    result['grippers'][side] = (dict(error=tap.error) if tap.error else
                                                dict(tap.value) if tap.value is not None else None)
                finally:
                    tap.lock.release()
            else:
                result['grippers'][side] = dict(error='Gripper feedback lock busy at diagnostic capture')
        for name, gate in endpoint.direct_gates.items():
            row = result['devices'][name] = {}
            if gate.lock.acquire(blocking=False):
                try:
                    row['gate'] = dict(phase=gate.phase, fault=gate.fault,
                        sequence=gate.sequence, deadline=gate.deadline,
                        rejected_packet=getattr(gate, 'hand_rejection', None),
                        counts=dict(gate.counts), first_dispatch=gate.first_dispatch,
                        first_dispatch_timing=list(gate.first_dispatch_timing.copy()),
                        authorization_timing=list(gate.authorization_timing.copy()))
                finally:
                    gate.lock.release()
            else:
                row['gate_error'] = 'Sender lock busy at diagnostic capture'
            try:
                # API construction follows tap order, which can differ from
                # the sorted order used to lock gates during group admission.
                api = endpoint.apis[list(endpoint.taps).index(name)]
                client = getattr(api, '_HexDeviceApi__kcp_client', None)
                transport = getattr(client, '_policy_udp', None)
                row['transport_fault'] = getattr(transport, 'fault', None)
                for key, attribute in [('udp_send', 'rows'), ('udp_receive', 'receive_rows'),
                                       ('kcp_process', 'process_rows')]:
                    values = getattr(transport, attribute, None)
                    row[key] = list(values.copy()) if values is not None else []
                tap = endpoint.taps[name]
                row['feedback'] = list(tap.frames.copy())
                latest = getattr(tap, 'value', None) or {}
                row['latest_health'] = {key: latest[key] for key in
                    ('received', 'parking_stop', 'session_holder', 'mine', 'initialized', 'state')
                    if key in latest}
                row['clock_note'] = 'Host send return is not device receipt; kernel clock conversion is estimated'
            except Exception as error:
                row['capture_error'] = f'{type(error).__name__}: {error}'
        result['capture_ms'] = (time.monotonic()-started)*1000
        return result
