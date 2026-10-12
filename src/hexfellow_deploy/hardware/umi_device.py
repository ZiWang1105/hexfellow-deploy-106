"""Native UMI MIT publisher in a separate process; same public sensor protocol."""
import copy
from contextlib import ExitStack
from dataclasses import dataclass, fields
import multiprocessing as mp
import queue
import signal
import time
from types import SimpleNamespace

import numpy as np
import yaml

from ..robot.joints import JOINTS, LEFT, RIGHT, SIDES
from ..control.gripper import validate_gripper
from ..robot.upstream import load_umi
from ..control.velocity_limits import arm_velocity_bounds

load_umi()
from policy_sim.hardware.whole_body import RobotMapping
from policy_sim.hardware.state_mailbox import MailboxBusy


@dataclass(frozen=True)
class DeploymentMapping(RobotMapping):
    """Apply explicit deployment ceilings without changing upstream bindings."""
    deployment_arm_speed: float | str = 'profile'
    deployment_lift_speed: float = .025

    def runtime_limits(self, profile, state=None, **kwargs):
        limits = super().runtime_limits(profile, state, **kwargs)
        caps = arm_velocity_bounds(self.deployment_arm_speed, profile)
        for i, name in enumerate(JOINTS):
            cap = self.deployment_lift_speed if i == 0 else caps[name]
            limits['runtime_velocity_bounds'][name] = min(limits['runtime_velocity_bounds'][name], cap)
        return limits


def native_configuration(config, *, execute=False):
    load_umi()
    from policy_sim.hardware.session import load_configuration
    research = config.get('umi_research', False)
    if research and not config.get('research_release'):
        raise ValueError('UMI research execution requires the existing research_release binding')
    args = SimpleNamespace(robot_config=config['mapping'], feedback_policy=None,
        latency_policy='strict', demo_authorized=research, research_release=config.get('research_release'),
        verification_report=config.get('verification_report'), demo_low_speed=False)
    mapping, profile, _, _ = load_configuration(args)
    if config['network'] != {k: mapping.data[k] for k in config['network']}:
        raise ValueError('Network differs from the native UMI mapping; select the actual local mapping')
    settings = yaml.safe_load(open(config['controller']))
    from policy_sim.control.motion_profile import load_motion_profile
    selected = load_motion_profile(settings['motion_profile'])
    if selected.sha256 != profile.sha256:
        raise ValueError('Hardware IK profile must match the reviewed UMI mapping profile exactly')
    if research:
        from policy_sim.hardware.highspeed_trial import require_research_release
        require_research_release(profile, config['research_release'], config['mapping'])
    mapping = DeploymentMapping(**{f.name: getattr(mapping, f.name) for f in fields(mapping)},
        deployment_arm_speed=settings['arm_velocity_limit_rad_s'],
        deployment_lift_speed=settings['lift_velocity_limit_m_s'])
    if execute:
        # UMI's own authorize() selects commissioned or explicitly requested
        # research execution. No hardware review annotations are rewritten.
        effort = mapping.authorize(profile, 'whole-body', profile.sha256)
    else:
        effort = np.minimum(mapping.data['hardware_effort_limits_nm'], profile.mit_servo.torque_limit_nm)
    return mapping, profile, effort


def sensor_packet(state, rows, grippers):
    """Strip the native endpoint's private base/health state at the IPC boundary."""
    return dict(timestamp_s=min(state.timestamp, *(r['received'] for r in rows.values())),
        joint_positions={n: float(state.q[i + 3]) for i, n in enumerate(JOINTS)},
        joint_velocities={n: float(state.v[i + 3]) for i, n in enumerate(JOINTS)},
        gripper_state={s: grippers[s].measured(rows[s]['pos']) for s in rows})


def feedback_key(measured):
    """Separate asynchronous samples sharing the conservative public epoch."""
    return (measured['timestamp_s'],
            *(measured['joint_positions'][n] for n in JOINTS),
            *(measured['joint_velocities'][n] for n in JOINTS),
            *(measured['gripper_state'][s] for s in SIDES))


def native_packet(bundle, state):
    from policy_sim.hardware.whole_body import WholeBodyCommand
    # Only the SDK publisher sees base odometry. IK solves exactly 15 encoders.
    source = np.r_[state.q[:3], bundle['source_q']]
    positions = np.r_[state.q[:3], bundle['positions']]
    velocity = np.r_[np.zeros(3), bundle['velocity']]
    return WholeBodyCommand(bundle['sequence'], bundle['created'], bundle['expires'],
        bundle['received'], source, positions, velocity, bundle['mit'], np.zeros(3),
        cycle_started=bundle['created'], enqueued_at=bundle.get('enqueued_at'))


def publisher_main(config, execute, states, commands, statuses, activate, stop, endpoint_factory=None,
                   configuration_factory=None):
    load_umi()
    from policy_sim.hardware.gateway import GatewaySupervisor, GatewayShutdownRequested
    from policy_sim.hardware.whole_body import CommandGuard
    from .umi_endpoint import MobileEndpoint
    endpoint, supervisor = None, None
    phase, fault_reported = 'startup', False
    last_bundle = None
    reference_seed = None
    scheduling = ExitStack()
    cpu_scheduling = {}
    timing = None
    def status(**value):
        try:
            statuses.put_nowait(value)
            return True
        except queue.Full:
            return False
    def report_fault(reason):
        nonlocal fault_reported
        if fault_reported:
            return
        observed = time.monotonic()
        state = None if supervisor is None else supervisor.last_state
        wire = {name: dict(sequence=gate.sequence, deadline=gate.deadline,
                sources=[dict(sequence=s[0], lease_deadline=s[1], expires=s[3], created=s[4])
                         for s in list(gate.sources)[-2:]], fault=gate.fault,
                rejected_packet=getattr(gate, 'hand_rejection', None),
                authorization_timing=[dict(sequence=seq, authorized_at=stamp)
                                      for seq, stamp in getattr(gate, 'authorization_timing', [])],
                counts=dict(getattr(gate, 'counts', {})))
                for name, gate in getattr(endpoint, 'direct_gates', {}).items()}
        fault_reported = status(kind='fault', detail=str(reason), phase=phase,
            observed_s=observed, source_age_ms=({n: 1000 * (observed - stamp)
                for n, stamp in state.received.items()} if state is not None else {}),
            command_timing=list(supervisor.command_timing)[-16:] if supervisor else [],
            snapshot_timing=list(timing.snapshots)[-16:] if timing else [],
            wire=wire, cpu_scheduling=cpu_scheduling,
            latest_parent_command=({k: last_bundle[k] for k in ('sequence', 'created', 'expires', 'received')}
                                   if last_bundle is not None else None),
            reference_seed=reference_seed,
            latest_reference=({n: float(last_bundle['positions'][i]) for i,n in enumerate(JOINTS)}
                              if last_bundle is not None else None))
    if mp.current_process().name != 'MainProcess':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        from .scheduling import control_memory, control_scheduling
        if config.get('timing_diagnostics'):
            from .timing_diagnostics import GatewayTiming
            # Observer starts before CPU affinity is narrowed and before ownership.
            timing = GatewayTiming(config['timing_diagnostics'])
        cpu_scheduling.update(scheduling.enter_context(control_memory()))
        cpu_scheduling.update(scheduling.enter_context(control_scheduling('gateway')))
        mapping, profile, effort = (configuration_factory or native_configuration)(config, execute=execute)
        prepared = CommandGuard.prepare(profile, mapping, 'whole-body', effort) if execute else None
        from .scheduling import gateway_thread_pools, thread_affinity
        pools = gateway_thread_pools(cpu_scheduling.get('affinity', []))
        cpu_scheduling['gateway_threads'] = pools
        # SDK event loops, KCP receive/update threads, and their future children
        # inherit the SDK core pool. Restore and pin only this owner afterwards.
        # No other process's affinity or native command deadline is changed.
        if pools['sdk']:
            with thread_affinity(pools['sdk']):
                endpoint = (endpoint_factory or MobileEndpoint)(mapping, config, execute=execute)
            scheduling.enter_context(thread_affinity(pools['owner']))
        else:
            endpoint = (endpoint_factory or MobileEndpoint)(mapping, config, execute=execute)
        if timing:
            # Measure actual construction, now owned by the feedback thread
            # during control. Test endpoints without a builder retain polling.
            name = '_build_state' if hasattr(endpoint, '_build_state') else 'state'
            setattr(endpoint, name, timing.instrument_snapshot(getattr(endpoint, name)))
        if pools['sdk']:
            from functools import partial
            endpoint.feedback_worker_context = partial(thread_affinity, pools['sdk'])
        endpoint.cancel_requested = stop.is_set
        supervisor = GatewaySupervisor(endpoint, profile, mapping, 'whole-body', effort,
                                       stationary_reference=True, on_fault=report_fault)
        supervisor.prepared_guard = prepared
        # Use the same command/feedback wake pipe as the upstream gateway.
        # Always clear BEFORE draining commands, so a concurrent enqueue wakes us.
        wake = commands.wake
        endpoint.feedback_notify = wake.set
        for tap in getattr(endpoint, 'taps', {}).values():
            tap.on_update = wake.set
        acquisition_pending = False
        while not stop.is_set():
            if activate.is_set() and supervisor.guard is None:
                if timing:
                    timing.mark(0)  # Acquisition has intentional bounded waits.
                if not execute:
                    raise PermissionError('Passive publisher cannot acquire')
                phase = 'acquisition'
                supervisor.activate()
                # Freeze only measured encoder provenance for the one-time
                # history initialization. Live feedback continues separately.
                reference_seed = sensor_packet(supervisor.guard.initial,
                    endpoint.hand_feedback(), endpoint.grippers)
                acquisition_pending = True
                phase = 'control'
                if timing:
                    timing.observe(supervisor.last_state.received)
            wake.clear()
            if timing:
                timing.mark(1)
            try:
                bundle = commands.get_nowait()
            except queue.Empty:
                bundle = None
            if bundle is not None:
                last_bundle = bundle
                if supervisor.guard is None:
                    raise PermissionError('Command before native acquisition')
                endpoint.extras = bundle
                packet = native_packet(bundle, supervisor.last_state)
            else:
                packet = None
            if timing:
                timing.mark(2)
            state = supervisor.tick(packet)
            if timing:
                timing.observe(state.received)
                timing.mark(3)
            rows = endpoint.hand_feedback()
            measured = sensor_packet(state, rows, endpoint.grippers)
            if max([*state.received.values(), *(r['received'] for r in rows.values())]) - measured['timestamp_s'] > config['feedback_max_skew_s']:
                raise RuntimeError('Native robot/GP80 feedback skew exceeded')
            try:
                if timing:
                    timing.mark(4)
                # Private provenance stays outside the policy-facing sensor packet.
                telemetry = dict(published_s=time.monotonic(), source_received={
                    **state.received, **{s + '_gp80': r['received'] for s, r in rows.items()}})
                states.put_nowait((measured, state.received, telemetry))
                # Parent may enter control only after post-acquisition sensor
                # feedback is published, not after a pre-initialization cache.
                if acquisition_pending and status(kind='acquired', reference_seed=reference_seed):
                    acquisition_pending = False
            except queue.Full:
                pass  # Latest feedback may be overwritten; command references may not.
            # A command wakes immediately; 4 ms is only the health fallback,
            # matching upstream's SDK report period. No command is coalesced.
            if timing:
                timing.mark(5)
            wake.wait(.004)
    except GatewayShutdownRequested:
        # Upstream raises this when a requested shutdown races a control tick.
        # Its finally path still confirms device stops; it is not a wire fault.
        if not stop.is_set():
            report_fault('Unexpected gateway shutdown request')
            if supervisor:
                supervisor.fail('Unexpected gateway shutdown request')
    except BaseException as error:
        report_fault(f'{type(error).__name__}: {error}')
        if supervisor:
            supervisor.fail(str(error))
    finally:
        if timing:
            timing.mark(0)
        failures = []
        if supervisor:
            failures.extend(supervisor.stop_failures)
        if endpoint:
            try:
                failures.extend(endpoint.close())
            except BaseException as error:
                failures.append(str(error))
        diagnostic_error = None
        if timing:
            try:
                timing.save(endpoint, supervisor)
            except Exception as error:
                diagnostic_error = f'{type(error).__name__}: {error}'
            finally:
                timing.close()
        status(kind='closed', stop_failures=failures,
               feedback_worker=getattr(endpoint, 'feedback_worker_report', None),
               timing_diagnostic_error=diagnostic_error,
               cpu_scheduling=cpu_scheduling,
               stop_observations=getattr(endpoint, 'stop_observations', {}),
               stop_feedback_errors=getattr(endpoint, 'stop_feedback_errors', []),
               initialization_checkpoints=getattr(endpoint, 'initialization_checkpoints', []))
        scheduling.close()


class UMIHardwareDevice:
    def __init__(self, config, ik, *, execute=False):
        load_umi()
        from policy_sim.hardware.state_mailbox import LatestStateMailbox, CommandMailbox
        self.config, self.ik, self.execute = config, ik, execute
        self.ctx = mp.get_context('spawn')
        self.states, self.commands = LatestStateMailbox(self.ctx), CommandMailbox(self.ctx)
        self.statuses = self.ctx.Queue(maxsize=16)
        self.activate_event, self.stop_event = self.ctx.Event(), self.ctx.Event()
        self.process = self.ctx.Process(target=publisher_main, args=(config, execute, self.states,
            self.commands, self.statuses, self.activate_event, self.stop_event), name='umi-mit-publisher')
        self.measured, self.source_received = None, {}
        self.feedback_records = {}
        self.acquired, self.fault, self.closed = False, None, None
        self.reference_seed = None
        self.fault_report, self.feedback_diagnostics = None, {}
        self.telemetry = {}
        self.sequence = 0
        self.ik.control_clock = time.monotonic
        self.ik.feedback_provider = self.feedback_for

    def feedback_for(self, measured):
        received = self.feedback_records.get(feedback_key(measured))
        if received is None:
            raise RuntimeError('UMI source feedback provenance unavailable')
        return dict(received)

    def poll(self):
        for _ in range(16):
            try:
                value = self.statuses.get_nowait()
            except queue.Empty:
                break
            if value['kind'] == 'acquired':
                self.reference_seed = value['reference_seed']
                self.acquired = True
            elif value['kind'] == 'fault':
                if not self.fault:
                    self.fault, self.fault_report = value['detail'], value
            elif value['kind'] == 'closed':
                self.closed = value
        if self.fault:
            raise RuntimeError(self.fault)
        if self.process.pid is not None and self.process.exitcode is not None:
            raise RuntimeError(f'UMI publisher exited: {self.closed}')

    def open(self):
        native_configuration(self.config, execute=self.execute)
        self.process.start()
        deadline = time.monotonic() + 30
        while self.measured is None:
            self.poll()
            try:
                self.read()
            except queue.Empty:
                if time.monotonic() >= deadline:
                    self.close()
                    raise RuntimeError('Native UMI telemetry startup timed out')
                time.sleep(.005)

    def read(self):
        self.poll()
        try:
            # Upstream returns (published payload, mailbox diagnostics). Our
            # payload contains public sensors, native timestamps and optional
            # private publication diagnostics (old two-field fixtures supported).
            payload, mailbox_diagnostics = self.states.get_latest()
            measured, received = payload[:2]
            self.telemetry = payload[2] if len(payload) == 3 else {}
            self.feedback_diagnostics = dict(mailbox_diagnostics, mailbox='publication')
            self.measured = measured
            self.source_received[measured['timestamp_s']] = received
            # Identical samples at one oldest-source epoch keep their earliest
            # provenance: another read must never renew their motion lease.
            self.feedback_records.setdefault(feedback_key(measured), dict(received))
            while len(self.feedback_records) > 256:
                del self.feedback_records[next(iter(self.feedback_records))]
            while len(self.source_received) > 256:
                del self.source_received[next(iter(self.source_received))]
        except queue.Empty as error:
            self.feedback_diagnostics['mailbox'] = 'busy' if isinstance(error, MailboxBusy) else 'no_new_publication'
            if self.measured is None:
                raise
        now = time.monotonic()
        sources = self.telemetry.get('source_received', self.source_received.get(self.measured['timestamp_s'], {}))
        self.feedback_diagnostics.update(sensor_age_ms=1000 * (now - self.measured['timestamp_s']),
            source_age_ms={n: 1000 * (now - stamp) for n, stamp in sources.items()},
            publication_age_ms=(1000 * (now - self.telemetry['published_s'])
                                if 'published_s' in self.telemetry else None))
        if now - self.measured['timestamp_s'] > self.config['feedback_max_age_s']:
            self.poll()  # Prefer an original gateway fault reported before blocking stop.
            raise RuntimeError(f'Native UMI feedback expired: {self.feedback_diagnostics}')
        return copy.deepcopy(self.measured)

    def acquire(self):
        if not self.execute:
            raise PermissionError('Native UMI observation-only device')
        self.activate_event.set()
        deadline = time.monotonic() + 10
        while not self.acquired:
            # Upstream acquire is synchronous and includes lift configuration
            # waits. Its own health checks supervise that bounded initialization;
            # the parent must not treat its pre-acquisition cache as live control.
            self.poll()
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError('Native UMI acquisition timed out')
            time.sleep(.001)
        self.read()  # The acquired acknowledgement follows fresh publication.
        if self.reference_seed is None:
            raise RuntimeError('Native acquisition did not provide its encoder reference seed')
        self.ik.initialize_reference(self.reference_seed, stationary_reference=True)

    def prepare(self, command):
        if set(command['gripper_open']) != set(SIDES):
            raise ValueError('Gripper commands require both hands')
        grippers = {s: validate_gripper(command['gripper_open'][s], allow_hold=True) for s in SIDES}
        self.poll()
        result, source = self.ik.last_result, self.ik.last_source
        if result is None or result.mit is None or not result.executable or result.diagnostics.fault:
            raise RuntimeError('No executable native UMI MIT result')
        received = source.get('feedback_received')
        if received is None:
            received = self.source_received.get(source['timestamp_s'])
        if received is None:
            raise RuntimeError('UMI source feedback provenance unavailable')
        positions = np.array([result.positions[self.ik.indices[n]] for n in JOINTS])
        if not np.array_equal(positions, [command['joint_positions'][n] for n in JOINTS]):
            raise RuntimeError('Actuator command differs from native UMI result')
        created = result.mit.timestamp_s
        from policy_sim.hardware.whole_body import COMMAND_TTL
        from policy_sim.control.feedback_policy import received_deadline
        expires = min(created + COMMAND_TTL, received_deadline(received, 'strict'))
        if not created <= time.monotonic() < expires:
            raise RuntimeError('Native MIT result expired before submission')
        bundle = dict(sequence=self.sequence, created=created, expires=expires,
            received=dict(received), source_q=[source['joint_positions'][n] for n in JOINTS],
            positions=positions, velocity=np.array([result.velocity[self.ik.indices[n]] for n in JOINTS]),
            mit=result.mit, base=list(command['base_velocity_body']),
            grippers=grippers)
        self.sequence += 1
        return bundle

    def send(self, bundle):
        if not self.acquired or time.monotonic() >= bundle['expires']:
            raise RuntimeError('Native MIT command expired or publisher inactive')
        self.poll()
        try:
            bundle['enqueued_at'] = time.monotonic()
            self.commands.put_nowait(bundle)
        except queue.Full:
            self.stop_event.set()
            raise RuntimeError('Native MIT mailbox full; reference must not be skipped')

    def stop(self):
        self.stop_event.set()
        self.acquired = False
        if self.process.pid is None:
            return
        # Drain diagnostics while waiting: a large final report can otherwise
        # block the child Queue feeder and prevent the process from exiting.
        deadline = time.monotonic() + 45
        while self.process.is_alive() and time.monotonic() < deadline:
            try:
                self.poll()
            except RuntimeError:
                pass  # Preserve the report; shutdown still requires confirmation.
            self.process.join(timeout=min(.02, max(0., deadline - time.monotonic())))
        try:
            self.poll()
        except RuntimeError:
            pass
        if self.process.is_alive() or self.closed is None or self.closed['stop_failures']:
            raise RuntimeError(f'Native UMI shutdown not confirmed: {self.closed}')

    def close(self):
        self.stop()
