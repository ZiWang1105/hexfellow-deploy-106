"""Mouse TCP targets through the existing feedback-driven whole-body session."""

import argparse
import fcntl
import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

from ..control.target_replay import tcp_poses
from ..control.target_timing import TargetSample
from ..paths import PROJECT_ROOT
from . import session
from .drag_viewer import DragViewer
from .research_profiles import PROFILE as PROFILE
from .research_profiles import RUNTIME as RUNTIME
from .research_profiles import SUPPORTED_PROFILES
from .research_profiles import HIGHSPEED_PROFILES
from .quest_trial import audit_passed
from .verification import device_binding, digest, software_binding
from .x7 import HardwareSafetyError

MAX_INPUT_AGE = .1
STABLE_INPUT_TIME = .5


class DragController(session.SessionController):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source
        self.default_complete = False
        self.prepared = False

    def command(self, name, payload, state, now):
        if name == 'drag':
            if not self.idle or not self.prepared:
                raise ValueError('Complete D default, then I init; wait for READY before T.')
            if np.any(abs(state.v) > session.SETTLED_SPEED):
                raise ValueError('Wait for the measured robot to settle before T.')
            if np.any(abs(self.core._last_velocity) > session.SETTLED_SPEED):
                raise ValueError('Wait for the command reference to settle before T.')
            if not self.source.stable():
                raise ValueError('Wait for 0.5 seconds of fresh viewer input before T.')
            self.source.anchor(tcp_poses(self.core, state.q), 'Arming: aligning targets to measured TCPs')
            self.hold_q = self.core._last_safe_positions.copy()
            self.enter('drag-arming', now)
            return
        if name == 'init' and not self.default_complete:
            raise ValueError('Press D and wait for default arrival before I.')
        if name not in {'default', 'init', 'stop', 'quit'}:
            raise ValueError('Drag session only accepts D / I / T / Space / Q.')
        if name == 'default' and self.idle:
            self.default_complete = self.prepared = False
        if name == 'init' and self.idle:
            self.prepared = False
        if name in {'stop', 'quit'}:
            self.source.anchor(tcp_poses(self.core, state.q), 'Stopping; wait for HOLD, then T to re-engage')
        super().command(name, payload, state, now)

    def sample(self, state, now, quest=None):
        old_phase = self.phase
        if self.phase in {'drag', 'drag-arming'}:
            if not self.source.fresh():
                # Allow one producer interval to acknowledge a new anchor epoch.
                if self.phase == 'drag-arming' and self.source.clock() - self.source.anchored_at <= MAX_INPUT_AGE:
                    return tcp_poses(self.core, self.hold_q), self.hold_q, np.zeros(18)
                self.command('stop', None, state, now)
                self.source.message = 'Viewer input stale; stopped. Fresh input + T required.'
                self.source.input_holds += 1
            else:
                if self.phase == 'drag-arming':
                    if self.core.control_state == 'HOLD':
                        self.core.resume_tracking()
                    self.source.command_anchors = self.core.reanchor_tracking_tasks(state.q, state.v)
                    self.enter('drag', now)
                    self.source.message = 'Following; T or Space stops'
                frame = self.source.frame
                self.target_sample = TargetSample(frame['at'], frame['sequence'], frame['epoch'], 'viewer_monotonic')
                return self.source.command_targets(measured=self.core.measured_pose_tracking), None, None
        result = super().sample(state, now, quest)
        if old_phase == 'default' and self.phase == 'hold':
            self.default_complete = True
            self.source.message = 'Default reached; press I'
        if old_phase == 'init' and self.phase == 'ready':
            self.prepared = True
            self.source.anchor(tcp_poses(self.core, state.q), 'READY; press T to follow')
        if old_phase == 'stopping' and self.phase == 'hold':
            self.source.anchor(tcp_poses(self.core, state.q), 'HOLD; press T to re-engage')
        return result


class DragInput:
    def __init__(self, viewer, *, binding=None, clock=None):
        self.clock = clock or (lambda: time.monotonic())
        self.viewer = viewer
        self.binding = binding
        self.epoch = 0
        self.anchors = None
        self.anchored_at = self.clock()
        self.frame = None
        self.targets = None
        self.command_anchors = None
        self.sequence = -1
        self.last_at = -1.
        self.stable_since = None
        self.message = 'Press D for default, wait, then I for init'
        self.suspended = False
        self.input_holds = 0
        self.device_holds = 0
        self.active_cycles = 0
        self.ik_ms = 0.
        self.errors_mm = [0., 0.]
        self.raw_errors_mm = [0., 0.]
        self.raw_rotation_deg = [0., 0.]
        self.last_display = None
        self.recovery_default_pending = False
        self.simulation_plant = None
        self.retain_simulation = False

    def validate_session(self, args, mapping, profile):
        if args.mode not in {'sim', 'live'} or args.commands:
            raise HardwareSafetyError('drag supports interactive sim/live only')
        if (profile.profile_id not in SUPPORTED_PROFILES or profile.motion_envelope != 'none'
                or mapping.feedback_policy != 'strict' or mapping.latency_policy != 'hold-resume'):
            raise HardwareSafetyError('drag requires its dedicated profile, strict feedback and hold-resume')
        if args.mode == 'live' and (not mapping.demo_authorized or self.binding is None
                                   or software_binding(mapping) != self.binding):
            raise HardwareSafetyError('drag requires its current independent read-only preflight binding')

    def controller_factory(self, *args, **kwargs):
        return DragController(*args, source=self, **kwargs)

    def before_acquire(self, args, mapping, profile):
        self.validate_session(args, mapping, profile)
        self.check_viewer_ready()

    def check_viewer_ready(self):
        self.ingest(self.viewer.targets.get(), self.clock())
        if (self.viewer.quit.is_set() or self.viewer.stop.is_set()
                or not self.viewer.process.is_alive() or not self.fresh()):
            raise HardwareSafetyError('viewer stopped or became stale during acquisition preparation')

    def anchor(self, targets, message):
        self.epoch += 1
        self.anchors = {s: p.homogeneous.tolist() for s, p in targets.items()}
        self.frame = self.targets = None
        self.command_anchors = None
        self.stable_since = None
        self.anchored_at = self.clock()
        self.message = message

    def command_targets(self, *, measured=False):
        """Use exact viewer goals, or preserve legacy command-anchor offsets."""
        if measured:
            return {side: target.copy() for side, target in self.targets.items()}
        result = {}
        for side, target in self.targets.items():
            start = np.asarray(self.anchors[side])
            command = self.command_anchors[side]
            result[side] = pin.SE3(target.rotation @ start[:3, :3].T @ command.rotation,
                                  command.translation + target.translation - start[:3, 3])
        return result

    def ingest(self, frame, now):
        if frame is None:
            return
        sequence, at, epoch = frame['sequence'], float(frame['at']), frame['epoch']
        if sequence <= self.sequence or epoch != self.epoch:
            return
        if not np.isfinite(at) or at <= self.last_at or not 0 <= now - at <= MAX_INPUT_AGE:
            return
        targets = {}
        for side in ('left', 'right'):
            matrix = np.asarray(frame['targets'][side], dtype=float)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
                raise HardwareSafetyError('invalid viewer TCP transform')
            targets[side] = pin.SE3(matrix)
        if self.stable_since is None or at - self.last_at > MAX_INPUT_AGE:
            self.stable_since = at
        self.sequence, self.last_at = sequence, at
        self.frame, self.targets = frame, targets

    def fresh(self, now=None):
        now = self.clock() if now is None else now
        return self.frame is not None and 0 <= now - self.frame['at'] <= MAX_INPUT_AGE

    def stable(self, now=None):
        now = self.clock() if now is None else now
        return (self.fresh(now) and self.stable_since is not None
                and self.last_at - self.stable_since >= STABLE_INPUT_TIME)

    def suspend(self, engine):
        if not self.suspended:
            self.suspended = True
            self.device_holds += 1
            self.anchor(tcp_poses(engine.core, engine.hold_q),
                        'Device delay: holding; manual T required after recovery')

    def recovered(self, engine, state, now):
        self.suspended = False
        self.anchor(tcp_poses(engine.core, state.q), 'Recovered into HOLD; wait for fresh input, then T')
        if engine.phase in {'drag', 'drag-arming', 'stopping'}:
            engine.pending = None
            engine.enter('hold', now)

    def poll(self, state, engine, control_now, paused):
        if self.anchors is None:
            # Use the measured-state FK model already owned by the session when
            # possible. Before acquisition, the display initializes its markers.
            if engine is not None:
                self.anchor(tcp_poses(engine.core, state.q), self.message)
            else:
                self.anchors = {}
        delayed = paused or (state.timing or {}).get('recovery', {}).get('phase', 'RUNNING') != 'RUNNING'
        if delayed and engine is not None:
            self.suspend(engine)
        frame = self.viewer.targets.get()
        now = self.clock()
        self.ingest(frame, now)
        phase = 'readonly' if engine is None else engine.phase
        self.last_display = dict(q=state.q.tolist(), phase=phase, epoch=self.epoch,
                                     anchors=self.anchors, message=self.message,
                                     age_ms=1000 * (now - self.last_at) if self.frame else -1.,
                                     ik_ms=self.ik_ms, errors_mm=self.errors_mm, raw_errors_mm=self.raw_errors_mm,
                                     raw_rotation_deg=self.raw_rotation_deg)
        self.viewer.display.put(self.last_display)
        if self.viewer.quit.is_set() or not self.viewer.process.is_alive():
            error = self.viewer.errors.get()
            if error or (not self.viewer.process.is_alive() and self.viewer.process.exitcode not in (None, 0)):
                raise HardwareSafetyError(f'viewer failed: {error}')
            if engine is not None and engine.quit_requested:
                return None
            return ('quit', None)
        if self.viewer.stop.is_set():
            self.recovery_default_pending = False
            self.viewer.stop.clear()
            self._drain_commands()
            if engine is None:
                # Multiprocessing Queue has a feeder thread: a D event can arrive
                # after the drain. Invalidate that event before any acquisition.
                self.epoch += 1
                self.frame = self.targets = None
                self.stable_since = None
                self.message = 'Cancelled; press D to start preparation'
            return ('stop', None) if engine is not None and engine.phase != 'stopping' else None
        if self.recovery_default_pending and engine is None and self.fresh(now) and not delayed:
            self.recovery_default_pending = False
            return ('default', None)
        if (engine is not None and engine.phase in {'default', 'init', 'lift-retreat'}
                and not self.fresh(now) and not delayed):
            self.input_holds += 1
            return ('stop', None)
        try:
            command = self.viewer.commands.get_nowait()
        except queue.Empty:
            return None
        if (command['epoch'] != self.epoch or not 0 <= now - command['at'] <= .25
                or delayed or self.suspended):
            return None
        key = command['key']
        if engine is None and key != 'D':
            self.message = 'Press D first; no hardware has been acquired'
            return None
        return ({'D': 'default', 'I': 'init', 'T': 'stop' if phase in {'drag', 'drag-arming'} else 'drag'}[key], None)

    def _drain_commands(self):
        while True:
            try:
                self.viewer.commands.get_nowait()
            except queue.Empty:
                break

    def diagnostics(self, engine, command, targets):
        self.active_cycles += int(engine.phase == 'drag')
        self.ik_ms = command.diagnostics.solve_time_ms
        self.errors_mm = [1000 * command.diagnostics.left_position_error,
                          1000 * command.diagnostics.right_position_error]
        measured = {s: engine.core._measured_configuration.get_transform_frame_to_world(task.frame)
                    for s, task in engine.core.frame_tasks.items()}
        self.raw_errors_mm = [1000*np.linalg.norm(targets[s].translation-measured[s].translation)
                              for s in ('left', 'right')]
        self.raw_rotation_deg = [np.rad2deg(np.linalg.norm(pin.log3(measured[s].rotation.T @ targets[s].rotation)))
                                 for s in ('left', 'right')]
        return dict(raw_position_errors_mm=self.raw_errors_mm, raw_rotation_errors_deg=self.raw_rotation_deg,
                    limited_targets={s: p.homogeneous for s, p in engine.core.limited_targets().items()},
                    planner_failures=command.diagnostics.target_planner_failures,
                    planner_error=command.diagnostics.last_target_planner_error,
                    recovery_reason=command.diagnostics.recovery_reason,
                    epoch=self.epoch, sequence=self.sequence,
                    input_age_ms=(self.clock() - self.last_at) * 1000 if self.frame else None,
                    active=engine.phase == 'drag', requested_targets={s: p.homogeneous for s, p in targets.items()},
                    viewer_targets=(None if self.targets is None else
                                    {s: p.homogeneous for s, p in self.targets.items()}),
                    command_anchors=(None if self.command_anchors is None else
                                     {s: p.homogeneous for s, p in self.command_anchors.items()}),
                    base_anchor_gain=engine.core.config.base_anchor_gain,
                    posture_preference=engine.core.arm_posture_task.target_q,
                    target_sample=engine.target_sample, message=self.message,
                    position_errors_mm=self.errors_mm,
                    rotation_errors_rad=[command.diagnostics.left_orientation_error,
                                         command.diagnostics.right_orientation_error])

    def summary(self):
        return dict(active_cycles=self.active_cycles, input_holds=self.input_holds,
                    device_holds=self.device_holds, epochs=self.epoch,
                    max_input_age_ms=MAX_INPUT_AGE * 1000, manual_reengagement=True,
                    viewer_error=self.viewer.errors.get())

    def check_command(self, command, *, phase):
        # Persistent Cartesian error is a status, not necessarily an infeasible
        # QP. In interactive drag it still warrants pausing, before dispatch.
        if phase == 'drag' and command.diagnostics.status == 'large/unreachable target error':
            raise HardwareSafetyError('persistent large/unreachable target error')
        if phase == 'drag' and command.diagnostics.recovery_reason in {
                'target_trajectory_infeasible', 'boundary_constraints_infeasible'}:
            raise HardwareSafetyError(command.diagnostics.recovery_reason)


def stop_confirmed(result):
    gateway = result.get('gateway', {})
    observations = gateway.get('stop_observations', {})
    return (gateway.get('kind') == 'closed' and not gateway.get('stop_failures')
            and all(observations.get(n, {}).get('continuous_stop_confirmed') is True
                    for n in ('left', 'right', 'theta', 'base')))


def wait_fault_paused(source, result, *, live):
    """Keep the UI alive after stop/close; never publish or reconnect here.

    D requests a checked return to default, not resumption of the failed target.
    An unconfirmed physical stop cannot be acknowledged away with a keypress.
    """
    viewer = source.viewer
    can_retry = not live or stop_confirmed(result)
    reason = str(result.get('finish_reason', 'unknown control fault'))
    action = ('D: checked return to default from current pose; Q: quit'
              if can_retry else 'STOP UNCONFIRMED: use physical E-stop. Recovery locked; Q: quit')
    print(f'WARNING: motion paused: {reason}\n{action}', flush=True)
    source.epoch += 1
    source._drain_commands()
    source.frame = source.targets = source.command_anchors = None
    source.stable_since = None
    source.suspended = True
    snapshot = dict(source.last_display or dict(
        q=result.get('initial_q', [0.] * 18), ik_ms=0., errors_mm=[0., 0.]))
    snapshot.update(phase='fault-paused', epoch=source.epoch, anchors={},
                    # Keep full diagnostics in the report/terminal, not the
                    # bounded shared-memory viewer snapshot.
                    message=f'{reason[:600]}\n{action}', age_ms=-1., disconnected=True,
                    controls=action)
    source.last_display = snapshot
    while not viewer.quit.is_set() and viewer.process.is_alive():
        viewer.display.put(snapshot)
        if viewer.stop.is_set():
            viewer.stop.clear()
            source._drain_commands()
            source.epoch += 1
            snapshot['epoch'] = source.epoch
        try:
            command = viewer.commands.get_nowait()
        except queue.Empty:
            command = None
        if (command is not None and command.get('key') == 'D' and can_retry
                and command.get('epoch') == source.epoch
                and 0 <= source.clock() - command.get('at', -1.) <= .25):
            return True
        time.sleep(.02)
    return False


def run(args, *, viewer_factory=DragViewer):
    from .demo import ACCEPTED
    from .verify import save_json, timing_probe

    parser = argparse.ArgumentParser()
    session.add_arguments(parser)
    values = parser.parse_args(['--mode', args.mode, '--robot-config', str(args.robot_config),
                               '--run-seconds', str(args.run_seconds)])
    values.demo_authorized = args.mode == 'live'
    values.research_release = getattr(args, 'research_release', None)
    values.plant_calibration = getattr(args, 'plant_calibration', None)
    values.control_group = getattr(args, 'control_group', 'coupled')
    values.tracking_state = getattr(args, 'tracking_state', 'measured')
    mapping, profile, _, _ = session.load_configuration(values)
    # Validate the selected contract before connecting any device or acquiring.
    if (profile.profile_id not in SUPPORTED_PROFILES or mapping.feedback_policy != 'strict'
            or mapping.latency_policy != 'hold-resume' or profile.motion_envelope != 'none'):
        raise HardwareSafetyError('drag requires its dedicated runtime with strict feedback and hold-resume')
    from .research_profiles import describe, require_offline_validation
    print(describe(profile, control_group=values.control_group), flush=True)
    print(f'IK tracking state: {values.tracking_state}', flush=True)
    if args.mode == 'live':
        require_offline_validation(profile, mapping=mapping)
    root = args.save_trace.resolve()
    root.mkdir(parents=True, exist_ok=False)
    viewer = viewer_factory(mapping.data['fixed_gripper_positions'], args.mode,
                            robot_model=profile.robot_model, collision_enabled=profile.self_collision_enabled)
    lock = None
    ledger = dict(suite='drag-trial', mode=args.mode, hardware_qualified=False)
    try:
        # Fail GUI initialization before starting the hardware audit.
        viewer.start()
        source = DragInput(viewer)
        source.retain_simulation = True
        if args.mode == 'live':
            (PROJECT_ROOT / 'runs').mkdir(exist_ok=True)
            lock = (PROJECT_ROOT / 'runs/.hexfellow-onsite.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise HardwareSafetyError('Another verification/control process is running') from error
            print('Independent drag research trial; incomplete firmware protection, disconnect, and startup checks remain within the existing research acceptance scope.', flush=True)
            answer = 'y' if getattr(args, 'confirm_onsite_ready', False) else input('Confirm empty grippers, mechanical zero, clearance/cables, charger disconnected, and accessible E-stop; enter y to begin the 30-second read-only check:')
            if answer.strip().lower() != 'y':
                ledger.update(passed=False, finish_reason='operator_cancelled', motion_commands_sent=0)
                save_json(root / 'drag-trial.json', ledger)
                return ledger
            values.confirm_onsite_ready = True
            values.allow_uncommissioned_profile_sha256 = profile.sha256
            mapping.authorize(profile, 'whole-body', profile.sha256)
            binding = software_binding(mapping)
            ledger.update(binding=binding, experimental_checks_accepted=list(ACCEPTED))
            audit_dir = root / 'audit'
            audit = timing_probe(mapping, profile, audit_dir, 30, sdk_sink=True,
                                 allow_lift_retreat=profile.profile_id in HIGHSPEED_PROFILES)
            ledger.update(audit_passed=audit_passed(audit), audit_sha256=digest(audit_dir / 'summary.json'))
            save_json(root / 'drag-trial.json', ledger)
            if not ledger['audit_passed']:
                raise HardwareSafetyError(f'Read-only check failed；{audit_dir / "summary.json"}')
            source.binding = binding
            values.trial_binding = binding
            values.demo_device_binding = device_binding(SimpleNamespace(health=audit['initial_health']))
        attempt, total_commands = 0, 0
        ledger['attempts'] = []
        normal_exits = {'controlled_session_exit', 'readonly_exit', 'operator_quit_during_delay_hold'}
        while True:
            values.save_trace = root / ('session' if attempt == 0 else f'session-{attempt:03d}')
            print('Window controls: D -> wait for completion -> I -> wait for READY -> T to drag; Space to stop, Q to quit.', flush=True)
            result = session.run(values, input_source=source)
            total_commands += result.get('motion_commands_sent', 0)
            ledger['attempts'].append(dict(report=str(values.save_trace.relative_to(root) / 'summary.json'),
                passed=result['passed'], finish_reason=result.get('finish_reason'),
                motion_commands_sent=result.get('motion_commands_sent', 0),
                stop_confirmed=stop_confirmed(result) if args.mode == 'live' else None))
            ledger.update(motion_commands_sent=total_commands, report=ledger['attempts'][-1]['report'])
            save_json(root / 'drag-trial.json', ledger)
            faulted = (result.get('finish_reason') not in normal_exits
                       or bool(result.get('gateway', {}).get('stop_failures'))
                       or bool(result.get('gateway', {}).get('fault')))
            if not faulted or not wait_fault_paused(source, result, live=args.mode == 'live'):
                break

            # D authorizes a new checked default movement, never the failed
            # target. Preserve simulation physics instead of rebuilding at home.
            epoch = source.epoch + 1
            snapshot = dict(source.last_display)
            retained_plant = source.simulation_plant
            source = DragInput(viewer)
            source.retain_simulation = True
            source.simulation_plant = retained_plant
            source.recovery_default_pending = True
            source.epoch = epoch
            source._drain_commands()
            # Preserve Space pressed after D, including during the audit.
            # The next poll consumes it and cancels pending preparation.
            snapshot.update(phase='rechecking', epoch=epoch,
                            message='Default recovery requested; checking before movement', controls='Space: cancel | Q: quit')
            source.last_display = snapshot
            viewer.display.put(snapshot)
            attempt += 1
            if args.mode == 'live':
                try:
                    fresh_mapping, fresh_profile, _, _ = session.load_configuration(values)
                    if fresh_mapping.sha256 != mapping.sha256 or fresh_profile.sha256 != profile.sha256:
                        raise HardwareSafetyError('configuration changed; restart after review')
                    require_offline_validation(profile, mapping=mapping)
                    mapping.authorize(profile, 'whole-body', profile.sha256)
                    binding = software_binding(mapping)
                    audit_dir = root / f'audit-{attempt:03d}'
                    audit = timing_probe(mapping, profile, audit_dir, 30, sdk_sink=True,
                                         allow_lift_retreat=profile.profile_id in HIGHSPEED_PROFILES)
                    if not audit_passed(audit):
                        raise HardwareSafetyError(f'Read-only recovery check failed; {audit_dir / "summary.json"}')
                    source.binding = values.trial_binding = binding
                    values.demo_device_binding = device_binding(SimpleNamespace(health=audit['initial_health']))
                    ledger.update(binding=binding, audit_sha256=digest(audit_dir / 'summary.json'))
                except Exception as error:
                    # No successful recovery audit: retain the UI, but no live
                    # session can be opened by another keypress in this process.
                    result = dict(passed=False, finish_reason=f'{type(error).__name__}: {error}',
                                  motion_commands_sent=0)
                    ledger['recovery_error'] = result['finish_reason']
                    save_json(root / 'drag-trial.json', ledger)
                    wait_fault_paused(source, result, live=True)
                    break
            source.epoch += 1  # Ignore keys/targets generated during the audit.
            source._drain_commands()
            if viewer.quit.is_set() or not viewer.process.is_alive():
                break

        ledger.update(session_passed=result['passed'],
                      motion_commands_sent=total_commands)
        ledger['report'] = str(values.save_trace.relative_to(root) / 'summary.json')
        if args.mode == 'live':
            ledger['stop_confirmed'] = stop_confirmed(result)
            ledger['machine_passed'] = bool(result['passed'] and ledger['stop_confirmed']
                                            and result.get('trace_complete') is True
                                            and source.active_cycles > 0 and not source.summary()['viewer_error'])
            ledger['operator_normal'] = False
            save_json(root / 'drag-trial.json', ledger)
            if ledger['machine_passed']:
                answer = input('Session stopped and released; confirm normal physical motion and stopping, then enter y:')
                ledger['operator_normal'] = answer.strip().lower() == 'y'
            result['passed'] = ledger['machine_passed'] and ledger['operator_normal']
        save_json(root / 'drag-trial.json', ledger)
        print(f'Report:{root}', flush=True)
        return result
    except BaseException as error:
        ledger['error'] = f'{type(error).__name__}: {error}'
        save_json(root / 'drag-trial.json', ledger)
        raise
    finally:
        if 'source' in locals() and source.simulation_plant is not None:
            source.simulation_plant.close()
        viewer.close()
        if lock is not None:
            lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sim', 'live'), default='sim')
    parser.add_argument('--control-group', choices=('coupled', 'arms'), default='coupled')
    parser.add_argument('--tracking-state', choices=('measured', 'reference'), default='measured')
    parser.add_argument("--research-release", type=Path, help="local research binding for GP80 live control")
    parser.add_argument('--robot-config', type=Path, default=RUNTIME)
    parser.add_argument('--run-seconds', type=float, default=3600.)
    parser.add_argument('--save-trace', type=Path)
    args = parser.parse_args(argv)
    if not np.isfinite(args.run_seconds) or args.run_seconds <= 0:
        parser.error('--run-seconds must be positive and finite')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    args.save_trace = args.save_trace or PROJECT_ROOT / 'runs' / f'hexfellow-drag-{tag}'
    try:
        return 0 if run(args)['passed'] else 1
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        print(f'Drag session failed:{type(error).__name__}: {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
