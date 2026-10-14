"""Independent SDK publisher/watchdog process, using bounded nonblocking IPC."""

from __future__ import annotations

import dataclasses
import gc
import multiprocessing as mp
import os
import queue
import signal
import time
import threading
from collections import deque
from contextlib import ExitStack

from .control_timing import COHERENCE_WAIT, feedback_deadline, select_control_state
from .recovery import DelayRecovery, ResumeRequest
from .state_mailbox import CommandMailbox, LatestStateMailbox, MailboxBusy
from .whole_body import COMMAND_WATCHDOG, CommandGuard
from .x7 import ControlDelay, FeedbackSkewSafetyError, FeedbackUnavailable, HardwareSafetyError


def offer_latest(mailbox, value):
    """Best effort, bounded; never block the safety loop on a reader/feeder."""
    try:
        mailbox.put_nowait(value)
        return True
    except queue.Full:
        if isinstance(mailbox, LatestStateMailbox):
            return False  # Already overwrites; never consume a state to resolve lock contention.
        try:
            mailbox.get_nowait()
        except queue.Empty:
            return False
        try:
            mailbox.put_nowait(value)
            return True
        except queue.Full:
            return False


def latest(mailbox, previous=None):
    # Bound work even if a broken producer floods a test double/unbounded queue.
    for _ in range(4):
        try:
            previous = mailbox.get_nowait()
        except queue.Empty:
            break
    return previous


class GatewayShutdownRequested(Exception):
    """Internal normal-shutdown signal; endpoint.close still confirms stopping."""


class GatewaySupervisor:
    """Deterministic per-cycle logic, also exercised with fake hardware."""

    def __init__(self, endpoint, profile, mapping, stage, effort_limits, clock=time.monotonic, on_fault=None, stationary_reference=False, sleep=None):
        self.sleep = sleep or time.sleep
        self.stationary_reference = stationary_reference
        self.on_fault = on_fault
        self.command_timing = deque(maxlen=512)
        mapping.validate_experiment_options()
        if profile.motion_envelope != mapping.motion_envelope:
            raise HardwareSafetyError("publisher motion envelope mismatch")
        self.endpoint, self.profile, self.mapping, self.stage = endpoint, profile, mapping, stage
        self.effort_limits, self.clock = effort_limits, clock
        self.guard = None
        self.last_command_at = None
        self.fault = None
        self.stop_failures = []
        self.last_state = None
        self.fault_observed_at = None
        self.startup_timing = {}
        self.prepared_guard = None
        self.coherence_events = deque(maxlen=64)
        self.fault_window = None
        self.first_feedback_anomaly = None
        self.observed_valid_feedback = False
        self.recovery = DelayRecovery(clock) if mapping.latency_policy == "hold-resume" else None
        from .feedback_monitor import FeedbackObservations
        self.feedback_observations = FeedbackObservations()

    def coherent_state(self, state, command, on_state):
        deadline = None
        event = None
        while True:
            now = self.clock()
            if getattr(self.endpoint, "cancel_requested", lambda: False)():
                raise GatewayShutdownRequested()
            if now - self.last_command_at > COMMAND_WATCHDOG:
                raise ControlDelay("IK process command watchdog expired")
            if command is not None and now >= command.expires:
                raise ControlDelay("command expired while checking feedback")
            if deadline is not None and now >= deadline:
                raise FeedbackSkewSafetyError("feedback coherence recovery missed its deadline")
            try:
                self.endpoint.check_health(state, self.stage)
                if event is not None:
                    event.update(recovered=True, ended=self.clock())
                return state
            except FeedbackSkewSafetyError:
                if deadline is None:
                    deadline = min(now + COHERENCE_WAIT, feedback_deadline(state),
                                   self.last_command_at + COMMAND_WATCHDOG,
                                   command.expires if command is not None else float('inf'))
                    event = dict(started=now, deadline=deadline, recovered=False, received=state.received)
                    self.coherence_events.append(event)
                if now >= deadline:
                    raise FeedbackSkewSafetyError("robot feedback skew exceeds 20 ms after bounded coherence wait")
                if on_state is not None:
                    on_state(state)
                self.sleep(min(.0002, deadline - now))
                state = self.endpoint.state()
                self.last_state = state
            except ControlDelay as error:
                from ..control.feedback_policy import FeedbackAgeError, THETA_BOUNDED
                from .control_timing import MAX_WAIT

                # Only age delays can wait here. Device/transport faults retain
                # their existing stop behavior; no stale command is published.
                if ((self.mapping.latency_policy != "hold-resume" and self.mapping.feedback_policy != THETA_BOUNDED)
                        or not isinstance(error.__cause__, FeedbackAgeError)):
                    raise
                if deadline is None:
                    deadline = min(now + MAX_WAIT, self.last_command_at + COMMAND_WATCHDOG,
                                   command.expires if command is not None else float('inf'))
                    event = dict(started=now, deadline=deadline, recovered=False,
                                 reason="feedback_age", received=state.received)
                    self.coherence_events.append(event)
                if now >= deadline:
                    raise ControlDelay("feedback age recovery missed its deadline") from error
                self.sleep(min(.0002, deadline - now))
                state = self.endpoint.state()
                self.last_state = state

    def activate(self):
        if self.fault or self.guard is not None:
            raise HardwareSafetyError("gateway cannot be rearmed within a session")
        try:
            if self.prepared_guard is None:
                self.startup_timing["preparation_started"] = self.clock()
                self.prepared_guard = CommandGuard.prepare(self.profile, self.mapping, self.stage, self.effort_limits)
                self.startup_timing["preparation_completed"] = self.clock()
            prepared = self.prepared_guard
            state = self.endpoint.acquire(self.stage, self.profile)
            self.startup_timing["acquisition_completed"] = self.clock()
            self.last_state = state
            self.guard = CommandGuard(self.profile, self.mapping, self.stage, state, self.effort_limits,
                                      prepared=prepared, stationary_reference=self.stationary_reference)
            self.startup_timing["guard_bound"] = self.clock()
            start_feedback = getattr(self.endpoint, "start_feedback_worker", None)
            if start_feedback is not None:
                start_feedback(state)
            self.last_command_at = self.clock()
            return state
        except BaseException as error:
            self.fail(f"acquisition failed: {type(error).__name__}: {error}")
            raise

    def fail(self, reason):
        if self.fault is None:
            self.fault = str(reason)
            self.fault_observed_at = self.clock()
            if self.on_fault is not None:
                try:
                    self.on_fault(self.fault)
                except Exception:
                    pass  # A broken diagnostic channel must never prevent stopping.
            try:
                self.stop_failures = self.endpoint.stop()
            except Exception as error:
                self.stop_failures = [str(error)]
            try:
                self.fault_window = self.endpoint.diagnostic_window()
            except Exception:
                pass

    def tick(self, command=None, on_state=None):
        if self.fault:
            raise HardwareSafetyError(self.fault)
        try:
            row = None
            if command is not None and not isinstance(command, ResumeRequest):
                row = {"sequence": command.sequence, "created": command.created,
                       "enqueued_at": command.enqueued_at, "dequeued_at": self.clock()}
                self.command_timing.append(row)
            state = self.endpoint.state()
            if row is not None:
                row["snapshot_completed_at"] = self.clock()
                row.update({k: v for k, v in (state.timing or {}).items()
                            if k.startswith("snapshot_")})
            self.last_state = state
            if hasattr(self, 'feedback_observations'):
                observations = self.feedback_observations.observe(
                    state, self.clock(), self.endpoint if self.guard is None else None)
                state = dataclasses.replace(state, timing={**(state.timing or {}),
                    'feedback_observations': observations})
            if getattr(self, "recovery", None) is not None and self.guard is not None:
                return self.recovery_tick(state, command, on_state)
            if self.guard is None:
                try:
                    state.validate(self.clock())
                    self.observed_valid_feedback = True
                except HardwareSafetyError as error:
                    if self.observed_valid_feedback and self.first_feedback_anomaly is None:
                        self.first_feedback_anomaly = {"at": self.clock(), "error": str(error),
                                                       "received": state.received}
                        try:
                            self.first_feedback_anomaly['window'] = self.endpoint.diagnostic_window()
                        except Exception as diagnostic_error:
                            self.first_feedback_anomaly['window_error'] = str(diagnostic_error)
            if self.guard is not None:
                now = self.clock()
                state = self.coherent_state(state, command, on_state)
                if now - self.last_command_at > COMMAND_WATCHDOG:
                    raise HardwareSafetyError("IK process command watchdog expired")
                if on_state is not None:
                    on_state(state)
                if command is not None:
                    self.startup_timing.setdefault("first_command_received", now)
                    self.guard.validate(command, state, self.clock())
                    row["validation_completed_at"] = self.clock()
                    if getattr(self.endpoint, "cancel_requested", lambda: False)():
                        raise GatewayShutdownRequested()
                    if hasattr(self.endpoint, "publish_validated"):
                        self.endpoint.publish_validated(command, self.stage, state)
                    else:
                        self.endpoint.publish(command, self.stage)
                    self.last_command_at = self.clock()
                    row["publish_returned_at"] = self.last_command_at
            elif command is not None:
                raise HardwareSafetyError("motion command before explicit acquisition")
            if self.guard is None and on_state is not None:
                on_state(state)
            return state
        except GatewayShutdownRequested:
            raise  # The process finally block closes/brakes; this is not a fault.
        except ControlDelay as error:
            if getattr(self, "recovery", None) is None or self.guard is None:
                self.fail(error)
                raise
            self.enter_delay(error)
            if self.last_state is not None:
                return self.deliver_recovery(self.last_state, on_state)
        except Exception as error:
            self.fail(error)
            raise

    def enter_delay(self, reason):
        if self.recovery.enter(reason):
            self.endpoint.delay_hold(reason, self.recovery.epoch)

    def deliver_recovery(self, state, on_state):
        timing = {**(state.timing or {}), "recovery": self.recovery.report()}
        if self.recovery.seed is not None:
            timing["recovery_seed"] = self.recovery.seed
        state = dataclasses.replace(state, timing=timing)
        if on_state is not None:
            on_state(state)
        return state

    def recovery_tick(self, state, command, on_state):
        # Hard health faults are checked even while clocks are too old for motion.
        self.endpoint.check_health(state, self.stage, check_timing=False)
        for gate in self.endpoint.direct_gates.values():
            if gate.delay_reason:
                self.enter_delay(gate.delay_reason)
        recovery = self.recovery
        if recovery.phase != "RUNNING":
            self.endpoint.maintain_delay_hold()
            ready = recovery.observe(state, drained=self.endpoint.delay_drained(recovery.epoch))
            if isinstance(command, ResumeRequest) and command.epoch == recovery.epoch and ready:
                self.guard = CommandGuard(self.profile, self.mapping, self.stage, state, self.effort_limits,
                                          prepared=self.prepared_guard, stationary_reference=True)
                self.endpoint.resume_delay(recovery.epoch)
                self.last_command_at = self.clock()
                recovery.resume(dataclasses.replace(state, timing=None))
            return self.deliver_recovery(state, on_state)
        if isinstance(command, ResumeRequest):
            return self.deliver_recovery(state, on_state)
        if command is not None and command.recovery_epoch != recovery.epoch:
            return self.deliver_recovery(state, on_state)  # Stale parent work, not a new motion source.
        state = self.coherent_state(state, command, None)
        if command is not None:
            self.guard.validate(command, state, self.clock())
            if getattr(self.endpoint, "cancel_requested", lambda: False)():
                raise GatewayShutdownRequested()
            self.endpoint.publish_validated(command, self.stage, state)
            self.last_command_at = self.clock()
            recovery.seed = None
        return self.deliver_recovery(state, on_state)


def _gateway_main(mapping, profile, stage, mode, effort, state_queue, command_queue,
                  status_queue, activate_event, stop_event, transport_options=None, stationary_reference=False):
    from .hexfellow_control_console import _load_vendor_sdk, _suppress_interactive_idle_logs
    from .hexfellow_endpoint import HexWholeBodyEndpoint
    from .x7_joint_console import _suppress_known_hands_timeout_log

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Parent requests stop via independent event.
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    endpoint = None
    supervisor = None
    recorder = None
    report = {"kind": "closed", "motion_commands_sent": 0, "stop_failures": []}
    logging_context = ExitStack()
    try:
        from .realtime import control_cpu, control_priority
        report["cpu_scheduling"] = {**logging_context.enter_context(control_cpu('gateway')),
                                    **logging_context.enter_context(control_priority())}
        _load_vendor_sdk()  # initialize SDK logging before installing instance filters
        from .realtime import sdk_thread_fairness
        logging_context.enter_context(sdk_thread_fairness(.0002))
        logging_context.enter_context(_suppress_known_hands_timeout_log())
        logging_context.enter_context(_suppress_interactive_idle_logs())
        prepared_guard = None
        preparation_timing = {}
        if mode == "live":
            preparation_timing["preparation_started"] = time.monotonic()
            prepared_guard = CommandGuard.prepare(profile, mapping, stage, effort)
            preparation_timing["preparation_completed"] = time.monotonic()
        responsive = os.environ.get('POLICY_SIM_RESPONSIVE_CPUS')
        sdk_scope = ExitStack()
        previous_threads = {thread.native_id for thread in threading.enumerate()}
        if responsive:
            import json
            from .responsive_scheduling import thread_cpu, promote_new_threads
            pools = json.loads(responsive)
            report['cpu_scheduling']['gateway_threads'] = pools
            sdk_scope.enter_context(thread_cpu(pools['sdk']))
        with sdk_scope:
            if transport_options:
                if mode not in {"probe", "shadow"}:
                    raise ValueError("transport diagnostics are read-only")
                if transport_options.get("diagnostic_path"):
                    from pathlib import Path

                    from .transport_diagnostics import TransportRecorder, record_sdk_identity
                    recorder = TransportRecorder(Path(transport_options["diagnostic_path"]))
                    record_sdk_identity(recorder)
                endpoint = HexWholeBodyEndpoint(mapping, read_only=True,
                                                transport_options=transport_options, recorder=recorder)
            else:
                endpoint = HexWholeBodyEndpoint(mapping, read_only=mode != "live")
        if responsive:
            # All broker work happens before ownership/initialization. No D-Bus
            # call is introduced into a live control/receive path.
            report['cpu_scheduling']['sdk_threads'] = promote_new_threads(previous_threads)
            logging_context.enter_context(thread_cpu(pools['owner']))
            from functools import partial
            endpoint.feedback_worker_context = partial(thread_cpu, pools['sdk'])
        def notify_fault(reason):
            from .shutdown import compact_state
            try:
                status_queue.put_nowait({"kind": "fault", "detail": str(reason),
                    "fault_state": compact_state(supervisor.last_state),
                    "fault_observed_at": supervisor.fault_observed_at})
            except queue.Full:
                pass

        def shutdown_progress(phase, event, device=None, error=None):
            row = dict(phase=phase, event=event, device=device, error=error, at=time.monotonic())
            report.setdefault('shutdown_stages', []).append(row)
            try:
                status_queue.put_nowait(dict(kind='shutdown_progress', **row))
            except queue.Full:
                pass

        endpoint.shutdown_progress = shutdown_progress
        endpoint.estop_armed_at = lambda: (getattr(command_queue, 'estop_armed_at', None).value or None
            if getattr(command_queue, 'estop_armed_at', None) is not None else None)

        def deliver_state(state):
            state = dataclasses.replace(state, timing={**(state.timing or {}), "offered_at": time.monotonic()})
            offer_latest(state_queue, state)

        supervisor = GatewaySupervisor(endpoint, profile, mapping, stage, effort, on_fault=notify_fault,
                                       stationary_reference=stationary_reference)
        supervisor.prepared_guard = prepared_guard
        supervisor.startup_timing.update(preparation_timing)
        endpoint.cancel_requested = stop_event.is_set
        wake = getattr(command_queue, "wake", None)
        if wake is not None:
            endpoint.feedback_notify = wake.set
        if wake is not None:
            for tap in getattr(endpoint, "taps", {}).values():
                tap.on_update = wake.set
        gc.collect()
        gc.disable()  # Avoid stopping all four SDK receive threads for cyclic GC.
        status_queue.put_nowait({"kind": "connected"})
        next_snapshot = 0.0
        feedback_woke = False
        while not stop_event.is_set():
            if recorder is not None and (recorder.error or recorder.dropped):
                raise HardwareSafetyError('diagnostic recording incomplete; stopping read-only collection')
            if mode == "live" and activate_event.is_set() and supervisor.guard is None:
                status_queue.put_nowait({"kind": "initializing"})
                state = supervisor.activate()
                deliver_state(state)
                status_queue.put_nowait({"kind": "acquired", "state": state})
            if wake is not None:
                wake.clear()  # Clear BEFORE draining, so publication cannot lose a wake.
            try:
                command = command_queue.get_nowait()  # Exactly one reference; never coalesce.
            except queue.Empty:
                command = None
            now = time.monotonic()
            if supervisor.recovery is not None and supervisor.guard is not None and command_queue.hold_requested.is_set():
                command_queue.hold_requested.clear()
                supervisor.enter_delay("parent requested delay hold")
                command = None  # Hold/cancel wins over a concurrently queued resume or motion.
            if command is not None or feedback_woke or now >= next_snapshot:
                supervisor.tick(command, on_state=deliver_state)
                next_snapshot = time.monotonic() + .004
            elif hasattr(endpoint, "check_transport"):
                endpoint.check_transport()
                if supervisor.guard is not None and now - supervisor.last_command_at > COMMAND_WATCHDOG:
                    if supervisor.recovery is not None:
                        supervisor.enter_delay("IK process command watchdog expired")
                    else:
                        raise HardwareSafetyError("IK process command watchdog expired")
            if wake is not None:
                feedback_woke = wake.wait(max(0., next_snapshot - time.monotonic()))  # Timeout is only the health/watchdog fallback.
            else:
                stop_event.wait(.004)
    except GatewayShutdownRequested:
        pass  # Explicit parent shutdown; retain normal close and stop verification.
    except BaseException as error:
        report["fault"] = f"{type(error).__name__}: {error}"
        if getattr(error, "feedback_failure", None) is not None:
            report["feedback_failure"] = error.feedback_failure
        report["fault_observed_at"] = (supervisor.fault_observed_at if supervisor is not None
                                       and supervisor.fault_observed_at is not None else time.monotonic())
        if supervisor is not None:
            if supervisor.recovery is not None:
                report["recovery"] = supervisor.recovery.report()
            report["fault_state"] = supervisor.last_state
        if supervisor is not None:
            supervisor.fail(report["fault"])
        try:
            status_queue.put_nowait({"kind": "fault", "detail": report["fault"]})
        except queue.Full:
            pass
    finally:
        if endpoint is not None:
            try:
                report["feedback_window"] = (supervisor.fault_window if supervisor is not None
                    and supervisor.fault_window is not None else endpoint.diagnostic_window())
            except Exception as error:
                report["feedback_window_error"] = str(error)
            report["motion_commands_sent"] = endpoint.write_count
            report["blocked_control_messages"] = sum(getattr(api, "blocked_control_messages", 0)
                                                     for api in endpoint.apis)
            report["state_overwrites"] = state_queue.overwrites.value
            report['sdk_reconnects'] = sum(getattr(api, 'lifecycle_reconnects', 0) for api in endpoint.apis)
            try:
                report["stop_failures"] = endpoint.close()
            except Exception as error:
                report["stop_failures"] = [f'endpoint close: {error}']
                report['kind'] = 'unconfirmed'
            report['feedback_worker'] = getattr(endpoint, 'feedback_worker_report', None)
            report['feedback_gap_state'] = {name: tap.gap_state.report()
                for name, tap in getattr(endpoint, 'taps', {}).items() if getattr(tap, 'gap_state', None) is not None}
            try:
                report["stop_window"] = getattr(endpoint, "stop_window", {})
                if getattr(endpoint, "stop_window_error", None):
                    report["stop_window_error"] = endpoint.stop_window_error
            except Exception as error:
                report["stop_window_error"] = str(error)
            report["stop_observations"] = getattr(endpoint, "stop_observations", {})
            report["stop_feedback_errors"] = getattr(endpoint, "stop_feedback_errors", [])
            report["feedback_latency_all_frames"] = {name: tap.latencies.report()
                for name, tap in getattr(endpoint, "taps", {}).items() if hasattr(tap, "latencies")}
            report["initialization_checkpoints"] = getattr(endpoint, "initialization_checkpoints", [])
            if getattr(endpoint, "feedback_failure", None) is not None:
                report["feedback_failure"] = endpoint.feedback_failure
            if getattr(endpoint, "direct_gates", {}):
                report["direct_downlink"] = {
                    name: {"phase": gate.phase, "fault": gate.fault, "blocked": gate.blocked,
                           "counts": getattr(gate, "counts", {}),
                           "latency_all_cycles": gate.latencies.report() if hasattr(gate, "latencies") else {},
                           "startup_timing": getattr(gate, "startup_timing", {}),
                           "first_dispatch_timing": list(getattr(gate, "first_dispatch_timing", [])),
                           "last_source_sequence": getattr(gate, "sequence", None)}
                    for name, gate in endpoint.direct_gates.items()}
        if supervisor is not None:
            report["stop_failures"].extend(supervisor.stop_failures)
            if supervisor.recovery is not None:
                report["recovery"] = supervisor.recovery.report()
            report["startup_timing"] = supervisor.startup_timing
            report["command_timing"] = list(supervisor.command_timing)
            report["coherence_events"] = list(supervisor.coherence_events)
            report["first_feedback_anomaly"] = supervisor.first_feedback_anomaly
            report["feedback_observations"] = supervisor.feedback_observations.report()
        if recorder is not None:
            report["transport_diagnostics"] = recorder.close()
        from .shutdown import export_diagnostics
        if endpoint is not None and hasattr(endpoint, 'shutdown_progress'):
            endpoint.shutdown_progress('diagnostics', 'started')
        export_diagnostics(report)
        if endpoint is not None and hasattr(endpoint, 'shutdown_progress'):
            endpoint.shutdown_progress('diagnostics', 'completed' if report.get('diagnostics', {}).get('complete', True) else 'failed')
        if report.get("fault_state") is not None or report.get("feedback_failure") is not None:
            from .ik_runner import jsonable

            for key in ("fault_state", "feedback_failure"):
                if report.get(key) is not None:
                    report[key] = jsonable(report[key])
        try:
            status_queue.put(report, timeout=1.)
        except queue.Full:
            pass
        # A stalled/dead parent must not trap SDK cleanup in Queue feeder joins.
        for mailbox in (state_queue, command_queue):
            mailbox.cancel_join_thread()
        # Flush the final small report after all SDKs have closed. The parent
        # drains status while joining; cancelling this feeder loses the report.
        status_queue.close()
        status_queue.join_thread()
        logging_context.close()
        gc.enable()


class HardwareGateway:
    def __init__(self, mapping, profile, stage, mode, effort_limits, transport_options=None, *, stationary_reference=False):
        mapping.validate_experiment_options()
        if profile.motion_envelope != mapping.motion_envelope:
            raise HardwareSafetyError("gateway motion envelope mismatch")
        self.stage = stage
        self.latency_policy = mapping.latency_policy
        context = mp.get_context("spawn")
        self.state_queue = LatestStateMailbox(context)
        self.command_queue = CommandMailbox(context)
        self.status_queue = context.Queue(maxsize=8)
        self.activate_event, self.stop_event = context.Event(), context.Event()
        self.process = context.Process(
            target=_gateway_main,
            args=(mapping, profile, stage, mode, effort_limits, self.state_queue, self.command_queue,
                  self.status_queue, self.activate_event, self.stop_event, transport_options, stationary_reference),
            name="hexfellow-sdk-watchdog",
        )
        self.last_state = None
        self.last_report = None
        self.fault_evidence = None
        self.shutdown_stages = []
        self.evidence_callback = None
        self.parent_timing = deque(maxlen=512)
        self.acquired_state = None
        self.initialization_started = False

    def start(self):
        self.process.start()

    def poll_status(self, *, raise_fault=True):
        for _ in range(8):
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if status["kind"] == "acquired":
                self.acquired_state = status["state"]
            if status["kind"] == "initializing":
                self.initialization_started = True
            if status["kind"] == "closed":
                self.last_report = status
            if status["kind"] == "unconfirmed":
                self.last_report = status
            if status["kind"] == "shutdown_progress":
                self.shutdown_stages.append(status)
            if status["kind"] == "fault" and status.get('fault_state') is not None:
                self.fault_evidence = status
                callback = getattr(self, 'evidence_callback', None)
                if callback is not None:
                    try:
                        callback(status)
                    except Exception as error:
                        self.evidence_error = str(error)
            if status["kind"] == "fault" and raise_fault:
                raise HardwareSafetyError(status["detail"])

    def state(self, *, fresh=True, deadline=None):
        self.poll_status()
        if self.process.exitcode is not None:
            raise HardwareSafetyError(f"hardware gateway exited: {self.last_report}")
        mailbox_timing = {"cache_reused": False}
        if hasattr(self.state_queue, "get_latest"):
            try:
                timeout = .0005 if deadline is None else min(.0005, max(0., deadline - time.monotonic()))
                self.last_state, metadata = self.state_queue.get_latest(timeout=timeout)
                mailbox_timing.update(metadata)
            except MailboxBusy as error:
                raise FeedbackUnavailable("feedback mailbox lock busy") from error
            except queue.Empty:
                mailbox_timing["cache_reused"] = True
        else:
            self.last_state = latest(self.state_queue, self.last_state)
        if self.last_state is None:
            raise FeedbackUnavailable("waiting for hardware telemetry")
        if fresh:
            self.last_state.validate(time.monotonic())
        timing = {**(self.last_state.timing or {}), **mailbox_timing, "consumed_at": time.monotonic()}
        self.parent_timing.append({**timing, "received": self.last_state.received})
        return dataclasses.replace(self.last_state, timing=timing)

    def control_state(self, slot_deadline):
        return select_control_state(self.state, slot_deadline)

    def activate(self):
        self.activate_event.set()
        deadline = time.monotonic() + 8.0
        while self.acquired_state is None:
            self.poll_status()
            if self.process.exitcode is not None or time.monotonic() > deadline:
                raise HardwareSafetyError("gateway acquisition failed/timed out")
            time.sleep(0.002)
        return self.acquired_state

    def publish(self, command):
        # Dropping a reference breaks its derivative history. Unlike pose input,
        # a full command mailbox is a stop condition, not a catch-up opportunity.
        try:
            now = time.monotonic()
            if not command.created <= now < command.expires:
                if getattr(self, "latency_policy", "strict") == "hold-resume" and command.created <= now:
                    self.request_hold()
                    raise ControlDelay("command expired before enqueue")
                self.stop_event.set()
                raise HardwareSafetyError("command expired before enqueue")
            self.command_queue.put_nowait(dataclasses.replace(command, enqueued_at=now))
        except queue.Full as error:
            if getattr(self, "latency_policy", "strict") == "hold-resume":
                self.request_hold()
                raise ControlDelay("command mailbox full") from error
            self.stop_event.set()
            raise HardwareSafetyError("command mailbox full; stopping instead of skipping references") from error

    def request_hold(self):
        self.command_queue.hold_requested.set()
        self.command_queue.wake.set()

    def request_resume(self, epoch):
        try:
            self.command_queue.put_nowait(ResumeRequest(epoch))
        except queue.Full:
            return False
        return True

    def await_dispatch(self, sequence, deadline):
        """Finish the last qualifying hold sample before requesting normal stop."""
        from .whole_body import STAGES
        coherence_deadline = deadline
        while time.monotonic() < min(deadline, coherence_deadline):
            try:
                state = self.state(deadline=coherence_deadline)
            except FeedbackUnavailable:
                if time.monotonic() >= coherence_deadline:
                    raise
                time.sleep(.0002)
                continue
            except FeedbackSkewSafetyError:
                coherence_deadline = min(coherence_deadline, time.monotonic() + COHERENCE_WAIT)
                if time.monotonic() >= coherence_deadline:
                    raise
                time.sleep(.0002)
                continue
            wire = (state.dispatch or {}).get("wire", {})
            if time.monotonic() >= coherence_deadline:
                break
            if all(wire.get(n) and wire[n].get("sequence", -1) >= sequence for n in STAGES[self.stage]):
                return
            time.sleep(.0002)
        raise HardwareSafetyError("final hold command was not dispatched before its slot deadline")

    def close(self):
        from .shutdown import SHUTDOWN_TIMEOUT
        if self.process.pid is None:
            return {"kind": "not_started", "motion_commands_sent": 0}
        self.stop_event.set()
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        printed = None
        while self.process.is_alive() and time.monotonic() < deadline:
            self.poll_status(raise_fault=False)
            if self.shutdown_stages:
                row = self.shutdown_stages[-1]
                label = (row['phase'], row.get('device'), row['event'])
                if label != printed:
                    print(f"Shutdown progress:{label[0]} {label[1] or ''} {label[2]}", flush=True)
                    printed = label
            self.process.join(timeout=0.02)
        self.poll_status(raise_fault=False)
        if self.process.is_alive():
            # Never SIGKILL the only surviving brake publisher automatically.
            return {"kind": "unconfirmed", "stop_failures": ["SDK gateway did not close; physical E-stop required"],
                    "fault_state": (self.fault_evidence or {}).get('fault_state'),
                    "fault": (self.fault_evidence or {}).get('detail'),
                    "shutdown_stages": self.shutdown_stages, "gateway_pid": self.process.pid}
        report = self.last_report or {"kind": "unconfirmed", "stop_failures": ["missing final gateway report"]}
        if self.fault_evidence is not None:
            report.setdefault('fault_state', self.fault_evidence.get('fault_state'))
            report.setdefault('fault', self.fault_evidence.get('detail'))
        report.setdefault('shutdown_stages', self.shutdown_stages).append(dict(
            phase='gateway_exit', event='completed' if self.process.exitcode == 0 else 'failed',
            at=time.monotonic(), exitcode=self.process.exitcode))
        if self.process.exitcode != 0:
            report['kind'] = 'unconfirmed'
            report.setdefault('stop_failures', []).append(f'gateway exit code {self.process.exitcode}')
        return {**report, "parent_state_timing": list(self.parent_timing)}
