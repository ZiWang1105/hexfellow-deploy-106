"""Resumable Chinese onsite wizard. No motion on import or status/help."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..control.first_trial import apply_envelope
from ..control.motion_profile import load_motion_profile
from ..paths import CONFIG_DIR, PROJECT_ROOT
from . import commission, session
from .direct_quest import check_power, check_start_position
from .gateway import HardwareGateway
from .ik_runner import TraceWriter, _wait_state, jsonable
from .verification import (
    RESEARCH_ACCEPTANCE,
    SESSION_COMMANDS,
    STEP_BY_NAME,
    STEPS,
    checked_results,
    device_binding,
    digest,
    machine_pass,
    permission,
    read_json,
    software_binding,
)
from .whole_body import SLICES, RobotMapping
from .x7 import HardwareSafetyError, FeedbackSkewSafetyError, FeedbackUnavailable
from .feedback_monitor import attach_feedback_result, feedback_metrics

DEFAULT_RUN = PROJECT_ROOT / "runs/hexfellow-onsite"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_progress(root, mapping):
    path = root / "verification.json"
    binding = software_binding(mapping)
    if not path.exists():
        return {
            "schema_version": 1,
            "binding": binding,
            "research_acceptance": RESEARCH_ACCEPTANCE,
            "completed": {},
            "attempts": [],
            "device_binding": None,
        }
    value = read_json(path)
    if value.get("schema_version") != 1 or value.get("binding") != binding:
        raise HardwareSafetyError(
            "Configuration, control source, or SDK changed.\nStart a new round with --run-dir runs/hexfellow-onsite-new; retain old reports."
        )
    return value


def invalidate_from(ledger, name):
    names = [s.name for s in STEPS]
    index = names.index(name)
    for later in names[index:]:
        ledger["completed"].pop(later, None)
    if index == 0:
        ledger["device_binding"] = None


def confirm_start(step):
    print(f"\nCurrent step:{step.name}\n{step.explanation}", flush=True)
    if step.test != "probe":
        print("Empty grippers; charger disconnected; sufficient clearance and cable slack; E-stop accessible. Use E-stop if abnormal.", flush=True)
    answer = input("Press Enter to start this step, or q to quit:").strip().lower()
    return answer == ""


def healthy_readonly(state, *, readiness=True, allow_lift_retreat=False):
    if readiness:
        check_power(state, acquiring=True)
        check_start_position(state, allow_lift_retreat=allow_lift_retreat)
    else:
        check_power(state, acquiring=False, ignore_low_battery=True, allow_charging_readonly=True)
    for name, row in state.health.items():
        if name == 'base' and row.get('warning') is not None:
            raise HardwareSafetyError(f"base warning: {row['warning']}")
        if row.get("parking_stop") is not None:
            raise HardwareSafetyError(f"{name}: Stop/fault not cleared; inspect the robot and remote-control state")
        if name == "theta":
            if not row.get("calibrated") or row.get("state") not in {"LsBrake", "LsAlgorithmControl"}:
                raise HardwareSafetyError("Theta is uncalibrated or unavailable for control; this script does not home automatically")
        elif not row.get("motor_states") or any(v != "normal" for v in row["motor_states"]):
            raise HardwareSafetyError(f"{name}: Abnormal motor state")
        elif row.get("session_holder") != 0:
            raise HardwareSafetyError(f"{name}: Control not released; exit other control programs")


def probe(mapping, profile, directory, *, before=None):
    writer = TraceWriter(directory)
    report = {
        "mode": "probe",
        "passed": False,
        "motion_commands_sent": 0,
        "start_checked": False,
        "mapping_sha256": mapping.sha256,
        "profile_sha256": profile.sha256,
    }
    gateway = HardwareGateway(mapping, profile, "whole-body", "probe", profile.mit_servo.torque_limit_nm)
    try:
        gateway.start()
        previous, stable = {}, dict.fromkeys(SLICES, 0.0)
        previous_sequences = {}
        seen_invalid = 0
        report['recoverable_observations'] = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                state = gateway.state(fresh=False, deadline=deadline)
                # Check real faults before considering skew recovery.
                state.validate(time.monotonic(), check_skew=False)
                healthy_readonly(state)
                for name, stamp in state.received.items():
                    if name in previous and (stamp < previous[name] or
                            state.sequences[name] < previous_sequences[name]):
                        raise HardwareSafetyError(f'{name} feedback timestamp/sequence regressed')
                stats = (state.timing or {}).get('feedback_observations', {})
                invalid = stats.get('invalid_samples', 0)
                if invalid != seen_invalid:
                    previous, stable = {}, dict.fromkeys(SLICES, 0.0)
                    seen_invalid = invalid
                report['last_feedback_metrics'] = feedback_metrics(state, time.monotonic())
                state.validate(time.monotonic())
            except (FeedbackSkewSafetyError, FeedbackUnavailable) as error:
                report['recoverable_observations'] += 1
                report['last_recoverable_error'] = str(error)
                previous, stable = {}, dict.fromkeys(SLICES, 0.0)
                writer.record({'kind': 'readonly_retry', 'at': time.monotonic(), 'error': str(error)})
                time.sleep(min(.004, max(0., deadline-time.monotonic())))
                continue
            writer.record({"kind": "readonly_state", "state": state})
            report["device_binding"] = device_binding(state)
            report["state"] = jsonable(state)
            for name, part in SLICES.items():
                stamp = state.received[name]
                old = previous.get(name)
                if old is not None and stamp > old:
                    dt = stamp - old
                    speed_limit = 0.002 if name in {"theta", "base"} else 0.01
                    stable[name] = stable[name] + dt if dt <= 0.04 and np.max(abs(state.v[part])) <= speed_limit else 0
                previous[name] = stamp
                previous_sequences[name] = state.sequences[name]
            if min(stable.values()) >= 0.5:
                if before is not None:
                    delta = state.q - np.asarray(before["q"])
                    # Endpoint connections have independent odometry origins. Compare raw odometry.
                    delta[:3] = np.asarray(state.health["base"]["odom"]) - np.asarray(before["health"]["base"]["odom"])
                    delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
                    report["post_event_displacement"] = delta.tolist()
                    if np.any(abs(delta) > np.r_[0.003, 0.003, 0.01, 0.003, np.full(14, 0.02)]):
                        raise HardwareSafetyError("Measured pose change across the event exceeds the stationary verification range")
                report.update(passed=True, start_checked=True, stable_seconds=stable)
                break
            time.sleep(0.01)
        if not report["passed"]:
            raise HardwareSafetyError("No continuous 0.5-second fresh stationary feedback from every device; retry, but do not use stale frames to confirm stopping")
        print(f"Read-only stationary check: lift {state.q[3] * 1000:.1f} mm; valid arm/lift/base feedback.", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["gateway"] = gateway.close()
        report["motion_commands_sent"] = report["gateway"].get("motion_commands_sent", 0)
        if report["gateway"].get("fault") or report["gateway"].get("stop_failures") or report["motion_commands_sent"]:
            report["passed"] = False
        attach_feedback_result(report, require_clean=report.get('mode') in {'feedback-only', 'audit', 'timing'})
        writer.close(report)
    return report


def feedback_probe(mapping, profile, directory, seconds):
    """Communication-only observation; no IK, ownership or ledger unlock."""
    from .realtime import control_runtime
    from .x7 import FeedbackUnavailable
    report = dict(mode="feedback-only", passed=False, diagnostic_only=True,
        hardware_execution_qualified=False, cycles=0, failed_cycles=0, skipped_slots=0,
        requested_seconds=seconds, software_binding=software_binding(mapping))
    writer = TraceWriter(directory)
    gateway = HardwareGateway(mapping, profile, "whole-body", "probe", profile.mit_servo.torque_limit_nm)
    try:
        with control_runtime():
            gateway.start()
            state = _wait_state(gateway, fresh=False)
            healthy_readonly(state, readiness=False)
            try:
                healthy_readonly(state)
                report["motion_readiness_passed"] = True
            except HardwareSafetyError as error:
                report.update(motion_readiness_passed=False, motion_readiness_error=str(error))
            start = time.monotonic()
            total = math.ceil(seconds / .02)
            tick = 0
            while tick < total:
                row = dict(kind="feedback", sequence=tick, at=time.monotonic())
                try:
                    state = gateway.state(fresh=False)
                    healthy_readonly(state, readiness=False)
                    row['state'] = state
                    try:
                        state.validate(time.monotonic())
                    except HardwareSafetyError as error:
                        row['feedback_error'] = str(error)
                        report['failed_cycles'] += 1
                except FeedbackUnavailable as error:
                    row['feedback_error'] = str(error)
                    report['failed_cycles'] += 1
                writer.record(row)
                report['cycles'] += 1
                tick += 1
                while tick < total and start + tick * .02 < time.monotonic():
                    tick += 1
                    report['skipped_slots'] += 1
                time.sleep(max(0., start + tick * .02 - time.monotonic()))
            report.update(elapsed_seconds=time.monotonic()-start,
                          passed=not report['failed_cycles'] and not report['skipped_slots'])
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['gateway'] = gateway.close()
        report['motion_commands_sent'] = report['gateway'].get('motion_commands_sent', 0)
        if report['motion_commands_sent'] or report['gateway'].get('fault') or report['gateway'].get('stop_failures'):
            report['passed'] = False
        if not report['passed']:
            report.setdefault('error', 'communication observation failed; inspect feedback errors and gateway window')
        attach_feedback_result(report, require_clean=report.get('mode') in {'feedback-only', 'audit', 'timing'})
        writer.close(report)
    return report


def timing_probe(mapping, profile, directory, seconds, *, sdk_sink=False, allow_lift_retreat=False):
    """Read-only counterfactual IK. Performance misses are counted over the window."""
    from contextlib import ExitStack
    from .control_timing import check_compute_state, dispatch_deadline, select_control_state
    from .realtime import control_runtime

    if allow_lift_retreat and (not sdk_sink or not mapping.runtime_session
                              or mapping.data.get("lift_position_mode") != "range_max_minus_sdk"):
        raise ValueError("lift-retreat audit requires a reverse-mapped runtime session and local SDK sink")
    readiness_options = {"allow_lift_retreat": True} if allow_lift_retreat else {}
    writer = TraceWriter(directory)
    report = dict(mode="audit" if sdk_sink else "timing", passed=False, cycles=0, attempted_cycles=0,
                  failed_cycles=0, skipped_slots=0, motion_commands_sent=0,
                  requested_seconds=seconds, diagnostic_only=True, lift_retreat_allowed=allow_lift_retreat,
                  scope="real feedback + counterfactual IK" + (" + local SDK/KCP/UDP sink" if sdk_sink else ""),
                  hardware_execution_qualified=False, software_binding=software_binding(mapping),
                  mapping_sha256=mapping.sha256, profile_sha256=profile.sha256)
    gateway = HardwareGateway(mapping, profile, "whole-body", "probe", profile.mit_servo.torque_limit_nm)
    metrics = []
    sink = None
    scope = ExitStack()
    report["cpu_scheduling"] = scope.enter_context(control_runtime())
    try:
        gateway.start()
        state = _wait_state(gateway)
        report["initial_health"] = jsonable(state.health)
        writer.record(dict(kind="initial_state", state=state))
        try:
            healthy_readonly(state, **readiness_options)
            report["motion_readiness_passed"] = True
        except HardwareSafetyError as readiness_error:
            report["motion_readiness_passed"] = False
            report["motion_readiness_error"] = str(readiness_error)
            if not sdk_sink:
                raise
            healthy_readonly(state, readiness=False)
        grippers = np.asarray(mapping.data["fixed_gripper_positions"], dtype=float)
        try:
            check_start_position(state, **readiness_options)
            from ..control.robot import LIFT_JOINT
            from ..control.ik_core import LIFT_REFERENCE_TOP_RESERVE_M
            retreat = bool(allow_lift_retreat and state.q[3] > mapping.physical_bounds(state)[LIFT_JOINT][1]
                           - .002 - LIFT_REFERENCE_TOP_RESERVE_M)
            options = {"allow_upper_lift_retreat": True} if retreat else {}
            core = commission.make_core(
                profile, dataclasses.replace(mapping, commissioning_stage="whole-body"), state, **options)
            poses = commission.initialize_commission_core(core, state, grippers, stationary_reference=retreat)
            report['lift_retreat_required'] = retreat
            report['audit_control_phase'] = 'lift-retreat' if retreat else 'tracking'
            if retreat:
                print('Theta is at the top: read-only preflight permits holding there; after entering the session, D lowers slowly before resetting pose.', flush=True)
            # Test a stationary reference through the same inward-only preparation
            # constraints as the session. Packets go solely to the local sink.
            def phase_options():
                return (dict(control_phase='lift-retreat', posture_target=core._last_safe_positions.copy(),
                             posture_velocity=np.zeros(core.model.nv)) if retreat else {})
        except (ValueError, HardwareSafetyError) as error:
            report['ik_eligible'] = False
            raise HardwareSafetyError(f"IK initial pose unavailable:{error}. Use remote control to move inside the planning range; audit --feedback-only checks communication alone.") from error
        report['ik_eligible'] = True
        if sdk_sink:
            from .loopback_sink import LoopbackSink
            sink = LoopbackSink()
        core.step(state.q, poses, measured_velocity=state.v, state_timestamp=state.timestamp,
                  control_timestamp=time.monotonic(), measured_gripper_positions=grippers,
                  **phase_options(), **state.ik_feedback())
        start = next_tick = time.monotonic()
        total_slots = math.ceil(seconds / .02)
        tick = 0
        while tick < total_slots:
            slot_deadline = next_tick + .02
            cycle_started = time.monotonic()
            row = dict(cycle_started=cycle_started, slot_deadline=slot_deadline,
                       sequence=report["attempted_cycles"], preparation_started=cycle_started)
            report["attempted_cycles"] += 1
            try:
                # Reset against an explicitly recorded preceding observation BEFORE
                # selecting the frame used for this solve. Never advance an executed
                # reference history on a real robot that this audit cannot command.
                row["preparation_received"] = state.received
                core.reset(state.q, poses, measured_velocity=state.v, state_timestamp=state.timestamp,
                           measured_gripper_positions=grippers, stationary_reference=True)
                row["preparation_completed"] = time.monotonic()
                state = select_control_state(gateway.state, slot_deadline,
                    preferred_age_s=.010 if os.environ.get('POLICY_SIM_RESPONSIVE_CPUS') else None)
                if sdk_sink:
                    healthy_readonly(state, readiness=False)
                else:
                    healthy_readonly(state)
                now = time.monotonic()
                row.update(state.timing or {})
                row.update(compute_started=now, source_age_ms=(now - state.timestamp) * 1000,
                           theta_age_ms=(now - state.received["theta"]) * 1000)
                check_compute_state(state, now, slot_deadline)
                candidate = core.step(state.q, poses, measured_velocity=state.v, state_timestamp=state.timestamp,
                                      control_timestamp=now, measured_gripper_positions=grippers,
                                      **phase_options(), **state.ik_feedback())
                finished = time.monotonic()
                row.update(compute_completed=finished, compute_ms=(finished-now)*1000,
                           preparation_ms=(row["preparation_completed"]-cycle_started)*1000,
                           slot_remaining_ms=(slot_deadline-finished)*1000)
                if not candidate.executable or candidate.diagnostics.fault or finished >= slot_deadline:
                    raise HardwareSafetyError(f"timing IK/deadline failure: {candidate.diagnostics.fault}")
                if sink is not None:
                    sends = sink.send(candidate, row["sequence"], now, dispatch_deadline(state, now, slot_deadline))
                    row["local_sdk_dispatch"] = sends
                    last = max(r["first_udp_send_started"] for r in sends.values())
                    row["cycle_to_udp_ms"] = (last - cycle_started) * 1000
                    row["source_to_udp_ms"] = (last - state.timestamp) * 1000
                report["cycles"] += 1
                row["passed"] = True
            except HardwareSafetyError as error:
                row.update(passed=False, failed_at=time.monotonic(), error=str(error))
                report["failed_cycles"] += 1
                # Only computational/admission performance misses can continue in
                # read-only mode. Broken health/transport/telemetry remains fatal.
                if not any(token in str(error) for token in ("budget", "slot expired", "IK/deadline", "command expired", "expired MIT publisher")):
                    raise
            finally:
                metrics.append(row)
                writer.record(dict(kind="timing", state=state, timing=row))
            tick += 1
            next_tick = start + tick * .02
            while tick < total_slots and next_tick < time.monotonic():
                report["skipped_slots"] += 1
                tick += 1
                next_tick = start + tick * .02
            time.sleep(max(0, next_tick - time.monotonic()))
        report["elapsed_seconds"] = time.monotonic() - start
        report["passed"] = not report["failed_cycles"] and not report["skipped_slots"]
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            report["gateway"] = gateway.close()
            report["motion_commands_sent"] = report["gateway"].get("motion_commands_sent", 0)
            if report["gateway"].get("fault") or report["gateway"].get("stop_failures") or report["motion_commands_sent"]:
                report["passed"] = False
        except Exception as error:
            report.update(passed=False, close_error=str(error))
        if sink is not None:
            report["local_sink_received"] = sink.received.copy()
            sink.close()
        scope.close()
        report["timing_ms_p50_p95_p99_max"] = {
            key: np.percentile([r[key] for r in metrics if key in r], [50, 95, 99, 100]).tolist()
            for key in sorted({k for r in metrics for k in r if k.endswith("_ms")})}
        if sdk_sink:
            from .timing_stats import qualify_performance
            stats = report["timing_ms_p50_p95_p99_max"]
            report.update(qualify_performance({'local_sink': {
                key.removesuffix('_ms'): {'count': sum(key in r for r in metrics), 'p50_p95_p99_max_ms': values}
                for key, values in stats.items()}}, report['attempted_cycles']))
            report["passed"] &= report["performance_passed"]
        if not report["passed"] and "error" not in report:
            report["error"] = "audit performance/deadline targets not met; inspect all-cycle trace"
        attach_feedback_result(report, require_clean=report.get('mode') in {'feedback-only', 'audit', 'timing'})
        writer.close(report)
    return report


class ExternalEvent:
    """UI/carrier observation stays off the command-producing thread."""

    def __init__(self, kind, interface):
        self.kind, self.interface = kind, interface
        self.armed, self.finished = threading.Event(), threading.Event()
        self.armed_at = None
        self.before = None
        self.carrier_down_at = None
        self.carrier_up_at = None
        self.thread = threading.Thread(target=self.watch, daemon=True, name="onsite-observer")

    def carrier(self):
        return (Path("/sys/class/net") / self.interface / "carrier").read_text().strip() == "1"

    def start(self):
        if self.kind == "disconnect" and not self.carrier():
            raise HardwareSafetyError("Wired interface must report link-up before the disconnect test")
        self.thread.start()

    def arm(self, state):
        if not self.armed.is_set():
            self.before = state
            self.armed_at = time.monotonic()
            self.armed.set()

    def watch(self):
        while not self.armed.wait(0.05):
            if self.finished.is_set():
                return
        print(
            "\nWhole robot is holding stationary. Now "
            + (
                "press the physical E-stop."
                if self.kind == "estop"
                else "unplug the robot data cable at the computer; watch for drops or drift and use E-stop if abnormal."
            ),
            flush=True,
        )
        prompted = False
        while not self.finished.wait(0.02):
            if self.kind != "disconnect":
                continue
            try:
                up = self.carrier()
            except OSError:
                continue  # An unreadable carrier is not proof of a disconnected cable.
            now = time.monotonic()
            if not up and self.carrier_down_at is None:
                self.carrier_down_at = now
            if self.carrier_down_at is not None and now - self.carrier_down_at >= 3 and not prompted:
                print("\nDisconnect observed. Reconnect the data cable; the program will not automatically reacquire control.", flush=True)
                prompted = True
            if self.carrier_down_at is not None and up and self.carrier_up_at is None:
                self.carrier_up_at = now

    def close(self):
        self.finished.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1)


def event_observed(kind, outcome, monitor):
    if not monitor.armed.is_set():
        return False
    if kind == "disconnect":
        at = outcome.get("fault_observed_at", 0)
        reason = str(outcome.get("fault", "")).lower()
        return (
            monitor.carrier_down_at is not None
            and at >= monitor.carrier_down_at - 0.1
            and any(word in reason for word in ("stale", "expired", "transport", "disconnect", "send failed"))
        )
    state = outcome.get("fault_state")
    if dataclasses.is_dataclass(state):
        state = jsonable(state)
    if not state:
        return False
    return any(
        (row.get("parking_stop") or {}).get("category") in {0, 8} and state["received"][name] >= monitor.armed_at
        for name, row in state["health"].items()
    )


def external_test(values, mapping, profile, directory):
    monitor = ExternalEvent(values.test, mapping.data["interface"])
    result = {
        "passed": False,
        "execute": True,
        "test": values.test,
        "event_observed": False,
        "post_event_stationary": False,
        "recovery_motion_commands": 0,
    }
    try:
        monitor.start()
        values.save_trace = directory / "event"
        raw = commission.run(values, external_event=monitor, quiet=True)
        if raw.get("trace_error"):
            raise HardwareSafetyError("Event log write failed; cannot serve as passing evidence")
        result.update(
            device_binding=raw.get("device_binding"),
            raw_event_report="event/summary.json",
            pre_event_hold_confirmed=raw.get("pre_event_hold_confirmed", False),
        )
        outcome = raw.get("gateway", {})
        result["event_observed"] = event_observed(values.test, outcome, monitor)
        if not result["event_observed"]:
            raise HardwareSafetyError("Required physical E-stop/network disconnect not detected; other faults do not count as test success")
        if outcome.get("kind") != "closed":
            raise HardwareSafetyError("Control process exit unconfirmed; press physical E-stop first; recovery check blocked")
        print("\nControl process exited. Confirm the robot is stationary; reconnect the cable or manually reset E-stop. Do not move the robot.", flush=True)
        if input("Press Enter after reconnection with the robot stationary, or q if abnormal:").strip():
            raise HardwareSafetyError("Operator cancelled event recovery check")
        recovery = probe(mapping, profile, directory / "recovery", before=jsonable(monitor.before))
        recovery_ok = machine_pass(STEP_BY_NAME["probe"], recovery)
        result["post_event_stationary"] = recovery_ok
        result["recovery_motion_commands"] = recovery["motion_commands_sent"]
        result["recovery_device_binding"] = recovery.get("device_binding")
        result["passed"] = (
            recovery_ok
            and recovery.get("device_binding") == result["device_binding"]
            and not recovery["motion_commands_sent"]
        )
        if values.test == "disconnect":
            duration = (
                monitor.carrier_up_at - monitor.carrier_down_at
                if monitor.carrier_up_at is not None and monitor.carrier_down_at is not None
                else 0.0
            )
            result["carrier_down_duration_s"] = duration
            if duration < 3:
                raise HardwareSafetyError("Disconnect observed for less than 3 seconds; retry and wait for the reconnect prompt")
        result["scope"] = "stationary event observation; missing feedback during event is not stop confirmation"
        result["expected_transport_cleanup_errors"] = outcome.get("stop_failures", [])
    except (Exception, KeyboardInterrupt) as error:
        result["passed"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["mapping_sha256"], result["profile_sha256"] = mapping.sha256, profile.sha256
        result["carrier_down_at"], result["carrier_up_at"] = monitor.carrier_down_at, monitor.carrier_up_at
        monitor.close()
        save_json(directory / "summary.json", result)
    return result


def commission_args(step, mapping_path, root, ledger, directory):
    parser = argparse.ArgumentParser()
    commission.add_arguments(parser)
    values = parser.parse_args(
        [
            "--robot-config",
            str(mapping_path),
            "--stage",
            step.stage,
            "--test",
            step.test,
            "--feedback-policy",
            RobotMapping.load(mapping_path).feedback_policy,
            "--execute",
            "--confirm-onsite-ready",
            "--save-trace",
            str(directory),
        ]
    )
    if step.name == "body-hold":
        values.hold_seconds = 60.0
    if step.test != "hold":
        hold = next(s for s in STEPS if s.test == "hold" and s.stage == step.stage)
        values.previous_report = root / ledger["completed"][hold.name]["report"]
    return values


def session_args(mapping_path, report_path, directory, *, step=None, rate="auto"):
    parser = argparse.ArgumentParser()
    session.add_arguments(parser)
    values = parser.parse_args(
        [
            "--mode",
            "live",
            "--robot-config",
            str(mapping_path),
            "--verification-report",
            str(report_path),
            "--save-trace",
            str(directory),
            "--confirm-onsite-ready",
            "--quest-pose-rate",
            rate,
        ]
    )
    values.allow_uncommissioned_profile_sha256 = RobotMapping.load(mapping_path).data["commissioning"]["profile_sha256"]
    values.commands = SESSION_COMMANDS.get(step)
    return values


def latency_comparison(root):
    latest = {}
    for path in sorted(root.glob("latency-*/summary.json")):
        report = read_json(path)
        rate = report.get("quest_pose_rate")
        if report.get("passed") and report.get("quest_active_cycles", 0) >= 50 and rate in {"50", "auto"}:
            latest[rate] = (path, report)
    if set(latest) != {"50", "auto"}:
        print("Data for the other pose rate is missing; run latency with --quest-pose-rate 50 and auto.")
        return
    a, b = latest["50"][1], latest["auto"][1]
    if any(a.get(key) != b.get(key) for key in ("mapping_sha256", "profile_sha256", "device_binding")):
        print("Configuration or devices differ between runs; direct comparison is invalid.")
        return
    rows = {}
    for key in sorted(a.get("metrics", {}).keys() & b.get("metrics", {}).keys()):
        rows[key] = {
            rate: {stat: latest[rate][1]["metrics"][key][stat] for stat in ("p95", "p99")} for rate in ("50", "auto")
        }
    value = {
        "reports": {rate: str(latest[rate][0]) for rate in latest},
        "metrics_ms": rows,
        "interpretation": "operator-matched small motions; SDK timing is not mechanical response latency",
    }
    save_json(root / "latency-comparison.json", value)
    print("50 / auto comparison (milliseconds; SDK send time is not mechanical response time):")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def execute_step(step, args, mapping, profile, ledger, directory):
    if step.test == "probe":
        return probe(mapping, profile, directory)
    # Re-read physical state before each motion, without changing any device configuration.
    preflight = probe(mapping, profile, directory.parent / (directory.name + "-preflight"))
    preflight_path = directory.parent / (directory.name + '-preflight') / 'summary.json'
    if not preflight['passed']:
        raise HardwareSafetyError(f"Read-only initial check failed:{preflight.get('error', 'No continuous stationary feedback')}；"
            f"feedback metrics={preflight.get('last_feedback_metrics')}; detailed report:{preflight_path}; control not acquired")
    if preflight.get('device_binding') != ledger['device_binding']:
        raise HardwareSafetyError(f"Devices differ from this qualification binding; detailed report:{preflight_path}; control not acquired")
    if step.test == "session":
        if step.name == "quest":
            print(
                "Enter default, init, quest in order; wait for each step to complete.\n"
                "After headset connection, move both arms slightly for at least 2 seconds, release squeeze, then enter stop and quit.",
                flush=True,
            )
        values = session_args(args.robot_config, args.run_dir / "verification.json", directory, step=step.name)
        return session.run(values, verification_scope=step.name)
    values = commission_args(step, args.robot_config, args.run_dir, ledger, directory)
    if step.test in {"estop", "disconnect"}:
        return external_test(values, mapping, profile, directory)
    return commission.run(values, quiet=True)


def add_arguments(parser):
    parser.add_argument("--suite", choices=("full", "quest-trial"), default="full")
    parser.add_argument("action", nargs="?", choices=("status", "next", "run", "control", "latency", "timing", "audit"), default="status")
    parser.add_argument("--seconds", type=float, default=300, help="audit/timing: diagnostic duration (1–600 seconds)")
    parser.add_argument("--feedback-only", action="store_true", help="audit: observe communication without IK or motion readiness")
    parser.add_argument("step", nargs="?", help="run: step name or number")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--research-release", type=Path, help="local binding for the GP80 Quest research suite")
    parser.add_argument("--quest-stream-config", type=Path, default=None)
    parser.add_argument("--quest-pose-rate", choices=("auto", "30", "50", "60", "90"), default=None)
    parser.add_argument("--quest-link-policy", choices=("strict", "target-200ms", "buffered"), default=None,
                        help="quest-trial input policy; other suites retain strict")
    parser.add_argument("--bind-ip", default="auto", help="quest-trial: HTTPS private LAN IPv4 address")
    parser.add_argument("--port", type=int, default=8443, help="quest-trial: HTTPS port")


def run(args):
    trial = getattr(args, 'suite', 'full') == 'quest-trial'
    if args.robot_config is None:
        from .research_profiles import RUNTIME
        args.robot_config = RUNTIME if trial else CONFIG_DIR / 'hardware/hexfellow_runtime.json'
    args.quest_pose_rate = args.quest_pose_rate or ('50' if trial else 'auto')
    args.quest_link_policy = getattr(args, 'quest_link_policy', None) or ('buffered' if trial else 'strict')
    if not trial and args.quest_link_policy != 'strict':
        raise HardwareSafetyError('--quest-link-policy buffered/target-200ms requires --suite quest-trial')
    if getattr(args, 'suite', 'full') == 'quest-trial':
        from .quest_trial import run as run_trial

        if args.run_dir == DEFAULT_RUN:
            args.run_dir = PROJECT_ROOT / 'runs/hexfellow-quest-trial'
        return run_trial(args)
    if getattr(args, 'feedback_only', False) and args.action != 'audit':
        raise ValueError('--feedback-only is supported only by audit')
    args.run_dir = args.run_dir.resolve()
    args.robot_config = args.robot_config.resolve()
    mapping = dataclasses.replace(
        RobotMapping.load(args.robot_config), motion_envelope="first-trial"
    )
    if not mapping.runtime_session:
        raise HardwareSafetyError("Verification wizard requires a hexfellow_runtime configuration")
    profile = apply_envelope(load_motion_profile(mapping.data["commissioning"]["motion_profile"]), "first-trial")
    ledger = load_progress(args.run_dir, mapping)
    good = checked_results(ledger, args.run_dir)
    if args.action == "status":
        for index, step in enumerate(STEPS, 1):
            print(f"{'passed' if step.name in good else 'pending'} {index:02d} {step.name}: {step.explanation}")
        print(f"\nCompleted {len(good)}/{len(STEPS)}; firmware clamps unknown and accepted for research; hardware certification not asserted.")
        print("Next step:" + ("control" if len(good) == len(STEPS) else "next → " + STEPS[len(good)].name))
        return
    # One owner for the whole wizard/control session. Never interleave two robot tests.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "runs").mkdir(exist_ok=True)
    with (PROJECT_ROOT / "runs/.hexfellow-onsite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise HardwareSafetyError("Another verification/control process is running; exit it first") from error
        ledger = load_progress(args.run_dir, mapping)
        good = checked_results(ledger, args.run_dir)
        if args.action in {"timing", "audit"}:
            if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 600:
                raise ValueError("--seconds must be between 1 and 600 seconds")
            directory = args.run_dir / (args.action + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
            print("Read-only diagnostics:" + ("observe communication only" if getattr(args, 'feedback_only', False) else "read feedback and run IK")
                  + "; no control acquisition or live motion commands; keep the robot stationary.", flush=True)
            result = (feedback_probe(mapping, profile, directory, args.seconds) if getattr(args, 'feedback_only', False)
                      else timing_probe(mapping, profile, directory, args.seconds, sdk_sink=True) if args.action == "audit"
                      else timing_probe(mapping, profile, directory, args.seconds))
            print("Report:", directory / "summary.json")
            if not result["passed"]:
                raise HardwareSafetyError(result.get("error", result.get("close_error", "Read-only timing failed; see report")))
            print("Read-only timing passed; motion verification is still locked. Run next to start the first step.")
            if result.get('performance_grade') == 'relaxed':
                print('Meets relaxed P99 <=15 ms acceptance; the <=10 ms optimization target remains unmet.')
            return
        if args.action in {"control", "latency"}:
            if len(good) != len(STEPS):
                raise HardwareSafetyError(f"Onsite verification incomplete ({len(good)}/{len(STEPS)}); run next first")
            permission(args.run_dir / "verification.json", mapping)
            input("Confirm empty grippers, charger disconnected, clearance/cables, and ready E-stop, then press Enter; Ctrl-C cancels:")
            directory = args.run_dir / (args.action + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
            values = session_args(
                args.robot_config,
                args.run_dir / "verification.json",
                directory,
                step="quest" if args.action == "latency" else None,
                rate=args.quest_pose_rate,
            )
            if args.action == "latency":
                print("Low-speed latency comparison: default -> init -> quest; repeat the same small motions for about 60 seconds, then stop and quit.")
            result = session.run(values, verification_scope="quest" if args.action == "latency" else "control")
            print("Report:", directory / "summary.json")
            if not result["passed"]:
                raise HardwareSafetyError(result.get("finish_reason", "Control session failed"))
            if args.action == "latency":
                print(json.dumps(result.get("metrics", {}), ensure_ascii=False, indent=2))
                latency_comparison(args.run_dir)
            return
        names = [s.name for s in STEPS]
        if args.action == "run":
            name = args.step
            if name and name.isdigit() and 1 <= int(name) <= len(STEPS):
                name = names[int(name) - 1]
            if name not in STEP_BY_NAME:
                raise HardwareSafetyError("After run, specify the step number or name shown by status")
            index = names.index(name)
            if index > len(good):
                raise HardwareSafetyError("Complete prerequisite steps first; skipping is not allowed")
        else:
            index = len(good)
            if index == len(STEPS):
                print("Verification complete. Run control to enter the full control terminal.")
                return
        step = STEPS[index]
        if not confirm_start(step):
            return
        invalidate_from(ledger, step.name)
        save_json(args.run_dir / "verification.json", ledger)
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        directory = args.run_dir / f"{index + 1:02d}-{step.name}-{tag}"
        try:
            result = execute_step(step, args, mapping, profile, ledger, directory)
        except (Exception, KeyboardInterrupt) as error:
            result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            if not (directory / "summary.json").exists():
                save_json(directory / "summary.json", result)
        report_path = directory / "summary.json"
        okay = machine_pass(step, result)
        normal = False
        if okay:
            try:
                answer = input("Automated checks passed. Were physical motion, direction, and stopping normal? Enter y to confirm; any other answer records an anomaly:")
                normal = answer.strip().lower() == "y"
            except (EOFError, KeyboardInterrupt):
                normal = False
        if okay and step.test == "probe":
            ledger["device_binding"] = result["device_binding"]
        entry = {
            "step": step.name,
            "report": str(report_path.relative_to(args.run_dir)),
            "sha256": digest(report_path),
            "operator_normal": normal,
            "machine_passed": okay,
            "at": tag,
        }
        ledger["attempts"].append(entry)
        if okay and normal and result.get("device_binding") == ledger["device_binding"]:
            ledger["completed"][step.name] = entry
        save_json(args.run_dir / "verification.json", ledger)
        print("Report:", report_path)
        if step.name not in ledger["completed"]:
            reason = result.get("error", result.get("finish_reason", "Onsite observation or automated checks failed"))
            gateway = result.get("gateway", {})
            for failure in gateway.get("stop_failures", []):
                print("Stop confirmation fault:", failure)
            if gateway.get("stop_failures") and not result.get("error"):
                reason = "; ".join(gateway["stop_failures"])
            if gateway.get("fault"):
                print("Publisher process fault:", gateway["fault"])
            for name, row in gateway.get("direct_downlink", {}).items():
                if row.get("fault"):
                    print(f"{name} downlink gate fault:{row['fault']}")
            raise HardwareSafetyError("This step failed; subsequent steps remain locked. Reason:" + str(reason))
        print("This step passed." + ("All complete; run control." if index == len(STEPS) - 1 else "Run next again for the next step."))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hexfellow step-by-step onsite verification; shows progress by default")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (HardwareSafetyError, OSError, ValueError, EOFError, KeyboardInterrupt) as error:
        print(f"Not continued:{error}", flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
