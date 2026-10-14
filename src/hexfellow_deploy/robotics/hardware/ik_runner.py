"""Headless measured-feedback IK commissioning and Quest teleoperation.

Probe/shadow never acquire control. Live requires explicit hardware review,
the immutable profile hash, prerequisite reports, and local operator approval.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pinocchio as pin

from ..control.ik_core import TCP_FRAMES, PinkIKCore
from ..control.motion_profile import controller_config_for_profile, load_motion_profile
from ..control.posture import NullPose
from ..control.target_timing import TargetSample
from ..paths import CONFIG_DIR
from .gateway import HardwareGateway
from .whole_body import COMMAND_TTL, STAGES, RobotMapping, WholeBodyCommand, locked_names
from .x7 import ControlDelay, HardwareSafetyError


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


from .trace_process import TraceWriter


def tcp_poses(core, q):
    if not hasattr(core, "_hardware_fk_data"):
        core._hardware_fk_data = core.model.createData()
    data = core._hardware_fk_data
    pin.forwardKinematics(core.model, data, q)
    pin.updateFramePlacements(core.model, data)
    return {side: data.oMf[core.model.getFrameId(frame)].copy() for side, frame in TCP_FRAMES.items()}


def record_feedback_diagnostics(writer, summary, state, now):
    theta_age_ms = (now - state.received["theta"]) * 1000
    summary["theta_feedback_max_age_ms"] = max(summary["theta_feedback_max_age_ms"], theta_age_ms)
    skew = max(abs(state.received["theta"] - state.received[n]) for n in ("left", "right", "base"))
    if state.feedback_policy != "strict" and (theta_age_ms > 40 or skew > .02):
        summary["theta_gap_warning_samples"] += 1
        writer.record({"kind": "theta_short_gap_warning", "age_ms": theta_age_ms,
                       "received": state.received, "sequences": state.sequences})


def initialize_core(core, state, grippers, *, stationary_reference=False):
    # Initial measured posture, rather than a distant simulation home, is the
    # commissioning nullspace reference. Profile weights/gains remain unchanged.
    if core.config.posture_policy is None:
        core.set_nominal_pose(NullPose(float(state.q[3]), tuple(state.q[4:11]), tuple(state.q[11:18])))
    poses = tcp_poses(core, state.q)
    core.reset(state.q, poses, measured_velocity=state.v, state_timestamp=state.timestamp,
               measured_gripper_positions=grippers, stationary_reference=stationary_reference)
    return poses


def manual_distance(value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 0.05:
        raise ValueError("manual-distance-m must be finite and in (0, 0.05]")
    return value


def manual_targets(initial, stage, distance_m=0.01):
    distance_m = manual_distance(distance_m)
    if distance_m != 0.01 and stage not in {"left-arm", "right-arm"}:
        raise ValueError("custom manual distance requires a single-arm nudge")
    delta = np.array([0.04, 0.0, 0.0]) if stage == "whole-body" else np.array(
        [0.0, 0.0, 0.02 if stage == "arms-lift" else distance_m]
    )
    outward = {side: pose.copy() for side, pose in initial.items()}
    for side in ("left", "right"):
        if side in STAGES[stage]:
            outward[side].translation += delta
    return [outward, {side: pose.copy() for side, pose in initial.items()}]


def target_errors(actual, target, stage):
    return {side: {"position_m": float(np.linalg.norm(actual[side].translation - target[side].translation)),
                   "rotation_rad": float(np.linalg.norm(pin.log3(actual[side].rotation.T @ target[side].rotation)))}
            for side in ("left", "right") if side in STAGES[stage]}


def strict_quest_reasons(report):
    from ..quest.link_policy import link_policy
    return link_policy().preflight_reasons(report)


class QuestInput:
    def __init__(self, bind_ip, port, *, pose_rate="50", link_policy="strict", stream_config=None):
        from ..quest.bridge import QuestBridge
        from ..quest.link_policy import link_policy as get_policy
        from ..quest.mapping import QuestTargetMapper

        self._link_policy = get_policy(link_policy, stream_config)
        self.bridge = QuestBridge(bind_ip=bind_ip, port=port, transport="webrtc", pose_rate=pose_rate,
                                  hardware_control=True, stream_config=self.policy.stream_config)
        self.mapper = QuestTargetMapper(translation_scale=1.0, rotation_scale=1.0,
                                        stale_after_ms=self.policy.max_age_ms)
        self.generation = None
        self.widths = {"left": 0.0, "right": 0.0}  # Gripper outputs are intentionally ignored.
        self.stream = None
        if self.policy.stream_config is not None:
            from ..quest.target_source import QuestTargetSource
            self.stream = QuestTargetSource(self.bridge.mailbox, self.mapper, self.policy.stream_config)

    @property
    def policy(self):
        from ..quest.link_policy import link_policy
        return getattr(self, '_link_policy', link_policy())

    def preflight(self, timeout=120.0, *, robot_holding=False, window_seconds=10.5):
        self.bridge.start()
        print(f"Quest: {self.bridge.url}  PIN: {self.bridge.pin}; enter XR, release both squeezes, keep still.")
        start = time.monotonic()
        window_start = None
        generation = None
        last_status = 0.0
        sync_started = None
        self.last_preflight_report = None
        while timeout is None or time.monotonic() - start < timeout:
            if self.bridge.finish_requested:
                raise HardwareSafetyError("Quest finish requested before arming")
            sample = self.bridge.latest()
            now = time.monotonic_ns()
            if time.monotonic() - last_status >= 1:
                elapsed = 0 if window_start is None else time.monotonic() - window_start
                label = "Synchronizing clocks" if self.policy.include_clock_margin and window_start is None else "Preflight in progress"
                self.bridge.send_status("preflight", f"{label} / {min(elapsed, window_seconds):.0f}/{window_seconds:g} s. " +
                                        ("Robot control acquired and holding; not following yet." if robot_holding else "Not enabled yet.") +
                                        "Keep both controllers visible and still, with both squeeze buttons released.")
                last_status = time.monotonic()
            if not self.bridge.connected or sample is None or generation != self.bridge.connection_generation:
                window_start = None
                sync_started = None
                generation = self.bridge.connection_generation
                self.bridge.monitor.reset()
            else:
                if self.policy.include_clock_margin and window_start is None:
                    estimate = self.bridge.clock.estimate()
                    config = self.policy.stream_config
                    qualified = (estimate is not None and estimate.exchanges >= (8 if config is None else config.sync_exchanges)
                                 and np.isfinite(estimate.uncertainty_ms)
                                 and 0 <= estimate.uncertainty_ms <= self.policy.max_clock_uncertainty_ms
                                 and self.policy.timing_reason(sample, now) is None)
                    if not qualified:
                        sync_started = None
                    elif sync_started is None:
                        sync_started = time.monotonic()
                    if sync_started is None or time.monotonic() - sync_started < (1. if config is None else config.sync_stable_ms / 1000):
                        time.sleep(.02)
                        continue
                    self.bridge.monitor.reset()
                if window_start is None:
                    window_start = time.monotonic()
                self.bridge.monitor.record_ik_age(timestamp_ns=now, age_ms=sample.age_ms(now),
                                                 effective_age_ms=self.policy.effective_age(sample, now))
                if time.monotonic() - window_start >= window_seconds:
                    config = self.policy.stream_config
                    report = self.bridge.monitor.report(window_s=window_seconds - .5, require_static=True,
                        **({} if config is None else {'minimum_receive_hz': config.minimum_receive_hz}))
                    reasons = (self.policy.preflight_reasons(report) if self.policy.include_clock_margin
                               else strict_quest_reasons(report))
                    if self.policy.include_clock_margin:
                        latest_reason = self.policy.timing_reason(sample, now)
                        if latest_reason:
                            reasons += (latest_reason,)
                    if any(p.squeeze > 0.2 for p in sample.controllers.values()):
                        reasons += ("release_both_squeezes",)
                    displayed = jsonable(report)
                    displayed.update(monitor_passed=displayed.get("passed"),
                                     passed=not reasons, reasons=list(reasons), scope="hardware",
                                     link_policy=self.policy.report())
                    self.last_preflight_report = displayed
                    print("Quest hardware preflight:", json.dumps(displayed))
                    if reasons:
                        if getattr(self, 'hold_resume', False):
                            self.bridge.send_status("preflight", "Waiting for input quality to recover: " + ", ".join(reasons))
                            window_start = None
                            sync_started = None
                            self.bridge.monitor.reset()
                            continue
                        self.bridge.send_status("failed", "Quest preflight failed; following not started: " + ", ".join(reasons))
                        raise HardwareSafetyError("Quest hardware link not qualified: " + ", ".join(reasons))
                    self.generation = generation
                    self.bridge.send_status("preflight_passed", "PREFLIGHT PASSED. "
                                            "Keep controllers still and wait for the ready prompt.")
                    print("Quest PREFLIGHT PASSED; waiting for the robot to be ready.", flush=True)
                    return displayed if self.policy.include_clock_margin else report
            time.sleep(0.02)
        self.bridge.send_status("failed", "Quest preflight timed out; following not started.")
        raise HardwareSafetyError("Quest preflight timed out; tracking not started")

    def reset(self, poses, *, require_release=True):
        if getattr(self, 'stream', None) is not None:
            self.stream.reset({k: p.homogeneous for k, p in poses.items()}, self.widths,
                              require_release=require_release)
            return
        self.mapper.reset({k: p.homogeneous for k, p in poses.items()}, self.widths, require_release=require_release)

    def suspend_targets(self):
        if getattr(self, 'stream', None) is not None:
            self.stream.suspend()

    def _stream_call(self, method, *args):
        from ..quest.target_source import SourceFault, SourceTimeout
        try:
            return method(*args)
        except SourceTimeout as error:
            raise ControlDelay(str(error)) from error
        except SourceFault as error:
            raise HardwareSafetyError(str(error)) from error

    def check_fresh(self):
        from .x7 import ControlDelay
        sample = self.bridge.latest()
        if not self.bridge.connected or self.bridge.connection_generation != self.generation:
            raise HardwareSafetyError("Quest disconnected/reconnected; new preflight and rearm required")
        if getattr(self, 'stream', None) is not None:
            return self._stream_call(self.stream.poll, time.monotonic_ns())
        if sample is None:
            raise ControlDelay("waiting for Quest input")
        reason = self.policy.timing_reason(sample, time.monotonic_ns())
        if reason:
            raise (HardwareSafetyError if reason == "invalid Quest clock" else ControlDelay)(reason)
        from ..quest.protocol import SESSION_FLAG_VISIBLE
        if not sample.session_flags & SESSION_FLAG_VISIBLE:
            raise HardwareSafetyError("Quest session is not visible")
        previous_epoch = self.mapper._reference_epoch
        if previous_epoch is not None and sample.reference_epoch != previous_epoch:
            raise HardwareSafetyError("Quest reference space changed")
        from ..quest.protocol import ControllerPose
        for side in ("left", "right"):
            pose = sample.controllers.get(side, ControllerPose.missing())
            fault = self.mapper._tracking_fault(pose)
            if fault:
                raise HardwareSafetyError(f"Quest tracking fault: {fault}")
        return sample

    def targets(self, poses):
        if getattr(self, 'stream', None) is not None:
            self.check_fresh()
            result, active, sample = self._stream_call(self.stream.targets,
                {k: p.homogeneous for k, p in poses.items()}, self.widths, time.monotonic_ns())
            self.target_sample = self.stream.target_sample
            return {k: pin.SE3(v.transform) for k, v in result.items()}, active, sample
        hold_resume = getattr(self, 'hold_resume', False)
        sample = self.check_fresh() if hold_resume else self.bridge.latest()
        now = time.monotonic_ns()
        if not self.bridge.connected or self.bridge.connection_generation != self.generation:
            raise HardwareSafetyError("Quest disconnected/reconnected; new preflight and rearm required")
        reason = "waiting for Quest input" if sample is None else self.policy.timing_reason(sample, now)
        if reason:
            raise (ControlDelay if hold_resume and reason != "invalid Quest clock" else HardwareSafetyError)(reason)
        result = self.mapper.update(sample, now_ns=now, measured_tcp={k: p.homogeneous for k, p in poses.items()},
                                    gripper_widths=self.widths)
        if any(result.faults.values()):
            raise HardwareSafetyError(f"Quest tracking/reference fault: {result.faults}")
        return {k: pin.SE3(v.transform) for k, v in result.targets.items()}, result.active, sample

    def close(self):
        self.bridge.stop()


def runtime_readonly_diagnostic(args, mapping, profile):
    return (
        args.mode == "probe" and getattr(args, "diagnostic_transport", False) and mapping.runtime_session
        and args.input == "manual" and args.stage == "whole-body"
        and profile.profile_id == mapping.data["commissioning"]["motion_profile"]
        and profile.sha256 == mapping.data["commissioning"]["profile_sha256"]
        and getattr(args, "manual_test", None) is None
    )


def validate_prerequisite(args, mapping, profile):
    from .direct_quest import validate_scope

    if not runtime_readonly_diagnostic(args, mapping, profile):
        validate_scope(args, mapping, profile)
    if mapping.direct_quest:
        return None  # Explicit experiment skips reports, never hardware authorization.
    if args.mode != "live":
        return
    stages = list(STAGES)
    first_arm = getattr(args, "first_arm", "right")
    if first_arm == "left":
        stages[:2] = ["left-arm", "right-arm"]
    test = getattr(args, "manual_test", None)
    commissioning = mapping.data.get("commissioning")
    experiment = mapping.quest_experiment
    if experiment:
        if (first_arm != "left" or not (
            args.stage == "left-arm" and args.input == "manual" and test in {"hold", "nudge"}
            or args.stage == "whole-body" and args.input == "quest" and test is None
        )):
            raise HardwareSafetyError("experiment permits only left hold/nudge then whole-body Quest")
        if test == "nudge" and getattr(args, "manual_distance_m", 0.01) != 0.05:
            raise HardwareSafetyError("experiment requires a 0.05 m left-arm nudge")
    elif commissioning and (args.stage not in commissioning["allowed_live_stages"]
                          or first_arm != commissioning["first_arm"]
                          or test not in {"hold", "nudge"} or args.input != "manual"):
        raise HardwareSafetyError("commissioning requires approved first-arm, stage and manual hold/nudge test")
    expected = args.stage if args.input == "quest" else (stages[stages.index(args.stage) - 1]
                                                         if stages.index(args.stage) else None)
    requires_hold = args.input == "manual" and test == "nudge"
    if requires_hold:
        expected = args.stage
    if experiment and args.input == "quest":
        expected = "left-arm"
    if expected is None:
        return
    if args.previous_report is None:
        raise HardwareSafetyError(f"live {args.input}/{args.stage} requires --previous-report for {expected}")
    report = json.loads(args.previous_report.read_text())
    if not passed_manual_report(report, mapping, profile, expected, first_arm, requires_hold):
        raise HardwareSafetyError("prerequisite report did not pass the matching hardware/profile/stage checks")
    if experiment and args.input == "quest":
        preview = report.get("manual_path_preview")
        if (report.get("manual_test") != "nudge" or report.get("manual_distance_m") != 0.05
                or report.get("commissioning_sequence") != commissioning["sequence"]
                or not isinstance(preview, dict) or preview.get("passed") is not True
                or not passed_manual_report(report.get("prerequisite_report", {}), mapping, profile,
                                            "left-arm", "left", True)):
            raise HardwareSafetyError("Quest experiment requires a verified 5 cm nudge with its live hold evidence")
    return report


def passed_manual_report(report, mapping, profile, stage, first_arm, requires_hold):
    if not isinstance(report, dict):
        return False
    gateway = report.get("gateway", {})
    return (isinstance(gateway, dict)
            and report.get("mode") == "live" and report.get("input") == "manual"
            and report.get("stage") == stage
            and report.get("first_arm", "right") == first_arm
            and (not requires_hold or report.get("manual_test") == "hold")
            and report.get("hold_passed" if requires_hold else "fixed_targets_passed") is True
            and report.get("operator_motion_checked") is True
            and report.get("profile_sha256") == profile.sha256 and report.get("mapping_sha256") == mapping.sha256
            and report.get("finish_reason") == ("manual_hold_complete" if requires_hold else "manual_targets_complete")
            and not report.get("trace_error") and not gateway.get("fault")
            and not gateway.get("stop_failures") and gateway.get("kind") == "closed")


def smooth_targets(start, goal, elapsed):
    """Three-second minimum-jerk translation; orientation remains unchanged."""
    u = float(np.clip(elapsed / 3., 0., 1.))
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    result = {side: pose.copy() for side, pose in start.items()}
    for side in result:
        result[side].translation = start[side].translation + blend * (goal[side].translation - start[side].translation)
    return result


def preview_manual_path(profile, mapping, measured, stage, distance_m):
    """No SDK: ideal kinematic rollout, never hardware tracking qualification.

    Predicted q is feedback ONLY in this isolated preview, never in the live core.
    Check both three-second legs and one-second endpoint dwell with normal IK
    collision/dynamic checks and the publisher's bounds/locked-joint checks.
    """
    from .whole_body import CommandGuard

    core = PinkIKCore(config=controller_config_for_profile(profile), base_mode="planar",
                      locked_joint_names=locked_names(stage),
                      runtime_position_bounds=mapping.physical_bounds(measured),
                          **mapping.runtime_limits(profile, measured))
    grippers = np.asarray(mapping.data["fixed_gripper_positions"], dtype=float)
    start_time = time.monotonic()
    virtual = dataclasses.replace(measured, q=measured.q.copy(), v=measured.v.copy(),
                                  received={n: start_time for n in measured.received})
    initial = initialize_core(core, virtual, grippers, stationary_reference=True)
    goals = manual_targets(initial, stage, distance_m)
    effort = mapping.data.get("hardware_effort_limits_nm")
    guard = CommandGuard(profile, mapping, stage, virtual,
                         profile.mit_servo.torque_limit_nm if effort is None else effort, stationary_reference=True)
    result = {"kind": "ideal_kinematic_preview", "hardware_qualified": False, "passed": False,
              "manual_distance_m": distance_m, "cycles": 0, "max_arm_excursion_rad": 0.0,
              "hardware_effort_verified": effort is not None, "endpoint_errors": []}
    dt = profile.control_dt_s
    sequence = 0
    try:
        for segment, goal in enumerate(goals):
            origin = initial if segment == 0 else goals[segment - 1]
            for tick in range(1, round(4.0 / dt) + 1):
                now = start_time + sequence * dt
                virtual = dataclasses.replace(virtual, received={n: now for n in virtual.received})
                targets = smooth_targets(origin, goal, tick * dt)
                command = core.step(virtual.q, targets, measured_velocity=virtual.v,
                                    state_timestamp=now, control_timestamp=now,
                                    measured_gripper_positions=grippers)
                if not command.executable or command.mit is None or command.diagnostics.fault:
                    raise HardwareSafetyError(f"preview IK failed: {command.diagnostics.fault}")
                excursion = float(np.max(np.abs(command.positions[4:] - measured.q[4:])))
                result["max_arm_excursion_rad"] = max(result["max_arm_excursion_rad"], excursion)
                guard.validate(WholeBodyCommand(sequence, now, now + COMMAND_TTL, virtual.received,
                                               virtual.q, command.positions, command.velocity, command.mit,
                                               command.base.velocity_body), virtual, now)
                virtual = dataclasses.replace(virtual, q=command.positions.copy(), v=command.velocity.copy())
                sequence += 1
                result["cycles"] = sequence
                if tick * dt >= 3.0:
                    errors = target_errors(tcp_poses(core, virtual.q), goal, stage)
                    if any(e["position_m"] > 0.003 or e["rotation_rad"] > np.deg2rad(2)
                           for e in errors.values()):
                        raise HardwareSafetyError("preview endpoint did not settle within 3 mm / 2 degrees")
            result["endpoint_errors"].append(errors)
        result["passed"] = True
    except (RuntimeError, ValueError) as error:
        result["fault"] = str(error)
    return result


class QuestStartupHold:
    """Require three settled seconds, then rebase with a new squeeze release."""

    def __init__(self, poses, now, q=None):
        self.initial = {s: p.copy() for s, p in poses.items()}
        self.initial_q = None if q is None else q.copy()
        self.started = now
        self.stable_since = None
        self.ready = False

    def update(self, poses, now, q=None):
        if self.ready:
            return True
        errors = target_errors(poses, self.initial, "whole-body")
        settled = all(e["position_m"] <= 0.003 and e["rotation_rad"] <= np.deg2rad(2)
                      for e in errors.values())
        if self.initial_q is not None:
            tolerances = np.r_[.003, .003, np.deg2rad(2), .003, np.full(14, np.deg2rad(2))]
            settled = settled and q is not None and bool(np.all(np.abs(q - self.initial_q) <= tolerances))
        self.stable_since = (now if self.stable_since is None else self.stable_since) if settled else None
        if self.stable_since is not None and now - self.stable_since >= 3.0:
            self.ready = True
        elif now - self.started >= 10.0:
            raise HardwareSafetyError("Quest startup hold did not settle within 10 seconds")
        return self.ready


def _wait_state(gateway, *, fresh=True, timeout=25):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return gateway.state(fresh=fresh)
        except HardwareSafetyError:
            if time.monotonic() >= deadline or gateway.process.exitcode is not None:
                raise
            time.sleep(0.01)


def qualify_feedback(gateway, duration_s=2.0):
    """A stale/skewed frame before acquisition blocks motion, not just logging."""
    from .x7 import ControlDelay
    deadline = time.monotonic() + duration_s
    samples = 0
    while time.monotonic() < deadline:
        try:
            gateway.state().validate(time.monotonic())
        except ControlDelay:
            if getattr(gateway, 'latency_policy', 'strict') != 'hold-resume':
                raise
            deadline = time.monotonic() + duration_s
            samples = 0
        samples += 1
        time.sleep(0.02 if getattr(gateway, 'latency_policy', 'strict') == 'hold-resume' else 0.005)
    return {"continuous_valid_window_s": duration_s, "samples": samples}


from .realtime import realtime_run


@realtime_run
def run(args):
    transport_options = {"theta_transport": getattr(args, "theta_transport", "kcp"),
                         "theta_report_hz": getattr(args, "theta_report_hz", 250)}
    diagnostic_transport = getattr(args, "diagnostic_transport", False)
    transport_experiment = diagnostic_transport or transport_options != {
        "theta_transport": "kcp", "theta_report_hz": 250}
    if transport_experiment and args.mode not in {"probe", "shadow"}:
        raise HardwareSafetyError("Theta transport experiments are read-only")
    candidate_geometry = getattr(args, "allow_candidate_geometry", False)
    if candidate_geometry and args.mode not in {"probe", "shadow"}:
        raise HardwareSafetyError("candidate geometry is read-only; live bypass forbidden")
    manual_test = getattr(args, "manual_test", None)
    distance_m = manual_distance(getattr(args, "manual_distance_m", 0.01))
    if distance_m != 0.01 and (manual_test != "nudge" or args.input != "manual"
                              or args.stage not in {"left-arm", "right-arm"}):
        raise ValueError("custom manual distance requires a single-arm manual nudge")
    if manual_test and (args.input != "manual" or args.stage not in {"left-arm", "right-arm"}):
        raise ValueError("hold/nudge commissioning tests require manual input and a single arm")
    diagnostic_load = getattr(args, "diagnostic_load", "none")
    if diagnostic_load != "none" and args.mode != "probe":
        raise ValueError("diagnostic-load is only available in read-only probe mode")
    profile = load_motion_profile(args.motion_profile)
    mapping = RobotMapping.load(args.robot_config)
    from ..control.first_trial import apply_envelope, envelope_report

    envelope = getattr(args, "motion_envelope", None) or ("first-trial" if mapping.direct_quest else "none")
    mapping = dataclasses.replace(
        mapping, motion_envelope=envelope, latency_policy="strict",
        feedback_policy=getattr(args, "feedback_policy", "strict"),
        allow_unverified_hardware_limits=getattr(args, "allow_unverified_hardware_limits", False))
    mapping.validate_experiment_options()
    profile = apply_envelope(profile, envelope)
    from .direct_quest import hardware_review_report, validate_scope

    runtime_diagnostic = runtime_readonly_diagnostic(args, mapping, profile)
    if not runtime_diagnostic:
        validate_scope(args, mapping, profile)
    if mapping.direct_quest:
        review = hardware_review_report(mapping)
        print("Direct Quest hardware review:", json.dumps(review, indent=2))
        if args.mode == "live" and review["blocking"]:
            output = args.save_trace or Path("runs") / (
                "hardware-ik-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
            writer = TraceWriter(output)
            writer.record({"kind": "hardware_review", "report": review})
            writer.close({"mode": "live", "input": "quest", "stage": "whole-body",
                          "experiment": "direct-quest-mit-v10-v1", "profile_sha256": profile.sha256,
                          "motion_envelope": envelope_report(profile), "feedback_policy": mapping.feedback_policy,
                          "allow_unverified_hardware_limits": mapping.allow_unverified_hardware_limits,
                          "mapping_sha256": mapping.sha256, "hardware_review": review,
                          "hardware_commissioned": False, "finish_reason": "hardware_review_required",
                          "gateway": {"kind": "not_started", "motion_commands_sent": 0}})
            print("Review report:", output / "summary.json")
            raise HardwareSafetyError("hardware review required: " + ", ".join(review["blocking"]))
    if not runtime_diagnostic and profile.profile_id not in {"x7-mit-highspeed-v8", "x7-mit-highspeed-v9"} and not (
        profile.profile_id == "x7-mit-highspeed-v10" and (
            args.mode != "live" or mapping.quest_experiment)
    ):
        raise ValueError("hardware-ik supports v8/v9; live v10 requires the dedicated Quest experiment")
    effort = profile.mit_servo.torque_limit_nm  # Used only offline unless authorize() succeeds.
    if args.mode == "live":
        effort = mapping.authorize(profile, args.stage, args.allow_uncommissioned_profile_sha256)
        if not (args.confirm_workspace_clear and args.confirm_estop_ready and args.confirm_arm_zero_checked):
            raise HardwareSafetyError("live requires workspace, E-stop and arm-zero confirmations")
        if "theta" in STAGES[args.stage] and not args.confirm_theta_clear:
            raise HardwareSafetyError("live lift requires --confirm-theta-clear")
        if "base" in STAGES[args.stage] and not (args.confirm_base_clear and args.confirm_base_untethered):
            raise HardwareSafetyError("live base requires space and cable-clearance confirmations")
        if mapping.direct_quest and not getattr(args, "confirm_charger_disconnected", False):
            raise HardwareSafetyError("direct whole-body Quest requires physical charger-disconnected confirmation")
    prerequisite = validate_prerequisite(args, mapping, profile)
    output = args.save_trace or Path("runs") / (
        "hardware-ik-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    writer = TraceWriter(output)  # Refuse existing paths BEFORE opening any robot connection.
    summary = {"mode": args.mode, "input": args.input, "stage": args.stage, "profile_id": profile.profile_id,
               "profile_sha256": profile.sha256, "mapping_sha256": mapping.sha256,
               "hardware_commissioned": False, "fixed_targets_passed": False, "operator_motion_checked": False,
               "translation_scale": 1.0, "rotation_scale": 1.0,
               "nullspace_reference": ("baseline_posture_blend_from_measured" if profile.posture_policy
                                       else "startup_measured"),
               "gripper_geometry_source": "reviewed fixed configuration; no gripper commands",
               "finish_reason": "not_started", "cycles": 0, "output": str(output.resolve())}
    summary.update(first_arm=getattr(args, "first_arm", "right"), manual_test=manual_test, hold_passed=False)
    summary.update(motion_envelope=envelope_report(profile), feedback_policy=mapping.feedback_policy,
                   allow_unverified_hardware_limits=mapping.allow_unverified_hardware_limits,
                   theta_gap_warning_samples=0, theta_feedback_max_age_ms=0.0)
    summary.update(manual_distance_m=distance_m if manual_test == "nudge" else None,
                   commissioning_sequence=mapping.data.get("commissioning", {}).get("sequence"),
                   prerequisite_report=prerequisite)
    if mapping.direct_quest:
        summary.update(experiment="direct-quest-mit-v10-v1", hardware_review=hardware_review_report(mapping),
                       execution_modes={"left": "MIT", "right": "MIT", "theta": "POSITION", "base": "velocity"},
                       charger_disconnected_confirmed=bool(getattr(args, "confirm_charger_disconnected", False)))
    summary["theta_transport"] = dict(transport_options)
    if diagnostic_transport:
        transport_options["diagnostic_path"] = str((output / "transport.jsonl").resolve())
    gateway = HardwareGateway(mapping, profile, args.stage, args.mode, effort,
                              stationary_reference=args.mode == "live",
                              **({"transport_options": transport_options} if transport_experiment else {}))
    core = None
    quest = None
    gc_was_enabled = gc.isenabled()
    try:
        writer.record({"kind": "configuration", "robot_mapping": mapping.data, "profile": profile.raw})
        gateway.start()
        state = _wait_state(gateway, fresh=args.mode != "probe")
        print("Robot feedback:", json.dumps(jsonable(state), indent=2))
        writer.record({"kind": "initial_state", "state": state})
        if mapping.direct_quest:
            from .direct_quest import check_power

            if runtime_diagnostic:
                # Observe power warnings in a passive diagnostic; never grant
                # control authority or hide them from the report.
                try:
                    summary["power_preflight"] = check_power(state, acquiring=False,
                        ignore_low_battery=True, allow_charging_readonly=True)
                except HardwareSafetyError as error:
                    summary["power_preflight"] = {"error": str(error), "read_only": True}
            else:
                summary["power_preflight"] = check_power(state, acquiring=True)
            if args.mode != "probe":
                from .direct_quest import check_start_position

                check_start_position(state)
        if args.mode == "probe":
            diagnostic_core = None
            if diagnostic_load == "ik":
                diagnostic_core = PinkIKCore(config=controller_config_for_profile(
                                                profile, self_collision_require_approved=not candidate_geometry),
                                            base_mode="planar",
                                            locked_joint_names=locked_names(args.stage),
                                            runtime_position_bounds=mapping.physical_bounds(state),
                          **mapping.runtime_limits(profile, state))
                grippers = np.asarray(mapping.data["fixed_gripper_positions"], dtype=float)
                poses = initialize_core(diagnostic_core, state, grippers)
            summary['measurement_start_monotonic'] = time.monotonic()
            deadline = summary['measurement_start_monotonic'] + args.run_seconds
            timing_rows = []
            detailed = []
            next_tick = time.monotonic()
            violations = 0
            longest = 0.0
            violation_start = None
            while time.monotonic() < deadline:
                state = gateway.state(fresh=False)
                sampled = time.monotonic()
                record_feedback_diagnostics(writer, summary, state, sampled)
                ages = {name: (sampled - stamp) * 1000 for name, stamp in state.received.items()}
                skew = (max(state.received.values()) - min(state.received.values())) * 1000
                if os.environ.get('POLICY_SIM_LIFECYCLE_DIAGNOSTICS') == '1' and max(ages.values()) > 2000:
                    raise HardwareSafetyError('lifecycle measurement has no new feedback for over 2 seconds')
                timing_rows.append((ages, skew))
                stages = dict(state.timing or {})
                stages["schedule_lateness_ms"] = max(0, sampled - next_tick) * 1000
                stages["snapshot_ms"] = (stages["snapshot_completed"] - stages["snapshot_started"]) * 1000
                stages["offer_to_consume_ms"] = (stages["consumed_at"] - stages["offered_at"]) * 1000
                for name, row in state.health.items():
                    stages[f"{name}_capture_ms"] = (row["capture_completed"] - row["received"]) * 1000
                    stages[f"{name}_sdk_dispatch_ms"] = (row["received"] - row["sdk_message_entered"]) * 1000
                valid = True
                try:
                    state.validate(sampled)
                except HardwareSafetyError:
                    valid = False
                if not valid:
                    violations += 1
                    if violation_start is None:
                        violation_start = sampled
                    longest = max(longest, sampled - violation_start)
                else:
                    violation_start = None
                if diagnostic_core is not None:
                    stages["ik_started"] = time.monotonic()
                    try:
                        state.validate(stages["ik_started"])
                        # Each read-only sample is independent: the robot never
                        # executed our last candidate. Do not benchmark a latched
                        # fault's cheap early return as if it were a real solve.
                        diagnostic_core.reset(state.q, poses, measured_velocity=state.v,
                                              state_timestamp=state.timestamp,
                                              measured_gripper_positions=grippers)
                        stages["ik_solve_started"] = time.monotonic()
                        candidate = diagnostic_core.step(
                            state.q, poses, measured_velocity=state.v,
                            state_timestamp=state.timestamp, control_timestamp=stages["ik_started"],
                            measured_gripper_positions=grippers, **state.ik_feedback())
                        stages["ik_solve_ms"] = (time.monotonic() - stages["ik_solve_started"]) * 1000
                        if not candidate.executable or candidate.diagnostics.fault:
                            stages["ik_error"] = str(candidate.diagnostics.fault or "non-executable")
                    except (RuntimeError, ValueError) as error:
                        stages["ik_error"] = str(error)
                    stages["ik_ms"] = (time.monotonic() - stages["ik_started"]) * 1000
                detailed.append(stages)
                writer.record({"kind": "probe", "sampled_at": sampled, "ages_ms": ages,
                               "skew_ms": skew, "state": state, "timing": stages})
                summary["cycles"] += 1
                next_tick += .02
                if next_tick < time.monotonic():
                    next_tick = time.monotonic()
                time.sleep(max(0, next_tick - time.monotonic()))
            summary['measurement_end_monotonic'] = time.monotonic()
            summary["diagnostic_load"] = diagnostic_load
            summary["timing_ms_p50_p95_p99_max"] = {
                key: np.percentile([row[key] for row in detailed if key in row], [50, 95, 99, 100]).tolist()
                for key in sorted({key for row in detailed for key in row if key.endswith("_ms")})
            }
            summary["ik_solve_cycles"] = sum("ik_solve_ms" in row for row in detailed)
            summary["ik_sample_policy"] = "independent_measured_reset_v1"
            summary["feedback_violations"] = violations
            summary["longest_observed_violation_s"] = longest
            summary["ik_error_cycles"] = sum("ik_error" in row for row in detailed)
            summary["duplicate_frames"] = {n: r["duplicate_frames"] for n, r in state.health.items()}
            summary["feedback_age_ms_p50_p95_p99_max"] = {
                name: np.percentile([r[0][name] for r in timing_rows], [50, 95, 99, 100]).tolist()
                for name in state.received
            }
            summary["feedback_skew_ms_p50_p95_p99_max"] = np.percentile(
                [r[1] for r in timing_rows], [50, 95, 99, 100]
            ).tolist()
            summary["feedback_gate_passed"] = violations == 0
            summary["finish_reason"] = "probe_complete_no_control_acquired"
            return summary
        core = PinkIKCore(config=controller_config_for_profile(
                             profile, self_collision_require_approved=not candidate_geometry), base_mode="planar",
                          locked_joint_names=locked_names(args.stage),
                          runtime_position_bounds=mapping.physical_bounds(state),
                          **mapping.runtime_limits(profile, state))
        grippers = np.asarray(mapping.data["fixed_gripper_positions"], dtype=float)
        state = _wait_state(gateway)
        poses = initialize_core(core, state, grippers, stationary_reference=args.mode == "live")
        # Warm native kernels while read-only; reference history is reset again
        # on acquisition. No warm-up command is ever submitted to the SDK.
        for _ in range(3):
            state = _wait_state(gateway)
            poses = initialize_core(core, state, grippers, stationary_reference=args.mode == "live")
            warmup = core.step(state.q, poses, measured_velocity=state.v, state_timestamp=state.timestamp,
                               control_timestamp=time.monotonic(), measured_gripper_positions=grippers,
                               **state.ik_feedback())
            time.sleep(0.02)
        if not warmup.executable or warmup.mit is None or warmup.diagnostics.fault:
            if core.self_collision_system is not None:
                writer.record({"kind": "fault_collision_diagnostics",
                               "pairs": core.self_collision_system.diagnostics(max_pairs=12)})
            raise HardwareSafetyError(f"read-only IK preflight failed: {warmup.diagnostics.fault}")
        preview_q = None
        if manual_test == "nudge" or (mapping.quest_experiment and manual_test == "hold"):
            state = _wait_state(gateway)
            preview_q = state.q.copy()
            preview_distance = 0.05 if manual_test == "hold" else distance_m
            summary["manual_path_preview"] = preview_manual_path(profile, mapping, state, args.stage, preview_distance)
            writer.record({"kind": "manual_path_preview", "report": summary["manual_path_preview"]})
            print("Manual path preview:", json.dumps(jsonable(summary["manual_path_preview"])))
            if not summary["manual_path_preview"]["passed"]:
                raise HardwareSafetyError("manual path preview failed; no control acquired")
        if args.input == "quest":
            quest = QuestInput(args.quest_bind_ip, args.quest_port)
            writer.record({"kind": "quest_preflight", "report": quest.preflight()})
        gc.collect()
        gc.disable()
        if args.mode == "live":
            approved_q = gateway.state().q.copy()
            if preview_q is not None and np.max(np.abs(approved_q - preview_q)) > 0.002:
                raise HardwareSafetyError("robot moved since path preview; start a new session")
            poses = tcp_poses(core, approved_q)
            print(f"LIVE {args.stage}, {profile.profile_id}, exact hash {profile.sha256}.")
            print("Start from measured posture; grippers fixed/empty. Verify the physical robot and E-stop.")
            if args.input == "manual":
                approved_targets = [poses] if manual_test == "hold" else manual_targets(poses, args.stage, distance_m)
                for index, target in enumerate(approved_targets):
                    print(f"Target {index + 1} (startup-base frame):",
                          {k: p.translation.tolist() for k, p in target.items()})
                    prompt = "Approve this segment and SDK startup with physical E-stop ready? Type y: "
                    if input(prompt).strip() != "y":
                        raise HardwareSafetyError("operator cancelled target before acquisition")
            elif input("Acquire robot control for Quest? Type y: ").strip() != "y":
                raise HardwareSafetyError("operator cancelled Quest acquisition")
            print("Checking continuous robot feedback before acquisition (2 seconds)...")
            writer.record({"kind": "feedback_preflight", "report": qualify_feedback(gateway)})
            if quest:
                # Human confirmation may have taken arbitrarily long. Do not
                # acquire on the strength of a now-disconnected XR session.
                quest.targets(tcp_poses(core, gateway.state().q))
            state = gateway.activate()
            summary["acquired_state"] = state
            summary["reference_initialization"] = "stationary"
            if np.max(np.abs(state.q - approved_q)) > 0.002:
                raise HardwareSafetyError("robot moved since target approval; preview and rearm again")
        else:
            state = _wait_state(gateway)
        poses = initialize_core(core, state, grippers, stationary_reference=args.mode == "live")
        initial = {k: v.copy() for k, v in poses.items()}
        commissioning_q = state.q.copy()
        goals = [] if manual_test == "hold" or quest else manual_targets(initial, args.stage, distance_m)
        writer.record({"kind": "manual_targets", "distance_m": summary["manual_distance_m"],
                       "targets": [{k: p.homogeneous for k, p in goal.items()} for goal in goals]})
        segment_origin = initial
        if quest:
            quest.reset(poses)
        quest_hold = QuestStartupHold(poses, time.monotonic(), state.q) if quest and args.mode == "live" else None
        if quest_hold:
            core.request_stop()
            print("Quest: holding measured posture for 3 settled seconds before engagement.")
        stage_start = time.monotonic()
        next_tick = stage_start
        target_index, stable_since, target_start = -1, None, stage_start
        targets = initial
        stopping = False
        stop_started = None
        last_active = {"left": False, "right": False}
        # Collection/initialization belongs outside the realtime deadline.
        stage_start = next_tick = target_start = time.monotonic()
        while True:
            now = time.monotonic()
            if now - next_tick >= 0.02:
                raise HardwareSafetyError("hardware IK scheduling deadline missed by >=20 ms")
            from .control_timing import check_compute_state, dispatch_deadline, select_control_state

            state = select_control_state(gateway.state, next_tick + COMMAND_TTL)
            now = time.monotonic()
            poses = tcp_poses(core, state.q)
            raw_sample = None
            if not stopping and (quest_hold is None or quest_hold.ready) and now - stage_start >= args.run_seconds:
                summary["finish_reason"] = "run_seconds_elapsed"
                stopping, stop_started = True, now
                core.request_stop()
            if quest and not stopping:
                try:
                    if quest.bridge.finish_requested:
                        summary["finish_reason"] = "quest_finish_requested"
                        stopping, stop_started = True, now
                        core.request_stop()
                    else:
                        targets, active, raw_sample = quest.targets(poses)
                        if quest_hold is not None and not quest_hold.ready:
                            targets = quest_hold.initial
                            active = {"left": False, "right": False}
                            if now - quest_hold.started >= 10 and core.control_state != "HOLD":
                                raise HardwareSafetyError("Quest startup brake did not reach HOLD within 10 seconds")
                            if quest_hold.update(poses, now, state.q) and core.control_state == "HOLD":
                                quest.reset(poses)
                                stage_start = now
                                summary["quest_startup_hold_passed"] = True
                                print("Quest hold passed; release and press squeeze to engage full-body IK.")
                            else:
                                # Do not finish the startup hold until the IK brake is also at HOLD.
                                quest_hold.ready = False
                        # Inactive/non-allowlisted arms never receive Quest goals.
                        for side in ("left", "right"):
                            if side not in STAGES[args.stage]:
                                targets[side] = initial[side]
                                active[side] = False
                        if any(last_active.values()) and not any(active.values()):
                            core.request_stop()
                        if any(active.values()) and core.control_state == "HOLD":
                            core.resume_tracking()
                        last_active = dict(active)
                except HardwareSafetyError as error:
                    summary["finish_reason"] = str(error)
                    stopping, stop_started = True, now
                    core.request_stop()
            elif not stopping and args.input == "manual":
                if args.mode == "shadow":
                    # This is counterfactual IK against a stationary measured
                    # robot, NOT a simulated plant or a tracking qualification.
                    phase = (now - stage_start) % 8
                    targets = (initial if manual_test == "hold" else
                               smooth_targets(initial if phase < 4 else goals[0],
                                              goals[0] if phase < 4 else initial, phase % 4))
                    # No plant executed the previous candidate: reset only in
                    # shadow, never feed candidate q back as measured state.
                    core.reset(state.q, poses, measured_velocity=state.v, state_timestamp=state.timestamp,
                               measured_gripper_positions=grippers)
                else:
                    final_target = initial if target_index == -1 else goals[target_index]
                    errors = target_errors(poses, final_target, args.stage)
                    settled = all(e["position_m"] <= 0.003 and e["rotation_rad"] <= np.deg2rad(2)
                                  for e in errors.values())
                    if target_index >= 0 and manual_test and now - target_start < 3:
                        settled = False
                    stable_since = (now if stable_since is None else stable_since) if settled else None
                    dwell = 3.0 if target_index == -1 else 1.0
                    if stable_since is not None and now - stable_since >= dwell:
                        target_index += 1
                        if target_index == len(goals):
                            summary["hold_passed" if manual_test == "hold" else "fixed_targets_passed"] = True
                            summary["finish_reason"] = ("manual_hold_complete" if manual_test == "hold"
                                                        else "manual_targets_complete")
                            stopping, stop_started = True, now
                            core.request_stop()
                        else:
                            segment_origin = initial if target_index == 0 else goals[target_index - 1]
                            targets = goals[target_index]
                            target_start, stable_since = now, None
                    elif now - target_start > 10:
                        summary["finish_reason"] = "manual_target_timeout"
                        stopping, stop_started = True, now
                        core.request_stop()
                    if not stopping and target_index >= 0 and manual_test:
                        targets = smooth_targets(segment_origin, goals[target_index], now - target_start)
            now = time.monotonic()
            check_compute_state(state, now, next_tick + COMMAND_TTL)
            command = core.step(state.q, targets, measured_velocity=state.v, state_timestamp=state.timestamp,
                                target_sample=(None if raw_sample is None or stopping or not any(active.values())
                                               else TargetSample.from_quest(raw_sample)),
                                control_timestamp=now, measured_gripper_positions=grippers, **state.ik_feedback())
            record_feedback_diagnostics(writer, summary, state, now)
            writer.record({"kind": "cycle", "sequence": summary["cycles"], "state": state,
                           "targets": {s: p.homogeneous for s, p in targets.items()}, "quest": raw_sample,
                           "input_latency": (None if raw_sample is None
                                             else raw_sample.latency_breakdown(int(now * 1e9))),
                           "positions": command.positions, "velocity": command.velocity, "mit": command.mit,
                           "mit_effort": core.last_mit_effort,
                           "timings": command.diagnostics.timings, "fault": command.diagnostics.fault,
                           "control_state": core.control_state, "errors": target_errors(poses, targets, args.stage)})
            if not command.executable or command.mit is None or command.diagnostics.fault:
                if core.self_collision_system is not None:
                    writer.record({"kind": "fault_collision_diagnostics",
                                   "pairs": core.self_collision_system.diagnostics(max_pairs=12)})
                raise HardwareSafetyError(f"non-executable IK: {command.diagnostics.fault}")
            excursion_limit = mapping.arm_excursion_limit(args.stage)
            if excursion_limit is not None:
                if (np.max(np.abs(command.positions[4:] - commissioning_q[4:])) > excursion_limit
                        or np.max(np.abs(state.q[4:] - commissioning_q[4:])) > excursion_limit):
                    raise HardwareSafetyError("commissioning arm excursion exceeds 5 degrees")
            writer.record({"kind": "sdk_preview", "sequence": summary["cycles"],
                           "arms": {side: mapping.arm_command(side, command.mit)
                                    for side in ("left", "right") if side in STAGES[args.stage]}})
            if args.mode == "live":
                gateway.publish(WholeBodyCommand(summary["cycles"], now, dispatch_deadline(state, now, next_tick + COMMAND_TTL), state.received,
                                                 state.q, command.positions, command.velocity, command.mit,
                                                 command.base.velocity_body, cycle_started=(state.timing or {}).get("feedback_wait_started", now)))
            summary["cycles"] += 1
            if stopping:
                if core.control_state == "HOLD":
                    break
                if now - stop_started > 2:
                    raise HardwareSafetyError("controlled stop timed out; gateway brake requested")
            next_tick += 0.02
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        summary["finish_reason"] = "keyboard_interrupt_gateway_brake"
    except Exception as error:
        summary["finish_reason"] = f"{type(error).__name__}: {error}"
        failed_state = getattr(gateway, "last_state", None)
        if failed_state is not None:
            writer.record({"kind": "fault_state", "observed_at": time.monotonic(),
                           "reason": summary["finish_reason"], "state": failed_state})
    finally:
        try:
            summary["gateway"] = gateway.close()
        except Exception as error:
            summary["gateway"] = {"kind": "unconfirmed", "stop_failures": [str(error)]}
        if gc_was_enabled:
            gc.enable()
        if quest:
            try:
                quest.close()
            except Exception as error:
                summary["quest_cleanup_error"] = str(error)
        if (summary["finish_reason"] != "manual_targets_complete"
                or summary["gateway"].get("kind") != "closed" or summary["gateway"].get("stop_failures")):
            summary["fixed_targets_passed"] = False
        if (summary["finish_reason"] != "manual_hold_complete" or summary["gateway"].get("kind") != "closed"
                or summary["gateway"].get("stop_failures")):
            summary["hold_passed"] = False
        if (summary["fixed_targets_passed"] or summary["hold_passed"]) and not summary["gateway"].get("stop_failures"):
            try:
                summary["operator_motion_checked"] = input(
                    "Robot stopped. Independently checked direction/displacement and no oscillation? Type y: "
                ).strip() == "y"
            except (EOFError, KeyboardInterrupt):
                summary["operator_motion_checked"] = False
        from .failure_evidence import save_failure_evidence

        save_failure_evidence(writer, summary, core,
                              {"state": getattr(gateway, "last_state", None)} if core is not None and core.last_qp_failure is not None else None)
        writer.close(summary)
        print(json.dumps(jsonable(summary), indent=2))
    return summary


def add_arguments(parser):
    parser.add_argument("--diagnostic-transport", action="store_true",
                        help="probe/shadow only: bounded per-frame Theta transport recording")
    parser.add_argument("--theta-transport", choices=("kcp", "websocket"), default="kcp",
                        help="probe/shadow transport experiment; default keeps existing KCP")
    parser.add_argument("--theta-report-hz", choices=(100, 250), type=int, default=250,
                        help="probe/shadow report frequency only; IK/control periods unchanged")
    parser.add_argument("--mode", choices=("probe", "shadow", "live"), default="probe")
    parser.add_argument("--input", choices=("manual", "quest"), default="manual")
    parser.add_argument("--stage", choices=tuple(STAGES), default="right-arm")
    parser.add_argument("--robot-config", type=Path, default=CONFIG_DIR / "hardware" / "hexfellow_ipv6.json")
    parser.add_argument("--motion-profile", default="x7-mit-highspeed-v10")
    parser.add_argument("--allow-candidate-geometry", "--allow-candidate-collision-policy",
                        dest="allow_candidate_geometry", action="store_true",
                        help="probe/shadow only: inspect candidate collision policy; never enables live")
    parser.add_argument("--save-trace", type=Path)
    parser.add_argument("--run-seconds", type=float, default=60)
    parser.add_argument("--diagnostic-load", choices=("none", "ik"), default="none",
                        help="probe only: add v9 IK computation without publishing commands")
    parser.add_argument("--previous-report", type=Path)
    parser.add_argument("--first-arm", choices=("left", "right"), default="right")
    parser.add_argument("--manual-test", choices=("hold", "nudge"))
    parser.add_argument("--direct-quest-experiment", action="store_true",
                        help="v10 quest/whole-body, including read-only probe/shadow")
    parser.add_argument("--motion-envelope", choices=("none", "first-trial"), default=None,
                        help="direct Quest defaults to first-trial; other workflows unchanged")
    parser.add_argument("--feedback-policy", choices=("strict", "theta-80ms-first-trial"), default="strict")
    parser.add_argument("--allow-unverified-hardware-limits", action="store_true",
                        help="direct first-trial only: accept unknown numeric hardware bounds, not other reviews")
    parser.add_argument("--confirm-charger-disconnected", action="store_true")
    parser.add_argument("--manual-distance-m", type=manual_distance, default=0.01,
                        help="single-arm nudge upward distance in metres, (0, 0.05]; default 0.01")
    parser.add_argument("--allow-uncommissioned-profile-sha256")
    parser.add_argument("--quest-bind-ip", default="auto")
    parser.add_argument("--quest-port", type=int, default=8443)
    for name in ("workspace-clear", "estop-ready", "arm-zero-checked", "theta-clear", "base-clear", "base-untethered"):
        parser.add_argument("--confirm-" + name, action="store_true")


def handle(args, parser):
    try:
        if not np.isfinite(args.run_seconds) or not 0 < args.run_seconds <= 600:
            raise ValueError("run-seconds must be in (0, 600]")
        summary = run(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    good = {"probe_complete_no_control_acquired", "manual_targets_complete", "manual_hold_complete",
            "run_seconds_elapsed",
            "quest_finish_requested"}
    if summary["finish_reason"] not in good or summary.get("gateway", {}).get("stop_failures"):
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="policy-sim hardware-ik", description=__doc__)
    add_arguments(parser)
    handle(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
