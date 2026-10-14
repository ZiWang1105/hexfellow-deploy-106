"""Bounded baseline commissioning, separate from normal teleoperation authority.

Default is a read-only preview. Reports never rewrite hardware review flags.
Initialization abort and publisher stall are host-path tests, NOT cable-loss tests.
"""

import argparse
import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pinocchio as pin

from ..control.first_trial import apply_envelope, envelope_report
from ..control.ik_core import PinkIKCore
from ..control.motion_profile import controller_config_for_profile, load_motion_profile
from ..control.posture import NullPose
from ..paths import CONFIG_DIR
from .commission_motion import MOTIONS
from .commission_acceptance import ArmMotionAccuracy, arm_tracking_errors, settling_report
from .direct_quest import check_power, check_start_position
from .gateway import HardwareGateway
from .ik_runner import TraceWriter, _wait_state, initialize_core, target_errors, tcp_poses, jsonable
from .control_timing import check_compute_state, dispatch_deadline, select_control_state
from .whole_body import COMMAND_TTL, JOINTS, STAGES, CommandGuard, RobotMapping, WholeBodyCommand, locked_names
from .x7 import HardwareSafetyError

BODY_TESTS = ("lift-move", "base-x", "base-y", "base-yaw")
ARM_TESTS = ("left-nudge", "right-nudge")
POSTURE_TESTS = (*BODY_TESTS, *ARM_TESTS)
TESTS = (
    "hold",
    "nudge",
    "translation",
    "rotation",
    "initialization-abort",
    "publisher-stall",
    *BODY_TESTS,
    *ARM_TESTS,
    "estop",
    "disconnect",
)


def add_arguments(parser):
    parser.add_argument("--test", choices=TESTS, default="hold")
    parser.add_argument("--stage", choices=tuple(STAGES), default="left-arm")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-onsite-ready", action="store_true")
    parser.add_argument("--robot-config", type=Path, default=CONFIG_DIR / "hardware/hexfellow_direct_quest.json")
    parser.add_argument(
        "--feedback-policy", choices=("strict", "theta-80ms-first-trial", "theta-record-only"), default="strict"
    )
    parser.add_argument("--previous-report", type=Path)
    parser.add_argument("--distance-m", type=float, default=0.005)
    parser.add_argument("--angle-deg", type=float, default=5.0)
    parser.add_argument("--speed-cm-s", type=float, default=1.0)
    parser.add_argument("--save-trace", type=Path)


def configuration(args):
    if (
        not np.isfinite([args.distance_m, args.angle_deg, args.speed_cm_s]).all()
        or not 0 < args.distance_m <= 0.1
        or not 0 < args.angle_deg <= 15
        or not 1 <= args.speed_cm_s <= 15
    ):
        raise ValueError("distance (0, .1] m, angle (0, 15] degrees, speed [1, 15] cm/s required")
    if args.test == "nudge" and args.distance_m > 0.005:
        raise ValueError("nudge is limited to 5 mm; use translation for longer paths")
    if args.execute and not args.confirm_onsite_ready:
        raise HardwareSafetyError(
            "confirm onsite empty tools, clear workspace, correct mapping, charger removed and E-stop ready"
        )
    mapping = dataclasses.replace(
        RobotMapping.load(args.robot_config),
        motion_envelope="first-trial",
        feedback_policy=args.feedback_policy,
        latency_policy="strict",
        commissioning_stage=args.stage,
        initialization_abort_after=(
            (
                "theta-speed"
                if args.stage == "arms-lift"
                else "base-ready"
                if args.stage == "whole-body"
                else f"{STAGES[args.stage][-1]}-ready"
            )
            if args.test == "initialization-abort"
            else None
        ),
    )
    if args.test in BODY_TESTS and args.stage != ("arms-lift" if args.test == "lift-move" else "whole-body"):
        raise HardwareSafetyError("body-axis test requires its exact device stage")
    if args.test in ARM_TESTS and args.stage != args.test.split("-")[0] + "-arm":
        raise HardwareSafetyError("arm nudge requires its exact device stage")
    mapping.validate_experiment_options()
    profile = apply_envelope(load_motion_profile(mapping.data["commissioning"]["motion_profile"]), "first-trial")
    if (
        mapping.data.get("read_only_candidate")
        or not mapping.direct_quest
        or profile.sha256 != mapping.data["commissioning"]["profile_sha256"]
    ):
        raise HardwareSafetyError("commissioning requires the pinned baseline and corrected direct mapping")
    for key in ("mapping_checked", "same_tool_empty", "mit_units_gains_checked"):
        if mapping.data.get("review", {}).get(key) is not True:
            raise HardwareSafetyError(f"commissioning requires {key}")
    if args.execute and args.test != "hold":
        if args.previous_report is None:
            raise HardwareSafetyError("a successful same-stage baseline hold report is required")
        previous = json.loads(args.previous_report.read_text())
        expected = {
            "test": "hold",
            "stage": args.stage,
            "passed": True,
            "execute": True,
            "profile_sha256": profile.sha256,
            "mapping_sha256": mapping.sha256,
            "evidence_version": 2,
            "feedback_policy": mapping.feedback_policy,
            "effective_limits_sha256": envelope_report(profile)["effective_limits_sha256"],
            "tested_devices": sorted(STAGES[args.stage]),
            "stop_confirmed": True,
        }
        outcome = previous.get("gateway", {})
        if (
            outcome.get("kind") != "closed"
            or outcome.get("stop_failures")
            or outcome.get("fault")
            or not outcome.get("motion_commands_sent", 0)
            or not all(
                outcome.get("direct_downlink", {}).get(n, {}).get("counts", {}).get("motion", 0) > 0
                for n in STAGES[args.stage]
            )
            or not all(
                outcome.get("stop_observations", {}).get(n, {}).get("continuous_stop_confirmed")
                for n in STAGES[args.stage]
            )
        ):
            raise HardwareSafetyError("hold evidence lacks transmitted commands and continuous stop confirmation")
        if any(previous.get(k) != v for k, v in expected.items()):
            raise HardwareSafetyError("hold evidence does not match this stage, mapping and baseline")
    return mapping, profile


def runtime_binding(mapping, profile, state):
    """Bind acceptance to SDK limits/protocol as observed, not only YAML caps."""
    value = {
        "limits": mapping.runtime_limits(profile, state),
        "positions": mapping.physical_bounds(state),
        "devices": {
            name: {
                k: v
                for k, v in state.health[name].items()
                if k
                in {
                    "protocol_version",
                    "joint_names",
                    "joint_limits",
                    "series",
                    "range",
                    "pulse_per_meter",
                    "maximum_speed",
                }
            }
            for name in STAGES[mapping.commissioning_stage]
        },
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=lambda v: np.asarray(v).tolist(), allow_nan=False).encode()
    ).hexdigest()


def motion_targets(initial, test, stage, elapsed, distance, angle, speed):
    """Out/dwell/back quintic path. Speed is TCP target speed, not IK frequency."""
    rotation = test == "rotation"
    duration = max(1.0, 1.875 * (np.deg2rad(angle) / 0.15 if rotation else distance / (speed / 100)))
    leg = np.clip(elapsed / duration if elapsed < duration + 1 else 1 - (elapsed - duration - 1) / duration, 0, 1)
    blend = leg**3 * (10 + leg * (-15 + 6 * leg))
    targets = {s: p.copy() for s, p in initial.items()}
    for side in targets:
        if side in STAGES[stage]:
            if rotation:
                targets[side].rotation = initial[side].rotation @ pin.exp3(
                    np.array([0.0, 0.0, blend * np.deg2rad(angle)])
                )
            else:
                targets[side].translation = initial[side].translation + np.array([blend * distance, 0.0, 0.0])
    return targets, elapsed >= 2 * duration + 2


def body_delta(test, initial_q):
    delta = np.zeros(18)
    amplitude = MOTIONS[test].amplitude
    if test in ARM_TESTS:
        delta[5 if test == "left-nudge" else 12] = np.deg2rad(amplitude)
    elif test == "lift-move":
        delta[3] = amplitude / 100
    elif test == "base-yaw":
        delta[2] = np.deg2rad(amplitude)
    elif test in {"base-x", "base-y"}:
        yaw = initial_q[2] + (np.pi / 2 if test == "base-y" else 0)
        delta[:2] = amplitude / 100 * np.array([np.cos(yaw), np.sin(yaw)])
    return delta


def body_target(test, initial_q, elapsed):
    """Quintic out/return legs with visible endpoint dwells."""
    delta = body_delta(test, initial_q)
    motion = MOTIONS[test]
    duration = motion.leg_seconds
    returning = duration + motion.dwell_seconds
    leg = np.clip(elapsed / duration if elapsed <= returning else 1 - (elapsed - returning) / duration, 0, 1)
    blend = leg**3 * (10 + leg * (-15 + 6 * leg))
    derivative = 30 * leg**2 * (1 - leg) ** 2 / duration
    derivative *= 1 if elapsed < duration else -1 if returning < elapsed < returning + duration else 0
    return initial_q + delta * blend, delta * derivative, elapsed >= motion.duration


def make_core(profile, mapping, state, test=None, *, allow_upper_lift_retreat=False):
    locked = locked_names(mapping.commissioning_stage)
    if test in POSTURE_TESTS:
        movable = (
            {5 if test == "left-nudge" else 12}
            if test in ARM_TESTS
            else {3}
            if test == "lift-move"
            else {2}
            if test == "base-yaw"
            else {0, 1}
        )
        locked = tuple(name for i, name in enumerate(JOINTS) if i not in movable)
    return PinkIKCore(
        config=controller_config_for_profile(profile),
        base_mode="planar",
        locked_joint_names=locked,
        allow_upper_lift_retreat=allow_upper_lift_retreat,
        runtime_position_bounds=mapping.physical_bounds(state),
        **mapping.runtime_limits(profile, state),
    )


def initialize_commission_core(core, state, grippers, *, stationary_reference=False):
    targets = initialize_core(core, state, grippers, stationary_reference=stationary_reference)
    # A 5 mm commissioning nudge must not also try to fold hanging arms toward
    # the distant human null pose. This is a test-local measured posture anchor;
    # normal session initialization and all profile weights/gains are unchanged.
    core.set_nominal_pose(NullPose(float(state.q[3]), tuple(state.q[4:11]), tuple(state.q[11:18])))
    core._posture_transition = None
    return targets


from .realtime import offline_work


@offline_work
def preview(args, mapping, profile, state):
    """Ideal feedback is used only in this explicit offline feasibility preview."""
    core = make_core(profile, mapping, state, args.test)
    grippers = np.asarray(mapping.data["fixed_gripper_positions"])
    simulated_state = dataclasses.replace(state, received=dict.fromkeys(state.received, 0.0))
    initial = initialize_commission_core(core, simulated_state, grippers, stationary_reference=True)
    q, v = state.q.copy(), state.v.copy()
    motion = args.test in {"nudge", "translation", "rotation", *POSTURE_TESTS}
    core.request_stop()
    guard = CommandGuard(
        profile,
        mapping,
        args.stage,
        simulated_state,
        mapping.runtime_limits(profile, state)["runtime_effort_limits_nm"],
        stationary_reference=True,
    )
    excursion_limit = mapping.arm_excursion_limit(args.stage)
    for tick in range(2500):
        elapsed = tick * 0.02
        if motion and tick == 150:
            core.resume_tracking()
        targets, done = (
            motion_targets(
                initial, args.test, args.stage, max(0.0, elapsed - 3), args.distance_m, args.angle_deg, args.speed_cm_s
            )
            if motion
            else (initial, tick >= 150)
        )
        posture, posture_v = None, None
        if args.test in POSTURE_TESTS:
            posture, posture_v, done = body_target(args.test, state.q, max(0, elapsed - 3))
            targets = tcp_poses(core, posture)
        command = core.step(
            q,
            targets,
            measured_velocity=v,
            state_timestamp=elapsed,
            control_timestamp=elapsed,
            measured_gripper_positions=grippers,
            posture_target=posture,
            posture_velocity=posture_v,
        )
        if not command.executable or command.mit is None or command.diagnostics.fault:
            raise HardwareSafetyError(f"offline path infeasible: {command.diagnostics.fault}")
        if np.max(np.abs(command.positions[4:] - state.q[4:])) > excursion_limit:
            raise HardwareSafetyError(
                f"offline path exceeds commissioning {np.rad2deg(excursion_limit):g}-degree excursion"
            )
        feedback = dataclasses.replace(simulated_state, q=q, v=v, received=dict.fromkeys(state.received, elapsed))
        guard.validate(
            WholeBodyCommand(
                tick,
                elapsed,
                elapsed + COMMAND_TTL,
                feedback.received,
                q,
                command.positions,
                command.velocity,
                command.mit,
                command.base.velocity_body,
            ),
            feedback,
            elapsed,
        )
        q, v = command.positions.copy(), command.velocity.copy()
        if done:
            return {"passed": True, "kind": "ideal_kinematic_preview", "hardware_qualified": False, "cycles": tick + 1}
    raise HardwareSafetyError("preview exceeded bounded duration")


from .realtime import realtime_run


@realtime_run
def run(args, *, external_event=None, quiet=False):
    if args.execute and args.test in {"estop", "disconnect"} and external_event is None:
        raise HardwareSafetyError("use hardware-verify for guided external-event tests")
    mapping, profile = configuration(args)
    output = args.save_trace or Path("runs") / (
        "hardware-commission-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    writer = TraceWriter(output)
    report = dict(
        evidence_version=2,
        tested_devices=sorted(STAGES[args.stage]),
        effective_limits_sha256=envelope_report(profile)["effective_limits_sha256"],
        test=args.test,
        stage=args.stage,
        execute=args.execute,
        passed=False,
        profile_sha256=profile.sha256,
        mapping_sha256=mapping.sha256,
        hardware_commissioned=False,
        review_flags_modified=False,
        disconnect_stop_tested=False,
        motion_envelope=envelope_report(profile),
        feedback_policy=mapping.feedback_policy,
        execution_modes={"left": "MIT", "right": "MIT", "theta": "POSITION", "base": "velocity"},
        gripper_commands_sent=0,
        posture_reference="measured_at_acquisition_for_bounded_commissioning",
        cycles=0,
        test_completed=False,
    )
    gateway = None
    core = None
    failure_input = None
    arm_accuracy = None
    if args.test in MOTIONS:
        report["test_motion"] = MOTIONS[args.test].report()
    try:
        effort = mapping.runtime_limits(profile)["runtime_effort_limits_nm"]
        gateway = HardwareGateway(mapping, profile, args.stage, "probe", effort)
        gateway.start()
        state = _wait_state(gateway)
        check_power(state, acquiring=True)
        check_start_position(state)
        if np.max(np.abs(state.v)) > 0.01:
            raise HardwareSafetyError("commissioning requires a stationary robot")
        report["runtime_binding_sha256"] = runtime_binding(mapping, profile, state)
        from .verification import device_binding

        report["device_binding"] = device_binding(state)
        if args.execute and args.test != "hold":
            previous = json.loads(args.previous_report.read_text())
            if previous.get("runtime_binding_sha256") != report["runtime_binding_sha256"]:
                raise HardwareSafetyError("hold evidence SDK limits/protocol differ from current hardware")
        writer.record({"kind": "initial_state", "state": state})
        report["preview"] = preview(args, mapping, profile, state)
        passive = gateway.close()
        gateway = None
        report["readonly_gateway"] = passive
        if passive.get("stop_failures") or passive.get("motion_commands_sent"):
            raise HardwareSafetyError("read-only preview did not close cleanly")
        if not args.execute:
            report["test_completed"] = True
        else:
            gateway = HardwareGateway(mapping, profile, args.stage, "live", effort, stationary_reference=True)
            gateway.start()
            current = _wait_state(gateway)
            if runtime_binding(mapping, profile, current) != report["runtime_binding_sha256"]:
                raise HardwareSafetyError("hardware limits/protocol changed after preview")
            if np.max(np.abs(current.q - state.q)) > 0.002:
                raise HardwareSafetyError("robot moved after preview; no control acquired")
            core = make_core(profile, mapping, current, args.test)
            grippers = np.asarray(mapping.data["fixed_gripper_positions"])
            initialize_commission_core(core, current, grippers, stationary_reference=True)
            if args.test == "initialization-abort":
                gateway.activate_event.set()
                deadline = time.monotonic() + 12
                while gateway.process.is_alive() and time.monotonic() < deadline:
                    gateway.poll_status(raise_fault=False)
                    time.sleep(0.01)
                if gateway.process.is_alive():
                    raise HardwareSafetyError("initialization cancellation timed out")
                report["test_completed"] = True
            else:
                current = gateway.activate()
                report["acquired_state"] = current
                report["reference_initialization"] = "stationary"
                initial = initialize_commission_core(core, current, grippers, stationary_reference=True)
                report["initial_reference"] = {"q": current.q.copy(), "v": core._last_velocity.copy(),
                                               "a": core._last_acceleration.copy()}
                q0 = current.q.copy()
                if args.test in ARM_TESTS:
                    motion = MOTIONS[args.test]
                    arm_accuracy = ArmMotionAccuracy(motion.amplitude, motion.leg_seconds, motion.dwell_seconds)
                    report["arm_motion_accuracy"] = arm_accuracy.report()
                excursion_limit = mapping.arm_excursion_limit(args.stage)
                core.request_stop()
                next_tick = time.monotonic()
                stable_since = None
                observed_at = {}
                stable_observations = dict.fromkeys(STAGES[args.stage], 0.0)
                unsettled_observations = dict.fromkeys(STAGES[args.stage], 0.0)
                moving = False
                motion_started = None
                peak_progress = 0.0
                report["actual_motion_confirmed"] = False
                for tick in range(max(3000, int((getattr(args, "hold_seconds", 3) + 10) * 50))):
                    now = time.monotonic()
                    failure_input = {"kind": "failed_cycle", "sequence": tick, "attempt_started": now,
                                     "slot_deadline": next_tick + COMMAND_TTL}
                    if now - next_tick >= 0.02:
                        raise HardwareSafetyError("commissioning scheduling deadline missed")
                    slot_deadline = next_tick + COMMAND_TTL
                    current = select_control_state(gateway.state, slot_deadline)
                    failure_input["state"] = current
                    now = time.monotonic()
                    poses = tcp_poses(core, current.q)
                    targets, done = (initial, False)
                    if moving:
                        targets, done = motion_targets(
                            initial,
                            args.test,
                            args.stage,
                            now - motion_started,
                            args.distance_m,
                            args.angle_deg,
                            args.speed_cm_s,
                        )
                    posture, posture_v = None, None
                    if args.test in POSTURE_TESTS:
                        posture, posture_v, done = body_target(args.test, q0, now - motion_started if moving else 0)
                        targets = tcp_poses(core, posture)
                        delta = body_delta(args.test, q0)
                        progress = float(np.dot(current.q - q0, delta) / np.dot(delta, delta))
                        peak_progress = max(peak_progress, progress)
                        if progress < -0.3 or progress > 1.5:
                            raise HardwareSafetyError("body-axis motion direction/excursion invalid")
                        moving_indices = np.flatnonzero(abs(delta) > 1e-10)
                        fixed = [i for i in range(18) if i not in moving_indices]
                        if np.any(
                            abs(current.q[fixed] - q0[fixed])
                            > np.array([0.003, 0.003, 0.01, 0.003, *([0.02] * 14)])[fixed]
                        ):
                            raise HardwareSafetyError("inactive joints moved during body-axis test")
                        report["actual_motion_confirmed"] = peak_progress >= 0.5
                        report["peak_body_progress"] = peak_progress
                    elif moving and args.test in {"nudge", "translation"}:
                        progresses = [
                            (poses[s].translation[0] - initial[s].translation[0]) / args.distance_m
                            for s in ("left", "right")
                            if s in STAGES[args.stage]
                        ]
                        peak_progress = max(peak_progress, min(progresses))
                        report["actual_motion_confirmed"] = peak_progress >= 0.5
                        report["peak_tcp_progress"] = peak_progress
                    now = time.monotonic()
                    failure_input = {"kind": "failed_cycle", "sequence": tick, "state": current,
                                     "control_timestamp": now, "targets": {s: p.homogeneous.copy() for s, p in targets.items()},
                                     "posture_target": posture, "posture_velocity": posture_v}
                    check_compute_state(current, now, slot_deadline)
                    command = core.step(
                        current.q,
                        targets,
                        measured_velocity=current.v,
                        state_timestamp=current.timestamp,
                        control_timestamp=now,
                        measured_gripper_positions=grippers,
                        **current.ik_feedback(),
                        posture_target=posture,
                        posture_velocity=posture_v,
                    )
                    failure_input["diagnostics"] = command.diagnostics
                    if not command.executable or command.mit is None or command.diagnostics.fault:
                        raise HardwareSafetyError(f"commissioning IK fault: {command.diagnostics.fault}")
                    if np.max(np.abs(current.q[4:] - q0[4:])) > excursion_limit:
                        raise HardwareSafetyError(
                            f"commissioning measured excursion exceeds {np.rad2deg(excursion_limit):g} degrees"
                        )
                    errors = target_errors(poses, initial, args.stage)
                    settlement = settling_report(
                        errors, current.q, q0, current.v,
                        posture_test=args.test in POSTURE_TESTS, control_state=core.control_state,
                    )
                    settled = settlement["settled"]
                    report["settling"] = settlement
                    arm_tracking = None
                    if arm_accuracy is not None:
                        index = 5 if args.test == "left-nudge" else 12
                        side = args.test.split("-")[0]
                        arm_tracking = arm_tracking_errors(q0, posture, command.positions, current.q, index)
                        if moving:
                            arm_accuracy.observe(
                                now - motion_started, current.received[side],
                                arm_tracking["measured_angle_deg"], current.v[index], now=now,
                            )
                        report["arm_motion_accuracy"] = arm_accuracy.report()
                    # Repeated Theta frames cannot advance arrival or failure clocks.
                    # Gaps themselves are diagnostic; only new non-stationary observations fail a hold.
                    for name in STAGES[args.stage]:
                        stamp = current.received[name]
                        if stamp > observed_at.get(name, -np.inf):
                            observed_at[name] = stamp
                            stable_observations[name] = stable_observations[name] + 0.02 if settled else 0.0
                            unsettled_observations[name] = 0.0 if settled else unsettled_observations[name] + 0.02
                    stable_since = now - min(stable_observations.values()) if settled else None
                    writer.record(
                        {
                            "kind": "cycle",
                            "sequence": tick,
                            "state": current,
                            "mit": command.mit,
                            "positions": command.positions,
                            "timings": command.diagnostics.timings,
                            "targets": {s: p.homogeneous for s, p in targets.items()},
                            "errors": target_errors(poses, targets, args.stage),
                            "control_state": core.control_state,
                            "arm_tracking": arm_tracking,
                            "settling": settlement,
                        }
                    )
                    gateway.publish(
                        WholeBodyCommand(
                            tick,
                            now,
                            dispatch_deadline(current, now, slot_deadline),
                            current.received,
                            current.q,
                            command.positions,
                            command.velocity,
                            command.mit,
                            command.base.velocity_body,
                            cycle_started=(current.timing or {}).get("feedback_wait_started", now),
                        )
                    )
                    failure_input = None
                    report["cycles"] += 1
                    if external_event is not None and external_event.armed.is_set():
                        if now - external_event.armed_at > 45:
                            raise HardwareSafetyError("onsite event not observed within 45 seconds")
                    if (
                        not moving
                        and stable_since is not None
                        and now - stable_since >= (getattr(args, "hold_seconds", 3) if args.test == "hold" else 3)
                        and core.control_state == "HOLD"
                    ):
                        if args.test == "publisher-stall":
                            time.sleep(0.08)  # Intentional bounded foreground stall; SDK watchdog stays independent.
                            gateway.poll_status(raise_fault=False)
                            report["test_completed"] = True
                            break
                        if args.test == "hold":
                            gateway.await_dispatch(tick, slot_deadline)
                            report["test_completed"] = True
                            break
                        if external_event is not None:
                            report["pre_event_hold_confirmed"] = True
                            external_event.arm(current)
                        else:
                            core.resume_tracking()
                            moving, motion_started = True, now
                    elif not moving and max(unsettled_observations.values()) > 10:
                        raise HardwareSafetyError(
                            "hold failed to settle in 10 seconds of new observations: "
                            + "; ".join(settlement["unmet_conditions"])
                        )
                    if moving and done:
                        core.request_stop()
                        if core.control_state == "HOLD" and settled:
                            gateway.await_dispatch(tick, slot_deadline)
                            report["test_completed"] = True
                            if arm_accuracy is not None and not arm_accuracy.passed:
                                raise HardwareSafetyError(arm_accuracy.failure_message())
                            break
                        if now - motion_started > 45:
                            raise HardwareSafetyError(
                                "motion did not settle: " + "; ".join(settlement["unmet_conditions"])
                            )
                    next_tick += 0.02
                    time.sleep(max(0.0, next_tick - time.monotonic()))
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if gateway is not None:
            try:
                report["gateway"] = gateway.close()
            except Exception as error:
                report["gateway"] = {"stop_failures": [str(error)]}
        outcome = report.get("gateway", {})
        if args.execute and args.test == "hold" and args.stage == "whole-body" and getattr(args, "hold_seconds", 3) >= 60:
            latencies = {n: outcome.get("direct_downlink", {}).get(n, {}).get("latency_all_cycles", {})
                         for n in ("left", "right", "theta", "base")}
            from .timing_stats import qualify_performance
            qualification = qualify_performance(latencies, report['cycles'])
            qualified = qualification['performance_passed']
            report["latency_qualification"] = {"passed": qualified, "devices": latencies,
                **qualification,
                "scope": "host UDP submission, not controller receipt or mechanical response"}
            if not qualified:
                report["error"] = report.get("error") or "60-second whole-body latency qualification failed"
        if args.execute:
            observations = outcome.get("stop_observations", {})
            confirmed = all(
                observations.get(n, {}).get("continuous_stop_confirmed")
                and observations.get(n, {}).get("stationary")
                and observations.get(n, {}).get("released_or_braked")
                for n in STAGES[args.stage]
            )
            if args.test == "initialization-abort":
                counts = outcome.get("direct_downlink", {})
                confirmed = (
                    bool(observations)
                    and all(
                        v.get("continuous_stop_confirmed") and v.get("stationary") and v.get("released_or_braked")
                        for v in observations.values()
                    )
                    and mapping.initialization_abort_after in outcome.get("initialization_checkpoints", [])
                    and (
                        mapping.initialization_abort_after == "theta-speed"
                        or any(v.get("counts", {}).get("initialize", 0) > 0 for v in counts.values())
                    )
                )
            if args.test == "publisher-stall":
                reason = str(outcome.get("fault", "")).lower()
                confirmed = confirmed and ("watchdog" in reason or "source lease expired" in reason)
            report["stop_confirmed"] = bool(confirmed)
            report["wire_tested_devices"] = sorted(
                n
                for n, value in outcome.get("direct_downlink", {}).items()
                if value.get("counts", {}).get("motion", 0) > 0
            )
        reason = str(outcome.get("fault", "")).lower()
        expected_fault = (
            args.test == "publisher-stall" and ("watchdog" in reason or "source lease expired" in reason)
        ) or (args.test == "initialization-abort" and "initialization cancelled" in reason)
        report["passed"] = bool(
            report["test_completed"]
            and not report.get("error")
            and not outcome.get("stop_failures")
            and (not outcome.get("fault") or expected_fault)
            and (
                not args.execute
                or (
                    report.get("stop_confirmed")
                    and outcome.get("kind") == "closed"
                    and (
                        args.test == "initialization-abort"
                        or (
                            outcome.get("motion_commands_sent", 0) > 0
                            and report.get("wire_tested_devices") == sorted(STAGES[args.stage])
                        )
                    )
                )
            )
        )
        if args.execute and args.test in {"nudge", *POSTURE_TESTS}:
            report["passed"] = report["passed"] and report.get("actual_motion_confirmed", False)
        if args.execute and args.test in ARM_TESTS:
            report["passed"] = report["passed"] and bool(arm_accuracy is not None and arm_accuracy.passed)
        from .failure_evidence import save_failure_evidence

        save_failure_evidence(writer, report, core, failure_input)
        writer.close(report)
        if not quiet:
            print(json.dumps(jsonable(report), indent=2))
            print("Report:", output / "summary.json")
    return report


def handle(args, parser):
    try:
        if not run(args)["passed"]:
            raise SystemExit(1)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    handle(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
