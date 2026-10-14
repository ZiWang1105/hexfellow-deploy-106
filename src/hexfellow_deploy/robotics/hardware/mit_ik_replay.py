"""Whole-body mapped-target replay through measured-feedback IK and arm MIT.

Simulation is the default. Shadow never acquires; live uses the existing MIT
hardware authorization, publisher, watchdog and brake/release implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from hexfellow_deploy.robotics.paths import PROJECT_ROOT

import numpy as np
import pinocchio as pin

from ..control.ik_core import PinkIKCore
from ..control.motion_profile import load_motion_profile
from ..control.target_replay import COLLISION_MODES, RecordedTargets, ReplaySequence, collision_config, tcp_poses
from ..paths import CONFIG_DIR
from .gateway import HardwareGateway
from .ik_runner import TraceWriter, _wait_state, initialize_core, jsonable, qualify_feedback
from .whole_body import COMMAND_TTL, RobotMapping, WholeBodyCommand, WholeBodyState
from .x7 import HardwareSafetyError


class SimulationPlant:
    def __init__(self, recording, config, grippers, initial_q=None, *, scene_path=None):
        import mujoco

        from ..control.ik import MaverX4PinkController
        from ..control.robot_model import load_robot_model
        from ..scene.builder import build_tabletop_scene

        spec = load_robot_model(config.robot_model)
        self.mujoco = mujoco
        self.directory = tempfile.TemporaryDirectory(prefix="mit-ik-replay-scene-")
        try:
            scene = Path(scene_path) if scene_path is not None else Path(self.directory.name) / "robot_workspace.xml"
            if scene_path is None:
                build_tabletop_scene(output_path=scene, base_mode="planar", object_ids=(),
                                     include_table=False, include_box=False, include_trashcan=False,
                                     layout_seed=int(recording.summary.get("seed", 0)), appearance_seed=0,
                                     preserve_fixed_inertias=True, base_config=config.base, robot_model=spec)
            self.model = mujoco.MjModel.from_xml_path(str(scene))
            self.data = mujoco.MjData(self.model)
            home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(self.model, self.data, home)
            self.core = MaverX4PinkController(self.model, config=config)
            # Initial conditions only: all subsequent movement comes from mj_step.
            if initial_q is not None:
                q = self.core._validate_configuration_vector(initial_q)
                self.data.qpos[self.core._mj_qpos_indices] = q
            self.data.qvel[:] = 0.
            self.data.qpos[self.core._mj_gripper_qpos_indices] = grippers
            for name, value in zip(spec.auxiliary_joints, grippers, strict=True):
                actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"position_{name}")
                self.data.ctrl[actuator] = value
            mujoco.mj_forward(self.model, self.data)
            self.peak_requested_torque = np.zeros(14)
            self.peak_actual_torque = np.zeros(14)
            self.saturation_steps = np.zeros(14, dtype=int)
            self.saturation_streak = np.zeros(14)
            self.maximum_saturation_streak_s = np.zeros(14)
            self.last_command = None
            from .effort_monitor import EffortLoadMonitor
            from ..control.hardware_capabilities import load_hardware_capabilities
            self.actual_load = EffortLoadMonitor(load_hardware_capabilities().continuous_torque_nm)
        except BaseException:
            self.directory.cleanup()
            raise

    def state(self):
        stamp = float(self.data.time)
        return WholeBodyState(self.core.measured_configuration(self.data), self.core.measured_velocity(self.data),
                              dict.fromkeys(("left", "right", "theta", "base"), stamp),
                              dict.fromkeys(("left", "right", "theta", "base"), round(stamp / .02)), {}, {})

    def publish(self, command):
        self.core.apply_command(self.data, command)
        self.last_command = command
        for _ in range(self.core.control_period_steps):
            self._before_substep()
            self.mujoco.mj_step(self.model, self.data)
            self._observe_effort()
            observer = getattr(self, "physics_observer", None)
            if observer is not None:
                observer(self)

    def _before_substep(self):
        # Identified plant loads (simulation-only calibration) act every substep.
        hook = getattr(self, "substep_hook", None)
        if hook is not None:
            hook(self)

    def _observe_effort(self):
        if self.last_command is None:
            return
        cmd = self.last_command.mit
        q = self.data.qpos[np.asarray(self.core._mj_qpos_indices)[4:]]
        v = self.data.qvel[np.asarray(self.core._mj_qvel_indices)[4:]]
        requested = cmd.kp * (cmd.position - q) + cmd.kd * (cmd.velocity - v) + cmd.torque_ff
        actual = self.data.actuator_force[self.core._mj_upper_actuator_ids[1:]]
        self.actual_load.observe(float(self.data.time), actual)
        self.peak_requested_torque = np.maximum(self.peak_requested_torque, np.abs(requested))
        self.peak_actual_torque = np.maximum(self.peak_actual_torque, np.abs(actual))
        limits = np.asarray(self.core.effective_mit_servo.torque_limit_nm)
        saturated = (np.abs(requested) > limits + 1e-7) | (np.abs(actual) >= limits - 1e-7)
        self.saturation_steps += saturated
        self.saturation_streak = np.where(saturated, self.saturation_streak + self.model.opt.timestep, 0.)
        self.maximum_saturation_streak_s = np.maximum(self.maximum_saturation_streak_s, self.saturation_streak)

    def effort_report(self):
        return dict(actual_load=self.actual_load.report(), peak_requested_nm=self.peak_requested_torque, peak_actual_nm=self.peak_actual_torque,
                    saturated_substeps=self.saturation_steps,
                    maximum_saturation_streak_s=self.maximum_saturation_streak_s,
                    physical_substep_s=self.model.opt.timestep,
                    interpretation="sampled simulation PD+FF and finite motor; not firmware clamp evidence")

    def observe_fault(self):
        # Observe the existing command, without controller resets or hidden brakes.
        start = float(self.data.time)
        violation = 0.
        for _ in range(round(2. / self.model.opt.timestep)):
            self._before_substep()
            self.mujoco.mj_step(self.model, self.data)
            self._observe_effort()
            q = self.core.measured_configuration(self.data)
            violation = max(violation, float(np.max(self.core._physical_lower_position_limit - q)),
                            float(np.max(q - self.core._physical_upper_position_limit)))
        return dict(observation_s=float(self.data.time)-start, commands_sent=0,
                    final_q=self.core.measured_configuration(self.data),
                    final_v=self.core.measured_velocity(self.data), maximum_joint_limit_excess=violation,
                    controlled_stop_passed=False)

    def close(self):
        self.directory.cleanup()


def _warmup(core, state, grippers):
    """Independent stationary candidates; never sent to a plant or gateway."""
    candidate = None
    for _ in range(10):
        targets = initialize_core(core, state, grippers)
        candidate = core.step(state.q, targets, measured_velocity=state.v,
                              measured_gripper_positions=grippers, state_timestamp=state.timestamp,
                              control_timestamp=state.timestamp, **state.ik_feedback())
    if not candidate.executable or candidate.diagnostics.fault:
        raise HardwareSafetyError(f"initial IK preflight: {candidate.diagnostics.fault}")
    initialize_core(core, state, grippers)


def percentiles(values):
    if not values:
        return None
    return dict(zip(("p50", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100])), strict=True))


def summarize(samples):
    valid = [s for s in samples if s["executable"] and not s["fault"]]
    return dict(
        attempted_cycles=len(samples), valid_solves=len(valid), failed_solves=len(samples) - len(valid),
        control_cycle_ms=percentiles([s["control_cycle_ms"] for s in valid]),
        ik_ms=percentiles([s["timings"]["total_ms"] for s in valid]),
        collision_ms=percentiles([s["timings"]["collision_distance_ms"]
                                  + s["timings"]["swept_collision_ms"] for s in valid]),
        deadline_overruns=sum(s["control_cycle_ms"] >= 20. for s in samples),
        phase_timings={phase: percentiles([s["control_cycle_ms"] for s in valid if s["phase"] == phase])
                       for phase in dict.fromkeys(s["phase"] for s in samples)},
        tracking={side: {"position_m": percentiles([s["errors"][side]["position_m"] for s in valid
                                                   if s["phase"] in {"replay", "settle"}]),
                         "rotation_rad": percentiles([s["errors"][side]["rotation_rad"] for s in valid
                                                       if s["phase"] in {"replay", "settle"}])}
                  for side in ("left", "right")},
    )


def initial_from_report(path):
    if path is None:
        return None
    report = json.loads(Path(path).read_text())
    value = report.get("preflight_q", report.get("q", report.get("initial_q")))
    q = np.asarray(value, dtype=float)
    if q.shape != (18,) or not np.isfinite(q).all():
        raise ValueError("initial-state report requires a finite 18D preflight_q, initial_q or q")
    return q


def run(args, *, gateway_factory=HardwareGateway, plant_factory=SimulationPlant):
    writer = TraceWriter(args.save_trace)
    report = dict(mode=args.mode, collision_mode=args.collision_mode,
                  collision_enabled=args.collision_mode == "enabled",
                  hardware_qualified=False, tracking_qualified=False, motion_commands_sent=0, passed=False,
                  finish_reason="not_started", execution_modes={"left": "MIT", "right": "MIT",
                                                                 "theta": "POSITION", "base": "velocity"})
    plant = gateway = sequence = None
    samples = []
    gc_enabled = gc.isenabled()
    try:
        recording = RecordedTargets(args.replay_trace, args.start_tick, args.end_tick,
                                    clock_mode=getattr(args, "replay_clock", "recorded"))
        profile = load_motion_profile(args.motion_profile)
        if args.mode != "sim" and profile.profile_id != "x7-mit-highspeed-v10":
            raise ValueError("MIT replay requires the current v10 control contract")
        config = collision_config(profile, args.collision_mode)
        mapping = RobotMapping.load(args.robot_config)
        if args.mode != "sim" and (not mapping.direct_quest or mapping.data.get("read_only_candidate", False)):
            raise ValueError("MIT replay requires the direct v10 robot mapping")
        if profile.sha256 != mapping.data["commissioning"]["profile_sha256"]:
            raise ValueError("mapping and v10 profile hash differ")
        grippers = np.asarray(mapping.data["fixed_gripper_positions"], dtype=float)
        report.update(recording=recording.metadata, profile_id=profile.profile_id, profile_sha256=profile.sha256,
                      mapping_sha256=mapping.sha256, speed=args.speed, control_dt_s=config.control_dt,
                      collision_status="enabled" if config.self_collision_enabled else "not_checked",
                      nullspace_reference="human_blend" if config.posture_policy is not None else "startup_measured",
                      motion_envelope="none",
                      feedback_policy="strict", grippers="fixed", warmup_cycles=10,
                      simulation_scene="empty_robot_workspace_with_physical_contacts" if args.mode == "sim" else None)
        writer.record(dict(kind="configuration", **report, profile=profile.raw, robot_mapping=mapping.data))
        if args.mode == "sim":
            plant = plant_factory(recording, config, grippers, initial_from_report(args.initial_state_report))
            core, state = plant.core, plant.state()
        else:
            from .direct_quest import check_power, check_start_position, hardware_review_report

            effort = profile.mit_servo.torque_limit_nm
            if args.mode == "live":
                report["hardware_review"] = hardware_review_report(mapping)
                if not args.confirm_onsite_ready:
                    raise HardwareSafetyError("live replay requires --confirm-onsite-ready")
                effort = mapping.authorize(profile, "whole-body", args.allow_uncommissioned_profile_sha256)
            gateway = gateway_factory(mapping, profile, "whole-body", args.mode, effort)
            gateway.start()
            state = _wait_state(gateway)
            report["power_preflight"] = check_power(state, acquiring=True)
            check_start_position(state)
            core = PinkIKCore(config=config, base_mode="planar", runtime_position_bounds=mapping.physical_bounds(state),
                          **mapping.runtime_limits(profile, state))
        report["initial_q"] = state.q.tolist()
        writer.record(dict(kind="initial_state", state=state))
        _warmup(core, state, grippers)
        # Build and validate every preparation endpoint before taking control.
        sequence = ReplaySequence(core, recording, state.q, args.speed)
        if args.mode == "live":
            approved_q = state.q.copy()
            report["feedback_preflight"] = qualify_feedback(gateway)
            state = gateway.state()
            if np.max(abs(state.q - approved_q)) > .002:
                raise HardwareSafetyError("robot moved during preflight; restart replay")
            state = gateway.activate()
            if np.max(abs(state.q - approved_q)) > .002:
                raise HardwareSafetyError("robot moved during acquisition; restart replay")
            initialize_core(core, state, grippers)
            sequence = ReplaySequence(core, recording, state.q, args.speed)
        elif args.mode == "shadow":
            report["counterfactual_only"] = True
        report["target_frame_transform"] = sequence.transform.homogeneous.tolist()
        gc.collect()
        gc.disable()
        epoch = state.timestamp if plant is not None else time.monotonic()
        next_tick = time.monotonic()
        previous_sequences = None
        for tick in range(int(np.ceil(args.run_seconds / config.control_dt))):
            started = time.monotonic()
            if gateway is not None and started - next_tick >= config.control_dt:
                raise HardwareSafetyError("replay scheduling deadline missed by >=20 ms")
            state = plant.state() if plant is not None else gateway.state()
            now = state.timestamp if plant is not None else time.monotonic()
            elapsed = now - epoch
            fresh = previous_sequences is None or all(state.sequences[n] != previous_sequences[n]
                                                      for n in state.sequences)
            if fresh:
                previous_sequences = state.sequences.copy()
            phase = sequence.phase
            targets, posture, posture_velocity = sequence.sample(elapsed, state.q, state.v,
                                                                 fresh=fresh, observe=args.mode != "shadow")
            if sequence.phase == "complete":
                report["finish_reason"] = "replay_complete"
                report["passed"] = True
                break
            if args.mode == "shadow":
                # The stationary real robot never executed the previous candidate.
                # Preserve the target timeline, but do not feed candidate q back.
                initialize_core(core, state, grippers)
                if phase == "stop":
                    core.request_stop()
            command = core.step(state.q, targets, measured_velocity=state.v,
                                measured_gripper_positions=grippers, state_timestamp=state.timestamp,
                                control_timestamp=now, target_sample=sequence.target_sample,
                                posture_target=posture, posture_velocity=posture_velocity,
                                **state.ik_feedback())
            actual = tcp_poses(core, state.q)
            sample = dict(kind="cycle", tick=tick, elapsed_s=elapsed, phase=phase,
                          measured_q=state.q, measured_v=state.v,
                          targets={s: t.homogeneous for s, t in targets.items()},
                          positions=command.positions, velocity=command.velocity, mit=command.mit,
                          mit_effort=core.last_mit_effort,
                          limited_targets={s: t.homogeneous for s,t in core.limited_targets().items()},
                          planner_diagnostics={s:dict(p.diagnostics) for s,p in core._target_planners.items()},
                          executable=command.executable, fault=command.diagnostics.fault,
                          control_state=core.control_state, stop_reason=command.diagnostics.stop_reason,
                          recovery_reason=command.diagnostics.recovery_reason,
                          collision_status=report["collision_status"],
                          collision_passed=(True if config.self_collision_enabled and command.executable else None),
                          timings=dataclasses.asdict(command.diagnostics.timings),
                          errors={s: dict(position_m=float(np.linalg.norm(actual[s].translation - t.translation)),
                                          rotation_rad=float(np.linalg.norm(
                                              pin.log3(actual[s].rotation.T @ t.rotation))))
                                  for s, t in targets.items()},
                          control_cycle_ms=(time.monotonic() - started) * 1000., published=False)
            samples.append(sample)
            if not command.executable or command.mit is None or command.diagnostics.fault:
                writer.record(sample)
                if core.last_qp_failure is not None:
                    np.savez_compressed(writer.directory / "failed_qp.npz", **{
                        "failed_qp_" + k: v for k, v in core.last_qp_failure.arrays.items()})
                raise HardwareSafetyError(f"{phase} tick {tick}: {command.diagnostics.fault}")
            if plant is not None:
                plant.publish(command)
            elif args.mode == "live":
                group = WholeBodyCommand(tick, now, now + COMMAND_TTL, state.received, state.q,
                                         command.positions, command.velocity, command.mit, command.base.velocity_body)
                gateway.publish(group)
                sample["published"] = True
            # Physics integration is separate from the portable controller budget.
            if plant is None:
                sample["control_cycle_ms"] = (time.monotonic() - started) * 1000.
            writer.record(sample)
            if gateway is not None:
                next_tick += config.control_dt
                time.sleep(max(0., next_tick - time.monotonic()))
        else:
            raise HardwareSafetyError("replay exceeded run-seconds")
    except (Exception, KeyboardInterrupt) as error:
        report["finish_reason"] = f"{type(error).__name__}: {error}"
    finally:
        # Stop hardware before any trace flush, GC or simulator teardown.
        if gateway is not None:
            try:
                report["gateway"] = gateway.close()
                report["motion_commands_sent"] = report["gateway"].get("motion_commands_sent", 0)
                if report["gateway"].get("stop_failures"):
                    report["passed"] = False
            except Exception as error:
                report["stop_error"], report["passed"] = str(error), False
        if plant is not None:
            if not report["passed"] and hasattr(plant, "observe_fault"):
                report["fault_observation"] = plant.observe_fault()
            if hasattr(plant, "effort_report"):
                report["physical_effort"] = plant.effort_report()
            plant.close()
        if gc_enabled:
            gc.enable()
        report["metrics"] = summarize(samples)
        if sequence is not None:
            report["phases"] = sequence.events
        writer.close(report)
    return report


def add_arguments(parser):
    parser.add_argument("--mode", choices=("sim", "shadow", "live"), default="sim")
    parser.add_argument("--collision-mode", choices=COLLISION_MODES, default="enabled")
    parser.add_argument("--replay-trace", type=Path, default=PROJECT_ROOT / "artifacts/reference/synthetic-replay")
    parser.add_argument("--start-tick", type=int)
    parser.add_argument("--end-tick", type=int)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--replay-clock", choices=("recorded", "legacy_fixed_20ms"), default="recorded")
    parser.add_argument("--motion-profile", default="x7-mit-highspeed-v10")
    parser.add_argument("--run-seconds", type=float, default=180.)
    parser.add_argument("--robot-config", type=Path, default=CONFIG_DIR / "hardware" / "hexfellow_direct_quest.json")
    parser.add_argument("--initial-state-report", type=Path, help="sim only: initial 18D posture from a JSON report")
    parser.add_argument("--save-trace", type=Path)
    parser.add_argument("--compare-collision-modes", action="store_true",
                        help="sim only: three serial runs of each mode")
    parser.add_argument("--confirm-onsite-ready", action="store_true",
                        help="confirm clear workspace/cables, correct zeros, disconnected charger and E-stop ready")
    parser.add_argument("--allow-uncommissioned-profile-sha256")


def handle(args, parser):
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be in (0, 1]")
    if not np.isfinite(args.run_seconds) or not 0 < args.run_seconds <= 900:
        parser.error("run-seconds must be in (0, 900]")
    if args.mode != "sim" and (args.compare_collision_modes or args.initial_state_report):
        parser.error("comparison and initial-state-report are simulation only")
    if args.save_trace is None:
        args.save_trace = Path("runs") / ("mit-ik-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    if args.compare_collision_modes:
        root = args.save_trace
        root.mkdir(parents=True, exist_ok=False)
        results = []
        for repeat in range(3):
            for mode in COLLISION_MODES:
                run_args = argparse.Namespace(**vars(args))
                run_args.collision_mode, run_args.save_trace = mode, root / f"{mode}-{repeat + 1}"
                result = run(run_args)
                results.append(dict(result, repeat=repeat + 1))
                print(mode, repeat + 1, result["finish_reason"], flush=True)
        report = dict(kind="serial_collision_comparison", runs=results,
                      passed=all(r["passed"] for r in results), hardware_qualified=False)
        (root / "comparison.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    else:
        report = run(args)
    print(json.dumps(jsonable(report), indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="policy-sim mit-ik-replay", description=__doc__)
    add_arguments(parser)
    handle(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
