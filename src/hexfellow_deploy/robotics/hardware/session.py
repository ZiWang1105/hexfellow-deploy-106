"""Persistent, feedback-driven default/init/Quest/replay session.

No hardware is acquired until a motion command passes the existing review.
Preparation adds constraints to the same IK history and MIT publisher.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import queue
import shlex
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

from ..control.feedback_policy import FEEDBACK_POLICIES, coupled_groups
from ..control.ik_core import LIFT_REFERENCE_TOP_RESERVE_M, PinkIKCore
from ..control.motion_profile import controller_config_for_profile, load_motion_profile
from ..control.posture import load_null_pose
from ..control.preparation import ACCELERATION, JERK, velocity_limits
from ..control.robot_model import load_robot_model
from ..control.target_replay import PostureSegment, RecordedTargets, planar_transform, tcp_poses
from ..control.target_timing import TargetSample
from ..paths import CONFIG_DIR, PROJECT_ROOT
from .control_timing import check_compute_state, dispatch_deadline, select_control_state
from .direct_quest import check_power, check_start_position, hardware_review_report
from .gateway import HardwareGateway
from .ik_runner import QuestInput, TraceWriter, _wait_state, initialize_core, jsonable, qualify_feedback
from .whole_body import COMMAND_TTL, CommandGuard, RobotMapping, WholeBodyCommand, WholeBodyState
from .x7 import ControlDelay, HardwareSafetyError
from .research_profiles import HIGHSPEED_PROFILES

# Shared by demo and normal-session arrival checks; this is not a tracking deadband.
TOLERANCE = np.r_[0.003, 0.003, np.deg2rad(2.0), 0.003, np.full(14, 0.02)]
SETTLED_SPEED = np.r_[0.003, 0.003, 0.02, 0.003, np.full(14, 0.02)]
# Replay final-arrival acceptance (TCP position); not a motion guard.
REPLAY_SETTLE_POSITION_M = 0.008


def pose_vector(pose, base):
    return np.r_[base, pose.lift_m, pose.left_arm_rad, pose.right_arm_rad]


def load_configuration(args, *, verification_scope="control"):
    source = RobotMapping.load(args.robot_config)
    mapping = dataclasses.replace(source,
                                  feedback_policy=(args.feedback_policy
                                                   or source.feedback_policy),
                                  latency_policy=(getattr(args, "latency_policy", None)
                                                  or source.latency_policy),
                                  demo_authorized=getattr(args, "demo_authorized", False),
                                  research_release=getattr(args, "research_release", None))
    if not mapping.runtime_session:
        raise ValueError("hardware-session requires its dedicated runtime mapping")
    profile = load_motion_profile(mapping.data["commissioning"]["motion_profile"])
    if mapping.data.get("base_control_enabled", True) != profile.base_control_enabled:
        raise ValueError("runtime/profile base control mismatch")
    if profile.sha256 != mapping.data["commissioning"]["profile_sha256"]:
        raise ValueError("runtime profile hash mismatch")
    reference = mapping.data["default_pose"]
    from ..paths import config_reference
    path = config_reference(args.robot_config, reference["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("default pose hash mismatch; review its captured mapping before rebinding")
    default_pose = load_null_pose(path)
    captured = default_pose.metadata["capture_mapping"]
    if any(mapping.data.get(key) != value for key, value in captured.items()):
        raise ValueError("default pose mapping differs from the captured calibration")
    if load_robot_model(mapping.data.get("robot_model")) != load_robot_model(profile.robot_model):
        raise ValueError("runtime and profile robot models differ")
    null_pose = load_null_pose()
    if getattr(args, "verification_report", None):
        from .verification import permission

        permit = permission(args.verification_report, mapping, verification_scope)
        if verification_scope == "poses":
            from ..control.first_trial import apply_envelope

            profile = apply_envelope(profile, "first-trial")
        mapping = dataclasses.replace(mapping, verification=permit, motion_envelope=profile.motion_envelope)
    if getattr(args, "demo_low_speed", False):
        from ..control.first_trial import apply_envelope
        profile = apply_envelope(profile, "first-trial")
        mapping = dataclasses.replace(mapping, motion_envelope="first-trial")
    mapping.validate_experiment_options()
    return mapping, profile, default_pose, null_pose


def parsed_command(line):
    parts = shlex.split(line)
    if not parts:
        return ("status", None)
    name = parts[0]
    if name == "replay":
        parser = argparse.ArgumentParser(prog="replay", exit_on_error=False)
        parser.add_argument("path", type=Path)
        parser.add_argument("--start-tick", type=int)
        parser.add_argument("--end-tick", type=int)
        parser.add_argument("--speed", type=float, default=1.0)
        args = parser.parse_args(parts[1:])
        if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
            raise ValueError("replay speed must be in (0, 1]")
        return name, (RecordedTargets(args.path, args.start_tick, args.end_tick), args.speed)
    if len(parts) != 1 or name not in {"default", "init", "quest", "stop", "status", "quit", "help", "body-check", "estop", "hold"}:
        raise ValueError("commands: default, init, quest, replay PATH [--speed N], stop, status, quit")
    return name, None


class Arrival:
    """Only new per-device observations advance arrival/failure clocks."""

    def __init__(self, *, theta_progress_timeout=False, tolerance=None):
        self.tolerance = TOLERANCE.copy() if tolerance is None else np.asarray(tolerance).copy()
        self.previous = {}
        self.stable = dict.fromkeys(("left", "right", "theta", "base"), 0.0)
        self.failed = dict.fromkeys(self.stable, 0.0)
        self.theta_progress_timeout = theta_progress_timeout
        self.theta_best_error = None
        self.details = {}

    def update(self, state, target, dt, *, timeout_active=False):
        slices = {"left": slice(4, 11), "right": slice(11, 18), "theta": slice(3, 4), "base": slice(0, 3)}
        for name, part in slices.items():
            stamp = state.received[name]
            if stamp <= self.previous.get(name, -np.inf):
                continue
            interval = stamp - self.previous.get(name, stamp)
            # A newly delivered frame cannot certify the intervening unobserved gap.
            if stamp - self.previous.get(name, stamp) > 0.04 + 1e-9:
                self.stable[name] = 0.0
            self.previous[name] = stamp
            arrived = np.all(abs(state.q[part] - target[part]) <= self.tolerance[part]) and np.all(
                abs(state.v[part]) <= SETTLED_SPEED[part]
            )
            self.stable[name] = self.stable[name] + dt if arrived else 0.0
            if timeout_active:
                error = float(np.max(abs(state.q[part] - target[part])))
                progress_mode = name == 'theta' and self.theta_progress_timeout
                if progress_mode:
                    if self.theta_best_error is None or error <= self.theta_best_error - .0005:
                        self.theta_best_error = error
                        self.failed[name] = 0.0
                    elif arrived:
                        self.failed[name] = 0.0
                    elif 0 < interval <= .04 + 1e-9:
                        self.failed[name] += min(dt, interval)
                else:
                    self.failed[name] = 0.0 if arrived else self.failed[name] + dt
                self.details[name] = dict(target=target[part].tolist(), actual=state.q[part].tolist(),
                    error=error, tolerance=self.tolerance[part].tolist(), no_progress_s=self.failed[name], progress_timeout=progress_mode)
                if self.failed[name] > 10.0:
                    if progress_mode:
                        raise HardwareSafetyError(f'theta made no progress for 10 seconds of new observations; '
                            f'target={target[3]*1000:.2f} mm, actual={state.q[3]*1000:.2f} mm, '
                            f'error={error*1000:.2f} mm, no_progress={self.failed[name]:.2f} s')
                    raise HardwareSafetyError(f"{name} did not settle in 10 seconds of new observations")
        return min(self.stable.values()) >= 0.5


class SessionController:
    def __init__(self, core, state, default_pose, null_pose, *, observe=True, theta_progress_timeout=False, init_arm_tolerance_rad=0.02):
        self.core, self.default_pose, self.null_pose = core, default_pose, null_pose
        self.init_tolerance = TOLERANCE.copy()
        self.init_tolerance[4:] = init_arm_tolerance_rad
        self.observe = observe
        self.theta_progress_timeout = theta_progress_timeout
        self.phase = "hold"
        self.hold_q = state.q.copy()
        self.started = 0.0
        self.segment = None
        self.arrival = self.new_arrival()
        self.ready = False
        self.pending = None
        self.recording = None
        self.speed = 1.0
        self.transform = None
        self.target_sample = None
        self._replay_reanchor_pending = False
        self.events = []
        self.quit_requested = False
        self.stop_phase = "tracking"
        self.idle_control_phase = "tracking"
        self.quest_active = False

    def new_arrival(self):
        return Arrival(theta_progress_timeout=self.theta_progress_timeout,
                       # Replay alignment is the same preparation arrival as init (null-pose family).
                       tolerance=self.init_tolerance if self.phase in {"init", "alignment"} else TOLERANCE)

    @property
    def idle(self):
        return self.phase in {"hold", "ready"}

    def enter(self, phase, now):
        if phase in {"hold", "ready", "quest-preflight"}:
            # Waiting after preparation must retain its base lock and motion
            # limits. Switching to tracking here can create a nonzero base
            # reference which conflicts with the next preparation's hard lock.
            self.idle_control_phase = self.control_phase
        self.phase, self.started = phase, now
        self.arrival = self.new_arrival()
        self.events.append({"phase": phase, "at": now})

    def posture(self, target, state, now, phase):
        target = target.copy()
        target[self.core._locked_indices] = self.core._last_safe_positions[self.core._locked_indices]
        self.core._validate_configuration_vector(target)
        # Generate from the last commanded reference without erasing its history.
        origin = self.core._last_safe_positions.copy()
        proxy = SimpleNamespace(
            model=self.core.model,
            _acceleration_limits=np.minimum(self.core._acceleration_limits, ACCELERATION),
            _jerk_limits=np.minimum(self.core._jerk_limits, JERK),
        )
        self.segment = PostureSegment(origin, target, proxy)
        self.segment.duration = max(self.segment.duration, float(np.max(1.875 * abs(self.segment.delta) /
                        velocity_limits(self.core.config.preparation_lift_velocity_m_s))))
        if self.core.control_state == "HOLD":
            self.core.resume_tracking()
        self.enter(phase, now)

    def command(self, name, payload, state, now):
        if name in {"stop", "quit"}:
            self.ready = False
            self.quit_requested |= name == "quit"
            self.stop_phase = self.control_phase
            self.core.request_stop()
            self.enter("stopping", now)
            self.pending = None
            return
        if not self.idle:
            raise ValueError("motion in progress; use stop before another command")
        if name == "hold":
            self.hold_q = self.core._last_safe_positions.copy()
            self.ready = False
            self.enter("hold", now)
            return
        if name == "body-check":
            if self.core._normal_lift_upper is not None:
                self.ready = False
                self.pending = (name, None)
                retreat = self.core._last_safe_positions.copy()
                retreat[3] = self.core._normal_lift_upper - 0.008
                self.posture(retreat, state, now, "lift-retreat")
                return
            self.body_origin = state.q.copy()
            self.body_peaks = [0., 0., 0.]
            self.ready = False
            if self.core.control_state == "HOLD":
                self.core.resume_tracking()
            self.enter("body-check", now)
            return
        if name in {"quest", "replay"} and (
            not self.ready
            or (
                self.observe
                and not np.all(abs(state.q[3:] - pose_vector(self.null_pose, state.q[:3])[3:]) <= self.init_tolerance[3:])
            )
        ):
            raise ValueError("run init and wait for READY before quest/replay")
        if name in {"default", "init"}:
            self.ready = False
            target = pose_vector(self.default_pose if name == "default" else self.null_pose, state.q[:3])
            if self.core._normal_lift_upper is not None:
                self.pending = (name, target)
                retreat = self.core._last_safe_positions.copy()
                retreat[3] = self.core._normal_lift_upper - 0.008
                self.posture(retreat, state, now, "lift-retreat")
            else:
                self.posture(target, state, now, name)
        elif name == "replay":
            self.recording, self.speed = payload
            self._replay_reanchor_pending = self.core.config.robot_model is not None
            self.transform = planar_transform(state.q) * planar_transform(self.recording.initial_q).inverse()
            aligned = self.recording.initial_q.copy()
            aligned[:3] = state.q[:3]
            self.posture(aligned, state, now, "alignment")
        elif name == "quest":
            # Qualification and an unengaged controller are a real constrained
            # HOLD, not Cartesian tracking with stationary hand targets.
            self.core.request_stop()
            self.enter("quest-preflight", now)
        else:
            raise ValueError(f"unknown motion command {name}")

    @property
    def control_phase(self):
        if self.phase == "lift-retreat" or (
            self.phase in {"hold", "ready"} and self.core._normal_lift_upper is not None
        ):
            return "lift-retreat"
        if self.phase == "stopping":
            return self.stop_phase
        if self.phase in {"hold", "ready", "quest-preflight"}:
            return self.idle_control_phase
        return "preparation" if self.phase in {"default", "init", "alignment"} else "tracking"

    def sample(self, state, now, quest=None):
        dt = self.core.config.control_dt
        self.target_sample = None
        phase = self.phase
        elapsed = now - self.started
        posture, velocity = None, None
        if phase == "body-check":
            from .commission import body_target
            from .commission_motion import MOTIONS
            names = ("base-x", "base-y", "base-yaw")
            if elapsed < 10.:
                return tcp_poses(self.core, self.body_origin), self.body_origin, np.zeros(18)
            t = elapsed - 10.
            index = min(int(t // MOTIONS["base-x"].duration), 2)
            posture, velocity, _ = body_target(names[index], self.body_origin,
                                            t - index * MOTIONS["base-x"].duration)
            delta = state.q[:3] - self.body_origin[:3]
            c, s = np.cos(self.body_origin[2]), np.sin(self.body_origin[2])
            observed = [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]]
            self.body_peaks = np.maximum(self.body_peaks, np.abs(observed)).tolist()
            if t >= sum(MOTIONS[n].duration for n in names):
                if self.arrival.update(state, self.body_origin, dt, timeout_active=True) or not self.observe:
                    if self.observe and np.any(np.asarray(self.body_peaks) < [.05, .05, np.deg2rad(5)]):
                        raise HardwareSafetyError("base movement below half the requested excursion")
                    self.hold_q = self.core._last_safe_positions.copy()
                    self.enter("hold", now)
            return tcp_poses(self.core, posture), posture, velocity
        if phase in {"default", "init", "alignment", "lift-retreat"}:
            posture, velocity = self.segment.sample(elapsed)
            arrived = self.arrival.update(
                state, self.segment.target, dt, timeout_active=elapsed > self.segment.duration
            )
            if elapsed >= self.segment.duration and (arrived or not self.observe):
                self.hold_q = self.core._last_safe_positions.copy()
                if phase == "lift-retreat":
                    if self.observe:
                        self.core.finish_lift_retreat()
                    name, target = self.pending
                    self.pending = None
                    if name == "body-check":
                        self.enter("hold", now + dt)
                        self.command(name, None, state, now + dt)
                    else:
                        self.posture(target, state, now + dt, name)
                elif phase == "alignment":
                    self.enter("replay", now + dt)
                else:
                    self.ready = phase == "init"
                    self.enter("ready" if phase == "init" else "hold", now + dt)
            return tcp_poses(self.core, posture), posture, velocity
        if phase in {"hold", "ready", "quest-preflight"}:
            posture, velocity = self.hold_q, np.zeros(18)
            return tcp_poses(self.core, posture), posture, velocity
        if phase == "quest":
            targets, active, sample = quest.targets(tcp_poses(self.core, state.q))
            if quest.bridge.finish_requested:
                self.command("stop", None, state, now)
            elif any(active.values()):
                if self.core.control_state == "HOLD":
                    self.core.resume_tracking()
                    if self.core.config.robot_model is not None:
                        self.core.reanchor_tracking_tasks(state.q, state.v)
                self.quest_active = True
                self.target_sample = getattr(quest, 'target_sample', None) or TargetSample.from_quest(sample)
            else:
                self.core.request_stop()
                self.quest_active = False
            self.last_quest_sample = sample
            return targets, None, None
        if phase in {"replay", "settle"}:
            if self._replay_reanchor_pending:
                # Preparation bypasses Cartesian tracking. Its old soft-task
                # anchors must not pull back toward the pre-alignment posture.
                # Keep all MIT position/velocity/acceleration history intact.
                self.core.reanchor_tracking_tasks(state.q, state.v)
                self._replay_reanchor_pending = False
            t = elapsed if phase == "replay" else self.recording.duration / self.speed
            self.replay_elapsed_s = t
            targets = self.recording.targets(t, self.speed, self.transform)
            if phase == "replay":
                self.target_sample = self.recording.sample(t, self.speed)
                if elapsed >= self.recording.duration / self.speed:
                    self.enter("settle", now + dt)
            else:
                actual = tcp_poses(self.core, state.q)
                # Acceptance only: measured static error on hardware is ~6-7 mm (J1 friction band).
                arrived = all(
                    np.linalg.norm(actual[s].translation - target.translation) <= REPLAY_SETTLE_POSITION_M
                    and np.linalg.norm(pin.log3(actual[s].rotation.T @ target.rotation)) <= np.deg2rad(2)
                    for s, target in targets.items()
                )
                # New arm observations, not Theta delivery, qualify TCP arrival.
                fresh = all(state.received[s] > self.arrival.previous.get(s, -np.inf) for s in ("left", "right"))
                if fresh:
                    self.arrival.previous = dict(state.received)
                    self.arrival.stable["left"] = self.arrival.stable["left"] + dt if arrived else 0.0
                if self.arrival.stable["left"] >= 0.5 or (not self.observe and elapsed >= 0.5):
                    self.command("stop", None, state, now)
                elif elapsed > 10.0:
                    raise HardwareSafetyError("replay final TCP targets did not settle")
            return targets, None, None
        if phase == "stopping":
            if self.core.control_state == "HOLD":
                self.hold_q = self.core._last_safe_positions.copy()
                self.enter("hold", now)
            elif elapsed > 2.0:
                raise HardwareSafetyError("controlled stop timed out")
            return tcp_poses(self.core, self.core._last_safe_positions), None, None
        raise ValueError(f"unknown session phase {phase}")


def read_commands(mailbox):
    while True:
        try:
            line = input("hexfellow> ")
        except EOFError:
            line = "quit"
        try:
            command = parsed_command(line)  # Parsing/loading stays outside the control thread.
            mailbox.put(command)
            if command[0] == "quit":
                return
        except (ValueError, OSError, SystemExit) as error:
            print(f"Command rejected: {error}", flush=True)


from .realtime import realtime_run


@realtime_run
def run(args, *, gateway_factory=HardwareGateway, verification_scope="control", input_source=None):
    mapping, profile, default_pose, null_pose = load_configuration(args, verification_scope=verification_scope)
    if input_source is not None:
        input_source.validate_session(args, mapping, profile)
    from ..quest.link_policy import link_policy
    quest_input_policy = link_policy(getattr(args, 'quest_link_policy', 'strict'), getattr(args, 'quest_stream_config', None))
    quest_trial = getattr(args, 'quest_trial', False)
    require_quest_release = (getattr(args, 'quest_resume', 'auto') == 'require-release'
                             or quest_input_policy.name == 'buffered')
    if quest_trial:
        from .quest_trial import COMMANDS as trial_commands
        from .verification import software_binding

        if (not mapping.demo_authorized or not require_quest_release
                or software_binding(mapping) != getattr(args, 'trial_binding', None)):
            raise HardwareSafetyError('quest-trial requires its current preflight binding and manual re-engagement')
    if verification_scope != "control":
        from .verification import SESSION_COMMANDS

        if mapping.verification is None:
            raise HardwareSafetyError("verification session requires completed prerequisite reports")
        if verification_scope in SESSION_COMMANDS and args.commands != SESSION_COMMANDS[verification_scope]:
            raise HardwareSafetyError("verification session commands differ from the reviewed step")
    config = controller_config_for_profile(profile)
    arms_only = getattr(args, 'control_group', 'coupled') == 'arms'
    measured_tracking = getattr(args, 'tracking_state', 'reference') == 'measured'
    from ..control.robot import BASE_JOINTS, LIFT_JOINT
    locked_joints = (*BASE_JOINTS, LIFT_JOINT) if arms_only else ()
    period = profile.control_dt_s
    grippers = np.asarray(mapping.data["fixed_gripper_positions"])
    # Scripted recordings are fully loaded before any robot connection.
    script = deque(parsed_command(line) for line in (args.commands or []))
    writer = TraceWriter(args.save_trace)
    writer.drop_on_full = mapping.latency_policy == "hold-resume"
    report = dict(
        mode=args.mode,
        latency_policy=mapping.latency_policy,
        demo_authorized=mapping.demo_authorized,
        passed=False,
        hardware_qualified=False,
        motion_commands_sent=0,
        robot_model=load_robot_model(profile.robot_model).report(),
        base_control_enabled=profile.base_control_enabled,
        ik_variables=14 if arms_only else (18 if profile.base_control_enabled else 15),
        control_group='arms' if arms_only else 'coupled',
        tracking_state='measured' if measured_tracking else 'reference',
        research_trial=profile.profile_id in HIGHSPEED_PROFILES,
        self_collision_enabled=profile.self_collision_enabled,
        profile_sha256=profile.sha256,
        mapping_sha256=mapping.sha256,
        feedback_policy=mapping.feedback_policy,
        feedback_groups=list(coupled_groups(mapping.feedback_policy)),
        quest_pose_rate=args.quest_pose_rate,
        quest_link_policy=getattr(args, 'quest_link_policy', 'strict'),
        quest_input_limits=quest_input_policy.report(),
        quest_resume='require-release' if require_quest_release else 'auto',
        quest_trial=quest_trial,
        replay_skipped_by_operator=quest_trial,
        software_binding=getattr(args, 'trial_binding', None),
        hardware_review=hardware_review_report(mapping),
        default_pose=default_pose.to_dict(),
        null_pose=null_pose.to_dict(),
        theta_gap_warning_samples=0,
        theta_feedback_max_age_ms=0.0,
        cycles=0,
        verification_scope=verification_scope,
        body_resume=getattr(args, 'body_resume', None),
        effective_outer_frame_gain=config.frame_gain,
        mit_planning_effort_fraction=config.mit_planning_effort_fraction,
        lift_reference_max_error_m=config.lift_reference_max_error_m,
        motion_envelope=profile.motion_envelope,
        quest_active_cycles=0,
        quest_release_events=0,
        counterfactual_only=args.mode == "shadow",
    )
    gateway = plant = engine = core = quest = None
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quest-preflight")
    quest_future = None
    inbox = queue.Queue(maxsize=16)
    failure_input = None
    active = False
    simulation_guard = None
    simulation_endpoint = simulation_supervisor = None
    simulation_wall_origin = simulation_time_origin = None
    # Bounded trailing metrics; full per-cycle evidence remains in trace.jsonl.
    metrics = defaultdict(lambda: deque(maxlen=100000))
    wire_sequences = {}
    recovery_epoch = 0
    wire_sequence = 0
    paused_at = None
    pause_cancelled = False
    recovery_log_at = 0.
    expected_estop = False
    estop_armed_at = None
    duration_stop_requested = False
    from .recovery import FreshInputWindow
    quest_fresh = FreshInputWindow(
        stable_s=.5 if quest_input_policy.stream_config is None else quest_input_policy.stream_config.recovery_stable_ms / 1000,
        max_gap_s=.1 if quest_input_policy.stream_config is None else quest_input_policy.stream_config.recovery_max_gap_ms / 1000)
    quest_delay_hold = False
    report["quest_delay_holds"] = 0
    last_quest_status = None
    previous_solve_at = None
    following_slowdown_active = False
    previous_limited = None

    def announce_quest(status, detail):
        nonlocal last_quest_status
        if quest is not None and (status, detail) != last_quest_status:
            try:
                quest.bridge.send_status(status, detail)
            except Exception as error:
                # A display failure must not bypass the independent stop/close path.
                report['quest_status_error'] = str(error)
            last_quest_status = status, detail

    def reanchor_quest(state):
        quest.reset(tcp_poses(core, state.q), require_release=require_quest_release)
        engine.quest_active = False
        if require_quest_release:
            announce_quest('ready', 'Stopped and re-anchored; release both squeeze buttons, then squeeze again to follow.')
        else:
            announce_quest('ready', 'Stopped and re-anchored; squeeze to resume following.')

    def begin_quest_hold(reason, now):
        nonlocal quest_delay_hold
        quest_fresh.reset()
        if not quest_delay_hold:
            core.request_stop()
            engine.quest_active = False
            quest_delay_hold = True
            report['quest_delay_holds'] += 1
            if hasattr(quest, 'suspend_targets'):
                quest.suspend_targets()
            writer.record(dict(kind='source_hold', reason=str(reason), at=now))
            announce_quest('holding', 'Quest input timed out or clock invalid; controlled braking in progress; waiting for stable input.')

    try:
        writer.record(dict(kind="configuration", **report, profile=profile.raw))
        if args.mode == "sim":
            from .mit_ik_replay import SimulationPlant, initial_from_report

            initial = (
                initial_from_report(args.initial_state_report)
                if args.initial_state_report
                else pose_vector(default_pose, np.zeros(3))
            )
            plant = getattr(input_source, 'simulation_plant', None)
            reused_plant = plant is not None
            if plant is None:
                plant = SimulationPlant(SimpleNamespace(summary={}), config, grippers, initial)
            plant_calibration = None
            if getattr(args, "plant_calibration", None) is not None:
                # Identified simulation-only plant correction; never a hardware setting.
                from .plant_calibration import PlantCalibration
                plant_calibration = PlantCalibration.load(args.plant_calibration)
                if not reused_plant:
                    plant_calibration.apply(plant)
                report["plant_calibration"] = plant_calibration.describe()
            state = plant.state()
            # The simulation adapter applies MIT commands; the portable core owns references.
            audit = json.loads((PROJECT_ROOT / "artifacts/reference/initial-state.json").read_text())["state"]
            audited_state = WholeBodyState(**{**audit, "q": np.asarray(audit["q"]), "v": np.asarray(audit["v"])})
            bounds = mapping.physical_bounds(audited_state)
            import mujoco

            from ..control.robot import LIFT_JOINT

            for joint, limits in bounds.items():
                index = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                plant.model.jnt_range[index] = limits
            retreat = not arms_only and state.q[3] > bounds[LIFT_JOINT][1] - 0.002 - LIFT_REFERENCE_TOP_RESERVE_M
            core = PinkIKCore(
                config=config,
                base_mode="planar",
                runtime_position_bounds=bounds,
                allow_upper_lift_retreat=retreat,
                locked_joint_names=locked_joints,
                measured_pose_tracking=measured_tracking,
                **mapping.runtime_limits(profile, audited_state),
            )
            if profile.schema_version in {22, 23}:
                from .simulation_endpoint import SimulationEndpoint
                simulation_endpoint = SimulationEndpoint(
                    plant, mapping, profile,
                    **({} if plant_calibration is None else plant_calibration.endpoint_options()))
                state = simulation_endpoint.state()
        else:
            gateway = gateway_factory(mapping, profile, "whole-body", args.mode, profile.mit_servo.torque_limit_nm,
                                      stationary_reference=args.mode == "live")
            def save_fault_event(event):
                # Small durable evidence is independent of the final SDK report.
                payload = dict(event, estop_armed_at=estop_armed_at)
                temporary = writer.directory / 'fault-event.tmp'
                temporary.write_text(json.dumps(jsonable(payload), allow_nan=False))
                temporary.replace(writer.directory / 'fault-event.json')
            gateway.evidence_callback = save_fault_event
            gateway.start()
            state = _wait_state(gateway, fresh=False)
        report["initial_q"] = state.q.tolist()
        from ..control.first_trial import envelope_report

        report["effective_runtime_limits"] = mapping.runtime_limits(profile, state)
        report["velocity_limit_audit"] = mapping.velocity_audit(profile, state)
        report["effective_profile"] = envelope_report(profile)
        if gateway is not None:
            from .verification import check_devices, device_binding

            report["device_binding"] = device_binding(state)
            check_devices(mapping.verification, state)
            expected_devices = getattr(args, "demo_device_binding", None)
            if expected_devices is not None and device_binding(state) != expected_devices:
                raise HardwareSafetyError("demo device identity/configuration changed")
        writer.record(dict(kind="initial_state", state=state))
        if not args.commands and input_source is None:
            threading.Thread(target=read_commands, args=(inbox,), daemon=True).start()
        print("Window: D default | I init | T follow/stop | Space stop | Q quit" if input_source is not None else
              "Commands: default | init | quest | stop | status | quit" if quest_trial else
              "Commands: default | init | quest | replay PATH [--speed N] | stop | status | quit", flush=True)
        epoch = next_tick = time.monotonic()
        while True:
            try:
                started = time.monotonic()
                now = float(plant.data.time) if plant is not None else started
                if simulation_endpoint is not None and active and not args.commands:
                    simulation_endpoint.advance_to(simulation_time_origin+max(0., started-simulation_wall_origin))
                    now = simulation_endpoint.clock()
                if started - epoch > args.run_seconds and not duration_stop_requested:
                    duration_stop_requested = True
                    inbox.put_nowait(("quit", None))
                if active and gateway is not None and paused_at is None and started - next_tick >= period:
                    writer.record(dict(kind='schedule_miss', reason='scheduling deadline', at=started,
                                       lateness_ms=(started - next_tick) * 1000))
                    if mapping.latency_policy == "hold-resume":
                        gateway.request_hold()
                        paused_at = paused_at or started
                        next_tick = started
                    else:
                        raise HardwareSafetyError("session scheduling deadline exceeded one control period")
                try:
                    state = (
                        (simulation_supervisor.tick() if active else simulation_endpoint.state())
                        if simulation_endpoint is not None else
                        dataclasses.replace(plant.state(), feedback_policy=mapping.feedback_policy)
                        if plant is not None
                        else gateway.state(fresh=active and mapping.latency_policy != "hold-resume")
                    )
                except ControlDelay:
                    if not active or mapping.latency_policy != "hold-resume" or gateway.last_state is None:
                        raise
                    if paused_at is None:
                        paused_at = started
                        gateway.request_hold()
                    state = gateway.last_state  # UI only; paused_at prevents computing from this cache.

                request = None
                try:
                    request = inbox.get_nowait()
                except queue.Empty:
                    if script and (engine is None or engine.idle):
                        request = script.popleft()
                if input_source is not None:
                    source_request = input_source.poll(state, engine, now, paused_at is not None)
                    if request is None or (source_request is not None and source_request[0] in {"stop", "quit"}
                                           and request[0] != "quit"):
                        request = source_request
                if (request is None and engine is not None and engine.phase in {"quest", "quest-preflight"}
                        and quest is not None and quest.bridge.finish_requested):
                    request = ("stop", None)
                if request is not None:
                    name, payload = request
                    if quest_trial and name not in trial_commands:
                        print('Only default/init/quest/stop/quit/status/help are allowed here.', flush=True)
                        if args.commands:
                            raise HardwareSafetyError('command outside Quest trial scope')
                        continue
                    was_cancelled = pause_cancelled
                    if name in {"stop", "quit"}:
                        pause_cancelled = input_source is None or name == "quit"
                        announce_quest('stopping', 'Stopping following; wait until stationary.')
                    elif name in {"default", "init", "hold", "body-check", "drag"}:
                        pause_cancelled = False
                    if name in {"body-check", "estop", "hold"} and not mapping.demo_authorized:
                        raise HardwareSafetyError("demo command requires the live-demo suite")
                    recovering = active and mapping.latency_policy == "hold-resume" and (
                        paused_at is not None or (state.timing or {}).get("recovery", {}).get("phase", "RUNNING") != "RUNNING")
                    if recovering and name in {"stop", "quit"}:
                        pause_cancelled = True
                        if simulation_supervisor is not None:
                            simulation_supervisor.enter_delay('operator hold during recovery')
                        else:
                            gateway.request_hold()
                        engine.pending = None
                        engine.ready = False
                        engine.enter("hold", now)
                        announce_quest('holding', 'Automatic recovery cancelled; enter init to prepare again, or quit to release.')
                        if name == "quit":
                            report["passed"], report["finish_reason"] = True, "operator_quit_during_delay_hold"
                            break
                        if input_source is not None:
                            # Transport may recover into stationary hold; the input
                            # adapter requires a separate, new operator engagement.
                            pause_cancelled = False
                        print("Holding; enter default/init to resume explicitly, or quit to release control.", flush=True)
                        continue
                    if name == "estop":
                        if engine is None or not engine.idle or np.any(abs(state.v) > SETTLED_SPEED):
                            raise HardwareSafetyError("estop observation requires stationary hold")
                        expected_estop = True
                        estop_armed_at = time.monotonic()
                        gateway.command_queue.estop_armed_at.value = estop_armed_at
                        report['estop_armed_at'] = estop_armed_at
                        print("Robot is holding stationary: press and hold the physical E-stop now; wait for the explicit prompt before resetting it.", flush=True)
                        continue
                    if recovering and name not in {"status", "help"}:
                        if was_cancelled and name in {"default", "init"}:
                            pause_cancelled = False
                            engine.command(name, payload, state, now)
                        else:
                            print("Delay hold; waiting for recovery; stop/quit are available.", flush=True)
                        continue
                    if verification_scope == "quest" and name not in {
                        "default",
                        "init",
                        "quest",
                        "stop",
                        "quit",
                        "status",
                        "help",
                    }:
                        print("Only default/init/quest/stop/quit/status are allowed here.", flush=True)
                        continue
                    writer.record(dict(kind="terminal_command", command=name, at=now))
                    if name in {"status", "help"}:
                        print(
                            json.dumps(
                                jsonable(
                                    dict(
                                        phase="readonly" if engine is None else engine.phase,
                                        q=state.q,
                                        review=report["hardware_review"],
                                        quest_source=(None if getattr(quest, 'stream', None) is None
                                                      else quest.stream.diagnostics),
                                    )
                                ),
                                indent=2,
                            ),
                            flush=True,
                        )
                    elif name == "quit" and not active:
                        report["passed"], report["finish_reason"] = True, "readonly_exit"
                        break
                    elif args.mode == "probe":
                        print("Read-only probe: motion commands are disabled.", flush=True)
                        if args.commands:
                            raise HardwareSafetyError("probe cannot execute scripted motion")
                    else:
                        if not active:
                            if name not in {"default", "init", "body-check", "hold"}:
                                print("Start with default or init.", flush=True)
                                if args.commands:
                                    raise HardwareSafetyError("start with default or init")
                                continue
                            if args.mode == "live":
                                if not args.confirm_onsite_ready:
                                    raise HardwareSafetyError("live acquisition requires --confirm-onsite-ready")
                                if quest_trial and software_binding(mapping) != args.trial_binding:
                                    raise HardwareSafetyError('software changed after Quest trial preflight')
                                mapping.authorize(profile, "whole-body", args.allow_uncommissioned_profile_sha256)
                                if input_source is not None:
                                    input_source.validate_session(args, mapping, profile)
                            if gateway is not None:
                                if mapping.latency_policy == "hold-resume":
                                    qualify_feedback(gateway)
                                    state = gateway.state()
                                check_power(state, acquiring=True)
                                check_start_position(state, allow_lift_retreat=True)
                                bounds = mapping.physical_bounds(state)
                                from ..control.robot import LIFT_JOINT

                                retreat = (not arms_only and args.mode == "live" and state.q[3]
                                           > bounds[LIFT_JOINT][1] - 0.002 - LIFT_REFERENCE_TOP_RESERVE_M)
                                core = PinkIKCore(
                                    config=config,
                                    base_mode="planar",
                                    runtime_position_bounds=bounds,
                                    allow_upper_lift_retreat=retreat,
                                    locked_joint_names=locked_joints,
                                    measured_pose_tracking=measured_tracking,
                                    **mapping.runtime_limits(profile, state),
                                )
                            if simulation_endpoint is not None and reused_plant:
                                settled = simulation_endpoint.settle_for_restart()
                                report['simulation_recovery_settle'] = settled
                                if not settled['settled']:
                                    raise HardwareSafetyError('simulation hold has not settled; recovery remains paused')
                                state = simulation_endpoint.state()
                                writer.record(dict(kind='recovery_settled_state', state=state))
                            stationary_start = args.mode == 'live' or simulation_endpoint is not None
                            # Warm the native kernels before acquisition, never under the source watchdog.
                            phase = "lift-retreat" if core._normal_lift_upper is not None else "preparation"
                            warmed = 0
                            while warmed < 10:
                                if gateway is not None and mapping.latency_policy == "hold-resume":
                                    try:
                                        state = gateway.state()
                                    except ControlDelay:
                                        time.sleep(.02)
                                        continue
                                targets = initialize_core(core, state, grippers, stationary_reference=stationary_start)
                                warm = core.step(
                                    state.q,
                                    targets,
                                    measured_velocity=state.v,
                                    measured_gripper_positions=grippers,
                                    state_timestamp=state.timestamp,
                                    control_timestamp=max(state.received.values()),
                                    control_phase=phase,
                                    posture_target=core._last_safe_positions,
                                    **state.ik_feedback(),
                                )
                                if not warm.executable:
                                    if (mapping.latency_policy == "hold-resume"
                                            and str(warm.diagnostics.fault).startswith("ControlDeadlineExceeded:")):
                                        time.sleep(.02)
                                        continue
                                    raise HardwareSafetyError(f"preflight IK: {warm.diagnostics.fault}")
                                warmed += 1
                                if mapping.latency_policy == "hold-resume":
                                    time.sleep(.002)
                            initialize_core(core, state, grippers, stationary_reference=stationary_start)
                            if args.mode == "live":
                                if not args.confirm_onsite_ready:
                                    raise HardwareSafetyError("live acquisition requires --confirm-onsite-ready")
                                mapping.authorize(profile, "whole-body", args.allow_uncommissioned_profile_sha256)
                                if input_source is not None:
                                    input_source.before_acquire(args, mapping, profile)
                                report["feedback_preflight"] = qualify_feedback(gateway)
                                approved = state.q.copy()
                                if input_source is not None:
                                    input_source.check_viewer_ready()
                                state = gateway.activate()
                                report["acquired_state"] = state
                                report["reference_initialization"] = "stationary"
                                check_devices(mapping.verification, state)
                                if input_source is not None and device_binding(state) != args.demo_device_binding:
                                    raise HardwareSafetyError("drag device identity changed during acquisition")
                                if np.max(abs(state.q - approved)) > 0.002:
                                    raise HardwareSafetyError("robot moved during preflight/acquisition")
                                initialize_core(core, state, grippers, stationary_reference=args.mode == "live")
                            if stationary_start:
                                report["initial_reference"] = {"q": state.q.copy(), "v": core._last_velocity.copy(),
                                                               "a": core._last_acceleration.copy(),
                                                               "measured_v": state.v.copy()}
                            controller_type = SessionController if input_source is None else input_source.controller_factory
                            engine = controller_type(core, state, default_pose, null_pose, observe=args.mode != "shadow",
                                                       theta_progress_timeout=mapping.demo_authorized,
                                                       init_arm_tolerance_rad=(np.deg2rad(2.0) if profile.profile_id
                                                           in HIGHSPEED_PROFILES else 0.02))
                            if plant is not None:
                                # Explicit mock bounds for the simulation guard, never hardware evidence.
                                simulation_mapping = mapping
                                mock_limits = mapping.data.get("hardware_base_velocity_limits") is None
                                if mock_limits:
                                    simulation_mapping = dataclasses.replace(
                                        mapping,
                                        data={
                                            **mapping.data,
                                            "hardware_base_velocity_limits": [
                                                2.9,
                                                2.9,
                                                profile.base_angular_velocity_rad_s,
                                            ],
                                        },
                                    )
                                simulation_guard = CommandGuard(
                                    profile,
                                    simulation_mapping,
                                    "whole-body",
                                    state,
                                    core.effective_mit_servo.torque_limit_nm,
                                    stationary_reference=stationary_start,
                                )
                                if simulation_endpoint is not None:
                                    from .gateway import GatewaySupervisor
                                    simulation_supervisor = GatewaySupervisor(
                                        simulation_endpoint, profile, simulation_mapping, "whole-body",
                                        core.effective_mit_servo.torque_limit_nm, clock=simulation_endpoint.clock,
                                        stationary_reference=stationary_start,
                                        sleep=lambda dt: simulation_endpoint.advance_to(simulation_endpoint.clock()+dt))
                                    simulation_supervisor.activate()
                                report["simulation_publisher_checked"] = True
                                report["simulation_base_limits_are_mock"] = mock_limits
                            active = True
                            now = float(plant.data.time) if plant is not None else time.monotonic()
                            epoch = started = next_tick = time.monotonic()
                            if simulation_endpoint is not None:
                                simulation_wall_origin, simulation_time_origin = started, now
                        try:
                            engine.command(name, payload, state, now)
                            if name in {"stop", "quit"} and quest_future is not None and quest is not None:
                                quest.bridge._finish_requested.set()
                            if name == "quest":
                                last_quest_status = None

                                def prepare_quest():
                                    nonlocal quest
                                    if quest is None:
                                        quest = QuestInput(args.bind_ip, args.port, pose_rate=args.quest_pose_rate,
                                                           link_policy=getattr(args, 'quest_link_policy', 'strict'),
                                                           stream_config=quest_input_policy.stream_config)
                                        quest.hold_resume = mapping.latency_policy == "hold-resume"
                                    quest.bridge.clear_finish_request()
                                    return quest.preflight(
                                        timeout=None if mapping.latency_policy == "hold-resume" else 120.,
                                        robot_holding=True)

                                quest_future = worker.submit(prepare_quest)
                        except ValueError as error:
                            if input_source is not None:
                                input_source.message = str(error)
                            writer.record(dict(kind="command_rejected", reason=str(error)))
                            print(str(error), flush=True)
                            if args.commands:
                                raise HardwareSafetyError(str(error)) from error
                if not active:
                    time.sleep(0.02)
                    continue
                if (gateway is not None or simulation_supervisor is not None) and mapping.latency_policy == "hold-resume":
                    recovery = (state.timing or {}).get("recovery", {})
                    phase = recovery.get("phase", "RUNNING")
                    new_epoch = recovery.get("epoch", 0)
                    if phase != "RUNNING":
                        if input_source is not None:
                            input_source.suspend(engine)
                        announce_quest('holding', 'Robot feedback or publishing delayed; holding; follow terminal instructions to re-engage after recovery.')
                        if started - recovery_log_at >= 1.:
                            writer.record(dict(kind="delay_status", recovery=recovery, at=started))
                            recovery_log_at = started
                        paused_at = paused_at or started
                        if quest is not None and hasattr(quest, 'suspend_targets'):
                            quest.suspend_targets()
                        if phase == "RECOVERING" and not pause_cancelled:
                            if engine.phase == "quest" and quest is not None:
                                try:
                                    fresh = quest.check_fresh()
                                except ControlDelay:
                                    quest_fresh.reset()
                                    raise
                                if not quest_fresh.observe(fresh.sequence, started):
                                    time.sleep(.02)
                                    continue
                            if simulation_supervisor is not None:
                                from .recovery import ResumeRequest
                                simulation_supervisor.tick(ResumeRequest(new_epoch))
                            else:
                                gateway.request_resume(new_epoch)
                        if simulation_endpoint is not None and args.commands:
                            simulation_endpoint.advance_to(simulation_endpoint.clock()+period)
                        next_tick = time.monotonic() + (period if simulation_endpoint is not None else 0.)
                        time.sleep(.02)
                        continue
                    if new_epoch != recovery_epoch:
                        seed = (state.timing or {}).get("recovery_seed")
                        if seed is None:
                            raise HardwareSafetyError("missing gateway recovery reference")
                        paused_duration = started - (paused_at or started)
                        initialize_core(core, seed, grippers, stationary_reference=True)
                        engine.hold_q = seed.q.copy()
                        engine.started += paused_duration
                        engine.arrival = engine.new_arrival()
                        if input_source is not None:
                            input_source.recovered(engine, seed, now)
                        if engine.phase in {"default", "init", "alignment", "lift-retreat"}:
                            engine.posture(engine.segment.target.copy(), seed, now, engine.phase)
                        if engine.phase == "quest" and quest is not None:
                            reanchor_quest(seed)
                            quest_delay_hold = False
                            quest_fresh.reset()
                        recovery_epoch, wire_sequence = new_epoch, 0
                        paused_at = None
                        next_tick = time.monotonic()
                        writer.record(dict(kind="delay_resumed", epoch=new_epoch, paused_seconds=paused_duration))
                    elif paused_at is not None:
                        # Parent hold request has not reached the independent gateway yet.
                        next_tick = time.monotonic()
                        time.sleep(.002)
                        continue
                if quest_future is not None and quest_future.done():
                    if engine.phase == "quest-preflight":
                        report["quest_preflight"] = quest_future.result()
                        quest.reset(tcp_poses(core, state.q))
                        engine.enter("quest", now)
                        announce_quest('ready', 'Robot holding, Quest ready; release both squeeze buttons, then squeeze to follow.')
                    quest_future = None
                if args.mode == "shadow":
                    # A counterfactual solve never pretends the previous command moved the robot.
                    initialize_core(core, state, grippers, stationary_reference=args.mode == "live")
                failure_input = {"kind": "failed_cycle", "sequence": report["cycles"],
                                 "attempt_started": time.monotonic(), "slot_deadline": next_tick + min(COMMAND_TTL, period)}
                if gateway is not None:
                    state = select_control_state(gateway.state, next_tick + min(COMMAND_TTL, period),
                        preferred_age_s=.010 if os.environ.get('POLICY_SIM_RESPONSIVE_CPUS') else None)
                    failure_input["state"] = state
                    now = time.monotonic()
                old_phase = engine.phase
                control_phase = engine.control_phase
                was_quest_active = engine.quest_active
                if engine.phase == "quest" and mapping.latency_policy == "hold-resume":
                    try:
                        fresh = quest.check_fresh()
                        if quest_delay_hold and quest_fresh.observe(fresh.sequence, now) and core.control_state == "HOLD":
                            reanchor_quest(state)
                            if not require_quest_release:
                                core.resume_tracking()
                            quest_delay_hold = False
                            quest_fresh.reset()
                    except ControlDelay as source_delay:
                        begin_quest_hold(source_delay, now)
                else:
                    quest_delay_hold = False
                    quest_fresh.reset()
                if quest_delay_hold:
                    engine.target_sample = None
                    targets, posture, velocity = tcp_poses(core, core._last_safe_positions), None, None
                else:
                    try:
                        targets, posture, velocity = engine.sample(state, now, quest)
                    except ControlDelay as source_delay:
                        if old_phase != 'quest' or mapping.latency_policy != 'hold-resume':
                            raise
                        # A timeout can cross its threshold after check_fresh;
                        # it still belongs to source braking, not gateway delay.
                        begin_quest_hold(source_delay, now)
                        engine.target_sample = None
                        targets, posture, velocity = tcp_poses(core, core._last_safe_positions), None, None
                if old_phase == "quest":
                    report["quest_active_cycles"] += int(engine.quest_active)
                    report["quest_release_events"] += int(
                        was_quest_active and not engine.quest_active and not quest_delay_hold)
                    if engine.quest_active and not was_quest_active:
                        announce_quest('live', 'Following; release both squeeze buttons to brake, or click Stop following.')
                    elif was_quest_active and not engine.quest_active and not quest_delay_hold:
                        writer.record(dict(kind='operator_release', reason='operator_release', at=now))
                        announce_quest('holding', 'Squeeze released; braking; squeeze again to engage, or stop to end following.')
                if gateway is not None:
                    now = time.monotonic()
                failure_input = {"kind": "failed_cycle", "state": state, "control_timestamp": now,
                                 "phase": old_phase, "targets": {s: p.homogeneous.copy() for s, p in targets.items()}, "posture_target": posture,
                                 "posture_velocity": velocity, "target_sample": engine.target_sample}
                if gateway is not None:
                    check_compute_state(state, now, next_tick + min(COMMAND_TTL, period))
                command = core.step(
                    state.q,
                    targets,
                    measured_velocity=state.v,
                    measured_gripper_positions=grippers,
                    state_timestamp=state.timestamp,
                    control_timestamp=now,
                    target_sample=engine.target_sample,
                    posture_target=posture,
                    posture_velocity=velocity,
                    control_phase=control_phase,
                    **state.ik_feedback(),
                )
                failure_input["diagnostics"] = command.diagnostics
                following = None if core.following_governor is None else core.following_governor.observation()
                if following is not None:
                    metrics['following_error_rad'].append(following['error_rad'])
                    metrics['following_reference_scale'].append(following['scale'])
                    if following['active'] != following_slowdown_active:
                        following_slowdown_active = following['active']
                        if following_slowdown_active:
                            print('WARNING: tracking lag; slowing reference '
                                  f"(joint gap {np.rad2deg(following['error_rad']):.2f} deg, "
                                  f"projected {np.rad2deg(following['predicted_error_rad']):.2f} deg). "
                                  'Following continues; speed recovers as tracking improves.', flush=True)
                        else:
                            print('Tracking recovered; reference slowdown released.', flush=True)
                sample = dict(
                    kind="cycle",
                    at=started,
                    loop_dt_ms=None if previous_solve_at is None else (started - previous_solve_at) * 1000,
                    schedule_lateness_ms=max(0., started - next_tick) * 1000,
                    qp_solve_ms=command.diagnostics.solve_time_ms,
                    control_state=core.control_state,
                    hold_reason=command.diagnostics.stop_reason,
                    tick=report["cycles"],
                    phase=old_phase,
                    control_phase=control_phase,
                    state=state,
                    positions=command.positions,
                    velocity=command.velocity,
                    mit=command.mit,
                    predicted_mit_effort=core.last_mit_effort,
                    tracking_observation=core.tracking_observation(),
                    executable=command.executable,
                    fault=command.diagnostics.fault,
                    timings=dataclasses.asdict(command.diagnostics.timings),
                    control_cycle_ms=(time.monotonic() - started) * 1000,
                )
                if old_phase in {"replay", "settle"}:
                    # Record the actual sampled clock and world targets for offline A/B.
                    # Wall time includes delay holds; cycle count can miss scheduling gaps.
                    sample["replay"] = dict(
                        elapsed_s=engine.replay_elapsed_s, speed=engine.speed,
                        recording_sha256=engine.recording.metadata["sha256"],
                        start_tick=engine.recording.metadata["start_tick"],
                        end_tick=engine.recording.metadata["end_tick"],
                        requested_targets={s: t.homogeneous.copy() for s, t in targets.items()},
                    )
                if input_source is not None:
                    sample["drag"] = input_source.diagnostics(engine, command, targets)
                    if sample["drag"]["input_age_ms"] is not None:
                        metrics["drag_input_age_ms"].append(sample["drag"]["input_age_ms"])
                    if sample["drag"]["active"]:
                        for side, error in zip(("left", "right"), sample["drag"]["position_errors_mm"], strict=True):
                            metrics[f"drag_{side}_position_error_mm"].append(error)
                previous_solve_at = started
                if old_phase == 'quest' and getattr(quest, 'stream', None) is not None:
                    sample['source'] = dict(quest.stream.diagnostics)
                    sample['quest_link'] = quest.bridge.runtime_metrics
                    sample['requested_targets'] = {s: t.homogeneous for s, t in targets.items()}
                    sample['tracking_errors'] = dict(
                        left_position_m=command.diagnostics.left_position_error,
                        right_position_m=command.diagnostics.right_position_error,
                        left_rotation_rad=command.diagnostics.left_orientation_error,
                        right_rotation_rad=command.diagnostics.right_orientation_error)
                    limited = core._limited_targets
                    if previous_limited is not None:
                        sample['limited_reference_velocity'] = {
                            s: dict(linear_m_s=float(np.linalg.norm(limited[s].translation - previous_limited[s].translation) / period),
                                    angular_rad_s=float(np.linalg.norm(pin.log3(previous_limited[s].rotation.T @ limited[s].rotation)) / period))
                            for s in limited}
                    previous_limited = {s: t.copy() for s, t in limited.items()}
                if old_phase == "quest" and hasattr(engine, "last_quest_sample"):
                    sample["input_latency"] = engine.last_quest_sample.latency_breakdown(time.monotonic_ns())
                theta_age = max(0.0, now - state.received["theta"])
                report["theta_feedback_max_age_ms"] = max(report["theta_feedback_max_age_ms"], theta_age * 1000)
                sample["theta_age_ms"] = theta_age * 1000
                sample["theta_skew_ms"] = 1000 * max(
                    abs(state.received["theta"] - state.received[n]) for n in ("left", "right", "base")
                )
                if theta_age > 0.04 or sample["theta_skew_ms"] > 20:
                    report["theta_gap_warning_samples"] += 1
                metrics["control_cycle_ms"].append(sample["control_cycle_ms"])
                metrics["ik_ms"].append(command.diagnostics.timings.total_ms)
                for key, value in sample.get("input_latency", {}).items():
                    if key.endswith("_ms") and value is not None:
                        metrics[key].append(value)
                for device, dispatch in (state.dispatch or {}).get("wire_first", {}).items():
                    key = (dispatch.get("epoch", 0), dispatch["sequence"]) if dispatch else (-1, -1)
                    if dispatch is not None and key > wire_sequences.get(device, (-1, -1)):
                        wire_sequences[device] = key
                        metrics[f"{device}_command_to_first_sdk_ms"].append(
                            1000 * (dispatch["sdk_send_started"] - dispatch["created"])
                        )
                writer.record(sample)
                if input_source is not None and hasattr(input_source, 'check_command'):
                    input_source.check_command(command, phase=old_phase)
                if not command.executable or command.mit is None or command.diagnostics.fault:
                    if str(command.diagnostics.fault).startswith("ControlDeadlineExceeded:"):
                        raise ControlDelay(str(command.diagnostics.fault))
                    raise HardwareSafetyError(f"{old_phase}: {command.diagnostics.fault}")
                if plant is not None:
                    packet = WholeBodyCommand(
                        wire_sequence, now, now+COMMAND_TTL, state.received, state.q,
                        command.positions, command.velocity, command.mit, command.base.velocity_body,
                        control_phase, recovery_epoch=recovery_epoch)
                    if simulation_supervisor is not None:
                        # Account for real GUI/control stalls; physics and the
                        # modeled SDK lease continue while IK is computing.
                        elapsed = time.monotonic()-started if not args.commands else 0.
                        simulation_endpoint.advance_to(now+elapsed)
                        simulation_supervisor.tick(packet)
                        if args.commands:
                            simulation_endpoint.advance_to(now+period)
                    else:
                        simulation_guard.validate(packet, state, now)
                        plant.core._gravity_effort[:] = core._gravity_effort
                        plant.publish(command)
                elif args.mode == "live":
                    gateway.publish(
                        WholeBodyCommand(
                            wire_sequence,
                            now,
                            dispatch_deadline(state, now, next_tick + min(COMMAND_TTL, period)),
                            state.received,
                            state.q,
                            command.positions,
                            command.velocity,
                            command.mit,
                            command.base.velocity_body,
                            control_phase,
                            cycle_started=(state.timing or {}).get("feedback_wait_started", now),
                            recovery_epoch=recovery_epoch,
                        )
                    )
                failure_input = None
                report["cycles"] += 1
                wire_sequence += 1
                if engine.phase != old_phase:
                    print(f"{old_phase} -> {engine.phase}", flush=True)
                    if engine.phase == 'hold' and old_phase == 'stopping':
                        announce_quest('holding', 'Stopped and holding; enter init to prepare again, or quit to release.')
                if engine.quit_requested and engine.idle:
                    report["passed"], report["finish_reason"] = True, "controlled_session_exit"
                    break
                if gateway is not None or not args.commands:
                    next_tick += period
                    time.sleep(max(0.0, next_tick - time.monotonic()))
            except ControlDelay as delay:
                if mapping.latency_policy != "hold-resume" or not active or (
                        gateway is None and simulation_supervisor is None):
                    raise
                if engine is not None and engine.quit_requested:
                    report["passed"], report["finish_reason"] = True, "operator_quit_during_delay_hold"
                    break
                if paused_at is None:
                    paused_at = time.monotonic()
                    if simulation_supervisor is not None:
                        simulation_supervisor.enter_delay(str(delay))
                    else:
                        gateway.request_hold()
                    writer.record(dict(kind="delay_hold", reason=str(delay), at=paused_at))
                if input_source is not None:
                    input_source.suspend(engine)
                next_tick = time.monotonic()
                time.sleep(.02)
    except (Exception, KeyboardInterrupt) as error:
        report["finish_reason"] = f"{type(error).__name__}: {error}"
        announce_quest('failed', 'Session ended; stopping and releasing; see the terminal fault report.')
    finally:
        if gateway is not None:
            try:
                report["gateway"] = gateway.close()
            except Exception as error:
                evidence = getattr(gateway, 'fault_evidence', None) or {}
                report["gateway"] = {"kind": "unconfirmed", "stop_failures": [str(error)],
                    'fault_state': evidence.get('fault_state'), 'fault': evidence.get('detail'),
                    'shutdown_stages': getattr(gateway, 'shutdown_stages', [])}
            if getattr(gateway, 'evidence_error', None):
                report['fault_evidence_error'] = gateway.evidence_error
            diagnostic = report['gateway'].get('diagnostics')
            if diagnostic and diagnostic.get('complete'):
                import shutil
                try:
                    destination = writer.directory / 'gateway-diagnostics.json'
                    shutil.copyfile(diagnostic['path'], destination)
                    diagnostic['path'] = str(destination.resolve())
                except OSError as error:
                    diagnostic.update(complete=False, error=f'diagnostic archive: {error}')
            report["motion_commands_sent"] = report["gateway"].get("motion_commands_sent", 0)
            if report["gateway"].get("stop_failures") or report["gateway"].get("fault"):
                report["passed"] = False
        if quest is not None:
            announce_quest('complete' if report['passed'] else 'failed',
                           'Session ended; see the terminal stop and release report.')
            quest.bridge._finish_requested.set()
            quest.close()
        worker.shutdown(wait=False, cancel_futures=True)
        if plant is not None:
            if getattr(input_source, 'retain_simulation', False):
                # Brake through physics and retain the measured pose/velocity.
                # Recovery must not teleport to a fresh scene's initial pose.
                if simulation_endpoint is not None:
                    report['simulation_stop_settle'] = simulation_endpoint.settle_for_restart()
                input_source.simulation_plant = plant
                final_state = plant.state()
                report['final_q'] = final_state.q.tolist()
                report['final_v'] = final_state.v.tolist()
                if input_source.last_display is not None:
                    input_source.last_display['q'] = final_state.q.tolist()
            if simulation_endpoint is not None:
                report["simulation_endpoint"] = simulation_endpoint.report()
            report["physical_effort"] = plant.effort_report()
            if getattr(input_source, 'simulation_plant', None) is not plant:
                plant.close()
        if expected_estop and report.get("gateway", {}).get("fault_state"):
            from .shutdown import expected_estop_success, physical_estop
            fault_state = report["gateway"]["fault_state"]
            report["estop_observed"] = physical_estop(fault_state, estop_armed_at)
            outcome = report['gateway']
            if expected_estop_success(outcome, estop_armed_at):
                report["passed"] = True
                report["finish_reason"] = "expected_physical_estop"
        if core is not None:
            report["session_completed"] = report["passed"]
            report["target_planner_failures"] = core.target_planner_failures
            report["last_target_planner_error"] = core.last_target_planner_error
            report["tracking_passed"] = None  # Interactive traces need independent performance scoring.
            if core.target_planner_failures:
                report["tracking_passed"] = False
                if profile.schema_version in {22, 23}:
                    report["passed"] = False
        if engine is not None:
            report["body_peaks"] = getattr(engine, "body_peaks", None)
            report["phases"] = engine.events
            report['arrival_details'] = engine.arrival.details
        if input_source is not None:
            report["drag"] = input_source.summary()
        report["metrics"] = {
            key: dict(
                zip(("p50", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100])), strict=True),
                samples=len(values),
            )
            for key, values in metrics.items()
            if values
        }
        report["metrics_window"] = "last 100000 observations per metric; complete trace retained"
        # Gateway histograms include every first dispatch (including the final
        # command), independent of supervisor snapshot sampling and resends.
        for device, details in report.get("gateway", {}).get("direct_downlink", {}).items():
            hist = details.get("latency_all_cycles", {}).get("command_to_first_sdk")
            if hist and hist["count"]:
                report["metrics"][f"{device}_command_to_first_sdk_ms"] = {
                    **dict(zip(("p50", "p95", "p99", "max"), hist["p50_p95_p99_max_ms"], strict=True)),
                    "samples": hist["count"], "scope": "all first dispatches; 0.01 ms bins"}
        from .failure_evidence import save_failure_evidence

        save_failure_evidence(writer, report, core, failure_input)
        writer.close(report)
    return report


def add_arguments(parser):
    parser.add_argument("--mode", choices=("probe", "shadow", "sim", "live"), default="probe")
    parser.add_argument("--research-release", type=Path, help="local research binding for GP80 live control")
    parser.add_argument("--robot-config", type=Path, default=CONFIG_DIR / "hardware/hexfellow_runtime.json")
    parser.add_argument("--feedback-policy", choices=FEEDBACK_POLICIES, default=None)
    parser.add_argument("--latency-policy", choices=("strict", "hold-resume"), default=None)
    parser.add_argument("--quest-stream-config", type=Path, default=None)
    parser.add_argument("--quest-pose-rate", choices=("auto", "30", "50", "60", "90"), default="auto")
    parser.add_argument("--quest-link-policy", choices=("strict", "target-200ms", "buffered"), default="strict")
    parser.add_argument("--quest-resume", choices=("auto", "require-release"), default="auto",
                        help="Quest clutch behavior after a recovered input/device delay")
    parser.add_argument("--bind-ip", default="auto")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--commands", nargs="+", help="sequential commands; quote replay PATH as one argument")
    parser.add_argument("--initial-state-report", type=Path, help="sim only: initial measured pose report")
    parser.add_argument("--plant-calibration", type=Path,
                        help="sim only: identified plant calibration JSON (gravity, friction, inertia)")
    parser.add_argument("--run-seconds", type=float, default=3600.0)
    parser.add_argument("--save-trace", type=Path)
    parser.add_argument("--confirm-onsite-ready", action="store_true")
    parser.add_argument("--allow-uncommissioned-profile-sha256")
    parser.add_argument("--verification-report", type=Path, help="completed onsite research-control report")


def handle(args, parser):
    if not np.isfinite(args.run_seconds) or args.run_seconds <= 0:
        parser.error("run-seconds must be positive and finite")
    if args.initial_state_report is not None and args.mode != "sim":
        parser.error("initial-state-report is simulation only")
    if getattr(args, "plant_calibration", None) is not None and args.mode != "sim":
        parser.error("plant-calibration is simulation only")
    args.save_trace = args.save_trace or Path("runs") / (
        "hardware-session-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    report = run(args)
    print(json.dumps(jsonable(report), indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="policy-sim hardware-session", description=__doc__)
    add_arguments(parser)
    handle(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
