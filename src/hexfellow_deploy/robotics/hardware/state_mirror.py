"""Measured-state display, with optional explicit single-arm zero-torque release."""

import argparse
import os
import queue
import select
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..control.motion_profile import load_motion_profile
from ..control.robot import GRIPPER_JOINTS, build_scene
from ..control.robot_model import load_robot_model
from .gateway import HardwareGateway
from .mapping_comparison import (
    LEGACY_DRAG_MAPPING,
    CURRENT_HIGHSPEED_MAPPING,
    ReplayTrace,
    comparison_labels,
    comparison_mappings,
    joint_text,
    remap_state,
    replay_mapping_path,
)
from .whole_body import JOINTS, RobotMapping, vector
from .x7 import HardwareSafetyError


class TerminalCommands:
    """Poll complete terminal lines without blocking rendering or its heartbeat."""

    def __init__(self, stream=None):
        self.buffer = b""
        try:
            self.fd = (sys.stdin if stream is None else stream).fileno()
        except (OSError, ValueError):
            self.fd = None

    def poll(self):
        if self.fd is None:
            return []
        ready, _, _ = select.select([self.fd], [], [], 0)
        if not ready:
            return []
        data = os.read(self.fd, 4096)
        if not data:
            self.fd = None
            self.buffer = b""
            print("Terminal closed; stopping release and exiting.", flush=True)
            return ["q"]
        self.buffer += data
        commands = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            command = line.decode("utf-8", errors="replace").strip().lower()
            if command in {"u", "b", "q", "1", "2", "3", "4"}:
                commands.append(command)
            elif command:
                print("Commands: u release | b brake | q quit | 1/2/3/4 display; press Enter.", flush=True)
        if len(self.buffer) > 4096:
            self.buffer = b""
        # A stop in the same input batch must supersede a queued release.
        if "q" in commands:
            return ["q"]
        if "b" in commands:
            return ["b"]
        return commands


def apply_state(model, data, state, grippers, auxiliary_joints=GRIPPER_JOINTS):
    import mujoco

    values = np.concatenate((vector(state.q, 18, "measured pose"),
                             vector(grippers, len(auxiliary_joints), "fixed grippers")))
    addresses = []
    outside = []
    for name, value in zip((*JOINTS, *auxiliary_joints), values, strict=True):
        joint = model.joint(name)
        addresses.append(int(joint.qposadr[0]))
        if joint.limited[0] and not joint.range[0] <= value <= joint.range[1]:
            outside.append(name)
    data.qpos[addresses] = values
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return outside


def status_text(state, now):
    ages = {n: (now - t) * 1000 for n, t in state.received.items()}
    skew = np.ptp(list(state.received.values())) * 1000
    stale = max(ages.values()) > 40 or min(ages.values()) < 0 or skew > 20
    warning = "STALE / UNSYNCHRONIZED" if stale else "RECEIVING"
    return (f"READ ONLY | {warning}\n"
            f"ages ms { {n: round(v, 1) for n, v in ages.items()} } | skew {skew:.1f} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path,
                        help="Default: corrected J3/J5/J7 runtime; live comparison uses legacy runtime; "
                             "replay selects a known runtime by its recorded hash")
    parser.add_argument("--rate-hz", type=float, default=30)
    parser.add_argument("--run-seconds", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--show-joints", action="store_true", help="Overlay all SDK/model arm joint angles")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-j3", action="store_true", help="Alias for --compare-joints 3")
    comparison.add_argument("--compare-joints",
                            help="Comma-separated joints to reverse, e.g. 3,5; keys 1/2/3/4 select sides")
    parser.add_argument("--replay-run", type=Path, help="Recorded run or session directory; no hardware connection")
    parser.add_argument("--release-arm", choices=("left", "right"),
                        help="Terminal u + Enter releases this arm only; b + Enter brakes")
    args = parser.parse_args(argv)
    if args.release_arm and (args.replay_run or args.headless or args.rate_hz < 20):
        parser.error("release-arm requires a live window and rate-hz >= 20; no replay/headless release")
    try:
        joints = (tuple(int(value) for value in args.compare_joints.split(","))
                  if args.compare_joints is not None else (3,))
        if len(set(joints)) != len(joints) or any(j < 1 or j > 7 for j in joints):
            raise ValueError
    except ValueError:
        parser.error("compare-joints must contain distinct joint numbers from 1 to 7, e.g. 3,5")
    compare = args.compare_j3 or args.compare_joints is not None
    if not np.isfinite(args.rate_hz) or not 0 < args.rate_hz <= 120:
        parser.error("rate-hz must be in (0, 120]")
    if args.run_seconds is not None and (not np.isfinite(args.run_seconds) or args.run_seconds <= 0):
        parser.error("run-seconds must be positive")
    import mujoco
    import mujoco.viewer

    if args.robot_config is None:
        try:
            args.robot_config = (replay_mapping_path(args.replay_run) if args.replay_run
                                 else LEGACY_DRAG_MAPPING if compare else CURRENT_HIGHSPEED_MAPPING)
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"Replay rejected: {error}", flush=True)
            return 1
    mapping = RobotMapping.load(args.robot_config)
    robot_model = load_robot_model(mapping.data.get("robot_model"))
    if compare and mapping.data.get("candidate_source"):
        parser.error("comparison requires the source mapping, not a derived sign candidate")
    print(f"Robot mapping: {args.robot_config.resolve()}", flush=True)
    if mapping.data.get("read_only_candidate", False):
        print("UNVERIFIED READ-ONLY CANDIDATE:", mapping.data.get("candidate_note", ""), flush=True)
    if not compare:
        joints = tuple(mapping.data.get("candidate_joints", [3]))
    labels = comparison_labels(joints)
    variants = comparison_mappings(mapping, joints) if compare else [mapping]
    # Deliberately construct no gateway (including on exit) for offline replay.
    try:
        replay = ReplayTrace(args.replay_run, mapping) if args.replay_run else None
    except (OSError, KeyError, TypeError, ValueError, HardwareSafetyError) as error:
        print(f"Replay rejected: {error}", flush=True)
        return 1
    gateway = None
    if replay is None:
        profile = load_motion_profile("x7-mit-highspeed-v10")
        gateway = HardwareGateway(mapping, profile, "whole-body", "probe", profile.mit_servo.torque_limit_nm)
    keys = queue.SimpleQueue()
    selected = 0
    frozen = False
    playing = False
    quit_requested = False
    replay_due = 0.
    viewer = None
    release = None
    terminal = None
    exit_code = 0
    print("READ ONLY: no enable/calibration/control. Base origin = first odometry; grippers = FIXED, NOT MEASURED.")
    with tempfile.TemporaryDirectory(prefix="hexfellow-mirror-") as directory:
        path = build_scene(urdf_path=robot_model.urdf, output_path=Path(directory) / "robot.xml", base_mode="planar",
                           preserve_fixed_inertias=True)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        try:
            if replay is None:
                gateway.start()
                deadline = time.monotonic() + 15
                while True:
                    try:
                        state = gateway.state(fresh=False)
                        break
                    except HardwareSafetyError:
                        if gateway.process.exitcode is not None or time.monotonic() > deadline:
                            raise
                        time.sleep(.02)
            else:
                state = replay.frame.state
            apply_state(model, data, state, mapping.data["fixed_gripper_positions"], robot_model.auxiliary_joints)
            if not args.headless:
                viewer = mujoco.viewer.launch_passive(
                    model, data, key_callback=keys.put, show_left_ui=False, show_right_ui=False)
                geometry_groups = viewer.opt.geomgroup.copy()
            if args.ready_file is not None:
                args.ready_file.write_text("ready\n")
            if args.release_arm:
                from .manual_release import ManualRelease

                release = ManualRelease(mapping, args.release_arm)
                terminal = TerminalCommands()
                print(f"{args.release_arm.upper()}: support the arm; zero torque does not support weight.\n"
                      "TERMINAL: u + Enter release | b + Enter brake | q + Enter quit | 1/2/3/4 + Enter display.\n"
                      "Window U is disabled. One release per run; other devices remain read-only.", flush=True)
            started = time.monotonic()
            next_print = 0
            fault = None
            # Headless replay traverses every frame from the initial state.
            if replay is not None and args.headless:
                replay.index = 0
            while viewer is None or viewer.is_running():
                now = time.monotonic()
                if args.run_seconds is not None and now - started >= args.run_seconds:
                    break
                changed = False
                if terminal is not None:
                    for command in terminal.poll():
                        keys.put("release" if command == "u" else ord(command.upper()))
                while not keys.empty():
                    key = keys.get()
                    if key == ord("Q"):
                        quit_requested = True
                        if release is not None:
                            release.request_stop()
                    elif release is not None and key == "release":
                        release.start()
                    elif release is not None and key == ord("U"):
                        print("Release from terminal: support arm, type u and press Enter.", flush=True)
                    elif release is not None and key == ord("B"):
                        release.request_stop()
                    elif compare and key in tuple(map(ord, "1234")):
                        selected = key - ord("1")
                        changed = True
                    elif key == 32:
                        if replay is not None:
                            playing = not playing
                            replay_due = now
                        else:
                            frozen = not frozen
                    elif replay is not None and key in (ord("N"), ord("P"), 262, 263):
                        playing = False
                        if key in (ord("N"), ord("P")):
                            replay.jump(1 if key == ord("N") else -1)
                        else:
                            replay.step(1 if key == 262 else -1)
                        changed = True
                if quit_requested:
                    break
                outside = []
                if fault is None:
                    try:
                        if replay is not None:
                            if playing:
                                while replay.index < len(replay.frames) - 1 and now >= replay_due:
                                    previous = replay.frame.at
                                    replay.step(1)
                                    replay_due += max(.001, replay.frame.at - previous)
                                if replay.index == len(replay.frames) - 1:
                                    playing = False
                            state = replay.frame.state
                        else:
                            latest = gateway.state(fresh=False)
                            if release is not None:
                                release.update(latest, labels[selected] if compare else "configured mapping")
                            if not frozen:
                                state = latest
                        displayed = remap_state(state, variants[selected]) if compare else state
                        if viewer is None:
                            outside = apply_state(model, data, displayed, mapping.data["fixed_gripper_positions"],
                                                      robot_model.auxiliary_joints)
                        else:
                            with viewer.lock():
                                if compare and changed:
                                    # Native MuJoCo number keys also toggle geom groups.
                                    # Mapping selection must not hide the robot meshes.
                                    viewer.opt.geomgroup[:] = geometry_groups
                                outside = apply_state(model, data, displayed, mapping.data["fixed_gripper_positions"],
                                                      robot_model.auxiliary_joints)
                    except (HardwareSafetyError, KeyError, ValueError) as error:
                        fault = str(error)
                        exit_code = 1
                        if release is not None:
                            release.request_stop()
                if fault:
                    text = f"DISCONNECTED/INVALID: {fault}; last valid pose"
                elif replay is not None:
                    text = (f"OFFLINE REPLAY | {replay.frame.phase} | frame {replay.index + 1}/{len(replay.frames)}"
                            f" | recorded host time {replay.frame.at:.6f} | {'PLAYING' if playing else 'PAUSED'}")
                else:
                    text = status_text(state, now) + (" | FROZEN SNAPSHOT" if frozen else "")
                if release is not None:
                    text = text.replace("READ ONLY", "MEASURED FEEDBACK") + "\n" + release.message
                if compare:
                    text += f"\nUNVERIFIED DISPLAY: {labels[selected]} (relative to source mapping)"
                if not fault and (compare or mapping.data.get("read_only_candidate", False)):
                    text += "\n" + joint_text(displayed, variants[selected], joints)
                elif not fault and args.show_joints:
                    text += "\n" + joint_text(displayed, variants[selected], tuple(range(1, 8)))
                if outside:
                    text += f" | MODEL LIMIT MISMATCH: {outside}"
                if mapping.data.get("read_only_candidate", False):
                    text = "UNVERIFIED READ-ONLY MAPPING\n" + text
                if not fault and "pos" in state.health.get("theta", {}):
                    text += (f"\nLift model: {displayed.q[3] * 1000:.2f} mm"
                             f" | SDK: {state.health['theta']['pos'] * 1000:.2f} mm")
                controls = "Q quit | Grippers fixed, not measured"
                if release is not None:
                    controls += f"\nTERMINAL u + Enter: release {args.release_arm} | b + Enter: brake | Window B: brake"
                if compare:
                    controls += f"\n1 original | 2 {labels[1]} | 3 right | 4 both"
                controls += ("\nSpace play/pause | N/P next/previous phase | arrows step frame"
                             if replay is not None else "\nSpace freeze/unfreeze feedback display")
                if viewer is not None:
                    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       text, controls)])
                    viewer.sync(state_only=True)
                if (release is None and now >= next_print) or changed or (replay is not None and args.headless
                                                    and replay.index in replay.phase_starts):
                    print(text, flush=True)
                    if compare and args.headless and not fault:
                        for label, variant in zip(labels, variants, strict=True):
                            print(label, joint_text(remap_state(state, variant), variant, joints), flush=True)
                    elif not fault:
                        print("left deg", np.rad2deg(displayed.q[4:11]).round(2),
                              "right deg", np.rad2deg(displayed.q[11:]).round(2),
                              "base m,m,rad", displayed.q[:3].round(4),
                              "calibrated", {n: r.get("calibrated") for n, r in state.health.items()}, flush=True)
                    next_print = now + 1
                if fault and args.headless:
                    break
                if replay is not None and args.headless:
                    if replay.index == len(replay.frames) - 1:
                        break
                    replay.step(1)
                else:
                    time.sleep(1 / args.rate_hz)
        except KeyboardInterrupt:
            pass
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Read-only mirror failed: {error}", flush=True)
            exit_code = 1
        finally:
            # Stop before potentially slow window/feedback-gateway shutdown.
            if release is not None and not release.close():
                exit_code = 1
            if viewer is not None:
                viewer.close()
                deadline = time.monotonic() + 2
                while viewer.m is not None and time.monotonic() < deadline:
                    time.sleep(.01)
            if gateway is not None:
                print("Passive close:", gateway.close(), flush=True)
            else:
                print("Offline replay closed: no hardware connection or commands.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
