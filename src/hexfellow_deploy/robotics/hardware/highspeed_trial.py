"""Operator-authorized GP80 research trial; simulation is always the default."""

import argparse
import fcntl
import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..paths import PROJECT_ROOT, REPOSITORY_ROOT
from . import session
from .research_profiles import HIGHSPEED_PROFILE, HIGHSPEED_PROFILES, HIGHSPEED_V2_PROFILE, HIGHSPEED_V3_PROFILE
from .x7 import HardwareSafetyError

RUNTIME = PROJECT_ROOT / "config/hardware/hexfellow_gp80_highspeed_j357.json"
RELEASE = PROJECT_ROOT / "config/local/hexfellow_gp80_highspeed_research.json"
# Profile variant -> (runtime mapping, research release binding). v1 stays the default entry.
VARIANTS = {
    "v1": (HIGHSPEED_PROFILE, RUNTIME, RELEASE),
    "v2": (HIGHSPEED_V2_PROFILE, PROJECT_ROOT / "config/hardware/hexfellow_gp80_highspeed_j357_v2.json",
           PROJECT_ROOT / "config/local/hexfellow_gp80_highspeed_v2_research.json"),
    "v3": (HIGHSPEED_V3_PROFILE, PROJECT_ROOT / "config/hardware/hexfellow_gp80_highspeed_j357_v3.json",
           PROJECT_ROOT / "config/local/hexfellow_gp80_highspeed_v3_research.json"),
}


def _variant_files(variant):
    """(runtime, release); v1 resolves the module attributes at call time."""
    if variant == "v1":
        return RUNTIME, RELEASE
    return VARIANTS[variant][1:]


def _release_for(profile_id):
    for variant, (candidate, _, _) in VARIANTS.items():
        if candidate == profile_id:
            return _variant_files(variant)[1]
    raise KeyError(profile_id)
OUTPUT = PROJECT_ROOT / "runs/highspeed-fixed-base"


def require_research_release(profile, release_path=None, runtime_path=None):
    """Bind the explicit research authorization, never reinterpret a failed audit."""
    try:
        release = json.loads(Path(release_path or _release_for(profile.profile_id)).read_text())
        valid = (
            profile.profile_id in HIGHSPEED_PROFILES
            and release.get("profile_id", profile.profile_id) == profile.profile_id
            and profile.schema_version == 23
            and not profile.base_control_enabled
            and profile.lift_velocity_m_s == 0.24
            and release["profile_sha256"] == profile.sha256
            and release["authorization"] == "operator_requested_research_trial"
            and release["hardware_qualified"] is False
            and bool(release["source_sha256"])
        )
        if runtime_path is not None:
            valid = valid and release.get("runtime_sha256") == hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest()
        valid = valid and all(
            hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest() == value
            for name, value in release["source_sha256"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise HardwareSafetyError("GP80 research configuration/source binding mismatch; regenerate the release binding; no motion control requested")


def show_session(directory, profile):
    """Visualize measured simulation feedback after the physical session finishes."""
    import time

    import mujoco
    import mujoco.viewer

    from ..control.motion_profile import controller_config_for_profile
    from .mit_ik_replay import SimulationPlant

    rows = []
    with (directory / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "cycle":
                rows.append((min(row["state"]["received"].values()), row["state"]["q"], row["phase"]))
    if not rows:
        return
    times = np.asarray([row[0] for row in rows])
    times -= times[0]
    plant = SimulationPlant(
        SimpleNamespace(summary={}), controller_config_for_profile(profile), np.array([]), np.array(rows[0][1])
    )
    try:
        with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
            viewer.cam.lookat[:] = (0.4, 0, 0.7)
            viewer.cam.distance = 2.6
            viewer.cam.azimuth, viewer.cam.elevation = 135, -20
            began = time.monotonic()
            while viewer.is_running():
                i = int(np.clip(np.searchsorted(times, time.monotonic() - began, side="right") - 1, 0, len(rows) - 1))
                with viewer.lock():
                    plant.data.qpos[plant.core._mj_qpos_indices] = rows[i][1]
                    plant.data.qvel[:] = 0.0
                    mujoco.mj_forward(plant.model, plant.data)
                viewer.set_texts(
                    [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "GP80 / 15-variable IK / BASE DISABLED",
                            rows[i][2] + " | measured simulation replay",
                        )
                    ]
                )
                viewer.sync()
                time.sleep(0.01)
    finally:
        plant.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "live"), default="sim")
    parser.add_argument("--input", choices=("replay", "drag", "quest"), default="replay")
    parser.add_argument('--control-group', choices=('coupled', 'arms'), default='coupled',
                        help='drag only: arms excludes chassis and lift from IK')
    parser.add_argument('--tracking-state', choices=('measured', 'reference'), default='measured',
                        help='drag IK state: current measured joints, or legacy command reference')
    parser.add_argument("--replay-trace", type=Path, default=PROJECT_ROOT / "artifacts/reference/synthetic-replay")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-tick", type=int)
    parser.add_argument("--end-tick", type=int)
    parser.add_argument("--save-trace", type=Path)
    parser.add_argument("--run-seconds", type=float, default=3600.0)
    parser.add_argument("--bind-ip", default="auto")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--confirm-onsite-ready", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help="headless simulation replay")
    parser.add_argument("--robot-config", type=Path, help="local robot mapping; required for live mode")
    parser.add_argument("--research-release", type=Path, help="local source/configuration binding; required for live mode")
    parser.add_argument("--profile", choices=tuple(VARIANTS), default="v1",
                        help="profile variant (v1 = default, v2 = tracking, v3 = quiet research candidate)")
    parser.add_argument("--plant-calibration", type=Path,
                        help="sim only: identified plant calibration JSON (gravity, friction, inertia)")
    args = parser.parse_args(argv)
    if args.control_group != 'coupled' and args.input != 'drag':
        parser.error('--control-group arms requires --input drag')
    if args.plant_calibration is not None and args.mode != "sim":
        parser.error("--plant-calibration is simulation only")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("--speed must be in (0,1]")
    if not np.isfinite(args.run_seconds) or args.run_seconds <= 0:
        parser.error("--run-seconds must be positive")
    if args.mode == "live" and not args.confirm_onsite_ready:
        parser.error("After confirming empty grippers, clearance/cables, charger disconnected, and accessible E-stop onsite, use --confirm-onsite-ready")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.save_trace = args.save_trace or OUTPUT / f"{args.mode}-{args.input}-{tag}"
    if args.mode == "live" and (args.robot_config is None or args.research_release is None):
        parser.error("live mode requires --robot-config and --research-release; see docs/hardware.md")
    runtime = args.robot_config or _variant_files(args.profile)[0]
    args.robot_config = runtime
    base_parser = argparse.ArgumentParser()
    session.add_arguments(base_parser)
    values = base_parser.parse_args(["--mode", args.mode, "--robot-config", str(runtime)])
    values.plant_calibration = args.plant_calibration
    values.research_release = args.research_release
    values.demo_authorized = args.mode == "live"
    mapping, profile, _, _ = session.load_configuration(values)
    from .research_profiles import describe

    if profile.profile_id != VARIANTS[args.profile][0]:
        parser.error("--profile does not match the selected robot configuration")
    print(describe(profile, control_group=args.control_group), flush=True)
    if args.mode == "live":
        require_research_release(profile, args.research_release, runtime)
    if args.input == "drag":
        from .drag_control import run

        return 0 if run(args)["passed"] else 1
    replay = "replay " + shlex.quote(str(args.replay_trace)) + " --speed " + str(args.speed)
    for key in ("start_tick", "end_tick"):
        if getattr(args, key) is not None:
            replay += " --" + key.replace("_", "-") + " " + str(getattr(args, key))
    # Load/validate the recording before any live connection.
    if args.input == "replay":
        session.parsed_command(replay)
    args.save_trace.mkdir(parents=True, exist_ok=False)
    values.save_trace = args.save_trace / "session"
    values.run_seconds = args.run_seconds
    values.bind_ip, values.port = args.bind_ip, args.port
    values.commands = ["default", "init", replay, "quit"] if args.input == "replay" else None
    values.confirm_onsite_ready = args.confirm_onsite_ready
    values.allow_uncommissioned_profile_sha256 = profile.sha256
    report = None
    lock = None
    try:
        if args.mode == "live":
            from .demo import ACCEPTED
            from .quest_trial import audit_passed
            from .verification import device_binding, software_binding
            from .verify import save_json, timing_probe

            (PROJECT_ROOT / "runs").mkdir(exist_ok=True)
            lock = (PROJECT_ROOT / "runs/.hexfellow-onsite.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            mapping.authorize(profile, "whole-body", profile.sha256)
            audit = timing_probe(mapping, profile, args.save_trace / "audit", 30, sdk_sink=True,
                                 allow_lift_retreat=profile.profile_id in HIGHSPEED_PROFILES)
            save_json(
                args.save_trace / "research-trial.json",
                dict(
                    authorization="operator_requested_research_trial",
                    hardware_qualified=False,
                    binding=software_binding(mapping),
                    experimental_checks_accepted=list(ACCEPTED),
                    audit_passed=audit_passed(audit),
                ),
            )
            if not audit_passed(audit):
                raise HardwareSafetyError("30-second read-only feedback check failed; motion not started")
            values.demo_device_binding = device_binding(SimpleNamespace(health=audit["initial_health"]))
        if args.input == "quest":
            print("Enter default, init, quest in order, waiting for each step; stop to stop, quit to exit.", flush=True)
        report = session.run(values)
    finally:
        if lock is not None:
            lock.close()
    print("Report:", values.save_trace, flush=True)
    if args.mode == "sim" and not args.no_viewer:
        print("Showing measured simulation feedback, including the pre-stop segment after failure; close the window to exit.", flush=True)
        show_session(values.save_trace, profile)
    return 0 if report["passed"] else 1
