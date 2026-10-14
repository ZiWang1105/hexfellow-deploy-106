"""Onsite evidence and narrowly scoped research-control authorization.

These are local operator records, not signed certificates or hardware certification.
No device connection or motion is performed by this module.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from ..paths import CONFIG_DIR, PROJECT_ROOT, REPOSITORY_ROOT
from .commission_motion import MOTIONS
from .x7 import HardwareSafetyError


@dataclass(frozen=True)
class Step:
    name: str
    stage: str
    test: str
    explanation: str


STEPS = (
    Step("probe", "whole-body", "probe", "Read-only check of connectivity, measured height, device state, and battery; no control acquisition."),
    Step("left-hold", "left-arm", "hold", "Acquire left-arm control and hold for 3 seconds; check for jumps, vibration, or drops."),
    Step("left-nudge", "left-arm", "left-nudge", MOTIONS["left-nudge"].prompt()),
    Step("right-hold", "right-arm", "hold", "Acquire right-arm control and hold for 3 seconds; check for jumps, vibration, or drops."),
    Step("right-nudge", "right-arm", "right-nudge", MOTIONS["right-nudge"].prompt()),
    Step("dual-hold", "dual-arm", "hold", "Hold both arms for 3 seconds and confirm stable combined control."),
    Step("lift-hold", "arms-lift", "hold", "Add lift control; keep both arms and lift in place."),
    Step("body-hold", "whole-body", "hold", "Add base control; hold the whole robot in place."),
    Step("lift-move", "arms-lift", "lift-move", MOTIONS["lift-move"].prompt()),
    Step("base-x", "whole-body", "base-x", MOTIONS["base-x"].prompt()),
    Step("base-y", "whole-body", "base-y", MOTIONS["base-y"].prompt()),
    Step("base-yaw", "whole-body", "base-yaw", MOTIONS["base-yaw"].prompt()),
    Step("left-abort", "left-arm", "initialization-abort", "Cancel left-arm initialization immediately after feedback; confirm stopping and release."),
    Step("right-abort", "right-arm", "initialization-abort", "Cancel right-arm initialization immediately after feedback; confirm stopping and release."),
    Step("lift-abort", "arms-lift", "initialization-abort", "Cancel after lift velocity configuration feedback; confirm lift braking."),
    Step("body-abort", "whole-body", "initialization-abort", "Cancel whole-body initialization after feedback; confirm all acquired devices stop."),
    Step("left-stall", "left-arm", "publisher-stall", "Pause publishing for 80 ms during hold; check that the independent watchdog stops the left arm."),
    Step("right-stall", "right-arm", "publisher-stall", "Pause publishing for 80 ms during hold; check that the independent watchdog stops the right arm."),
    Step("lift-stall", "arms-lift", "publisher-stall", "Pause publishing for 80 ms during hold; check that both arms and lift stop."),
    Step("body-stall", "whole-body", "publisher-stall", "Pause publishing for 80 ms during hold; check that the whole robot stops."),
    Step("estop", "whole-body", "estop", "Press the physical E-stop when prompted; reset only after program exit and instructions; no automatic reacquisition."),
    Step(
        "disconnect",
        "whole-body",
        "disconnect",
        "Wait for the stationary-hold prompt, then unplug the robot data cable at the computer; observe the robot and reconnect when prompted. Do not unplug power.",
    ),
    Step("poses", "whole-body", "session", "Execute default -> init -> default -> init slowly; observe pose, base hold, and transitions."),
    Step("replay", "whole-body", "session", "Run default/init, then replay frames 250-399 at 0.25 speed with small motions and settle."),
    Step("quest", "whole-body", "session", "Follow terminal instructions to enter Quest, move both arms slightly, release squeeze, then stop and quit."),
)
STEP_BY_NAME = {step.name: step for step in STEPS}
REPLAY_COMMAND = "replay " + shlex.quote(str(PROJECT_ROOT / "artifacts/reference/x7-small-replay")) + " --start-tick 250 --end-tick 399 --speed 0.25"
SESSION_COMMANDS = {
    "poses": ["default", "init", "default", "init", "quit"],
    "replay": ["default", "init", REPLAY_COMMAND, "quit"],
}
RESEARCH_ACCEPTANCE = {
    "drive_total_torque_clamp": "unknown_accepted_for_research",
    "source": "operator choice in 2026-09-19 planning conversation",
    "hardware_certified": False,
    "disconnect_scope": "stationary cable-loss observation only; moving cable-loss stopping is unqualified",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def software_binding(mapping):
    """Include code, geometry/config and the actual installed SDK, never run logs."""
    from ..control.feedback_policy import coupled_groups
    files = set((REPOSITORY_ROOT / "src/hexfellow_deploy").rglob("*.py"))
    files.update((REPOSITORY_ROOT / "src").glob("_hexfellow_collision_kernel*.so"))
    files.update((REPOSITORY_ROOT / "src/hexfellow_deploy/robotics/control").glob("*.cpp"))
    files.update(REPOSITORY_ROOT / name for name in ("setup.py", "pyproject.toml"))
    files.update((CONFIG_DIR / "control").rglob("*.yaml"))
    files.update((CONFIG_DIR / "control").rglob("*.json"))
    files.update((CONFIG_DIR / "quest").glob("*.json"))
    from ..control.robot_model import load_robot_model
    model = load_robot_model(mapping.data.get("robot_model"))
    files.add(model.urdf)
    if model.descriptor is not None:
        files.add(model.descriptor)
    files.update(PROJECT_ROOT / path for path, _ in model.assets)
    from ..paths import config_reference
    files.add(config_reference(mapping.source_path, mapping.data["default_pose"]["path"]))
    files.add(PROJECT_ROOT / mapping.data["hardware_capabilities"]["path"])
    sources = {(str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)): digest(p) for p in sorted(files)}
    sdk = importlib.metadata.distribution("hex_device")
    sdk_sources = {
        str(p): digest(sdk.locate_file(p))
        for p in (sdk.files or [])
        if str(p).startswith("hex_device/") and str(p).endswith(".py")
    }
    return {
        "mapping_sha256": mapping.sha256,
        "profile_sha256": mapping.data["commissioning"]["profile_sha256"],
        "feedback_policy": mapping.feedback_policy,
        "feedback_groups": list(coupled_groups(mapping.feedback_policy)),
        "latency_policy": mapping.latency_policy,
        "sources": sources,
        "sdk_version": sdk.version,
        "sdk_sources": sdk_sources,
    }


def device_binding(state):
    keys = (
        "protocol_version",
        "joint_names",
        "joint_limits",
        "series",
        "range",
        "pulse_per_meter",
        "maximum_speed",
        "serial_number",
        "robot_type",
        "firmware",
    )
    return {name: {key: row[key] for key in keys if key in row} for name, row in state.health.items()}


def machine_pass(step, result):
    """A CLI exit status or a human 'yes' alone is never execution evidence."""
    if result.get("passed") is not True or result.get("trace_error"):
        return False
    if step.test == "probe":
        return (
            result.get("mode") == "probe"
            and result.get("motion_commands_sent") == 0
            and result.get("start_checked") is True
            and bool(result.get("device_binding"))
            and result.get("gateway", {}).get("kind") == "closed"
            and not result["gateway"].get("stop_failures")
            and not result["gateway"].get("fault")
        )
    if step.test in {"estop", "disconnect"}:
        return (
            result.get("execute") is True
            and result.get("event_observed") is True
            and result.get("test") == step.test
            and result.get("pre_event_hold_confirmed") is True
            and result.get("post_event_stationary") is True
            and result.get("recovery_motion_commands") == 0
        )
    gateway = result.get("gateway", {})
    if gateway.get("kind") != "closed" or gateway.get("stop_failures"):
        return False
    from .whole_body import STAGES

    observed = gateway.get("stop_observations", {})
    touched = ("theta",) if step.name == "lift-abort" else STAGES[step.stage]
    if not all(
        observed.get(name, {}).get("continuous_stop_confirmed") is True
        and observed[name].get("stationary") is True
        and observed[name].get("released_or_braked") is True
        for name in touched
    ):
        return False
    if step.test != "initialization-abort" and not all(
        gateway.get("direct_downlink", {}).get(name, {}).get("counts", {}).get("motion", 0) > 0
        for name in STAGES[step.stage]
    ):
        return False
    if step.test in {"left-nudge", "right-nudge"}:
        accuracy = result.get("arm_motion_accuracy", {})
        if accuracy.get("passed") is not True or accuracy.get("finalized") is not True:
            return False
    if step.test == "session":
        if result.get("mode") != "live" or result.get("verification_scope") != step.name:
            return False
        phases = [row["phase"] for row in result.get("phases", [])]
        required = {
            "poses": {"default", "init", "ready"},
            "replay": {"replay", "settle", "hold"},
            "quest": {"quest", "stopping", "hold"},
        }[step.name]
        if not required.issubset(phases) or gateway.get("fault"):
            return False
        if step.name == "poses" and (phases.count("default") != 2 or phases.count("ready") != 2):
            return False
        if step.name == "quest" and not (
            result.get("quest_active_cycles", 0) >= 50 and result.get("quest_release_events", 0) >= 1
        ):
            return False
        return result.get("motion_commands_sent", 0) > 0
    return (
        result.get("execute") is True
        and result.get("stage") == step.stage
        and result.get("test") == step.test
        and result.get("stop_confirmed") is True
        and (
            step.test not in {"left-nudge", "right-nudge", "lift-move", "base-x", "base-y", "base-yaw"}
            or result.get("actual_motion_confirmed") is True
        )
    )


def checked_results(ledger, root):
    """Only a contiguous prefix of genuine successful onsite results is reusable."""
    good = {}
    for step in STEPS:
        entry = ledger.get("completed", {}).get(step.name)
        if not entry:
            break
        path = (root / entry["report"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != entry.get("sha256"):
            raise HardwareSafetyError(f"Qualification report modified:{step.name}; repeat this and subsequent steps")
        report = read_json(path)
        if any(report.get(key) != ledger["binding"].get(key) for key in ("mapping_sha256", "profile_sha256")):
            raise HardwareSafetyError(f"Qualification report configuration/profile mismatch:{step.name}")
        if entry.get("operator_normal") is not True or not machine_pass(step, report):
            raise HardwareSafetyError(f"Invalid onsite evidence:{step.name}")
        if report.get("device_binding") != ledger.get("device_binding"):
            raise HardwareSafetyError(f"Device identity/range/protocol evidence mismatch:{step.name}")
        good[step.name] = report
    return good


def permission(path, mapping, scope="control"):
    path = Path(path).resolve()
    ledger = read_json(path)
    if ledger.get("schema_version") != 1 or ledger.get("binding") != software_binding(mapping):
        raise HardwareSafetyError("Qualification configuration, source, or SDK changed; repeat onsite verification")
    if ledger.get("research_acceptance") != RESEARCH_ACCEPTANCE:
        raise HardwareSafetyError("Missing unresolved-item record for this research run")
    good = checked_results(ledger, path.parent)
    if scope not in {"control", "poses", "replay", "quest"}:
        raise HardwareSafetyError("invalid verification scope")
    needed = len(STEPS) if scope == "control" else [s.name for s in STEPS].index(scope)
    if len(good) < needed:
        raise HardwareSafetyError("Onsite verification incomplete; run hardware-verify status for the next step")
    return {
        "scope": scope,
        "mapping_sha256": mapping.sha256,
        "profile_sha256": mapping.data["commissioning"]["profile_sha256"],
        "device_binding": ledger["device_binding"],
        "report": str(path),
        "checks": {
            "sdk_startup_checked": True,
            "initialization_stop_checked": True,
            "disconnect_stop_checked": True,
            "theta_stop_checked": True,
            "base_axes_checked": True,
        },
        "research_acceptance": RESEARCH_ACCEPTANCE,
    }


def check_devices(permit, state):
    if permit and permit["device_binding"] != device_binding(state):
        raise HardwareSafetyError("Current device protocol/range/identity differs from onsite qualification; acquisition blocked")
