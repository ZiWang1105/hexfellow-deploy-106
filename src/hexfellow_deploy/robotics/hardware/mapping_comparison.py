"""Pure pose comparison and recorded feedback loading; no device connections."""

import copy
import json
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..paths import CONFIG_DIR, PROJECT_ROOT, config_reference
from .research_profiles import RUNTIME as PRECISION_MAPPING
from .whole_body import SLICES, RobotMapping, vector

LEGACY_DRAG_MAPPING = CONFIG_DIR / "hardware/hexfellow_drag_runtime.json"
CORRECTED_DRAG_MAPPING = CONFIG_DIR / "hardware/hexfellow_drag_j35_runtime.json"
CURRENT_HIGHSPEED_MAPPING = CONFIG_DIR / "hardware/hexfellow_gp80_highspeed_j357.json"


def replay_mapping_path(path):
    """Select a known runtime by the recorded hash, never by approximate signs."""
    directory = Path(path)
    if (directory / "session").is_dir():
        directory /= "session"
    with (directory / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "configuration":
                recorded_hash = row.get("mapping_sha256")
                for config in (LEGACY_DRAG_MAPPING, CORRECTED_DRAG_MAPPING, PRECISION_MAPPING,
                               CONFIG_DIR / 'hardware/hexfellow_gp80_highspeed_fixed_base.json',
                               CURRENT_HIGHSPEED_MAPPING):
                    if RobotMapping.load(config).sha256 == recorded_hash:
                        return config
                raise ValueError("unknown recorded mapping hash; supply its exact --robot-config")
            if row.get("kind") in {"initial_state", "cycle"}:
                break
    raise ValueError("recording has no preceding mapping binding")


def comparison_labels(joints):
    label = "+".join(f"J{joint}" for joint in joints)
    return ("original", f"left {label} reversed", f"right {label} reversed", f"both {label} reversed")


def comparison_mappings(mapping, joints=(3,)):
    """Variants are relative to the explicitly selected source mapping."""
    result = []
    for sides in ((), ("left",), ("right",), ("left", "right")):
        data = copy.deepcopy(mapping.data)
        for side in sides:
            for joint in joints:
                data["arms"][side]["signs"][joint - 1] *= -1
        result.append(RobotMapping(data, "display-only"))
    return result


def remap_state(state, mapping):
    """Recompute arms from raw SDK feedback, never from already remapped q."""
    q = vector(state.q, 18, "recorded/measured pose")
    for side in ("left", "right"):
        raw = state.health[side]["pos"]
        q[SLICES[side]], _ = mapping.arm_state(side, raw, np.zeros(7))
    return SimpleNamespace(q=q, health=state.health, received=state.received)


def joint_text(state, mapping, joints=(3,)):
    rows = []
    for side in ("left", "right"):
        entry = mapping.data["arms"][side]
        raw = vector(state.health[side]["pos"], 7, f"{side} SDK positions")
        model = state.q[SLICES[side]]
        for joint in joints:
            index = joint - 1
            rows.append(f"{side} J{joint} SDK {np.rad2deg(raw[entry['sdk_indices'][index]]):+.2f} deg"
                        f" -> model {np.rad2deg(model[index]):+.2f} deg")
    return "\n".join(rows)


def replay_source_hash(mapping):
    source = mapping.data.get("candidate_source")
    if source is None:
        return mapping.sha256
    original = RobotMapping.load(config_reference(mapping.source_path, source["path"]))
    if original.sha256 != source["sha256"]:
        raise ValueError("candidate source mapping hash mismatch")
    # Recorded base/lift q cannot be reinterpreted without their full raw state.
    for key in ("lift_position_mode", "lift_zero_offset_m", "base_sdk_to_model_yaw_rad"):
        if mapping.data.get(key) != original.data.get(key):
            raise ValueError(f"replay body mapping differs: {key}")
    return original.sha256


class ReplayTrace:
    """Compact chronological snapshots, including the final failed cycle."""

    def __init__(self, path, mapping):
        directory = Path(path)
        if (directory / "session").is_dir():
            directory /= "session"
        expected_hash = replay_source_hash(mapping)
        self.frames = []
        checked = False
        with (directory / "trace.jsonl").open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if row.get("kind") == "configuration":
                        if row.get("mapping_sha256") != expected_hash:
                            raise ValueError("recorded mapping hash differs from selected source mapping")
                        checked = True
                    if row.get("kind") not in {"initial_state", "cycle"}:
                        continue
                    if not checked:
                        raise ValueError("recording has no preceding mapping binding")
                    self._append(row, mapping)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"trace.jsonl line {line_number}: {error}") from error
        failure = directory / "failed_cycle.json"
        if failure.exists():
            if not checked:
                raise ValueError("recording has no mapping binding")
            row = json.loads(failure.read_text())
            row["phase"] = "fault"
            self._append(row, mapping)
        if not self.frames:
            raise ValueError("recording contains no feedback frames")
        self.phase_starts = [i for i, frame in enumerate(self.frames)
                             if i == 0 or frame.phase != self.frames[i - 1].phase]
        self.index = next((i for i in self.phase_starts if self.frames[i].phase == "ready"), 0)

    def _append(self, row, mapping):
        raw = row["state"]
        received = {k: float(v) for k, v in raw["received"].items()}
        if not received or not np.all(np.isfinite(list(received.values()))):
            raise ValueError("invalid recorded receive times")
        at = float(row.get("at", max(received.values())))
        if not np.isfinite(at):
            raise ValueError("invalid recorded time")
        # A failed cycle can omit its host timestamp; feedback still orders it.
        if self.frames:
            at = max(at, self.frames[-1].at)
        health = {side: {key: value for key, value in values.items()
                         if key in {"pos", "vel", "calibrated", "range"}}
                  for side, values in raw["health"].items()}
        state = remap_state(SimpleNamespace(q=raw["q"], health=health, received=received), mapping)
        self.frames.append(SimpleNamespace(state=state, at=at, phase=row.get("phase", "initial")))

    @property
    def frame(self):
        return self.frames[self.index]

    def step(self, amount):
        self.index = int(np.clip(self.index + amount, 0, len(self.frames) - 1))

    def jump(self, direction):
        position = bisect_right(self.phase_starts, self.index) - 1
        position = int(np.clip(position + direction, 0, len(self.phase_starts) - 1))
        self.index = self.phase_starts[position]
