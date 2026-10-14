"""Deterministic, collision-free object layouts for versioned tasks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .catalog import OBJECT_CATALOG
from .constants import (
    OBJECT_GAP,
    ORGANIZE_BOX_INNER_HALF_EXTENTS,
    ORGANIZE_BOX_OUTER_HALF_EXTENTS,
    TABLE_CENTER,
    TABLE_USABLE_RADIUS,
)

ObjectPoses = dict[str, tuple[float, float, float]]
LayoutSelector = Callable[[tuple[str, ...], int], ObjectPoses]
OrganizeDifficulty = str


def _validate(object_ids: tuple[str, ...], seed: int, *, minimum: int, maximum: int, name: str) -> None:
    if not minimum <= len(object_ids) <= maximum:
        raise ValueError(f"{name} layouts require {minimum} to {maximum} objects")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("layout object IDs must be unique")
    unknown = sorted(set(object_ids) - set(OBJECT_CATALOG))
    if unknown:
        raise ValueError(f"layout contains unknown objects: {unknown}")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def select_organize_object_positions(object_ids: tuple[str, ...], seed: int) -> ObjectPoses:
    """Place objects beside the organizer in deterministic wrist-visible slots."""
    _validate(object_ids, seed, minimum=1, maximum=3, name="organizer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4F5247]))
    specs = sorted((OBJECT_CATALOG[name] for name in object_ids), key=lambda item: -item.footprint_radius)
    first_sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
    roles = [("side", first_sign)]
    if len(specs) >= 2:
        roles.append(("side", -first_sign))
    if len(specs) == 3:
        roles.append(("far", first_sign))
    outer_x, outer_y = ORGANIZE_BOX_OUTER_HALF_EXTENTS
    for _restart in range(64):
        placed: ObjectPoses = {}
        occupied: list[tuple[float, float, float]] = []
        for spec, (role, sign) in zip(specs, roles, strict=True):
            radius = spec.footprint_radius
            accepted: tuple[float, float, float] | None = None
            for _ in range(200):
                if role == "side":
                    local_x = -0.05 + float(rng.uniform(-0.01, 0.01))
                    extra_y = 0.045 + float(rng.uniform(-0.01, 0.01))
                else:
                    local_x = 0.14 + float(rng.uniform(-0.01, 0.01))
                    extra_y = 0.075 + float(rng.uniform(-0.01, 0.01))
                local_y = sign * (outer_y + radius + OBJECT_GAP + extra_y)
                if local_x * local_x + local_y * local_y > (TABLE_USABLE_RADIUS - radius) ** 2:
                    continue
                dx = max(abs(local_x) - outer_x, 0.0)
                dy = max(abs(local_y) - outer_y, 0.0)
                if dx * dx + dy * dy < (radius + OBJECT_GAP) ** 2:
                    continue
                if any(
                    (local_x - other_x) ** 2 + (local_y - other_y) ** 2
                    < (radius + other_radius + OBJECT_GAP) ** 2
                    for other_x, other_y, other_radius in occupied
                ):
                    continue
                accepted = (
                    float(TABLE_CENTER[0] + local_x),
                    float(TABLE_CENTER[1] + local_y),
                    float(rng.uniform(-np.pi, np.pi)),
                )
                occupied.append((local_x, local_y, radius))
                break
            if accepted is None:
                break
            placed[spec.object_id] = accepted
        if len(placed) == len(specs):
            return {name: placed[name] for name in object_ids}
    raise ValueError(f"could not create a wrist-visible, collision-free layout for {object_ids}")


def select_organize_batch_object_positions(
    object_ids: tuple[str, ...],
    seed: int,
    *,
    difficulty: OrganizeDifficulty = "standard",
) -> ObjectPoses:
    """Sample bounded source poses for sequential organizer demonstrations.

    The standard distribution retains the established lateral/far source
    regions, while the challenge tier only reduces the object-to-box standoff.
    Both tiers use the same overlap and tabletop checks as task resets.
    """

    _validate(object_ids, seed, minimum=1, maximum=3, name="organizer")
    if difficulty not in {"standard", "challenge"}:
        raise ValueError("organize batch difficulty must be 'standard' or 'challenge'")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4F524742]))
    specs = sorted((OBJECT_CATALOG[name] for name in object_ids), key=lambda item: -item.footprint_radius)
    first_sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
    roles = [("side", first_sign)]
    if len(specs) >= 2:
        roles.append(("side", -first_sign))
    if len(specs) == 3:
        roles.append(("far", first_sign))
    outer_x, outer_y = ORGANIZE_BOX_OUTER_HALF_EXTENTS
    if difficulty == "standard":
        # Keep ordinary demonstrations comfortably clear of the organizer.
        # The tighter side band made an otherwise valid second pick skim a
        # box wall with only millimetres of link clearance.
        side_extra = (0.085, 0.110)
        far_extra = (0.110, 0.140)
        x_jitter = 0.018
    else:
        side_extra = (0.022, 0.040)
        far_extra = (0.040, 0.064)
        x_jitter = 0.026
    for _restart in range(64):
        placed: ObjectPoses = {}
        occupied: list[tuple[float, float, float]] = []
        for spec, (role, sign) in zip(specs, roles, strict=True):
            radius = spec.footprint_radius
            accepted: tuple[float, float, float] | None = None
            for _ in range(240):
                if role == "side":
                    local_x = -0.05 + float(rng.uniform(-x_jitter, x_jitter))
                    extra_y = float(rng.uniform(*side_extra))
                else:
                    local_x = 0.14 + float(rng.uniform(-x_jitter, x_jitter))
                    extra_y = float(rng.uniform(*far_extra))
                local_y = sign * (outer_y + radius + OBJECT_GAP + extra_y)
                if local_x * local_x + local_y * local_y > (TABLE_USABLE_RADIUS - radius) ** 2:
                    continue
                dx = max(abs(local_x) - outer_x, 0.0)
                dy = max(abs(local_y) - outer_y, 0.0)
                if dx * dx + dy * dy < (radius + OBJECT_GAP) ** 2:
                    continue
                if any(
                    (local_x - other_x) ** 2 + (local_y - other_y) ** 2
                    < (radius + other_radius + OBJECT_GAP) ** 2
                    for other_x, other_y, other_radius in occupied
                ):
                    continue
                accepted = (
                    float(TABLE_CENTER[0] + local_x),
                    float(TABLE_CENTER[1] + local_y),
                    float(rng.uniform(-np.pi, np.pi)),
                )
                occupied.append((local_x, local_y, radius))
                break
            if accepted is None:
                break
            placed[spec.object_id] = accepted
        if len(placed) == len(specs):
            return {name: placed[name] for name in object_ids}
    raise ValueError(f"could not create a {difficulty} organizer batch layout for {object_ids}")


def select_organize_box_targets(
    object_ids: tuple[str, ...],
    seed: int,
    *,
    difficulty: OrganizeDifficulty = "standard",
) -> dict[str, tuple[float, float]]:
    """Plan conservative, non-overlapping organizer-relative release points.

    The circle packing approximation deliberately rejects some valid dense
    arrangements.  It keeps the generated corpus away from rim strikes and
    makes room for the gripper to release each object from above.
    """

    _validate(object_ids, seed, minimum=1, maximum=3, name="organizer")
    if difficulty not in {"standard", "challenge"}:
        raise ValueError("organize target difficulty must be 'standard' or 'challenge'")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x424F5854]))
    specs = sorted((OBJECT_CATALOG[name] for name in object_ids), key=lambda item: -item.footprint_radius)
    inner_x, inner_y = ORGANIZE_BOX_INNER_HALF_EXTENTS
    # The target point must leave room not only for the object but also for
    # the wrist and open jaws during the vertical release.  Challenge scenes
    # retain the original denser packing; standard scenes stay well centred.
    wall_margin = 0.040 if difficulty == "standard" else 0.008
    object_gap = 0.025 if difficulty == "standard" else 0.008
    for _restart in range(128):
        placed: dict[str, tuple[float, float, float]] = {}
        for spec in specs:
            radius = spec.footprint_radius
            limit_x = inner_x - radius - wall_margin
            limit_y = inner_y - radius - wall_margin
            if limit_x <= 0.0 or limit_y <= 0.0:
                break
            accepted: tuple[float, float] | None = None
            for _ in range(800):
                local_x = float(rng.uniform(-limit_x, limit_x))
                local_y = float(rng.uniform(-limit_y, limit_y))
                if any(
                    (local_x - other_x) ** 2 + (local_y - other_y) ** 2
                    < (radius + other_radius + object_gap) ** 2
                    for other_x, other_y, other_radius in placed.values()
                ):
                    continue
                accepted = (local_x, local_y)
                placed[spec.object_id] = (local_x, local_y, radius)
                break
            if accepted is None:
                break
        if len(placed) == len(specs):
            return {name: tuple(placed[name][:2]) for name in object_ids}
    raise ValueError(f"could not pack {object_ids} into the organizer for {difficulty} difficulty")


def select_open_table_object_positions(object_ids: tuple[str, ...], seed: int) -> ObjectPoses:
    """Place objects on an open table in deterministic cross-camera slots."""
    _validate(object_ids, seed, minimum=1, maximum=3, name="open-table")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x544852]))
    specs = sorted((OBJECT_CATALOG[name] for name in object_ids), key=lambda item: -item.footprint_radius)
    first_sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
    roles = [("center", 0.0)] if len(specs) == 1 else [("side", first_sign), ("side", -first_sign)]
    if len(specs) == 3:
        roles.append(("center", 0.0))
    side_reference_y = ORGANIZE_BOX_OUTER_HALF_EXTENTS[1]
    for _restart in range(64):
        placed: ObjectPoses = {}
        occupied: list[tuple[float, float, float]] = []
        for spec, (role, sign) in zip(specs, roles, strict=True):
            radius = spec.footprint_radius
            accepted: tuple[float, float, float] | None = None
            for _ in range(200):
                if role == "side":
                    local_x = -0.02 + float(rng.uniform(-0.04, 0.04))
                    local_y = sign * (
                        side_reference_y + radius + OBJECT_GAP + 0.035 + float(rng.uniform(-0.01, 0.01))
                    )
                else:
                    local_x = 0.05 + float(rng.uniform(-0.02, 0.02))
                    local_y = float(rng.uniform(-0.02, 0.02))
                if local_x * local_x + local_y * local_y > (TABLE_USABLE_RADIUS - radius) ** 2:
                    continue
                if any(
                    (local_x - other_x) ** 2 + (local_y - other_y) ** 2
                    < (radius + other_radius + OBJECT_GAP) ** 2
                    for other_x, other_y, other_radius in occupied
                ):
                    continue
                accepted = (
                    float(TABLE_CENTER[0] + local_x),
                    float(TABLE_CENTER[1] + local_y),
                    float(rng.uniform(-np.pi, np.pi)),
                )
                occupied.append((local_x, local_y, radius))
                break
            if accepted is None:
                break
            placed[spec.object_id] = accepted
        if len(placed) == len(specs):
            return {name: placed[name] for name in object_ids}
    raise ValueError(f"could not create a wrist-visible, collision-free layout for {object_ids}")


def select_lift_object_positions(
    object_ids: tuple[str, ...],
    seed: int,
    *,
    x_jitter_m: float = 0.015,
    y_jitter_m: float = 0.015,
) -> ObjectPoses:
    """Place exactly two objects in opposing wrist-visible lateral slots."""
    _validate(object_ids, seed, minimum=2, maximum=2, name="lift")
    if not 0 <= x_jitter_m <= 0.04 or not 0 <= y_jitter_m <= 0.04:
        raise ValueError("lift placement jitter must be within [0, 0.04] m")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4C494654]))
    assigned = list(object_ids)
    if bool(rng.integers(0, 2)):
        assigned.reverse()
    placed: ObjectPoses = {}
    occupied: list[tuple[float, float, float]] = []
    for object_id, sign in zip(assigned, (1.0, -1.0), strict=True):
        radius = OBJECT_CATALOG[object_id].footprint_radius
        accepted: tuple[float, float, float] | None = None
        for _ in range(100):
            local_x = -0.11 + float(rng.uniform(-x_jitter_m, x_jitter_m))
            local_y = sign * (0.20 + float(rng.uniform(-y_jitter_m, y_jitter_m)))
            if local_x * local_x + local_y * local_y > (TABLE_USABLE_RADIUS - radius) ** 2:
                continue
            if any(
                (local_x - other_x) ** 2 + (local_y - other_y) ** 2
                < (radius + other_radius + OBJECT_GAP) ** 2
                for other_x, other_y, other_radius in occupied
            ):
                continue
            accepted = (
                float(TABLE_CENTER[0] + local_x),
                float(TABLE_CENTER[1] + local_y),
                float(rng.uniform(-np.pi, np.pi)),
            )
            occupied.append((local_x, local_y, radius))
            break
        if accepted is None:
            raise ValueError(f"could not place lift object {object_id!r} for seed {seed}")
        placed[object_id] = accepted
    return {name: placed[name] for name in object_ids}


LAYOUT_SELECTORS: dict[str, LayoutSelector] = {
    "bimanual_lateral": select_lift_object_positions,
    "organizer_lateral": select_organize_object_positions,
    "open_table_cross": select_open_table_object_positions,
}


def _circle_clear_of_box(x: float, y: float, radius: float) -> bool:
    dx = max(abs(x) - ORGANIZE_BOX_OUTER_HALF_EXTENTS[0], 0.0)
    dy = max(abs(y) - ORGANIZE_BOX_OUTER_HALF_EXTENTS[1], 0.0)
    return dx * dx + dy * dy >= (radius + OBJECT_GAP) ** 2


def layout_objects(
    selected_ids: Sequence[str],
    layout_seed: int,
    include_box: bool,
    fixed_positions: Mapping[str, tuple[float, float, float | None]] | None = None,
) -> dict[str, tuple[tuple[float, float], float]]:
    """Create a reproducible, non-overlapping tabletop layout.

    Fixed coordinates are world-frame ``(x, y, optional yaw)`` values.  All
    remaining poses are sampled around them with the same overlap checks.
    """
    if layout_seed < 0:
        raise ValueError("layout_seed must be non-negative")
    selected_ids = tuple(selected_ids)
    fixed_positions = dict(fixed_positions or {})
    unknown_fixed = sorted(set(fixed_positions) - set(selected_ids))
    if unknown_fixed:
        raise ValueError(f"Positions were supplied for unselected objects: {', '.join(unknown_fixed)}")

    all_specs = sorted(
        (OBJECT_CATALOG[object_id] for object_id in selected_ids),
        key=lambda spec: (-spec.footprint_radius, spec.object_id),
    )
    random_specs = [spec for spec in all_specs if spec.object_id not in fixed_positions]
    layout_rng = np.random.default_rng(np.random.SeedSequence([layout_seed, 0x4C4159]))
    yaw_rng = np.random.default_rng(np.random.SeedSequence([layout_seed, 0x594157]))
    yaws = {
        spec.object_id: float(yaw_rng.uniform(-math.pi, math.pi))
        for spec in sorted(all_specs, key=lambda item: item.object_id)
    }

    fixed_trial: dict[str, tuple[float, float, float]] = {}
    for object_id, (world_x, world_y, supplied_yaw) in fixed_positions.items():
        if not all(math.isfinite(value) for value in (world_x, world_y)):
            raise ValueError(f"Position for {object_id} must contain finite values")
        if supplied_yaw is not None:
            if not math.isfinite(supplied_yaw):
                raise ValueError(f"Yaw for {object_id} must be finite")
            yaws[object_id] = supplied_yaw
        radius = OBJECT_CATALOG[object_id].footprint_radius
        x = world_x - TABLE_CENTER[0]
        y = world_y - TABLE_CENTER[1]
        max_center_radius = TABLE_USABLE_RADIUS - radius
        if x * x + y * y > max_center_radius * max_center_radius:
            raise ValueError(f"Position for {object_id} lies outside the usable tabletop")
        if include_box and not _circle_clear_of_box(x, y, radius):
            raise ValueError(f"Position for {object_id} overlaps the central box")
        if any(
            (x - other_x) ** 2 + (y - other_y) ** 2 < (radius + other_radius + OBJECT_GAP) ** 2
            for other_x, other_y, other_radius in fixed_trial.values()
        ):
            raise ValueError(f"Position for {object_id} overlaps another fixed object")
        fixed_trial[object_id] = (x, y, radius)

    placed: dict[str, tuple[float, float, float]] | None = None
    for _restart in range(48):
        trial = dict(fixed_trial)
        restart_offset = float(layout_rng.uniform(-math.pi, math.pi))
        for spec in random_specs:
            radius = spec.footprint_radius
            max_center_radius = TABLE_USABLE_RADIUS - radius
            if max_center_radius <= 0:
                break
            candidates: list[tuple[float, float]] = []
            for _ in range(2500):
                polar_radius = max_center_radius * math.sqrt(float(layout_rng.random()))
                angle = float(layout_rng.uniform(-math.pi, math.pi))
                candidates.append((polar_radius * math.cos(angle), polar_radius * math.sin(angle)))
            # Deterministic rings are a packing fallback for crowded layouts.
            for fraction in (1.0, 0.92, 0.84, 0.74, 0.62, 0.48, 0.32):
                ring_radius = max_center_radius * fraction
                for angle_index in range(24):
                    angle = restart_offset + angle_index * 2 * math.pi / 24
                    candidates.append((ring_radius * math.cos(angle), ring_radius * math.sin(angle)))

            position: tuple[float, float] | None = None
            for x, y in candidates:
                if x * x + y * y > max_center_radius * max_center_radius:
                    continue
                if include_box and not _circle_clear_of_box(x, y, radius):
                    continue
                if any(
                    (x - other_x) ** 2 + (y - other_y) ** 2 < (radius + other_radius + OBJECT_GAP) ** 2
                    for other_x, other_y, other_radius in trial.values()
                ):
                    continue
                position = (x, y)
                break
            if position is None:
                break
            trial[spec.object_id] = (*position, radius)
        if len(trial) == len(all_specs):
            placed = trial
            break

    if placed is None:
        joined = ", ".join(selected_ids) or "none"
        suggestion = (
            " Try fewer objects, another --layout-seed, or --no-box."
            if include_box
            else " Try fewer objects or another --layout-seed."
        )
        raise ValueError(f"Could not place object selection on the round table: {joined}.{suggestion}")

    return {
        object_id: (
            (TABLE_CENTER[0] + placed[object_id][0], TABLE_CENTER[1] + placed[object_id][1]),
            yaws[object_id],
        )
        for object_id in selected_ids
    }
