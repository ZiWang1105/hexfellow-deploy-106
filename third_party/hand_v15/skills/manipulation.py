"""Hand-designed manipulation skills (library ``hand_v6``@3.2.0 = hand_v5@3.1.0 + candidate cand_20260910_121426_b48287, taken as
validated: hand_v1 lineage; hand_v4 status semantics + the reach-edge / motion-time revision of candidate cand_20260909_221511_026ccd
merged by hand).

Composed from primitives through ``ctx.call``. All effect checks use robot observables only: fingertip FK, gripper
aperture after closing (blocked close => object between fingers) and a re-check after lifting (object still held).

hand_v4 behaviour kept unchanged: ``place`` landed-early success + post-release fingers-opened check
(cand_20260909_214510_0ef7c7). Candidate 026ccd changes (P6 reach-edge accuracy, P9 motion time):
  * every ``move_to`` omits ``duration`` (distance-scaled, 0.4-1.5 s) instead of the fixed 1.0-1.5 s; approach, lift and
    transit moves run in the fast mode (``correction_passes=0``) - the descent / release point is what gets corrected;
  * approach / lift heights come from a ONE-branch column check (``controller.solve_column``) and back off from the reach
    ceiling to a pose with >= MIN_COLUMN_MARGIN_RAD joint-limit margin, so the descent cannot flip the wrist branch;
  * descent residuals (``tip_error_m`` before/after correction, ``limit_margin_rad``) and the robot time of every skill are
    reported; a descent that misses by more than the tolerance is retried once from 3 cm up before failing, and the
    failure message says when the pose is at the reach edge (pull_toward_base);
  * ``place`` transits along a straight Cartesian line (no dipping of the held object) with a joint-space fallback.

Candidate cand_20260910_121426_b48287 (truth-checked diagnoses of the remaining hand_v5 single-seed failures, docs in the candidate record):
  * ``_clear_other_arm``: a ``place`` retreats straight up and stays parked above its release point; the other arm's next
    transit / approach at the same xy ran into it (every oracle stack_blocks failure: 11-16 cm transit misses, a 6.5 cm
    approach miss). ``pick`` and ``place`` now move the idle arm toward its own side first when its fingertip is parked
    within IDLE_ARM_CLEAR_RADIUS_M of the target column (proprioception only, ~0.6 s, only when needed).
  * ``place`` accepts a release point that is reachable although no 3 cm column exists above it (a stack layer at the reach
    ceiling: the old check refused "release point not reachable" although the point was reachable), retries a missed
    transit at lower heights (margin-preferred height first, then 2 cm steps down to MIN_TRANSIT_M) instead of the single
    -4 cm retry that a low transit could not use, and retreats at least RETREAT_MIN_M after the release (the open pads end
    1.7 cm below the tip: a 1-3 cm retreat left them beside the placed object).
  * ``pick`` lowers its approach height when the move itself reports the column-solver height unreachable (IK seeds
    disagree at the reach edge; oracle stack_blocks s1 lost a block to that).
  * ``move_above`` clamps the requested height to the reach ceiling above the point instead of failing (cand_20260910_100451_f2c92a).

hand_v8@3.4.0 (parent hand_v7): the held / empty decision is no longer a constant in this library. ``_held`` reads the reading's calibrated width rule (``GripperReading.held_by_width``: opening >= the backend/side ``held_width_min_m`` of configs/gripper/<backend>_<side>.yaml; ambiguous and empty count as not held) and ``_held_width_min`` exposes that threshold to the pinch / transfer checks. No other behaviour change.

hand_v9@3.5.0 (parent hand_v8): correction of the hand_v7 integration defect (docs/evaluation_protocol.md amendment 9 addendum). This file is hand_v8's with the candidate cand_20260910_164848_9a670e ``Place`` hunk (transit_height lower bound 0.05 -> 0.01, the skill already lands on a >= 1 cm column; place 1.6.0 = hand_v8's 1.5.0 threshold change + the candidate's 1.5.0 transit change). The calibrated held / empty width rule of hand_v8 (``_held`` / ``_held_width_min`` from configs/gripper/<backend>_<side>.yaml) is kept everywhere; no width constant.

Candidate cand_20260911_135918_55d735 (parent hand_v13@3.9.0): base Pick 1.3.0 - optional ``pad_floor_z`` (the closed-finger pad bottoms must clear a caller's floor, e.g. the rim of a container the object stands in): ``_pad_floor`` takes the floor, ``_pad_gate`` measures the pad height after the descent (fingertip FK minus the calibrated pad offset) and raises a low tip by the measured deficit before the fingers close. Additive; every other skill of this module unchanged.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field

from physapi.schemas.core import SkillStatus
from physapi.skills.base import Params, Skill, SkillError

Side = Literal["left", "right"]
#: hand_v8: no held / empty width constant here - the threshold is the gripper calibration's ``held_width_min_m`` (per backend and
#: side, configs/gripper/*.yaml), read through the executor context (see ``_held`` / ``_held_width_min``).
DESCENT_TOLERANCE_M = 0.015
MIN_COLUMN_MARGIN_RAD = 0.06  # preferred joint-limit margin of approach/lift/transit heights: poses closer to the reach ceiling never settle (2 s, 1.5-2 cm off)
MAX_MARGIN_BACKOFF_M = 0.03  # the margin preference may lower an approach / lift / transit height by at most this much below the reach ceiling
REACH_EDGE_HINT = "the grasp pose is at the arm's reach edge (low joint-limit margin); pull_toward_base brings the object into the accurate zone, or use the other arm"
#: The other arm's fingertip closer than this (xy) to a target column, and lower than IDLE_ARM_WORKSPACE_Z_M above the table,
#: is in the way of the active arm's approach / transit (the gripper bodies are ~10 cm wide, the wrists reach further).
IDLE_ARM_CLEAR_RADIUS_M = 0.18
IDLE_ARM_WORKSPACE_Z_M = 0.22  # a tip higher than this above the table (e.g. the home pose, +0.36 m) is out of the way
IDLE_ARM_CLEAR_STEP_M = 0.22  # distance of the clearing pose from the target column, toward the idle arm's own side
#: Place transit: the smallest carry height above the release point (a stack layer at the reach ceiling may have no 3 cm
#: column above it), the step of the lower retries and their count.
MIN_TRANSIT_M = 0.01
TRANSIT_STEP_M = 0.02
MAX_TRANSIT_RETRIES = 3
#: Post-release retreat: at least this far above the release point when reachable (open pads end 1.7 cm below the tip).
RETREAT_MIN_M = 0.06
#: Lateral alignment over the release point before the release descent: a fast transit that ends further off than this is
#: corrected in place (nothing to push against up there), and lowered to the highest height with joint-limit margin when the
#: correction cannot settle (a ceiling pose sags 1-3 cm; the release descent has no correction passes by design, so the error it
#: starts with is the error the object lands with - 1.7 cm lateral on a stack layer tipped the block off its support).
ALIGN_TOL_M = 0.006
#: Approach lowering when move_to rejects the column-solver height as unreachable (at most this many 1 cm steps).
MAX_APPROACH_LOWERING = 3


def _held_width_min(ctx, side: str) -> float:
    """The calibrated held threshold [m] of this backend / side (configs/gripper/<backend>_<side>.yaml ``held_width_min_m``) -
    the ONLY source of the held / empty width decision in this library (hand_v8; hand_v7 used a 1 cm constant)."""

    r = ctx.gripper_reading(side)
    if getattr(r, "held_width_min_m", None) is not None:
        return float(r.held_width_min_m)
    return float(ctx.gripper_adapter.calibration(side).held_width_min_m)


def _held(ctx, side: str) -> tuple[bool, float | None]:
    """(something between the fingers, opening width m) from the calibrated reading's width rule: ``held_by_width`` (opening >=
    held_width_min_m); ``ambiguous`` and ``empty`` count as not held. Raw fallback through the same threshold in raw units when
    the reading is invalid."""

    r = ctx.gripper_reading(side)
    if r.opening_width_m is not None and r.valid:
        return bool(r.held_by_width), r.opening_width_m
    return ctx.gripper(side) > ctx.gripper_adapter.calibration(side).raw_held_threshold, None


def _xyz(v):
    return [float(x) for x in v]


#: Lowest admissible finger-pad bottom above the table for a grasp [m]: pads commanded into the table stall the descent 1-1.5 cm high
#: and then cannot slide closed (friction with the table) - the blocked opening reads as "held" with nothing between the pads
#: (formal_v11 capx_nut_assembly s102 / s103: 1 cm tall handle, tip asked 0.7-1.2 cm above the table = pads 0.5-1 cm below it).
PAD_FLOOR_CLEARANCE_M = 0.003


def _pad_floor(ctx, z: float, floor_z: float | None = None) -> tuple[float, float]:
    """Raise a grasp height so the closed-finger pad bottoms stay PAD_FLOOR_CLEARANCE_M above the table (the pad offset is the
    cell calibration's ``finger_pad_below_tip_m``), or above ``floor_z`` when a higher floor is given (pick 1.8.0 ``pad_floor_z``:
    the rim of a container the object stands in). Returns (z, raised_by_m)."""

    floor = float(ctx.table_z) + float(ctx.cell_calibration.finger_pad_below_tip_m) + PAD_FLOOR_CLEARANCE_M
    if floor_z is not None:
        floor = max(floor, float(floor_z) + float(ctx.cell_calibration.finger_pad_below_tip_m) + PAD_FLOOR_CLEARANCE_M)
    if z < floor:
        return floor, round(floor - z, 4)
    return z, 0.0


#: pick 1.8.0 measured pad gate: after the descent the pad bottoms (fingertip FK minus the calibrated pad offset) must clear the caller's
#: ``pad_floor_z`` by at least this much before the fingers close; a lower tip is raised by the MEASURED deficit (at most PAD_GATE_PASSES
#: moves, each biased by the residual of the previous one). formal_v13 abc_extract s103: a descent accepted 0.6 cm low (tolerance 1.5 cm)
#: put the pads 3 mm below the cup rim and the fingers closed on peg AND cup.
PAD_GATE_MARGIN_M = 0.001
PAD_GATE_PASSES = 2


def _pad_gate(ctx, side: str, x: float, y: float, z_cmd: float, yaw: float, floor_z: float) -> dict:
    """Raise the fingertip until the closed-finger pad bottoms measure >= ``floor_z`` + PAD_GATE_MARGIN_M (deployable: fingertip FK and
    the calibrated pad offset). Returns a record {pad_bottom_before_m, pad_bottom_after_m, deficit_m, raised_m, moves, ok}."""

    pad = float(ctx.cell_calibration.finger_pad_below_tip_m)
    tip = ctx.tip(side)
    before = float(tip[2]) - pad
    rec = {"floor_z": round(float(floor_z), 4), "pad_bottom_before_m": round(before, 4), "deficit_m": 0.0, "raised_m": 0.0, "moves": 0, "ok": True}
    target_cmd = float(z_cmd)
    for _ in range(PAD_GATE_PASSES):
        tip = ctx.tip(side)
        deficit = float(floor_z) + PAD_GATE_MARGIN_M - (float(tip[2]) - pad)
        if deficit <= 0.0:
            break
        # command relative to where the tip IS (the descent's residual at this pose is systematic: bias the command by it)
        target_cmd = target_cmd + deficit + 0.001
        r = ctx.call("move_to", side=side, xyz=[x, y, target_cmd], yaw=yaw, tolerance_m=0.005, correction_passes=2)  # move_to's tolerance floor is 5 mm
        rec["moves"] += 1
        if r.status == SkillStatus.error:
            rec["error"] = r.message[:120]
            break
    tip = ctx.tip(side)
    after = float(tip[2]) - pad
    rec.update({"pad_bottom_after_m": round(after, 4), "deficit_m": round(max(0.0, float(floor_z) + PAD_GATE_MARGIN_M - before), 4), "raised_m": round(after - before, 4), "ok": bool(after >= float(floor_z))})
    return rec


def _robot_time(ctx, tick0: int) -> float:
    return round((ctx.backend.control_step_index - tick0) / ctx.control_hz, 3)


def _reachable_height(ctx, side: str, x: float, y: float, z: float, wanted: float, yaw: float, *, minimum: float = 0.03, prefer_margin: bool = True) -> float | None:
    """Largest height <= wanted (step 1 cm) above (x, y, z) such that the grasp point AND the point at that height are
    reachable on ONE wrist branch; None if even the grasp point is not reachable.

    ``prefer_margin`` (default) backs off from the reach ceiling to the highest height whose pose keeps
    MIN_COLUMN_MARGIN_RAD of joint-limit margin (approach / lift / transit poses of pick and place: ceiling poses never
    settle). ``prefer_margin=False`` returns the TRUE ceiling (highest reachable column height) for callers that need every
    centimetre, e.g. the lift that has to bring an object's bottom above a container rim (extract_from_container: the 1-2 cm
    back-off left the peg 0.1 cm below the rim in the hand_v5 regression)."""

    ctrl = ctx.controller
    if ctrl.solve_ik(side, [x, y, z], yaw) is None:
        return None
    first_reachable = None
    h = float(wanted)
    while h >= minimum - 1e-9:
        chain = ctrl.solve_column(side, (x, y), yaw, [z, z + h])
        if chain is not None:
            if first_reachable is None:
                first_reachable = round(h, 3)
            if not prefer_margin:
                return first_reachable
            # a pose ON a joint limit (margin ~0) cannot even hold position under load; back off a little from the
            # reach ceiling when a slightly lower height has margin, otherwise accept the highest reachable one.
            # The margin is judged on the solution move_to will actually use (branch continuity from the current
            # configuration), not only on the column chain: at ceiling poses the continuity branch can sit on a limit
            # while the other branch, a 2-3 rad wrist swing away, has margin.
            actual = ctrl.solve_ik_detailed(side, [x, y, z + h], yaw)
            if min(c.limit_margin_rad for c in chain) >= MIN_COLUMN_MARGIN_RAD and actual is not None and actual.limit_margin_rad >= MIN_COLUMN_MARGIN_RAD and actual.joint_distance_rad < 1.5:
                return round(h, 3)
            if first_reachable - h >= MAX_MARGIN_BACKOFF_M - 1e-9:
                # the margin would cost more than MAX_MARGIN_BACKOFF_M of height: a transit / approach that low can collide
                # with a container rim or a stack (the peg place in the hand_v5 regression transited 5 cm above the target
                # instead of 10), so the ceiling pose is used and the arrival check (3 cm) / the lower retry handle it
                return first_reachable
        h -= 0.01
    if first_reachable is not None:
        return first_reachable
    if ctrl.solve_column(side, (x, y), yaw, [z, z + minimum]) is not None:
        return round(minimum, 3)
    return None


def _margin_height(ctx, side: str, x: float, y: float, z: float, wanted: float, yaw: float, *, minimum: float = MIN_TRANSIT_M) -> float | None:
    """Highest height <= wanted (1 cm grid) above (x, y, z) whose IK solution keeps MIN_COLUMN_MARGIN_RAD of joint-limit
    margin (the pose the servos can actually hold), or None."""

    h = float(wanted)
    while h >= minimum - 1e-9:
        sol = ctx.controller.solve_ik_detailed(side, [x, y, z + h], yaw)
        if sol is not None and sol.limit_margin_rad >= MIN_COLUMN_MARGIN_RAD:
            return round(h, 3)
        h -= 0.01
    return None


def _ceiling_above(ctx, side: str, x: float, y: float, z: float, yaw: float) -> float | None:
    """Highest top-down reachable z above (x, y) between z - 0.05 and z + 0.25 (1 cm grid), for failure messages."""

    best = None
    for dz in np.arange(-0.05, 0.251, 0.01):
        if ctx.controller.solve_ik(side, [x, y, z + dz], yaw) is not None:
            best = round(float(z + dz), 3)
    return best


def _clear_other_arm(ctx, side: str, x: float, y: float) -> dict | None:
    """Move the OTHER arm's fingertip out of the neighbourhood of the target column (x, y) when it is parked there.

    A place retreats straight up and leaves the arm above its release point; the other arm's next transit / approach at
    the same xy then collides with it (every oracle stack_blocks failure of hand_v5: transit misses of 11-16 cm and a
    6.5 cm approach miss at the parked arm's column). Deployable evidence only (fingertip FK of both arms). The idle arm
    is shifted toward its own side at its current height (or a little up / back when that pose is unreachable). Returns
    a small record for the metrics, or None when the other arm was not in the way."""

    other = "right" if side == "left" else "left"
    tip = ctx.tip(other)
    dist = math.hypot(float(tip[0]) - x, float(tip[1]) - y)
    if dist >= IDLE_ARM_CLEAR_RADIUS_M or float(tip[2]) > ctx.table_z + IDLE_ARM_WORKSPACE_Z_M:
        return None
    own = 1.0 if other == "left" else -1.0
    z = max(float(tip[2]), ctx.table_z + 0.10)
    # the clearing pose sits IDLE_ARM_CLEAR_STEP_M from the target column toward the idle arm's own side (wherever the idle
    # tip is now - an arm parked exactly on the column needs the full step), at its current height or a little back / up
    ty = y + own * IDLE_ARM_CLEAR_STEP_M
    candidates = (
        [float(tip[0]), ty, z],
        [float(tip[0]) - 0.05, ty, z],
        [float(tip[0]) - 0.08, ty, z + 0.04],
        [float(tip[0]), ty, z + 0.06],
        [float(tip[0]) - 0.10, y + own * (IDLE_ARM_CLEAR_STEP_M - 0.03), z + 0.04],
    )
    for cand in candidates:
        if math.hypot(cand[0] - x, cand[1] - y) < IDLE_ARM_CLEAR_RADIUS_M:
            continue
        if ctx.controller.solve_ik(other, cand, 0.0) is None:
            continue
        r = ctx.call("move_to", side=other, xyz=[round(float(v), 4) for v in cand], yaw=0.0, tolerance_m=0.05, correction_passes=0)
        return {"arm": other, "from": np.round(tip, 3).tolist(), "to": [round(float(v), 3) for v in cand], "distance_before_m": round(dist, 3), "moved": r.status == SkillStatus.success}
    return {"arm": other, "from": np.round(tip, 3).tolist(), "to": None, "distance_before_m": round(dist, 3), "moved": False}


def _approach(ctx, side: str, x: float, y: float, z: float, approach: float, yaw: float, open_aperture: float, *, minimum: float = 0.03):
    """Approach move above the grasp point; when move_to rejects the column-solver height as unreachable (the IK seeds
    disagree at the reach edge) the height is lowered in 1 cm steps (at most MAX_APPROACH_LOWERING). Returns (result, height)."""

    above = ctx.call("move_to", side=side, xyz=[x, y, z + approach], yaw=yaw, gripper=open_aperture, correction_passes=0)
    lowered = 0
    while above.status != SkillStatus.success and above.metrics.get("reachable") is False and lowered < MAX_APPROACH_LOWERING and approach - 0.01 >= minimum - 1e-9:
        approach = round(approach - 0.01, 3)
        lowered += 1
        above = ctx.call("move_to", side=side, xyz=[x, y, z + approach], yaw=yaw, gripper=open_aperture, correction_passes=0)
    return above, approach


def _descend(ctx, side: str, xyz, yaw: float, approach: float, *, tolerance_m: float = DESCENT_TOLERANCE_M):
    """Descend to the grasp point; one retry from 3 cm up when the fingertip misses. Returns (result, retried)."""

    x, y, z = xyz
    down = ctx.call("move_to", side=side, xyz=[x, y, z], yaw=yaw, tolerance_m=tolerance_m)
    if down.status == SkillStatus.success or down.metrics.get("reachable") is False:
        return down, False
    ctx.call("move_to", side=side, xyz=[x, y, z + min(0.03, approach)], yaw=yaw)
    down = ctx.call("move_to", side=side, xyz=[x, y, z], yaw=yaw, tolerance_m=tolerance_m)
    return down, True


def _transit(ctx, side: str, x: float, y: float, z_low: float, transit: float, yaw: float):
    """Carry move to ``transit`` above (x, y, z_low): a straight Cartesian line (no dipping of the held object while the
    arm reconfigures) with a joint-space fallback. A miss (a ceiling pose: the servos sag at a joint limit) is retried
    lower - first at the highest height with joint-limit margin, then in TRANSIT_STEP_M steps down to MIN_TRANSIT_M, at
    most MAX_TRANSIT_RETRIES times (hand_v5 had one retry 4 cm lower that a transit below 7 cm could never use).
    Returns (result, transit_height_used, heights_tried)."""

    r = ctx.call("move_to", side=side, xyz=[x, y, z_low + transit], yaw=yaw, cartesian=True, correction_passes=0)
    if r.status == SkillStatus.failure and r.metrics.get("reachable") is False:
        r = ctx.call("move_to", side=side, xyz=[x, y, z_low + transit], yaw=yaw, correction_passes=0)
    tried = [round(transit, 3)]
    if r.status == SkillStatus.success:
        return r, transit, tried
    h = float(transit)
    margin_h = _margin_height(ctx, side, x, y, z_low, transit - 0.01, yaw)
    while r.status != SkillStatus.success and len(tried) <= MAX_TRANSIT_RETRIES:
        nxt = round(h - TRANSIT_STEP_M, 3)
        if margin_h is not None and margin_h < h - 1e-9:
            nxt = min(nxt, margin_h) if nxt >= MIN_TRANSIT_M - 1e-9 else margin_h
            margin_h = None
        if nxt < MIN_TRANSIT_M - 1e-9:
            if h > MIN_TRANSIT_M + 1e-9:
                nxt = MIN_TRANSIT_M
            else:
                break
        h = nxt
        r = ctx.call("move_to", side=side, xyz=[x, y, z_low + h], yaw=yaw, correction_passes=0)
        tried.append(round(h, 3))
    return r, (round(h, 3) if r.status == SkillStatus.success else tried[-1]), tried


def _align_over(ctx, side: str, x: float, y: float, z_low: float, transit: float, yaw: float, over):
    """Measured-pose gate over the release point: correct a transit residual > ALIGN_TOL_M in place (correction passes), and
    when the pose cannot settle (ceiling), lower to the highest height with joint-limit margin and correct there. Returns
    (transit_height, tip_error_m, correction_moves)."""

    err = (over.metrics or {}).get("tip_error_m")
    if err is None or err <= ALIGN_TOL_M:
        return transit, err, 0
    moves = 0
    r = ctx.call("move_to", side=side, xyz=[x, y, z_low + transit], yaw=yaw, tolerance_m=0.03, correction_passes=2)
    moves += 1
    err = float(r.metrics.get("tip_error_m", err))
    if err > ALIGN_TOL_M:
        lower = _margin_height(ctx, side, x, y, z_low, transit - 0.01, yaw)
        if lower is not None and lower < transit - 1e-9:
            r2 = ctx.call("move_to", side=side, xyz=[x, y, z_low + lower], yaw=yaw, tolerance_m=0.03, correction_passes=2)
            moves += 1
            if r2.status == SkillStatus.success:
                transit, err = lower, float(r2.metrics.get("tip_error_m", err))
    return transit, err, moves


def _retreat_height(ctx, side: str, x: float, y: float, z_low: float, transit: float, yaw: float) -> float:
    """Height of the post-release retreat above the release point: at least RETREAT_MIN_M when reachable (the open pads
    end 1.7 cm below the tip, so a 1-3 cm retreat leaves them beside the placed object and the next lateral move would
    clip it), otherwise the highest reachable of 5 / 4 cm, otherwise the transit height."""

    for h in sorted({round(max(transit, RETREAT_MIN_M), 3), 0.05, 0.04}, reverse=True):
        if h <= transit + 1e-9 or ctx.controller.solve_ik(side, [x, y, z_low + h], yaw) is not None:
            return h
    return transit


#: A release descent that stops at least this high above the release point is a touchdown of the held object (or a
#: collision): no correction pass may push on (it drags the object sideways - 1.6 cm in the hand_v5 regression).
TOUCHDOWN_DZ_M = 0.01
#: Slowest / fastest release descent: the distance-scaled 0.4 s minimum lands a held object at ~35 cm/s, and the impact of
#: an early touchdown deflected the cube 1.6 cm sideways; hand_v4 descended in 1.0 s (0.7 cm).
RELEASE_DESCENT_MIN_S = 0.6
RELEASE_DESCENT_SPEED_MPS = 0.25


def _release_descent(ctx, side: str, target, yaw: float, transit: float):
    """Descent to the release point WITHOUT Cartesian correction passes but with the full joint-space settle
    (``correction_passes=0, fast=False``) and a gentler duration (``dist / RELEASE_DESCENT_SPEED_MPS``, at least
    RELEASE_DESCENT_MIN_S): a held object that touches down early stops the arm, and a Cartesian correction that
    re-solves the residual would push on and drag the object sideways (1.6 cm in the hand_v5 regression; the hand_v4
    landed-early semantics rely on the touchdown being felt, not pushed). A two-segment variant with a corrected
    pre-stop was tried and cost ~1 s per place, which pushed the 4-object relays over the 60 s robot-time budget.
    Returns the move_to result; metrics.touchdown_suspected marks a stop >= TOUCHDOWN_DZ_M above the target."""

    target = np.asarray(target, dtype=float)
    dist = float(np.linalg.norm(ctx.tip(side) - target))
    duration = float(np.clip(dist / RELEASE_DESCENT_SPEED_MPS, RELEASE_DESCENT_MIN_S, 1.5))
    down = ctx.call("move_to", side=side, xyz=target.tolist(), yaw=yaw, tolerance_m=0.02, correction_passes=0, fast=False, duration=round(duration, 3))
    down.metrics["touchdown_suspected"] = bool(float(ctx.tip(side)[2] - target[2]) >= TOUCHDOWN_DZ_M)
    down.metrics["transit_height"] = transit
    return down


def _descent_metrics(down) -> dict:
    m = (down.metrics or {}) if down is not None else {}
    return {k: m.get(k) for k in ("tip_error_m", "tip_error_m_before_correction", "correction_passes", "limit_margin_rad", "manipulability") if k in m}


def _edge_hint(down) -> str:
    margin = ((down.metrics or {}) if down is not None else {}).get("limit_margin_rad")
    return "; " + REACH_EDGE_HINT if margin is not None and margin < 0.03 else ""


class Pick(Skill):
    name = "pick"
    version = "1.3.0"
    description = "Top-down pick: open, approach above, descend, close, lift. Evidence: calibrated opening after closing and after lifting. Distance-scaled durations; a descent > 1.5 cm off is retried once; residual and joint-limit margin reported (reach edge: pull_toward_base). A grasp height that would press the finger pads into the table is raised to the pad floor (pads 3 mm above the table; metrics.grasp_z_raised_m) - thin flat objects are grasped at that floor. `pad_floor_z` (e.g. a container rim the object stands in): the pads must clear it - the grasp height is raised to it and the MEASURED pad height after the descent is checked and corrected before closing (metrics.pad_gate)."
    returns = "success if held after closing and after lifting (outputs.lift_pos, held, opening_width_m); failure: unreachable, descent missed, closed on nothing, lost in lift. metrics: tip_error_m, limit_margin_rad, descent_retried, robot_time_s"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "calibrated opening after close and lift; descent FK residual"
    observations_used = ["proprio"]
    dependencies = ["move_to", "set_gripper"]
    known_limitations = ["top-down grasps; yaw across the object's short axis"]
    timeout_s = 15.0

    class Params(Params):
        side: Side
        xyz: list[float] = Field(min_length=3, max_length=3, description="object centre [m]")
        yaw: float = Field(default=0.0, description="[rad]; 0 closes along x")
        approach_height: float = Field(default=0.10, ge=0.03, le=0.25, description="[m]")
        lift_height: float = Field(default=0.12, ge=0.03, le=0.30, description="[m]")
        grasp_aperture: float = Field(default=0.0, ge=0.0, le=0.8)
        open_aperture: float = Field(default=1.0, ge=0.3, le=1.0)
        object_top_z: float | None = Field(default=None, description="object top z [m]")
        pad_floor_z: float | None = Field(default=None, description="pads stay above [m] (a rim)")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        x, y, z = _xyz(p.xyz)
        if z < ctx.table_z - 0.01:
            raise SkillError(f"grasp point z={z:.3f} is below the table", error_type="precondition")
        z, raised = _pad_floor(ctx, z, p.pad_floor_z)
        wanted_approach = p.approach_height
        if p.object_top_z is not None:
            wanted_approach = max(wanted_approach, float(p.object_top_z) - z + 0.03)
        approach = _reachable_height(ctx, p.side, x, y, z, wanted_approach, p.yaw)
        if approach is None:
            return self.result(SkillStatus.failure, f"grasp point {np.round([x, y, z], 3).tolist()} is not reachable top-down by the {p.side} arm", verified=True, metrics={"stage": "approach", "reachable": False})
        cleared = _clear_other_arm(ctx, p.side, x, y)
        above, approach = _approach(ctx, p.side, x, y, z, approach, p.yaw, p.open_aperture)
        if above.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {above.message}", verified=True, metrics={"stage": "approach", "approach_height": approach, "other_arm_cleared": cleared, "robot_time_s": _robot_time(ctx, tick0)})
        down, retried = _descend(ctx, p.side, (x, y, z), p.yaw, approach)
        if down.status != SkillStatus.success:
            ctx.call("move_to", side=p.side, xyz=[x, y, z + approach], yaw=p.yaw)
            return self.result(SkillStatus.failure, f"descent failed: {down.message}{_edge_hint(down)}", verified=True, metrics={"stage": "descend", "descent_retried": retried, **_descent_metrics(down), "robot_time_s": _robot_time(ctx, tick0)})
        pad_gate = _pad_gate(ctx, p.side, x, y, z, p.yaw, float(p.pad_floor_z)) if p.pad_floor_z is not None else None
        grip = ctx.call("set_gripper", side=p.side, aperture=p.grasp_aperture, duration=0.5)
        grasp_aperture = float(grip.outputs.get("aperture", 0.0))
        held_now, width_now = _held(ctx, p.side)
        if not held_now:
            ctx.call("set_gripper", side=p.side, aperture=p.open_aperture, duration=0.3)
            ctx.call("move_to", side=p.side, xyz=[x, y, z + approach], yaw=p.yaw)
            return self.result(SkillStatus.failure, f"fingers closed on nothing (opening {0 if width_now is None else round(width_now * 100, 1)} cm)", verified=True, metrics={"stage": "grasp", "grasp_aperture": grasp_aperture, "grasp_width_m": width_now, **_descent_metrics(down), "robot_time_s": _robot_time(ctx, tick0)})
        lift = _reachable_height(ctx, p.side, x, y, z, p.lift_height, p.yaw) or approach
        lift_res = ctx.call("move_to", side=p.side, xyz=[x, y, z + lift], yaw=p.yaw, correction_passes=0)
        held_after, width_after = _held(ctx, p.side)
        held = ctx.gripper(p.side)
        if not held_after:
            return self.result(SkillStatus.failure, f"object lost during lift (opening {0 if width_after is None else round(width_after * 100, 1)} cm)", verified=True, metrics={"stage": "lift", "grasp_aperture": round(grasp_aperture, 3), "aperture_after_lift": round(held, 3), "width_after_lift_m": width_after, "robot_time_s": _robot_time(ctx, tick0)})
        msg = "object grasped and lifted" + (f" (lift limited to {lift:.2f} m above grasp by reach)" if lift < p.lift_height - 1e-6 else "") + (f" (grasp height raised {raised * 100:.1f} cm to the pad floor)" if raised >= 0.005 else "")
        return self.result(SkillStatus.success, msg, verified=True, metrics={"stage": "lift", "grasp_aperture": round(grasp_aperture, 3), "aperture_after_lift": round(held, 3), "grasp_width_m": width_now, "width_after_lift_m": width_after, "lift_height": lift, "lift_move": lift_res.status.value, "descent_retried": retried, "other_arm_cleared": cleared, "grasp_z_raised_m": raised, "pad_gate": pad_gate, **_descent_metrics(down), "robot_time_s": _robot_time(ctx, tick0)}, outputs={"lift_pos": np.round(ctx.tip(p.side), 4).tolist(), "held": True, "opening_width_m": width_after, "grasp_xy": [round(x, 4), round(y, 4)], "grasp_z": round(z, 4)})


class Place(Skill):
    name = "place"
    version = "1.6.0"
    description = "Carry a held object over a world point (straight Cartesian transit), descend to `release_height` above it, open, retreat >= 6 cm. Checks held before / fingers opened after the release. A descent the object stops early (landed_early tolerances) is a landing: success, metrics.landed_early. Idle other arm over the column: moved aside. Reach-ceiling release point: uses the column that exists (transit_height >= 0.01); missed transit: retried lower."
    returns = "success if held on arrival and the fingers opened (outputs.release_pos; metrics.landed_early); failure: nothing held, unreachable (metrics.reach_ceiling_z), lateral miss / too high, fingers wedged. metrics: tip_error_xy_m, touchdown_dz_m, transit_heights_tried, other_arm_cleared"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "opening before / after release, release point FK; landing not observed"
    dependencies = ["move_to", "set_gripper"]
    known_limitations = []
    timeout_s = 15.0

    class Params(Params):
        side: Side
        xyz: list[float] = Field(min_length=3, max_length=3, description="object centre landing point [m]")
        yaw: float = Field(default=0.0)
        transit_height: float = Field(default=0.15, ge=0.01, le=0.35, description="carry height above xyz [m]")
        release_height: float = Field(default=0.04, ge=0.0, le=0.2, description="grasp point above xyz at release [m]")
        open_aperture: float = Field(default=1.0, ge=0.3, le=1.0)
        landed_early_xy_tolerance_m: float = Field(default=0.015, ge=0.002, le=0.05, description="[m]")
        landed_early_max_dz_m: float = Field(default=0.05, ge=0.005, le=0.15, description="[m]")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        x, y, z = _xyz(p.xyz)
        held_before = ctx.gripper(p.side)
        is_held, width_before = _held(ctx, p.side)
        if not is_held and ctx.backend.commanded_action()[6 if p.side == "left" else 13] < 0.3:
            return self.result(SkillStatus.failure, f"nothing appears to be held (opening {0 if width_before is None else round(width_before * 100, 1)} cm)", verified=True, metrics={"stage": "precondition"})
        z_low = z + p.release_height
        wanted = max(p.transit_height - p.release_height, 0.03)
        transit = _reachable_height(ctx, p.side, x, y, z_low, wanted, p.yaw)
        if transit is None:
            # the release point itself may be reachable while no 3 cm column above it is (a stack layer at the reach
            # ceiling; hand_v5 refused it as "release point not reachable"): approach with whatever column exists
            transit = _reachable_height(ctx, p.side, x, y, z_low, wanted, p.yaw, minimum=MIN_TRANSIT_M)
        if transit is None:
            ceiling = _ceiling_above(ctx, p.side, x, y, z_low, p.yaw)
            hint = f" (reach ceiling above the point: z={ceiling:.3f})" if ceiling is not None else ""
            return self.result(SkillStatus.failure, f"release point {np.round([x, y, z_low], 3).tolist()} is not reachable top-down by the {p.side} arm{hint}: lower the release, place closer to the arm base or use the other arm", verified=True, metrics={"stage": "transit", "reachable": False, "reach_ceiling_z": ceiling})
        cleared = _clear_other_arm(ctx, p.side, x, y)
        over, transit, transit_tried = _transit(ctx, p.side, x, y, z_low, transit, p.yaw)
        if over.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"transit failed: {over.message}", verified=True, metrics={"stage": "transit", "transit_height": transit, "transit_heights_tried": transit_tried, "other_arm_cleared": cleared, "robot_time_s": _robot_time(ctx, tick0)})
        transit, over_err, align_moves = _align_over(ctx, p.side, x, y, z_low, transit, p.yaw, over)
        target = np.array([x, y, z_low])
        down = _release_descent(ctx, p.side, target, p.yaw, transit)
        release_pos = ctx.tip(p.side)
        xy_err = float(np.linalg.norm(release_pos[:2] - target[:2]))
        dz = float(release_pos[2] - target[2])  # > 0: the fingertip stopped ABOVE the release point (the held object touched down first)
        opened = ctx.call("set_gripper", side=p.side, aperture=p.open_aperture, duration=0.4)
        width_open = (opened.outputs or {}).get("opening_width_m")
        commanded_open = (opened.outputs or {}).get("commanded_width_m") or 0.096
        # post-release observation: the fingers must have opened to (about) the commanded width, or at least 2 cm beyond the
        # width they held; a few millimetres of opening means they are wedged (e.g. inside a cup mouth) and the object -
        # and whatever the fingers are wedged in - is still clamped and would be carried away by the retreat
        released = width_open is None or width_before is None or float(width_open) >= min(float(width_before) + 0.02, float(commanded_open) - 0.01)
        retreat = _retreat_height(ctx, p.side, x, y, z_low, transit, p.yaw)
        ctx.call("move_to", side=p.side, xyz=[x, y, z_low + retreat], yaw=p.yaw, correction_passes=0)
        metrics = {"aperture_before_release": round(held_before, 3), "width_before_release_m": width_before, "width_after_open_m": width_open, "released": released, **_descent_metrics(down), "tip_error_xy_m": round(xy_err, 4), "touchdown_dz_m": round(dz, 4), "landed_early": False, "transit_height": transit, "transit_heights_tried": transit_tried, "over_tip_error_m": None if over_err is None else round(float(over_err), 4), "align_moves": align_moves, "retreat_height": retreat, "other_arm_cleared": cleared, "robot_time_s": _robot_time(ctx, tick0)}
        outputs = {"release_pos": np.round(release_pos, 4).tolist()}
        if not released:
            return self.result(SkillStatus.failure, f"the fingers did not open after the release command (opening {round(float(width_open) * 100, 1)} cm vs {round(float(width_before) * 100, 1)} cm held, {round(float(commanded_open) * 100, 1)} cm commanded): the fingers are wedged and the object may still be clamped - re-observe before the next action", verified=True, metrics={**metrics, "stage": "release", "commanded_open_width_m": commanded_open}, outputs=outputs)
        if down.status != SkillStatus.success:
            if xy_err <= p.landed_early_xy_tolerance_m and 0.0 < dz <= p.landed_early_max_dz_m:
                metrics.update({"landed_early": True, "landed_early_dz_m": round(dz, 4)})
                return self.result(SkillStatus.success, f"released (landed early: the held object touched down {dz * 100:.1f} cm above the nominal release height, lateral error {xy_err * 100:.1f} cm; it may rest on a surface higher than assumed - re-observe if the landing matters)", verified=True, metrics=metrics, outputs=outputs)
            return self.result(SkillStatus.failure, f"released, but the fingertip missed the release point: {down.message} (lateral {xy_err * 100:.1f} cm, height offset {dz * 100:+.1f} cm){_edge_hint(down)}", verified=True, metrics=metrics, outputs=outputs)
        return self.result(SkillStatus.success, "released", verified=True, metrics=metrics, outputs=outputs)


class PickAndPlace(Skill):
    name = "pick_and_place"
    version = "1.1.0"
    description = "Pick an object at `pick_xyz` and place it at `place_xyz` with the same arm (pick + place composed)."
    returns = "success if pick reported a held object and place released it; metrics.robot_time_s"
    dependencies = ["pick", "place"]
    effect_verification = "inherits pick's grasp verification; place location from FK"
    timeout_s = 30.0

    class Params(Params):
        side: Side
        pick_xyz: list[float] = Field(min_length=3, max_length=3)
        place_xyz: list[float] = Field(min_length=3, max_length=3)
        pick_yaw: float = 0.0
        place_yaw: float = 0.0
        lift_height: float = Field(default=0.15, ge=0.03, le=0.3)
        release_height: float = Field(default=0.04, ge=0.0, le=0.2)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        pick = ctx.call("pick", side=p.side, xyz=p.pick_xyz, yaw=p.pick_yaw, lift_height=p.lift_height)
        if pick.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"pick failed: {pick.message}", verified=True, metrics={"stage": "pick", **pick.metrics, "robot_time_s": _robot_time(ctx, tick0)})
        place = ctx.call("place", side=p.side, xyz=p.place_xyz, yaw=p.place_yaw, transit_height=p.lift_height, release_height=p.release_height)
        ok = place.status == SkillStatus.success
        return self.result(SkillStatus.success if ok else SkillStatus.failure, "placed" if ok else f"place failed: {place.message}", verified=True, metrics={"stage": "place", **place.metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs=place.outputs)


class TransferBetweenArms(Skill):
    """hand_v1 fixed-zone transfer; superseded by skills/transfer_between_arms.py (disabled in the manifest, kept for lineage)."""

    name = "transfer_between_arms"
    version = "1.0.0"
    description = "Move a held object from one arm to the other via the table: the giver places the object in the small zone both arms can reach top-down (around x=0.50, y=-0.10), releases and retreats; the receiver picks it up from there. In-air handover is not possible with top-down wrists on this robot (the arms share no workspace above the table)."
    returns = "success if the receiver holds the object afterwards (outputs.lift_pos); failure otherwise"
    preconditions = ["`giver` holds an object", "the object fits the gripper when grasped at yaw `regrasp_yaw`"]
    effects = ["object held by `receiver`, lifted ~10 cm above the shared zone"]
    failure_conditions = ["nothing held", "object rolls or tips after release", "receiver grasp misses"]
    dependencies = ["place", "pick", "wait", "locate_objects"]
    effect_verification = "receiver gripper aperture after pick (via pick's own verification)"
    known_limitations = ["shared zone is tiny (~7 x 5 cm); rolling objects (rods) may leave it", "receiver regrasp uses the release point (plus an optional RGB re-localisation) - it does not track the object"]
    timeout_s = 40.0

    class Params(Params):
        giver: Side
        receiver: Side
        object_height: float = Field(default=0.03, ge=0.005, le=0.3, description="object height [m], used for release/regrasp heights")
        regrasp_yaw: float = Field(default=0.0, description="wrist yaw for the receiver's grasp")
        relocate_color: str | None = Field(default=None, description="if given, re-localise the object by colour (locate_objects) before regrasping")

    def run(self, ctx, p):
        if p.giver == p.receiver:
            raise SkillError("giver and receiver must differ", error_type="invalid_params")
        zone = [0.50, -0.10, ctx.table_z + p.object_height / 2]
        place = ctx.call("place", side=p.giver, xyz=zone, yaw=0.0, transit_height=0.12, release_height=0.01 + p.object_height / 2)
        if place.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"giver could not place in the shared zone: {place.message}", verified=True, metrics={"stage": "place"})
        ctx.call("wait", seconds=0.5)
        clear_y = 0.25 if p.giver == "left" else -0.3
        ctx.call("move_to", side=p.giver, xyz=[0.45, clear_y, ctx.table_z + 0.15], yaw=0.0)
        target = list(zone)
        if p.relocate_color:
            det = ctx.call("locate_objects", color=p.relocate_color, object_height_m=p.object_height)
            blobs = det.outputs.get("detections") or []
            near = [d for d in blobs if d.get("world_xyz") and abs(d["world_xyz"][0] - zone[0]) < 0.08 and abs(d["world_xyz"][1] - zone[1]) < 0.08]
            if near:
                target[0], target[1] = near[0]["world_xyz"][0], near[0]["world_xyz"][1]
        pick = ctx.call("pick", side=p.receiver, xyz=target, yaw=p.regrasp_yaw, lift_height=0.10, object_top_z=ctx.table_z + p.object_height)
        ok = pick.status == SkillStatus.success
        return self.result(SkillStatus.success if ok else SkillStatus.failure, "object transferred" if ok else f"receiver regrasp failed: {pick.message}", verified=True, metrics={"stage": "regrasp", **pick.metrics}, outputs=pick.outputs)


class MoveAbove(Skill):
    name = "move_above"
    version = "1.2.0"
    description = "Move an arm's fingertip to a point `height` above a world xy location (safe transit / inspection pose). Duration scales with the distance. A height above the reach ceiling at that xy is clamped to the highest reachable one (>= 3 cm) instead of failing (metrics.clamped, height_used)."
    dependencies = ["move_to"]
    effect_verification = "fingertip FK vs target"
    timeout_s = 8.0

    class Params(Params):
        side: Side
        xy: list[float] = Field(min_length=2, max_length=2)
        height: float = Field(default=0.15, ge=0.03, le=0.4, description="height above the table [m]; clamped to the reach ceiling")
        yaw: float = 0.0
        gripper: float | None = Field(default=None, ge=0.0, le=1.0)

    def run(self, ctx, p):
        x, y = float(p.xy[0]), float(p.xy[1])
        height = float(p.height)
        clamped = False
        if ctx.controller.solve_ik(p.side, [x, y, ctx.table_z + height], p.yaw) is None:
            h = height
            while h - 0.01 >= 0.03 - 1e-9:
                h = round(h - 0.01, 3)
                if ctx.controller.solve_ik(p.side, [x, y, ctx.table_z + h], p.yaw) is not None:
                    height, clamped = h, True
                    break
        r = ctx.call("move_to", side=p.side, xyz=[x, y, ctx.table_z + height], yaw=p.yaw, gripper=p.gripper)
        msg = r.message + (f" (height {p.height:.2f} m above the table is beyond the reach ceiling here: clamped to {height:.2f} m)" if clamped else "")
        return self.result(r.status, msg, verified=True, metrics={**r.metrics, "requested_height": float(p.height), "height_used": round(height, 3), "clamped": clamped}, outputs=r.outputs)


PUSH_STALL_WINDOW_M = 0.025  # commanded stroke distance without >= 3 mm of fingertip progress that counts as a stall
PUSH_STALL_PROGRESS_M = 0.003
PUSH_FINGER_LEAD_M = 0.025  # the closed finger's leading edge is ahead of the fingertip point in the push direction (measured 2.8 cm on the drawer front at yaw 0; the fingertip therefore stops that far short of the commanded goal when the part reaches it)


class Push(Skill):
    """Joint-space push with a fingertip-progress stall monitor (cand_20260909_214510_0ef7c7). Superseded by the Cartesian
    pad-height push of skills/contact_container.py; disabled in the manifest, kept for lineage."""

    name = "push"
    version = "1.1.0"
    description = "Push an object on the table: place closed fingertips `standoff` behind the object (opposite the push direction), then stroke toward the goal at contact height while watching the fingertip's progress. A stroke that STALLS (the fingertip stops advancing although the command continues) ends immediately instead of pressing on. Stalled within `stall_tolerance_m` of the goal after the object moved at least `min_travel_fraction` of the requested distance = the pushed part reached its goal or end-stop (drawer shut, part against its stop) -> success with metrics.stalled=true; stalled earlier -> failure (blocked). The object's own motion is not observed - re-observe when it matters."
    returns = "success if the fingertip reached the goal, or stalled within `stall_tolerance_m` of it with object_travel_est_m >= min_travel_fraction * requested (metrics: reached_goal, stalled, travel_m = fingertip travel along the push direction, object_travel_est_m = travel_m - standoff + finger lead (estimate of the object's displacement), requested_travel_m, shortfall_m = remaining fingertip distance to the goal, stall_point); failure if the start/goal is unreachable, the stroke stalled early (blocked) or missed the goal without a stall"
    preconditions = ["object rests on the table", "start and goal within reach"]
    effect_verification = "fingertip FK during the stroke (progress monitor) and at its end; object motion is NOT observed"
    dependencies = ["move_to", "set_gripper"]
    known_limitations = ["no verification of object displacement (only the fingertip's travel)", "the closed finger pads reach ~1.7 cm below the fingertip point and the finger's leading edge is ~2.5-3 cm ahead of it in the push direction, so the fingertip stops short of the goal by about that much (plus any overshoot margin in goal_xy) when the part reaches its stop", "a stall within stall_tolerance_m of the goal cannot be distinguished from an obstacle that close to the goal without exteroception"]
    timeout_s = 12.0

    class Params(Params):
        side: Side
        start_xyz: list[float] = Field(min_length=3, max_length=3, description="object centre (or the face to push) [m]")
        goal_xy: list[float] = Field(min_length=2, max_length=2, description="where that point should end [m]")
        contact_height: float = Field(default=0.015, ge=0.005, le=0.08, description="fingertip height above the table during the stroke [m]")
        standoff: float = Field(default=0.04, ge=0.02, le=0.1, description="distance behind `start_xyz` where the closed fingers touch down [m]")
        stall_tolerance_m: float = Field(default=0.06, ge=0.005, le=0.15, description="a stroke that stalls within this fingertip distance of the goal counts as having reached it (end-stop / goal contact) [m]; default covers the ~2.5-3 cm finger lead plus a 2-3 cm overshoot margin")
        min_travel_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="a stall only counts as reaching the goal when the estimated object displacement is at least this fraction of the requested distance")
        duration: float = Field(default=1.5, ge=0.3, le=6.0, description="nominal stroke duration [s]")

    def run(self, ctx, p):
        s = np.asarray(p.start_xyz[:2], dtype=float)
        g = np.asarray(p.goal_xy, dtype=float)
        d = g - s
        if np.linalg.norm(d) < 1e-3:
            raise SkillError("goal equals start", error_type="invalid_params")
        u = d / np.linalg.norm(d)
        u3 = np.array([u[0], u[1], 0.0])
        behind = s - u * p.standoff
        z = ctx.table_z + p.contact_height
        goal3 = np.array([g[0], g[1], z])
        requested = float(np.linalg.norm(d))
        ctx.call("set_gripper", side=p.side, aperture=0.0, duration=0.3)
        r = ctx.call("move_to", side=p.side, xyz=[behind[0], behind[1], z + 0.08], yaw=0.0, duration=1.2)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"cannot reach start: {r.message}", verified=True, metrics={"stage": "approach"})
        sol = ctx.controller.solve_ik(p.side, goal3, 0.0)
        if sol is None:
            return self.result(SkillStatus.failure, f"goal {np.round(goal3, 3).tolist()} is not reachable top-down by the {p.side} arm", verified=True, metrics={"stage": "approach", "reachable": False})
        ctx.call("move_to", side=p.side, xyz=[behind[0], behind[1], z], yaw=0.0, duration=0.8)
        start_tip = ctx.tip(p.side).copy()
        stroke_len = max(1e-3, float((goal3 - start_tip) @ u3))
        progress = {"best_m": 0.0, "fraction_at_best": 0.0}

        def monitor(fraction, state):
            # stall = the command advanced PUSH_STALL_WINDOW_M along the stroke while the fingertip gained < 3 mm
            prog = float((ctx.tip(p.side) - start_tip) @ u3)
            if prog >= progress["best_m"] + PUSH_STALL_PROGRESS_M:
                progress["best_m"], progress["fraction_at_best"] = prog, fraction
                return None
            if (fraction - progress["fraction_at_best"]) * stroke_len >= PUSH_STALL_WINDOW_M:
                return f"fingertip stopped advancing after {progress['best_m'] * 100:.1f} cm (command {(fraction - progress['fraction_at_best']) * stroke_len * 100:.1f} cm ahead)"
            return None

        stroke = ctx.controller.move_joints_both({p.side: sol}, duration=p.duration, monitor=monitor)
        tip_end = ctx.tip(p.side).copy()
        travel = float((tip_end - start_tip) @ u3)
        shortfall = float((goal3 - tip_end) @ u3)
        err = float(np.linalg.norm(tip_end - goal3))
        stalled = stroke["aborted_reason"] is not None
        reached = err <= 0.03
        ctx.call("move_to", side=p.side, xyz=[float(tip_end[0]), float(tip_end[1]), z + 0.08], yaw=0.0, duration=0.8)
        object_est = max(0.0, travel - p.standoff + PUSH_FINGER_LEAD_M)
        metrics = {"stage": "stroke", "reached_goal": reached, "stalled": stalled, "stall_reason": stroke["aborted_reason"], "stroke_fraction": stroke["fraction"], "travel_m": round(travel, 4), "object_travel_est_m": round(object_est, 4), "requested_travel_m": round(requested, 4), "shortfall_m": round(shortfall, 4), "finger_lead_m": PUSH_FINGER_LEAD_M, "tip_error_m": round(err, 4), "stall_point": np.round(tip_end, 4).tolist()}
        if reached:
            return self.result(SkillStatus.success, "push stroke complete (fingertip reached the goal)", verified=True, metrics=metrics)
        if stalled and shortfall <= p.stall_tolerance_m and object_est >= p.min_travel_fraction * requested:
            return self.result(SkillStatus.success, f"push stalled {shortfall * 100:.1f} cm short of the goal after {travel * 100:.1f} cm of stroke: the pushed part is at its goal / end-stop (object moved about {object_est * 100:.1f} cm of {requested * 100:.1f} cm requested; its position was not observed)", verified=True, metrics=metrics)
        if stalled:
            return self.result(SkillStatus.failure, f"push blocked: fingertip stalled {shortfall * 100:.1f} cm short of the goal after {travel * 100:.1f} cm (object moved about {object_est * 100:.1f} cm of {requested * 100:.1f} cm requested)", verified=True, metrics=metrics)
        return self.result(SkillStatus.failure, f"fingertip missed the goal (error {err * 100:.1f} cm) without a detected stall", verified=True, metrics=metrics)
