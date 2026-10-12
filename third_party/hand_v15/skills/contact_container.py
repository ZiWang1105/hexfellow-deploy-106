"""Contact-aware motion helpers and container skills (``hand_v6``: the hand_v5 file - hand_v4's from candidate cand_20260909_214530_669f26,
skill documentation within the runtime's 1900-character tool-description cap - plus insert_into_container 1.1.0 of candidate
cand_20260910_121426_b48287: reach-edge carry-pose guard, hover, guided two-step release; extract / sweep / push_until_stop / push unchanged).

Finger geometry (measured by FK on the robot model's collision geometry, both arms, both yaws, fingers open or closed):
  * the lowest point of the closed fingers (the blade tips / pad bottoms) sits ``ctx.cell_calibration.finger_pad_below_tip_m`` (0.017 m in the sim model, ~0.024 m real-cell estimate)
    BELOW the fingertip grasp point that every Cartesian target refers to;
  * the lowest ~2 cm of the closed fingers is a narrow column centred on the tip: ~1.2 x 1.2 cm
    (:data:`FINGER_BLADE_HALF_M`). There is NO wide low plate: the finger links only widen 2-5 cm higher
    (:data:`FINGER_LEAD_PROFILE_M`), which is what touches a tall face first when pushing. (Earlier "6.5 cm plate"
    estimates were bounding-box artefacts of a tilted strut; a sweep band was measured directly: items move only when the
    tip passes within ~1 cm of them.)
Every contact-height parameter in this module means "pad bottom above the table"; the tip offset is applied inside.

Skills
  * ``extract_from_container`` - grasp a protruding object 1-2 cm below its top (never at the rim: the pads must stay
    above the container edge), lift, verify, and move laterally clear of the container only once the object's bottom
    is above the rim (a drag would carry the container).
  * ``insert_into_container``  - carry a held object over a container opening at a reachable height, descend inside the
    footprint in 1 cm Cartesian steps until contact or until the object bottom is below the rim, release, retreat out
    of the top camera's line of sight and confirm the placement from the top depth image (surface height at the
    opening rises).
  * ``sweep_area``             - closed-finger wiping passes across an axis-aligned patch; the blade tips ride at a
    constant height above the table along several Cartesian waypoints (no joint-space bowing), pass spacing from the
    measured blade width plus the debris size.
  * ``push_until_stop``        - closed-finger push stroke in short Cartesian steps; a stall (end-stop, obstacle) with
    the travel criterion met is a SUCCESS, not a "fingertip missed target" failure.
  * ``push`` (override)        - same parameters as before, ``contact_height`` now means pad bottom above the table and
    an end-stop stall after >= half the requested stroke counts as success (``metrics.stalled``); it also reports the
    status-semantics candidate's metrics (``reached_goal``, ``requested_travel_m``, ``shortfall_m``, ``object_travel_est_m``).
Only proprioception / FK, the calibrated gripper reading and the top RGB-D camera are used. Nothing is task specific.

Integration follow-up (hand_v4): the insertion's peg-in-hole search defaults to :data:`SEARCH_OFFSETS_DEFAULT` = 3 offsets
(was 8) and stops when the remaining robot-time budget drops below :data:`SEARCH_MIN_REMAINING_S` - an 8-offset search on a
mis-estimated opening consumed the whole 60 s budget in the candidate's abc_insert validation episodes.

hand_v9@3.5.0 (parent hand_v8): correction of the hand_v7 integration defect (docs/evaluation_protocol.md amendment 9 addendum). This file is candidate cand_20260910_164848_9a670e's (extract_from_container 1.1.0: the container is re-detected after the vertical lift, a mixed peg + container grasp is released back; first loaded by a released library here - hand_v7 / hand_v8 kept hand_v6's 1.0.0) with hand_v8's base-module import keyed by the resolved skills-directory hash; ``_held`` comes from that base module, so the held checks follow the calibration. The calibrated held / empty width rule of hand_v8 (``_held`` / ``_held_width_min`` from configs/gripper/<backend>_<side>.yaml) is kept everywhere; no width constant.

Candidate cand_20260911_003629_419496 (parent hand_v9@3.5.0): extract_from_container 1.2.0, insert_into_container 1.1.1.
formal_v9 abc_extract s104 (runs/analysis_v3): "target-held verification was not equivalent to isolated extraction; first verdict a genuine
missed-held" - a 5.2 cm cup-sized closure passed the wrapper's SAM3 check only when a prompt was given, and the agent's primitive retry then
dragged the cup 14.8 cm. 1.2.0 makes the check DECISIVE and prompt-independent: the container's rim ring in the top depth image (displaced
> 2 cm, or gone with the table showing where it stood) and the SAM3 re-detection (moved > 2 cm / risen > 1 cm) are the witnesses; a positive on
either = lowered back and released with ``metrics.reason`` mixed_grasp and a message that forbids primitive retries. The surface height at the
object's place is recorded but never decides (the lifted object stands in that column and the arm occludes it in the oblique view: a first
version misread it as "target not lifted" on abc_extract s0); ``pick`` receives ``exclusive_of`` = the container prompt so the verifier is asked
whether ONLY the target is between the pads. formal_v9 abc_insert s101: ``extract_from_container`` was called as a RECOVERY for an already-held peg and its open-gripper approach
discarded the grasp - both extract_from_container and pick now refuse without motion while the arm holds something (commanded closed AND
calibrated held_by_width; failure already_holding), and insert_into_container's docs / refusals say the object is STILL HELD and steer to
transfer_between_arms or a lower re-grasp, never to pick / extract.

hand_v11@3.7.0 (parent hand_v10): cell-calibration plumbing (docs/evaluation_protocol.md, hand_v11 addendum). The module's own ``FINGER_PAD_BELOW_TIP_M = 0.017`` and ``PAD_MIN_TIP_ABOVE_TABLE_M`` are gone: ``_pad_below_tip_m(ctx)`` / ``_pad_min_tip_above_table_m(ctx)`` read ``ctx.cell_calibration.finger_pad_below_tip_m`` (extract_from_container 1.2.1 grasp floor, insert_into_container 1.1.2 descent floor, sweep_area 1.0.1 / push 1.1.1 / push_until_stop 1.0.1 contact heights); the push descriptions no longer quote 0.017 m. The values come from configs/cell_calibration/<backend>.yaml through ``ctx.cell_calibration`` (physapi.calibration.cell), loaded per backend kind (sim -> sim.yaml = the constants this file used to hard-code, so simulation decisions are identical; real / dry -> real.yaml). No skill branches on the backend kind. No other behaviour change.

Candidate cand_20260911_135918_55d735 (parent hand_v13@3.9.0): extract_from_container 1.3.0 - the nominal grasp adds RIM_DESCENT_ALLOWANCE_M above the rim floor when the protrusion allows, pick is asked to keep the pads above rim + RIM_PAD_CLEARANCE_M (``pad_floor_z``, measured), a container-sized or mixed grasp is released back and re-grasped ONCE REGRASP_HIGHER_M higher before the skill gives up (formal_v13 abc_extract s103: the agent's open-loop primitive retries after the wrapper's honest rejection never freed the peg). ``_carry`` gains ``speed_mps`` (default unchanged). insert_into_container / sweep_area / push_until_stop / push unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from physapi.schemas.core import SkillStatus
from physapi.skills.base import Params, Skill, SkillError

_name = f"physapi_libmod_{hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]}_manipulation_base"
if _name in sys.modules:
    _base = sys.modules[_name]
else:
    _spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name("manipulation.py"))
    _base = importlib.util.module_from_spec(_spec)
    assert _spec.loader
    sys.modules[_name] = _base
    _spec.loader.exec_module(_base)
_held = _base._held
_reachable_height = _base._reachable_height
Side = Literal["left", "right"]

# ----------------------------------------------------------------------------------------------- finger geometry
#: Closed-finger pad bottom below the fingertip grasp point [m]: hand_v11 reads it from the cell calibration
#: (``ctx.cell_calibration.finger_pad_below_tip_m``, configs/cell_calibration/<backend>.yaml: 0.017 in the sim model by FK probe, ~0.024
#: estimated on the real cell) through :func:`_pad_below_tip_m`; no constant here.
#: Half extent of the lowest ~2 cm of the closed fingers (the blade tips), centred on the tip, both horizontal axes [m].
FINGER_BLADE_HALF_M = 0.006
#: Lowest admissible pad height above the table (2 mm above the surface); the tip height follows from the calibrated pad offset.
PAD_MIN_CLEARANCE_M = 0.002


def _pad_below_tip_m(ctx) -> float:
    """The cell calibration's ``finger_pad_below_tip_m`` [m] of this backend (sim 0.017; real-cell estimate 0.024)."""

    return float(ctx.cell_calibration.finger_pad_below_tip_m)


def _pad_min_tip_above_table_m(ctx) -> float:
    """Lowest admissible tip height above the table: pads PAD_MIN_CLEARANCE_M above the surface."""

    return _pad_below_tip_m(ctx) + PAD_MIN_CLEARANCE_M
#: Default number of peg-in-hole search offsets in ``insert_into_container`` (each costs ~4-6 s of robot time).
SEARCH_OFFSETS_DEFAULT = 3
#: The search is not started / continued when less robot time than this remains (release, retreat and the top-camera check
#: still have to fit into the episode's robot-time budget).
SEARCH_MIN_REMAINING_S = 12.0


def _remaining_robot_time_s(ctx) -> float | None:
    """Remaining robot-time (simulated/control) budget of the episode, or None when the context has no budget."""

    budget = getattr(getattr(ctx, "executor", None), "budget", None)
    if budget is None or not hasattr(budget, "remaining_execution_s"):
        return None
    try:
        return float(budget.remaining_execution_s(ctx.backend))
    except Exception:  # noqa: BLE001 - a missing backend attribute must not break the skill
        return None
#: Closed-finger half extent ALONG the closing axis (how far the fingers lead the tip when pushing) as a function of the height
#: above the pad bottom: (up to this height [m], half extent [m]). Blade tips 0.5 cm, finger links 1.0-3.2 cm, wrist 4.6 cm
#: (exact collision-surface sampling, both yaws).
FINGER_LEAD_PROFILE_M = ((0.020, 0.005), (0.030, 0.010), (0.045, 0.020), (0.060, 0.023), (0.080, 0.032), (0.130, 0.046))


def finger_lead_m(face_height_above_pad: float | None) -> float:
    """Distance the closed fingers lead the fingertip point along the closing axis when pushing a face that reaches
    ``face_height_above_pad`` above the pad bottom (None = tall face of ~8 cm, wrist not involved: 0.032)."""

    if face_height_above_pad is None:
        return 0.032
    lead = FINGER_LEAD_PROFILE_M[0][1]
    lower = 0.0
    for height, half in FINGER_LEAD_PROFILE_M:  # the face reaches slab (lower, height] -> that slab's half extent leads
        if face_height_above_pad > lower:
            lead = half
        lower = height
    return lead


def _xy(v) -> np.ndarray:
    return np.asarray([float(v[0]), float(v[1])], dtype=float)


def _unit(v) -> np.ndarray:
    v = _xy(v)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        raise SkillError("direction must be non-zero", error_type="invalid_params")
    return v / n


def _wrap_half(yaw: float) -> float:
    """Fold a yaw into (-pi/2, pi/2] (parallel gripper: yaw and yaw + pi are the same grasp)."""

    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw <= -math.pi / 2:
        yaw += math.pi
    return yaw


def _holding(ctx, side: str) -> tuple[bool, float | None]:
    """(an object is between the fingers, opening width): the gripper is COMMANDED closed (< half open) AND the measured
    opening is still above the empty-closed value. An open gripper is not "holding" whatever its width reads."""

    commanded = float(ctx.backend.commanded_action()[6 if side == "left" else 13])
    held, width = _held(ctx, side)
    return (held and commanded < 0.5), width


def _tip_z_for_pad_height(ctx, pad_height: float) -> float:
    """Fingertip z that puts the closed-finger pad bottoms ``pad_height`` above the table (>= -6 mm: pressing onto it)."""

    return ctx.table_z + _pad_below_tip_m(ctx) + max(-0.006, float(pad_height))


# --------------------------------------------------------------------------------------------- Cartesian motion
def _joints(ctx, side: str) -> np.ndarray:
    q = np.asarray(ctx.backend.state().joint_pos, dtype=float)
    return q[0:6] if side == "left" else q[7:13]


def _ik_near(ctx, side: str, target, yaw: float, *, tolerance_m: float = 0.006):
    """IK solution on the SAME arm configuration branch: among the yaw-equivalent wrist solutions (yaw, yaw +- pi) the one
    closest to the current joints. ``controller.solve_ik`` alone returns the first converging branch, which at the reach
    edge can be a wrist flip 2-3 rad away; interpolating to it swings the fingertip through an 8 cm arc.
    ``tolerance_m``: IK position tolerance (the controller default 6 mm is too coarse for precise centring).
    Returns (solution, max |dq|) or None."""

    q = _joints(ctx, side)
    best = None
    for cand in (float(yaw), float(yaw) + math.pi, float(yaw) - math.pi):
        sol = ctx.controller.solve_ik(side, target, cand, symmetric=False, tolerance_m=tolerance_m, max_iterations=240 if tolerance_m >= 0.006 else 480)
        if sol is None:
            continue
        dq = float(np.abs(np.asarray(sol) - q).max())
        if best is None or dq < best[1]:
            best = (np.asarray(sol), dq)
    return best


def _cartesian_path(ctx, side: str, waypoints, yaw: float, *, speed_mps: float = 0.15, hold_s: float = 0.1, correction_passes: int = 1, stall_fraction: float | None = None, z_log: list | None = None, allow_jump: bool = False, max_jump_rad: float = 0.6, stall_patience: int = 1) -> dict:
    """Drive the fingertip through ``waypoints`` (each IK-solved on the current configuration branch, joint interpolation
    only between neighbours a few cm apart).

    Returns {"n_reached", "n_waypoints", "stop_reason": None|"unreachable"|"stall"|"ik_jump", "tip_errors_m", "stalled",
    "jumps", "tip"}. ``stall_fraction``: progress along the segment below this fraction of the segment length = stall.
    A configuration jump (max |dq| > ``max_jump_rad``) stops the path unless ``allow_jump`` (free-space moves), in which
    case it is executed slowly with extra correction. ``z_log`` collects the tip z at every control step. A suspected stall is
    re-commanded ``stall_patience`` times (0.2 s each) before it counts: a servo still settling, or an object caught on an edge
    that slides on when pushed a little longer, must not look like an end-stop.
    """

    ctrl = ctx.controller
    errors: list[float] = []
    stop = None
    stalled = False
    jumps = 0
    listener = None
    if z_log is not None:
        def listener(cmd, state, i):  # noqa: E306
            z_log.append(float(ctx.tip(side)[2]))

        ctx.backend.add_step_listener(listener)
    n = 0
    try:
        for wp in waypoints:
            wp = np.asarray(wp, dtype=float)
            best = _ik_near(ctx, side, wp, yaw)
            if best is None:
                stop = "unreachable"
                break
            sol, dq = best
            before = ctx.tip(side)
            seg = wp - before
            dist = float(np.linalg.norm(seg))
            duration = float(np.clip(dist / max(speed_mps, 1e-3), 0.15, 1.5))
            passes = correction_passes
            if dq > max_jump_rad:
                if not allow_jump:
                    stop = "ik_jump"
                    break
                jumps += 1
                duration = float(min(4.0, max(duration, dq / 0.6)))
                passes = max(2, correction_passes)
            ctrl.move_joints(side, sol, duration=duration, hold_s=hold_s, correction_passes=passes)
            tip = ctx.tip(side)
            n += 1
            if stall_fraction is not None and dist > 1e-4:
                progress = float((tip - before) @ (seg / dist))
                if progress < stall_fraction * dist:
                    # confirm before calling it a stall: a servo still settling from the previous move can look like one
                    for _ in range(max(1, stall_patience)):
                        ctrl.move_joints(side, sol, duration=0.2, hold_s=0.2, correction_passes=1)
                        tip = ctx.tip(side)
                        progress = float((tip - before) @ (seg / dist))
                        if progress >= stall_fraction * dist:
                            break
                    if progress < stall_fraction * dist:
                        errors.append(round(float(np.linalg.norm(tip - wp)), 4))
                        stalled = True
                        stop = "stall"
                        break
            errors.append(round(float(np.linalg.norm(tip - wp)), 4))
    finally:
        if listener is not None:
            ctx.backend.remove_step_listener(listener)
    return {"n_reached": n, "n_waypoints": len(list(waypoints)), "stop_reason": stop, "tip_errors_m": errors, "stalled": stalled, "jumps": jumps, "tip": ctx.tip(side)}


def _settle_precise(ctx, side: str, target, yaw: float, *, tol_m: float = 0.002, passes: int = 3) -> float:
    """Bring the fingertip within ``tol_m`` of ``target`` by biasing the commanded point with the measured residual.

    ``move_joints`` stops correcting at 0.01 rad (~5 mm at the tip) and the IK itself tolerates 6 mm, so a single move
    leaves 3-6 mm of error; re-targeting at target - residual (measured by FK) removes it. Returns the final error [m]."""

    target = np.asarray(target, dtype=float)
    goal = target.copy()
    for _ in range(max(0, passes)):
        e = ctx.tip(side) - target
        if float(np.linalg.norm(e)) <= tol_m:
            break
        goal = goal - e
        best = _ik_near(ctx, side, goal, yaw, tolerance_m=0.0015)  # tight IK: the residual must be the servo's, not the solver's
        if best is None or best[1] > 0.6:
            break
        ctx.controller.move_joints(side, best[0], duration=0.3, hold_s=0.2, correction_passes=2)
    return float(np.linalg.norm(ctx.tip(side) - target))


def _line(start, end, step: float):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    d = float(np.linalg.norm(end - start))
    k = max(1, int(math.ceil(d / max(step, 1e-3))))
    return [start + (end - start) * (i / k) for i in range(1, k + 1)]


def _highest_reachable(ctx, side: str, x: float, y: float, z_low: float, z_high: float, yaw: float, step: float = 0.01) -> float | None:
    """Highest z in [z_low, z_high] (1 cm grid, z_high included) that is top-down reachable at (x, y); None if z_low is not."""

    z = float(z_high)
    while z > z_low + 1e-9:
        if ctx.controller.solve_ik(side, [x, y, z], yaw) is not None:
            return round(z, 4)
        z -= step
    return round(float(z_low), 4) if ctx.controller.solve_ik(side, [x, y, z_low], yaw) is not None else None


def _carry(ctx, side: str, target_xy, z_min: float, z_pref: float, yaw: float, *, step: float = 0.05, speed_mps: float = 0.25) -> dict:
    """Horizontal Cartesian carry to ``target_xy``: each waypoint at the highest reachable height in [z_min, z_pref]. ``speed_mps``
    (default the previous 0.25) lets a caller carry an object that hangs off-centre in the pinch (a rim-held bowl) more gently."""

    tip = ctx.tip(side)
    pts = []
    for p in _line(tip[:2], _xy(target_xy), step):
        z = _highest_reachable(ctx, side, float(p[0]), float(p[1]), z_min, z_pref, yaw)
        if z is None:
            continue
        pts.append([float(p[0]), float(p[1]), z])
    if not pts or np.linalg.norm(np.asarray(pts[-1][:2]) - _xy(target_xy)) > 0.005:
        z = _highest_reachable(ctx, side, float(target_xy[0]), float(target_xy[1]), z_min, z_pref, yaw)
        if z is None:
            return {"ok": False, "reason": "target unreachable"}
        pts.append([float(target_xy[0]), float(target_xy[1]), z])
    r = _cartesian_path(ctx, side, pts, yaw, speed_mps=float(speed_mps), allow_jump=True)
    return {"ok": r["stop_reason"] is None, "reason": r["stop_reason"], **r}


def _depth_points(ctx, xy, at_z: float, radius_m: float) -> np.ndarray | None:
    """World points (N x 3) of the top depth image within ``radius_m`` (pixel radius at that range) of world point (xy, at_z)."""

    from physapi.perception.locator import world_to_pixel

    K = ctx.backend.camera_intrinsics("top")
    c2w = ctx.backend.camera_to_world("top")
    centre = np.array([float(xy[0]), float(xy[1]), float(at_z)])
    uv = world_to_pixel(centre, K, c2w)
    if uv is None:
        return None
    depth = ctx.capture_depth("top")
    dist = float(np.linalg.norm(centre - c2w[:3, 3]))
    r_px = int(np.clip(round(float(K[0, 0]) * radius_m / max(dist, 0.3)), 2, 60))
    u = int(np.clip(round(uv[0]), 0, depth.shape[1] - 1))
    v = int(np.clip(round(uv[1]), 0, depth.shape[0] - 1))
    v0, v1 = max(0, v - r_px), min(depth.shape[0], v + r_px + 1)
    u0, u1 = max(0, u - r_px), min(depth.shape[1], u + r_px + 1)
    patch = depth[v0:v1, u0:u1]
    vv, uu = np.mgrid[v0:v1, u0:u1]
    valid = patch > 0
    if not valid.any():
        return None
    rays = np.linalg.inv(K) @ np.stack([uu[valid].ravel(), vv[valid].ravel(), np.ones(int(valid.sum()))])
    p_cam = rays / rays[2] * patch[valid].ravel()
    return (c2w[:3, :3] @ p_cam).T + c2w[:3, 3]


def _top_surface_z(ctx, xy, at_z: float, *, radius_m: float = 0.02) -> float | None:
    """Highest surface the top depth camera sees within ``radius_m`` of world point (xy, at_z); None without depth.

    The top camera looks obliquely (from the robots' side along +x), so the single pixel of a container's centre sees its
    far inner wall; the HIGHEST point in the neighbourhood is robust to that: an empty container reads its rim, a seated
    object that protrudes reads its top.
    """

    pts = _depth_points(ctx, xy, at_z, radius_m)
    if pts is None:
        return None
    return float(np.percentile(pts[:, 2], 98))  # robust maximum


def _rim_ring(ctx, xy, rim_z: float, *, radius_m: float = 0.04, band_m: float = 0.006, min_points: int = 15, exclude_xy=None, exclude_r: float = 0.012, pts: np.ndarray | None = None) -> dict | None:
    """Locate a container's rim in the top depth image: pixels within ``radius_m`` of (xy, rim_z) whose surface height is
    within ``band_m`` of the rim height (optionally ignoring a disc around ``exclude_xy``, e.g. an object standing there).
    Returns {"centre": xy of those points, "n": count, "spread_m": rms radius} or None.
    The centroid of a complete rim ring is the opening centre (the caller's estimate may be 1-2 cm off; depth is 2-3 mm)."""

    if pts is None:
        pts = _depth_points(ctx, xy, rim_z, radius_m)
    if pts is None:
        return None
    mask = (np.abs(pts[:, 2] - rim_z) <= band_m) & (np.linalg.norm(pts[:, :2] - _xy(xy), axis=1) <= radius_m)
    if exclude_xy is not None:
        mask &= np.linalg.norm(pts[:, :2] - _xy(exclude_xy), axis=1) >= exclude_r
    sel = pts[mask]
    if len(sel) < min_points:
        return None
    centre = sel[:, :2].mean(axis=0)
    return {"centre": centre, "n": int(len(sel)), "spread_m": float(np.sqrt(np.mean(np.sum((sel[:, :2] - centre) ** 2, axis=1))))}


def _placement_view(ctx, xy, rim_z: float, *, radius_m: float = 0.045) -> dict:
    """What the top depth camera sees around a release point after letting go: the highest surface, the xy of the object if
    something protrudes >= 1 cm above the rim, the rim ring centre (object pixels excluded) and the object-to-ring distance."""

    pts = _depth_points(ctx, xy, rim_z, radius_m)
    out = {"surface_max": None, "object_xy": None, "ring_xy": None, "object_to_ring_m": None}
    if pts is None:
        return out
    out["surface_max"] = float(np.percentile(pts[:, 2], 98))
    top = pts[pts[:, 2] >= rim_z + 0.01]
    if len(top) >= 6:
        out["object_xy"] = top[:, :2].mean(axis=0)
    ring = _rim_ring(ctx, xy, rim_z, radius_m=radius_m, exclude_xy=out["object_xy"], pts=pts)
    if ring is not None:
        out["ring_xy"] = ring["centre"]
        if out["object_xy"] is not None:
            out["object_to_ring_m"] = float(np.linalg.norm(out["object_xy"] - ring["centre"]))
    return out


def _pick_params_supported(ctx) -> set[str]:
    try:
        return set(ctx.library.get("pick").Params.model_fields)
    except Exception:  # noqa: BLE001
        return set()


# =========================================================================================== extract_from_container
#: extract_from_container 1.1.0 target-only check: the container is re-detected from the top camera after the vertical lift
#: (before any lateral move); a displacement above CONTAINER_MOVED_M or a rise of its top above CONTAINER_RISEN_M means the fingers
#: hold the container too (mixed peg + cup grasp after a tolerated descent error, formal_v7 abc_extract s104: the cup travelled 6.6 cm).
#: extract_from_container 1.2.0 (cand_20260911_003629_419496, formal_v9 abc_extract s104: "target-held verification was not equivalent to
#: isolated extraction"): the check is DECISIVE and independent of the SAM3 prompt - the container's rim ring in the top depth image
#: (centre before the pick vs after the vertical lift; gone = the table shows where the rim was) decides, the SAM3 re-detection
#: (container_prompt) is a second witness. Thresholds tightened to the measurement (ring centroid within a few mm; abc_extract s0: the cup
#: truly slides 7 mm as the peg leaves, ring 9.9 mm; SAM3 centroid jitter ~1 cm): moved > 2 cm or risen > 1 cm = mixed grasp, released back.
#: The surface height at the object's place is recorded (target_rise_m) but never decides: the lifted object stands in that column.
CONTAINER_MOVED_M = 0.02
CONTAINER_RISEN_M = 0.01
ALREADY_HOLDING_HINT = ("pick / extract_from_container start a NEW grasp: they open the fingers and would drop what is held. If the held object is the "
                        "one to insert or place, call insert_into_container / place; if it is already extracted you are done with it; to discard the grasp "
                        "deliberately call set_gripper(aperture=1) first")


def _container_view(ctx, container_xy, rim_z: float, object_xy, object_top_z: float | None) -> dict:
    """Top-depth witness of an extraction: the container's rim ring centre (``_rim_ring``), the highest surface within 1.5 cm of the
    object's place BELOW a ceiling 2 cm above the object's top (the arm that holds the lifted object stands over the container and the
    oblique camera sees it in the same pixels: anything above the ceiling is the arm / the lifted object, never the scene), and the
    number of table-level points where the container stands (the container is GONE only when the table shows there). Before the pick
    the surface is the protruding object's top; after a real extraction it drops to the rim / interior. None fields = not visible."""

    top = rim_z if object_top_z is None else float(object_top_z)
    ring = _rim_ring(ctx, container_xy, rim_z, radius_m=0.045, exclude_xy=object_xy, exclude_r=0.012)
    pts = _depth_points(ctx, object_xy, top, 0.015)
    surface = None
    occluded = None
    if pts is not None and len(pts):
        above = pts[:, 2] > top + 0.02
        occluded = round(float(above.mean()), 3)
        # after the lift the arm (and the lifted object) usually stand over the container in the oblique view: the patch then shows the
        # arm, not the scene, and says nothing about the object's place - the witness is withheld rather than misread
        if occluded <= 0.2 and int((~above).sum()) >= 4:
            surface = float(np.percentile(pts[~above, 2], 98))
    cpts = _depth_points(ctx, container_xy, rim_z, 0.03)
    table_n = 0 if cpts is None else int(((np.abs(cpts[:, 2] - ctx.table_z) <= 0.01) & (np.linalg.norm(cpts[:, :2] - _xy(container_xy), axis=1) <= 0.02)).sum())
    return {"ring_xy": None if ring is None else [round(float(v), 4) for v in ring["centre"]], "ring_n": None if ring is None else ring["n"], "surface_at_object_z": None if surface is None else round(surface, 4), "occluded_fraction": occluded, "table_points_at_container": table_n}


def _detect_near(ctx, prompt: str, xy, radius_m: float = 0.12) -> dict | None:
    """segment_objects detection of ``prompt`` nearest ``xy`` (within radius), or None (no server / no detection)."""

    if "segment_objects" not in ctx.library:
        return None
    try:
        r = ctx.call("segment_objects", text_prompt=prompt, max_results=4, near_xy=[float(xy[0]), float(xy[1])], near_radius_m=radius_m)
    except Exception:  # noqa: BLE001
        return None
    dets = [d for d in (r.outputs or {}).get("detections") or [] if d.get("world_xyz")]
    if not dets:
        return None
    d = min(dets, key=lambda d: float(np.linalg.norm(np.asarray(d["world_xyz"][:2]) - np.asarray(xy, dtype=float))))
    top = d.get("top_z")
    if top is None and d.get("estimated_height_m") is not None:
        top = float(ctx.table_z) + float(d["estimated_height_m"])
    return {"xy": [round(float(d["world_xyz"][0]), 4), round(float(d["world_xyz"][1]), 4)], "top_z": None if top is None else round(float(top), 4)}


#: extract_from_container 1.3.0 (formal_v13 abc_extract s103): the pads stay RIM_PAD_CLEARANCE_M above the rim by the measured pad gate of pick
#: 1.8.0 (``pad_floor_z``), and the NOMINAL grasp height adds RIM_DESCENT_ALLOWANCE_M for the descent's tolerated residual (0.6 cm accepted
#: under the 1.5 cm tolerance put the pads 3 mm below the rim: a peg + cup grasp of 5.4 cm) whenever the protrusion leaves room for it. A
#: container-sized grasp or a mixed grasp is released back and RETRIED ONCE this many metres higher on the object before the skill gives up.
RIM_PAD_CLEARANCE_M = 0.005
RIM_DESCENT_ALLOWANCE_M = 0.008
REGRASP_HIGHER_M = 0.012
#: The fingertip must stay at least this far below the object top (the 2 cm pad column then covers >= 1.3 cm of the object).
TIP_BELOW_TOP_MIN_M = 0.008


class ExtractFromContainer(Skill):
    name = "extract_from_container"
    version = "1.3.0"
    kind = "code"
    description = (
        "Take an object protruding >= 3 cm from a container out (refused while the arm holds something): verified `pick` `grasp_below_top_m` below the object top "
        "(pads kept above the rim by a measured gate), lift to the reach ceiling, DECISIVE target-only check (top depth rim ring, SAM3 `container_prompt`; a "
        "container-sized or mixed grasp is released back and re-grasped ONCE higher), then move `clearance_m` sideways once the object bottom clears the rim. "
        "Heights = table z + estimated_height_m."
    )
    returns = (
        "success if held, lifted, clear, container stayed; failure: already_holding, short protrusion, unreachable, closed on nothing, mixed_grasp / container_grasp "
        "(released back), not clear (held). metrics: reason, pad_gate"
    )
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "openings after close / lift / clearance; fingertip FK; verify_grasp"
    observations_used = ["proprio", "rgb:left_wrist", "rgb:right_wrist", "rgb:top", "depth:top"]
    dependencies = ["pick", "move_to", "set_gripper"]
    known_limitations = []
    timeout_s = 45.0

    class Params(Params):
        side: Side
        object_xyz: list[float] = Field(min_length=3, max_length=3, description="[m]")
        object_top_z: float = Field(description="[m]")
        container_xyz: list[float] = Field(min_length=3, max_length=3, description="opening xy [m]")
        container_top_z: float = Field(description="rim z [m]")
        object_bottom_z: float | None = Field(default=None, description="table")
        yaw: float | None = Field(default=None, description="null = auto")
        grasp_below_top_m: float = Field(default=0.02, ge=0.01, le=0.05, description="[m]")
        clearance_m: float = Field(default=0.10, ge=0.03, le=0.25, description="lateral move [m]")
        object_width_m: float | None = Field(default=None, ge=0.005, le=0.09, description="grasp width [m]")
        target_object: str = Field(default="the target object", max_length=120)
        container_prompt: str = Field(default="", max_length=80, description="container SAM3 prompt")

    def run(self, ctx, p):
        x, y = float(p.object_xyz[0]), float(p.object_xyz[1])
        rim = float(p.container_top_z)
        # precondition (abc_insert s101: extract_from_container was called as a RECOVERY for an already-held peg and its open-gripper
        # approach discarded the grasp): a gripper commanded closed with the calibrated width rule saying held is refused, nothing moves
        holding, width_held = _holding(ctx, p.side)
        if holding:
            return self.result(SkillStatus.failure, f"the {p.side} gripper already holds an object (opening {0 if width_held is None else round(width_held * 100, 1)} cm, fingers commanded closed): nothing moved. {ALREADY_HOLDING_HINT}", verified=True, metrics={"stage": "precondition", "reason": "already_holding", "opening_width_m": width_held}, outputs={"held": True, "cleared": False})
        container_before = _detect_near(ctx, p.container_prompt, p.container_xyz[:2]) if p.container_prompt else None
        top = float(p.object_top_z)
        view_before = _container_view(ctx, p.container_xyz[:2], rim, (x, y), top)
        bottom = ctx.table_z if p.object_bottom_z is None else float(p.object_bottom_z)
        pad_floor_z = rim + RIM_PAD_CLEARANCE_M  # the pad bottoms never below this (measured by pick's gate)
        min_grasp_z = pad_floor_z + _pad_below_tip_m(ctx)
        grasp_z = max(top - p.grasp_below_top_m, min_grasp_z)
        protrusion = top - rim
        tip_max = top - TIP_BELOW_TOP_MIN_M
        if grasp_z > tip_max:
            return self.result(SkillStatus.failure, f"object protrudes only {protrusion * 100:.1f} cm above the rim: a grasp {p.grasp_below_top_m * 100:.0f} cm below its top would put the finger pads ({_pad_below_tip_m(ctx) * 100:.1f} cm below the tip) on the container; need >= {(_pad_below_tip_m(ctx) + 0.013) * 100:.1f} cm", verified=True, metrics={"stage": "precondition", "protrusion_m": round(protrusion, 4), "min_grasp_z": round(min_grasp_z, 4)})
        # 1.3.0: the nominal grasp also allows for the descent's tolerated residual when the protrusion leaves room (the measured gate
        # in pick is the guarantee; the allowance saves the gate's correction move)
        grasp_z = min(max(grasp_z, min_grasp_z + RIM_DESCENT_ALLOWANCE_M), tip_max)
        # yaw: given, or the most reachable one (highest lift ceiling) for yaw-symmetric objects
        yaws = [float(p.yaw)] if p.yaw is not None else [0.0, math.pi / 2, math.pi / 4, -math.pi / 4]
        best = None
        for yaw in yaws:
            # the TRUE reach ceiling (no joint-limit-margin back-off): the lift is a straight Cartesian path and every
            # centimetre counts for bringing the object's bottom above the rim
            h = _reachable_height(ctx, p.side, x, y, grasp_z, 0.30, yaw, prefer_margin=False)
            if h is not None and (best is None or h > best[1]):
                best = (yaw, h)
        if best is None:
            return self.result(SkillStatus.failure, f"grasp point {np.round([x, y, grasp_z], 3).tolist()} is not reachable top-down by the {p.side} arm", verified=True, metrics={"stage": "approach", "reachable": False, "grasp_z": round(grasp_z, 4)})
        yaw, ceiling_h = best
        supported = _pick_params_supported(ctx)
        attempts = 0
        grasp_log: list[dict] = []
        pick = None
        lift = None
        width = None
        container_check: dict = {}
        while True:
            attempts += 1
            # pick with the SMALLEST lift (3 cm): its joint-space lift arcs several cm sideways, which drags the container while the
            # object is still inside; the lift to the reach ceiling is done below as a straight Cartesian path
            params = {"side": p.side, "xyz": [x, y, grasp_z], "yaw": yaw, "approach_height": float(np.clip(top - grasp_z + 0.04, 0.06, 0.25)), "lift_height": 0.03, "object_top_z": top}
            if "target_object" in supported:
                params["target_object"] = p.target_object
            if "exclusive_of" in supported:
                # the verifier is asked the right question: is ONLY the target between the pads (not the container's rim)?
                params["exclusive_of"] = p.container_prompt or "the container it stands in"
            if "pad_floor_z" in supported:
                params["pad_floor_z"] = pad_floor_z  # 1.3.0: pick measures the pad height after the descent and raises a low tip before closing
            pick = ctx.call("pick", **params)
            if pick.status in (SkillStatus.failure, SkillStatus.error):
                return self.result(SkillStatus.failure, f"pick failed: {pick.message}", verified=True, metrics={"stage": "pick", "grasp_z": round(grasp_z, 4), "yaw": round(yaw, 3), "attempts": attempts, "grasp_log": grasp_log, "pick_status": pick.status.value, **{k: v for k, v in pick.metrics.items() if k in ("verification_status", "width_after_lift_m", "tip_error_m", "pad_gate")}})
            held, width = _held(ctx, p.side)
            lift = None
            if held:
                tip0 = ctx.tip(p.side)
                z_top = grasp_z + ceiling_h
                pts = [[x, y, float(z)] for z in np.arange(tip0[2] + 0.015, z_top - 1e-6, 0.015)] + [[x, y, float(z_top)]]
                lift = _cartesian_path(ctx, p.side, pts, yaw, speed_mps=0.1, allow_jump=False)  # straight up, same arm configuration
                held, width = _held(ctx, p.side)
            if not held:
                return self.result(SkillStatus.failure, "nothing held after the lift", verified=True, metrics={"stage": "lift", "grasp_z": round(grasp_z, 4), "attempts": attempts, "grasp_log": grasp_log, "pick_status": pick.status.value, "width_after_lift_m": width})
            reasons: list[str] = []
            reason_code = None
            if p.object_width_m is not None and width is not None and width > p.object_width_m + 0.015:
                # the fingers closed on something much wider than the object (the container)
                reasons.append(f"grasp width {width * 100:.1f} cm is far wider than the object ({p.object_width_m * 100:.1f} cm): the fingers closed on the container")
                reason_code = "container_grasp"
                container_check = {"grasp_width_m": round(width, 4), "object_width_m": p.object_width_m, "mixed_grasp": True, "decisive": True}
            else:
                # DECISIVE target-only check after the vertical lift, before any lateral move (a mixed grasp after a tolerated descent error closes
                # on peg AND cup with an object-plausible width and "held" verdicts; the lateral move would then drag the cup - formal_v7 / formal_v9
                # abc_extract s104). Three witnesses: the rim ring in the top depth image (container displacement, or gone), the surface height at
                # the object's place (the target must have risen out: the object top before, the rim / interior after) and, with a prompt, SAM3.
                view_after = _container_view(ctx, p.container_xyz[:2], rim, (x, y), top)
                ring_moved = None
                if view_before["ring_xy"] is not None and view_after["ring_xy"] is not None:
                    ring_moved = float(np.linalg.norm(np.asarray(view_after["ring_xy"]) - np.asarray(view_before["ring_xy"])))
                ring_gone = view_before["ring_xy"] is not None and view_after["ring_xy"] is None and view_after["table_points_at_container"] >= 20  # the table shows where the rim was
                target_rise = None
                if view_before["surface_at_object_z"] is not None and view_after["surface_at_object_z"] is not None:
                    target_rise = float(view_before["surface_at_object_z"] - view_after["surface_at_object_z"])  # how far the surface at the object's place dropped
                after = _detect_near(ctx, p.container_prompt, container_before["xy"]) if container_before is not None else None
                moved = None if after is None or container_before is None else float(np.linalg.norm(np.asarray(after["xy"]) - np.asarray(container_before["xy"])))
                risen = None if after is None or container_before is None or after["top_z"] is None or container_before["top_z"] is None else float(after["top_z"] - container_before["top_z"])
                if ring_moved is not None and ring_moved > CONTAINER_MOVED_M:
                    reasons.append(f"its rim ring moved {ring_moved * 100:.1f} cm (top depth)")
                if ring_gone:
                    reasons.append("its rim ring is no longer where it was (top depth): the container came up with the object")
                if moved is not None and moved > CONTAINER_MOVED_M:
                    reasons.append(f"it moved {moved * 100:.1f} cm (SAM3 '{p.container_prompt}')")
                if risen is not None and risen > CONTAINER_RISEN_M:
                    reasons.append(f"its top rose {risen * 100:.1f} cm (SAM3 '{p.container_prompt}')")
                if reasons:
                    reason_code = "mixed_grasp"
                    reasons = [f"the container {' and '.join(reasons)} with the lift: the fingers hold the object AND the container (mixed grasp, opening {0 if width is None else round(width * 100, 1)} cm)"]
                # the surface at the object's place is informational only: the lifted object stands in that very column (a vertical lift), so the
                # patch reads its body, not the scene - the rim ring (depth) and the SAM3 re-detection decide
                container_check = {"depth_before": view_before, "depth_after": view_after, "ring_moved_m": None if ring_moved is None else round(ring_moved, 4), "ring_gone": bool(ring_gone), "target_rise_m": None if target_rise is None else round(target_rise, 4),
                                   "sam3_before": container_before, "sam3_after": after, "moved_m": None if moved is None else round(moved, 4), "risen_m": None if risen is None else round(risen, 4), "mixed_grasp": bool(reasons), "decisive": bool(reasons or ring_moved is not None or moved is not None)}
            grasp_log.append({"attempt": attempts, "grasp_z": round(grasp_z, 4), "grasp_width_m": None if width is None else round(width, 4), "reason": reason_code, "pad_gate": pick.metrics.get("pad_gate")})
            if not reasons:
                break
            # put the assembly back where it was and let go
            ctx.call("move_to", side=p.side, xyz=[x, y, grasp_z], yaw=yaw, duration=1.2, tolerance_m=0.03, correction_passes=0)
            ctx.call("set_gripper", side=p.side, aperture=1.0, duration=0.4)
            ctx.call("move_to", side=p.side, xyz=[x, y, grasp_z + 0.06], yaw=yaw, duration=0.8, tolerance_m=0.03)
            # 1.3.0: one internal re-grasp HIGHER on the object (the pads further from the rim) when the protrusion leaves room for it
            higher = min(grasp_z + REGRASP_HIGHER_M, tip_max)
            if attempts == 1 and higher >= grasp_z + 0.004 and _reachable_height(ctx, p.side, x, y, higher, 0.30, yaw, prefer_margin=False) is not None:
                ctx.log(f"extract_from_container: {reason_code} at grasp z {grasp_z:.3f}; re-grasping {(higher - grasp_z) * 100:.1f} cm higher")
                grasp_z = higher
                ceiling_h = _reachable_height(ctx, p.side, x, y, grasp_z, 0.30, yaw, prefer_margin=False) or ceiling_h
                continue
            return self.result(SkillStatus.failure, f"{reasons[0]}; lowered back and released, the object is back in the container ({attempts} attempt(s), the last {(grasp_z - rim - _pad_below_tip_m(ctx)) * 100:.1f} cm of pad clearance above the rim). Re-detect both, then call extract_from_container again with a smaller grasp_below_top_m (grasp higher on the object) or the other arm; do NOT retry with move_to / set_gripper primitives and never move laterally while the container follows", verified=True, metrics={"stage": "container_check", "reason": reason_code, "grasp_z": round(grasp_z, 4), "yaw": round(yaw, 3), "grasp_width_m": width, "attempts": attempts, "grasp_log": grasp_log, "container_check": container_check}, outputs={"held": False, "cleared": False})
        tip = ctx.tip(p.side)
        bottom_offset = max(0.0, grasp_z - bottom)
        bottom_above_rim = float(tip[2] - bottom_offset - rim)
        cxy = _xy(p.container_xyz)
        lateral = float(np.linalg.norm(tip[:2] - cxy))
        moved = None
        if bottom_above_rim >= 0.005:
            z = float(tip[2])
            for d in ([-1.0, 0.0], [0.0, 1.0 if p.side == "left" else -1.0], [0.0, -1.0 if p.side == "left" else 1.0], [1.0, 0.0]):
                target = tip[:2] + np.asarray(d) * p.clearance_m
                pts = [[float(q[0]), float(q[1]), z] for q in _line(tip[:2], target, 0.03)]
                if all(ctx.controller.solve_ik(p.side, q, yaw) is not None for q in pts):
                    moved = _cartesian_path(ctx, p.side, pts, yaw, speed_mps=0.2)
                    moved["direction"] = d
                    break
            if moved is None:
                # no full-length lateral path at this height: try a shorter one (>= 4 cm) in the same order
                for d in ([-1.0, 0.0], [0.0, 1.0 if p.side == "left" else -1.0], [0.0, -1.0 if p.side == "left" else 1.0], [1.0, 0.0]):
                    for frac in (0.7, 0.5):
                        if p.clearance_m * frac < 0.04:
                            continue
                        target = tip[:2] + np.asarray(d) * p.clearance_m * frac
                        pts = [[float(q[0]), float(q[1]), z] for q in _line(tip[:2], target, 0.03)]
                        if all(ctx.controller.solve_ik(p.side, q, yaw) is not None for q in pts):
                            moved = _cartesian_path(ctx, p.side, pts, yaw, speed_mps=0.2)
                            moved["direction"] = d
                            break
                    if moved is not None:
                        break
        held_after, width_after = _held(ctx, p.side)
        tip = ctx.tip(p.side)
        lateral = float(np.linalg.norm(tip[:2] - cxy))
        rise = float(tip[2] - grasp_z)
        cleared = bottom_above_rim >= 0.005 or lateral >= p.clearance_m - 0.02
        metrics = {"stage": "clearance", "grasp_z": round(grasp_z, 4), "yaw": round(yaw, 3), "rise_m": round(rise, 4), "lateral_clearance_m": round(lateral, 4), "object_bottom_above_rim_m": round(bottom_above_rim, 4), "width_after_lift_m": width, "width_after_clearance_m": width_after, "pick_status": pick.status.value, "verification_status": pick.metrics.get("verification_status"), "pad_gate": pick.metrics.get("pad_gate"), "attempts": attempts, "grasp_log": grasp_log, "lift_ceiling_above_grasp_m": ceiling_h, "lift": None if lift is None else {"n_reached": lift["n_reached"], "n_waypoints": lift["n_waypoints"], "stop_reason": lift["stop_reason"]}, "lateral_move": None if moved is None else {"direction": moved["direction"], "n_reached": moved["n_reached"], "n_waypoints": moved["n_waypoints"]}, "container_check": container_check}
        outputs = {"held": bool(held_after), "cleared": bool(cleared and held_after), "tip_pos": np.round(tip, 4).tolist(), "opening_width_m": width_after}
        if not held_after:
            return self.result(SkillStatus.failure, f"object lost during the clearance move (opening {0 if width_after is None else round(width_after * 100, 1)} cm)", verified=True, metrics=metrics, outputs=outputs)
        if not cleared:
            return self.result(SkillStatus.failure, f"object held and lifted {rise * 100:.1f} cm, but its bottom is still {-bottom_above_rim * 100:.1f} cm below the rim at this arm's reach ceiling; no lateral move made (it would drag the container). The object is STILL HELD - do not release it over the container", verified=True, metrics=metrics, outputs=outputs)
        return self.result(SkillStatus.success, f"object extracted: held, lifted {rise * 100:.1f} cm above the grasp point and moved {lateral * 100:.1f} cm clear of the container", verified=True, metrics=metrics, outputs=outputs)


# ============================================================================================ insert_into_container
#: Finger pads stay at least this far above the rim during the descent (hand_v5 value kept: a larger clearance shortens the
#: achievable release depth, and the toppling of a released peg is decided by the drop, not by the pads).
INSERT_PAD_RIM_CLEARANCE_M = 0.003
#: The precise settle over the opening happens at the highest reachable height within this much above the nominal carry height:
#: a carry pose at the reach ceiling is held 1-3 cm low, so the extra room keeps the object bottom off the rim.
INSERT_HOVER_EXTRA_M = 0.03
#: A carry pose over the opening with less joint-limit margin than this is a ceiling pose (1-3 cm tracking error).
INSERT_LOW_MARGIN_RAD = 0.03
#: Guided release: the fingers first open to the held width plus this, so the object is bracketed while it drops in.
INSERT_GUIDED_OPEN_M = 0.012


class InsertIntoContainer(Skill):
    name = "insert_into_container"
    version = "1.1.2"
    kind = "code"
    description = (
        "Put the object the arm ALREADY HOLDS INTO a container - also the call to re-align a held object: carry over the opening, re-centre on the rim ring (top "
        "depth), descend in 1 cm steps to contact or `max_descent_m` (pads above the rim), guided release, retreat, confirm (depth); a stall at rim height is retried "
        "on `search_offsets` offsets. A reach-edge carry pose fails BEFORE moving, object still held: never call pick / extract_from_container then (they open the "
        "gripper) - transfer_between_arms, or place and re-pick lower."
    )
    returns = (
        "success if released inside and the camera confirms or cannot judge; failure: nothing held, unreachable / reach edge, on the rim / beside the container "
        "(still held), no visual change. metrics: contact, search_attempts, visual_confirmed"
    )
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK contact; top depth before vs after"
    observations_used = ["proprio", "depth:top", "rgb:top", "rgb:left_wrist", "rgb:right_wrist"]
    dependencies = ["move_to", "set_gripper", "wait"]
    known_limitations = []
    timeout_s = 60.0

    class Params(Params):
        side: Side
        container_xyz: list[float] = Field(min_length=3, max_length=3, description="opening centre xy [m]")
        container_top_z: float = Field(description="[m]")
        object_bottom_offset_m: float = Field(ge=0.0, le=0.30, description="grasp point to object bottom [m]")
        yaw: float | None = Field(default=None)
        clearance_m: float = Field(default=0.01, ge=0.0, le=0.10, description="[m]")
        release_depth_m: float = Field(default=0.01, ge=0.0, le=0.10)
        max_descent_m: float = Field(default=0.08, ge=0.0, le=0.30, description="[m]")
        search_step_m: float = Field(default=0.01, ge=0.0, le=0.03)
        search_offsets: int = Field(default=SEARCH_OFFSETS_DEFAULT, ge=0, le=12)
        use_depth_recentre: bool = Field(default=True)
        target_object: str = Field(default="the held object", max_length=120)

    def run(self, ctx, p):
        cx, cy = float(p.container_xyz[0]), float(p.container_xyz[1])
        rim = float(p.container_top_z)
        off = float(p.object_bottom_offset_m)
        is_held, width_before = _holding(ctx, p.side)
        if not is_held:
            commanded = float(ctx.backend.commanded_action()[6 if p.side == "left" else 13])
            why = "the gripper is commanded open" if commanded >= 0.5 else "the fingers are closed on nothing"
            return self.result(SkillStatus.failure, f"nothing is held by the {p.side} gripper: {why} (opening {0 if width_before is None else round(width_before * 100, 1)} cm)", verified=True, metrics={"stage": "precondition", "opening_width_m": width_before, "commanded_aperture": round(commanded, 3)})
        z_transit = rim + off + p.clearance_m
        # the descent goes to contact or to z_deep; the finger pads must stay ABOVE the rim (a narrow opening would trap the
        # blades: the gripper cannot open inside it and would carry the container away on retreat)
        z_deep = max(rim + _pad_below_tip_m(ctx) + INSERT_PAD_RIM_CLEARANCE_M, rim + off - p.max_descent_m)
        z_target = max(z_deep, rim + off - p.release_depth_m)
        yaws = [float(p.yaw)] if p.yaw is not None else [0.0, math.pi / 2, math.pi / 4, -math.pi / 4]
        yaw = None
        ceilings = {}
        for cand in yaws:
            top_ok = ctx.controller.solve_ik(p.side, [cx, cy, z_transit], cand) is not None
            low_ok = ctx.controller.solve_ik(p.side, [cx, cy, z_target], cand) is not None
            ceilings[round(cand, 2)] = _highest_reachable(ctx, p.side, cx, cy, z_target, z_transit + 0.06, cand)
            if top_ok and low_ok:
                yaw = cand
                break
        if yaw is None:
            best = max((v for v in ceilings.values() if v is not None), default=None)
            return self.result(SkillStatus.failure, f"the opening at ({cx:.3f}, {cy:.3f}) is not reachable with the object bottom above the rim: the tip would need z={z_transit:.3f} but the reach ceiling there is {best}. The object is STILL HELD - do not call pick or extract_from_container (they open the gripper): hand it to the other arm with transfer_between_arms, or place it on the table and re-grasp the object lower (closer to its bottom), or use the other arm from the start", verified=True, metrics={"stage": "transit", "required_tip_z": round(z_transit, 4), "reach_ceiling_z": best, "ceiling_by_yaw": ceilings}, outputs={"held": True})
        # a carry pose ON a joint limit is held 1-3 cm low by the servos: with the object bottom nominally `clearance_m` above the
        # rim it would drag over the rim and push the container over (abc_insert s101: the cup tipped before the release). Only
        # accept such a pose when the 1.5 cm higher point is reachable too (the settle then has room); otherwise fail before moving.
        carry_sol = ctx.controller.solve_ik_detailed(p.side, [cx, cy, z_transit], yaw)
        if carry_sol is not None and carry_sol.limit_margin_rad < INSERT_LOW_MARGIN_RAD and ctx.controller.solve_ik(p.side, [cx, cy, z_transit + 0.015], yaw) is None:
            return self.result(SkillStatus.failure, f"the carry pose over the opening (tip z={z_transit:.3f}, object bottom {p.clearance_m * 100:.1f} cm above the rim) is at the {p.side} arm's reach edge (joint-limit margin {carry_sol.limit_margin_rad:.2f} rad): the servos hold such a pose 1-3 cm low and the object bottom would drag over the rim. Nothing was moved and the object is STILL HELD - do not call pick or extract_from_container (they open the gripper) and do not substitute place over the container: hand it to the other arm with transfer_between_arms, or place it on the table and re-grasp it lower (closer to its bottom)", verified=True, metrics={"stage": "transit", "required_tip_z": round(z_transit, 4), "limit_margin_rad": round(float(carry_sol.limit_margin_rad), 4), "reach_ceiling_z": ceilings.get(round(yaw, 2))}, outputs={"held": True})
        surface_before = _top_surface_z(ctx, (cx, cy), rim)
        if surface_before is not None and surface_before > rim + 0.03:
            surface_before = None  # arm or held object in the line of sight (an empty opening reads about its rim)
        # re-centre the estimate on the rim ring seen by the top depth camera (the caller's centroid can be 1-2 cm off; the
        # ring centroid is within a few mm); ignored when no ring is visible (large container, occlusion) or implausibly far
        rim_ring = _rim_ring(ctx, (cx, cy), rim) if (p.use_depth_recentre and surface_before is not None) else None
        estimate_correction = None
        if rim_ring is not None and float(np.linalg.norm(rim_ring["centre"] - [cx, cy])) <= 0.03:
            c0 = np.array([cx, cy])
            centre = rim_ring["centre"]
            if float(np.linalg.norm(centre - c0)) > 0.005:
                again = _rim_ring(ctx, centre, rim)  # second fit around the corrected centre: the first is truncated by the search radius
                if again is not None and float(np.linalg.norm(again["centre"] - c0)) <= 0.035:
                    centre = again["centre"]
            estimate_correction = [round(float(centre[0] - cx), 4), round(float(centre[1] - cy), 4)]
            cx, cy = float(centre[0]), float(centre[1])
        carry = _carry(ctx, p.side, (cx, cy), z_transit, z_transit + 0.04, yaw)
        if not carry.get("ok"):
            r = ctx.call("move_to", side=p.side, xyz=[cx, cy, z_transit], yaw=yaw, duration=1.5, tolerance_m=0.02)
            if r.status != SkillStatus.success:
                return self.result(SkillStatus.failure, f"could not carry the object over the opening: {r.message}", verified=True, metrics={"stage": "transit", "carry": carry.get("reason")})
        # centre precisely over the opening: the radial clearance is small, so the 3-6 mm a plain move leaves matters; hover as
        # high as reachable within INSERT_HOVER_EXTRA_M above the nominal carry height (extra clearance at the reach edge)
        z_hover = _highest_reachable(ctx, p.side, cx, cy, z_transit, z_transit + INSERT_HOVER_EXTRA_M, yaw) or z_transit
        _settle_precise(ctx, p.side, [cx, cy, z_hover], yaw, tol_m=0.002)
        tip = ctx.tip(p.side)
        xy_err = float(np.linalg.norm(tip[:2] - [cx, cy]))
        held_now, _ = _held(ctx, p.side)
        if not held_now:
            return self.result(SkillStatus.failure, "object lost during the carry", verified=True, metrics={"stage": "transit", "xy_error_m": round(xy_err, 4)})
        # descend inside the footprint in 1 cm steps until contact (or z_deep)
        def descend(x0: float, y0: float) -> tuple[dict, float]:
            pts = [[x0, y0, float(z)] for z in np.arange(z_hover - 0.01, z_deep - 1e-6, -0.01)] + [[x0, y0, float(z_deep)]]
            pts = [q for i, q in enumerate(pts) if i == 0 or abs(q[2] - pts[i - 1][2]) > 1e-4]
            # patience 3: an object whose edge catches the inner rim edge usually slides in when pushed a little longer
            d = _cartesian_path(ctx, p.side, pts, yaw, speed_mps=0.06, hold_s=0.1, correction_passes=1, stall_fraction=0.4, stall_patience=3)
            return d, float(rim - (ctx.tip(p.side)[2] - off))

        # a stall while the object bottom is at the rim height means it rests on the edge; for a long object (hangs >= 3 cm below the
        # grasp) any stall within 2 cm of the rim is treated the same way: its edge is caught on the inner rim edge and would fall
        # anywhere if released (a container floor that shallow would need a shorter object)
        rim_zone = 0.02 if off >= 0.03 else 0.003

        def on_rim(d: dict, below: float) -> bool:
            return d["stalled"] and below < rim_zone

        def on_table(below: float) -> bool:
            return (rim - below) <= ctx.table_z + 0.008  # (short objects only) object bottom at table level: no container floor met

        def outcome(d: dict, below: float) -> str:
            if on_rim(d, below):
                return "rim"
            if on_table(below):
                return "table"
            return "inside" if d["stalled"] else "deep"  # contact inside (floor / contents / pads on the rim) or max_descent reached

        fx, fy = cx, cy
        desc, bottom_below_rim = descend(cx, cy)
        state = outcome(desc, bottom_below_rim)
        attempts = 1
        search_offset = None
        search_stopped = None
        if state in ("rim", "table") and p.search_offsets > 0 and p.search_step_m > 0:
            # peg-in-hole search: the opening estimate is off by more than the radial clearance; lift back over the rim and retry the
            # descent on a ring of offsets around the estimate (touch tells where the opening is; no truth involved). The search is
            # bounded by the remaining robot-time budget: release + retreat + camera check must still fit.
            for k in range(p.search_offsets):
                remaining = _remaining_robot_time_s(ctx)
                if remaining is not None and remaining < SEARCH_MIN_REMAINING_S:
                    search_stopped = f"robot-time budget: {remaining:.1f} s left after {attempts - 1} search offset(s)"
                    break
                _cartesian_path(ctx, p.side, [[fx, fy, z_transit]], yaw, speed_mps=0.12, allow_jump=True)
                ang = 2 * math.pi * k / p.search_offsets
                ox, oy = cx + p.search_step_m * math.cos(ang), cy + p.search_step_m * math.sin(ang)
                lateral = _cartesian_path(ctx, p.side, [[ox, oy, z_transit]], yaw, speed_mps=0.12)
                if lateral["stop_reason"] is not None:
                    continue
                _settle_precise(ctx, p.side, [ox, oy, z_transit], yaw, tol_m=0.002)
                fx, fy = ox, oy
                desc, bottom_below_rim = descend(ox, oy)
                attempts += 1
                state = outcome(desc, bottom_below_rim)
                if state in ("inside", "deep"):
                    search_offset = [round(ox - cx, 4), round(oy - cy, 4)]
                    break
        if state in ("rim", "table"):
            _cartesian_path(ctx, p.side, [[fx, fy, z_transit]], yaw, speed_mps=0.1, allow_jump=True)
            where = f"stalled on the container rim ({-bottom_below_rim * 100:.1f} cm above it)" if state == "rim" else f"reached the table level ({bottom_below_rim * 100:.1f} cm below the rim) without meeting a container floor or rim: it is BESIDE the container"
            tried = f" at the estimate and at {attempts - 1} search offsets of {p.search_step_m * 100:.1f} cm" if attempts > 1 else ""
            stopped = f" (search stopped early: {search_stopped})" if search_stopped else ""
            return self.result(SkillStatus.failure, f"the object {where}{tried}{stopped}: opening centre off by more than the search covers (tip xy error at the estimate {xy_err * 100:.1f} cm). The object is STILL HELD; re-localise the opening and retry", verified=True, metrics={"stage": "descent", "xy_error_m": round(xy_err, 4), "stalled": bool(desc["stalled"]), "contact": state, "search_attempts": attempts, "search_stopped": search_stopped, "object_bottom_below_rim_m": round(bottom_below_rim, 4)}, outputs={"held": True})
        tip = ctx.tip(p.side)
        if desc["stalled"]:
            _cartesian_path(ctx, p.side, [[fx, fy, float(tip[2]) + 0.004]], yaw, speed_mps=0.05)  # unload the contact before letting go
            tip = ctx.tip(p.side)
        release_pos = np.round(tip, 4).tolist()
        # guided release: open to the held width plus INSERT_GUIDED_OPEN_M first so the pads bracket the object while it drops the
        # remaining way to the container floor (a tall peg released from a few cm inside toppled out in 1 of 5 oracle runs), then open
        width_held = ctx.gripper_width(p.side)
        if width_held is not None and width_held + INSERT_GUIDED_OPEN_M < 0.09:
            ctx.call("set_gripper", side=p.side, opening_width_m=round(float(width_held) + INSERT_GUIDED_OPEN_M, 4), duration=0.3)
            ctx.call("wait", seconds=0.2)
        ctx.call("set_gripper", side=p.side, aperture=1.0, duration=0.4)
        ctx.call("wait", seconds=0.3)
        # retreat: up, then sideways out of the top camera's line of sight (the camera looks along +x from the robots' side,
        # so a retreat toward the robots leaves the forearm in the view; a 14 cm step toward the arm's own side does not)
        up = _highest_reachable(ctx, p.side, fx, fy, z_transit, z_transit + 0.05, yaw) or z_transit
        _cartesian_path(ctx, p.side, [[fx, fy, up]], yaw, speed_mps=0.15)
        own = 1.0 if p.side == "left" else -1.0
        for d in ([0.0, own], [-0.6, 0.8 * own], [-1.0, 0.0], [0.0, -own]):
            target = np.array([fx, fy]) + np.asarray(d) * 0.14
            pts = [[float(q[0]), float(q[1]), up] for q in _line([fx, fy], target, 0.04)]
            if all(ctx.controller.solve_ik(p.side, q, yaw) is not None for q in pts):
                _cartesian_path(ctx, p.side, pts, yaw, speed_mps=0.25)
                break
        view = _placement_view(ctx, (fx, fy), rim)
        surface_after = view["surface_max"]
        if surface_after is not None and surface_after > rim + off + 0.10:
            surface_after = None  # something (the arm) is still in the line of sight: no visual judgement
        ring_offset = view["object_to_ring_m"]
        _, wrist_ref = ctx.capture(f"{p.side}_wrist", tag="post_insert")
        _, top_ref = ctx.capture("top", tag="post_insert")
        # visual judgement: something protrudes where the opening was (or the surface there rose) AND, when both the object and
        # the rim ring are visible, the object stands within the ring (an object 1.8+ cm from the ring centre stands BESIDE it;
        # the object is compared with the ring, not the release point, because it may have slid in from an offset release)
        confirmed = None
        if surface_after is not None:
            if surface_after >= rim + 0.01:
                confirmed = True
            elif surface_before is not None:
                confirmed = (surface_after - surface_before) >= 0.015
                if not confirmed and (surface_after - surface_before) >= 0.005:
                    confirmed = None
        beside = ring_offset is not None and ring_offset > 0.018
        if beside:
            confirmed = False
        elif confirmed and view["object_xy"] is not None and (ring_offset is None or ring_offset > 0.012):
            confirmed = None  # object seen but no clean ring (hidden / large container) or marginal: cannot confirm
        metrics = {"stage": "verify", "xy_error_m": round(xy_err, 4), "estimate_correction_m": estimate_correction, "release_tip_z": round(float(release_pos[2]), 4), "object_bottom_below_rim_m": round(bottom_below_rim, 4), "stalled": desc["stalled"], "contact": state, "descent_steps": desc["n_reached"], "search_attempts": attempts, "search_offset_m": search_offset, "yaw": round(yaw, 3), "surface_before_z": None if surface_before is None else round(surface_before, 4), "surface_after_z": None if surface_after is None else round(surface_after, 4), "object_to_rim_centre_m": None if ring_offset is None else round(ring_offset, 4), "visual_confirmed": confirmed}
        outputs = {"release_pos": release_pos, "image": top_ref.model_dump() if top_ref else None, "wrist_image": wrist_ref.model_dump() if wrist_ref else None}
        inside = bottom_below_rim >= 0.0
        if beside:
            return self.result(SkillStatus.failure, f"released, but the top camera sees the object {ring_offset * 100:.1f} cm from the centre of the container rim: it stands BESIDE the container (opening estimate off by more than the search covers)", verified=True, metrics=metrics, outputs=outputs)
        if confirmed is False:
            return self.result(SkillStatus.failure, f"released with the object bottom {bottom_below_rim * 100:.1f} cm below the rim, but the top camera reads the opening at the same height as before ({surface_after:.3f} m vs {surface_before:.3f} m): the object is not inside the container", verified=True, metrics=metrics, outputs=outputs)
        if not inside:
            return self.result(SkillStatus.failure, f"released with the object bottom {-bottom_below_rim * 100:.1f} cm ABOVE the rim (reach limit); it may have fallen beside the container", verified=True, metrics=metrics, outputs=outputs)
        if confirmed is None and surface_after is not None and view["object_xy"] is None and surface_before is None and off >= 0.03:
            # no usable baseline, a long object (hung >= 3 cm below the grasp) and nothing protruding above the rim: cannot be called verified
            return self.result(SkillStatus.unknown, f"released with the object bottom {bottom_below_rim * 100:.1f} cm below the rim, but the top camera sees nothing protruding above the rim ({surface_after:.3f} m vs rim {rim:.3f} m) and no pre-insertion baseline is available: placement not confirmed - re-observe the container", verified=False, metrics=metrics, outputs=outputs)
        how = "confirmed by the top camera" if confirmed else "the top camera cannot judge"
        found = f"; opening found {np.hypot(*search_offset) * 100:.1f} cm from the estimate by the search" if search_offset else ""
        contact = " on contact inside (floor, contents or pads on the rim)" if state == "inside" else " at max_descent without contact"
        return self.result(SkillStatus.success, f"object released inside the container ({bottom_below_rim * 100:.1f} cm below the rim){contact}{found}; {how}", verified=bool(confirmed) or state == "inside", metrics=metrics, outputs=outputs)


# ==================================================================================================== sweep_area
class SweepArea(Skill):
    name = "sweep_area"
    version = "1.0.1"
    kind = "code"
    description = (
        "Wipe an axis-aligned table patch with the closed fingers: parallel Cartesian strokes along `direction_xy` keeping the finger tips "
        "`contact_height` above the table (no bowing). The low part of the closed fingers is a ~1.2 cm column, so each stroke clears ~1.2 cm + "
        "the debris size; passes are spaced by that band up to `max_passes` (a 15 cm patch needs ~8 passes, ~45 s). Strokes start `run_in` "
        "before the patch and end `overshoot` beyond it. Default direction: along y away from the table centre-line."
    )
    returns = "success if at least one pass ran (metrics.passes per pass: status, max_height_deviation_m; band_m, coverage_fraction); failure if no pass is reachable. Item displacement is not observed - re-observe"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK per pass (reach, height deviation)"
    dependencies = ["set_gripper", "move_to"]
    known_limitations = []
    timeout_s = 120.0

    class Params(Params):
        side: Side
        center_xy: list[float] = Field(min_length=2, max_length=2, description="patch centre [m]")
        half_extents_xy: list[float] = Field(min_length=2, max_length=2, description="half sizes along x and y [m]")
        direction_xy: list[float] | None = Field(default=None, description="null = along y away from the centre-line")
        contact_height: float = Field(default=0.0, ge=-0.006, le=0.05, description="tip bottom above the table [m]; 0 = on it")
        debris_size_m: float = Field(default=0.01, ge=0.0, le=0.05, description="item size [m]")
        run_in: float = Field(default=0.04, ge=0.0, le=0.15, description="[m]")
        overshoot: float = Field(default=0.04, ge=0.0, le=0.15, description="[m]")
        pass_overlap: float = Field(default=0.004, ge=0.0, le=0.02)
        max_passes: int = Field(default=10, ge=1, le=16)
        speed: float = Field(default=0.15, ge=0.05, le=0.4, description="[m/s]")

    def run(self, ctx, p):
        c = _xy(p.center_xy)
        hx, hy = abs(float(p.half_extents_xy[0])), abs(float(p.half_extents_xy[1]))
        if p.direction_xy is None:
            u = np.array([0.0, 1.0 if (c[1] > 0 or (c[1] == 0 and p.side == "left")) else -1.0])
        else:
            u = _unit(p.direction_xy)
        n = np.array([-u[1], u[0]])
        yaw = _wrap_half(math.atan2(u[1], u[0]))  # fingers close along the stroke -> plate perpendicular to it
        half_u = abs(hx * u[0]) + abs(hy * u[1])
        half_n = abs(hx * n[0]) + abs(hy * n[1])
        W = 2 * FINGER_BLADE_HALF_M + p.debris_size_m  # band cleared by one stroke (blade width + item size)
        spacing = max(W - p.pass_overlap, 0.008)
        if 2 * half_n <= W:
            offsets = [0.0]
        else:
            k = int(math.ceil((2 * half_n - W) / spacing)) + 1
            k = min(k, p.max_passes)
            offsets = list(np.linspace(-(half_n - W / 2), half_n - W / 2, k))
        actual_spacing = (offsets[-1] - offsets[0]) / (len(offsets) - 1) if len(offsets) > 1 else 0.0
        covered = len(offsets) * W - max(0, len(offsets) - 1) * max(0.0, W - actual_spacing)  # bands minus their overlaps
        coverage = min(1.0, covered / max(2 * half_n, 1e-6))
        z = _tip_z_for_pad_height(ctx, p.contact_height)
        hover = z + 0.06
        ctx.call("set_gripper", side=p.side, aperture=0.0, duration=0.3)
        passes = []
        n_ok = 0
        for off in offsets:
            tip_line = c + n * off  # the blade column is centred on the fingertip point
            approach = ctx.call("move_to", side=p.side, xyz=[float(tip_line[0] - u[0] * (half_u + p.run_in)), float(tip_line[1] - u[1] * (half_u + p.run_in)), hover], yaw=yaw, duration=1.0, tolerance_m=0.03)
            if approach.status != SkillStatus.success:
                passes.append({"offset_m": round(float(off), 4), "status": "unreachable", "message": approach.message[:80]})
                continue
            run_in, overshoot = p.run_in, p.overshoot
            start = tip_line - u * (half_u + run_in)
            end = tip_line + u * (half_u + overshoot)
            for _ in range(3):
                if ctx.controller.solve_ik(p.side, [start[0], start[1], z], yaw) is not None and ctx.controller.solve_ik(p.side, [end[0], end[1], z], yaw) is not None:
                    break
                run_in, overshoot = run_in * 0.5, overshoot * 0.5
                start = tip_line - u * (half_u + run_in)
                end = tip_line + u * (half_u + overshoot)
            else:
                passes.append({"offset_m": round(float(off), 4), "status": "unreachable", "message": "stroke endpoints not reachable at pad height"})
                continue
            _cartesian_path(ctx, p.side, [[float(start[0]), float(start[1]), hover], [float(start[0]), float(start[1]), z + 0.02], [float(start[0]), float(start[1]), z]], yaw, speed_mps=0.15, allow_jump=True)
            zlog: list[float] = []
            stroke = _cartesian_path(ctx, p.side, [[float(q[0]), float(q[1]), z] for q in _line(start, end, 0.03)], yaw, speed_mps=p.speed, hold_s=0.1, correction_passes=1, z_log=zlog)
            _cartesian_path(ctx, p.side, [[float(ctx.tip(p.side)[0]), float(ctx.tip(p.side)[1]), hover]], yaw, speed_mps=0.2, allow_jump=True)
            dev = max((abs(v - z) for v in zlog), default=0.0)
            ok = stroke["stop_reason"] is None
            n_ok += ok
            passes.append({"offset_m": round(float(off), 4), "status": "complete" if ok else stroke["stop_reason"], "n_reached": stroke["n_reached"], "n_waypoints": stroke["n_waypoints"], "max_height_deviation_m": round(dev, 4), "run_in_m": round(run_in, 3), "overshoot_m": round(overshoot, 3)})
        metrics = {"passes": passes, "n_passes_complete": n_ok, "n_passes": len(offsets), "band_m": round(W, 4), "coverage_fraction": round(float(coverage), 3), "pad_height_m": p.contact_height, "tip_z": round(z, 4), "yaw": round(yaw, 3), "direction": np.round(u, 3).tolist(), "swept_area_m2": round(float(n_ok * W * 2 * half_u), 4)}
        if n_ok == 0:
            return self.result(SkillStatus.failure, "no sweep pass could be executed: " + "; ".join(str(q.get("message", q["status"])) for q in passes), verified=True, metrics=metrics)
        return self.result(SkillStatus.success, f"{n_ok}/{len(offsets)} passes complete along {np.round(u, 2).tolist()} ({W * 100:.1f} cm band each, {coverage * 100:.0f}% of the patch width covered) with the finger tips {p.contact_height * 100:.1f} cm above the table (max height deviation {max((q.get('max_height_deviation_m', 0) for q in passes), default=0) * 100:.1f} cm)", verified=True, metrics=metrics)


# ================================================================================================ push_until_stop
def _push_stroke(ctx, side: str, contact_xyz, direction_xy, *, max_travel: float, standoff: float, step_m: float, stall_fraction: float, yaw: float | None, approach_height: float, speed: float, lead: float) -> dict:
    """Closed-finger push: the leading finger surface (``lead`` ahead of the tip along the closing axis) starts ``standoff``
    behind the face, so object travel = tip travel - standoff."""

    u = _unit(direction_xy)
    z = max(float(contact_xyz[2]), ctx.table_z + _pad_min_tip_above_table_m(ctx))
    yaw = _wrap_half(math.atan2(u[1], u[0])) if yaw is None else float(yaw)
    cxy = _xy(contact_xyz)
    start = cxy - u * (standoff + lead)
    out = {"yaw": round(yaw, 3), "tip_z": round(z, 4), "standoff_m": standoff, "finger_lead_m": round(lead, 4)}
    ctx.call("set_gripper", side=side, aperture=0.0, duration=0.3)
    r = ctx.call("move_to", side=side, xyz=[float(start[0]), float(start[1]), z + approach_height], yaw=yaw, duration=1.2, tolerance_m=0.03)
    if r.status != SkillStatus.success:
        return {**out, "ok": False, "reason": f"cannot reach the start: {r.message}", "travel_m": 0.0, "tip_travel_m": 0.0, "stalled": False, "stop_reason": "start"}
    down = _cartesian_path(ctx, side, [[float(start[0]), float(start[1]), z + 0.02], [float(start[0]), float(start[1]), z]], yaw, speed_mps=0.12)
    if down["stop_reason"] is not None:
        return {**out, "ok": False, "reason": "cannot descend to contact height at the start", "travel_m": 0.0, "tip_travel_m": 0.0, "stalled": False, "stop_reason": "descend"}
    s0 = ctx.tip(side).copy()
    goal = start + u * (standoff + max_travel)  # tip goal: leading finger surface max_travel past the face
    pts = [[float(q[0]), float(q[1]), z] for q in _line(s0[:2], goal, step_m)]
    stroke = _cartesian_path(ctx, side, pts, yaw, speed_mps=speed, hold_s=0.1, correction_passes=1, stall_fraction=stall_fraction)
    tip = ctx.tip(side)
    tip_travel = float((tip - s0)[:2] @ u)
    travel = max(0.0, tip_travel - standoff)
    # relieve the contact force, then lift clear
    back = tip[:2] - u * 0.01
    _cartesian_path(ctx, side, [[float(back[0]), float(back[1]), z], [float(back[0]), float(back[1]), z + approach_height]], yaw, speed_mps=0.15)
    reason = stroke["stop_reason"] or "max_travel"
    return {**out, "ok": True, "travel_m": round(travel, 4), "tip_travel_m": round(tip_travel, 4), "stalled": bool(stroke["stalled"]), "stop_reason": reason, "n_steps": stroke["n_reached"], "tip_errors_m": stroke["tip_errors_m"][-3:]}


class PushUntilStop(Skill):
    name = "push_until_stop"
    version = "1.0.1"
    kind = "code"
    description = (
        "Push with the closed fingers along `direction_xy` against the object face at `contact_xyz` (fingertip height; pads kept >= 2 mm above the table; "
        "calibrated pad offset and 1-3 cm finger-link lead on tall faces accounted for) in `step_m` Cartesian steps at constant "
        "height until the tip stalls (end-stop, obstacle), the reach limit, or `max_travel`. A stall with travel >= `min_travel` is SUCCESS: "
        "what closing a drawer looks like."
    )
    returns = "success when the stroke ended (stall / reach limit / max_travel) with travel_m >= min_travel; failure if the start is unreachable or the travel is shorter. metrics: travel_m (object travel), stalled, stop_reason. Object motion inferred from the fingertip: re-observe"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK progress per step (stall detection) and total travel"
    dependencies = ["set_gripper", "move_to"]
    known_limitations = []
    timeout_s = 30.0

    class Params(Params):
        side: Side
        contact_xyz: list[float] = Field(min_length=3, max_length=3, description="point on the face [m]")
        direction_xy: list[float] = Field(min_length=2, max_length=2)
        max_travel: float = Field(ge=0.01, le=0.5, description="[m]")
        min_travel: float = Field(default=0.0, ge=0.0, le=0.5, description="for success [m]")
        face_top_z: float | None = Field(default=None, description="null = tall face")
        standoff: float = Field(default=0.03, ge=0.01, le=0.10, description="behind the contact point [m]")
        step_m: float = Field(default=0.01, ge=0.005, le=0.03)
        stall_fraction: float = Field(default=0.4, ge=0.1, le=0.9, description="stall if step progress below this")
        yaw: float | None = Field(default=None, description="null = close along the push")
        approach_height: float = Field(default=0.06, ge=0.02, le=0.2)
        speed: float = Field(default=0.08, ge=0.02, le=0.3, description="[m/s]")

    def run(self, ctx, p):
        pad_z = max(float(p.contact_xyz[2]), ctx.table_z + _pad_min_tip_above_table_m(ctx)) - _pad_below_tip_m(ctx)
        lead = finger_lead_m(None if p.face_top_z is None else float(p.face_top_z) - pad_z)
        r = _push_stroke(ctx, p.side, p.contact_xyz, p.direction_xy, max_travel=p.max_travel, standoff=p.standoff, step_m=p.step_m, stall_fraction=p.stall_fraction, yaw=p.yaw, approach_height=p.approach_height, speed=p.speed, lead=lead)
        metrics = {k: v for k, v in r.items() if k not in ("ok", "reason")}
        if not r["ok"]:
            return self.result(SkillStatus.failure, r["reason"], verified=True, metrics=metrics)
        if r["travel_m"] + 1e-6 < p.min_travel:
            return self.result(SkillStatus.failure, f"stroke stopped ({r['stop_reason']}) after {r['travel_m'] * 100:.1f} cm of travel, less than the {p.min_travel * 100:.1f} cm required", verified=True, metrics=metrics)
        how = {"stall": "stopped at an end-stop / obstacle", "unreachable": "reached the arm's reach limit", "ik_jump": "reached the arm's configuration limit", "max_travel": "completed max_travel"}.get(r["stop_reason"], f"stopped ({r['stop_reason']})")
        return self.result(SkillStatus.success, f"push {how} after {r['travel_m'] * 100:.1f} cm of travel past the contact point", verified=True, metrics=metrics, outputs={"tip_pos": np.round(ctx.tip(p.side), 4).tolist(), "travel_m": r["travel_m"], "stalled": r["stalled"]})


class Push(Skill):
    name = "push"
    version = "1.1.1"
    kind = "code"
    description = (
        "Push an object on the table: closed fingertips placed `standoff` behind the object (opposite the push direction), then a "
        "Cartesian stroke at constant height to the goal. `contact_height` is the height of the closed-finger PAD BOTTOMS above the "
        "table (the fingertip point sits the calibrated pad offset higher: 0.017 m in sim). The stroke stops when the fingertip stalls; a stall after at least half of the "
        "requested distance counts as success with metrics.stalled=true (end-stop reached). For drawers/lids prefer `push_until_stop`."
    )
    returns = "success if the goal was reached (metrics.reached_goal) or the object stopped at an end-stop after >= 50% of the stroke (metrics.stalled); metrics: travel_m (object travel estimate = tip travel - standoff), tip_travel_m, requested_travel_m, shortfall_m (remaining object travel to the goal), object_travel_est_m (= travel_m), stop_reason, tip_z, finger_lead_m; failure otherwise (stalled early = blocked, start unreachable)"
    preconditions = ["object rests on the table", "start and goal within reach"]
    effect_verification = "fingertip FK progress per step (stall detection) and total travel (object motion not verified)"
    dependencies = ["move_to", "set_gripper"]
    known_limitations = ["no verification of object displacement"]
    timeout_s = 20.0

    class Params(Params):
        side: Side
        start_xyz: list[float] = Field(min_length=3, max_length=3, description="object point to push (centre of the face; z unused)")
        goal_xy: list[float] = Field(min_length=2, max_length=2)
        contact_height: float = Field(default=0.010, ge=0.0, le=0.08, description="pad bottom height above the table [m]")
        standoff: float = Field(default=0.04, ge=0.02, le=0.1)

    def run(self, ctx, p):
        s = _xy(p.start_xyz)
        g = _xy(p.goal_xy)
        d = g - s
        if np.linalg.norm(d) < 1e-3:
            raise SkillError("goal equals start", error_type="invalid_params")
        z = _tip_z_for_pad_height(ctx, p.contact_height)
        # start_xyz is the object centre: its top is about twice the centre height above the table
        face_height_above_pad = 2.0 * (float(p.start_xyz[2]) - ctx.table_z) - max(0.0, p.contact_height)
        r = _push_stroke(ctx, p.side, [float(s[0]), float(s[1]), z], d, max_travel=float(np.linalg.norm(d)), standoff=p.standoff, step_m=0.01, stall_fraction=0.4, yaw=None, approach_height=0.08, speed=0.1, lead=finger_lead_m(face_height_above_pad))
        metrics = {k: v for k, v in r.items() if k not in ("ok", "reason")}
        if not r["ok"]:
            return self.result(SkillStatus.failure, r["reason"], verified=True, metrics=metrics)
        requested = float(np.linalg.norm(d))
        reached = r["stop_reason"] == "max_travel" or r["travel_m"] >= requested - 0.02
        # additive metrics shared with the status-semantics candidate's push (cand_20260909_214510_0ef7c7): reached_goal,
        # requested_travel_m, shortfall_m, object_travel_est_m (here the object estimate IS travel_m: tip travel minus standoff)
        metrics.update({"requested_m": round(requested, 4), "requested_travel_m": round(requested, 4), "reached_goal": bool(reached), "shortfall_m": round(max(0.0, requested - r["travel_m"]), 4), "object_travel_est_m": r["travel_m"]})
        if reached:
            return self.result(SkillStatus.success, f"push stroke complete ({r['travel_m'] * 100:.1f} cm; fingertip reached the goal)", verified=True, metrics=metrics)
        if r["stalled"] and r["travel_m"] >= 0.5 * requested:
            return self.result(SkillStatus.success, f"push stalled {metrics['shortfall_m'] * 100:.1f} cm short of the goal after {r['travel_m'] * 100:.1f} of {requested * 100:.1f} cm: the pushed part is at its goal / end-stop (its position was not observed)", verified=True, metrics=metrics)
        if r["stalled"]:
            return self.result(SkillStatus.failure, f"push blocked: stalled {metrics['shortfall_m'] * 100:.1f} cm short of the goal after {r['travel_m'] * 100:.1f} of {requested * 100:.1f} cm", verified=True, metrics=metrics)
        return self.result(SkillStatus.failure, f"stroke stopped ({r['stop_reason']}) after {r['travel_m'] * 100:.1f} of {requested * 100:.1f} cm without a detected stall", verified=True, metrics=metrics)
