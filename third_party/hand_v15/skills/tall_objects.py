"""Tall-object skills with a PITCHED wrist (candidate cand_20260911_085230_18233b, parent hand_v11@3.7.0; ported from
cand_20260910_221908_2ce510, parent hand_v8, and revised for put_bottles spec v2 - two specified bottles).

Why (measured in the put_bottles scene, docs/put_bottles_diagnosis_20260910.md, docs/results/put_bottles/README.md):
  * the bin rim is 0.950-0.963 m (about 20 cm above the table, inner radius 6.4 cm at the floor, 7.9 cm at the rim) while the
    TOP-DOWN reach ceiling over the bin centre is ~0.87 m for both arms at every yaw: no top-down pose can carry an object over
    the rim. Every method version and the oracle scored 0/5 (or a "success" that was a bottle lying on the TABLE against the
    bin wall, accepted by the wide referee window); the runtime agents correctly reported the carry infeasible;
  * with the approach axis tilted 60 deg toward +x about the finger-closing axis (fingers side by side, pads at one height:
    still a proper pinch across a vertical cylinder) both arms reach (0.65, 0, 1.02-1.10) with 0.65-1.07 rad of joint-limit
    margin; a pinched bottle stays vertical through lift and carry (pad friction);
  * a bottle released with its bottom 3-4 cm ABOVE the rim bounced out in 3 of 4 trials; released INSIDE the opening (bottom
    several cm below the rim) it stays in; a drop 6 cm off-centre landed on the rim and tilted the light free-standing bin;
  * the agent's perception under-estimates the rim (0.928 from a rim pixel vs 0.960) and the object tops (0.884 vs 0.927):
    a carry planned on those numbers catches the rim, and a grasp planned on them sits below the object's centre of mass.
    Both are re-measured here from the top depth camera before any motion (the rim ring and the object's cap are visible);
  * two objects fit side by side on the container floor only at about +-3 cm from the centre: the first drop of an empty
    opening is offset toward the arm's OWN side, the next drop (usually the other arm's) lands on the other side, away from
    what the depth camera sees standing and from this skill's release history.

Skills (deployable observations only: fingertip FK, calibrated gripper width, top depth camera; geometry parameters come
from perception - footprint centre, top height, rim height, opening radius, neighbouring footprints - never from task ids or
absolute positions; the finger-pad offset is the cell calibration's ``finger_pad_below_tip_m``, hand_v11 accessor):
  * ``pick_tall_object``: pitched pinch of a standing tall object at a chosen height above its bottom (default 60 % of its
    height: just above the centre of mass); top / footprint centre refined from depth; joint move to a hover above every
    object top on the max-margin wrist branch, then ONE continuous path down to the pre-grasp and along the approach axis;
    the tilt azimuth (+x, +x +/- 45 deg) is chosen by neighbour clearance and joint-limit margin; calibrated-width check,
    one optional verify_grasp, optional 3 cm test lift.
  * ``place_into_bin``: rim height and opening centre refined from depth (ring fit); one continuous path lifts the held
    object ``clearance_m`` over the rim and carries it over the chosen spot INSIDE the opening (own-side offset, depth + history);
    lateral gate 2 cm; lowered until the pads are just above the rim (object bottom several cm below it), released, short
    sideways retreat, top-depth confirmation. Never releases on the rim: a carry that ends off the spot keeps the object.
  * ``pick_and_place_tall_object``: the two composed (one call per object); ``side='auto'`` picks the arm by REACH (own side
    first: pick chain and carry over the container both reachable with margin), so two objects go one per arm when both
    arms reach; an object whose top is < 10 cm above the table is LYING and goes through ``pick`` (top-down across its long
    axis) + the pitched carry.
Core dependency: ``physapi.sim.control.ArmController`` ``pitch`` (additive; rotation about the closing axis) with the extra
IK seeds for pitched targets; ``move_to`` 1.3.0 / ``solve_ik`` 1.2.0 (this library's reach.py) expose ``pitch_deg``.

Candidate cand_20260911_133402_13dd11 (parent hand_v13@3.9.0): place_into_bin 1.2.0 / pick_and_place_tall_object 1.1.1, from the two
formal_v13 put_bottles failures (runs/analysis_v3/formal_v13/put_bottles, both reproduced in sim with the agent's inputs):
  * s100: the bottle was picked at a 45 deg fallback pitch near the arm base; the carry was planned at 60 deg on another tilt azimuth,
    its straight lift over the grasp was infeasible, the ``move_to`` fallback to the lift point FAILED (unreachable, no motion) and the
    code went on with a joint-space diagonal from the grasp pose (tip z 0.86) to over the bin (z 1.09): the bottle hanging 11 cm below
    the tip swept through the rim and toppled the bin (29 cm, 84 deg); the descent then stalled 10 cm early and the skill still opened
    2.4 cm above the rim. The carry is now TRANSACTIONAL: (1) a straight lift at the PICK wrist to the carry height (or the highest
    reachable height that still puts the object bottom above the rim), verified by FK before anything moves sideways; (2) the wrist
    change to the carry wrist at the lift point (a joint move with the tip fixed); (3) the Cartesian traverse at the carry height; the
    joint-space traverse remains the last resort only after a VERIFIED lift. Nothing is released with the object bottom more than
    RETAIN_ABOVE_RIM_M above the rim: the object is lifted back and the skill fails with ``outputs.held`` True (re-observe, other spot / arm).
  * s102: SAM3 never found the bin, the agent guessed a centre 12 cm short, ``refine_container_from_depth`` accepted only <= 3 cm
    corrections, both bottles were released onto the table in front of the bin and the skill said "released inside the opening"
    (0 of 15 seeds strictly inside with a 12 cm centre error, every call a success). The refinement now searches the top depth image
    for the rim RING within RING_SEARCH_RADIUS_M of the given centre (annulus scoring on a grid, Kasa refit, angular coverage), moves
    the centre to the ring it finds (any distance, reported), and when no ring with enough coverage exists the place fails BEFORE moving
    with the object still held ("no container opening found near ...").

Candidate cand_20260911_182411_f543ce (parent hand_v14@3.10.0, whose tall_objects.py is hand_v13's): the 13dd11 place_into_bin 1.2.0 ported
after formal_v14 put_bottles s100 failed by the same mechanism (GPT-6 truth diagnosis runs/analysis_v3/formal_v14/put_bottles/put_bottles__s100,
reproduced in simulation with the agent's inputs: bin toppled by the joint-space diagonal after the failed lift, release 2.6 cm above the rim),
plus the follow-ups the rejected candidate's agent validation exposed:
  * the pre-bin rise of a PARTIALLY lifted object (top-down / lying pick near the arm base) first slides the object at the LIFT wrist along the
    ray from the container centre to a pre-bin point whose distance also allows for the re-pitch swing of the hanging object (REPITCH_FOLLOW x
    the pitch change, toward the tilt direction = toward the container) and only then rises and re-wrists with the tip fixed in xy; the single
    joint move of the prior art (rise + traverse + re-wrist at once, 5 cm margin) swung the hanging bottle against the light bin (~2 cm nudge);
  * the second object of a container is aimed at the side OPPOSITE the first release (through the centre, at the search radius) instead of the
    arm's own side: identical when the other arm places it, no longer a same-spot drop when one arm places both;
  * the wrist yaw of a top-down grasp is read from the end-effector pose of the robot state (proprioception), not from the simulator model;
  * pick_and_place_tall_object 1.2.0: a caller's "lying" object (top < 10 cm) is checked against the top depth image before the top-down route is
    taken (``posture_from_depth``; additive ``check_posture``): the runtime model called a standing bottle half hidden under the idle forearm
    "lying" on development seed 103 and the descending open gripper knocked it over; a depth-confirmed standing object takes the pitched route
    with the measured top (``metrics.route.posture``).
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
from physapi.sim.control import OPEN_FINGER_HALF_SPAN_M
from physapi.skills.base import Params, Skill, SkillError

_spec = importlib.util.spec_from_file_location(f"physapi_libmod_{hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]}_manipulation_base", Path(__file__).with_name("manipulation.py"))
if _spec.name in sys.modules:  # one base module per library (shared with manipulation_gv.py / cloth_fold.py / contact_container.py)
    _base = sys.modules[_spec.name]
else:
    _base = importlib.util.module_from_spec(_spec)
    assert _spec.loader
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
_held = _base._held
_robot_time = _base._robot_time
_clear_other_arm = _base._clear_other_arm

Side = Literal["left", "right"]

#: Default wrist pitch [deg] about the closing axis (0 = straight down). 60 deg: best joint-limit margins over the bin
#: (0.67-1.07 rad at (0.65, 0, 1.02-1.10)) and at the far bottles (0.84 rad) in the measured scene.
DEFAULT_PITCH_DEG = 60.0
#: Steeper carry fallback when the requested pitch cannot hold the carry height over the opening.
CARRY_PITCH_FALLBACK_DEG = 75.0
#: Default grasp height as a fraction of the object height: above the centre of mass of a bottle (~0.5 of its height, 0.53
#: of the depth-visible height - the top depth camera reads bottle tops 0-4 cm low), so the pinched object hangs as a stable
#: pendulum instead of an indifferent one.
DEFAULT_GRASP_FRACTION = 0.65
#: A held object follows a wrist re-pitch (the carry pitched steeper than the pick) by this fraction of the angle: measured
#: 12-14 deg after a 15 deg re-pitch and 22-24 deg after 30 deg on two bottle types (the pinch is neither a free hinge nor
#: rigid); the drop point is compensated with it, and the +-0.15 spread narrows the drop offset.
REPITCH_FOLLOW = 0.8
REPITCH_FOLLOW_SPREAD = 0.15
#: Guided release: open by this much beyond the held width first (0.3 s) so the object drops free before the pads move fast -
#: a straight full opening dragged the object 4 cm sideways with the high-friction pad it stuck to (landed tilted 35 deg).
RELEASE_GUIDED_OPEN_M = 0.012
#: Which way a released object rolls: the two finger pads do not let go together (one finger is driven, the other follows through
#: a coupling), so the falling object pivots on the last pad and its top leans toward the arm's INNER side (toward the other arm)
#: for both arms on the only wrist branch that reaches over a raised rim (measured: 30-40 deg lean across the opening when dropped
#: on the arm's own side). +1 = toward the inner side. The drop point is therefore preferred on the side the object rolls TOWARD,
#: so it leans on the near wall (~15 deg) and leaves the rest of the opening free for the next object.
RELEASE_ROLL_SIGN = 1.0
#: An object whose top is lower than this above the table is treated as LYING (a 16-18 cm bottle on its side is 5-8 cm high).
LYING_MAX_TOP_M = 0.10
#: Pre-grasp distance along the approach axis (behind and above the grasp point); hover clearance above the object tops.
APPROACH_M = 0.10
HOVER_ABOVE_TOP_M = 0.05
#: Finger geometry for the clearance check: blade half-width (+ margin), blade length back toward the palm, palm (link_6)
#: centre offset and radius. How far the pads reach beyond the fingertip point along the approach axis is the cell
#: calibration's ``finger_pad_below_tip_m`` (passed in by the skills; no constant here).
FINGER_HALF_WIDTH_M = 0.012
BLADE_LENGTH_M = 0.06
PALM_OFFSET_M = 0.135
PALM_RADIUS_M = 0.05
#: Candidate tilt azimuths [deg] of the approach axis relative to +x (away from the robot bases): straight forward and
#: forward +/- 45 deg. The approach always comes from the robots' side, so a nearer object in the same row blocks it -
#: pick the nearer object first.
TILT_AZIMUTHS_DEG = (0.0, 45.0, -45.0)
#: Pitch fallbacks [deg] when the requested pitch has no joint margin: near the arm base (x <= 0.46 m) a 60 deg wrist would
#: sit above the shoulder; 45 / 30 deg keep 0.4-0.6 rad of margin there (ik_scan.json). The carry re-pitches when needed.
PITCH_FALLBACKS_DEG = (45.0, 30.0)
#: A wrist pose closer than this to a joint limit is not used for the hover / approach / carry (a limit pose tracks 3-5 cm off).
MIN_MARGIN_RAD = 0.05
TEST_LIFT_M = 0.03
TEST_LIFT_MIN_RISE_M = 0.01
#: Default obstacle footprint radius (typical bottle body) when a neighbour is given as [x, y] only.
DEFAULT_OBSTACLE_RADIUS_M = 0.035
#: place_into_bin: object bottom above the rim during the carry; lateral alignment tolerance over the drop point before the
#: descent; default opening radius used for the free-spot search; default held-object half span; retreat (sideways, toward
#: the arm's own side).
CARRY_CLEARANCE_M = 0.03
ALIGN_TOL_M = 0.02
DEFAULT_OPENING_RADIUS_M = 0.08
DEFAULT_OBJECT_RADIUS_M = 0.035
#: Drop-point search: grid cell, the largest offset of a drop point from the opening centre (two ~7 cm objects stand side by
#: side at +-3 cm; a fingertip 3 cm off-centre keeps the opening finger inside an 8 cm rim), the wall margin kept between the
#: object's edge and the rim, and the radius around an earlier release inside which a later drop would land on that object.
FREE_SPOT_CELL_M = 0.01
MAX_DROP_OFFSET_M = 0.03
WALL_MARGIN_M = 0.015
AVOID_EARLIER_DROP_M = 0.065
#: A surface inside the opening higher than rim - OCCUPIED_BELOW_RIM_M marks an occupied spot (a standing bottle's cap).
OCCUPIED_BELOW_RIM_M = 0.08
RETREAT_SIDE_M = 0.16
#: Minimum height of the fingertip point above the rim at the release (the pads reach pad_offset * cos(pitch) below it, and
#: a 1 cm settle sag must not put an opening finger on the rim of the light free-standing container).
RIM_TIP_MARGIN_M = 0.02
#: A descent that stops this much above its target met the container's contents (or the rim): release elsewhere.
STALL_DZ_M = 0.025
#: Release on top of the contents: an object released standing on the caps of others topples at random (out in ~half of the
#: trials); lifted LAYDOWN_LIFT_M and re-pitched to LAYDOWN_PITCH_DEG it already leans toward the far wall and lies down inside.
LAYDOWN_LIFT_M = 0.02
LAYDOWN_PITCH_DEG = 35.0
CARRY_SPEED_MPS = 0.4
APPROACH_SPEED_MPS = 0.4
#: Depth refinement: search radius around the given object footprint for its top / cap, band below the top that counts as
#: the cap, acceptance windows against the given estimates (perception under-estimates tops by 3-5 cm, never over-estimates
#: by much); ring band below the measured rim that counts as the rim top surface.
OBJECT_REFINE_RADIUS_M = 0.05
CAP_BAND_M = 0.02
ARC_BAND_M = 0.012
TOP_ACCEPT_ABOVE_M = 0.08
CENTRE_ACCEPT_M = 0.03
#: Drop-point search: a cell the depth camera did not see costs this (below the preference gradient of one cell).
UNSEEN_PENALTY = 0.003
RIM_ACCEPT_BELOW_M = 0.02
RIM_ACCEPT_ABOVE_M = 0.06
RIM_BAND_M = 0.012
MIN_REFINE_POINTS = 12
#: Wide rim-ring search (candidate 13dd11): the given opening centre may be far off (a guessed centre when SAM3 misses the container;
#: formal_v13 s102: 12 cm). Points within RING_SEARCH_RADIUS_M of the given centre whose height is within RING_SEARCH_BAND_M of the
#: given rim height are scored on a RING_GRID_M grid of candidate centres by how many lie in the rim annulus [r - 0.012, r + 0.035]
#: (opening radius r plus the wall) and how many of RING_BINS angular bins they cover; the best cell is refined by a Kasa fit on the
#: annulus points. A ring is accepted when it covers >= RING_MIN_COVERAGE of the bins and its fitted radius is plausible.
RING_SEARCH_RADIUS_M = 0.25
RING_SEARCH_BAND_M = 0.06
RING_GRID_M = 0.02
RING_BINS = 12
RING_MIN_COVERAGE = 0.5
RING_MIN_POINTS = 40
#: place_into_bin never opens with the object bottom more than this above the rim (the object cannot be inside): it lifts back
#: and returns failure with the object still held.
RETAIN_ABOVE_RIM_M = 0.02
#: Lift verification: the lift must end with the tip at least this close to its target height (FK).
LIFT_TOL_M = 0.015
#: When no wrist can lift the object above the rim straight over the grasp (top-down grasps near the arm base, lying objects), lift at
#: least this much, then rise to the carry height in a joint move to a pre-bin point PREBIN_MARGIN_M outside the rim + object radius.
PARTIAL_LIFT_MIN_M = 0.03
PREBIN_MARGIN_M = 0.05
#: Candidate f543ce: a caller's "lying" (top < LYING_MAX_TOP_M) is checked against the top depth image before the top-down route is taken - the
#: runtime model called a standing bottle half hidden under the idle forearm "lying" (development seed 103) and the top-down pick knocked it over.
#: Depth points within OBJECT_REFINE_RADIUS_M of the given centre that stand higher than LYING_MAX_TOP_M + POSTURE_MARGIN_M above the table,
#: whose median xy lies within POSTURE_CENTRE_M of the given centre (the object itself, not a standing neighbour), make it STANDING with that top.
POSTURE_MARGIN_M = 0.02
POSTURE_CENTRE_M = 0.035
POSTURE_MIN_POINTS = 12


# ------------------------------------------------------------------------------------------------------ geometry helpers
def _pad_below_tip_m(ctx) -> float:
    """The cell calibration's ``finger_pad_below_tip_m`` [m] (configs/cell_calibration/<backend>.yaml; hand_v11 accessor)."""

    return float(ctx.cell_calibration.finger_pad_below_tip_m)


def wrist_for_tilt(tilt_xy, pitch_deg: float) -> tuple[float, float, np.ndarray]:
    """(yaw, pitch [rad], approach axis) of a wrist whose approach axis tilts ``pitch_deg`` from straight down toward the
    horizontal direction ``tilt_xy``; the fingers close perpendicular to the tilt direction (``Rz(yaw) Rx(pitch) TOPDOWN``:
    the tilt direction is ``(-sin yaw, cos yaw)``)."""

    d = np.asarray(tilt_xy, dtype=float)[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        raise SkillError("tilt_direction_xy must be a non-zero horizontal direction", error_type="invalid_params")
    d = d / n
    p = math.radians(float(pitch_deg))
    yaw = math.atan2(d[1], d[0]) - math.pi / 2
    axis = np.array([math.sin(p) * d[0], math.sin(p) * d[1], -math.cos(p)])
    return float(yaw), float(p), axis


def tilt_of(yaw: float, pitch: float) -> list[float]:
    """Horizontal tilt direction of the approach axis of the wrist ``(yaw, pitch)`` (the sign of the pitch flips it)."""

    s = 1.0 if pitch >= 0 else -1.0
    return [round(-math.sin(yaw) * s, 3), round(math.cos(yaw) * s, 3)]


def _obstacle_list(obstacles, default_radius: float, default_top_z: float) -> list[tuple[float, float, float, float]]:
    """[(x, y, radius, top_z)] from ``[[x, y], [x, y, radius_m], [x, y, radius_m, top_z]]`` entries."""

    out = []
    for o in obstacles or []:
        vals = [float(v) for v in o]
        if len(vals) < 2:
            raise SkillError("each obstacle needs [x, y], [x, y, radius_m] or [x, y, radius_m, top_z]", error_type="invalid_params")
        r = vals[2] if len(vals) >= 3 and vals[2] > 0 else float(default_radius)
        top = vals[3] if len(vals) >= 4 else float(default_top_z)
        out.append((vals[0], vals[1], r, top))
    return out


def _gripper_points(tip, yaw: float, axis, half_span: float, finger_beyond_tip: float):
    """Representative points of the open gripper at fingertip ``tip``: (point, radius) for the pads, blade samples and palm."""

    c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    points = []
    for sgn in (-1.0, 1.0):
        pad = tip + sgn * half_span * c + finger_beyond_tip * axis
        for s in (0.0, 0.02, 0.04, BLADE_LENGTH_M):
            points.append((pad - s * axis, FINGER_HALF_WIDTH_M))
    points.append((tip - PALM_OFFSET_M * axis, PALM_RADIUS_M))
    return points


def approach_clearance(grasp, yaw: float, axis, obstacles, obstacle_top_z: float, *, finger_beyond_tip_m: float, approach_m: float = APPROACH_M, hover_z: float | None = None, half_span: float = OPEN_FINGER_HALF_SPAN_M) -> tuple[float, int | None]:
    """Smallest horizontal clearance [m] between the open gripper (finger pads, blades, palm) swept along the approach
    (hover -> pre-grasp straight down, pre-grasp -> grasp along the axis) and vertical-cylinder obstacles ``[(x, y, radius[, top_z])]``
    (default top ``obstacle_top_z``); negative = collision. ``finger_beyond_tip_m`` is the calibrated pad offset below the
    fingertip point. Returns (clearance, index of the closest obstacle or None)."""

    grasp = np.asarray(grasp, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if not obstacles:
        return float("inf"), None
    pre = grasp - approach_m * axis
    tips = [grasp - axis * (approach_m - 0.01 * k) for k in range(int(round(approach_m / 0.01)) + 1)]
    if hover_z is not None and hover_z > pre[2] + 1e-6:
        tips += [np.array([pre[0], pre[1], z]) for z in np.arange(pre[2] + 0.01, hover_z + 1e-6, 0.01)]
    best, who = float("inf"), None
    for tip in tips:
        for i, obs in enumerate(obstacles):
            ox, oy, r = obs[0], obs[1], obs[2]
            top = obs[3] if len(obs) >= 4 else obstacle_top_z
            for p, rad in _gripper_points(tip, yaw, axis, half_span, finger_beyond_tip_m):
                if p[2] - rad * 0.6 >= top:
                    continue  # above the obstacle (the palm is treated as a sphere)
                d = math.hypot(p[0] - ox, p[1] - oy) - r - rad
                if d < best:
                    best, who = d, i
    return float(best), who


def _depth_points(ctx, xy, at_z: float, radius_m: float) -> np.ndarray | None:
    """World points (N x 3) of the top depth image within ``radius_m`` (pixel radius at that range) of (xy, at_z); None without depth."""

    try:
        from physapi.perception.locator import world_to_pixel

        K = ctx.backend.camera_intrinsics("top")
        c2w = ctx.backend.camera_to_world("top")
        centre = np.array([float(xy[0]), float(xy[1]), float(at_z)])
        uv = world_to_pixel(centre, K, c2w)
        if uv is None:
            return None
        depth = ctx.capture_depth("top")
        dist = float(np.linalg.norm(centre - c2w[:3, 3]))
        r_px = int(np.clip(round(float(K[0, 0]) * radius_m / max(dist, 0.3)), 2, 120))
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
        pts = (c2w[:3, :3] @ p_cam).T + c2w[:3, 3]
        return pts[np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) <= radius_m]
    except Exception:  # noqa: BLE001 - the depth camera is a best-effort source here
        return None


def _top_surface_z(ctx, xy, at_z: float, radius_m: float) -> float | None:
    pts = _depth_points(ctx, xy, at_z, radius_m)
    if pts is None or not len(pts):
        return None
    return float(np.percentile(pts[:, 2], 98))


def refine_object_from_depth(ctx, xy, top_z: float, table_z: float, grasp_z: float | None = None) -> dict:
    """Re-measure a standing object's top height and axis from the top depth camera.

    Top: the 99th-percentile height of the points within OBJECT_REFINE_RADIUS_M of the given centre that stand clear of the
    table; the HIGHER of the given and the measured top is kept (both under-read: perception's ``top_z`` - a depth percentile
    inside a SAM mask - was 3-5 cm low on every bottle of the evidence, and the depth camera itself misses a thin spout), unless
    the measurement is implausibly high. Centre: a least-squares circle through the body points in a +-ARC_BAND_M band at the
    grasp height (the camera sees the near half of the body as an arc; the fit recovers the axis to ~2 mm and the body radius
    there), falling back to the centroid of the top CAP_BAND_M (biased toward the camera by up to 2 cm). Measurements outside
    the acceptance windows (or too few points) keep the given values."""

    x, y = float(xy[0]), float(xy[1])
    out = {"top_z_given": round(float(top_z), 4), "centre_given": [round(x, 4), round(y, 4)], "top_z": float(top_z), "centre_xy": [x, y], "n_points": 0, "top_refined": False, "centre_refined": False, "centre_method": None}
    pts = _depth_points(ctx, (x, y), float(top_z), OBJECT_REFINE_RADIUS_M)
    if pts is None or not len(pts):
        return out
    pts = pts[pts[:, 2] > float(table_z) + 0.03]
    out["n_points"] = int(len(pts))
    if len(pts) < MIN_REFINE_POINTS:
        return out
    top = float(np.percentile(pts[:, 2], 99))
    out["top_z_measured"] = round(top, 4)
    if float(top_z) < top <= float(top_z) + TOP_ACCEPT_ABOVE_M:
        out["top_z"] = top
        out["top_refined"] = True
    gz = float(grasp_z) if grasp_z is not None else float(table_z) + DEFAULT_GRASP_FRACTION * (out["top_z"] - float(table_z))
    band = pts[(pts[:, 2] >= gz - ARC_BAND_M) & (pts[:, 2] <= gz + ARC_BAND_M)]
    fit = _fit_circle(band[:, :2]) if len(band) >= MIN_REFINE_POINTS else None
    if fit is not None and 0.012 <= fit[2] <= 0.06 and math.hypot(fit[0] - x, fit[1] - y) <= CENTRE_ACCEPT_M:
        out.update({"centre_xy": [fit[0], fit[1]], "centre_measured": [round(fit[0], 4), round(fit[1], 4)], "body_radius_m": round(fit[2], 4), "centre_refined": True, "centre_method": "arc_fit", "n_arc": int(len(band))})
        return out
    cap = pts[pts[:, 2] >= top - CAP_BAND_M]
    if len(cap) >= 5:
        cx, cy = float(np.median(cap[:, 0])), float(np.median(cap[:, 1]))
        out["centre_measured"] = [round(cx, 4), round(cy, 4)]
        if math.hypot(cx - x, cy - y) <= CENTRE_ACCEPT_M:
            out.update({"centre_xy": [cx, cy], "centre_refined": True, "centre_method": "cap_centroid"})
    return out


def posture_from_depth(ctx, xy, table_z: float) -> dict:
    """Is the object at ``xy`` STANDING according to the top depth image? (candidate f543ce; constants POSTURE_*). Returns
    {"n_points", "n_high", "standing": True | False | None (nothing seen), "top_z" (99th percentile of the high points when standing),
    "high_centre_xy"}. High points whose median lies farther than POSTURE_CENTRE_M from the given centre belong to a neighbour: not standing."""

    x, y = float(xy[0]), float(xy[1])
    out = {"n_points": 0, "n_high": 0, "standing": None, "top_z": None, "high_centre_xy": None}
    pts = _depth_points(ctx, (x, y), float(table_z) + 0.06, OBJECT_REFINE_RADIUS_M)
    if pts is None or not len(pts):
        return out
    pts = pts[pts[:, 2] > float(table_z) + 0.03]
    out["n_points"] = int(len(pts))
    high = pts[pts[:, 2] > float(table_z) + LYING_MAX_TOP_M + POSTURE_MARGIN_M]
    out["n_high"] = int(len(high))
    if len(high) < POSTURE_MIN_POINTS:
        out["standing"] = False if len(pts) >= POSTURE_MIN_POINTS else None
        return out
    cx, cy = float(np.median(high[:, 0])), float(np.median(high[:, 1]))
    out["high_centre_xy"] = [round(cx, 4), round(cy, 4)]
    if math.hypot(cx - x, cy - y) <= POSTURE_CENTRE_M:
        out["standing"] = True
        out["top_z"] = round(float(np.percentile(high[:, 2], 99)), 4)
    else:
        out["standing"] = False
    return out


def _fit_circle(xy: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic (Kasa) least-squares circle through points (N x 2): (cx, cy, r) or None."""

    if len(xy) < 6:
        return None
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, bb, c = (float(v) for v in sol)
    cx, cy = -a / 2.0, -bb / 2.0
    r2 = cx * cx + cy * cy - c
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _ring_coverage(pts_xy: np.ndarray, centre, bins: int = RING_BINS) -> float:
    """Fraction of ``bins`` equal angular sectors around ``centre`` that contain at least one of ``pts_xy``."""

    if pts_xy is None or not len(pts_xy):
        return 0.0
    ang = np.arctan2(pts_xy[:, 1] - float(centre[1]), pts_xy[:, 0] - float(centre[0]))
    idx = np.floor((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    return float(len(np.unique(idx)) / bins)


def find_rim_ring(band_xyz: np.ndarray, given_xy, opening_radius_m: float, *, search_radius_m: float = RING_SEARCH_RADIUS_M, grid_m: float = RING_GRID_M) -> dict | None:
    """Locate a container's rim ring among rim-height depth points (N x 3, world) anywhere within ``search_radius_m`` of ``given_xy``.

    Candidate centres on a ``grid_m`` grid are scored by the number of points in the rim annulus [r - 0.012, r + 0.035] around them
    (``r`` = the opening radius; the annulus spans the rim's wall) weighted by the angular coverage of those points; the best cell is
    refined by a Kasa circle fit on its annulus points (one re-selection of the annulus around the fitted centre). Returns
    {"centre_xy", "radius_m", "coverage", "n_ring", "rim_z", "score"} or None when no cell has RING_MIN_POINTS annulus points."""

    if band_xyz is None or len(band_xyz) < RING_MIN_POINTS:
        return None
    gx, gy = float(given_xy[0]), float(given_xy[1])
    r = float(opening_radius_m)
    lo, hi = r - 0.012, r + 0.035
    xy = band_xyz[:, :2]
    n = int(math.floor(search_radius_m / grid_m + 1e-9))
    best = None
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            cx, cy = gx + i * grid_m, gy + j * grid_m
            if math.hypot(i * grid_m, j * grid_m) > search_radius_m + 1e-9:
                continue
            d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
            sel = (d >= lo) & (d <= hi)
            k = int(sel.sum())
            if k < RING_MIN_POINTS:
                continue
            cov = _ring_coverage(xy[sel], (cx, cy))
            score = k * cov * cov
            if best is None or score > best[0]:
                best = (score, cx, cy, sel)
    if best is None:
        return None
    _score, cx, cy, sel = best
    for _ in range(2):  # refine: fit the circle through the annulus points, re-select the annulus around the fitted centre
        fit = _fit_circle(xy[sel])
        if fit is None:
            break
        cx, cy = float(fit[0]), float(fit[1])
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        sel = (d >= lo) & (d <= hi)
        if int(sel.sum()) < RING_MIN_POINTS:
            return None
    fit = _fit_circle(xy[sel])
    radius = float(fit[2]) if fit is not None else float(np.median(np.hypot(xy[sel, 0] - cx, xy[sel, 1] - cy)))
    cov = _ring_coverage(xy[sel], (cx, cy))
    return {"centre_xy": [round(cx, 4), round(cy, 4)], "radius_m": round(radius, 4), "coverage": round(cov, 3), "n_ring": int(sel.sum()), "rim_z": round(float(np.percentile(band_xyz[sel, 2], 90)), 4), "score": round(float(_score), 1)}


def refine_container_from_depth(ctx, xy, rim_z: float, opening_radius_m: float) -> dict:
    """Re-measure an open container's rim height and opening centre from the top depth camera.

    Local pass (hand_v12): among the points within opening_radius + 4 cm of the given centre and within the acceptance band of the
    given rim height, the 97th-percentile height is the rim; the points within RIM_BAND_M below it form the rim ring, whose
    least-squares circle gives the opening centre (accepted within CENTRE_ACCEPT_M). Wide pass (candidate 13dd11): the rim RING is
    searched within RING_SEARCH_RADIUS_M of the given centre (:func:`find_rim_ring`) and validated by its angular coverage
    (>= RING_MIN_COVERAGE) and radius (0.6-1.6 x the opening radius): a ring found there moves the centre by any distance
    (``centre_correction_m`` reported; formal_v13 s102 handed the skill a centre 12 cm short of the bin). ``ring_found`` False with
    ``wide_searched`` True means no container opening is visible near the given centre - the caller must not release there."""

    x, y = float(xy[0]), float(xy[1])
    out = {"rim_z_given": round(float(rim_z), 4), "centre_given": [round(x, 4), round(y, 4)], "rim_z": float(rim_z), "centre_xy": [x, y], "n_points": 0, "n_ring": 0, "rim_refined": False, "centre_refined": False, "ring_found": False, "ring_coverage": None, "centre_correction_m": 0.0, "wide_searched": False}
    pts = _depth_points(ctx, (x, y), float(rim_z), float(opening_radius_m) + 0.04)
    local_ok = False
    if pts is not None and len(pts):
        band = pts[(pts[:, 2] >= float(rim_z) - RIM_ACCEPT_BELOW_M - 0.04) & (pts[:, 2] <= float(rim_z) + RIM_ACCEPT_ABOVE_M)]
        out["n_points"] = int(len(band))
        if len(band) >= MIN_REFINE_POINTS:
            rim = float(np.percentile(band[:, 2], 97))
            out["rim_z_measured"] = round(rim, 4)
            ring = band[band[:, 2] >= rim - RIM_BAND_M]
            out["n_ring"] = int(len(ring))
            if len(ring) >= MIN_REFINE_POINTS:
                # accept the local ring only through a validated circle FIT: a plausible radius, a centre within CENTRE_ACCEPT_M of the
                # given one and the ring covering most angles around the FITTED centre. The median of a partial arc (the given centre 6-7 cm
                # off: only the far half of the rim inside the local window) lies near the arc and passes a coverage test around itself -
                # it put a drop point on the far wall in a validation episode; such a case goes to the wide search below.
                fit = _fit_circle(ring[:, :2])
                cx, cy = float(np.median(ring[:, 0])), float(np.median(ring[:, 1]))
                fit_ok = fit is not None and 0.6 * float(opening_radius_m) <= fit[2] <= 1.6 * float(opening_radius_m) and math.hypot(fit[0] - x, fit[1] - y) <= CENTRE_ACCEPT_M
                if fit_ok:
                    cx, cy = fit[0], fit[1]
                    out["ring_fit_radius_m"] = round(fit[2], 4)
                cov = _ring_coverage(ring[:, :2], (cx, cy))
                out["centre_measured"] = [round(cx, 4), round(cy, 4)]
                out["ring_coverage"] = round(cov, 3)
                if fit_ok and cov >= RING_MIN_COVERAGE:
                    local_ok = True
                    if float(rim_z) - RIM_ACCEPT_BELOW_M <= rim <= float(rim_z) + RIM_ACCEPT_ABOVE_M:
                        out["rim_z"] = rim
                        out["rim_refined"] = True
                    out["centre_xy"] = [cx, cy]
                    out["centre_refined"] = True
                    out["ring_found"] = True
                    out["centre_correction_m"] = round(math.hypot(cx - x, cy - y), 4)
    if local_ok:
        return out
    # wide pass: the ring is not where the caller said (or not complete there): look for it anywhere within the search radius
    out["wide_searched"] = True
    wide = _depth_points(ctx, (x, y), float(rim_z), RING_SEARCH_RADIUS_M + float(opening_radius_m) + 0.04)
    if wide is None or not len(wide):
        return out
    band = wide[(wide[:, 2] >= float(rim_z) - RING_SEARCH_BAND_M) & (wide[:, 2] <= float(rim_z) + RING_SEARCH_BAND_M)]
    out["n_points_wide"] = int(len(band))
    ring = find_rim_ring(band, (x, y), float(opening_radius_m))
    out["wide_ring"] = ring
    if ring is None:
        return out
    plausible = 0.6 * float(opening_radius_m) <= ring["radius_m"] <= 1.6 * float(opening_radius_m) and ring["coverage"] >= RING_MIN_COVERAGE
    if not plausible:
        return out
    cx, cy = ring["centre_xy"]
    out.update({"centre_xy": [cx, cy], "centre_measured": [cx, cy], "centre_refined": True, "ring_found": True, "ring_coverage": ring["coverage"], "ring_fit_radius_m": ring["radius_m"], "n_ring": ring["n_ring"], "centre_correction_m": round(math.hypot(cx - x, cy - y), 4)})
    # the rim height at the ring: the top of the annulus points (97th percentile, as in the local pass)
    d = np.hypot(band[:, 0] - cx, band[:, 1] - cy)
    ann = band[(d >= float(opening_radius_m) - 0.012) & (d <= float(opening_radius_m) + 0.035)]
    if len(ann) >= MIN_REFINE_POINTS:
        rim = float(np.percentile(ann[:, 2], 97))
        out["rim_z_measured"] = round(rim, 4)
        if float(rim_z) - RING_SEARCH_BAND_M <= rim <= float(rim_z) + RING_SEARCH_BAND_M:
            out["rim_z"] = rim
            out["rim_refined"] = True
    return out


def free_spots(centre_xy, rim_z: float, search_r: float, pts: np.ndarray | None, history, *, prefer_xy=None, cell: float = FREE_SPOT_CELL_M, avoid_r: float = AVOID_EARLIER_DROP_M) -> list[tuple[float, float, float | None, float]]:
    """Candidate drop points inside an opening, best first: grid cells within ``search_r`` of the centre scored by the surface the
    top depth camera sees there (``pts`` world points; a surface within OCCUPIED_BELOW_RIM_M of the rim = a standing object; a cell
    without points is unseen and costs a little), by the distance to earlier release points (``history``: (x, y) pairs; a cell
    closer than ``avoid_r`` to one would drop the object onto the earlier one) and by the distance to the preferred point
    ``prefer_xy`` (default the centre; place_into_bin prefers the arm's own side so a second object lands beside the first).
    Returns [(x, y, surface_z or None, score)] sorted by score (lower is better)."""

    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    px, py = (cx, cy) if prefer_xy is None else (float(prefer_xy[0]), float(prefer_xy[1]))
    n = int(math.floor(search_r / cell + 1e-9))
    cells = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = cx + i * cell, cy + j * cell
            if math.hypot(x - cx, y - cy) <= search_r + 1e-9:
                cells.append((x, y))
    if not cells:
        cells = [(cx, cy)]
    out = []
    for x, y in cells:
        surface = None
        if pts is not None and len(pts):
            near = pts[(np.abs(pts[:, 0] - x) <= cell) & (np.abs(pts[:, 1] - y) <= cell)]
            if len(near):
                surface = float(np.percentile(near[:, 2], 95))
        occupied_depth = 0.0 if surface is None else max(0.0, surface - (rim_z - OCCUPIED_BELOW_RIM_M))  # how far a seen surface pokes into the top 8 cm
        unseen = UNSEEN_PENALTY if surface is None else 0.0
        hist_pen = 0.0
        for hx, hy in history or []:
            d = math.hypot(x - hx, y - hy)
            if d < avoid_r:
                hist_pen = max(hist_pen, avoid_r - d)
        score = 10.0 * occupied_depth + 4.0 * hist_pen + unseen + 0.5 * math.hypot(x - px, y - py)
        out.append((round(x, 4), round(y, 4), None if surface is None else round(surface, 4), round(score, 4)))
    out.sort(key=lambda t: t[3])
    return out


def _grasp_record(ctx, side: str) -> dict | None:
    return (ctx.scratch.get("tall_grasp") or {}).get(side)


def _current_wrist_yaw(ctx, side: str) -> float | None:
    """Yaw of the arm's current (assumed top-down) wrist from the end-effector pose of the robot state (proprioception: ``R = Rz(yaw)
    TOPDOWN``); None when the state carries no pose (a top-down grasp made by ``pick`` leaves no tall_grasp record)."""

    try:
        R = np.asarray(ctx.backend.state().ee_pose[side], dtype=float).reshape(4, 4)[:3, :3] @ ctx.controller.TOPDOWN.T
        return float(math.atan2(R[1, 0], R[0, 0]))
    except Exception:  # noqa: BLE001
        return None


def _remaining_robot_time_s(ctx) -> float | None:
    """Robot time left in the episode budget (None when the executor has no budget)."""

    try:
        return float(ctx.executor.budget.remaining_execution_s(ctx.backend))
    except Exception:  # noqa: BLE001
        return None


def _r4(v) -> list[float]:
    return [round(float(x), 4) for x in v]


def _best_branch(ctrl, side: str, points, yaw: float, pitch: float):
    """Wrist branch ((yaw, pitch) or its half-turn twin) whose chained IK over ``points`` (first point free, the rest seeded
    from the previous solution) has the largest minimum joint-limit margin. Returns (yaw, pitch, margin, solutions) or None."""

    best = None
    for byaw, bpitch in ctrl.symmetric_branches(yaw, pitch)[:2]:
        sols = []
        seed = None
        for p in points:
            sol = ctrl.solve_ik_detailed(side, p, byaw, pitch=bpitch, symmetric=False, seed_qpos=seed, reference_qpos=seed)
            if sol is None:
                sols = None
                break
            sols.append(sol)
            seed = sol.qpos
        if sols is None:
            continue
        margin = min(s.limit_margin_rad for s in sols)
        if best is None or margin > best[2]:
            best = (byaw, bpitch, margin, sols)
    return best


def plan_pitched_pick(ctx, side: str, grasp, top_z: float, obstacles, pitch_deg: float, tilt_direction_xy=None) -> list[tuple]:
    """Every (tilt azimuth, pitch) option of a pitched pick, no motion: the record (reachable with margin, clearance, blocker,
    margin), the wrist branch, the approach axis, the pre-grasp and the hover point. Stops at the first pitch (requested,
    then the flatter fallbacks) that has a reachable AND clear option. Shared by pick_tall_object and the arm choice of
    pick_and_place_tall_object."""

    ctrl = ctx.controller
    pad = _pad_below_tip_m(ctx)
    grasp = np.asarray(grasp, dtype=float)
    azimuths = [np.asarray(tilt_direction_xy, dtype=float)] if tilt_direction_xy is not None else [np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]) for a in TILT_AZIMUTHS_DEG]
    pitches = [float(pitch_deg)] + [q for q in PITCH_FALLBACKS_DEG if abs(q - float(pitch_deg)) > 1e-6]
    options = []
    for pd in pitches:
        for d in azimuths:
            yaw, pitch, axis = wrist_for_tilt(d, pd)
            pre = grasp - APPROACH_M * axis
            hover_z = max(float(pre[2]), float(top_z) + HOVER_ABOVE_TOP_M + pad * math.cos(pitch))
            hover = np.array([pre[0], pre[1], hover_z])
            branch = _best_branch(ctrl, side, [hover, pre, grasp, grasp + [0.0, 0.0, TEST_LIFT_M]], yaw, pitch)
            clearance, blocker = approach_clearance(grasp, yaw, axis, obstacles, float(top_z), finger_beyond_tip_m=pad, hover_z=hover_z)
            margin = None if branch is None else round(branch[2], 3)
            rec = {"tilt_direction_xy": [round(float(d[0]), 3), round(float(d[1]), 3)], "pitch_deg": pd, "yaw": round(yaw, 3), "reachable": branch is not None and branch[2] >= MIN_MARGIN_RAD, "clearance_m": None if math.isinf(clearance) else round(clearance, 4), "blocker": blocker, "limit_margin_rad": margin}
            options.append((rec, branch, axis, pre, hover, clearance))
        if any(o[0]["reachable"] and o[5] >= 0.0 for o in options):
            break  # the requested (or first fallback) pitch works: do not go flatter than needed
    return options


def _feasible(options) -> list[tuple]:
    return [o for o in options if o[0]["reachable"] and o[5] >= 0.0]


def carry_reachable(ctx, side: str, centre_xy, rim_z: float, hang: float, clearance: float, pitch_deg: float) -> dict | None:
    """The carry / release column over the opening centre for one arm: the wrist (yaw, pitch) among the tilt azimuths at the
    requested pitch (then the steeper fallback) whose branch reaches both the carry height (object bottom ``clearance`` above
    the rim) and the release height with margin. None when no wrist does."""

    ctrl = ctx.controller
    pad = _pad_below_tip_m(ctx)
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    for pd in [float(pitch_deg)] + ([CARRY_PITCH_FALLBACK_DEG] if abs(float(pitch_deg) - CARRY_PITCH_FALLBACK_DEG) > 1e-6 else []):
        pitch = math.radians(pd)
        z_carry = float(rim_z) + float(hang) + float(clearance)
        z_release = min(float(rim_z) + pad * abs(math.cos(pitch)) + RIM_TIP_MARGIN_M, z_carry)
        for a in TILT_AZIMUTHS_DEG:
            yaw, _pp, _ax = wrist_for_tilt([math.cos(math.radians(a)), math.sin(math.radians(a))], pd)
            branch = _best_branch(ctrl, side, [[cx, cy, z_carry], [cx, cy, z_release]], yaw, pitch)
            if branch is not None and branch[2] >= MIN_MARGIN_RAD:
                return {"yaw": round(branch[0], 3), "pitch_deg": round(math.degrees(branch[1]), 1), "limit_margin_rad": round(branch[2], 3), "carry_z": round(z_carry, 4), "release_z": round(z_release, 4)}
    return None


# ------------------------------------------------------------------------------------------------------ pick_tall_object
class PickTallObject(Skill):
    name = "pick_tall_object"
    version = "1.1.0"
    kind = "code"
    description = (
        "Pick a TALL standing object (bottle, can) with a PITCHED wrist: approach tilted pitch_deg from vertical toward +x, fingers closing across the "
        "object grasp_height_m above the table (default 60 % of its height); reaches where a top-down wrist cannot. Top and footprint centre are re-measured "
        "from the top depth camera first. Swings in above all object tops, then down and forward from the robots' side: give the other objects as "
        "obstacles_xy, nearest object first. Lying objects (top < 10 cm): use pick."
    )
    returns = (
        "success when held: outputs grasp_height_above_bottom_m, grasp_xyz, yaw, pitch_deg, opening_width_m; metrics refined, clearance_m, "
        "limit_margin_rad, times. failure: unreachable / blocked approach, approach missed, closed on nothing"
    )
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "calibrated opening (close, test lift) + optional verify_grasp; fingertip FK"
    observations_used = ["proprio", "depth:top", "rgb:left_wrist", "rgb:right_wrist"]
    dependencies = ["move_to", "set_gripper", "verify_grasp"]
    known_limitations = []
    timeout_s = 25.0

    class Params(Params):
        side: Side
        object_xyz: list[float] = Field(min_length=2, max_length=3, description="footprint centre [m]")
        object_top_z: float = Field(description="top z [m]")
        grasp_height_m: float | None = Field(default=None, ge=0.02, le=0.20, description="[m] above the table")
        pitch_deg: float = Field(default=DEFAULT_PITCH_DEG, ge=20.0, le=85.0)
        tilt_direction_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="default auto")
        obstacles_xy: list[list[float]] = Field(default=[], description="[[x, y, radius_m, top_z]]")
        target_object: str = Field(default="the target object", max_length=120)
        refine_from_depth: bool = Field(default=True)
        test_lift: bool = True
        max_verifications: int = Field(default=1, ge=0, le=2)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        times: dict[str, float] = {}

        def lap(name):
            times[name] = round(_robot_time(ctx, tick0) - sum(times.values()), 3)

        table = float(ctx.table_z)
        x, y = float(p.object_xyz[0]), float(p.object_xyz[1])
        top_z = float(p.object_top_z)
        grasp_h0 = float(p.grasp_height_m) if p.grasp_height_m is not None else float(np.clip(DEFAULT_GRASP_FRACTION * (top_z - table), 0.03, 0.12))
        refined = refine_object_from_depth(ctx, (x, y), top_z, table, grasp_z=table + grasp_h0) if p.refine_from_depth else None
        if refined is not None:
            top_z = float(refined["top_z"])
            x, y = float(refined["centre_xy"][0]), float(refined["centre_xy"][1])
        height = top_z - table
        if height < 0.05:
            raise SkillError(f"object top {top_z:.3f} is only {height * 100:.1f} cm above the table: pick_tall_object is for objects taller than 5 cm (use pick)", error_type="precondition")
        grasp_h = float(p.grasp_height_m) if p.grasp_height_m is not None else float(np.clip(DEFAULT_GRASP_FRACTION * height, 0.03, 0.12))
        if grasp_h > height - 0.01:
            raise SkillError(f"grasp_height_m {grasp_h:.3f} is above the object top ({height:.3f} above the table)", error_type="invalid_params")
        grasp = np.array([x, y, table + grasp_h])
        obstacles = _obstacle_list(p.obstacles_xy, DEFAULT_OBSTACLE_RADIUS_M, top_z)
        pitch_deg = float(p.pitch_deg)
        ctrl = ctx.controller
        # ---- choose the tilt azimuth and wrist branch: reachable hover / pre-grasp / grasp / test-lift chain with margin,
        #      neighbour clearance, then margin
        options = plan_pitched_pick(ctx, p.side, grasp, top_z, obstacles, pitch_deg, p.tilt_direction_xy)
        feasible = _feasible(options)
        base_metrics = {"refined": refined, "grasp_height_m": round(grasp_h, 4)}
        if not feasible:
            reach = [o for o in options if o[0]["reachable"]]
            if not reach:
                return self.result(SkillStatus.failure, f"no pitched approach of the {p.side} arm reaches the grasp point {_r4(grasp)} with joint-limit margin (pitch {pitch_deg:.0f} / {' / '.join(str(int(q)) for q in PITCH_FALLBACKS_DEG)} deg, tilt toward +x / +x+-45 deg): use the other arm or a different grasp height", verified=True, metrics={**base_metrics, "stage": "plan", "reachable": False, "options": [o[0] for o in options]})
            worst = max(reach, key=lambda o: o[5])
            ox, oy = obstacles[worst[0]["blocker"]][:2]
            return self.result(SkillStatus.failure, f"every reachable approach of the {p.side} arm would hit the object at ({ox:.3f}, {oy:.3f}) (clearance {worst[5] * 100:.1f} cm): pick that object first (the gripper comes from the robots' side), or give tilt_direction_xy", verified=True, metrics={**base_metrics, "stage": "plan", "reachable": True, "blocked": True, "options": [o[0] for o in options]})
        rec, branch, axis, pre, hover, clearance = max(feasible, key=lambda o: (min(o[5], 0.03), o[0]["limit_margin_rad"] or 0.0))
        byaw, bpitch, _margin, _sols = branch
        bpitch_deg = math.degrees(bpitch)
        pitch_deg = float(rec["pitch_deg"])
        cleared = _clear_other_arm(ctx, p.side, x, y)
        # ---- hover (joint move above every top on the chosen branch, gripper open), then ONE continuous path: down to the pre-grasp, along the axis
        above = ctx.call("move_to", side=p.side, xyz=_r4(hover), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, gripper=1.0, correction_passes=0, tolerance_m=0.03)
        lap("hover_s")
        if above.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"hover move failed: {above.message}", verified=True, metrics={**base_metrics, "stage": "hover", "option": rec, "other_arm_cleared": cleared, "times": times, "robot_time_s": _robot_time(ctx, tick0)})
        _, before_ref = ctx.capture(f"{p.side}_wrist", tag="pre_grasp")
        path = ctrl.move_cartesian_path_both([{p.side: _r4(pre)}, {p.side: _r4(grasp)}], {p.side: byaw}, pitches={p.side: bpitch}, speed_mps=APPROACH_SPEED_MPS, symmetric=False)
        approach_plan = "path"
        if path["completed"]:
            _before_c, err, _passes = ctrl.correct_tip(p.side, grasp, byaw, pitch=bpitch, passes=1)
        else:
            approach_plan = f"fallback ({path.get('aborted_reason')})"
            ctx.call("move_to", side=p.side, xyz=_r4(pre), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, cartesian=True, correction_passes=0)
            down = ctx.call("move_to", side=p.side, xyz=_r4(grasp), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, cartesian=True, correction_passes=2, tolerance_m=0.015)
            err = float(down.metrics.get("tip_error_m") or np.linalg.norm(ctx.tip(p.side) - grasp))
        if err > 0.015:  # one joint-space re-solve at the grasp point (the path end can lag at a far reach)
            down = ctx.call("move_to", side=p.side, xyz=_r4(grasp), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=2, tolerance_m=0.015)
            err = float(down.metrics.get("tip_error_m") or np.linalg.norm(ctx.tip(p.side) - grasp))
            approach_plan += "+resolve"
        lap("approach_s")
        if err > 0.015:
            ctx.call("move_to", side=p.side, xyz=_r4(hover), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=0)
            return self.result(SkillStatus.failure, f"approach missed the grasp point by {err * 100:.1f} cm ({approach_plan})", verified=True, metrics={**base_metrics, "stage": "approach", "option": rec, "tip_error_m": round(err, 4), "times": times, "robot_time_s": _robot_time(ctx, tick0)})
        # ---- close, width rule, optional model check, optional test lift
        ctx.call("set_gripper", side=p.side, aperture=0.0, duration=0.4)
        held_now, width_now = _held(ctx, p.side)
        lap("close_s")
        if not held_now:
            ctx.call("set_gripper", side=p.side, aperture=1.0, duration=0.3)
            ctx.call("move_to", side=p.side, xyz=_r4(pre), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, cartesian=True, correction_passes=0)
            return self.result(SkillStatus.failure, f"fingers closed on nothing at {_r4(grasp)} (opening {0 if width_now is None else round(width_now * 100, 1)} cm): the footprint estimate is off, or the object is not standing (a lying object: use pick) - re-observe", verified=True, metrics={**base_metrics, "stage": "grasp", "option": rec, "grasp_width_m": width_now, "tip_error_m": round(err, 4), "times": times, "robot_time_s": _robot_time(ctx, tick0)})
        verify = None
        if p.max_verifications > 0 and "verify_grasp" in ctx.library:
            r = ctx.call("verify_grasp", side=p.side, target_object=p.target_object, context="closed the pitched gripper around the object", before_frames=[before_ref.path] if before_ref else [])
            verify = {"status": (r.outputs or {}).get("status", "uncertain"), "availability": (r.outputs or {}).get("availability", "ok" if r.status == SkillStatus.success else "unavailable"), "holding_target": (r.outputs or {}).get("holding_target")}
        rise = None
        width_final = width_now
        if p.test_lift:
            z_before = float(ctx.tip(p.side)[2])
            ctx.call("move_to", side=p.side, xyz=_r4(grasp + [0.0, 0.0, TEST_LIFT_M]), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, cartesian=True, correction_passes=0)
            rise = float(ctx.tip(p.side)[2]) - z_before
            held_lift, width_final = _held(ctx, p.side)
            lap("test_lift_s")
            if not held_lift or rise < TEST_LIFT_MIN_RISE_M:
                ctx.call("set_gripper", side=p.side, aperture=1.0, duration=0.3)
                ctx.call("move_to", side=p.side, xyz=_r4(hover), yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=0)
                return self.result(SkillStatus.failure, f"object lost in the test lift (opening {0 if width_final is None else round(width_final * 100, 1)} cm, rise {rise * 100:.1f} cm)", verified=True, metrics={**base_metrics, "stage": "test_lift", "option": rec, "verification": verify, "times": times, "robot_time_s": _robot_time(ctx, tick0)})
        status = "held" if verify and verify["status"] == "held" else "held_proprio_only"
        record = {"grasp_height_above_bottom_m": round(grasp_h, 4), "grasp_xyz": _r4(grasp), "yaw": round(byaw, 4), "pitch_deg": round(bpitch_deg, 3), "tilt_direction_xy": rec["tilt_direction_xy"], "opening_width_m": None if width_final is None else round(float(width_final), 4), "object_top_z": round(top_z, 4), "object_height_m": round(height, 4)}
        ctx.scratch.setdefault("tall_grasp", {})[p.side] = record
        conflict = verify is not None and verify["status"] == "not_held"
        how = "model-confirmed" if status == "held" else ("proprioception - the model said not_held, the held opening decides" if conflict else "proprioception")
        lifted = f" through the {TEST_LIFT_M * 100:.0f} cm test lift" if p.test_lift else ""
        refined_note = ""
        if refined is not None and (refined["top_refined"] or refined["centre_refined"]):
            refined_note = f"; depth re-measured the object (top {refined['top_z_given']:.3f} -> {top_z:.3f} m, centre moved {math.hypot(x - refined['centre_given'][0], y - refined['centre_given'][1]) * 100:.1f} cm)"
        msg = f"held (opening {width_final * 100:.1f} cm{lifted}; {how}); grasped {grasp_h * 100:.1f} cm above the object bottom, wrist pitched {pitch_deg:.0f} deg toward {rec['tilt_direction_xy']}" + (f" (requested {p.pitch_deg:.0f} deg had no joint margin here)" if abs(pitch_deg - float(p.pitch_deg)) > 1e-6 else "") + refined_note
        return self.result(SkillStatus.success, msg, verified=status == "held", metrics={**base_metrics, "stage": "test_lift" if p.test_lift else "grasp", "verification_status": status, "verification": verify, "verifier_conflict": conflict, "option": rec, "clearance_m": rec["clearance_m"], "limit_margin_rad": rec["limit_margin_rad"], "approach": approach_plan, "tip_error_m": round(err, 4), "test_lift_rise_m": None if rise is None else round(rise, 4), "other_arm_cleared": cleared, "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={**record, "held": True, "lift_pos": np.round(ctx.tip(p.side), 4).tolist()})


# ------------------------------------------------------------------------------------------------------ place_into_bin
class PlaceIntoBin(Skill):
    name = "place_into_bin"
    version = "1.2.0"
    kind = "code"
    description = (
        "Drop the held object INTO an open container (bin; rim above the top-down reach), wrist pitched. The rim ring is found in the top depth image within "
        "25 cm of container_xyz (an off centre is corrected; no ring = refused). Lifts straight up (verified), carries over a free spot inside the opening, "
        "lowers until the pads are just above the rim, releases. Never releases on or above the rim (object kept)."
    )
    returns = (
        "success when released with the object bottom below the rim (FK) and the fingers opened; outputs release_pos, drop_xy, object_bottom_below_rim_m; "
        "metrics refined, carry_plan. failure (outputs.held = still holding): no opening, lift / carry infeasible, descent stopped above the rim"
    )
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK at the release, opening after release, top depth before/after"
    observations_used = ["proprio", "depth:top"]
    dependencies = ["move_to", "set_gripper", "wait"]
    known_limitations = []
    timeout_s = 25.0

    class Params(Params):
        side: Side
        container_xyz: list[float] = Field(min_length=2, max_length=3, description="opening centre [m]")
        container_rim_z: float = Field(description="rim z [m]")
        object_bottom_offset_m: float | None = Field(default=None, ge=0.0, le=0.30, description="default: the pick's; give after a top-down pick")
        object_half_span_m: float = Field(default=DEFAULT_OBJECT_RADIUS_M, ge=0.01, le=0.15)
        pitch_deg: float | None = Field(default=None, ge=20.0, le=85.0)
        yaw: float | None = Field(default=None)
        opening_radius_m: float = Field(default=DEFAULT_OPENING_RADIUS_M, ge=0.03, le=0.30, description="inner radius [m]")
        clearance_m: float = Field(default=CARRY_CLEARANCE_M, ge=0.0, le=0.10)
        refine_from_depth: bool = Field(default=True)
        target_object: str = Field(default="the held object", max_length=120)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        times: dict[str, float] = {}

        def lap(name):
            times[name] = round(_robot_time(ctx, tick0) - sum(times.values()), 3)

        pad = _pad_below_tip_m(ctx)
        cx, cy = float(p.container_xyz[0]), float(p.container_xyz[1])
        rim = float(p.container_rim_z)
        is_held, width_before = _held(ctx, p.side)
        if not is_held:
            return self.result(SkillStatus.failure, f"nothing appears to be held by the {p.side} gripper (opening {0 if width_before is None else round(width_before * 100, 1)} cm)", verified=True, metrics={"stage": "precondition"})
        refined = refine_container_from_depth(ctx, (cx, cy), rim, float(p.opening_radius_m)) if p.refine_from_depth else None
        if refined is not None:
            if refined.get("wide_searched") and not refined.get("ring_found") and refined.get("n_points_wide", 0) >= RING_MIN_POINTS:
                # candidate 13dd11: rim-height surfaces are in view but none forms a ring around any point within the search radius:
                # the given centre is not a container opening (formal_v13 s102: a guessed centre 12 cm short of the bin) - releasing
                # there would put the object on the table. Nothing moved; the object is still held.
                return self.result(SkillStatus.failure, f"no container opening found within {RING_SEARCH_RADIUS_M * 100:.0f} cm of ({cx:.3f}, {cy:.3f}) at rim height ~{rim:.3f} m in the top depth image (rim ring coverage {refined.get('ring_coverage')}): the container centre is wrong - re-observe it (segment_objects on the container, or its footprint centre) and call place_into_bin again; nothing moved, the object is still held", verified=True, metrics={"stage": "container", "refined": refined, "reachable": None}, outputs={"held": True})
            rim = float(refined["rim_z"])
            cx, cy = float(refined["centre_xy"][0]), float(refined["centre_xy"][1])
        rec = _grasp_record(ctx, p.side) or {}
        hang0 = float(p.object_bottom_offset_m) if p.object_bottom_offset_m is not None else float(rec.get("grasp_height_above_bottom_m", 0.08))
        pitch_abs = float(p.pitch_deg) if p.pitch_deg is not None else DEFAULT_PITCH_DEG

        def swing(carry_pitch_deg: float) -> tuple[float, float, float, float]:
            """(re-pitch angle, forward offset of the object bottom, effective hang, compensation uncertainty) when the carry is
            pitched ``carry_pitch_deg`` and the object was picked at the record's pitch: the object follows REPITCH_FOLLOW of the
            re-pitch (measured), its bottom swings toward the tilt direction and rises."""

            if rec.get("pitch_deg") is None or p.yaw is not None:
                return 0.0, 0.0, hang0, 0.0
            d = float(carry_pitch_deg) - abs(float(rec["pitch_deg"]))
            if abs(d) < 1e-6:
                return 0.0, 0.0, hang0, 0.0
            theta = math.radians(REPITCH_FOLLOW * abs(d))
            spread = hang0 * (math.sin(math.radians((REPITCH_FOLLOW + REPITCH_FOLLOW_SPREAD) * abs(d))) - math.sin(math.radians((REPITCH_FOLLOW - REPITCH_FOLLOW_SPREAD) * abs(d)))) / 2.0
            return d, hang0 * math.sin(theta) * (1.0 if d > 0 else -1.0), hang0 * math.cos(theta), spread

        repitch_deg, forward, hang, uncertainty = swing(pitch_abs)
        z_carry = rim + hang + float(p.clearance_m)
        pitch = math.radians(pitch_abs)
        z_release = min(rim + pad * abs(math.cos(pitch)) + RIM_TIP_MARGIN_M, z_carry)  # pads stay above the rim
        ctrl = ctx.controller
        own = 1.0 if p.side == "left" else -1.0
        # ---- drop point: inside the opening (object edge inside the rim), preferring this arm's own side so a second object from
        #      the other arm lands beside the first; away from what the top depth camera sees standing and from earlier releases
        search_r = float(np.clip(float(p.opening_radius_m) - float(p.object_half_span_m) - WALL_MARGIN_M - uncertainty, 0.0, MAX_DROP_OFFSET_M))
        prefer = (cx, cy - own * RELEASE_ROLL_SIGN * search_r)
        history = [tuple(h) for h in (ctx.scratch.get("bin_drops") or [])]
        if history:
            # candidate f543ce: a later object is aimed at the side OPPOSITE the last release (through the centre, at the search radius): two
            # ~7 cm objects share a ~13 cm floor only side by side. The own-side rule above coincides with this when the OTHER arm places the
            # second object (the usual two-arm case) and would aim a same-arm second object at the first one's spot.
            hx, hy = float(history[-1][0]), float(history[-1][1])
            away = np.array([cx - hx, cy - hy])
            n_away = float(np.linalg.norm(away))
            if n_away > 1e-6:
                prefer = (cx + away[0] / n_away * search_r, cy + away[1] / n_away * search_r)
        pts = _depth_points(ctx, (cx, cy), rim, search_r + 0.03) if search_r > 0 else None
        if pts is not None:
            pts = pts[pts[:, 2] < rim + 0.05]  # points above the rim belong to the arm / held object, not to the opening
        spots = free_spots((cx, cy), rim, search_r, pts, history, prefer_xy=prefer) if search_r > 0 else [(cx, cy, None, 0.0)]
        surface_before = None if pts is None or not len(pts) else float(np.percentile(pts[:, 2], 98))
        # ---- carry wrist: the pick's own wrist (yaw AND pitch: no re-pitch, no swing) first, then the requested / pick's yaw at the
        #      requested pitch, then the tilt azimuth candidates at the requested pitch, then the steeper fallback
        groups: list[list[tuple[float, float]]] = [[]]
        if p.yaw is not None:
            groups[0].append((float(p.yaw), pitch))
        elif rec.get("yaw") is not None:
            if rec.get("pitch_deg") is not None and 20.0 <= abs(float(rec["pitch_deg"])) <= 85.0:
                groups[0].append((float(rec["yaw"]), math.radians(float(rec["pitch_deg"]))))
            groups[0].append((float(rec["yaw"]), math.copysign(pitch, float(rec.get("pitch_deg", pitch_abs)))))
        for pd in [pitch_abs] + ([CARRY_PITCH_FALLBACK_DEG] if abs(pitch_abs - CARRY_PITCH_FALLBACK_DEG) > 1e-6 else []):
            if pd != pitch_abs:
                groups.append([])
            for a in TILT_AZIMUTHS_DEG:
                yaw_a, _pp, _ax = wrist_for_tilt([math.cos(math.radians(a)), math.sin(math.radians(a))], pd)
                groups[-1].append((yaw_a, math.radians(pd)))
        tip = ctx.tip(p.side)
        lift_pt = [float(tip[0]), float(tip[1]), z_carry]
        chosen = None
        tried = []
        # candidate 13dd11: within each pitch group (the requested pitch first; the steeper fallback only when nothing at the requested pitch
        # reaches - it re-pitches the held object and shifts the drop) prefer a wrist whose branch reaches the lift point above the grasp AND
        # the drop columns (one continuous chain, no wrist change), then the drop columns alone (the wrist change happens at the lift point)
        for group in groups:
            for full_chain in (True, False):
                for yaw_c, pitch_c in group:
                    z_rel_c = min(rim + pad * abs(math.cos(pitch_c)) + RIM_TIP_MARGIN_M, z_carry)
                    for dx, dy, _surf, _score in spots[:3]:
                        pts_chain = ([lift_pt] if full_chain else []) + [[dx, dy, z_carry], [dx, dy, z_rel_c]]
                        branch = _best_branch(ctrl, p.side, pts_chain, yaw_c, pitch_c)
                        tried.append({"yaw": round(yaw_c, 3), "pitch_deg": round(math.degrees(pitch_c), 1), "drop_xy": [dx, dy], "full_chain": full_chain, "reachable": branch is not None, "limit_margin_rad": None if branch is None else round(branch[2], 3)})
                        if branch is not None and branch[2] >= MIN_MARGIN_RAD:
                            chosen = (branch[0], branch[1], dx, dy, full_chain)
                            z_release = z_rel_c
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            return self.result(SkillStatus.failure, f"the opening at ({cx:.3f}, {cy:.3f}) is not reachable by the {p.side} arm with the object bottom {p.clearance_m * 100:.0f} cm above the rim (tip z {z_carry:.3f}, pitch {pitch_abs:.0f} deg): re-grasp the object lower (smaller grasp height) or use the other arm", verified=True, metrics={"stage": "plan", "refined": refined, "reachable": False, "required_tip_z": round(z_carry, 4), "tried": tried[:12]}, outputs={"held": True})
        byaw, bpitch, dx, dy, full_chain = chosen
        bpitch_deg = math.degrees(bpitch)
        if abs(abs(bpitch_deg) - pitch_abs) > 1e-6:
            repitch_deg, forward, hang, uncertainty = swing(abs(bpitch_deg))  # the carry ended on a fallback pitch: recompute the swing
        spot = next((s for s in spots if s[0] == dx and s[1] == dy), spots[0])
        if abs(forward) > 1e-6:
            tdir = tilt_of(byaw, bpitch)
            dx, dy = round(dx - forward * tdir[0], 4), round(dy - forward * tdir[1], 4)
        # ---- TRANSACTIONAL carry (candidate 13dd11). (1) LIFT straight up at the pick's wrist (the arm is in it; no swing) to the carry
        #      height, or to the highest reachable height that still puts the object bottom above the rim; verified by FK before any
        #      sideways motion. (2) the wrist change to the carry wrist at the lift point (tip fixed). (3) the Cartesian traverse at the
        #      carry height; a joint-space traverse is the last resort and only after the verified lift. formal_v13 s100: the old
        #      fallback swept the hanging bottle through the rim from the grasp pose when its lift move had failed.
        z_min_lift = rim + hang + 0.01
        # (yaw, pitch, symmetric): the grasp wrist first. A top-down grasp made by ``pick`` (no tall_grasp record; the lying-object route)
        # lifts on ITS wrist - the caller's yaw when given, else the yaw read from the end-effector - with the symmetric branches allowed
        # (pick may have settled on the yaw + pi branch); planning the lift on another yaw rotates the wrist during the first 2 cm step and
        # the chain is refused as a branch flip
        lift_wrists: list[tuple[float | None, float, bool]] = []
        if rec.get("yaw") is not None and rec.get("pitch_deg") is not None:
            lift_wrists.append((float(rec["yaw"]), math.radians(float(rec["pitch_deg"])), False))
        else:
            grasp_yaw = float(p.yaw) if p.yaw is not None else _current_wrist_yaw(ctx, p.side)
            lift_wrists.append((grasp_yaw, 0.0, True))
        lift_wrists.append((byaw, bpitch, False))
        lift_plan = None
        for lw_yaw, lw_pitch, lw_sym in lift_wrists:
            z_l = z_carry
            while z_l >= z_min_lift - 1e-9:
                pl = ctrl.plan_cartesian_both({p.side: [lift_pt[0], lift_pt[1], z_l]}, {p.side: lw_yaw}, pitches={p.side: lw_pitch}, symmetric=lw_sym)
                if pl["feasible"]:
                    lift_plan = (lw_yaw, lw_pitch, lw_sym, z_l)
                    break
                z_l = round(z_l - 0.01, 4)
            if lift_plan is not None:
                break
        partial_lift = False
        if lift_plan is None:
            # no wrist lifts the object above the rim straight over the grasp (a top-down grasp near the arm base: the top-down ceiling is
            # ~0.93 m; also the lying-object route). Lift as high as the grasp wrist reaches (>= PARTIAL_LIFT_MIN_M), then rise the rest of
            # the way in a joint move to a PRE-BIN point outside the rim (+ object + margin) - the swing happens away from the container.
            tip0 = ctx.tip(p.side)
            for lw_yaw, lw_pitch, lw_sym in lift_wrists:
                z_l = z_min_lift
                while z_l >= float(tip0[2]) + PARTIAL_LIFT_MIN_M - 1e-9:
                    pl = ctrl.plan_cartesian_both({p.side: [lift_pt[0], lift_pt[1], z_l]}, {p.side: lw_yaw}, pitches={p.side: lw_pitch}, symmetric=lw_sym)
                    if pl["feasible"]:
                        lift_plan = (lw_yaw, lw_pitch, lw_sym, z_l)
                        partial_lift = True
                        break
                    z_l = round(z_l - 0.01, 4)
                if lift_plan is not None:
                    break
        if lift_plan is None:
            return self.result(SkillStatus.failure, f"the object cannot be lifted straight up from the grasp pose (needs tip z >= {z_min_lift:.3f} at ({lift_pt[0]:.3f}, {lift_pt[1]:.3f}) to clear the rim, and not even {PARTIAL_LIFT_MIN_M * 100:.0f} cm of lift is reachable): re-grasp the object lower (smaller grasp height) or use the other arm; nothing moved, the object is still held", verified=True, metrics={"stage": "lift_plan", "refined": refined, "reachable": False, "required_tip_z": round(z_min_lift, 4), "carry_z": round(z_carry, 4)}, outputs={"held": True})
        lw_yaw, lw_pitch, lw_sym, z_lift = lift_plan
        z_tip0 = float(tip[2])
        lift = ctrl.move_cartesian_path_both([{p.side: [lift_pt[0], lift_pt[1], z_lift]}], {p.side: lw_yaw}, pitches={p.side: lw_pitch}, speed_mps=CARRY_SPEED_MPS, symmetric=lw_sym)
        tip_l = ctx.tip(p.side)
        lifted = bool(lift["completed"]) and float(tip_l[2]) >= z_lift - LIFT_TOL_M
        if not lifted and float(tip_l[2]) - z_tip0 >= PARTIAL_LIFT_MIN_M - 0.005:
            partial_lift = True  # the arm sagged short of the planned height at its reach ceiling, but the object rose: continue via the pre-bin point
        elif not lifted:
            lap("lift_s")
            return self.result(SkillStatus.failure, f"the lift above the grasp did not reach its height (tip z {float(tip_l[2]):.3f} vs {z_lift:.3f}; {lift.get('aborted_reason') or 'tracking'}): the object was NOT carried over the container and is still held - re-observe, then retry with the other arm or a lower grasp", verified=True, metrics={"stage": "lift", "refined": refined, "carry_z": round(z_carry, 4), "lift_z": round(z_lift, 4), "tip_z": round(float(tip_l[2]), 4), "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True})
        carry_plan = "lift" if not partial_lift else "partial_lift"
        wrist_changed = False
        same_wrist = lw_yaw is not None and abs(lw_yaw - byaw) <= 1e-6 and abs(lw_pitch - bpitch) <= 1e-6
        if partial_lift:
            # rise to the carry height on the carry wrist at a PRE-BIN point: radially outside the rim by the object, a margin AND the swing of
            # the hanging object when the wrist re-pitches (REPITCH_FOLLOW of the pitch change, toward the tilt direction - i.e. toward the
            # container when the tilt azimuth points at it; candidate f543ce: the prior art's fixed 5 cm margin let that swing nudge the light bin).
            # The object first SLIDES there at the lift wrist along the ray from the container centre (never closer to the centre than the
            # pre-bin point: a bottom still below the rim stays outside the wall), then rises and re-wrists with the tip fixed in xy, away from
            # the container. Only when the slide is infeasible does the single joint move of the prior art carry it there.
            to_tip = np.array([float(tip_l[0]) - cx, float(tip_l[1]) - cy])
            dist_c = float(np.linalg.norm(to_tip))
            swing_m = float(hang) * math.sin(math.radians(REPITCH_FOLLOW * abs(abs(bpitch_deg) - abs(math.degrees(float(lw_pitch))))))
            r_out = float(p.opening_radius_m) + float(p.object_half_span_m) + PREBIN_MARGIN_M + swing_m
            u = to_tip / dist_c if dist_c > 1e-6 else np.array([-1.0, 0.0])
            r_pre = max(r_out, min(dist_c, r_out + 0.10))
            pre = [round(float(cx + u[0] * r_pre), 4), round(float(cy + u[1] * r_pre), 4), round(z_carry, 4)]
            slid = False
            if dist_c > r_pre + 0.02:
                slide = ctrl.move_cartesian_path_both([{p.side: [pre[0], pre[1], float(tip_l[2])]}], {p.side: lw_yaw}, pitches={p.side: lw_pitch}, speed_mps=CARRY_SPEED_MPS, symmetric=lw_sym)
                slid = bool(slide["completed"])
                carry_plan += "+slide" if slid else f"+slide_failed ({slide.get('aborted_reason')})"
            t_s = ctx.tip(p.side)
            rise_xy = [round(float(t_s[0]), 4), round(float(t_s[1]), 4)] if (slid or dist_c <= r_pre + 0.02) else pre[:2]
            rw = ctx.call("move_to", side=p.side, xyz=[rise_xy[0], rise_xy[1], round(z_carry, 4)], yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=0, tolerance_m=0.04)
            t_pre = ctx.tip(p.side)
            wrist_changed = rw.status == SkillStatus.success and float(t_pre[2]) >= z_min_lift - LIFT_TOL_M
            carry_plan += "+prebin" if wrist_changed else "+prebin_failed"
            if not wrist_changed:
                back = ctrl.move_cartesian_path_both([{p.side: [lift_pt[0], lift_pt[1], z_lift]}], {p.side: lw_yaw}, pitches={p.side: lw_pitch}, speed_mps=CARRY_SPEED_MPS, symmetric=lw_sym)
                lap("carry_s")
                return self.result(SkillStatus.failure, f"the object could not be raised to the carry height next to the container (pre-bin point {pre} on the carry wrist: {rw.message[:80]}); it was brought back above its pick point ({'ok' if back['completed'] else back.get('aborted_reason')}) and is still held - re-grasp lower / use the other arm", verified=True, metrics={"stage": "prebin", "refined": refined, "carry_z": round(z_carry, 4), "carry_plan": carry_plan, "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True})
        elif not same_wrist:
            # wrist change with the tip where it is (the object swings by REPITCH_FOLLOW of the re-pitch; compensated in the drop point)
            rw = ctx.call("move_to", side=p.side, xyz=[round(float(tip_l[0]), 4), round(float(tip_l[1]), 4), round(float(tip_l[2]), 4)], yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=0, tolerance_m=0.04)
            wrist_changed = rw.status == SkillStatus.success
            carry_plan += "+rewrist" if wrist_changed else "+rewrist_failed"
            if not wrist_changed:
                # the carry wrist has no solution at the lift point: bring the object toward the container along the carry height with
                # the lift wrist (stopping radially outside the rim + object + margin) and try there; still nothing = keep the object
                to = np.array([dx - lift_pt[0], dy - lift_pt[1]])
                dist = float(np.linalg.norm(to))
                stop = max(0.0, dist - (float(p.opening_radius_m) + float(p.object_half_span_m) + 0.04))
                if stop > 0.02:
                    mid = [round(float(lift_pt[0] + to[0] / dist * stop), 4), round(float(lift_pt[1] + to[1] / dist * stop), 4), round(float(tip_l[2]), 4)]
                    mv = ctrl.move_cartesian_path_both([{p.side: mid}], {p.side: lw_yaw}, pitches={p.side: lw_pitch}, speed_mps=CARRY_SPEED_MPS, symmetric=lw_sym)
                    if mv["completed"]:
                        t2 = ctx.tip(p.side)
                        rw = ctx.call("move_to", side=p.side, xyz=[round(float(t2[0]), 4), round(float(t2[1]), 4), round(float(t2[2]), 4)], yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=0, tolerance_m=0.04)
                        wrist_changed = rw.status == SkillStatus.success
                        carry_plan += "+approach" + ("+rewrist" if wrist_changed else "+rewrist_failed")
        t_now = ctx.tip(p.side)
        if wrist_changed or same_wrist:
            carry = ctrl.move_cartesian_path_both([{p.side: [float(t_now[0]), float(t_now[1]), z_carry]}, {p.side: [dx, dy, z_carry]}], {p.side: byaw}, pitches={p.side: bpitch}, speed_mps=CARRY_SPEED_MPS, symmetric=False)
            carry_plan += "+" + str(carry.get("plan", "path"))
        else:
            carry = {"completed": False, "aborted_reason": "carry wrist has no solution next to the lifted object"}
        if not carry["completed"]:
            # last resort: the joint-space traverse to the drop column at the carry height with the carry wrist. From the VERIFIED lift pose
            # (object bottom above the rim, both end poses at the carry height) it is what hand_v13 did from the grasp pose - safe here because
            # the object is already high; hand_v13 s100 swept the bottle through the rim only because its lift had silently failed.
            carry_plan += f"+joint_fallback ({carry.get('aborted_reason')})"
            over = ctx.call("move_to", side=p.side, xyz=[dx, dy, round(z_carry, 4)], yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=1, tolerance_m=0.03)
            if over.status != SkillStatus.success:
                back = ctrl.move_cartesian_path_both([{p.side: [lift_pt[0], lift_pt[1], z_lift]}], {p.side: lw_yaw}, pitches={p.side: lw_pitch}, speed_mps=CARRY_SPEED_MPS, symmetric=lw_sym)
                lap("carry_s")
                return self.result(SkillStatus.failure, f"could not carry the object over the opening: {over.message} (path: {carry.get('aborted_reason')}); the object was brought back above its pick point ({'ok' if back['completed'] else back.get('aborted_reason')}) and is still held - use the other arm or a lower grasp", verified=True, metrics={"stage": "carry", "refined": refined, "carry_z": round(z_carry, 4), "carry_plan": carry_plan, "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True})
        err_xy = float(np.hypot(*(ctx.tip(p.side)[:2] - [dx, dy])))
        align_moves = 0
        if err_xy > ALIGN_TOL_M / 2:
            ctx.call("move_to", side=p.side, xyz=[dx, dy, round(z_carry, 4)], yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=2, tolerance_m=0.03)
            align_moves = 1
            err_xy = float(np.hypot(*(ctx.tip(p.side)[:2] - [dx, dy])))
        lap("carry_s")
        held_over, _width_over = _held(ctx, p.side)
        if not held_over:
            return self.result(SkillStatus.failure, "object lost during the carry", verified=True, metrics={"stage": "carry", "refined": refined, "carry_z": round(z_carry, 4), "tip_error_xy_m": round(err_xy, 4), "times": times, "robot_time_s": _robot_time(ctx, tick0)})
        if err_xy > ALIGN_TOL_M:
            return self.result(SkillStatus.failure, f"the carry ended {err_xy * 100:.1f} cm off the drop point (limit {ALIGN_TOL_M * 100:.0f} cm) - not released; the object is still held: re-observe the container and call place_into_bin again", verified=True, metrics={"stage": "align", "refined": refined, "carry_z": round(z_carry, 4), "tip_error_xy_m": round(err_xy, 4), "carry_plan": carry_plan, "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True})

        # ---- lower the object into the opening (no correction / settle: a contact must stop, not push); a stall = contents there -> next spot
        def descend(x0, y0):
            d = ctrl.move_cartesian_path_both([{p.side: [x0, y0, z_release]}], {p.side: byaw}, pitches={p.side: bpitch}, speed_mps=0.3, symmetric=False, settle=False)
            t = ctx.tip(p.side)
            return d, float(t[2] - z_release), float(np.hypot(*(t[:2] - [x0, y0])))

        down, dz, xy_rel = descend(dx, dy)
        stalled = dz > STALL_DZ_M
        remaining = _remaining_robot_time_s(ctx)
        if stalled and (remaining is None or remaining > 4.0):
            alt = next((s for s in spots if (s[0], s[1]) != (dx, dy) and math.hypot(s[0] - dx, s[1] - dy) >= 0.04), None)
            if alt is not None:
                ctrl.move_cartesian_path_both([{p.side: [dx, dy, z_carry]}, {p.side: [alt[0], alt[1], z_carry]}], {p.side: byaw}, pitches={p.side: bpitch}, speed_mps=0.3, symmetric=False, settle=False)
                dx, dy = alt[0], alt[1]
                down, dz, xy_rel = descend(dx, dy)
                stalled = dz > STALL_DZ_M
        laydown = False
        if stalled and (remaining is None or remaining > 3.0) and abs(bpitch_deg) > LAYDOWN_PITCH_DEG + 5.0:
            # on top of the contents: lean the object toward the far wall before letting go (see LAYDOWN_PITCH_DEG)
            t = ctx.tip(p.side)
            lay_pitch = math.copysign(math.radians(LAYDOWN_PITCH_DEG), bpitch)
            lean = ctrl.move_to_report(p.side, [float(t[0]), float(t[1]), float(t[2]) + LAYDOWN_LIFT_M], byaw, pitch=lay_pitch, symmetric=False, correction_passes=0, fast=True)
            laydown = bool(lean.reachable)
        lap("descent_s")
        tip_rel = ctx.tip(p.side)
        bottom_below_rim = float(rim - (tip_rel[2] - hang))
        if not down["completed"] and dz > 0.08:
            ctx.call("move_to", side=p.side, xyz=[dx, dy, round(z_carry, 4)], yaw=byaw, pitch_deg=bpitch_deg, strict_yaw=True, correction_passes=0)
            return self.result(SkillStatus.failure, f"the descent into the opening did not happen ({down.get('aborted_reason')}); the object is still held above the rim", verified=True, metrics={"stage": "descent", "refined": refined, "carry_z": round(z_carry, 4), "tip_error_xy_m": round(xy_rel, 4), "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True})
        if bottom_below_rim < -RETAIN_ABOVE_RIM_M:
            # candidate 13dd11: the object bottom is still well above the rim (something blocks the opening there, or the container is not
            # where the descent went): opening now would drop it on the rim / outside. Lift back and keep it (formal_v13 s100 released
            # 2.4 cm above the rim of a toppled bin; hand_v13 regression s8: 2.3 cm above the rim onto another bottle's cap).
            ctrl.move_cartesian_path_both([{p.side: [dx, dy, z_carry]}], {p.side: byaw}, pitches={p.side: bpitch}, speed_mps=0.3, symmetric=False, settle=False)
            lap("retain_s")
            return self.result(SkillStatus.failure, f"the descent stopped with the object bottom {-bottom_below_rim * 100:.1f} cm ABOVE the rim at ({dx:.3f}, {dy:.3f}) ({'a stall: something is in the way there' if stalled else 'the release height was not reached'}); NOT released - the object is still held above the container: re-observe the container (segment_objects) and call place_into_bin again (another spot / the other arm)", verified=True, metrics={"stage": "retain", "refined": refined, "carry_z": round(z_carry, 4), "release_tip_z": round(float(tip_rel[2]), 4), "object_bottom_below_rim_m": round(bottom_below_rim, 4), "hang_m": round(hang, 4), "stalled": stalled, "stall_dz_m": round(dz, 4), "tip_error_xy_m": round(xy_rel, 4), "carry_plan": carry_plan, "times": times, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True, "drop_xy": [dx, dy], "rim_z": round(rim, 4), "opening_centre_xy": [round(cx, 4), round(cy, 4)]})
        # guided release: a small slow opening first so the object drops free of both pads before they move fast (a straight full
        # opening dragged the object sideways with the pad it stuck to), then the full opening
        width_held = ctx.gripper_width(p.side)
        guided = width_held is not None and float(width_held) + RELEASE_GUIDED_OPEN_M < 0.09
        if guided:
            ctx.call("set_gripper", side=p.side, opening_width_m=round(float(width_held) + RELEASE_GUIDED_OPEN_M, 4), duration=0.3)
            ctx.call("wait", seconds=0.2)
        ctx.call("set_gripper", side=p.side, aperture=1.0, duration=0.3)
        ctx.call("wait", seconds=0.1)
        width_open = ctx.gripper_width(p.side)
        released = width_open is None or width_before is None or float(width_open) >= min(float(width_before) + 0.02, 0.085)
        # the release history holds where the OBJECT was aimed (the free spot), not the forward-compensated fingertip point: with a 30 deg
        # re-pitch the two differ by 4.5 cm and the next drop's avoidance was off by that much (candidate 13dd11)
        ctx.scratch.setdefault("bin_drops", []).append((round(float(spot[0]), 4), round(float(spot[1]), 4)))
        lap("release_s")
        ret_pitch = math.copysign(math.radians(LAYDOWN_PITCH_DEG), bpitch) if laydown else bpitch
        ret = ctrl.move_cartesian_path_both([{p.side: [dx - 0.03, dy + own * RETREAT_SIDE_M, z_carry + 0.01]}], {p.side: byaw}, pitches={p.side: ret_pitch}, speed_mps=CARRY_SPEED_MPS, symmetric=False, settle=False)
        if not ret["completed"]:
            ctx.call("move_to", side=p.side, xyz=[round(dx - 0.03, 4), round(dy + own * RETREAT_SIDE_M, 4), round(z_carry + 0.01, 4)], yaw=byaw, pitch_deg=math.degrees(ret_pitch), strict_yaw=True, correction_passes=0, tolerance_m=0.05)
        lap("retreat_s")
        surface_after = _top_surface_z(ctx, (dx, dy), rim, 0.03)
        if surface_after is not None and surface_after > rim + hang + 0.10:
            surface_after = None  # the arm is still in the line of sight
        confirmed = None
        if surface_after is not None:
            if surface_after >= rim + 0.01:
                confirmed = True
            elif spot[2] is not None:
                confirmed = (surface_after - spot[2]) >= 0.02 or None
        metrics = {"stage": "release", "refined": refined, "carry_z": round(z_carry, 4), "release_tip_z": round(float(tip_rel[2]), 4), "object_bottom_below_rim_m": round(bottom_below_rim, 4), "hang_m": round(hang, 4), "forward_offset_m": round(forward, 4), "repitch_deg": round(repitch_deg, 1), "repitch_uncertainty_m": round(uncertainty, 4), "guided_release": guided, "laydown": laydown, "tip_error_xy_m": round(xy_rel, 4), "align_moves": align_moves, "stalled": stalled, "stall_dz_m": round(dz, 4), "released": released, "width_before_release_m": width_before, "width_after_open_m": width_open, "free_spot": {"xy": [dx, dy], "offset_from_centre_m": round(math.hypot(dx - cx, dy - cy), 4), "search_r_m": round(search_r, 4), "surface_z": spot[2], "n_history": len(history), "depth_points": 0 if pts is None else int(len(pts))}, "surface_before_z": None if surface_before is None else round(surface_before, 4), "surface_after_z": None if surface_after is None else round(surface_after, 4), "visual_confirmed": confirmed, "carry_plan": carry_plan, "wrist": {"yaw": round(byaw, 3), "pitch_deg": round(bpitch_deg, 1), "tilt_direction_xy": tilt_of(byaw, bpitch)}, "times": times, "robot_time_s": _robot_time(ctx, tick0)}
        outputs = {"release_pos": np.round(tip_rel, 4).tolist(), "drop_xy": [dx, dy], "object_bottom_below_rim_m": round(bottom_below_rim, 4), "rim_z": round(rim, 4), "opening_centre_xy": [round(cx, 4), round(cy, 4)]}
        ctx.scratch.get("tall_grasp", {}).pop(p.side, None)
        if not released:
            return self.result(SkillStatus.failure, f"the fingers did not open after the release command (opening {0 if width_open is None else round(float(width_open) * 100, 1)} cm): the object may still be clamped - re-observe", verified=True, metrics=metrics, outputs=outputs)
        if bottom_below_rim < -0.005:
            return self.result(SkillStatus.failure, f"released with the object bottom {-bottom_below_rim * 100:.1f} cm ABOVE the rim (the descent stopped early); it may have bounced out - re-observe the container", verified=True, metrics=metrics, outputs=outputs)
        how = "confirmed by the top depth camera" if confirmed else ("the top camera reads no change inside the opening - re-observe" if confirmed is False else "the top camera cannot judge")
        on = (" onto the contents, leaned toward the far wall" if laydown else " onto the contents") if stalled else ""
        refined_note = ""
        if refined is not None and (refined["rim_refined"] or refined["centre_refined"]):
            refined_note = f"; depth re-measured the container (rim {refined['rim_z_given']:.3f} -> {rim:.3f} m, centre moved {math.hypot(cx - refined['centre_given'][0], cy - refined['centre_given'][1]) * 100:.1f} cm)"
        return self.result(SkillStatus.success, f"released inside the opening{on} with the object bottom {bottom_below_rim * 100:.1f} cm below the rim at ({dx:.3f}, {dy:.3f}), {math.hypot(dx - cx, dy - cy) * 100:.1f} cm from the centre ({how}){refined_note}", verified=bool(confirmed), metrics=metrics, outputs=outputs)


# ------------------------------------------------------------------------------------------------------ composed
class PickAndPlaceTallObject(Skill):
    name = "pick_and_place_tall_object"
    version = "1.2.0"
    kind = "code"
    description = (
        "Bin one tall object (bottle, can): standing (top >= 10 cm) -> pick_tall_object (pitched wrist); LYING (top < 10 cm, depth-checked) -> "
        "top-down pick across its axis; then place_into_bin (carry over the rim, free spot inside; geometry from depth). side='auto' = arm by reach. "
        "Neighbours: obstacles_xy; nearest first."
    )
    returns = "success if released inside; metrics pick, place, route; failure names the stage (outputs.held = still held)"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "pick (calibrated opening) + place_into_bin (release FK, depth)"
    observations_used = ["proprio", "depth:top", "rgb:left_wrist", "rgb:right_wrist"]
    dependencies = ["pick_tall_object", "pick", "place_into_bin"]
    known_limitations = []
    timeout_s = 45.0

    class Params(Params):
        side: Literal["left", "right", "auto"] = Field(default="auto", description="auto = by reach")
        object_xyz: list[float] = Field(min_length=2, max_length=3, description="footprint centre [m]")
        object_top_z: float = Field(description="top z [m]")
        container_xyz: list[float] = Field(min_length=2, max_length=3, description="opening centre [m]")
        container_rim_z: float = Field(description="rim z [m]")
        object_yaw: float | None = Field(default=None, description="LYING: axis yaw [rad]")
        object_length_m: float | None = Field(default=None, ge=0.05, le=0.40, description="LYING: length [m]")
        object_half_span_m: float = Field(default=DEFAULT_OBJECT_RADIUS_M, ge=0.01, le=0.15, description="half width")
        grasp_height_m: float | None = Field(default=None, ge=0.02, le=0.20)
        pitch_deg: float = Field(default=DEFAULT_PITCH_DEG, ge=20.0, le=85.0)
        obstacles_xy: list[list[float]] = Field(default=[], description="[[x, y, radius_m, top_z]]")
        opening_radius_m: float = Field(default=DEFAULT_OPENING_RADIUS_M, ge=0.03, le=0.30, description="inner [m]")
        target_object: str = Field(default="the target object", max_length=120)
        check_posture: bool = True

    def _choose_side(self, ctx, p, lying: bool, top_z: float | None = None, grasp_height_m: float | None = None) -> tuple[str, dict]:
        """Arm by reach: own side (the object's side of the table) first; an arm qualifies when its pitched pick chain (or, for a
        lying object, its top-down grasp column) AND its carry over the container are reachable with joint margin. Falls back to
        the first arm whose pick is reachable, then to the own side (the pick reports the failure)."""

        table = float(ctx.table_z)
        x, y = float(p.object_xyz[0]), float(p.object_xyz[1])
        own = "left" if y >= 0.0 else "right"
        order = [own, "right" if own == "left" else "left"]
        top_z = float(p.object_top_z) if top_z is None else float(top_z)
        height = top_z - table
        checks = {}
        for side in order:
            if lying:
                grasp_z = table + max(0.5 * height, _pad_below_tip_m(ctx) + 0.004)
                yaw = (float(p.object_yaw) + math.pi / 2) if p.object_yaw is not None else 0.0
                pick_ok = ctx.controller.solve_ik(side, [x, y, grasp_z], yaw) is not None and ctx.controller.solve_ik(side, [x, y, grasp_z + 0.04], yaw) is not None
                hang = 0.5 * (float(p.object_length_m) if p.object_length_m is not None else 0.16)
            else:
                grasp_h = float(grasp_height_m) if grasp_height_m is not None else float(np.clip(DEFAULT_GRASP_FRACTION * height, 0.03, 0.12))
                obstacles = _obstacle_list(p.obstacles_xy, DEFAULT_OBSTACLE_RADIUS_M, top_z)
                options = plan_pitched_pick(ctx, side, [x, y, table + grasp_h], top_z, obstacles, float(p.pitch_deg))
                pick_ok = bool(_feasible(options))
                hang = grasp_h
            carry = carry_reachable(ctx, side, p.container_xyz, float(p.container_rim_z), hang, CARRY_CLEARANCE_M, float(p.pitch_deg))
            checks[side] = {"pick_reachable": pick_ok, "carry": carry}
            if pick_ok and carry is not None:
                return side, {"side": side, "reason": "own side reaches pick and carry" if side == own else "own side cannot; other arm reaches pick and carry", "checks": checks}
        for side in order:
            if checks[side]["pick_reachable"]:
                return side, {"side": side, "reason": "pick reachable, carry over the container not confirmed by IK", "checks": checks}
        return own, {"side": own, "reason": "no arm confirmed by IK; own side tries", "checks": checks}

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        table = float(ctx.table_z)
        top_z = float(p.object_top_z)
        height = top_z - table
        lying = height < LYING_MAX_TOP_M
        posture = None
        grasp_height_arg = p.grasp_height_m
        if lying and p.check_posture:
            # candidate f543ce: the caller's "lying" is checked against depth before the top-down route (a standing object read as lying is
            # knocked over by the descending open gripper); a confirmed standing object takes the pitched route with the measured top and the
            # default grasp height (the caller's grasp_height_m belongs to the lying geometry: 2.5 cm above the bottom of a 16 cm bottle cannot
            # be lowered below a rim the pads must stay above)
            posture = posture_from_depth(ctx, (float(p.object_xyz[0]), float(p.object_xyz[1])), table)
            if posture.get("standing"):
                lying = False
                top_z = float(posture["top_z"])
                height = top_z - table
                grasp_height_arg = None
        route = {"side": p.side, "reason": "given"}
        side = p.side
        if p.side == "auto":
            side, route = self._choose_side(ctx, p, lying, top_z, grasp_height_arg)
        half_span = float(p.object_half_span_m)
        if lying and "pick" in ctx.library:
            x, y = float(p.object_xyz[0]), float(p.object_xyz[1])
            length = float(p.object_length_m) if p.object_length_m is not None else 0.16
            yaw = (float(p.object_yaw) + math.pi / 2) if p.object_yaw is not None else 0.0  # close across the long axis
            grasp_z = table + max(0.5 * height, _pad_below_tip_m(ctx) + 0.004)
            pick = ctx.call("pick", side=side, xyz=[x, y, round(grasp_z, 4)], yaw=yaw, lift_height=0.04, object_top_z=float(p.object_top_z), target_object=p.target_object)
            pitch = math.radians(float(p.pitch_deg))
            hang = 0.5 * length * math.sin(pitch) + 0.5 * height * math.cos(pitch)  # far end of the re-pitched object below the grasp
            span = 0.5 * length * math.cos(pitch) + 0.5 * height * math.sin(pitch)  # the grasp is the horizontal centre of the tilted object
            place_kw = {"object_bottom_offset_m": round(float(min(hang, 0.30)), 4), "object_half_span_m": round(float(np.clip(span, 0.01, 0.15)), 4), "pitch_deg": float(p.pitch_deg), "yaw": yaw}
            mode = "lying"
            if pick.status != SkillStatus.success and pick.metrics.get("reachable") is False and p.object_yaw is not None:
                # beyond the top-down reach: pitched pinch along the object's axis from the robots' side (the object stays
                # horizontal in the carry; it is released across the opening)
                ax = np.array([math.cos(float(p.object_yaw)), math.sin(float(p.object_yaw))])
                if ax[0] < 0:
                    ax = -ax
                pick = ctx.call("pick_tall_object", side=side, object_xyz=p.object_xyz, object_top_z=float(p.object_top_z), grasp_height_m=round(max(0.5 * height, 0.025), 4), pitch_deg=p.pitch_deg, tilt_direction_xy=[float(ax[0]), float(ax[1])], obstacles_xy=p.obstacles_xy, target_object=p.target_object, test_lift=False, refine_from_depth=False)
                place_kw = {"object_bottom_offset_m": round(0.5 * height, 4), "object_half_span_m": round(float(np.clip(0.5 * length, 0.01, 0.15)), 4)}
                mode = "lying_pitched"
        else:
            pick = ctx.call("pick_tall_object", side=side, object_xyz=p.object_xyz, object_top_z=top_z, grasp_height_m=grasp_height_arg, pitch_deg=p.pitch_deg, obstacles_xy=p.obstacles_xy, target_object=p.target_object, test_lift=False)
            place_kw = {"object_bottom_offset_m": (pick.outputs or {}).get("grasp_height_above_bottom_m"), "object_half_span_m": round(half_span, 4)}
            mode = "standing" if posture is None or not posture.get("standing") else "standing (caller said lying; depth sees a top %.3f m)" % top_z
        route = {**route, "mode": mode, "posture": posture}
        if pick.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"pick failed ({side}, {mode}): {pick.message}", verified=True, metrics={"stage": "pick", "route": route, "pick": pick.metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"pick": pick.outputs, "held": False})
        place = ctx.call("place_into_bin", side=side, container_xyz=p.container_xyz, container_rim_z=p.container_rim_z, opening_radius_m=p.opening_radius_m, target_object=p.target_object, **place_kw)
        ok = place.status == SkillStatus.success
        ctx.scratch.setdefault("tall_routes", []).append({"side": side, "mode": mode, "ok": ok})
        return self.result(SkillStatus.success if ok else SkillStatus.failure, (f"binned ({mode}, {side} arm): {place.message}" if ok else f"place failed ({side}, {mode}): {place.message}"), verified=ok and place.effect_verified, metrics={"stage": "place", "route": route, "pick": {k: pick.metrics.get(k) for k in ("verification_status", "refined", "grasp_height_m", "clearance_m", "limit_margin_rad", "option", "approach", "times", "robot_time_s")}, "place": place.metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"pick": pick.outputs, "place": place.outputs, "held": bool((place.outputs or {}).get("held", False))})


def CANDIDATE_SMOKE(ctx) -> dict:
    """Offline smoke: the geometry helpers and the pitched IK over a raised opening."""

    pad = _pad_below_tip_m(ctx)
    yaw, pitch, axis = wrist_for_tilt([1.0, 0.0], 60.0)
    clear, who = approach_clearance([0.6, -0.4, 0.84], yaw, axis, [(0.66, -0.36, 0.035)], 0.92, finger_beyond_tip_m=pad)
    ok = ctx.controller.solve_ik("right", [0.65, 0.0, 1.0], yaw, pitch=pitch) is not None
    spots = free_spots((0.65, 0.0), 0.96, 0.03, None, [(0.65, -0.03)], prefer_xy=(0.65, 0.03))
    return {"yaw": yaw, "pitch": pitch, "clearance_m": clear, "blocker": who, "bin_reachable_pitched": ok, "first_free_spot": spots[0]}
