"""Rim grasps of wide / flared round objects, supported set-down with a settle check, and hole-over-peg insertion
(candidate cand_20260911_080228_cce58a, parent hand_v11@3.7.0; protocol v2 amendment 11: abc_rack, capx_nut_assembly).

Evidence (runs/analysis_v3/formal_v11, simulation measurements on seeds 0-4):
  * abc_rack - a body grasp of the flared bowl (pads on the outer wall, opening 7.5 cm) cannot be released by opening and
    retreating: the wall flares from 8.3 cm at the pads to 10.7 cm at the rim while the open fingers span 9.5 cm, so the rising
    fingers hook the rim from below and carry the bowl away (formal_v11 s102, GPT-6 diagnosis "bowl remained with the right
    arm after 'released'"; measured: released body-grasped bowls ended 12-16 cm above the table). A RIM grasp - one pad inside
    and one outside the wall at the rim, fingers closing radially - releases cleanly (measured: bowl back on the table at
    +1.9 cm, its resting height) because the inner finger rises inside the flare and the outer one is beyond the rim.
  * capx_nut_assembly - a nut held by its handle PIVOTS about the pinch axis when its ring lands on the peg tip instead of
    stalling the arm (the fingertip keeps reaching its z targets), so a plain descent cannot tell a seated nut from one
    resting tilted on the peg. The top depth camera can: at the hover the ring's top-surface points centre the hole to
    ~2 mm (measured 0.5-1.5 mm error at 0 / +10 / -10 mm offsets), and after the descent an annulus around the peg axis
    reads the ring top within 1 cm of the expected height when the peg is through the hole and >= 1.5 cm above it when the
    nut sits tilted on the tip. Insertion tolerance measured: 0 and 6 mm offsets seat, 10 mm does not (44 mm hole, 30 mm peg).

Skills (deployable observations only: proprioception / FK, the calibrated gripper reading, the top RGB-D camera):
  * ``pick_rim``        - grasp point on the rim nearest the arm base (or along a given direction), yaw radial, pads straddling
                          the wall PAD_DEPTH_BELOW_RIM_M below the rim top; executes the library's verified ``pick``; reports the
                          offsets the caller needs to set the object down: centre_offset_xy (tip -> object centre) and
                          object_bottom_offset_m (tip above the object bottom).
  * ``set_down``        - carry the held object over a landing point on a support surface, descend until touchdown (stall) or the
                          object bottom reaches the support, release, retreat out of the top camera's view and confirm from the
                          depth image that something rests there (settled, object_xy, top_z).
  * ``insert_onto_peg`` - carry a held object with a hole over a peg / post, re-centre the hole on the peg from the top depth
                          image, descend past the peg top, detect a tilted nut (depth annulus) and search around the estimate,
                          seat it on the support (touchdown), release, retreat, verify the seated height from depth.
Nothing here is task specific: radii, heights and offsets are parameters or measured.

Candidate cand_20260911_135918_55d735 (parent hand_v13@3.9.0): set_down 1.1.0 (raised narrow support: the long axis is measured from the top depth image and a rim-held object is turned so its hanging tilt lies along it; single slow joint-space carry with a full settle) and insert_onto_peg 1.1.0 (peg axis / top re-localised from depth before the carry; riding-height refinement only for a level ring with the finger links excluded; pause check on the low quartile of the ring top relative to the peg geometry; annulus outside the peg's corners; pre-release seat check - a jammed nut is lifted back and re-aligned, never released tilted; tightened post-release verification; optional ``grasp_yaw``). Evidence: formal_v13 abc_rack s102, capx_nut_assembly s100, hand_v13 regression abc_rack s2 / capx_nut_assembly s1 / s5. pick_rim unchanged.

Candidate cand_20260911_182411_f543ce (parent hand_v14@3.10.0): set_down 1.2.0 and pick_rim 1.1.0, from formal_v14 abc_rack s103 (GPT-6 truth
diagnosis runs/analysis_v3/formal_v14/abc_rack/abc_rack__s103, reproduced in simulation with the agent's calls):
  * the agent passed the rack's BASE PLATE height (table + 1.4 cm) as ``support_z`` for rails standing 5 cm above the table; 1.1.0 judged the
    support "not raised" (threshold table + 1.5 cm), skipped the rail alignment, and the rim-held bowl hanging tilted across the rails went down
    beside the far rail (landing 3.1 cm off, bowl 7.5 cm from the rack centre in y; the same call with the true rail height lands on the rails).
    The agents' support heights for that rack spread from table + 0.5 cm to + 4 cm (formal_v13 / v14, hand_v14 regression): set_down 1.2.0 re-measures
    the support's TOP from the top depth image around the landing point before the carry (highest populated 5 mm height band whose points lie
    under the landing point, within the acceptance window of the caller's value) and uses it for the raised-support test, the rail-axis band and
    the touchdown height (metrics.support_measured); the caller's value stays when nothing higher is seen.
  * the recovery ``pick_rim`` let ``pick`` re-acquire the bowl 7.7 cm away and grasp at the re-detected centroid, while pick_rim reported the offsets
    of the PLANNED rim point: set_down then landed 5.2 cm off and tipped the bowl. pick_rim 1.1.0 runs pick without its own re-acquisition; after
    a closing that meets nothing (or nothing that follows the gripper) it re-detects the object itself, re-plans the RIM point from the new
    centre (same direction) and picks once more, so centre_offset_xy / object_bottom_offset_m always describe the executed grasp
    (metrics.reacquisition).
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

_dir_hash = hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]
_base_name = f"physapi_libmod_{_dir_hash}_manipulation_base"
if _base_name in sys.modules:
    _base = sys.modules[_base_name]
else:
    _spec = importlib.util.spec_from_file_location(_base_name, Path(__file__).with_name("manipulation.py"))
    _base = importlib.util.module_from_spec(_spec)
    assert _spec.loader
    sys.modules[_base_name] = _base
    _spec.loader.exec_module(_base)
_cc_name = f"physapi_libmod_{_dir_hash}_contact_container"
if _cc_name in sys.modules:
    _cc = sys.modules[_cc_name]
else:
    _spec_cc = importlib.util.spec_from_file_location(_cc_name, Path(__file__).with_name("contact_container.py"))
    _cc = importlib.util.module_from_spec(_spec_cc)
    assert _spec_cc.loader
    sys.modules[_cc_name] = _cc
    _spec_cc.loader.exec_module(_cc)
_gv_name = f"physapi_libmod_{_dir_hash}_manipulation_gv"
if _gv_name in sys.modules:
    _gv = sys.modules[_gv_name]
else:
    _spec_gv = importlib.util.spec_from_file_location(_gv_name, Path(__file__).with_name("manipulation_gv.py"))
    _gv = importlib.util.module_from_spec(_spec_gv)
    assert _spec_gv.loader
    sys.modules[_gv_name] = _gv
    _spec_gv.loader.exec_module(_gv)
_held = _base._held
_reachable_height = _base._reachable_height
_robot_time = _base._robot_time
_holding = _cc._holding
_cartesian_path = _cc._cartesian_path
_settle_precise = _cc._settle_precise
_highest_reachable = _cc._highest_reachable
_carry = _cc._carry
_depth_points = _cc._depth_points
_line = _cc._line
_pad_below_tip_m = _cc._pad_below_tip_m

Side = Literal["left", "right"]
#: Rim grasp: the pad bottoms sit this far below the rim top (the pads straddle the wall; 2 cm keeps the pad column on the wall of a
#: shallow flared bowl while the finger links above stay clear of the rim).
PAD_DEPTH_BELOW_RIM_M = 0.02
#: Set-down: carry clearance of the object bottom above the support, descent overshoot past the nominal touchdown (stall decides).
SET_DOWN_CLEARANCE_M = 0.03
SET_DOWN_OVERSHOOT_M = 0.008
#: Insertion: hover clearance of the object bottom above the peg top, depth past the peg top that counts as "peg through the hole",
#: depth-image band for the ring top, annulus around the peg axis for the tilt test, tilt threshold.
INSERT_HOVER_CLEARANCE_M = 0.015
INSERT_PASS_DEPTH_M = 0.015
#: Height band around the ring's top surface for the ring points: wide enough for a nut that rides pitched in the pinch (formal
#: validation: a 7 mm band caught a partial ring - 275 points, half-size 1.95 cm - and mis-centred by 1.4 cm), still 1 cm clear of the
#: peg top at the hover / pause depths (the nut bottom is >= 1.5 cm below or above the peg top there).
RING_BAND_M = 0.02
RING_SEARCH_RADIUS_M = 0.038
RING_MIN_POINTS = 100
RING_MIN_HALF_SIZE_M = 0.015
#: insert_onto_peg 1.1.0: the annulus starts OUTSIDE the peg's corner radius (a 3 cm square peg reaches 2.1 cm from its axis: its top-face
#: corners inside the old 1.8 cm bound read as "ring 3.7 cm above the seat" and a seated nut looked jammed).
ANNULUS_R_M = (0.024, 0.034)
#: Seat check: a ring top (median) more than this above the seated height (support + thickness) is a nut jammed on the peg (a nut
#: through the hole but still pitched in the pinch reads ~0.5 cm; one resting on the peg tip reads the peg height, >= 2 cm).
TILT_ABOVE_EXPECTED_M = 0.015
#: Pause check: a nut that went over the peg has its ring top INSERT_PASS_DEPTH_M below the level of one resting on the peg tip; the
#: threshold sits halfway (1.1.0: the old 1.2 cm threshold let a nut resting on the tip - measured 1.1 cm above - pass, formal_v13 s100).
PAUSE_ABOVE_TOL_M = 0.5 * INSERT_PASS_DEPTH_M
#: insert_onto_peg 1.1.0: the ring-top height measured from depth replaces the nominal riding height only when the ring reads LEVEL (its
#: top band this narrow); a nut pitched in the pinch spans centimetres and its median top is not its bottom + thickness.
RING_LEVEL_MAX_SPREAD_M = 0.010
MAX_DEPTH_RECENTRE_M = 0.02
#: At the pause depth the ring centre measured from depth must lie within this of the peg axis, else the descent is retried on the
#: measured correction (a nut lowered BESIDE the peg is level and does not stall: height alone cannot tell it from a seated one).
#: insert_onto_peg 1.1.0: 5 mm (was 8): a square nut rotated 10-15 deg against a square peg keeps only ~3.5 mm of clearance per side
#: (formal_v13 capx_nut_assembly s100 jammed at a measured 6.2 mm lateral error).
PAUSE_LATERAL_TOL_M = 0.005
#: insert_onto_peg 1.1.0 riding-height measurement: the ring top of a handle-held object lies BELOW the fingertip (the handle is pinched
#: between pads that end at the tip) - at most this far above it for a nut riding pitched in the pinch; the finger blades / links stand
#: within FINGER_EXCLUDE_R_M of the fingertip and are excluded (formal_v13 capx_nut_assembly s100: finger points 0.8-2 cm ABOVE the tip
#: were read as the ring top, the nut was believed to ride 1.2 cm higher than it did, and a nut resting tilted on the peg passed the tilt
#: test and the seated-height check).
RING_ABOVE_TIP_MAX_M = 0.012
FINGER_EXCLUDE_R_M = 0.035
RIDE_HEIGHT_PLAUSIBLE_M = 0.02
#: Seat check before the release: a seat descent that stalls more than this above the seated tip height, or a ring top measured more than
#: TILT_ABOVE_EXPECTED_M above the seated height, is a nut jammed tilted on the peg - it is lifted back and re-aligned (at most SEAT_ROUNDS
#: rounds), never released tilted.
SEAT_STALL_TOL_M = 0.010
SEAT_ROUNDS = 2
#: Post-release verification band above the seated ring top (was 2 cm: a tilted nut passed) and the largest ring-top spread of a level nut.
SEATED_BAND_ABOVE_M = 0.010
SEATED_MAX_SPREAD_M = 0.025
#: set_down 1.1.0: a support this far above the table is RAISED (rack rails, a plate) and its long axis is measured from the top depth image;
#: an off-centre (rim) grasp is rotated so the held object's hanging tilt lies along that axis (formal_v13 abc_rack s102 + hand_v13
#: regression s2: the rim-held bowl hangs tilted ~20 deg toward its far side and toppled off the rails it crossed; oracle 10/10 along the axis).
RAISED_SUPPORT_MIN_M = 0.015
SUPPORT_AXIS_RADIUS_M = 0.14
SUPPORT_AXIS_BAND_M = 0.012
SUPPORT_AXIS_MIN_POINTS = 30
#: set_down 1.2.0: 1.3 (was 1.5). An unobstructed two-rail rack reads 17.3 x 6.5 cm from every landing point within 3.5 cm of its centre, but with
#: the other arm holding the object in the oblique camera's view one end of the rails can be cut off (hand_v14 / candidate regression abc_rack s4:
#: 11.6 x 8.2 cm = 1.41 -> "not narrow", no alignment, bowl slid off). The minor-extent cap below still excludes wide trays and plates, and aligning
#: a tilt along the long axis of a moderately elongated flat support is harmless.
SUPPORT_NARROW_RATIO = 1.3
SUPPORT_NARROW_MAX_MINOR_M = 0.12
SUPPORT_AXIS_MIN_MAJOR_M = 0.08
OFF_CENTRE_MIN_M = 0.02
GENTLE_CARRY_MPS = 0.15
#: set_down 1.2.0: the support's top surface is re-measured from the top depth image around the landing point when the caller gives a support
#: above the table (formal_v14 abc_rack s103: the agent passed the rack's base-plate height, table + 1.4 cm, for rails 5 cm above the table; 1.1.0
#: judged the support "not raised", skipped the rail alignment and the tilted rim-held bowl went down beside the far rail). Points within
#: SUPPORT_TOP_RADIUS_M of the landing point, above the table and below the caller's value + the acceptance window (the arm / held object are
#: higher), are binned in 5 mm height bands; the highest band with SUPPORT_TOP_MIN_POINTS points is the support top when those points' centroid
#: lies within SUPPORT_TOP_CENTRE_M of the landing point (a surface UNDER the object, not a neighbour) and the height is within
#: SUPPORT_TOP_ACCEPT_M (below, above) of the caller's value (callers under-read a base plate or over-read a rim pixel by centimetres, not more).
SUPPORT_TOP_RADIUS_M = 0.12
SUPPORT_TOP_BIN_M = 0.005
SUPPORT_TOP_MIN_POINTS = 25
SUPPORT_TOP_CENTRE_M = 0.06
SUPPORT_TOP_ACCEPT_M = (0.03, 0.06)
SUPPORT_TOP_MIN_ABOVE_TABLE_M = 0.004
SUPPORT_TOP_MIN_CHANGE_M = 0.005
#: pick_rim 1.1.0: after a re-grasp on a re-detected centre the held object's centre is measured from the top depth image after the lift - the
#: rim of a rim-held object hangs at about the fingertip height (tilted ~20 deg: +-2 cm across an 11 cm rim), so the points within
#: HELD_RIM_BAND_M of the fingertip height and within 2 x the rim radius + HELD_RIM_MARGIN_M of the tip (the fingers around the tip excluded)
#: are the rim and the upper inner wall; a Kasa circle fit with one outlier rejection gives the axis. Accepted for a plausible radius and a
#: centre within the rim radius + margin of the tip (the pads straddle the wall).
HELD_RIM_BAND_M = 0.03
HELD_RIM_MARGIN_M = 0.03
HELD_RIM_FINGER_EXCLUDE_M = 0.015
HELD_RIM_MIN_POINTS = 40
#: In-place wrist rotation rate and single-move carry rate for an object hanging off-centre in the pinch [rad/s, m/s].
ROTATE_RATE_RPS = 0.35
CARRY_RATE_MPS = 0.12
#: Offset search around the hole estimate when the nut rests tilted on the peg tip: ring 1 at this radius, ring 2 at twice it
#: (8 directions each; clearance 7 mm per side on the 44 / 30 mm nut, so 6 mm steps cannot skip the hole).
SEARCH_STEP_M = 0.006
#: Post-release lateral retreat (out of the oblique top camera's line of sight) [m].
VIEW_CLEAR_M = 0.14


def _own_side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def _clear_view(ctx, side: str, xy, z: float, yaw: float) -> None:
    """Retreat sideways from ``xy`` at height ``z`` so the arm leaves the top camera's line of sight over the point."""

    own = _own_side_sign(side)
    for d in ([0.0, own], [-0.6, 0.8 * own], [-1.0, 0.0], [0.0, -own]):
        target = np.asarray(xy, dtype=float) + np.asarray(d) * VIEW_CLEAR_M
        pts = [[float(q[0]), float(q[1]), z] for q in _line(xy, target, 0.04)]
        if all(ctx.controller.solve_ik(side, q, yaw) is not None for q in pts):
            _cartesian_path(ctx, side, pts, yaw, speed_mps=0.25)
            return


def _surface_blob(ctx, xy, ref_z: float, radius_m: float, above_m: float, z_cap: float | None = None) -> dict:
    """Depth points within ``radius_m`` of (xy, ref_z): the highest surface, and the centroid / top of the points that rise more than
    ``above_m`` above ref_z but no higher than ``z_cap`` (an object resting there; points above the cap are the arm or the wrist in
    the oblique view and are counted separately). None fields when nothing is visible."""

    pts = _depth_points(ctx, xy, ref_z, radius_m)
    out = {"surface_max": None, "object_xy": None, "object_top_z": None, "n_above": 0, "n_over_cap": 0}
    if pts is None or len(pts) == 0:
        return out
    out["surface_max"] = round(float(np.percentile(pts[:, 2], 98)), 4)
    band = pts[:, 2] >= ref_z + above_m
    if z_cap is not None:
        out["n_over_cap"] = int((pts[:, 2] > z_cap).sum())
        band &= pts[:, 2] <= z_cap
    above = pts[band]
    out["n_above"] = int(len(above))
    if len(above) >= 8:
        out["object_xy"] = [round(float(v), 4) for v in above[:, :2].mean(axis=0)]
        out["object_top_z"] = round(float(np.percentile(above[:, 2], 90)), 4)
    return out


def _support_axis(ctx, xy, support_z: float) -> dict | None:
    """Long axis of a raised support around ``xy`` from the top depth image: the points within SUPPORT_AXIS_BAND_M of ``support_z`` and
    SUPPORT_AXIS_RADIUS_M of the point (rail tops, a plate's surface), their principal axis and 2.5-97.5 % extents. ``narrow`` = an
    elongated support whose minor extent is smaller than a bowl (a rack of two rails, a slot), where a hanging object's tilt direction
    decides whether it lands on both rails. None when the camera sees nothing there."""

    pts = _depth_points(ctx, xy, support_z, SUPPORT_AXIS_RADIUS_M)
    if pts is None or len(pts) == 0:
        return None
    d = np.linalg.norm(pts[:, :2] - np.asarray(xy, dtype=float), axis=1)
    sel = pts[(d <= SUPPORT_AXIS_RADIUS_M) & (np.abs(pts[:, 2] - float(support_z)) <= SUPPORT_AXIS_BAND_M)]
    if len(sel) < SUPPORT_AXIS_MIN_POINTS:
        return {"n": int(len(sel)), "narrow": False}
    c = sel[:, :2].mean(axis=0)
    rel = sel[:, :2] - c
    w, v = np.linalg.eigh(np.cov(rel.T))
    major, minor = v[:, 1], v[:, 0]
    # set_down 1.2.0: a two-rail support seen partly occluded (the other arm, holding the object, stands between the oblique camera and one end
    # of a rail) has a whole-cloud PCA axis rotated toward the visible diagonal (candidate regression abc_rack s4: 32 deg off, "not narrow", no
    # alignment). Each rail's own points still lie on a straight line: split the cloud across the minor axis, take each half's principal
    # direction and re-estimate the axis from the two (weighted by their spread), re-splitting on the refined minor axis a few times.
    split = {"applied": False}
    for _ in range(3):
        pn0 = rel @ minor
        halves = [rel[pn0 >= 0], rel[pn0 < 0]]
        dirs = []
        for g in halves:
            if len(g) >= 15:
                gw, gv = np.linalg.eigh(np.cov((g - g.mean(axis=0)).T))
                d = gv[:, 1] if float(gv[:, 1] @ major) >= 0 else -gv[:, 1]
                dirs.append((d, float(np.sqrt(max(float(gw[1]), 0.0)))))
        if len(dirs) != 2 or abs(float(dirs[0][0] @ dirs[1][0])) < math.cos(math.radians(20.0)):
            break
        d = dirs[0][0] * dirs[0][1] + dirs[1][0] * dirs[1][1]
        nd = float(np.linalg.norm(d))
        if nd < 1e-9:
            break
        new_major = d / nd
        turned = math.degrees(math.acos(min(1.0, abs(float(new_major @ major)))))
        major = new_major
        minor = np.array([-major[1], major[0]])
        split = {"applied": True, "rail_dirs_deg_apart": round(math.degrees(math.acos(min(1.0, abs(float(dirs[0][0] @ dirs[1][0]))))), 1), "turned_deg": round(turned, 1)}
        if turned < 0.5:
            break
    pm, pn = rel @ major, rel @ minor
    ext_major = float(np.percentile(pm, 97.5) - np.percentile(pm, 2.5))
    ext_minor = float(np.percentile(pn, 97.5) - np.percentile(pn, 2.5))
    narrow = ext_major >= SUPPORT_AXIS_MIN_MAJOR_M and ext_major >= SUPPORT_NARROW_RATIO * max(ext_minor, 1e-3) and ext_minor <= SUPPORT_NARROW_MAX_MINOR_M
    return {"n": int(len(sel)), "centre": [round(float(q), 4) for q in c], "axis": [round(float(q), 4) for q in major], "yaw": round(math.atan2(float(major[1]), float(major[0])), 3), "extent_major_m": round(ext_major, 3), "extent_minor_m": round(ext_minor, 3), "narrow": bool(narrow), "split": split}


def _support_top_from_depth(ctx, xy, table_z: float, given_z: float) -> dict:
    """set_down 1.2.0: the top surface of a raised support under the landing point ``xy`` from the top depth image (constants above).

    Returns {"given", "measured", "used", "refined", "n", "n_top", "top_centre_offset_m", "reason"}: ``used`` is the measured top when it
    is accepted and differs from the caller's value by >= SUPPORT_TOP_MIN_CHANGE_M (``refined`` True), else the caller's value."""

    given = float(given_z)
    out = {"given": round(given, 4), "measured": None, "used": round(given, 4), "refined": False, "n": 0, "n_top": 0, "top_centre_offset_m": None, "reason": None}
    pts = _depth_points(ctx, xy, given, SUPPORT_TOP_RADIUS_M)
    if pts is None or len(pts) == 0:
        out["reason"] = "no depth"
        return out
    c0 = np.asarray(xy, dtype=float)
    d = np.linalg.norm(pts[:, :2] - c0, axis=1)
    sel = pts[(d <= SUPPORT_TOP_RADIUS_M) & (pts[:, 2] >= float(table_z) + SUPPORT_TOP_MIN_ABOVE_TABLE_M) & (pts[:, 2] <= given + SUPPORT_TOP_ACCEPT_M[1] + 0.01)]
    out["n"] = int(len(sel))
    if len(sel) < SUPPORT_TOP_MIN_POINTS:
        out["reason"] = "too few raised points"
        return out
    z = sel[:, 2]
    edges = np.arange(float(table_z), float(z.max()) + SUPPORT_TOP_BIN_M + 1e-9, SUPPORT_TOP_BIN_M)
    counts, _ = np.histogram(z, bins=edges)
    good = np.nonzero(counts >= SUPPORT_TOP_MIN_POINTS)[0]
    if len(good) == 0:
        out["reason"] = "no populated height band"
        return out
    k = int(good[-1])  # the highest populated band: the top of the support (lower bands are its sides / base seen obliquely)
    top_pts = sel[(z >= edges[k] - SUPPORT_TOP_BIN_M) & (z <= edges[k + 1] + SUPPORT_TOP_BIN_M)]
    top = float(np.percentile(top_pts[:, 2], 80))
    offset = float(np.linalg.norm(top_pts[:, :2].mean(axis=0) - c0))
    out.update({"measured": round(top, 4), "n_top": int(len(top_pts)), "top_centre_offset_m": round(offset, 4), "top_centre_xy": [round(float(v), 4) for v in top_pts[:, :2].mean(axis=0)]})
    if offset > SUPPORT_TOP_CENTRE_M:
        out["reason"] = "highest surface lies beside the landing point"
        return out
    if not (given - SUPPORT_TOP_ACCEPT_M[0] <= top <= given + SUPPORT_TOP_ACCEPT_M[1]):
        out["reason"] = "outside the acceptance window of the caller's value"
        return out
    if abs(top - given) < SUPPORT_TOP_MIN_CHANGE_M:
        out["reason"] = "agrees with the caller"
        return out
    out.update({"used": round(top, 4), "refined": True})
    return out


def _kasa_circle(xy: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic least-squares circle through points (N x 2): (cx, cy, r) or None."""

    if len(xy) < 6:
        return None
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, -(x * x + y * y), rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = (float(v) for v in sol)
    cx, cy = -a / 2.0, -b / 2.0
    r2 = cx * cx + cy * cy - c
    return None if r2 <= 0 else (cx, cy, math.sqrt(r2))


def _held_rim_centre_from_depth(ctx, side: str, rim_radius_m: float) -> dict:
    """pick_rim 1.1.0: the axis of the rim-held object hanging under the ``side`` fingertip, from the top depth image (constants above).

    Returns {"n", "centre_xy", "radius_m", "offset_xy" (tip -> centre), "accepted", "reason"}."""

    tip = ctx.tip(side)
    r = float(rim_radius_m)
    out = {"n": 0, "centre_xy": None, "radius_m": None, "offset_xy": None, "accepted": False, "reason": None}
    pts = _depth_points(ctx, tip[:2], float(tip[2]), 2.0 * r + HELD_RIM_MARGIN_M + 0.02)
    if pts is None or len(pts) == 0:
        out["reason"] = "no depth"
        return out
    d = np.linalg.norm(pts[:, :2] - tip[:2], axis=1)
    sel = pts[(d <= 2.0 * r + HELD_RIM_MARGIN_M) & (d >= HELD_RIM_FINGER_EXCLUDE_M) & (np.abs(pts[:, 2] - float(tip[2])) <= HELD_RIM_BAND_M)]
    out["n"] = int(len(sel))
    if len(sel) < HELD_RIM_MIN_POINTS:
        out["reason"] = "too few rim points"
        return out
    xy = sel[:, :2]
    fit = _kasa_circle(xy)
    if fit is not None:  # one outlier rejection: keep the points within 1.5 cm of the fitted circle (the deeper inner wall, the fingers)
        rad = np.hypot(xy[:, 0] - fit[0], xy[:, 1] - fit[1])
        keep = np.abs(rad - fit[2]) <= 0.015
        if int(keep.sum()) >= HELD_RIM_MIN_POINTS:
            fit = _kasa_circle(xy[keep]) or fit
    if fit is None:
        out["reason"] = "no circle fit"
        return out
    cx, cy, rr = fit
    off = np.array([cx - float(tip[0]), cy - float(tip[1])])
    out.update({"centre_xy": [round(cx, 4), round(cy, 4)], "radius_m": round(rr, 4), "offset_xy": [round(float(v), 4) for v in off]})
    if not (0.7 * r <= rr <= 1.3 * r):
        out["reason"] = "implausible radius"
        return out
    if float(np.linalg.norm(off)) > r + HELD_RIM_MARGIN_M:
        out["reason"] = "centre too far from the fingertip for a rim grasp"
        return out
    out["accepted"] = True
    return out


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


#: insert_onto_peg 1.1.0 peg re-localisation: search radius around the caller's peg estimate, height band of the peg's top face below /
#: above the caller's peg_top_z, minimum points, and the largest accepted shift of the estimate.
PEG_RELOCALISE_R_M = 0.045
PEG_TOP_BAND_M = (0.02, 0.015)
PEG_RELOCALISE_MIN_POINTS = 15
PEG_RELOCALISE_MAX_SHIFT_M = 0.03
#: The top face must be seen COMPLETELY for its centroid to be the axis: at least this many face points and this extent along both axes
#: (measured on the 3 cm peg over 15 seeds: a fully visible face gives 155-197 points, 2.9-3.3 cm x 2.7-3.0 cm and a centroid within 1.3 mm
#: of the axis; the arm holding the nut occluded part of it in 6 of 15 poses - 16-70 points, 0.2-1.1 cm along one axis, centroid 0.9-1.4 cm off).
PEG_FACE_MIN_POINTS = 100
PEG_FACE_MIN_EXTENT_M = 0.02


def _peg_top_from_depth(ctx, peg_xy, peg_top: float) -> dict | None:
    """The peg's top face in the top depth image: points within PEG_RELOCALISE_R_M of the estimate whose height lies within PEG_TOP_BAND_M of
    ``peg_top`` (the top face and the uppermost rim of the side faces; the plate / table far below and a held object far above are
    outside the band). Returns {centre (xy centroid), top_z (p80), n, extent_m, complete, shift_m} or None when too few points are seen;
    ``complete`` says whether the face was seen whole (else the centroid is biased toward the visible part and must not be used)."""

    pts = _depth_points(ctx, peg_xy, float(peg_top), PEG_RELOCALISE_R_M)
    if pts is None or len(pts) == 0:
        return None
    d = np.linalg.norm(pts[:, :2] - np.asarray(peg_xy, dtype=float), axis=1)
    sel = pts[(d <= PEG_RELOCALISE_R_M) & (pts[:, 2] >= float(peg_top) - PEG_TOP_BAND_M[0]) & (pts[:, 2] <= float(peg_top) + PEG_TOP_BAND_M[1])]
    if len(sel) < PEG_RELOCALISE_MIN_POINTS:
        return None
    top = float(np.percentile(sel[:, 2], 80))
    face = sel[sel[:, 2] >= top - 0.006]  # the top face itself (the side faces seen obliquely lie below it)
    if len(face) < PEG_RELOCALISE_MIN_POINTS:
        face = sel
    centre = face[:, :2].mean(axis=0)
    ext = [float(np.percentile(face[:, k], 97.5) - np.percentile(face[:, k], 2.5)) for k in (0, 1)]
    complete = len(face) >= PEG_FACE_MIN_POINTS and min(ext) >= PEG_FACE_MIN_EXTENT_M
    return {"centre": [round(float(v), 4) for v in centre], "top_z": round(top, 4), "n": int(len(face)), "extent_m": [round(v, 4) for v in ext], "complete": bool(complete), "shift_m": round(float(np.linalg.norm(centre - np.asarray(peg_xy, dtype=float))), 4)}


def _ring_top_from_depth(ctx, hole_est, z_lo: float, z_hi: float, *, radius_m: float = 0.048, bin_m: float = 0.005, min_points: int = 30, exclude_xy=None, exclude_r: float = 0.0) -> float | None:
    """Height of the ring's top surface: the most populated 5 mm height bin of the depth points within ``radius_m`` of the estimate
    between ``z_lo`` and ``z_hi`` (the highest such bin when several are populated). A handle-held object rides in the pinch: the nut
    measured 1.4 cm higher relative to the fingertip after the carry than at the grasp, which put the nominal band beside the ring.
    ``exclude_xy`` / ``exclude_r`` drop the points around the fingertip (the finger blades and links are not the ring)."""

    pts = _depth_points(ctx, hole_est, 0.5 * (z_lo + z_hi), radius_m + 0.02)
    if pts is None or len(pts) == 0:
        return None
    d = np.linalg.norm(pts[:, :2] - np.asarray(hole_est, dtype=float), axis=1)
    keep = (d <= radius_m) & (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
    if exclude_xy is not None and exclude_r > 0:
        keep &= np.linalg.norm(pts[:, :2] - np.asarray(exclude_xy, dtype=float), axis=1) > float(exclude_r)
    z = pts[keep, 2]
    if len(z) < min_points:
        return None
    edges = np.arange(z_lo, z_hi + bin_m, bin_m)
    counts, _ = np.histogram(z, bins=edges)
    good = np.nonzero(counts >= min_points)[0]
    if len(good) == 0:
        return None
    k = int(good[-1])  # the highest populated bin: the ring top (lower bins are the peg / support seen through the hole)
    sel = z[(z >= edges[k] - bin_m) & (z <= edges[k + 1] + bin_m)]
    return float(np.median(sel))


def _ring_top_cluster(ctx, hole_est, z_lo: float, z_hi: float, *, radius_m: float = 0.048, bin_m: float = 0.005, min_points: int = 30, exclude_xy=None, exclude_r: float = 0.0) -> dict | None:
    """The ring-top cluster of depth points (:func:`_ring_top_from_depth`'s selection widened to the populated bins adjoining the highest
    one): {top (median), spread (p90 - p10), n}. ``spread`` tells a level ring (< 1 cm) from one pitched in the pinch (centimetres)."""

    pts = _depth_points(ctx, hole_est, 0.5 * (z_lo + z_hi), radius_m + 0.02)
    if pts is None or len(pts) == 0:
        return None
    d = np.linalg.norm(pts[:, :2] - np.asarray(hole_est, dtype=float), axis=1)
    keep = (d <= radius_m) & (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
    if exclude_xy is not None and exclude_r > 0:
        keep &= np.linalg.norm(pts[:, :2] - np.asarray(exclude_xy, dtype=float), axis=1) > float(exclude_r)
    z = pts[keep, 2]
    if len(z) < min_points:
        return None
    edges = np.arange(z_lo, z_hi + bin_m, bin_m)
    counts, _ = np.histogram(z, bins=edges)
    good = np.nonzero(counts >= min_points)[0]
    if len(good) == 0:
        return None
    k = int(good[-1])
    j = k
    while j - 1 >= 0 and counts[j - 1] >= min_points:  # walk down the contiguous populated bins: the whole ring top, level or pitched
        j -= 1
    sel = z[(z >= edges[j] - bin_m) & (z <= edges[k + 1] + bin_m)]
    return {"top": float(np.median(sel)), "spread": float(np.percentile(sel, 90) - np.percentile(sel, 10)), "n": int(len(sel))}


def _ring_centre_from_depth(ctx, hole_est, top_z: float, toward_handle, *, radius_m: float = 0.048, band_m: float = RING_BAND_M) -> dict | None:
    """Centre of a square / round ring seen from above in the top depth image, robust to the handle that extends from one side:
    ring-top points (within ``band_m`` of ``top_z``, within ``radius_m`` of the estimate); the axis perpendicular to the handle
    gives the centre and the half-size from its 2-98 % extent (the handle is narrower than the ring, so it does not widen it);
    along the handle axis the centre is the clean far edge (away from the handle) plus that half-size. Measured on the 68 mm nut:
    0.3-3 mm error for estimates 10-15 mm off, where the plain centroid was 5-7 mm off (biased toward the handle)."""

    c0 = np.asarray(hole_est, dtype=float)
    u = np.asarray(toward_handle, dtype=float)
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return None
    u = u / n
    v = np.array([-u[1], u[0]])
    pts = _depth_points(ctx, c0, top_z, radius_m + 0.02)
    if pts is None or len(pts) == 0:
        return None
    d = np.linalg.norm(pts[:, :2] - c0, axis=1)
    sel = pts[(d <= radius_m) & (np.abs(pts[:, 2] - top_z) <= band_m)]
    if len(sel) < RING_MIN_POINTS:
        return None
    rel = sel[:, :2] - c0
    pu, pv = rel @ u, rel @ v
    v_lo, v_hi = float(np.percentile(pv, 2)), float(np.percentile(pv, 98))
    half = 0.5 * (v_hi - v_lo)
    if half < RING_MIN_HALF_SIZE_M:
        return None  # a partial ring (occluded / cut by the band): not a usable measurement
    centre = c0 + (float(np.percentile(pu, 2)) + half) * u + 0.5 * (v_lo + v_hi) * v
    return {"centre": centre, "n": int(len(sel)), "half_size_m": round(half, 4)}


class PickRim(Skill):
    name = "pick_rim"
    version = "1.1.0"
    kind = "code"
    description = "Grasp a wide / flared round object (bowl, plate) by its RIM: grasp point on the rim nearest the arm base (or along `grasp_direction_xy`), radial pinch across the wall, pads 2 cm below the rim. Use instead of pick for anything wider than ~8 cm (a body grasp hooks the flared rim on release). A rim-held bowl hangs tilted ~20 deg (far side down): on a NARROW support (rack) grasp along its long axis. Runs the verified pick; after a miss the rim point is re-planned once from a fresh detection."
    returns = "pick's result + outputs.grasp_xyz, yaw, side, centre_offset_xy (tip -> object centre) and object_bottom_offset_m (tip above the object bottom) for set_down; failure: no reachable rim point, or pick's failure"
    preconditions = ["object on the table; centre_xy / rim_top_z / rim_radius_m from segment_objects (footprint centre, top_z, size / 2)"]
    effects = []
    failure_conditions = []
    effect_verification = "pick's calibrated opening after close / test lift / lift (+ verify_grasp)"
    observations_used = ["proprio"]
    dependencies = ["pick"]
    known_limitations = []
    timeout_s = 30.0

    class Params(Params):
        side: Literal["left", "right", "auto"] = "auto"
        centre_xy: list[float] = Field(min_length=2, max_length=2, description="object centre [m]")
        rim_top_z: float = Field(description="rim top [m]")
        rim_radius_m: float = Field(ge=0.03, le=0.25, description="outer radius [m]")
        wall_thickness_m: float = Field(default=0.008, ge=0.002, le=0.03)
        grasp_direction_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="centre -> rim point")
        pad_depth_m: float = Field(default=PAD_DEPTH_BELOW_RIM_M, ge=0.005, le=0.05, description="pads below the rim top [m]")
        lift_height: float = Field(default=0.10, ge=0.03, le=0.25)
        target_object: str = Field(default="the bowl", max_length=120)
        regrasp_on_miss: bool = True

    def _rim_point(self, ctx, side: str, p, centre_xy=None) -> tuple[np.ndarray, float, np.ndarray]:
        c = np.asarray(p.centre_xy if centre_xy is None else centre_xy, dtype=float)
        if p.grasp_direction_xy is not None:
            u = np.asarray(p.grasp_direction_xy, dtype=float)
        else:
            u = ctx.controller.base_position(side)[:2] - c
        n = float(np.linalg.norm(u))
        if n < 1e-6:
            raise SkillError("grasp direction undefined (object centred on the arm base?)", error_type="invalid_params")
        u = u / n
        r = float(p.rim_radius_m) - 0.5 * float(p.wall_thickness_m)  # the wall's centre line
        tip_z = float(p.rim_top_z) - float(p.pad_depth_m) + float(ctx.cell_calibration.finger_pad_below_tip_m)
        gp = np.array([c[0] + u[0] * r, c[1] + u[1] * r, tip_z])
        yaw = math.atan2(u[1], u[0])  # fingers close along the radial direction: one pad inside, one outside the wall
        return gp, yaw, u

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        if p.rim_top_z < ctx.table_z + 0.01:
            raise SkillError(f"rim_top_z {p.rim_top_z:.3f} is at table level", error_type="precondition")
        sides = ["left", "right"] if p.side == "auto" else [p.side]
        best = None
        tried = {}
        for s in sides:
            gp, yaw, u = self._rim_point(ctx, s, p)
            sol = ctx.controller.solve_ik_detailed(s, gp, yaw)
            col = _reachable_height(ctx, s, *gp, 0.06, yaw)
            tried[s] = {"grasp_xyz": np.round(gp, 4).tolist(), "yaw": round(yaw, 3), "reachable": sol is not None and col is not None, "margin_rad": None if sol is None else round(float(sol.limit_margin_rad), 3)}
            if sol is not None and col is not None and (best is None or sol.limit_margin_rad > best[3]):
                best = (s, gp, yaw, float(sol.limit_margin_rad), u)
        if best is None:
            return self.result(SkillStatus.failure, f"no arm reaches a rim point of {p.target_object} top-down ({tried}); bring it closer (pull_toward_base) or use the other side", verified=True, metrics={"stage": "reach", "reachable": False, "candidates": tried})
        side, gp, yaw, margin, u = best
        centre = np.asarray(p.centre_xy, dtype=float)
        # 1.1.0: pick runs WITHOUT its own re-acquisition (retry_on_not_held=False keeps its test-lift rescue of a false not_held verdict but
        # not the blind re-grasp at a re-detected centroid, which left the offsets below describing a grasp that never happened)
        r = ctx.call("pick", side=side, xyz=gp.tolist(), yaw=float(yaw), lift_height=float(p.lift_height), object_top_z=float(p.rim_top_z), approach_height=0.06, target_object=p.target_object, retry_on_not_held=False)
        reacquisition = None
        if r.status != SkillStatus.success and p.regrasp_on_miss and r.metrics.get("reason") in ("grasp_not_confirmed", "reacquisition_failed"):
            # the closing met nothing (or nothing that follows the gripper): the CENTRE estimate is off. Re-detect the object with the arm out of the
            # camera's line of sight, re-plan the RIM point from the new centre (same direction) and pick once more (formal_v14 abc_rack s103)
            obs = _gv._observation_pose(ctx, side, float(gp[0]), float(gp[1]), float(gp[2]), 0.06)
            if obs is not None:
                ctx.call("move_to", side=side, xyz=obs, yaw=float(yaw), correction_passes=0)
            # the colour fallback of the re-detection projects the blob at 2 x (z - table): pass the object's mid-height so it lands on the rim height
            z_mid = float(ctx.table_z) + 0.5 * (float(p.rim_top_z) - float(ctx.table_z))
            found, source = _gv._reacquire(ctx, p.target_object, float(centre[0]), float(centre[1]), z_mid)
            reacquisition = {"source": source, "from_centre": np.round(centre, 4).tolist(), "to_centre": None if found is None else [round(found[0], 4), round(found[1], 4)], "retried": False}
            if found is not None and math.hypot(found[0] - centre[0], found[1] - centre[1]) >= 0.005:
                gp2, yaw2, u2 = self._rim_point(ctx, side, p, centre_xy=found)
                sol2 = ctx.controller.solve_ik_detailed(side, gp2, yaw2)
                col2 = _reachable_height(ctx, side, *gp2, 0.06, yaw2)
                reacquisition.update({"shift_m": round(float(math.hypot(found[0] - centre[0], found[1] - centre[1])), 4), "rim_point": np.round(gp2, 4).tolist(), "reachable": sol2 is not None and col2 is not None})
                if sol2 is not None and col2 is not None:
                    reacquisition["retried"] = True
                    centre, gp, yaw, u, margin = np.asarray(found, dtype=float), gp2, yaw2, u2, float(sol2.limit_margin_rad)
                    r = ctx.call("pick", side=side, xyz=gp.tolist(), yaw=float(yaw), lift_height=float(p.lift_height), object_top_z=float(p.rim_top_z), approach_height=0.06, target_object=p.target_object, retry_on_not_held=False)
        tip = ctx.tip(side)
        centre_offset = (centre - gp[:2]).tolist()
        held_depth = None
        if r.status == SkillStatus.success and reacquisition is not None and reacquisition.get("retried"):
            # the re-detected centre came from a fresh (possibly biased) detection: measure the held object's axis under the fingertip from depth
            # and report the offsets of the object actually in the hand
            held_depth = _held_rim_centre_from_depth(ctx, side, float(p.rim_radius_m))
            if held_depth.get("accepted"):
                centre = np.asarray(held_depth["centre_xy"], dtype=float)
                centre_offset = (centre - tip[:2]).tolist()
                reacquisition["offsets_from"] = "depth (held object)"
            else:
                reacquisition["offsets_from"] = "re-detected centre"
        bottom_off = float(gp[2] - ctx.table_z)
        metrics = {**r.metrics, "stage": r.metrics.get("stage"), "side": side, "grasp_xyz": np.round(gp, 4).tolist(), "yaw": round(float(yaw), 4), "limit_margin_rad": round(margin, 3), "candidates": tried, "reacquisition": reacquisition, "held_centre_depth": held_depth, "robot_time_s": _robot_time(ctx, tick0)}
        outputs = {**(r.outputs or {}), "side": side, "grasp_xyz": np.round(gp, 4).tolist(), "yaw": round(float(yaw), 4), "centre_xy": np.round(centre, 4).tolist(), "centre_offset_xy": [round(float(v), 4) for v in centre_offset], "object_bottom_offset_m": round(bottom_off, 4), "tip_pos": np.round(tip, 4).tolist()}
        if r.status != SkillStatus.success:
            return self.result(r.status, f"rim grasp ({side}, yaw {yaw:.2f}) failed: {r.message}", verified=r.effect_verified, metrics=metrics, outputs=outputs)
        regrasped = f" (re-detected {p.target_object} {reacquisition['shift_m'] * 100:.1f} cm from the given centre and re-planned the rim point)" if reacquisition and reacquisition.get("retried") else ""
        msg = (f"{p.target_object} held by the rim ({side} arm, opening {round((r.outputs or {}).get('opening_width_m', 0) * 100, 1)} cm): {r.message}{regrasped}. "
               f"To set it down use set_down(side='{side}', object_xy=<landing centre>, support_z=<surface height>, centre_offset_xy={[round(float(v), 3) for v in centre_offset]}, object_bottom_offset_m={bottom_off:.3f}, yaw={yaw:.3f})")
        return self.result(SkillStatus.success, msg, verified=r.effect_verified, metrics=metrics, outputs=outputs)


class SetDown(Skill):
    name = "set_down"
    version = "1.2.0"
    kind = "code"
    description = "Set the held object down onto a support (table, rack, tray) with its centre at `object_xy`: carry over the point (bottom 3 cm above), descend to touchdown (stall) or `support_z`, release, retreat, confirm from depth that it rests there (metrics.settled, landing_offset_m). Offsets from pick_rim: centre_offset_xy (tip -> object centre), object_bottom_offset_m (tip above the bottom); centred pick: [0, 0], grasp height. A raised support's TOP (rails above a base plate) is re-measured from depth (metrics.support_measured); on a RAISED narrow support a rim-held object is turned so its tilt lies along the rails (metrics.alignment)."
    returns = "success if released and a resting object is seen at the point (or the camera cannot judge); failure: nothing held, unreachable, fingers wedged, object missing / carried away on the fingers"
    preconditions = ["the arm holds the object"]
    effects = []
    failure_conditions = []
    effect_verification = "touchdown stall by FK; calibrated opening after release; top depth at the landing point after the retreat"
    observations_used = ["proprio", "depth:top"]
    dependencies = ["set_gripper", "wait", "move_to"]
    known_limitations = []
    timeout_s = 30.0

    class Params(Params):
        side: Side
        object_xy: list[float] = Field(min_length=2, max_length=2, description="landing point of the object centre [m]")
        support_z: float | None = Field(default=None, description="support height [m]; default table")
        centre_offset_xy: list[float] = Field(default=[0.0, 0.0], min_length=2, max_length=2, description="tip -> object centre [m]")
        object_bottom_offset_m: float = Field(ge=0.0, le=0.30, description="tip above the object bottom [m]")
        yaw: float = Field(default=0.0, description="wrist yaw to keep [rad]")
        settle_check: bool = True
        target_object: str = Field(default="the held object", max_length=120)
        align_to_support: bool = Field(default=True, description="turn a rim-held object's tilt along a narrow support")

    def _align(self, ctx, p, tip_xy, yaw: float, support: float, z_carry: float):
        """set_down 1.1.0: on a raised narrow support turn the wrist so the off-centre held object's tilt (along the tip -> centre line) lies
        along the support's long axis; rotates in place (slowly) before the carry. Returns (tip_xy, yaw, record)."""

        centre_off = np.asarray(p.centre_offset_xy, dtype=float)
        n = float(np.linalg.norm(centre_off))
        rec = {"applied": False, "reason": None}
        if not p.align_to_support or p.support_z is None or support < ctx.table_z + RAISED_SUPPORT_MIN_M:
            rec["reason"] = "support not raised"
            return tip_xy, yaw, rec
        if n < OFF_CENTRE_MIN_M:
            rec["reason"] = "centred grasp"
            return tip_xy, yaw, rec
        axis = _support_axis(ctx, p.object_xy, support)
        rec["support_axis"] = axis
        if not axis or not axis.get("narrow"):
            rec["reason"] = "support not seen as narrow / elongated"
            return tip_xy, yaw, rec
        a = np.asarray(axis["axis"], dtype=float)
        if a[0] < 0:
            a = -a  # the tip on the robots' side (-x) of the object first: the better-reached column
        best = None
        for sign in (1.0, -1.0):
            new_off = sign * a * n
            cand_xy = np.asarray(p.object_xy, dtype=float) - new_off
            dyaw = _wrap_pi(math.atan2(new_off[1], new_off[0]) - math.atan2(centre_off[1], centre_off[0]))
            cand_yaw = _wrap_pi(yaw + dyaw)
            sol = ctx.controller.solve_ik_detailed(p.side, [cand_xy[0], cand_xy[1], z_carry], cand_yaw)
            if sol is None:
                continue
            here = ctx.controller.solve_ik_detailed(p.side, ctx.tip(p.side), cand_yaw)
            if here is None:
                continue
            if best is None:
                best = (new_off, cand_xy, cand_yaw, dyaw, float(sol.limit_margin_rad))
        if best is None:
            rec["reason"] = "no reachable aligned pose"
            return tip_xy, yaw, rec
        new_off, cand_xy, cand_yaw, dyaw, margin = best
        if abs(dyaw) < math.radians(8.0):
            rec.update({"applied": False, "reason": "already aligned", "rotation_deg": round(math.degrees(dyaw), 1)})
            return tip_xy, yaw, rec
        # the wrist turn is folded into the single slow joint-space carry (_smooth_carry): a separate turn, or a turn inside a Cartesian
        # carry, swung the hanging bowl about the pinch and let it pivot there (its landing moved 4-7 cm)
        rec.update({"applied": True, "rotation_deg": round(math.degrees(dyaw), 1), "yaw_from": round(yaw, 3), "yaw_to": round(float(cand_yaw), 3), "centre_offset_used": [round(float(v), 4) for v in new_off], "limit_margin_rad": round(margin, 3)})
        return np.asarray(cand_xy, dtype=float), float(cand_yaw), rec

    @staticmethod
    def _smooth_carry(ctx, side: str, tip_xy, z_carry: float, yaw: float, off_centre: bool, yaw_from: float | None = None) -> dict:
        """Carry to the landing column. An object hanging off-centre (rim grasp) is carried in ONE slow joint-space move (the wrist turn
        of the alignment included, height up to 3 cm above the carry height when reachable) with a full settle - the waypoint carry
        stops and re-corrects every 5 cm and a Cartesian carry jitters at the waypoint IK: both swung the hanging bowl about the pinch
        and let it pivot there (measured: tilt 19 -> 33-50 deg, landing 4-7 cm off; the single joint move kept 18-20 deg and landed
        within 0.5 cm). A centred object keeps the waypoint carry. Returns {"ok", "reason", "mode", "z"}."""

        x, y = float(tip_xy[0]), float(tip_xy[1])
        if not off_centre:
            c = _carry(ctx, side, tip_xy, z_carry, z_carry + 0.03, yaw)
            c.update({"mode": "waypoints", "z": float(z_carry)})
            return c
        tip = ctx.tip(side)
        dist = float(np.linalg.norm(np.asarray(tip_xy, dtype=float) - tip[:2]))
        dyaw = 0.0 if yaw_from is None else abs(_wrap_pi(float(yaw) - float(yaw_from)))
        duration = float(np.clip(max(dist / CARRY_RATE_MPS, dyaw / ROTATE_RATE_RPS), 1.5, 5.0))
        z_use = _highest_reachable(ctx, side, x, y, z_carry, z_carry + 0.03, yaw) or float(z_carry)
        r = ctx.call("move_to", side=side, xyz=[x, y, float(z_use)], yaw=yaw, duration=duration, tolerance_m=0.02, correction_passes=2)
        if r.status == SkillStatus.success:
            return {"ok": True, "reason": None, "mode": "joint", "z": float(z_use), "duration_s": duration}
        c = _carry(ctx, side, tip_xy, z_carry, z_carry + 0.03, yaw, speed_mps=GENTLE_CARRY_MPS)
        c.update({"mode": "waypoints", "z": float(z_carry)})
        return c

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        held, width = _holding(ctx, p.side)
        if not held:
            return self.result(SkillStatus.failure, f"nothing is held by the {p.side} gripper (opening {0 if width is None else round(width * 100, 1)} cm)", verified=True, metrics={"stage": "precondition", "opening_width_m": width})
        support = float(ctx.table_z if p.support_z is None else p.support_z)
        support_meas = None
        if p.support_z is not None and support > ctx.table_z + SUPPORT_TOP_MIN_ABOVE_TABLE_M:
            # 1.2.0: the caller's support height is a perception estimate (a rack's base plate, a rim pixel): the top of what stands under the
            # landing point is re-measured from depth before the carry (the held object is still away from the landing point)
            support_meas = _support_top_from_depth(ctx, p.object_xy, ctx.table_z, support)
            if support_meas.get("refined"):
                support = float(support_meas["used"])
        off = float(p.object_bottom_offset_m)
        tip_xy = np.asarray(p.object_xy, dtype=float) - np.asarray(p.centre_offset_xy, dtype=float)
        z_touch = support + off
        z_carry = z_touch + SET_DOWN_CLEARANCE_M
        # 1.2.0: the descent may continue past the measured top down to the CALLER's (lower) height when nothing stops it - an object that misses the
        # rails (tilt across them) then settles between / beside them as it did before, instead of being dropped from the rail height
        support_low = min(support, float(p.support_z)) if p.support_z is not None else support
        yaw = float(p.yaw)
        yaw_from = yaw
        tip_xy, yaw, align = self._align(ctx, p, tip_xy, yaw, support, z_carry)
        off_centre = float(np.linalg.norm(np.asarray(p.centre_offset_xy, dtype=float))) >= OFF_CENTRE_MIN_M
        if ctx.controller.solve_ik(p.side, [tip_xy[0], tip_xy[1], z_carry], yaw) is None:
            ceiling = _highest_reachable(ctx, p.side, float(tip_xy[0]), float(tip_xy[1]), z_touch, z_carry, yaw)
            if ceiling is None:
                return self.result(SkillStatus.failure, f"the landing point {np.round(tip_xy, 3).tolist()} (tip) is not reachable at the carry height z={z_carry:.3f} by the {p.side} arm; the object is STILL HELD - choose a landing point closer to the arm base or use the other arm", verified=True, metrics={"stage": "transit", "reachable": False, "required_tip_z": round(z_carry, 4), "alignment": align, "support_z": round(support, 4), "support_measured": support_meas}, outputs={"held": True})
            z_carry = ceiling
        carry = self._smooth_carry(ctx, p.side, tip_xy, z_carry, yaw, off_centre, yaw_from=yaw_from)
        if not carry.get("ok"):
            r = ctx.call("move_to", side=p.side, xyz=[float(tip_xy[0]), float(tip_xy[1]), z_carry], yaw=yaw, tolerance_m=0.02)
            if r.status != SkillStatus.success:
                return self.result(SkillStatus.failure, f"could not carry {p.target_object} over the landing point: {r.message}; still held", verified=True, metrics={"stage": "transit", "carry": carry.get("reason"), "alignment": align}, outputs={"held": True})
        z_carry_nominal = float(z_carry)  # the post-release retreat / parking height stays the nominal one (hand_v13): the idle arm parked
        z_carry = float(carry.get("z") or z_carry)  # 3 cm higher after a raised joint carry collided with the other arm's next approach (oracle abc_rack s6)
        _settle_precise(ctx, p.side, [float(tip_xy[0]), float(tip_xy[1]), z_carry], yaw, tol_m=0.003)
        held_now, _ = _held(ctx, p.side)
        if not held_now:
            return self.result(SkillStatus.failure, f"{p.target_object} was lost during the carry", verified=True, metrics={"stage": "transit"}, outputs={"held": False})
        # descent to touchdown: 1 cm Cartesian steps, a stall (the object bottom on the support) ends it
        z_end = max(support_low + off - SET_DOWN_OVERSHOOT_M, ctx.table_z + _pad_below_tip_m(ctx) + 0.002)
        pts = [[float(tip_xy[0]), float(tip_xy[1]), float(z)] for z in np.arange(z_carry - 0.01, z_end - 1e-6, -0.01)] + [[float(tip_xy[0]), float(tip_xy[1]), float(z_end)]]
        pts = [q for i, q in enumerate(pts) if i == 0 or abs(q[2] - pts[i - 1][2]) > 1e-4]
        desc = _cartesian_path(ctx, p.side, pts, yaw, speed_mps=0.06, hold_s=0.1, correction_passes=1, stall_fraction=0.4, stall_patience=2)
        tip = ctx.tip(p.side)
        touchdown_dz = float(tip[2] - z_touch)
        if desc["stalled"]:
            _cartesian_path(ctx, p.side, [[float(tip_xy[0]), float(tip_xy[1]), float(tip[2]) + 0.003]], yaw, speed_mps=0.05)  # unload before letting go
        width_before = ctx.gripper_width(p.side)
        opened = ctx.call("set_gripper", side=p.side, aperture=1.0, duration=0.4)
        ctx.call("wait", seconds=0.2)
        width_open = (opened.outputs or {}).get("opening_width_m")
        commanded_open = (opened.outputs or {}).get("commanded_width_m") or 0.096
        released = width_open is None or width_before is None or float(width_open) >= min(float(width_before) + 0.02, float(commanded_open) - 0.01)
        up = _highest_reachable(ctx, p.side, float(tip_xy[0]), float(tip_xy[1]), float(tip[2]) + 0.03, z_carry_nominal + 0.05, yaw) or (float(tip[2]) + 0.03)
        _cartesian_path(ctx, p.side, [[float(tip_xy[0]), float(tip_xy[1]), up]], yaw, speed_mps=0.15)
        metrics = {"stage": "release", "support_z": round(support, 4), "support_measured": support_meas, "descent_floor_z": round(support_low, 4), "touchdown": bool(desc["stalled"]), "touchdown_dz_m": round(touchdown_dz, 4), "descent_steps": desc["n_reached"], "width_before_release_m": width_before, "width_after_open_m": width_open, "released": released, "carry_z": round(z_carry, 4), "alignment": align, "carry_mode": carry.get("mode", "waypoints"), "yaw_used": round(yaw, 3), "tip_target_xy": [round(float(v), 4) for v in tip_xy]}
        if not released:
            metrics["robot_time_s"] = _robot_time(ctx, tick0)
            return self.result(SkillStatus.failure, f"the fingers did not open after the release command (opening {round(float(width_open) * 100, 1)} cm vs {round(float(width_before) * 100, 1)} cm held): wedged - re-observe before the next action", verified=True, metrics=metrics)
        settled = None
        view = None
        if p.settle_check:
            _clear_view(ctx, p.side, tip_xy, up, yaw)
            # a resting object's top lies between the support and the grasp height + 6 cm (a rim-held bowl may rest leaning); points
            # above that cap are the arm / wrist in the oblique view and must not enter the blob (they made 3 of 5 rack landings that
            # the referee accepted read as "not settled")
            z_cap = support + off + 0.06
            view = _surface_blob(ctx, p.object_xy, support, 0.06, 0.01, z_cap=z_cap)
            metrics["depth_view"] = view
            if view["surface_max"] is not None:
                if view["object_xy"] is not None and view["object_top_z"] is not None:
                    landing_offset = float(np.linalg.norm(np.asarray(view["object_xy"]) - np.asarray(p.object_xy, dtype=float)))
                    metrics["landing_offset_m"] = round(landing_offset, 4)
                    metrics["object_top_z"] = view["object_top_z"]
                    settled = bool(landing_offset <= 0.06)
                elif view["n_over_cap"] > 20:
                    settled = None  # the arm still stands in the line of sight: cannot judge
                else:
                    settled = False
        metrics["settled"] = settled
        metrics["robot_time_s"] = _robot_time(ctx, tick0)
        outputs = {"release_pos": np.round(tip, 4).tolist(), "object_xy": None if view is None else view.get("object_xy"), "held": False}
        where = f"touched down {touchdown_dz * 100:+.1f} cm from the nominal contact height" if desc["stalled"] else "reached the support height without a felt touchdown"
        if settled is False:
            return self.result(SkillStatus.failure, f"released ({where}), but the top camera does not see {p.target_object} resting at the landing point (surface {view['surface_max']} m, expected >= {support + 0.01:.3f}): it may have been carried away on the fingers or slid off the support - re-observe with segment_objects", verified=True, metrics=metrics, outputs=outputs)
        how = "camera cannot judge" if settled is None else f"seen resting {metrics.get('landing_offset_m', 0) * 100:.1f} cm from the landing point, top {metrics.get('object_top_z')} m"
        turned = f"; turned {align.get('rotation_deg')} deg to lay its tilt along the support's axis" if align.get("applied") else ""
        remeasured = f"; support top re-measured from depth ({support_meas['given']:.3f} -> {support:.3f} m)" if support_meas and support_meas.get("refined") else ""
        return self.result(SkillStatus.success, f"{p.target_object} set down on the support at z={support:.3f} ({where}){turned}{remeasured}; {how}", verified=bool(settled), metrics=metrics, outputs=outputs)


class InsertOntoPeg(Skill):
    name = "insert_onto_peg"
    version = "1.1.0"
    kind = "code"
    description = "Lower a held object with a hole (nut, ring) onto a peg until it rests: peg re-localised from depth, hover 1.5 cm above the peg, re-centre the hole on it (depth), descend past the peg top in 1 cm steps (retries on measured / blind offsets), seat it, CHECK the seat before letting go (a jammed nut is lifted back, not released), release, verify. hole_xy_at_grasp = hole centre when grasped (segment_objects: the ring part's centre); grasp_xy / grasp_z / yaw = the pick's grasp (else pass the pick's yaw as grasp_yaw)."
    returns = "success if the object rests on the support around the peg (metrics.seated, seat_log); failure: nothing held, unreachable, not over the peg / jammed (STILL HELD), wedged, not seated"
    preconditions = ["the arm holds the object"]
    effects = []
    failure_conditions = []
    effect_verification = "top depth: ring top at the pause, at the seat (before release) and after"
    observations_used = ["proprio", "depth:top"]
    dependencies = ["set_gripper", "wait"]
    known_limitations = ["hole / peg clearance below ~3 mm"]
    timeout_s = 60.0

    class Params(Params):
        side: Side
        peg_xy: list[float] = Field(min_length=2, max_length=2, description="peg axis [m]")
        peg_top_z: float = Field(description="peg top [m]")
        hole_xy_at_grasp: list[float] = Field(min_length=2, max_length=2, description="hole centre when grasped [m]")
        grasp_xy: list[float] = Field(min_length=2, max_length=2, description="pick grasp xy [m]")
        grasp_z: float = Field(description="pick grasp z [m] (object on the table)")
        object_thickness_m: float = Field(default=0.012, ge=0.003, le=0.06)
        yaw: float = Field(description="grasp yaw [rad]")
        support_z: float | None = Field(default=None, description="landing surface [m]")
        search_offsets: int = Field(default=8, ge=0, le=16)
        target_object: str = Field(default="the nut", max_length=120)
        grasp_yaw: float | None = Field(default=None, description="the pick's yaw if `yaw` differs")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        held, width = _holding(ctx, p.side)
        if not held:
            return self.result(SkillStatus.failure, f"nothing is held by the {p.side} gripper (opening {0 if width is None else round(width * 100, 1)} cm)", verified=True, metrics={"stage": "precondition", "opening_width_m": width})
        hole_off = np.asarray(p.hole_xy_at_grasp, dtype=float) - np.asarray(p.grasp_xy, dtype=float)
        if float(np.linalg.norm(hole_off)) > 0.20:
            raise SkillError(f"hole centre {np.round(hole_off, 3).tolist()} m from the grasp point: not one held object", error_type="invalid_params")
        off = max(0.0, float(p.grasp_z) - float(ctx.table_z))  # tip above the object bottom (the object rested on the table when grasped)
        peg_xy = np.asarray(p.peg_xy, dtype=float)
        peg_top = float(p.peg_top_z)
        yaw = float(p.yaw)
        side = p.side
        thick = float(p.object_thickness_m)
        hole_off_rotated = None
        if p.grasp_yaw is not None and abs(_wrap_pi(yaw - float(p.grasp_yaw))) > 1e-3:
            # 1.1.0: the hole rides with the wrist - a yaw change since the pick turns the hole offset by the same angle (formal_v13
            # capx_nut_assembly s100: yaw -1.57 for a -1.32 grasp put the nominal hole 1.4 cm off; the depth re-centre absorbed only part)
            da = _wrap_pi(yaw - float(p.grasp_yaw))
            c, s = math.cos(da), math.sin(da)
            hole_off = np.array([c * hole_off[0] - s * hole_off[1], s * hole_off[0] + c * hole_off[1]])
            hole_off_rotated = round(math.degrees(da), 1)
        # 1.1.0: the peg axis and top are re-localised from the top depth image BEFORE the carry (the held object is still away from the
        # peg): the caller's estimate came from a footprint centroid (formal_v13 s100: 1.3 cm off) and the hole was then centred on that
        # wrong axis with 7 mm of clearance. The peg's top face is the highest surface within PEG_RELOCALISE_R_M of the estimate.
        peg_reloc = _peg_top_from_depth(ctx, peg_xy, peg_top)
        if peg_reloc is not None and peg_reloc["complete"] and peg_reloc["shift_m"] <= PEG_RELOCALISE_MAX_SHIFT_M:
            # a partially occluded face (the arm holding the object in the camera's line of sight) is biased toward its visible part: kept as a
            # record, not used (oracle abc_rack-style relay poses occluded 6 of 15 pegs by 0.9-1.4 cm)
            peg_xy = np.asarray(peg_reloc["centre"], dtype=float)
            peg_top = float(peg_reloc["top_z"]) if abs(float(peg_reloc["top_z"]) - peg_top) <= 0.02 else peg_top
        # the support the object lands on: given, or the lowest surface the top camera sees 3-4.5 cm from the peg axis (plate or table)
        support = p.support_z
        if support is None:
            pts = _depth_points(ctx, peg_xy, peg_top - 0.05, 0.05)
            if pts is not None and len(pts):
                d = np.linalg.norm(pts[:, :2] - peg_xy, axis=1)
                ring = pts[(d >= 0.03) & (d <= 0.045)]
                if len(ring) >= 10:
                    support = float(np.percentile(ring[:, 2], 20))
        support = float(ctx.table_z if support is None else max(support, ctx.table_z))
        tip_xy = peg_xy - hole_off
        # first hover 1 cm higher than nominal: the object may ride higher in the pinch than at the grasp (measured below from depth)
        z_hover = peg_top + off + INSERT_HOVER_CLEARANCE_M + 0.01
        if ctx.controller.solve_ik(side, [tip_xy[0], tip_xy[1], z_hover], yaw) is None:
            ceiling = _highest_reachable(ctx, side, float(tip_xy[0]), float(tip_xy[1]), peg_top + off, z_hover, yaw)
            return self.result(SkillStatus.failure, f"the hover pose over the peg (tip {np.round(tip_xy, 3).tolist()}, z={z_hover:.3f}) is not reachable by the {side} arm (ceiling {ceiling}); {p.target_object} is STILL HELD - use the other arm or re-grasp closer to the hole", verified=True, metrics={"stage": "transit", "reachable": False, "required_tip_z": round(z_hover, 4), "reach_ceiling_z": ceiling}, outputs={"held": True})
        carry = _carry(ctx, side, tip_xy, z_hover, z_hover + 0.04, yaw)
        if not carry.get("ok"):
            r = ctx.call("move_to", side=side, xyz=[float(tip_xy[0]), float(tip_xy[1]), z_hover], yaw=yaw, tolerance_m=0.02)
            if r.status != SkillStatus.success:
                return self.result(SkillStatus.failure, f"could not carry {p.target_object} over the peg: {r.message}; still held", verified=True, metrics={"stage": "transit", "carry": carry.get("reason")}, outputs={"held": True})
        _settle_precise(ctx, side, [float(tip_xy[0]), float(tip_xy[1]), z_hover], yaw, tol_m=0.002)
        held_now, _ = _held(ctx, side)
        if not held_now:
            return self.result(SkillStatus.failure, f"{p.target_object} was lost during the carry", verified=True, metrics={"stage": "transit"}, outputs={"held": False})
        # depth re-centre: the ring's top-surface points around the expected hole position; their centroid is the hole centre
        # where does the object ride in the pinch? the ring's top surface height from depth replaces the nominal grasp geometry
        # (a handle-held nut measured 1.4 cm higher relative to the tip after the carry than at the grasp)
        tip = ctx.tip(side)
        # 1.1.0: the band ends just above the fingertip and the finger blades / links around the tip are excluded (they read 0.8-2 cm ABOVE
        # the tip and were taken for the ring top: the nut was then believed to ride 1.2 cm higher than it did - formal_v13 s100); the
        # measurement replaces the nominal riding height only for a LEVEL ring (a pitched nut's median top is not its bottom + thickness)
        cluster = _ring_top_cluster(ctx, tip[:2] + hole_off, peg_top - 0.01, float(tip[2]) + RING_ABOVE_TIP_MAX_M, exclude_xy=tip[:2], exclude_r=FINGER_EXCLUDE_R_M)
        ring_top = None if cluster is None else cluster["top"]
        ring_spread = None if cluster is None else cluster["spread"]
        off_nominal = off
        ride_meas = None
        if ring_top is not None:
            # RECORDED, not used (1.1.0): the nominal riding height (grasp z above the table the object rested on) was within 2 mm of the
            # truth in every measured case, while this depth reading was 0.8-1.2 cm off in 3 of 5 (finger links and a pitched ring in the band) and
            # each error moved the hover, pause and seat heights and masked a nut resting on the peg tip (formal_v13 s100). A nut that rides
            # higher than nominal after the carry is caught by the pause check (low quartile of the ring top vs the peg geometry).
            ride_meas = round(float(tip[2]) - (ring_top - thick), 4)
        z_hover = peg_top + off + INSERT_HOVER_CLEARANCE_M
        if abs(float(tip[2]) - z_hover) > 0.005:
            _cartesian_path(ctx, side, [[float(tip_xy[0]), float(tip_xy[1]), z_hover]], yaw, speed_mps=0.1)
            _settle_precise(ctx, side, [float(tip_xy[0]), float(tip_xy[1]), z_hover], yaw, tol_m=0.002)
        # depth re-centre: the ring's top surface gives the hole centre (extent estimator, robust to the handle); a large correction is
        # re-measured once after the move
        recentre = None
        total_corr = np.zeros(2)
        for _pass in range(2):
            tip = ctx.tip(side)
            expected_top = float(tip[2]) - off + thick
            hole_est = tip[:2] + hole_off  # the object moves with the tip: its estimate relative to the tip is what the caller gave
            ring = _ring_centre_from_depth(ctx, hole_est, expected_top, tip[:2] - hole_est)
            if ring is None:
                break
            corr = peg_xy - ring["centre"]
            if float(np.linalg.norm(corr)) > MAX_DEPTH_RECENTRE_M:
                break
            total_corr = total_corr + corr
            tip_xy = tip_xy + corr
            recentre = {"n": ring["n"], "ring_half_size_m": ring["half_size_m"], "ring_centre_xy": [round(float(v), 4) for v in ring["centre"]], "correction_m": [round(float(v), 4) for v in total_corr], "passes": _pass + 1}
            _settle_precise(ctx, side, [float(tip_xy[0]), float(tip_xy[1]), z_hover], yaw, tol_m=0.002)
            if float(np.linalg.norm(corr)) <= 0.006:
                break
        z_pause = peg_top + off - INSERT_PASS_DEPTH_M
        z_seat = max(support + off + 0.002, ctx.table_z + _pad_below_tip_m(ctx) + 0.002)

        def descend_to(x0: float, y0: float, z_from: float, z_to: float, patience: int = 2) -> dict:
            zs = [float(z) for z in np.arange(z_from - 0.01, z_to - 1e-6, -0.01)] + [float(z_to)]
            pts_ = [[x0, y0, z] for i, z in enumerate(zs) if i == 0 or abs(z - zs[i - 1]) > 1e-4]
            return _cartesian_path(ctx, side, pts_, yaw, speed_mps=0.06, hold_s=0.1, correction_passes=1, stall_fraction=0.4, stall_patience=patience)

        def ring_top_above_expected(expected: float, pct: float) -> tuple[float | None, float]:
            """The ``pct`` percentile of the ring-top heights in the annulus around the peg axis (finger links excluded) minus ``expected``
            (1.1.0: the expectation comes from the peg / support geometry, not from the fingertip and the uncertain riding height: at the
            pause depth a nut that went over the peg has its top at peg_top + thick - pass depth, a seated one at support + thick). A
            handle-held nut rides PITCHED (its far side ~1 cm up): the pause check reads the LOW quartile - a nut resting on the peg tip
            has even its lowest ring point at peg_top + thick - and the seat check the median."""

            pts_ = _depth_points(ctx, peg_xy, expected, ANNULUS_R_M[1] + 0.01)
            if pts_ is None or len(pts_) == 0:
                return None, expected
            d_ = np.linalg.norm(pts_[:, :2] - peg_xy, axis=1)
            tip_now = ctx.tip(side)
            far_from_fingers = np.linalg.norm(pts_[:, :2] - tip_now[:2], axis=1) > FINGER_EXCLUDE_R_M  # the finger links slant into the annulus
            sel_ = pts_[(d_ >= ANNULUS_R_M[0]) & (d_ <= ANNULUS_R_M[1]) & far_from_fingers]
            if len(sel_) < 8:
                return None, expected
            return float(np.percentile(sel_[:, 2], pct)) - expected, expected

        expected_pause_top = peg_top + thick - INSERT_PASS_DEPTH_M
        expected_seat_top = support + thick

        attempts = 0
        tilt_log = []
        used_offset = None
        blind = 0
        fx, fy = float(tip_xy[0]), float(tip_xy[1])
        state = {"attempts": 0, "blind": 0, "used_offset": None}

        def align_over_peg(fx0: float, fy0: float) -> tuple[bool, float, float]:
            """Descend to the pause depth and check the ring from depth; back up and retry on the measured correction (else a blind offset)
            until the ring sits over the peg axis. Returns (aligned, fx, fy)."""

            fx_, fy_ = fx0, fy0
            while True:
                state["attempts"] += 1
                desc = descend_to(fx_, fy_, z_hover, z_pause)
                tip_ = ctx.tip(side)
                above, expected = ring_top_above_expected(expected_pause_top, 25)
                # where is the ring now relative to the peg axis? (a nut lowered beside the peg is level and does not stall)
                ring_now = _ring_centre_from_depth(ctx, peg_xy, expected, tip_[:2] - peg_xy)
                lateral_err = None if ring_now is None else float(np.linalg.norm(ring_now["centre"] - peg_xy))
                tilted = bool(desc["stalled"]) or (above is not None and above > PAUSE_ABOVE_TOL_M)
                off_target = lateral_err is not None and lateral_err > PAUSE_LATERAL_TOL_M
                tilt_log.append({"offset_m": None if state["used_offset"] is None else [round(v, 4) for v in state["used_offset"]], "stalled": bool(desc["stalled"]), "ring_above_expected_m": None if above is None else round(above, 4), "ring_lateral_error_m": None if lateral_err is None else round(lateral_err, 4), "tip_z": round(float(tip_[2]), 4)})
                if not tilted and not off_target:
                    return True, fx_, fy_
                if state["attempts"] > p.search_offsets:
                    return False, fx_, fy_
                # back up over the peg; retry on the MEASURED correction when the camera sees the ring, else on the next blind offset
                _cartesian_path(ctx, side, [[fx_, fy_, z_hover]], yaw, speed_mps=0.12, allow_jump=True)
                if lateral_err is not None and lateral_err <= MAX_DEPTH_RECENTRE_M and lateral_err > 0.003:
                    corr = peg_xy - ring_now["centre"]
                    fx_, fy_ = float(fx_ + corr[0]), float(fy_ + corr[1])
                else:
                    ring_k = 1 + (state["blind"] // 8)
                    ang = 2 * math.pi * (state["blind"] % 8) / 8 + (math.pi / 8 if ring_k % 2 == 0 else 0.0)
                    state["blind"] += 1
                    fx_, fy_ = float(tip_xy[0] + SEARCH_STEP_M * ring_k * math.cos(ang)), float(tip_xy[1] + SEARCH_STEP_M * ring_k * math.sin(ang))
                state["used_offset"] = (fx_ - float(tip_xy[0]), fy_ - float(tip_xy[1]))
                lateral = _cartesian_path(ctx, side, [[fx_, fy_, z_hover]], yaw, speed_mps=0.12)
                if lateral["stop_reason"] is not None:
                    continue
                _settle_precise(ctx, side, [fx_, fy_, z_hover], yaw, tol_m=0.002)

        seat_log: list[dict] = []
        seat = None
        rounds = 0
        while True:
            rounds += 1
            aligned, fx, fy = align_over_peg(fx, fy)
            attempts, used_offset = state["attempts"], state["used_offset"]
            metrics = {"stage": "insert", "support_z": round(support, 4), "hover_z": round(z_hover, 4), "bottom_offset_nominal_m": round(off_nominal, 4), "bottom_offset_used_m": round(off, 4), "bottom_offset_depth_m": ride_meas, "ring_top_measured_z": None if ring_top is None else round(ring_top, 4), "ring_top_spread_m": None if ring_spread is None else round(ring_spread, 4), "hole_offset_rotated_deg": hole_off_rotated, "peg_relocalised": peg_reloc, "peg_xy_used": [round(float(v), 4) for v in peg_xy], "peg_top_used": round(peg_top, 4), "depth_recentre": recentre, "search_attempts": attempts, "search_offset_m": None if used_offset is None else [round(v, 4) for v in used_offset], "tilt_log": tilt_log, "seat_log": seat_log}
            if not aligned:
                _cartesian_path(ctx, side, [[fx, fy, z_hover]], yaw, speed_mps=0.12, allow_jump=True)
                metrics["robot_time_s"] = _robot_time(ctx, tick0)
                return self.result(SkillStatus.failure, f"{p.target_object} did not go over the peg at the estimate and at {attempts - 1} retries (tilted on the tip or beside it; measured ring offsets / blind {SEARCH_STEP_M * 100:.1f} cm steps): the hole is further from the peg than the search covers. It is STILL HELD (lifted back over the peg) - re-localise the peg (segment_objects / locate_objects 'raised') and call again with the new peg_xy", verified=True, metrics=metrics, outputs={"held": True})
            # the peg is through the hole: seat the object on the support (touchdown stall) - and CHECK the seat before letting go (1.1.0)
            seat = descend_to(fx, fy, float(ctx.tip(side)[2]), z_seat, patience=1)
            tip = ctx.tip(side)
            above_seat, _expected_seat = ring_top_above_expected(expected_seat_top, 50)
            high_seat, _ = ring_top_above_expected(expected_seat_top, 90)
            high_stall = bool(seat["stalled"]) and (float(tip[2]) - z_seat) > SEAT_STALL_TOL_M
            # jammed: the ring's median well above the seated height (resting on the peg), or its high side far up (hanging tilted on the
            # peg's edge: the median can straddle the seated height while one side stands centimetres up)
            jammed = (above_seat is not None and above_seat > TILT_ABOVE_EXPECTED_M) or (high_seat is not None and high_seat > 2 * TILT_ABOVE_EXPECTED_M) or (above_seat is None and high_stall)
            seat_log.append({"round": rounds, "stalled": bool(seat["stalled"]), "tip_z": round(float(tip[2]), 4), "seat_z": round(z_seat, 4), "ring_above_expected_m": None if above_seat is None else round(above_seat, 4), "ring_p90_above_expected_m": None if high_seat is None else round(high_seat, 4), "jammed": bool(jammed)})
            if not jammed:
                break
            # lift back over the peg (the object stays held) and re-align from a fresh depth re-centre
            _cartesian_path(ctx, side, [[fx, fy, z_hover]], yaw, speed_mps=0.12, allow_jump=True)
            _settle_precise(ctx, side, [fx, fy, z_hover], yaw, tol_m=0.002)
            if rounds >= SEAT_ROUNDS:
                metrics["robot_time_s"] = _robot_time(ctx, tick0)
                why = f"the ring top reads {above_seat * 100:.1f} cm above the seated height" if above_seat is not None else f"the descent stalled {(float(tip[2]) - z_seat) * 100:.1f} cm above the seated tip height"
                return self.result(SkillStatus.failure, f"{p.target_object} jammed tilted on the peg at the seat ({why}) in {rounds} round(s); NOT released - lifted back over the peg, STILL HELD. Re-localise the peg (segment_objects / locate_objects 'raised') and call again with the new peg_xy, or re-grasp closer to the hole", verified=True, metrics=metrics, outputs={"held": True})
            tip = ctx.tip(side)
            ring = _ring_centre_from_depth(ctx, tip[:2] + hole_off, float(tip[2]) - off + thick, -hole_off)
            if ring is not None and float(np.linalg.norm(peg_xy - ring["centre"])) <= MAX_DEPTH_RECENTRE_M:
                corr = peg_xy - ring["centre"]
                fx, fy = float(fx + corr[0]), float(fy + corr[1])
                _settle_precise(ctx, side, [fx, fy, z_hover], yaw, tol_m=0.002)
        if seat["stalled"]:
            _cartesian_path(ctx, side, [[fx, fy, float(tip[2]) + 0.003]], yaw, speed_mps=0.05)
        width_before = ctx.gripper_width(side)
        opened = ctx.call("set_gripper", side=side, aperture=1.0, duration=0.4)
        ctx.call("wait", seconds=0.3)
        width_open = (opened.outputs or {}).get("opening_width_m")
        commanded_open = (opened.outputs or {}).get("commanded_width_m") or 0.096
        released = width_open is None or width_before is None or float(width_open) >= min(float(width_before) + 0.02, float(commanded_open) - 0.01)
        up = _highest_reachable(ctx, side, fx, fy, z_hover, z_hover + 0.05, yaw) or z_hover
        _cartesian_path(ctx, side, [[fx, fy, up]], yaw, speed_mps=0.15, allow_jump=True)
        metrics.update({"stage": "release", "seat_touchdown": bool(seat["stalled"]), "release_tip_z": round(float(tip[2]), 4), "width_before_release_m": width_before, "width_after_open_m": width_open, "released": released})
        if not released:
            metrics["robot_time_s"] = _robot_time(ctx, tick0)
            return self.result(SkillStatus.failure, f"the fingers did not open after the release command (opening {round(float(width_open) * 100, 1)} cm): wedged - re-observe", verified=True, metrics=metrics)
        _clear_view(ctx, side, [fx, fy], up, yaw)
        # verification: the ring top around the peg axis should sit thick above the support (seated); near the peg top = hung on the tip
        pts = _depth_points(ctx, peg_xy, support + thick, ANNULUS_R_M[1] + 0.01)
        seated = None
        ring_z = None
        spread = None
        if pts is not None and len(pts):
            d = np.linalg.norm(pts[:, :2] - peg_xy, axis=1)
            sel = pts[(d >= ANNULUS_R_M[0]) & (d <= ANNULUS_R_M[1])]
            if len(sel) >= 8:
                ring_z = float(np.percentile(sel[:, 2], 50))
                spread = float(np.percentile(sel[:, 2], 90) - np.percentile(sel[:, 2], 10))  # a level ring reads flat; a tilted one spans centimetres
                seated = bool(support + 0.002 <= ring_z <= support + thick + SEATED_BAND_ABOVE_M and spread <= SEATED_MAX_SPREAD_M)
        metrics.update({"stage": "verify", "seated": seated, "seated_top_z": None if ring_z is None else round(ring_z, 4), "seated_top_spread_m": None if spread is None else round(spread, 4), "expected_top_z": round(support + thick, 4), "robot_time_s": _robot_time(ctx, tick0)})
        outputs = {"release_pos": np.round(tip, 4).tolist(), "held": False, "seated": seated}
        if seated is False:
            hung = ring_z is not None and ring_z > peg_top - 0.03
            tilted_msg = f" and spans {spread * 100:.1f} cm in height (tilted)" if spread is not None and spread > SEATED_MAX_SPREAD_M else ""
            return self.result(SkillStatus.failure, f"released, but the top camera reads the ring top at {ring_z:.3f} m around the peg (expected {support + thick:.3f} when seated){tilted_msg}: {p.target_object} {'hangs on the peg tip' if hung else 'is not seated around the peg'}", verified=True, metrics=metrics, outputs=outputs)
        how = "seated height confirmed by the top camera" if seated else "the top camera cannot judge the seated height"
        found = f"; hole re-centred {np.linalg.norm(recentre['correction_m']) * 100:.1f} cm from depth" if recentre else ""
        searched = f"; aligned after {attempts - 1} search offset(s)" if attempts > 1 else ""
        return self.result(SkillStatus.success, f"{p.target_object} lowered over the peg and released {'on contact' if seat['stalled'] else 'at the support height'}{found}{searched}; {how}", verified=bool(seated), metrics=metrics, outputs=outputs)
