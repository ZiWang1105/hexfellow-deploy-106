"""Cloth folding skills (candidate cand_20260910_164848_9a670e, parent hand_v6@3.2.0): ``fold_cloth`` 1.1.0 (the fold API) and
``bimanual_fold`` 2.1.0 (same contract as hand_v5's 1.2.0, the 1.1.0 mechanics). Deployable observations only: fingertip FK,
calibrated gripper openings, the caller's corner estimates (segment_objects footprint corners).

hand_v6 follow-up 2 (docs/results/hand_v6_regression/decision_notes.md): with the runtime agent's SAM3 corners the fold lost a
pinch mid-carry in 2 of 6 folds (agent 2/5, oracle on truth corners 5/5), attributed there to the reach edge. The privileged probe
(runs/diag_fold/probe.py, seeds 0-9, both corner sources) showed otherwise:

* the SAM3 footprint corners lie 1-6 mm OUTSIDE the material corners (mean ~2.5 mm; the 3 mm inset corrects the mean) and the
  resulting pinch points differ from the truth-corner ones by only 1-2 mm - yet the SAM3-corner folds lost a pinch in 5 of 9
  seeds at path fractions 0.35-0.89, with the tips at x 0.51-0.71 and joint-limit margins 0.08-0.35 rad (truth-corner folds held
  with margins of 0.002 rad at x 0.74). The pinch, not the reach, was marginal: 10 mm along the edge the 12 mm pads span 4-16 mm
  from the edge and hold only the corner vertex of the 25 mm flex grid (1-2 vertices between the pads); the width then drops from
  ~9.5 to < 3 mm within 0.2 s (a yank, not creep) whenever the trailing cloth tugs;
* pinching 20 mm along the edge puts the second edge vertex between the pads: SAM3-corner referee successes 4/9 -> 7/10 (three
  configurations, 9/10 with the 5 cm release), truth corners 7-8/10 unchanged; 25 mm pinches two layers (3/10), a deeper inboard
  pinch (15-20 mm) holds but lands 5-8 cm off (3/10); carry speed, slack and lateral compensation change nothing;
* releasing from 6 cm the moved corner landed ~2 cm LONG in 2 of 10 seeds, off the fixed layer (referee "not on top": rise
  1.6-2.6 mm); from 5 cm every landing was 5-30 mm short and on the layer (rise 9-12 mm);
* the remaining losses are late (fraction 0.85-0.95, the descent onto the target: the dropped corner lands within tolerance, 5/5
  probes) or in the first 2 % of the lift (one seed: the flap slipped with a good 10 mm width; the cloth is still flat there and a
  re-pinch 1 cm further along the fold recovers it). A mid-carry loss no longer drops both corners into a heap: the held side
  carries its corner on to the target alone and the result names where the dropped corner lies;
* the far-edge fold (edge='far') pinches at the reach edge: 4/10, gathered cloth in 6 probes - 'near' stays the direction.

What holds the cloth and what made it slip (privileged diagnosis, runs/diag/pinch_probe.py, fold_towel seeds 0-9):

* The pinch holds ONE 12 mm flex layer of the corner flap standing between the pads (opening 10-12 mm) by pad friction:
  mu 4, ~2.4 N squeeze (finger position servo kp 400 N/m x 6 mm error), contacts on the pad faces 15-25 mm above the pad
  bottom. MuJoCo's soft friction has no static regime: a sustained tangential load creeps the pinched vertices out at
  ~(1-d)/(d b) F/m ~ 10 mm/s per newton (vertex mass 0.1-0.4 g), so the fold must keep the tangential load and its
  duration small. The fold-direction drag is normal to the pads (yaw 0 closes along the fold) and is held geometrically.
* Loads that were measured and removed:
  - pads closing AT the table pressed cloth vertices 13-16 mm into the table box; the stiff edge constraint to the pinched
    vertex (edge strain > 1.0) yanked the flap out within ~1 s of lifting -> pads 3 mm above the table, close ramped over
    0.4 s (strain stays < 0.3, 10/10 seeds keep both pinches through the lift);
  - a 5 cm carry ploughed the pads through the two-layer pile (0.3-0.8 N on the fingers) -> carry at 8 cm (0.1-0.15 N);
  - every intermediate STOP of a segmented two-arm carry (settle at each waypoint) let the trailing cloth swing and yank
    the pinch (losses at segment boundaries in 4/10 seeds) -> one continuous lift-carry-descend path
    (controller.move_cartesian_path_both), release right at its end without a settle (0/10 losses);
  - lowering onto the pile before release squeezed the flap out -> release from 6 cm above the pinch height; the corner
    then lands within +-2 cm of the tip along the fold, but 2-4 cm INWARD of it (the 3 cm slack that keeps the near edge
    from being tensioned between the two arms pulls the flap inward) -> end targets shifted 4 cm outward.
* Reach: far corners beyond x ~0.75 m are outside the top-down envelope at any usable height; the end pose is searched
  over (lateral compensation, release height, pull-back <= 4 cm) and the fold lands short when the pull-back is needed
  (the referee's 5 cm tolerance absorbs <= 2 cm).

hand_v8@3.4.0 (parent hand_v7): the held / empty decision is no longer a constant in this library. ``_held`` reads the reading's calibrated width rule (``GripperReading.held_by_width``: opening >= the backend/side ``held_width_min_m`` of configs/gripper/<backend>_<side>.yaml; ambiguous and empty count as not held) and ``_held_width_min`` exposes that threshold to the pinch / transfer checks. No other behaviour change.

hand_v9@3.5.0 (parent hand_v8): correction of the hand_v7 integration defect (docs/evaluation_protocol.md amendment 9 addendum). This file is hand_v8's ``cloth_fold_9a670e.py`` - candidate cand_20260910_164848_9a670e's fold_cloth 1.1.0 / bimanual_fold 2.1.0 / repair_corner 1.0.0 with the calibrated pinch floor and carry-loss monitor - now the library's single cloth_fold.py: fold_cloth 1.1.1, bimanual_fold 2.1.1, repair_corner 1.0.1 (unchanged from hand_v8). hand_v7 / hand_v8 loaded only repair_corner from this code and fold_cloth 1.0.x / bimanual_fold 2.0.x from hand_v6's cloth_fold.py. The calibrated held / empty width rule of hand_v8 (``_held`` / ``_held_width_min`` from configs/gripper/<backend>_<side>.yaml) is kept everywhere; no width constant.

Candidate cand_20260911_003629_419496 (parent hand_v9@3.5.0): fold_cloth 1.2.0 = the 1.1.1 fold + a MEASURED finish, align_corner 1.0.0 (new).
Evidence: docs/results/fold_towel_hard/README.md (every method 0/5 at 1 cm; clean folds land 10-27 mm off; 7/10 false success claims - SAM3
footprint checks cannot resolve 1 cm), formal_v9 fold_towel s100 / s101 (a partial fold treated as terminal; a halved footprint does not
establish layer order). Privileged probes (runs/probe_fold, fold_towel_hard seeds 0-9, the agent's SAM3 corners):

* the release height is not the lever: release 5 cm -> 1/10 at 1 cm (best corner offsets 6-33 mm, landings mostly SHORT along the fold by
  8-32 mm, lateral errors 0-12 mm), 4 cm -> 2/10 (8-33 mm), 3 cm -> 1/10 and worse (30-43 mm, corners landing on bunched cloth);
* the top camera is 640 x 480 with ~3 mm per pixel at the far corners (1.7 m above the robots' side, 30 deg tilt); its depth image resolves
  the two-layer region (moved layer on the fixed one, ~25 mm) from one layer (~13 mm) and the table once the heights are read relative to
  the table surface next to the cloth (bias +-2 mm cancels; 3 x 3 median removes the per-pixel salt). ``_measure_fold`` finds, per side and
  in the fold frame, the visible far / outer edges of ANY cloth and of the TWO-layer region by bin occupancy (4 mm bins, empty shadow bins
  skipped); the moved corner is the two-layer far corner, its target the visible fixed edge - or, where the top layer overhangs the fixed
  edge, the pre-fold edge measured the same way. Error vs truth: 3-6 mm on the offset vector where the fixed edge is visible, ~12 mm where
  it is hidden (the fold drags the fixed half by up to 1 cm). A corner that curled over (vertex 3 cm inward ON the layer, seed 1) shows more
  than two layers near the corner and is reported as not measurable; the skill never claims an alignment finer than +-5 mm;
* align_corner (pinch at the corner, lift 3 cm, carry, lower, open in place) is a bounded attempt: with truth offsets it brought 1 of 4
  reachable corners within 1 cm (16 -> 7 mm) and doubled the flap / dragged the fixed layer (mu = 4 cloth-on-cloth) in others; a slide on
  the layer moves nothing (the cloth creeps through the pinch); far corners beyond x ~0.75 m cannot be pinched at pad height. The finish
  is opt-in (``nudge``) and its outcome is re-measured and reported; the measurement itself always runs against ``corner_tolerance_m``
  (default 0.05 = the referee's fold_towel tolerance; the agent / oracle pass the task's 0.01 for fold_towel_hard) and a tight tolerance
  with an unmeasurable side is PARTIAL, never a pinch-evidence success. Releasing ON the far corners for a tight tolerance (landings
  average 1 cm short of the 1 cm-back target) was tried in the first oracle validation pass and dropped: the longer path tensioned the
  cloth and lost a pinch at 68-97 % of the carry in 3 of 5 seeds.

hand_v11@3.7.0 (parent hand_v10): cell-calibration plumbing (docs/evaluation_protocol.md, hand_v11 addendum). ``_FoldCore.execute`` (fold_cloth 1.2.1, bimanual_fold 2.1.2), ``RepairCorner.run`` (1.0.2) and ``AlignCorner.run`` (1.0.1) take the pinch height's pad offset from ``ctx.cell_calibration.finger_pad_below_tip_m``; ``FOOTPRINT_INSET_M`` is gone - fold_cloth's ``footprint_inset_m`` defaults to None = the cell calibration's ``footprint_inset_m`` (an explicit value still overrides it). The values come from configs/cell_calibration/<backend>.yaml through ``ctx.cell_calibration`` (physapi.calibration.cell), loaded per backend kind (sim -> sim.yaml = the constants this file used to hard-code, so simulation decisions are identical; real / dry -> real.yaml). No skill branches on the backend kind. No other behaviour change.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

import cv2

from physapi.perception.geometry import deproject
from physapi.schemas.core import SkillStatus
from physapi.skills.base import Params, Skill, SkillError

_spec = importlib.util.spec_from_file_location(f"physapi_libmod_{hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]}_manipulation_base", Path(__file__).with_name("manipulation.py"))
if _spec.name in sys.modules:
    _base = sys.modules[_spec.name]
else:
    _base = importlib.util.module_from_spec(_spec)
    assert _spec.loader
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
_robot_time = _base._robot_time
_held_width_min = _base._held_width_min

Side = Literal["left", "right"]
SIDES = ("left", "right")
OTHER = {"left": "right", "right": "left"}

# ------------------------------------------------------------------------------------- measured constants (see module doc)
PAD_CLEARANCE_M = 0.003  # pad bottom above the table at the pinch: 0 pins cloth into the table, >= 5 mm slides over the 6 mm-radius layer
GRASP_INBOARD_M = 0.012  # pinch this far from the corner along the fold: a single-layer flap (deeper insets pinch two layers that extrude; 15-20 mm landed 5-8 cm off)
#: Pinch this far from the corner along the edge (toward the other arm). hand_v6 used 10 mm: the 12 mm pads then span 4-16 mm from
#: the edge and hold only the corner vertex of the 25 mm flex grid (1-2 vertices between the pads) - a marginal pinch that a 1-2 mm
#: shift of the corner estimate flips: with the agent's SAM3 footprint corners (1-6 mm outside the material corners) the pinch was
#: yanked out mid-carry in 5 of 9 seeds at x 0.51-0.71, joint margins 0.08-0.35 rad (NOT the reach edge); with truth corners the
#: same geometry held 8/10. At 20 mm the pads span 14-26 mm and the second edge vertex sits between them: SAM3-corner folds 4/9 ->
#: 7-9/10 (probe sweeps, runs/diag_fold). 25 mm pinches two layers again (3/10).
EDGE_INSET_M = 0.02
CLOSE_RAMP_S = 0.4  # gripper command ramped open -> closed over this time (a step squeezed vertices into the table)
CLOSE_HOLD_S = 0.8  # total closing time before the width is read
LIFT_M = 0.03  # first waypoint: straight up, clears the 12 mm layer before moving
CARRY_HEIGHT_M = 0.08  # pads clear the two-layer pile with margin
#: Release from here. hand_v6 released from 6 cm; the moved corner then landed ~2 cm LONG in 2 of 10 seeds, off the fixed layer
#: onto the table (referee "not on top", rise 1.6-2.6 mm vs 3 mm). From 5 cm the landing is 5-30 mm short of the target corner and on
#: the layer (rise 9-12 mm) in 9/10 probes; lower squeezes the flap out on the pile.
RELEASE_HEIGHT_M = 0.05
#: Carry-loss handling (measured on the probe sweeps): a width below LOST_WIDTH_M is a lost pinch; a loss before the tip moved
#: EARLY_RECOVERY_PROGRESS_M along the fold happened during the lift with the cloth still flat (re-pinch once); a loss after
#: LATE_COMPLETE_FRACTION of the path is in the descent onto the target and the dropped corner lands within tolerance (5/5 probes),
#: so the path is completed; in between the held side carries its corner on alone.
#: hand_v8: a lost pinch reads below the calibrated held threshold (was the 3 mm constant LOST_WIDTH_M)
EARLY_RECOVERY_PROGRESS_M = 0.04
LATE_COMPLETE_FRACTION = 0.85
#: A re-pinch after an early slip must hold ONE layer (10-12 mm): the slipped flap can flip over and a 23 mm two-layer pinch then
#: carried the corner onto a pile 10 cm short while the skill believed it held (probe seed 7) - a half fold with an honest failure is
#: better than that.
RECOVERY_MAX_WIDTH_M = 0.016
#: Landing check (post-release top-camera re-detection of the cloth footprint, deployable): a complete single fold covers about
#: half of the flat footprint (referee window 0.42-0.58 on the convex hull; the SAM3 oriented box reads ~0.5-0.6), and no cloth
#: extends beyond the fold line toward the robots. A moved corner that dropped ON the fixed layer is invisible from above (same
#: cloth), so the pinch evidence decides first and the re-detection confirms / measures gross shortfalls.
LANDING_MAX_AREA_RATIO = 0.64
LANDING_MAX_OVERHANG_M = 0.04
#: End targets pulled back this far for reach land the corners 5-7 cm short of the far corners (formal_v7 fold_towel s100: 7.3 cm
#: offset after a 4 cm pull-back was reported as success): such a fold is PARTIAL and needs repair_corner on both sides.
PARTIAL_PULL_BACK_M = 0.04
#: repair_corner: pad clearance above the table for a corner lying on ONE fixed layer (12 mm flex layer) vs on the table.
LAYER_THICKNESS_M = 0.012
REPAIR_INBOARD_M = 0.012
REPAIR_CARRY_HEIGHT_M = 0.06
REPAIR_RELEASE_HEIGHT_M = 0.04
MIN_RELEASE_HEIGHT_M = 0.03
SLACK_M = 0.03  # both tips move this much toward each other during the lift: the near edge is never tensioned by tip drift (measured 10-20 mm)
LATERAL_COMP_M = 0.04  # end targets shifted outward (away from the other arm): the released corner lands 2-4 cm inward of the tip otherwise
TARGET_BACK_M = 0.01  # nominal end target this far short of the far corner (the pinched flap extends ~1 cm beyond the tip)
MAX_PULL_BACK_M = 0.04  # additional pull-back accepted for reachability
CARRY_SPEED_MPS = 0.3
#: hand_v8: the pinch floor ("closed on nothing") is the gripper calibration's held_width_min_m, resolved at run time (pinch_min_width_m=None);
#: one 12 mm layer reads 10-12 mm, an empty close 0.13-0.19 cm in sim.
PINCH_MAX_WIDTH_M = 0.03  # wider = several bunched layers (they extrude during the carry)
RECOVERY_INSET_M = 0.01  # a missed side is re-pinched this much further along the fold
YAW_SNAP_RAD = 0.35  # fold directions within this of the table x axis use yaw 0 (measured: yaw aligned to a skewed fold lost the right pinch)
#: hand_v11: the footprint inset (segment_objects / SAM3 footprint corners lie outside the material corners: sim ~3 mm rendered-cloth bias measured on
#: 5 agent episodes, 15-20 mm on the real placemat) is the cell calibration's ``footprint_inset_m`` (fold_cloth footprint_inset_m=None); no constant here.
APPROACH_M = 0.08
MIN_APPROACH_M = 0.03
RETREAT_M = 0.06
MIN_TIP_SEPARATION_M = 0.12

# ------------------------------------------------------------------- measured finish (candidate cand_20260911_003629_419496)
#: Depth-based measurement of the fold from the top camera (640 x 480, ~3 mm per pixel at the far corners, depth noise sigma up to
#: 4.5 mm per pixel + a per-episode bias of +-2 mm). Heights are read RELATIVE to the table surface seen in the same image (a
#: table patch next to the cloth), so the bias cancels; a 5 x 5 median filter removes the per-pixel salt. One flex layer of the
#: towel reads ~13 mm above the table, two layers ~25 mm.
MEASURE_ONE_LAYER_MIN_M = 0.006  # any cloth (one layer or more)
MEASURE_TWO_LAYER_MIN_M = 0.019  # the moved layer lying ON the fixed layer
MEASURE_ARM_MIN_M = 0.06  # anything this high in the region of interest is an arm / gripper: the measurement is invalid there
MEASURE_BIN_M = 0.004  # occupancy bins along an edge (>= the ~3 mm pixel pitch at the far corners, so bins inside a surface are never empty)
MEASURE_THREE_LAYER_MIN_M = 0.033  # more than two layers near the moved corner: the flap curled over / rolled - its vertex is not at the visible edge
MEASURE_CURL_MIN_PX = 30
MEASURE_ROI_OUTWARD_M = 0.05  # region of interest extends this far outward of the cloth edge (the home-pose arms sit ~9 cm outside)
MEASURE_STRIP_M = (0.012, 0.06)  # edge occupancy is read in a strip this far inside the perpendicular edge (avoids the rounded corner)
#: Resolution the measurement claims for a corner offset: two edges at ~3 mm per pixel, sub-pixel occupancy interpolation, depth
#: noise. Measured against truth on the probe seeds (runs/probe_fold); reported with every offset - the skill never claims an
#: alignment finer than this.
MEASURE_RESOLUTION_M = 0.005
#: A visible cloth edge more than this beyond the pre-fold far edge on a side = the moved layer landed LONG (hangs over the fixed
#: layer's far edge onto the table); otherwise the moved corner is the two-layer region's far corner (landed short, on the layer).
LONG_LANDING_MIN_M = 0.005
#: The finish: after the release the corner offsets are measured and a side farther off than the requested tolerance is nudged
#: by align_corner (pinch on the layer, slide the measured offset, release in place); at most this many rounds.
FINISH_MAX_ROUNDS = 2
FINISH_MAX_OFFSET_M = 0.06  # farther than this is not a nudge: repair_corner (lift and carry) is the right tool
TIGHT_TOLERANCE_M = 0.03  # below this an unmeasurable side is PARTIAL (the pinch evidence cannot vouch for a tight alignment)
#: A two-layer far edge more than this short of the cloth's far edge is not a flat two-layer stack (a completed fold released within 1-2 cm
#: of the far corners cannot be 6 cm short unless a pinch was lost): the side is reported as not measurable instead of a gross offset
MEASURE_MAX_PLAUSIBLE_SHORT_M = 0.06
#: align_corner geometry: the same 12 mm inboard / 20 mm along-the-edge pinch as the fold, pads 3 mm above the fixed layer; the
#: pinched flap is lifted a little so it slides over the layer instead of ploughing it, the slide stops short of the target so the
#: corner never overhangs the fixed layer's edge (an overhanging corner is not "on top": referee rise < 3 mm), and the gripper
#: opens IN PLACE (no drop height: the corner lands where the pads leave it).
#: Probe evidence (runs/probe_fold, seeds 2-5, truth-driven nudges): a slide ON the layer moves nothing - the pinched cloth creeps
#: through the pads against the mu = 4 cloth-on-cloth friction and the corner ends where it was or 1 cm back; a pinch descending to
#: 3 mm over the slack (bulged) top layer bulldozes a fold (18-23 mm, refused); lifting the flap 3 cm and carrying it fixed one corner
#: (16 -> 7 mm) and doubled / dragged the fixed layer in others (the two layers stick). The nudge is therefore a bounded attempt
#: whose outcome the re-measurement reports; it runs only when the caller asks for a tolerance the fold itself cannot meet.
ALIGN_LIFT_M = 0.03
ALIGN_STOP_SHORT_M = 0.004
ALIGN_SLIDE_SPEED_MPS = 0.08
ALIGN_PAD_CLEARANCE_M = 0.005
ALIGN_MAX_WIDTH_M = 0.016  # one layer; wider = several layers (the corner is crumpled or the fixed layer is pinched too)
ALIGN_MIN_SLIDE_M = 0.003
ALIGN_INBOARD_M = 0.006  # pinch this far behind the landed corner along the fold (the corner vertex itself sits between the pads)
ALIGN_EDGE_INSET_M = 0.008  # and this far along the edge into the cloth
ALIGN_MAX_SLIDE_M = 0.08
#: Arms parked here (x toward the robots, own side, 12 cm up) leave the top camera's rays to the cloth free (camera 1.7 m above
#: the robots' side looking along +x: a ray to the far edge passes at z > 1.1 m over x = 0.38).
CLEAR_POSES = ((0.38, 0.38, 0.12), (0.35, 0.35, 0.15), (0.32, 0.40, 0.10), (0.45, 0.40, 0.08))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _fold_geometry(corners_xy: list[list[float]], edge: str, footprint_inset_m: float = 0.0) -> dict[str, Any]:
    """Moving edge corners and their targets from the four footprint corners (any order).

    ``edge='near'`` moves the two corners nearest the robots (smallest x) onto the two far corners; ``'far'`` the reverse.
    The left arm takes the moving corner with the larger y; each moving corner is paired with the target corner on the
    same side (larger / smaller y). ``footprint_inset_m`` moves every corner that much toward the footprint centre along
    each axis (the visible cloth edge lies about half the cloth thickness outside the corner points the fold is scored on).
    Returns per side: corner, target, fold_dir (unit, corner -> target); plus ``inward`` per side (unit vector toward the
    other arm's corner) and the wrist yaw."""

    pts = np.asarray(corners_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise SkillError("corners_xy must hold exactly four [x, y] points", error_type="invalid_params")
    if footprint_inset_m > 0:
        centre = pts.mean(axis=0)
        pts = pts + np.sign(centre - pts) * footprint_inset_m
    order = np.argsort(pts[:, 0])
    moving = pts[order[:2]] if edge == "near" else pts[order[2:]]
    fixed = pts[order[2:]] if edge == "near" else pts[order[:2]]
    left_c, right_c = (moving[0], moving[1]) if moving[0][1] >= moving[1][1] else (moving[1], moving[0])
    left_t, right_t = (fixed[0], fixed[1]) if fixed[0][1] >= fixed[1][1] else (fixed[1], fixed[0])
    geo = _pair_geometry({"left": left_c, "right": right_c}, {"left": left_t, "right": right_t})
    geo["edge"] = edge
    return geo


def _pair_geometry(corner: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, Any]:
    geo: dict[str, Any] = {}
    for s in SIDES:
        c, t = np.asarray(corner[s], float)[:2], np.asarray(target[s], float)[:2]
        d = t - c
        if np.linalg.norm(d) < 0.08:
            raise SkillError(f"{s}: corner and target are {np.linalg.norm(d) * 100:.1f} cm apart (< 8 cm): not an edge-to-edge fold", error_type="invalid_params")
        geo[s] = {"corner": c, "target": t, "fold_dir": _unit(d)}
    across = _unit(corner["right"][:2] - corner["left"][:2])
    geo["inward"] = {"left": across, "right": -across}
    fd = _unit(geo["left"]["fold_dir"] + geo["right"]["fold_dir"])
    angle = math.atan2(float(fd[1]), float(fd[0]))
    geo["yaw"] = 0.0 if abs(angle) < YAW_SNAP_RAD or abs(abs(angle) - math.pi) < YAW_SNAP_RAD else angle
    return geo


def _set_grippers_ramped(ctx, targets: dict[str, float], ramp_s: float, total_s: float) -> None:
    action = ctx.backend.commanded_action()
    start = {s: float(action[6 if s == "left" else 13]) for s in targets}
    n_total = max(1, int(round(total_s * ctx.control_hz)))
    n_ramp = int(round(ramp_s * ctx.control_hz))
    for k in range(n_total):
        f = 1.0 if n_ramp <= 0 else min(1.0, (k + 1) / n_ramp)
        for s, g in targets.items():
            action[6 if s == "left" else 13] = start[s] + (float(g) - start[s]) * f
        ctx.backend.step(action)


def _widths(ctx) -> dict[str, float | None]:
    return {s: ctx.gripper_width(s) for s in SIDES}


def _mm(v: float | None) -> float | None:
    return None if v is None else round(float(v) * 1000, 1)


def _landing_check(ctx, geo: dict[str, Any], far: dict[str, np.ndarray], fd: dict[str, np.ndarray], prompt: str) -> dict[str, Any]:
    """Post-release re-detection of the cloth from the top camera (segment_objects, deployable): footprint area ratio vs the flat
    cloth (edge lengths from the caller's corners) and, per side, how far the detected footprint extends beyond the fold line
    toward the robots (positive = that side's moved corner did not reach the fixed half). None-safe; never raises."""

    out: dict[str, Any] = {"detected": False, "prompt": prompt, "area_ratio": None, "overhang_m": {s: None for s in SIDES}}
    if "segment_objects" not in ctx.library:
        out["reason"] = "segment_objects not in the library"
        return out
    try:
        r = ctx.call("segment_objects", text_prompt=prompt, max_results=4)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"{type(exc).__name__}: {exc}"[:120]
        return out
    # SAM3-verified detections only: an unverified colour / dark-on-light fallback blob is the table or a robot link (area ratio 7-8
    # measured on two seeds where SAM3 missed the folded towel) and would flag a complete fold as partial
    dets = [d for d in (r.outputs or {}).get("detections") or [] if d.get("corners_xy") and d.get("footprint") and d.get("world_xyz")]
    if not dets:
        out["reason"] = f"{r.status.value}: {r.message[:100]}"
        return out
    mid = (np.asarray(geo["left"]["corner"], float) + np.asarray(geo["right"]["corner"], float)) / 2
    centre = mid + (np.asarray(fd["left"], float) + np.asarray(fd["right"], float)) / 2 * 0.5 * float(np.linalg.norm(far["left"] - geo["left"]["corner"]))
    d = min(dets, key=lambda d: float(np.linalg.norm(np.asarray(d["world_xyz"][:2], float) - centre)))
    size = d["footprint"].get("size_xy") or [0.0, 0.0]
    flat_area = float(np.linalg.norm(far["left"] - geo["left"]["corner"]) * np.linalg.norm(np.asarray(geo["right"]["corner"]) - np.asarray(geo["left"]["corner"])))
    ratio = float(size[0]) * float(size[1]) / max(flat_area, 1e-6)
    if not 0.2 <= ratio <= 1.5:
        out["reason"] = f"implausible footprint (area ratio {ratio:.2f}): not the cloth"
        return out
    out.update({"detected": True, "source": d.get("source", "sam3"), "area_ratio": round(ratio, 3), "centre_xy": [round(float(v), 3) for v in d["world_xyz"][:2]], "size_xy": [round(float(v), 3) for v in size]})
    pts = np.asarray(d["corners_xy"], float)
    out["near_corner_xy"] = {s: None for s in SIDES}
    for s in SIDES:
        L = float(np.linalg.norm(far[s] - geo[s]["corner"]))
        fold_pt = np.asarray(geo[s]["corner"], float) + np.asarray(fd[s], float) * (L / 2)
        side_pts = [c for c in pts if float(np.dot(c - mid, geo["inward"][s])) < 0]  # the detected corners on this arm's side
        if side_pts:
            near = min(side_pts, key=lambda c: float(np.dot(c - fold_pt, fd[s])))  # the detected corner nearest the robots on this side
            out["overhang_m"][s] = round(float(np.dot(fold_pt - near, fd[s])), 4)
            out["near_corner_xy"][s] = [round(float(v), 3) for v in near]
    return out


def _landing_bad(landing: dict[str, Any] | None) -> bool:
    if not landing or not landing.get("detected"):
        return False
    if landing["area_ratio"] is not None and landing["area_ratio"] > LANDING_MAX_AREA_RATIO:
        return True
    return any(v is not None and v > LANDING_MAX_OVERHANG_M for v in landing["overhang_m"].values())


def _arms_clear(ctx, geo: dict[str, Any] | None = None, *, sides: tuple[str, ...] = SIDES) -> dict[str, Any]:
    """Park the arms out of the top camera's rays to the cloth (own side, toward the robots, 8-15 cm up: CLEAR_POSES) so the depth
    measurement sees the cloth alone. Grippers stay as they are. Never raises."""

    ctrl = ctx.controller
    tz = float(ctx.table_z)
    targets: dict[str, np.ndarray] = {}
    for s in sides:
        sign = 1.0 if s == "left" else -1.0
        for x, y, h in CLEAR_POSES:
            p = np.array([x, sign * y, tz + h])
            if ctrl.solve_ik(s, p, 0.0) is not None:
                targets[s] = p
                break
    out: dict[str, Any] = {"targets": {s: np.round(v, 3).tolist() for s, v in targets.items()}}
    try:
        if len(targets) == 2:
            r = ctx.call("move_both", left_xyz=targets["left"].tolist(), right_xyz=targets["right"].tolist(), left_yaw=0.0, right_yaw=0.0, correction_passes=0, tolerance_m=0.05)
            out["status"] = r.status.value
        else:
            for s, p in targets.items():
                r = ctx.call("move_to", side=s, xyz=p.tolist(), yaw=0.0, correction_passes=0, tolerance_m=0.05)
                out[s] = r.status.value
    except Exception as exc:  # noqa: BLE001 - parking is best effort; the measurement reports an arm in view
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def _edge_position(coord: np.ndarray, is_mask: np.ndarray, *, bin_m: float = MEASURE_BIN_M, min_px: int = 3, occupancy: float = 0.5) -> float | None:
    """Outer edge of a mask along ``coord`` (metres along one axis, every pixel of a strip; ``is_mask`` flags the cloth pixels):
    the outermost bin at least half occupied whose two inner neighbours are occupied too (isolated noise bins beyond the edge do
    not count), refined inside the bin by its occupancy and by a partly filled next bin. None when no bin is occupied."""

    coord = np.asarray(coord, float).ravel()
    is_mask = np.asarray(is_mask, bool).ravel()
    if coord.size == 0 or not is_mask.any():
        return None
    lo = float(coord.min())
    idx = np.floor((coord - lo) / bin_m).astype(int)
    n = int(idx.max()) + 1
    total = np.bincount(idx, minlength=n).astype(float)
    hit = np.bincount(idx, weights=is_mask.astype(float), minlength=n)
    occ = np.where(total >= min_px, hit / np.maximum(total, 1.0), np.nan)
    filled = [i for i in range(n) if not np.isnan(occ[i])]  # the oblique view leaves a shadow gap (no pixels) behind every step: skip empty bins
    good = np.nan_to_num(occ, nan=0.0) >= occupancy
    for k in range(len(filled) - 1, -1, -1):
        i = filled[k]
        if good[i] and all(good[filled[k - j]] for j in (1, 2) if k - j >= 0):
            nxt = float(np.nan_to_num(occ[i + 1], nan=0.0)) if i + 1 < n else 0.0
            frac = float(np.clip(np.nan_to_num(occ[i], nan=0.0), 0.0, 1.0))
            return lo + (i + frac + (nxt if nxt < occupancy else 0.0)) * bin_m
    return None


def _measure_fold(ctx, geo: dict[str, Any], far: dict[str, np.ndarray], fd: dict[str, np.ndarray], *, mode: str = "post", pre: dict[str, Any] | None = None, profile: bool = False) -> dict[str, Any]:
    """Deployable measurement of an edge-to-edge fold from the top depth camera (arms parked out of view first: ``_arms_clear``).

    Heights are read relative to the table surface seen next to the cloth (the strip the near edge left behind), so the depth bias
    cancels. Per side (in the fold frame of that side: u along the fold direction from the moving corner, v outward across):
    the visible far corner of ANY cloth (>= one layer) and the far corner of the TWO-layer region (the moved layer lying on the
    fixed one). ``pre`` = the same measurement taken before the fold (only the far corner of the flat cloth is used): a cloth
    edge more than LONG_LANDING_MIN_M beyond the pre-fold far edge means the moved layer landed long (over the edge, onto the
    table) and the moved corner is the cloth's far corner; otherwise the moved corner is the two-layer far corner and the target
    is the cloth's far corner (the fixed layer). Returns ``sides[s]`` with moved_xy, target_xy, offset_m, offset_vec_xy
    (target - moved), along_m (positive = landed short), on_layer, long, and per-side / overall validity. Never raises."""

    out: dict[str, Any] = {"valid": False, "sides": {s: {"valid": False} for s in SIDES}, "resolution_m": MEASURE_RESOLUTION_M}
    try:
        depth = np.asarray(ctx.capture_depth("top"), dtype=np.float32)
        K = np.asarray(ctx.backend.camera_intrinsics("top"), float)
        c2w = np.asarray(ctx.backend.camera_to_world("top"), float)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"no top depth: {type(exc).__name__}: {exc}"[:120]
        return out
    if depth.ndim != 2 or not (depth > 0).any():
        out["reason"] = "empty depth image"
        return out
    d = cv2.medianBlur(depth, 3)
    valid = d > 0
    vs, us = np.nonzero(valid)
    pts = deproject(us, vs, d[valid], K, c2w)
    xy, z = pts[:, :2], pts[:, 2]
    cl, cr = np.asarray(geo["left"]["corner"], float), np.asarray(geo["right"]["corner"], float)
    mid = (cl + cr) / 2
    width = float(np.linalg.norm(cr - cl))
    f_mean = _unit(np.asarray(fd["left"], float) + np.asarray(fd["right"], float))
    across = _unit(cr - cl)
    u_all = (xy - mid) @ f_mean
    v_all = (xy - mid) @ across
    table_sel = (u_all > -0.09) & (u_all < -0.025) & (np.abs(v_all) < width / 2 - 0.02)
    if int(table_sel.sum()) >= 40:
        table_ref = float(np.median(z[table_sel]))
        out["table_reference"] = {"z": round(table_ref, 4), "n": int(table_sel.sum()), "bias_vs_table_z_m": round(table_ref - float(ctx.table_z), 4)}
    else:
        table_ref = float(ctx.table_z)
        out["table_reference"] = {"z": round(table_ref, 4), "n": int(table_sel.sum()), "fallback": "ctx.table_z"}
    h = z - table_ref
    n_valid = 0
    for s in SIDES:
        rec: dict[str, Any] = {"valid": False}
        out["sides"][s] = rec
        c = np.asarray(geo[s]["corner"], float)
        f = np.asarray(fd[s], float)
        n_out = -np.asarray(geo["inward"][s], float)
        L = float(np.linalg.norm(np.asarray(far[s], float) - c))
        u = (xy - c) @ f
        v = (xy - c) @ n_out
        roi = (u > 0.35 * L) & (u < L + 0.10) & (v > -(width / 2 - 0.015)) & (v < MEASURE_ROI_OUTWARD_M)
        rec["n_roi"] = int(roi.sum())
        if rec["n_roi"] < 200:
            rec["reason"] = "region of interest not in view"
            continue
        arm = roi & (h > MEASURE_ARM_MIN_M)
        if int(arm.sum()) > 25:
            axy = xy[arm].mean(axis=0)
            rec["reason"] = f"an arm is in the top camera's view over this side ({int(arm.sum())} px above {MEASURE_ARM_MIN_M * 100:.0f} cm around ({axy[0]:.2f}, {axy[1]:.2f}))"
            rec["arm_xy"] = [round(float(v), 3) for v in axy]
            rec["arm_h_max_m"] = round(float(h[arm].max()), 3)
            continue
        cloth = roi & (h > MEASURE_ONE_LAYER_MIN_M) & (h < MEASURE_ARM_MIN_M)
        top = roi & (h > MEASURE_TWO_LAYER_MIN_M) & (h < MEASURE_ARM_MIN_M)
        rec["n_cloth"], rec["n_top"] = int(cloth.sum()), int(top.sum())
        strip = roi & (u > L - 0.09) & (u < L - 0.02)
        v_cloth = _edge_position(v[strip], cloth[strip])
        if v_cloth is None:
            rec["reason"] = "no cloth edge found near the far end of this side"
            continue
        strip = roi & (v < v_cloth - MEASURE_STRIP_M[0]) & (v > v_cloth - MEASURE_STRIP_M[1])
        u_cloth = _edge_position(u[strip], cloth[strip])
        u_top = _edge_position(u[strip], top[strip])
        if u_cloth is None:
            rec["reason"] = "no far cloth edge found on this side"
            continue
        far_corner = c + f * u_cloth + n_out * v_cloth
        rec.update({"far_corner_xy": [round(float(x), 4) for x in far_corner], "u_cloth_m": round(u_cloth, 4), "v_cloth_m": round(v_cloth, 4), "u_top_m": None if u_top is None else round(u_top, 4)})
        if profile:
            edges = np.arange(L - 0.08, L + 0.04, MEASURE_BIN_M)
            prof = []
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                sel = strip & (u >= lo_) & (u < hi_)
                prof.append({"u_mm": round(float(lo_ - L) * 1000), "n": int(sel.sum()), "h_med_mm": None if not sel.any() else round(float(np.median(h[sel])) * 1000, 1), "h_p80_mm": None if not sel.any() else round(float(np.percentile(h[sel], 80)) * 1000, 1), "cloth": None if not sel.any() else round(float(cloth[sel].mean()), 2), "top": None if not sel.any() else round(float(top[sel].mean()), 2)})
            rec["profile_u"] = prof
        if mode == "pre":
            rec["valid"] = True  # pre-fold call: the far corner of the flat cloth is all that is needed
            n_valid += 1
            continue
        pre_side = ((pre or {}).get("sides") or {}).get(s) or {}
        pre_u = pre_side.get("u_cloth_m") if pre_side.get("valid") else None
        pre_v = pre_side.get("v_cloth_m") if pre_side.get("valid") else None
        long_by = None if pre_u is None else u_cloth - float(pre_u)
        out_by = None if pre_v is None else v_cloth - float(pre_v)
        rec["long_by_m"] = None if long_by is None else round(long_by, 4)
        rec["outward_by_m"] = None if out_by is None else round(out_by, 4)
        # a flap that curled over / rolled at the corner stacks more than two layers there: its vertex is not at any visible edge
        thick = roi & (h > MEASURE_THREE_LAYER_MIN_M) & (h < MEASURE_ARM_MIN_M) & (u > L - 0.08) & (v > v_cloth - 0.06)
        rec["n_thick"] = int(thick.sum())
        if rec["n_thick"] >= MEASURE_CURL_MIN_PX:
            rec["curled"] = True
            rec["reason"] = f"more than two layers near the moved corner ({rec['n_thick']} px above {MEASURE_THREE_LAYER_MIN_M * 1000:.0f} mm): the flap curled over, its corner is not at the visible edge - offset not measurable"
            rec.update({"target_xy": [round(float(x), 4) for x in far_corner]})
            continue
        rec["curled"] = False
        # per axis: where the top layer overhangs the fixed layer's edge (landed long / outward) the fixed edge is hidden and the pre-fold
        # edge (same measurement, same bias; the fixed half may have been dragged ~1 cm) is the target; otherwise the visible edges decide
        long = long_by is not None and long_by > LONG_LANDING_MIN_M
        outward = out_by is not None and out_by > LONG_LANDING_MIN_M
        if u_top is None and not long:
            rec["reason"] = "no two-layer region on this side: the moved corner is not on the fixed layer"
            rec.update({"long": False, "on_layer": False, "target_xy": [round(float(x), 4) for x in far_corner]})
            continue
        if u_top is not None and not long and u_cloth - float(u_top) > MEASURE_MAX_PLAUSIBLE_SHORT_M:
            rec["reason"] = f"the two-layer edge lies {(u_cloth - float(u_top)) * 100:.0f} cm short of the cloth edge: not a flat two-layer stack there, offset not measurable"
            rec.update({"long": False, "on_layer": None, "target_xy": [round(float(x), 4) for x in far_corner], "u_top_m": round(float(u_top), 4)})
            continue
        v_top = None
        if u_top is not None:
            strip2 = roi & (u < u_top - MEASURE_STRIP_M[0]) & (u > u_top - MEASURE_STRIP_M[1])
            v_top = _edge_position(v[strip2], top[strip2])
        moved_u = u_cloth if long else float(u_top)
        target_u = float(pre_u) if long else u_cloth
        moved_v = v_cloth if (outward or v_top is None) else float(v_top)
        target_v = float(pre_v) if outward else v_cloth
        moved = c + f * moved_u + n_out * moved_v
        target = c + f * target_u + n_out * target_v
        rec.update({"long": bool(long), "outward": bool(outward), "on_layer": not long, "v_top_m": None if v_top is None else round(float(v_top), 4)})
        vec = target - moved
        rec.update({"valid": True, "moved_xy": [round(float(x), 4) for x in moved], "target_xy": [round(float(x), 4) for x in target], "offset_m": round(float(np.linalg.norm(vec)), 4), "offset_vec_xy": [round(float(x), 4) for x in vec], "along_m": round(float(vec @ f), 4), "across_m": round(float(vec @ n_out), 4)})
        n_valid += 1
    out["valid"] = n_valid == len(SIDES)
    out["n_valid_sides"] = n_valid
    return out


def _offset_text(m: dict[str, Any] | None, tol: float | None) -> str:
    """One clause with the measured corner offsets (mm) and the measurement's resolution."""

    if not m:
        return ""
    parts = []
    for s in SIDES:
        r = (m.get("sides") or {}).get(s) or {}
        if r.get("valid"):
            parts.append(f"{s} {r['offset_m'] * 1000:.0f} mm ({'long' if r.get('long') else 'short'} {abs(r['along_m']) * 1000:.0f}, across {r['across_m'] * 1000:+.0f})")
        else:
            parts.append(f"{s} not measurable ({r.get('reason', 'no depth')})")
    res = f"; resolution +-{m.get('resolution_m', MEASURE_RESOLUTION_M) * 1000:.0f} mm" + (f", tolerance {tol * 1000:.0f} mm" if tol is not None else "")
    return " Measured corner offsets (top depth camera): " + ", ".join(parts) + res + "."


def _landing_text(landing: dict[str, Any] | None) -> str:
    if not landing:
        return ""
    if not landing.get("detected"):
        return f" Top-camera re-detection of the cloth failed ({landing.get('reason', '')})."
    parts = [f"footprint area ratio {landing['area_ratio']:.2f} (complete fold ~0.5-0.6)"]
    for s, v in landing["overhang_m"].items():
        if v is not None and v > 0.02:
            parts.append(f"cloth extends {v * 100:.0f} cm beyond the fold line on the {s}")
    return " Top-camera re-detection: " + ", ".join(parts) + "."


class _FoldCore:
    """Shared implementation; the skills only translate their parameter contracts into ``execute``."""

    def __init__(self, skill: Skill) -> None:
        self.skill = skill

    def result(self, status, message, **kw):
        return self.skill.result(status, message, verified=True, **kw)

    def execute(self, ctx, geo: dict[str, Any], *, pad_clearance_m: float, inboard_m: float, edge_inset_m: float, carry_height_m: float, release_height_m: float, slack_m: float, lateral_comp_m: float, target_back_m: float, max_pull_back_m: float, carry_speed_mps: float, pinch_min_width_m: float, retry_pinch: bool, landing_prompt: str | None = None, corner_tolerance_m: float | None = None, finish_rounds: int = FINISH_MAX_ROUNDS):
        tick0 = ctx.backend.control_step_index
        ctrl = ctx.controller
        pz = float(ctx.table_z + ctx.cell_calibration.finger_pad_below_tip_m + pad_clearance_m)  # hand_v11: calibrated pad offset
        yaw = float(geo["yaw"])
        P = {s: np.array([*(geo[s]["corner"] + geo[s]["fold_dir"] * inboard_m + geo["inward"][s] * edge_inset_m), pz]) for s in SIDES}
        far = {s: np.asarray(geo[s]["target"], float) for s in SIDES}
        fd = {s: geo[s]["fold_dir"] for s in SIDES}
        metrics: dict[str, Any] = {"stage": "precondition", "pinch_points": {s: np.round(P[s], 4).tolist() for s in SIDES}, "yaw": round(yaw, 3), "sides": {s: {} for s in SIDES}}
        if float(np.linalg.norm(P["left"] - P["right"])) < MIN_TIP_SEPARATION_M:
            raise SkillError("the two pinch points are closer than 12 cm: the grippers would collide", error_type="precondition")
        # wrist branches, chosen once for the whole fold (a branch flip mid-path would twist the pinch): every combination of
        # the branches on which each pinch column is reachable is planned end to end BEFORE any motion, and the combination
        # with the least pull-back, then the highest release height, then the largest lateral compensation wins
        branches: dict[str, list[float]] = {}
        for s in SIDES:
            branches[s] = [b for b in (yaw, yaw + math.pi, yaw - math.pi) if ctrl.solve_ik(s, P[s], b, symmetric=False) is not None and ctrl.solve_ik(s, P[s] + [0, 0, LIFT_M], b, symmetric=False) is not None]
            if not branches[s]:
                return self.result(SkillStatus.failure, f"{s} pinch point {np.round(P[s][:2], 3).tolist()} is not reachable top-down (pads 3 mm above the table)", metrics={**metrics, "stage": "reach", "reachable": False, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        # the path plans below start from the pinch pose, which the arms are not in yet: plan_cartesian_path_both takes the
        # current tips / joints as its origin, so the plan is made from a virtual start (the pinch pose's IK solution)
        plan = self._plan_fold(ctrl, P, far, fd, geo, branches, pz, slack_m=slack_m, lateral_comp_m=lateral_comp_m, target_back_m=target_back_m, max_pull_back_m=max_pull_back_m, release_height_m=release_height_m, carry_height_m=carry_height_m)
        if plan is None:
            return self.result(SkillStatus.failure, "no reachable fold path: the far corners are beyond the top-down reach even with a 4 cm pull-back; nothing moved", metrics={**metrics, "stage": "plan", "reachable_targets": False, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        yaws = plan["yaws"]
        metrics["branch_yaw"] = {s: round(yaws[s], 3) for s in SIDES}
        metrics["plan"] = {k: plan[k] for k in ("lateral_comp_m", "release_height_m", "pull_back_m", "carry_height_m", "slack_used_m", "plans_tried")}
        # pre-fold depth measurement of the flat cloth's far corners (the fixed layer's corners after the fold): the finish compares the
        # post-fold cloth edge with it to tell a corner that landed long (over the edge) from one that landed short (on the layer)
        pre = None
        if corner_tolerance_m is not None:
            pre = _measure_fold(ctx, geo, far, fd, mode="pre")
            metrics["measurement_pre"] = {"valid": pre.get("valid"), "sides": {s: {k: v for k, v in (pre["sides"].get(s) or {}).items() if k in ("valid", "reason", "far_corner_xy", "u_cloth_m")} for s in SIDES}, "table_reference": pre.get("table_reference")}
        both = dict(left_yaw=yaws["left"], right_yaw=yaws["right"], strict_yaw=True)
        # approach above the pinch points (highest reachable height <= APPROACH_M), grippers open
        h_app = APPROACH_M
        while h_app > MIN_APPROACH_M and any(ctrl.solve_ik(s, P[s] + [0, 0, h_app], yaws[s], symmetric=False) is None for s in SIDES):
            h_app = round(h_app - 0.01, 3)
        metrics["stage"] = "approach"
        r = ctx.call("move_both", left_xyz=(P["left"] + [0, 0, h_app]).tolist(), right_xyz=(P["right"] + [0, 0, h_app]).tolist(), left_gripper=1.0, right_gripper=1.0, **both)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["stage"] = "pinch"
        r = ctx.call("move_both", left_xyz=P["left"].tolist(), right_xyz=P["right"].tolist(), tolerance_m=0.015, correction_passes=1, **both)
        metrics["descent_tip_error_m"] = r.metrics.get("tip_error_m")
        _set_grippers_ramped(ctx, {s: 0.0 for s in SIDES}, CLOSE_RAMP_S, CLOSE_HOLD_S)
        widths = _widths(ctx)
        if pinch_min_width_m is None:
            pinch_min_width_m = max(_held_width_min(ctx, s) for s in SIDES)
        metrics["pinch_min_width_m"] = pinch_min_width_m
        bad = lambda w: w is None or w < pinch_min_width_m or w > PINCH_MAX_WIDTH_M  # noqa: E731
        recovered: list[str] = []
        for s in [s for s in SIDES if bad(widths[s])]:
            metrics["sides"][s]["first_pinch_width_m"] = widths[s]
            if not retry_pinch:
                continue
            # one-sided recovery while the other arm holds: re-pinch 1 cm further along the fold (undisturbed flat cloth)
            _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
            ctx.call("move_to", side=s, xyz=(P[s] + [0, 0, 0.03]).tolist(), yaw=yaws[s], strict_yaw=True)
            P[s] = P[s] + np.array([fd[s][0], fd[s][1], 0.0]) * RECOVERY_INSET_M
            ctx.call("move_to", side=s, xyz=P[s].tolist(), yaw=yaws[s], tolerance_m=0.015, correction_passes=1, strict_yaw=True)
            _set_grippers_ramped(ctx, {s: 0.0}, CLOSE_RAMP_S, CLOSE_HOLD_S)
            widths[s] = ctx.gripper_width(s)
            if not bad(widths[s]):
                recovered.append(s)
        for s in SIDES:
            metrics["sides"][s]["pinch_width_m"] = widths[s]
        metrics["recovered_sides"] = recovered
        still = [s for s in SIDES if bad(widths[s])]
        if still:
            self._release_and_retreat(ctx, P, h_app, both)
            why = "bunched several layers" if any(widths[s] is not None and widths[s] > PINCH_MAX_WIDTH_M for s in still) else "missed"
            return self.result(SkillStatus.failure, f"pinch {why} on {still} (openings {[_mm(widths[s]) for s in SIDES]} mm); cloth released, arms retreated", metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        # the pinch points may have moved by the recovery inset: re-plan the path from the actual pinch pose (same branches)
        metrics["stage"] = "plan"
        if recovered:
            plan = self._plan_fold(ctrl, P, far, fd, geo, {s: [yaws[s]] for s in SIDES}, pz, slack_m=slack_m, lateral_comp_m=lateral_comp_m, target_back_m=target_back_m, max_pull_back_m=max_pull_back_m, release_height_m=release_height_m, carry_height_m=carry_height_m)
            if plan is None:
                self._release_and_retreat(ctx, P, h_app, both)
                return self.result(SkillStatus.failure, "no reachable fold path from the re-pinched points; cloth released where it was pinched", metrics={**metrics, "reachable_targets": False, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["slack_used_m"] = plan["slack_used_m"]
        metrics["path"] = {k: (v if not isinstance(v, dict) else {s: np.round(v[s], 4).tolist() for s in SIDES}) for k, v in plan.items() if k not in ("waypoints", "knots", "yaws", "slack_used_m")}
        metrics["stage"] = "carry"
        lost: list[str] = []
        lost_at: dict[str, dict[str, Any]] = {}

        def progress(s: str) -> float:
            """Fold progress of a tip [m]: its displacement from the pinch point along the fold direction."""

            return float(np.dot(ctx.tip(s)[:2] - P[s][:2], fd[s]))

        def monitor(fraction, state):
            # a side that opens to < LOST_WIDTH_M has lost the cloth: stop (never drag a half-held cloth) unless the path is
            # in its last part (the arms are descending onto the target: the dropped flap lands within a few cm of it)
            for s in SIDES:
                w = ctx.gripper_width(s)
                if w is not None and w < pinch_min_width_m and s not in lost:
                    lost.append(s)
                    lost_at[s] = {"fraction": round(float(fraction), 3), "progress_m": round(progress(s), 4), "tip": np.round(ctx.tip(s), 4).tolist()}
                    if fraction < LATE_COMPLETE_FRACTION:
                        return f"{s} pinch lost at path fraction {fraction:.2f}"
            return None

        res = ctrl.move_cartesian_path_both(plan["waypoints"], yaws, speed_mps=carry_speed_mps, monitor=monitor, symmetric=False, settle=False, fallback_knots=plan["knots"])
        early_recovered: list[str] = []
        if not res["completed"] and len(lost) == 1 and retry_pinch and lost_at[lost[0]]["progress_m"] < EARLY_RECOVERY_PROGRESS_M:
            # EARLY loss: the flap slipped out during the lift, before the cloth moved along the fold (measured: one seed in ten
            # loses a 10 mm pinch in the first 2 % of the path). The cloth still lies flat, so the side is re-pinched once,
            # 1 cm further along the fold, while the other arm keeps its corner (a few cm up), and the path is re-planned from
            # the actual pose of both arms.
            s = lost[0]
            w = None
            # two attempts: 1 cm further along the fold, then 2 cm further along the fold and 1 cm further inward (the slipped flap
            # may have flipped or moved a little; the first re-pinch closed on nothing in one probe seed)
            for k, (along, inward) in enumerate(((RECOVERY_INSET_M, 0.0), (2 * RECOVERY_INSET_M, RECOVERY_INSET_M))):
                _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
                ctx.call("move_to", side=s, xyz=(P[s] + [0, 0, 0.03]).tolist(), yaw=yaws[s], correction_passes=0, strict_yaw=True)
                P[s] = P[s] + np.array([*(fd[s] * along + geo["inward"][s] * inward), 0.0])
                if ctrl.solve_ik(s, P[s], yaws[s], symmetric=False) is None:
                    break
                ctx.call("move_to", side=s, xyz=P[s].tolist(), yaw=yaws[s], tolerance_m=0.015, correction_passes=1, strict_yaw=True)
                _set_grippers_ramped(ctx, {s: 0.0}, CLOSE_RAMP_S, CLOSE_HOLD_S)
                w = ctx.gripper_width(s)
                metrics["sides"][s][f"repinch_width_m_{k + 1}"] = w
                if not bad(w) and w <= RECOVERY_MAX_WIDTH_M:
                    break
            metrics["sides"][s]["repinch_width_m"] = w
            if not bad(w) and w <= RECOVERY_MAX_WIDTH_M:
                plan2 = self._plan_fold(ctrl, P, far, fd, geo, {t: [yaws[t]] for t in SIDES}, pz, slack_m=slack_m, lateral_comp_m=lateral_comp_m, target_back_m=target_back_m, max_pull_back_m=max_pull_back_m, release_height_m=release_height_m, carry_height_m=carry_height_m, from_current=True)
                if plan2 is not None:
                    early_recovered.append(s)
                    lost.clear()
                    lost_at.clear()
                    plan = plan2
                    metrics["path"] = {k: (v if not isinstance(v, dict) else {t: np.round(v[t], 4).tolist() for t in SIDES}) for k, v in plan.items() if k not in ("waypoints", "knots", "yaws", "slack_used_m")}
                    res = ctrl.move_cartesian_path_both(plan["waypoints"], yaws, speed_mps=carry_speed_mps, monitor=monitor, symmetric=False, settle=False, fallback_knots=plan["knots"])
        metrics["early_recovered"] = early_recovered
        finished_alone: dict[str, Any] | None = None
        if not res["completed"] and lost:
            # MID-CARRY loss: the lost side opens and lifts clear; the HELD side carries its corner on to its target alone
            # (hand_v6 released both where they were, which left both corners in the middle of a heap). The result is a clean
            # half fold the caller can see and finish; the dropped corner lies near the lost tip's xy.
            s_lost = lost[-1]
            s_held = OTHER[s_lost]
            _set_grippers_ramped(ctx, {s_lost: 1.0}, 0.0, 0.3)
            tip_l = ctx.tip(s_lost)
            for up in (RETREAT_M, 0.03):
                if ctrl.solve_ik(s_lost, tip_l + [0, 0, up], yaws[s_lost], symmetric=False) is not None:
                    ctx.call("move_to", side=s_lost, xyz=(tip_l + [0, 0, up]).tolist(), yaw=yaws[s_lost], correction_passes=0, strict_yaw=True)
                    break
            w_held = ctx.gripper_width(s_held)
            finished_alone = {"side": s_held, "completed": False, "held_at_start": bool(w_held is not None and w_held >= pinch_min_width_m)}
            if finished_alone["held_at_start"] and s_held not in lost:
                wps = self._remaining_waypoints(plan, s_held, ctx.tip(s_held), P[s_held], fd[s_held])
                one_yaw = {s_held: yaws[s_held]}
                if ctrl.plan_cartesian_path_both(wps, one_yaw, symmetric=False)["feasible"]:
                    res_one = ctrl.move_cartesian_path_both(wps, one_yaw, speed_mps=carry_speed_mps, monitor=monitor, symmetric=False, settle=False)
                    finished_alone.update({"completed": bool(res_one["completed"]), "aborted_reason": res_one["aborted_reason"], "n_waypoints": len(wps)})
                else:
                    finished_alone["aborted_reason"] = "no reachable single-arm path to the target"
        widths_end = _widths(ctx)
        for s in SIDES:
            metrics["sides"][s]["width_before_release_m"] = widths_end[s]
        metrics["path"].update({"completed": res["completed"], "plan": res.get("plan"), "duration_s": res["duration_s"], "max_relation_drift_m": res.get("max_relation_drift_m"), "tip_error_m": {s: round(float(v), 4) for s, v in res["tip_error_m"].items()}, "lost_sides": list(lost), "lost_at": lost_at, "finished_alone": finished_alone})
        # release right away (no settle: a hold lets the pinch creep) and retreat
        _set_grippers_ramped(ctx, {s: 1.0 for s in SIDES}, 0.0, 0.4)
        tips = {s: ctx.tip(s) for s in SIDES}
        for up in (RETREAT_M, 0.03):  # the release pose is often at the reach edge: retreat as far as both arms can
            if all(ctrl.solve_ik(s, tips[s] + [0, 0, up], yaws[s], symmetric=False) is not None for s in SIDES):
                ctx.call("move_both", left_xyz=(tips["left"] + [0, 0, up]).tolist(), right_xyz=(tips["right"] + [0, 0, up]).tolist(), correction_passes=0, **both)
                break
        metrics["lost_sides"] = list(lost)
        held = all(widths_end[s] is not None and widths_end[s] >= pinch_min_width_m for s in SIDES)
        targets = {s: np.round(far[s], 3).tolist() for s in SIDES}
        # post-release landing check from the top camera (deployable). The pinch evidence decides first: a corner that dropped ON
        # the fixed layer is invisible from above; the re-detection measures gross shortfalls and confirms complete folds.
        landing = _landing_check(ctx, geo, far, fd, landing_prompt) if landing_prompt else None
        metrics["landing_check"] = landing
        # measured finish (corner_tolerance_m given, path completed): park the arms, measure both moved corners from the top depth camera,
        # nudge every side farther off than the tolerance with align_corner, re-measure (at most finish_rounds rounds)
        finish = None
        meas = None
        if corner_tolerance_m is not None and res["completed"]:
            finish = self._finish(ctx, geo, far, fd, pre, corner_tolerance_m, finish_rounds)
            meas = finish["final"]
            metrics["finish"] = {k: v for k, v in finish.items() if k != "final"}
            metrics["measurement"] = meas
        metrics["robot_time_s"] = _robot_time(ctx, tick0)
        measured = _landing_text(landing) + _offset_text(meas, corner_tolerance_m)
        meas_ok = bool(meas and meas.get("valid"))
        within = bool(meas_ok and all(meas["sides"][s]["offset_m"] <= corner_tolerance_m for s in SIDES))
        if corner_tolerance_m is not None and res["completed"] and not meas_ok:
            measured += " The corner alignment is therefore NOT verified beyond the pinch evidence."
        L = {s: float(np.linalg.norm(far[s] - geo[s]["corner"])) for s in SIDES}
        def flap_corner_for(s: str, tip_xy) -> list[float]:
            # the pinch sits inboard_m along the fold and edge_inset_m along the edge inside the corner: the corner of a dropped or
            # released flap lies that far BEHIND the tip along the fold and outward along the edge (measured pinch geometry)
            return [round(float(v), 3) for v in (np.asarray(tip_xy[:2], float) - fd[s] * inboard_m - geo["inward"][s] * edge_inset_m)]

        dropped: dict[str, list[float]] = {s_: flap_corner_for(s_, lost_at[s_]["tip"]) for s_ in lost_at}
        outputs: dict[str, Any] = {"held": bool(res["completed"] and held), "release_points": {s: np.round(tips[s], 4).tolist() for s in SIDES}, "pinch_points": metrics["pinch_points"], "targets_xy": targets, "edge": geo.get("edge"), "lost_sides": list(lost),
                                   "dropped_corner_xy": dropped, "landing_check": landing, "next_action": None, "next_actions": [],
                                   "corner_offset_mm": {s: (round(meas["sides"][s]["offset_m"] * 1000, 1) if meas_ok or (meas and meas["sides"][s].get("valid")) else None) for s in SIDES},
                                   "measurement_resolution_mm": round(MEASURE_RESOLUTION_M * 1000, 1), "corner_tolerance_mm": None if corner_tolerance_m is None else round(corner_tolerance_m * 1000, 1),
                                   "finish_nudges": 0 if not finish else sum(len(r["actions"]) for r in finish["rounds"])}

        def repair(s: str, corner_xy, on_layer: bool) -> dict[str, Any]:
            return {"skill": "repair_corner", "params": {"side": s, "corner_xy": [round(float(v), 3) for v in corner_xy], "target_xy": targets[s], "edge_inward_xy": [round(float(v), 3) for v in geo["inward"][s]], "on_layer": bool(on_layer)}}

        def released_corner(s: str):
            return flap_corner_for(s, tips[s])

        def measured_actions() -> list[dict[str, Any]]:
            """Ready calls for the sides the measurement shows off tolerance: align_corner (<= 6 cm) or repair_corner."""

            acts: list[dict[str, Any]] = []
            for s in SIDES:
                r = ((meas or {}).get("sides") or {}).get(s) or {}
                if r.get("valid"):
                    if r["offset_m"] <= corner_tolerance_m:
                        continue
                    if r["offset_m"] <= FINISH_MAX_OFFSET_M:
                        acts.append({"skill": "align_corner", "params": {"side": s, "corner_xy": [round(float(v), 3) for v in r["moved_xy"]], "target_xy": [round(float(v), 3) for v in r["target_xy"]], "edge_inward_xy": [round(float(v), 3) for v in geo["inward"][s]], "on_layer": bool(r.get("on_layer", True))}})
                    else:
                        acts.append(repair(s, r["moved_xy"], bool(r.get("on_layer", True))))
                elif str(r.get("reason", "")).startswith("no two-layer"):
                    acts.append(repair(s, released_corner(s), False))  # the moved corner is not on the fixed layer: lift and carry it
            return acts

        def partial(msg: str, actions: list[dict[str, Any]], outcome: str = "partial"):
            metrics["outcome"] = outcome
            outputs["next_actions"] = actions
            outputs["next_action"] = actions[0] if actions else None
            what = "" if not actions else (" - it pinches the top layer 12 mm inside that corner and slides it by the measured offset, releasing in place; re-measure once afterwards (fold_cloth is not to be called again on this footprint)." if actions[0]["skill"] == "align_corner" else " - it pinches the top layer 12 mm inside that corner and carries it onto its target; re-observe once afterwards. Do not call fold_cloth again on this deformed footprint and do not re-grasp a landed corner by hand.")
            hint = "" if not actions else f" Next: {actions[0]['skill']}({', '.join(f'{k}={v}' for k, v in actions[0]['params'].items())})" + (" then the same for the other side" if len(actions) > 1 else "") + what
            return self.result(SkillStatus.failure, "PARTIAL fold: " + msg + measured + hint, metrics=metrics, outputs=outputs)

        def complete(msg: str):
            metrics["outcome"] = "complete"
            tail = (f" Both moved corners measured within the {corner_tolerance_m * 1000:.0f} mm tolerance (resolution +-{MEASURE_RESOLUTION_M * 1000:.0f} mm)." if within else "")
            return self.result(SkillStatus.success, msg + measured + tail + " The fold is complete: verify with ONE observation and finish; do not re-grasp or nudge a landed corner by hand.", metrics=metrics, outputs=outputs)

        if corner_tolerance_m is not None and corner_tolerance_m < TIGHT_TOLERANCE_M and res["completed"] and not meas_ok and not (lost and lost_at[lost[-1]]["fraction"] < LATE_COMPLETE_FRACTION):
            # a tight tolerance cannot be vouched for by the pinch evidence alone (the fold lands 5-30 mm off): an unmeasurable side is a
            # PARTIAL result, never a success claim beyond what was measured
            return partial(f"the fold completed but a moved corner could not be measured to the {corner_tolerance_m * 1000:.0f} mm tolerance.", measured_actions())

        if lost:
            s_lost = lost[-1]
            frac = lost_at[s_lost]["fraction"]
            drop = outputs["dropped_corner_xy"][s_lost]
            on_layer = lost_at[s_lost]["progress_m"] >= 0.5 * L[s_lost]  # dropped beyond the fold line = on top of the fixed half
            # a slipping flap is displaced from its dead-reckoned drop point: when the top camera sees cloth sticking out beyond the fold
            # line on that side, the detected footprint corner there is the measured position of the dropped corner (on the table)
            seen = (landing or {}).get("near_corner_xy", {}).get(s_lost) if landing and landing.get("detected") else None
            if seen is not None and (landing["overhang_m"].get(s_lost) or 0.0) > 0.02:
                drop = seen
                on_layer = False
                outputs["dropped_corner_xy"][s_lost] = drop
                outputs["dropped_corner_source"] = "top camera"
            mismatch = float(np.linalg.norm(np.asarray(drop) - far[s_lost][:2]))
            action = repair(s_lost, drop, on_layer)
            done = finished_alone or {}
            if not res["completed"]:
                other = f"the {OTHER[s_lost]} corner was carried on to its target {targets[OTHER[s_lost]]} and released" if done.get("completed") else f"the {OTHER[s_lost]} corner was released where it was ({'its pinch was lost too' if OTHER[s_lost] in lost else done.get('aborted_reason') or 'no single-arm path'})"
                return partial(f"{s_lost} pinch lost at {frac * 100:.0f} % of the carry (soft-friction creep of the corner-flap pinch): the {s_lost} corner dropped near {drop}, about {mismatch * 100:.0f} cm from its target {targets[s_lost]} ({'on top of the fixed layer' if on_layer else 'on the table, short of the fold line'}); {other}.", [action])
            # a slip in the final descent (>= LATE_COMPLETE_FRACTION): the flap drops onto the target region from a few cm; every probe
            # fold of this kind met the referee (9/9, corners within 16-42 mm). Success only when the top camera confirms a complete
            # footprint; otherwise PARTIAL with the repair action (hand_v6 called every such fold a failure and the agent's repairs gathered the cloth).
            slip_msg = f"both corners carried {plan['path_length_m'] * 100:.0f} cm; the {s_lost} pinch slipped at {frac * 100:.0f} % of the carry, in the final descent, and its corner dropped near {drop}; the other corner was released on its target."
            if meas_ok:
                if within:
                    return complete(slip_msg)
                return partial(f"the {s_lost} pinch slipped at {frac * 100:.0f} % of the carry (final descent) and the moved corners still measure outside the tolerance after the finish.", measured_actions() or [action])
            if landing is not None and landing.get("detected") and not _landing_bad(landing):
                return complete(slip_msg + f" Its corner lies within a few cm of its target {targets[s_lost]}.")
            return partial(f"the {s_lost} pinch slipped at {frac * 100:.0f} % of the carry (final descent) and its corner dropped near {drop}, ~{mismatch * 100:.0f} cm from its target {targets[s_lost]}; the landing is not confirmed from the top camera.", [action])
        if not res["completed"]:
            metrics["outcome"] = "aborted"
            return self.result(SkillStatus.failure, f"fold path aborted: {res['aborted_reason']}; cloth released where it was (partially moved).{measured} Re-observe with segment_objects before the next action.", metrics=metrics, outputs=outputs)
        if not held:
            bad_sides = [s for s in SIDES if widths_end[s] is None or widths_end[s] < pinch_min_width_m]
            metrics["lost_sides"] = bad_sides
            outputs["lost_sides"] = bad_sides
            outputs["dropped_corner_xy"] = {s: [round(float(v), 3) for v in released_corner(s)] for s in bad_sides}
            drop_msg = f"pinch lost on {bad_sides} at the release (openings {[_mm(widths_end[s]) for s in SIDES]} mm): that corner lies near {outputs['dropped_corner_xy'][bad_sides[0]]}; the other corner was carried."
            if meas_ok:
                return complete(drop_msg) if within else partial(drop_msg + " The moved corners measure outside the tolerance after the finish.", measured_actions() or [repair(s, released_corner(s), True) for s in bad_sides])
            return partial(drop_msg, [repair(s, released_corner(s), True) for s in bad_sides])
        if plan["pull_back_m"] >= PARTIAL_PULL_BACK_M - 1e-9:
            # the far corners are beyond the top-down reach: the corners were released short and land 5-7 cm from their targets
            back_msg = f"both corners were released {plan['pull_back_m'] * 100:.0f} cm short of the far corners because those lie beyond the top-down reach"
            if meas_ok:
                return complete(back_msg + ".") if within else partial(back_msg + "; the moved corners measure outside the tolerance after the finish.", measured_actions() or [repair(s, released_corner(s), True) for s in SIDES])
            return partial(back_msg + f"; the moved corners now lie about {(plan['pull_back_m'] + 0.02) * 100:.0f} cm short of their targets on top of the fixed layer (referee tolerance 5 cm).", [repair(s, released_corner(s), True) for s in SIDES])
        if meas_ok and not within:
            return partial(f"both pinches held to the release, but after {outputs['finish_nudges']} nudge(s) the moved corners still measure outside the {corner_tolerance_m * 1000:.0f} mm tolerance.", measured_actions())
        if _landing_bad(landing) and not meas_ok:
            worst = [s for s in SIDES if landing["overhang_m"].get(s) is not None and landing["overhang_m"][s] > LANDING_MAX_OVERHANG_M] or list(SIDES)
            return partial("both pinches held to the release but the top camera does not show a complete fold.", [repair(s, released_corner(s), False) for s in worst])
        msg = f"both corners carried {plan['path_length_m'] * 100:.0f} cm and released {plan['release_height_m'] * 100:.0f} cm above the pinch height" + (f" ({plan['pull_back_m'] * 100:.0f} cm short of the far corners for reach)" if plan["pull_back_m"] else "") + (f"; the {early_recovered[0]} corner was re-pinched once after slipping in the lift" if early_recovered else "") + (f"; {outputs['finish_nudges']} corner nudge(s) by align_corner" if outputs["finish_nudges"] else "") + "."
        return complete(msg)

    @staticmethod
    def _finish(ctx, geo, far, fd, pre, tol: float, rounds: int) -> dict[str, Any]:
        """Measured finish: park both arms, measure the moved corners from the top depth camera, nudge every side measured farther
        off than ``tol`` (and within align_corner's range) with align_corner, park that arm again, re-measure; at most ``rounds``
        rounds. Returns {"clear", "initial", "rounds": [{actions, after}], "final": measurement}. Never raises."""

        log: dict[str, Any] = {"tolerance_m": tol, "rounds": []}
        log["clear"] = _arms_clear(ctx, geo)
        m = _measure_fold(ctx, geo, far, fd, pre=pre)
        log["initial"] = {s: {k: (m["sides"][s] or {}).get(k) for k in ("valid", "offset_m", "along_m", "across_m", "long", "reason")} for s in SIDES}
        for _ in range(max(0, int(rounds))):
            todo = [s for s in SIDES if (m["sides"].get(s) or {}).get("valid") and tol < m["sides"][s]["offset_m"] <= FINISH_MAX_OFFSET_M]
            if not todo or "align_corner" not in ctx.library:
                break
            acts = []
            for s in todo:
                r = m["sides"][s]
                f = np.asarray(fd[s], float)
                # the corner should end ALIGN_STOP_SHORT_M short of the target ALONG THE FOLD (never over the fixed layer's far edge),
                # whichever direction the nudge goes; the pinch point is 12 mm toward the target from the corner - over the fixed
                # layer unless the corner hangs over its edge by more than that
                target_eff = np.asarray(r["target_xy"], float) - f * ALIGN_STOP_SHORT_M
                long_by = float(r.get("long_by_m") or 0.0)
                on_layer = (not r.get("long")) or long_by <= 0.008  # the pinch sits 12 mm behind the corner: over the fixed layer unless the overhang is longer
                try:
                    res = ctx.call("align_corner", side=s, corner_xy=[float(v) for v in r["moved_xy"]], target_xy=[float(v) for v in target_eff], edge_inward_xy=[float(v) for v in geo["inward"][s]], fold_dir_xy=[float(v) for v in f], on_layer=bool(on_layer), stop_short_m=0.0)
                    acts.append({"side": s, "before_m": r["offset_m"], "status": res.status.value, "message": res.message[:160], "pinch_width_m": res.metrics.get("pinch_width_m"), "slide_m": res.metrics.get("slide_m"), "on_layer": bool(on_layer)})
                except Exception as exc:  # noqa: BLE001 - a refused nudge (params / precondition) is recorded, the fold result stays honest
                    acts.append({"side": s, "before_m": r["offset_m"], "status": "error", "message": f"{type(exc).__name__}: {exc}"[:160]})
            _arms_clear(ctx, geo, sides=tuple(todo))
            m = _measure_fold(ctx, geo, far, fd, pre=pre)
            log["rounds"].append({"actions": acts, "after": {s: (m["sides"].get(s) or {}).get("offset_m") for s in SIDES}})
        log["final"] = m
        return log

    @staticmethod
    def _remaining_waypoints(plan, side: str, tip, P_side, fd_side) -> list[dict[str, np.ndarray]]:
        """The planned waypoints of ``side`` that still lie ahead of its tip (> 2 cm further along the fold), always ending at
        its end target: the single-arm continuation after the other side lost its pinch."""

        p_tip = float(np.dot(np.asarray(tip[:2]) - P_side[:2], fd_side))
        ahead = [wp for wp in plan["waypoints"][:-1] if float(np.dot(np.asarray(wp[side][:2]) - P_side[:2], fd_side)) > p_tip + 0.02]
        return [{side: np.asarray(wp[side], float)} for wp in ahead] + [{side: np.asarray(plan["waypoints"][-1][side], float)}]

    @staticmethod
    def _plan_fold(ctrl, P, far, fd, geo, branches, pz, *, slack_m, lateral_comp_m, target_back_m, max_pull_back_m, release_height_m, carry_height_m, max_plans: int = 40, from_current: bool = False):
        """End pose and continuous path for every branch combination, planned from the pinch pose (virtual start = the
        pinch IK solutions; ``from_current``: from the arms' actual tips and joints, for a re-plan after an early re-pinch).
        Candidates are ranked by (pull-back, -release height, -lateral compensation); the search of one combination stops at
        its first feasible plan in that order, and the whole search stops early on a perfect plan."""

        heights = [round(release_height_m - 0.01 * k, 3) for k in range(int(round((release_height_m - MIN_RELEASE_HEIGHT_M) / 0.01)) + 1)]
        backs = [b for b in (0.0, 0.02, 0.04, 0.06) if b <= max_pull_back_m + 1e-9]
        laterals = sorted({round(lateral_comp_m, 4), round(max(0.0, lateral_comp_m - 0.01), 4), round(max(0.0, lateral_comp_m - 0.02), 4), 0.0}, reverse=True)
        # one cost in metres of landing error: the pull-back lands the corner short 1:1; every centimetre of release height
        # below 5 cm costs about as much (the pads squeeze the flap out on the pile, measured); a side with less than 1.5 cm
        # of slack tensions the near edge (creep) and is charged 2 cm; less lateral compensation lands the corner inward
        candidates = sorted(((back + max(0.0, 0.05 - h_end) + 0.5 * (lateral_comp_m - lateral), back, h_end, lateral) for back in backs for h_end in heights for lateral in laterals), key=lambda c: (round(c[0], 4), -c[2], -c[3], c[1]))
        best, best_cost = None, None
        tried = 0
        for bl in branches["left"]:
            for br in branches["right"]:
                yaws = {"left": float(bl), "right": float(br)}
                start_q = None if from_current else {s: ctrl.solve_ik(s, P[s], yaws[s], symmetric=False) for s in SIDES}
                if start_q is not None and any(q is None for q in start_q.values()):
                    continue
                slack_used = {}
                for s in SIDES:
                    sl = slack_m
                    while sl > 1e-9 and ctrl.solve_ik(s, P[s] + np.array([*(geo["inward"][s] * sl), 0.0]) + [0, 0, LIFT_M], yaws[s], symmetric=False) is None:
                        sl = round(sl - 0.005, 4)
                    slack_used[s] = max(0.0, sl)
                slack_penalty = 0.02 if min(slack_used.values()) < 0.015 else 0.0
                Ps = {s: P[s] + np.array([*(geo["inward"][s] * slack_used[s]), 0.0]) for s in SIDES}
                for cost, back, h_end, lateral in candidates:
                    if best_cost is not None and cost + slack_penalty >= best_cost:
                        break  # this branch pair cannot beat the plan already found
                    end = {s: np.array([*(far[s] - fd[s] * (target_back_m + back) + geo["inward"][s] * (slack_used[s] - lateral)), pz + h_end]) for s in SIDES}
                    if any(ctrl.solve_ik(s, end[s], yaws[s], symmetric=False) is None for s in SIDES):
                        continue
                    found = None
                    for h_c in sorted({carry_height_m, round(max(h_end, min(carry_height_m, h_end + 0.02)), 3)}, reverse=True):
                        if tried >= max_plans:
                            break
                        wps = [{s: Ps[s] + [0, 0, LIFT_M] for s in SIDES}]
                        for frac in (0.25, 0.75):
                            wps.append({s: np.array([*(Ps[s][:2] + (end[s][:2] - Ps[s][:2]) * frac), pz + h_c]) for s in SIDES})
                        wps.append(end)
                        tried += 1
                        pl = ctrl.plan_cartesian_path_both(wps, yaws, symmetric=False, starts=None if from_current else P, start_qpos=start_q)
                        if pl["feasible"]:
                            found = {"waypoints": wps, "knots": pl["knots"], "end": end, "yaws": yaws, "slack_used_m": slack_used, "lateral_comp_m": lateral, "release_height_m": h_end, "pull_back_m": back, "carry_height_m": h_c, "path_length_m": pl["path_length_m"], "cost_m": round(cost + slack_penalty, 4)}
                            break
                    if found is not None:
                        best, best_cost = found, cost + slack_penalty
                        break
                if best_cost is not None and best_cost <= 1e-9:
                    break
            if best_cost is not None and best_cost <= 1e-9:
                break
        if best is not None:
            best["plans_tried"] = tried
        return best

    @staticmethod
    def _release_and_retreat(ctx, P, h, both) -> None:
        _set_grippers_ramped(ctx, {s: 1.0 for s in SIDES}, 0.0, 0.3)
        ctx.call("move_both", left_xyz=(P["left"] + [0, 0, h]).tolist(), right_xyz=(P["right"] + [0, 0, h]).tolist(), correction_passes=0, **both)


class FoldCloth(Skill):
    name = "fold_cloth"
    version = "1.2.1"
    kind = "code"
    description = ("Folds a flat cloth edge onto the opposite edge with both arms from its four footprint corners (segment_objects corners_xy), then MEASURES "
                   "both moved corners (top depth camera, +-5 mm) against `corner_tolerance_m`; `nudge` = align_corner on off-tolerance corners. "
                   "Lost pinch, short release, corners off or unmeasurable = PARTIAL (next_actions).")
    returns = "outputs.corner_offset_mm {left, right} (null = unmeasurable), measurement_resolution_mm, finish_nudges; success only within tolerance; PARTIAL: next_actions"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "openings; path completion; depth-measured corner offsets"
    dependencies = ["move_both", "move_to", "segment_objects", "align_corner"]
    known_limitations = []
    observations_used = ["proprio", "rgb:top", "depth:top"]
    timeout_s = 70.0

    class Params(Params):
        corners_xy: list[list[float]] = Field(min_length=4, max_length=4, description="four [x, y] corners [m]")
        edge: Literal["near", "far"] = Field(default="near")
        cloth_prompt: str = Field(default="towel", min_length=2, max_length=60)
        corner_tolerance_m: float = Field(default=0.05, ge=0.005, le=0.10, description="task tolerance [m]")
        nudge: bool = Field(default=False, description="align_corner on corners off tolerance")
        release_height_m: float = Field(default=RELEASE_HEIGHT_M, ge=0.03, le=0.10)
        carry_height_m: float = Field(default=CARRY_HEIGHT_M, ge=0.04, le=0.15)
        slack_m: float = Field(default=SLACK_M, ge=0.0, le=0.05)
        lateral_comp_m: float = Field(default=LATERAL_COMP_M, ge=0.0, le=0.06)
        carry_speed_mps: float = Field(default=CARRY_SPEED_MPS, ge=0.1, le=0.5)
        pad_clearance_m: float = Field(default=PAD_CLEARANCE_M, ge=0.0, le=0.005)
        footprint_inset_m: float | None = Field(default=None, ge=0.0, le=0.02, description="default: calibration")
        retry_pinch: bool = Field(default=True)
        verify_landing: bool = Field(default=True)

    def run(self, ctx, p):
        inset = float(ctx.cell_calibration.footprint_inset_m) if p.footprint_inset_m is None else float(p.footprint_inset_m)  # hand_v11: the cell's SAM3 footprint bias
        geo = _fold_geometry(p.corners_xy, p.edge, inset)
        # the release target stays 1 cm short of the far corners for every tolerance: releasing ON the far corners for a tight tolerance was tried
        # (oracle validation, first pass) and tensioned the cloth on the longer path - pinches lost at 68-97 % of the carry in 3 of 5 seeds
        return _FoldCore(self).execute(ctx, geo, pad_clearance_m=p.pad_clearance_m, inboard_m=GRASP_INBOARD_M, edge_inset_m=EDGE_INSET_M, carry_height_m=p.carry_height_m, release_height_m=p.release_height_m, slack_m=p.slack_m, lateral_comp_m=p.lateral_comp_m, target_back_m=TARGET_BACK_M, max_pull_back_m=MAX_PULL_BACK_M, carry_speed_mps=p.carry_speed_mps, pinch_min_width_m=None, retry_pinch=p.retry_pinch, landing_prompt=p.cloth_prompt if p.verify_landing else None, corner_tolerance_m=p.corner_tolerance_m, finish_rounds=FINISH_MAX_ROUNDS if p.nudge else 0)


class RepairCorner(Skill):
    name = "repair_corner"
    version = "1.0.2"
    kind = "code"
    description = ("Move ONE cloth corner onto its target with one arm - the repair after a PARTIAL fold_cloth (use its outputs.next_action as is): the corner at "
                   "`corner_xy` is pinched 12 mm inside / 20 mm along the edge, pads just above the fixed layer (`on_layer`, 12 mm) or the table, one layer only "
                   "(opening from the calibrated held threshold up to 16 mm; a crumpled corner is released untouched), on ONE continuous path (lift 3, carry 6, release 4 cm up) to 1 cm short of `target_xy`. Re-observe once afterwards.")
    returns = "success if the pinch held to the release near target_xy; failure: unreachable, pinched nothing / several layers (released), pinch lost (outputs.dropped_corner_xy). metrics: pinch_width_m, lost_fraction; outputs: release_point"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "openings after closing and before release; path completion"
    dependencies = ["move_to"]
    known_limitations = ["a crumpled corner (several layers) is released untouched"]
    observations_used = ["proprio"]
    timeout_s = 25.0

    class Params(Params):
        side: Side
        corner_xy: list[float] = Field(min_length=2, max_length=2, description="where the corner lies [m]")
        target_xy: list[float] = Field(min_length=2, max_length=2, description="where it belongs [m]")
        edge_inward_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="unit vector along the cloth edge from the corner into the cloth (fold_cloth supplies it)")
        on_layer: bool = Field(default=True, description="corner lies on the folded layer (pads 12 mm up)")
        carry_height_m: float = Field(default=REPAIR_CARRY_HEIGHT_M, ge=0.04, le=0.12)
        release_height_m: float = Field(default=REPAIR_RELEASE_HEIGHT_M, ge=0.02, le=0.08)
        carry_speed_mps: float = Field(default=CARRY_SPEED_MPS, ge=0.1, le=0.5)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        ctrl = ctx.controller
        s = p.side
        c = np.asarray(p.corner_xy, float)
        t = np.asarray(p.target_xy, float)
        d = t - c
        dist = float(np.linalg.norm(d))
        if dist < 0.03:
            raise SkillError(f"corner and target are {dist * 100:.1f} cm apart: nothing to repair", error_type="invalid_params")
        u = d / dist
        angle = math.atan2(float(u[1]), float(u[0]))
        yaw0 = 0.0 if abs(angle) < YAW_SNAP_RAD or abs(abs(angle) - math.pi) < YAW_SNAP_RAD else angle
        pz = float(ctx.table_z + ctx.cell_calibration.finger_pad_below_tip_m + PAD_CLEARANCE_M + (LAYER_THICKNESS_M if p.on_layer else 0.0))
        # the same pinch geometry as fold_cloth: 12 mm inside the corner along the direction of travel and 20 mm along the cloth edge
        # (the second flex vertex between the pads); without the edge direction the pinch sits on the corner flap itself
        inward = np.zeros(2) if p.edge_inward_xy is None else _unit(np.asarray(p.edge_inward_xy, float))
        P = np.array([*(c + u * REPAIR_INBOARD_M + inward * EDGE_INSET_M), pz])
        end = np.array([*(t - u * TARGET_BACK_M), pz + p.release_height_m])
        metrics: dict[str, Any] = {"stage": "reach", "pinch_point": np.round(P, 4).tolist(), "end": np.round(end, 4).tolist()}
        yaw = None
        for b in (yaw0, yaw0 + math.pi, yaw0 - math.pi):
            if ctrl.solve_ik(s, P, b, symmetric=False) is not None and ctrl.solve_ik(s, P + [0, 0, LIFT_M], b, symmetric=False) is not None and ctrl.solve_ik(s, end, b, symmetric=False) is not None:
                yaw = float(b)
                break
        if yaw is None:
            return self.result(SkillStatus.failure, f"the {s} arm cannot reach the pinch point {np.round(P[:2], 3).tolist()} and the release point {np.round(end[:2], 3).tolist()} top-down on one wrist branch; use the other arm or a lower release_height_m", verified=True, metrics={**metrics, "reachable": False, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["yaw"] = round(yaw, 3)
        pinch_min = _held_width_min(ctx, s)  # calibrated: below it the pads closed on nothing
        metrics["pinch_min_width_m"] = pinch_min
        h_app = APPROACH_M
        while h_app > MIN_APPROACH_M and ctrl.solve_ik(s, P + [0, 0, h_app], yaw, symmetric=False) is None:
            h_app = round(h_app - 0.01, 3)
        metrics["stage"] = "approach"
        r = ctx.call("move_to", side=s, xyz=(P + [0, 0, h_app]).tolist(), yaw=yaw, gripper=1.0, correction_passes=0, strict_yaw=True)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["stage"] = "pinch"
        widths = []
        for k in range(2):  # one re-pinch 1 cm further along the direction of the target when the pads closed on nothing
            r = ctx.call("move_to", side=s, xyz=P.tolist(), yaw=yaw, tolerance_m=0.015, correction_passes=1, strict_yaw=True)
            _set_grippers_ramped(ctx, {s: 0.0}, CLOSE_RAMP_S, CLOSE_HOLD_S)
            w = ctx.gripper_width(s)
            widths.append(_mm(w))
            if w is not None and pinch_min <= w <= RECOVERY_MAX_WIDTH_M:
                break
            _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
            if w is not None and w > RECOVERY_MAX_WIDTH_M:
                break  # several layers: the corner is crumpled; carrying it gathered the cloth into an 80 mm heap in the probe, a second attempt would gather more
            ctx.call("move_to", side=s, xyz=(P + [0, 0, 0.03]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
            P = P + np.array([*(u * RECOVERY_INSET_M), 0.0])
        metrics["pinch_widths_mm"] = widths
        w = ctx.gripper_width(s)
        last_mm = widths[-1] if widths else None
        if w is None or not (pinch_min <= w <= RECOVERY_MAX_WIDTH_M):
            _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
            ctx.call("move_to", side=s, xyz=(P + [0, 0, h_app]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
            why = "several layers (a crumpled or folded-under corner; carrying it would gather the cloth)" if last_mm is not None and last_mm > RECOVERY_MAX_WIDTH_M * 1000 else "nothing (no cloth edge to lift there)"
            return self.result(SkillStatus.failure, f"the {s} pinch closed on {why} at {np.round(P[:2], 3).tolist()} (openings {widths} mm); released, arm retreated. Re-observe the corner's position (segment_objects) before another attempt", verified=True, metrics={**metrics, "pinch_width_m": None if last_mm is None else last_mm / 1000, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["pinch_width_m"] = w
        metrics["stage"] = "carry"
        mid = np.array([*(P[:2] + (end[:2] - P[:2]) * 0.5), pz + p.carry_height_m])
        wps = [{s: P + [0, 0, LIFT_M]}, {s: mid}, {s: end}]
        yaws = {s: yaw}
        if not ctrl.plan_cartesian_path_both(wps, yaws, symmetric=False)["feasible"]:
            wps = [{s: P + [0, 0, LIFT_M]}, {s: end}]
            if not ctrl.plan_cartesian_path_both(wps, yaws, symmetric=False)["feasible"]:
                _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
                ctx.call("move_to", side=s, xyz=(P + [0, 0, h_app]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
                return self.result(SkillStatus.failure, "no reachable single-arm path from the pinch to the release point; corner released where it was", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        lost_at: dict[str, Any] = {}

        def monitor(fraction, state):
            wm = ctx.gripper_width(s)
            if wm is not None and wm < pinch_min and not lost_at:
                lost_at.update({"fraction": round(float(fraction), 3), "tip": np.round(ctx.tip(s), 4).tolist()})
                if fraction < LATE_COMPLETE_FRACTION:
                    return f"{s} pinch lost at path fraction {fraction:.2f}"
            return None

        res = ctrl.move_cartesian_path_both(wps, yaws, speed_mps=p.carry_speed_mps, monitor=monitor, symmetric=False, settle=False)
        w_end = ctx.gripper_width(s)
        _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.4)
        tip = ctx.tip(s)
        for up in (RETREAT_M, 0.03):
            if ctrl.solve_ik(s, tip + [0, 0, up], yaw, symmetric=False) is not None:
                ctx.call("move_to", side=s, xyz=(tip + [0, 0, up]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
                break
        metrics.update({"path_completed": bool(res["completed"]), "width_before_release_m": w_end, "lost_fraction": lost_at.get("fraction"), "robot_time_s": _robot_time(ctx, tick0)})
        outputs = {"held": bool(res["completed"] and w_end is not None and w_end >= pinch_min), "release_point": np.round(tip, 4).tolist(), "target_xy": [round(float(v), 3) for v in t]}
        if lost_at and not res["completed"]:
            outputs["dropped_corner_xy"] = [round(float(v), 3) for v in lost_at["tip"][:2]]
            return self.result(SkillStatus.failure, f"the {s} pinch was lost at {lost_at['fraction'] * 100:.0f} % of the repair carry; the corner dropped near {outputs['dropped_corner_xy']}, released. Re-observe before another attempt", verified=True, metrics=metrics, outputs=outputs)
        if not res["completed"]:
            return self.result(SkillStatus.failure, f"repair path aborted: {res['aborted_reason']}; corner released where it was", verified=True, metrics=metrics, outputs=outputs)
        if not outputs["held"]:
            return self.result(SkillStatus.failure, f"the {s} pinch slipped before the release (opening {_mm(w_end)} mm); the corner lies near {np.round(tip[:2], 3).tolist()}, within a few cm of {outputs['target_xy']}. Re-observe once", verified=True, metrics=metrics, outputs=outputs)
        return self.result(SkillStatus.success, f"the {s} corner was carried {dist * 100:.0f} cm and released {p.release_height_m * 100:.0f} cm above the layer, 1 cm short of {outputs['target_xy']}. Verify with ONE observation and finish; do not re-grasp landed corners", verified=True, metrics=metrics, outputs=outputs)


class AlignCorner(Skill):
    name = "align_corner"
    version = "1.0.1"
    kind = "code"
    description = ("Nudge ONE landed cloth corner by a measured offset with one arm (fold_cloth's finishing move; a remaining call is in its outputs.next_actions): pinch the "
                   "top-layer flap just inside `corner_xy` (ONE layer only - wider = crumpled, released untouched), lift, carry so the corner stops `stop_short_m` before "
                   "`target_xy` (offsets 3-80 mm), lower, open IN PLACE. Offsets > 6 cm: repair_corner. Re-measure afterwards.")
    returns = ("success if the pinch held to the release (outputs.corner_xy_estimate dead-reckoned, slide_m); failure: unreachable, pinched nothing / several layers "
               "(nothing moved), pinch lost (outputs.dropped_corner_xy). metrics: pinch_width_m, slide_m")
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "openings after closing, during the carry and before the release; path completion"
    dependencies = ["move_to"]
    known_limitations = []
    observations_used = ["proprio"]
    timeout_s = 20.0

    class Params(Params):
        side: Side
        corner_xy: list[float] = Field(min_length=2, max_length=2, description="where the corner lies [m]")
        target_xy: list[float] = Field(min_length=2, max_length=2, description="where it belongs [m]")
        edge_inward_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="unit, along the edge into the cloth")
        fold_dir_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="unit, near -> far edge")
        on_layer: bool = Field(default=True)
        stop_short_m: float = Field(default=ALIGN_STOP_SHORT_M, ge=0.0, le=0.02)
        lift_m: float = Field(default=ALIGN_LIFT_M, ge=0.0, le=0.03)
        pad_clearance_m: float = Field(default=ALIGN_PAD_CLEARANCE_M, ge=0.0, le=0.012)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        ctrl = ctx.controller
        s = p.side
        c = np.asarray(p.corner_xy, float)
        t = np.asarray(p.target_xy, float)
        d = t - c
        dist = float(np.linalg.norm(d))
        if dist < ALIGN_MIN_SLIDE_M:
            raise SkillError(f"corner and target are {dist * 1000:.0f} mm apart: nothing to align", error_type="invalid_params")
        if dist > ALIGN_MAX_SLIDE_M:
            raise SkillError(f"corner and target are {dist * 100:.1f} cm apart: not a nudge - use repair_corner", error_type="invalid_params")
        u = d / dist
        slide = max(0.0, dist - p.stop_short_m)
        # the top layer extends from its landed corner AGAINST the fold direction: the pinch goes 12 mm that way (into the flap), never toward
        # the target (a corner that landed short has only the fixed layer beyond it); the pads close along the fold so the slide is normal to them
        f = u if p.fold_dir_xy is None else _unit(np.asarray(p.fold_dir_xy, float))
        angle = math.atan2(float(f[1]), float(f[0]))
        yaw0 = 0.0 if abs(angle) < YAW_SNAP_RAD or abs(abs(angle) - math.pi) < YAW_SNAP_RAD else angle
        pz = float(ctx.table_z + ctx.cell_calibration.finger_pad_below_tip_m + p.pad_clearance_m + (LAYER_THICKNESS_M if p.on_layer else 0.0))
        inward = np.zeros(2) if p.edge_inward_xy is None else _unit(np.asarray(p.edge_inward_xy, float))
        inboard = ALIGN_INBOARD_M
        inset = ALIGN_EDGE_INSET_M
        P = np.array([*(c - f * inboard + inward * inset), pz])
        end = P + np.array([*(u * slide), 0.0])
        metrics: dict[str, Any] = {"stage": "reach", "pinch_point": np.round(P, 4).tolist(), "end": np.round(end, 4).tolist(), "requested_slide_m": round(slide, 4)}
        yaw = None
        for b in (yaw0, yaw0 + math.pi, yaw0 - math.pi):
            if all(ctrl.solve_ik(s, q, b, symmetric=False) is not None for q in (P, P + [0, 0, p.lift_m], end, end + [0, 0, p.lift_m])):
                yaw = float(b)
                break
        if yaw is None:
            return self.result(SkillStatus.failure, f"the {s} arm cannot reach the pinch point {np.round(P[:2], 3).tolist()} and the end point {np.round(end[:2], 3).tolist()} top-down on one wrist branch; use the other arm", verified=True, metrics={**metrics, "reachable": False, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["yaw"] = round(yaw, 3)
        pinch_min = _held_width_min(ctx, s)
        metrics["pinch_min_width_m"] = pinch_min
        h_app = APPROACH_M
        while h_app > MIN_APPROACH_M and ctrl.solve_ik(s, P + [0, 0, h_app], yaw, symmetric=False) is None:
            h_app = round(h_app - 0.01, 3)
        metrics["stage"] = "approach"
        r = ctx.call("move_to", side=s, xyz=(P + [0, 0, h_app]).tolist(), yaw=yaw, gripper=1.0, correction_passes=0, strict_yaw=True)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["stage"] = "pinch"
        widths: list[float | None] = []
        w = None
        for k in range(2):  # one re-pinch 1 cm further along the slide direction when the pads closed on nothing
            ctx.call("move_to", side=s, xyz=P.tolist(), yaw=yaw, tolerance_m=0.012, correction_passes=1, strict_yaw=True)
            _set_grippers_ramped(ctx, {s: 0.0}, CLOSE_RAMP_S, CLOSE_HOLD_S)
            w = ctx.gripper_width(s)
            widths.append(_mm(w))
            if w is not None and pinch_min <= w <= ALIGN_MAX_WIDTH_M:
                break
            _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
            if w is not None and w > ALIGN_MAX_WIDTH_M:
                break  # several layers: a crumpled corner or the fixed layer pinched too - sliding it would gather the cloth
            if k == 0:
                ctx.call("move_to", side=s, xyz=(P + [0, 0, 0.03]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
                P = P - np.array([*(f * RECOVERY_INSET_M), 0.0])  # 1 cm deeper into the flap
                inboard += RECOVERY_INSET_M
        metrics["pinch_widths_mm"] = widths
        metrics["inboard_m"] = round(inboard, 4)
        if w is None or not (pinch_min <= w <= ALIGN_MAX_WIDTH_M):
            ctx.call("move_to", side=s, xyz=(P + [0, 0, h_app]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
            why = "several layers (a crumpled corner, or the fixed layer as well)" if w is not None and w > ALIGN_MAX_WIDTH_M else "nothing (no cloth edge there)"
            return self.result(SkillStatus.failure, f"the {s} pinch closed on {why} at {np.round(P[:2], 3).tolist()} (openings {widths} mm); released untouched, arm retreated. Re-measure the corner before another attempt", verified=True, metrics={**metrics, "pinch_width_m": w, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["pinch_width_m"] = w
        # the corner ends stop_short_m before the target: the pinch point travels so that corner = P_end - u*inboard - inward*inset
        end = np.array([*(t - u * p.stop_short_m - f * inboard + inward * inset), pz])
        slide = float(np.linalg.norm(end[:2] - P[:2]))
        metrics["slide_m"] = round(slide, 4)
        metrics["stage"] = "slide"
        lost: dict[str, Any] = {}

        def monitor(fraction, state):
            wm = ctx.gripper_width(s)
            if wm is not None and wm < pinch_min and not lost:
                lost.update({"fraction": round(float(fraction), 3), "tip": np.round(ctx.tip(s), 4).tolist()})
                return f"{s} pinch lost at slide fraction {fraction:.2f}"
            return None

        wps = [{s: P + [0, 0, p.lift_m]}, {s: end + [0, 0, p.lift_m]}, {s: end}]
        yaws = {s: yaw}
        if not ctrl.plan_cartesian_path_both(wps, yaws, symmetric=False)["feasible"]:
            _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)
            ctx.call("move_to", side=s, xyz=(P + [0, 0, h_app]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
            return self.result(SkillStatus.failure, "no reachable slide path from the pinch point to the end point; corner released where it was", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        res = ctrl.move_cartesian_path_both(wps, yaws, speed_mps=ALIGN_SLIDE_SPEED_MPS, monitor=monitor, symmetric=False, settle=False)
        w_end = ctx.gripper_width(s)
        tip_end = ctx.tip(s)
        _set_grippers_ramped(ctx, {s: 1.0}, 0.0, 0.3)  # open in place: the corner lands where the pads leave it
        for up in (RETREAT_M, 0.03):
            if ctrl.solve_ik(s, tip_end + [0, 0, up], yaw, symmetric=False) is not None:
                ctx.call("move_to", side=s, xyz=(tip_end + [0, 0, up]).tolist(), yaw=yaw, correction_passes=0, strict_yaw=True)
                break
        estimate = [round(float(v), 4) for v in (tip_end[:2] + f * inboard - inward * inset)]
        metrics.update({"path_completed": bool(res["completed"]), "width_before_release_m": w_end, "lost": lost or None, "tip_end": np.round(tip_end, 4).tolist(), "robot_time_s": _robot_time(ctx, tick0)})
        held = bool(res["completed"] and w_end is not None and w_end >= pinch_min)
        outputs: dict[str, Any] = {"held": held, "corner_xy_estimate": estimate, "slide_m": metrics["slide_m"], "target_xy": [round(float(v), 4) for v in t]}
        if lost:
            outputs["dropped_corner_xy"] = [round(float(v), 4) for v in (np.asarray(lost["tip"][:2]) + f * inboard - inward * inset)]
            return self.result(SkillStatus.failure, f"the {s} pinch was lost at {lost['fraction'] * 100:.0f} % of the {slide * 1000:.0f} mm slide; the corner lies near {outputs['dropped_corner_xy']}. Re-measure before another attempt", verified=True, metrics=metrics, outputs=outputs)
        if not res["completed"]:
            return self.result(SkillStatus.failure, f"slide aborted: {res['aborted_reason']}; corner released near {estimate}", verified=True, metrics=metrics, outputs=outputs)
        if not held:
            return self.result(SkillStatus.failure, f"the {s} pinch slipped before the release (opening {_mm(w_end)} mm); the corner lies near {estimate}. Re-measure", verified=True, metrics=metrics, outputs=outputs)
        return self.result(SkillStatus.success, f"the {s} corner was slid {slide * 1000:.0f} mm and released in place; dead-reckoned corner {estimate} ({p.stop_short_m * 1000:.0f} mm short of the target). Re-measure once", verified=True, metrics=metrics, outputs=outputs)


class BimanualFold(Skill):
    name = "bimanual_fold"
    version = "2.1.2"
    kind = "code"
    description = ("Two-point cloth fold, the 1.2.0 contract on the fold_cloth 1.1 mechanics: the left arm pinches the corner at `left_xyz`, the right at `right_xyz` "
                   "(x, y only; pads 3 mm above the table, 12 mm inside the corner, 20 mm along the edge, ramped close, one re-pinch per side) and both are carried on ONE "
                   "continuous path (carry 8 cm, release 5 cm up) onto `left_target_xy` / `right_target_xy`. Prefer fold_cloth (four footprint corners).")
    returns = "as fold_cloth: success when both pinches held to the release; failure names the stage and sides. metrics: stage, sides, path, robot_time_s; outputs: held"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "calibrated openings after closing and before release; path completion"
    dependencies = ["move_both", "move_to"]
    known_limitations = ["targets beyond x ~0.75 m land up to 4 cm short"]
    timeout_s = 40.0

    class Params(Params):
        left_xyz: list[float] = Field(min_length=3, max_length=3, description="z ignored")
        right_xyz: list[float] = Field(min_length=3, max_length=3)
        left_target_xy: list[float] = Field(min_length=2, max_length=2)
        right_target_xy: list[float] = Field(min_length=2, max_length=2)
        yaw: float = Field(default=0.0, description="ignored")
        lift_height: float = Field(default=CARRY_HEIGHT_M, ge=0.03, le=0.2, description="carry height [m]")
        release_height: float = Field(default=RELEASE_HEIGHT_M, ge=0.01, le=0.08, description="[m] above the pinch")
        pinch_min_width_m: float | None = Field(default=None, ge=0.0, le=0.02, description="default: calibration held_width_min_m")
        pad_clearance_m: float = Field(default=PAD_CLEARANCE_M, ge=-0.003, le=0.006)
        slack_m: float = Field(default=SLACK_M, ge=0.0, le=0.05)
        centre_inset_m: float = Field(default=EDGE_INSET_M, ge=0.0, le=0.03)
        carry_speed_mps: float = Field(default=CARRY_SPEED_MPS, ge=0.02, le=0.5)
        gather_m: float = Field(default=0.0, ge=0.0, le=0.03, description="ignored")

    def run(self, ctx, p):
        geo = _pair_geometry({"left": np.asarray(p.left_xyz[:2], float), "right": np.asarray(p.right_xyz[:2], float)}, {"left": np.asarray(p.left_target_xy, float), "right": np.asarray(p.right_target_xy, float)})
        return _FoldCore(self).execute(ctx, geo, pad_clearance_m=float(np.clip(p.pad_clearance_m, 0.0, 0.005)), inboard_m=GRASP_INBOARD_M, edge_inset_m=p.centre_inset_m, carry_height_m=float(np.clip(p.lift_height, 0.04, 0.15)), release_height_m=float(np.clip(p.release_height, MIN_RELEASE_HEIGHT_M, 0.10)), slack_m=p.slack_m, lateral_comp_m=LATERAL_COMP_M, target_back_m=TARGET_BACK_M, max_pull_back_m=MAX_PULL_BACK_M, carry_speed_mps=max(0.1, p.carry_speed_mps), pinch_min_width_m=p.pinch_min_width_m, retry_pinch=True)
