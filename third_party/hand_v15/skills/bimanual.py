"""Bimanual coordination skills (library ``hand_v6``: the hand_v5 file, unchanged - hand_v2 lineage; hand_v4's bimanual_fold lift check
(cand_20260909_214510_0ef7c7) merged with the cloth-capable revision of cand_20260909_221511_026ccd, proposal P8). In hand_v6 the
``BimanualFold`` 1.2.0 of this file is DISABLED in the manifest: ``bimanual_fold`` 2.0.0 (same contract) and the new ``fold_cloth``
live in skills/cloth_fold.py (candidate cand_20260910_124648_ecccef); ``move_both``, ``pinch_cloth``, ``bimanual_lift`` and
``hold_and_push`` stay here.

Real coordination, not two wrapped single-arm calls:
  * ``move_both``          - one control loop drives both arms along the same progress fraction to two Cartesian
                             targets. By default both fingertips follow straight Cartesian lines (waypoint IK on one
                             wrist branch for each arm, ``controller.move_cartesian_both``) so the tip-to-tip relation
                             is held by construction; joint-space interpolation drifted 5-6 cm after 20-35 % of every
                             30 cm fold move and made the relation monitor abort. The monitor stays as a safety abort.
  * ``pinch_cloth``        - single-arm cloth pinch: the finger pads land at the table surface (the pads sit 17 mm below
                             the fingertip grasp point; a tip target at table + 4 mm pressed them 13 mm into the table
                             and the gripper could not close), close, verify the opening after closing and after a 2 cm
                             test lift (a > 50 % drop = slipping), optional recovery slide that raises a fold.
  * ``bimanual_lift``      - grasp two points on one object (roles left/right), verify BOTH grasps (calibrated
                             opening + optional model-backed verify_grasp), lift synchronously while monitoring
                             level-ness and per-side slip; one-sided failure -> stop, lower both, release, report side.
  * ``bimanual_fold``      - two-point cloth pinch (synchronised descent, per-side verification, one-sided re-pinch
                             with a slide), synchronised test lift with slip check, lift with relation monitor and a
                             per-side pinch check after it (hand_v4), coordinated Cartesian move of both grasp points to
                             their targets, lower and release; any lift / move failure lowers and releases where the
                             cloth is instead of dragging it.
  * ``hold_and_push``      - one arm holds an object (closed fingers at a point, drift-monitored) while the other pushes
                             from start to goal; the holder's drift aborts the operation.
All targets are in metres/radians in the world frame; nothing is task specific. Deployable observations only.

Candidate cand_20260911_080228_cce58a (parent hand_v11@3.7.0; protocol v2 amendment 11, capx_two_arm_lift regression hand_v4 5/5 ->
hand_v5+ 1-3/5). Root cause from the formal_v6..v11 traces (30/30 episodes): the first ``bimanual_lift`` fails at the APPROACH stage. From
the home pose the straight Cartesian path leaves the top-down reach, ``move_both`` 1.1.0 falls back to joint space and the two-arm
``correct_tips`` pass that follows is accepted when EITHER arm improves: the left tip went 3.2 -> 1.7 cm while the right went 1.8 -> 3.5 cm
(measured on seeds 0-4 from home: 2.0/1.7 cm before, 4.5/2.7 cm after the passes), above the 2 cm tolerance -> "approach failed". hand_v4's
move_both 1.0.0 had no Cartesian correction and passed at 1.6/0.5 cm. The recovered lifts of the agent then asked for 6-8 cm and rose
4.6-5.1 cm: the lift target was the NOMINAL grasp point + lift_height while the descent had ended 0.9-1.2 cm above it (tolerance 1.5 cm),
and the skill reported "lifted 0.06 m" without measuring anything (GPT-6 diagnoses capx_two_arm_lift s102/s103, runs/analysis_v3/formal_v11).
  * ``move_both`` 1.2.0 - after a joint-space fallback whose tips end outside the tolerance, a straight Cartesian FINISHING segment from
    the pose actually reached (feasible now that the arms are inside the top-down reach; measured 4.4/2.6 cm -> 0.9/0.3 cm on seeds 0-4);
    metrics.path = "joint+cartesian". Cartesian paths and everything else unchanged.
  * ``bimanual_lift`` 1.2.0 - the lift is MEASURED: the targets are the tips measured after closing (not the nominal grasp points) plus
    the lift height, the height is the true reach ceiling (no margin back-off), the per-side rise is read from proprioception after the
    lift and a shortfall > RISE_TOLERANCE_M is re-commanded once; the result reports rise_m per side, the requested / commanded lift and
    fails honestly when the rise stays short. Same parameters (additive: none removed).

hand_v8@3.4.0 (parent hand_v7): the held / empty decision is no longer a constant in this library. ``_held`` reads the reading's calibrated width rule (``GripperReading.held_by_width``: opening >= the backend/side ``held_width_min_m`` of configs/gripper/<backend>_<side>.yaml; ambiguous and empty count as not held) and ``_held_width_min`` exposes that threshold to the pinch / transfer checks. No other behaviour change.

hand_v11@3.7.0 (parent hand_v10): cell-calibration plumbing (docs/evaluation_protocol.md, hand_v11 addendum). ``_pinch_height`` (pinch_cloth 1.1.1, and the disabled bimanual_fold 1.3.0 copy) takes the pad offset from ``ctx.cell_calibration.finger_pad_below_tip_m`` instead of physapi.sim.control.FINGER_PAD_BELOW_TIP_M; the pinch_cloth documentation names the calibrated empty-close width instead of the sim value. The values come from configs/cell_calibration/<backend>.yaml through ``ctx.cell_calibration`` (physapi.calibration.cell), loaded per backend kind (sim -> sim.yaml = the constants this file used to hard-code, so simulation decisions are identical; real / dry -> real.yaml). No skill branches on the backend kind. No other behaviour change.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

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
_held = _base._held
_held_width_min = _base._held_width_min
_reachable_height = _base._reachable_height
_robot_time = _base._robot_time
Side = Literal["left", "right"]
SIDES = ("left", "right")
PINCH_CLOSE_S = 0.8
PINCH_SLIP_RATIO = 0.5  # opening after the test lift below this fraction of the opening after closing = slipping
PINCH_MAX_WIDTH_M = 0.025  # a close wider than this holds a bunched fold (several layers), which pulls out under load
RECOVERY_INSET_M = 0.015  # a missed pinch is retried this much further onto the cloth (along the fold direction)
PINCH_CENTRE_INSET_M = 0.01  # two-point pinch: each point is moved this much toward the other arm so the 12 mm wide pads sit fully on cloth (a pad straddling the side edge closes on nothing in ~40 % of the corners)
CLOTH_SPEED_MPS = 0.12  # carry speed with a pinched cloth. Measured: a pinch creeps out under SUSTAINED load, so slow carries
#: (0.03-0.06 m/s) lost the cloth already during the lift while 0.3 m/s carries held; end-of-move corrections are still avoided.
GATHER_M = 0.0  # optional slide of the open pads along the fold direction before closing: gathers a second cloth sphere into the
#: pinch (16-21 mm opening instead of 11 mm); single-sphere pinches survived lifts but crept out during the 33 cm carry


def _cloth_duration(dist_m: float, minimum: float = 0.8, speed: float = CLOTH_SPEED_MPS) -> float:
    return float(max(minimum, min(8.0, dist_m / max(speed, 0.01))))


def _both_reachable(ctx, targets: dict[str, np.ndarray], yaws: dict[str, float], *, symmetric: bool = True) -> dict[str, np.ndarray] | None:
    sols = {}
    for side in SIDES:
        sol = ctx.controller.solve_ik(side, targets[side], yaws[side], symmetric=symmetric)
        if sol is None:
            return None
        sols[side] = sol
    return sols


def _relation_monitor(ctx, expected_delta: np.ndarray, *, tol_dist: float, tol_dz: float, watch_grip: bool):
    """Abort when the tip-to-tip vector deviates from the commanded relation or a gripper opens/closes suddenly."""

    exp_dist = float(np.linalg.norm(expected_delta))
    exp_dz = float(expected_delta[2])
    widths0 = {s: ctx.gripper_width(s) for s in SIDES} if watch_grip else {}

    def monitor(fraction, state):
        lt, rt = ctx.tip("left"), ctx.tip("right")
        d = rt - lt
        if abs(float(np.linalg.norm(d)) - exp_dist) > tol_dist:
            return f"relative distance drift {abs(float(np.linalg.norm(d)) - exp_dist) * 100:.1f} cm"
        if abs(float(d[2]) - exp_dz) > tol_dz:
            return f"height difference drift {abs(float(d[2]) - exp_dz) * 100:.1f} cm (tilt)"
        if watch_grip:
            for s in SIDES:
                w = ctx.gripper_width(s)
                if w is not None and widths0.get(s) is not None and w < 0.008 and widths0[s] > 0.012:
                    return f"{s} gripper closed on nothing (slip)"
        return None

    return monitor


def _set_grippers(ctx, apertures: dict[str, float], duration: float) -> None:
    """Command both grippers in ONE action and wait (a sequential set_gripper per side doubles the wait)."""

    action = ctx.backend.commanded_action()
    for side, aperture in apertures.items():
        action[6 if side == "left" else 13] = float(ctx.gripper_command(side, opening_fraction=float(aperture)).raw)
    for _ in range(max(1, int(duration * ctx.control_hz))):
        ctx.backend.step(action)


def _pinch_height(ctx, pad_clearance_m: float) -> float:
    """Fingertip z that puts the pad bottoms ``pad_clearance_m`` above the table. The pad offset is the cell calibration's
    ``finger_pad_below_tip_m`` (configs/cell_calibration/<backend>.yaml; hand_v11), never a sim constant."""

    return float(ctx.table_z + ctx.cell_calibration.finger_pad_below_tip_m + pad_clearance_m)


class MoveBoth(Skill):
    name = "move_both"
    version = "1.2.0"
    kind = "code"
    description = "Move BOTH fingertip grasp points to two world targets (>= 12 cm apart) in one synchronised motion. With `cartesian=true` (default) both tips follow straight lines, so the tip-to-tip distance is held by construction (drift < 1 cm); a path leaving the top-down reach (e.g. from home) falls back to joint space plus a short Cartesian finish (metrics.path). `keep_relation` aborts on tip-to-tip distance / height drift (co-held object). No collision avoidance."
    returns = "success if both tips end within tolerance; failure with the abort reason. metrics: tip_error_m {left,right}, max_relation_drift_m, path, aborted_reason, fraction, robot_time_s; outputs.tips"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK of both arms after the motion; relation monitor during it"
    dependencies = []
    known_limitations = []
    timeout_s = 12.0

    class Params(Params):
        left_xyz: list[float] = Field(min_length=3, max_length=3)
        right_xyz: list[float] = Field(min_length=3, max_length=3)
        left_yaw: float = 0.0
        right_yaw: float = 0.0
        duration: float | None = Field(default=None, ge=0.3, le=8.0, description="[s]; default distance-scaled")
        left_gripper: float | None = Field(default=None, ge=0.0, le=1.0)
        right_gripper: float | None = Field(default=None, ge=0.0, le=1.0)
        keep_relation: bool = Field(default=False)
        relation_tolerance_m: float = Field(default=0.03, ge=0.005, le=0.2)
        tolerance_m: float = Field(default=0.02, ge=0.005, le=0.1)
        cartesian: bool = Field(default=True)
        correction_passes: int = Field(default=2, ge=0, le=4)
        strict_yaw: bool = Field(default=False)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        targets = {"left": np.asarray(p.left_xyz, float), "right": np.asarray(p.right_xyz, float)}
        yaws = {"left": p.left_yaw, "right": p.right_yaw}
        if float(np.linalg.norm(targets["left"] - targets["right"])) < 0.12:
            raise SkillError("targets closer than 12 cm: the two grippers/wrist cameras collide (measured: converge only at >= ~14 cm side by side, >= 12 cm fore-aft)", error_type="precondition")
        for side in SIDES:
            if targets[side][2] < ctx.table_z - 0.02:
                raise SkillError(f"{side} target below the table", error_type="precondition")
        sols = _both_reachable(ctx, targets, yaws, symmetric=not p.strict_yaw)
        if sols is None:
            return self.result(SkillStatus.failure, "one or both targets unreachable top-down", verified=True, metrics={"reachable": False})
        grips = {s: g for s, g in (("left", p.left_gripper), ("right", p.right_gripper)) if g is not None}
        monitor = _relation_monitor(ctx, targets["right"] - targets["left"], tol_dist=p.relation_tolerance_m, tol_dz=p.relation_tolerance_m, watch_grip=True) if p.keep_relation else None
        ctrl = ctx.controller
        path = "cartesian" if p.cartesian else "joint"
        res = None
        if p.cartesian:
            res = ctrl.move_cartesian_both(targets, yaws, duration=p.duration, grippers=grips or None, monitor=monitor, correction_passes=p.correction_passes, symmetric=not p.strict_yaw)
            if not res["completed"] and res.get("fraction", 0.0) == 0.0 and ("leaves the top-down reach" in (res.get("aborted_reason") or "") or "unreachable" in (res.get("aborted_reason") or "")):
                res = None  # the straight path is not reachable (e.g. from the home pose): fall back to joint space
                path = "joint"
        if res is None:
            dq_max = max(float(np.abs(np.asarray(sols[s]) - ctx.backend.state().joint_pos[ctrl._slice(s)]).max()) for s in SIDES)
            dist = max(float(np.linalg.norm(targets[s] - ctx.tip(s))) for s in SIDES)
            dur = ctrl.duration_for(dist, dq_max) if p.duration is None else float(p.duration)
            res = ctrl.move_joints_both(sols, duration=dur, grippers=grips or None, monitor=monitor)
            if res["completed"]:
                corr = ctrl.correct_tips(targets, yaws, passes=p.correction_passes)
                res["tip_error_m_before_correction"] = {s: corr[s][0] for s in SIDES}
                after_joint = {s: float(np.linalg.norm(ctx.tip(s) - targets[s])) for s in SIDES}
                if any(e > p.tolerance_m for e in after_joint.values()):
                    # 1.2.0: the joint path from a far pose (home) ends 2-4 cm off and the paired correction pass can leave one arm
                    # WORSE (accepted when the other improved); from the pose reached the straight Cartesian segment to the targets is
                    # inside the top-down reach and lands within a few mm (0.9 / 0.3 cm measured on seeds 0-4 of capx_two_arm_lift)
                    fin = ctrl.move_cartesian_both(targets, yaws, duration=None, grippers=grips or None, monitor=monitor, correction_passes=p.correction_passes, symmetric=not p.strict_yaw)
                    if fin["completed"] or fin.get("fraction", 0.0) > 0.0:
                        res = {**fin, "tip_error_m_before_correction": res.get("tip_error_m_before_correction"), "tip_error_m_before_finish": {s: round(v, 4) for s, v in after_joint.items()}}
                        path = "joint+cartesian"
        errs = {s: float(np.linalg.norm(ctx.tip(s) - targets[s])) for s in SIDES}
        if res["completed"] and p.correction_passes > 0 and any(e > p.tolerance_m for e in errs.values()):
            # 1.2.0: the controller's two-arm correction accepts a pass when EITHER arm improves, so one arm can end worse than before
            # it (right 1.8 -> 3.5 cm from home). A per-arm pass reverts on its own when it does not help: it can only recover.
            recovered = {}
            for s in SIDES:
                if errs[s] > p.tolerance_m:
                    before, after, used = ctrl.correct_tip(s, targets[s], yaws[s], passes=max(1, p.correction_passes))
                    recovered[s] = {"before": round(before, 4), "after": round(after, 4), "passes": used}
            res["per_arm_correction"] = recovered
            errs = {s: float(np.linalg.norm(ctx.tip(s) - targets[s])) for s in SIDES}
        ok = res["completed"] and all(e <= p.tolerance_m for e in errs.values())
        msg = "both arms reached their targets" if ok else (f"aborted: {res['aborted_reason']}" if res.get("aborted_reason") else f"tip errors left {errs['left'] * 100:.1f} cm right {errs['right'] * 100:.1f} cm")
        metrics = {"tip_error_m": {k: round(v, 4) for k, v in errs.items()}, "aborted_reason": res.get("aborted_reason"), "fraction": res.get("fraction"), "path": path, "robot_time_s": _robot_time(ctx, tick0)}
        if res.get("tip_error_m_before_correction") is not None:
            metrics["tip_error_m_before_correction"] = {k: round(float(v), 4) for k, v in res["tip_error_m_before_correction"].items()}
        if res.get("tip_error_m_before_finish") is not None:
            metrics["tip_error_m_before_finish"] = res["tip_error_m_before_finish"]
        if res.get("per_arm_correction"):
            metrics["per_arm_correction"] = res["per_arm_correction"]
        if res.get("max_relation_drift_m") is not None:
            metrics["max_relation_drift_m"] = res["max_relation_drift_m"]
        return self.result(SkillStatus.success if ok else SkillStatus.failure, msg, verified=True, metrics=metrics, outputs={"tips": {s: np.round(ctx.tip(s), 4).tolist() for s in SIDES}})


def _pinch_once(ctx, side: str, point: np.ndarray, yaw: float, *, slide_xy: np.ndarray | None, slide_m: float, strict: bool = False) -> float | None:
    """Descend to the pinch height (already approached above it), optional slide, close. Returns the opening width."""

    ctx.call("move_to", side=side, xyz=point.tolist(), yaw=yaw, tolerance_m=0.015, correction_passes=1, strict_yaw=strict)
    if slide_xy is not None and slide_m > 0:
        target = point + np.array([slide_m * slide_xy[0], slide_m * slide_xy[1], 0.0])
        ctx.call("move_to", side=side, xyz=target.tolist(), yaw=yaw, cartesian=True, tolerance_m=0.02, correction_passes=1, duration=0.5, strict_yaw=strict)
    _set_grippers(ctx, {side: 0.0}, PINCH_CLOSE_S)
    return ctx.gripper_width(side)


class PinchCloth(Skill):
    name = "pinch_cloth"
    version = "1.1.1"
    kind = "code"
    description = "Pinch flat cloth (towel corner / edge) at a table-plane point with one arm: open, approach, descend so the finger pads touch the table at the point (the tip height comes from the table - do NOT aim the tip at the cloth), close, verify the opening after closing and after a 2 cm test lift (one layer ~1 cm; empty = the calibrated empty-close width; > 2.5 cm = bunched layers that slip; a drop of more than half at the test lift = slipping). A failed first pinch is retried `slide_m` further onto the cloth along `slide_direction_xy`. Ends holding the pinch 2 cm up."
    returns = "success: outputs.held=true, tip_pos, opening_width_m; metrics.pinch_width_m, width_after_test_lift_m, slid, robot_time_s. failure 'pinch missed' / 'bunched' / 'slipped' with outputs.held=false (gripper re-opened, arm retreated above the point)"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "calibrated opening after closing and after the 2 cm test lift"
    dependencies = ["move_to", "set_gripper"]
    known_limitations = ["cloth thinner than ~4 mm is not caught", "one point per arm: bimanual_fold for two"]
    timeout_s = 12.0

    class Params(Params):
        side: Side
        xyz: list[float] = Field(min_length=3, max_length=3, description="cloth point [m]; only x, y used")
        yaw: float = Field(default=0.0, description="fingers close across this axis [rad]")
        pad_clearance_m: float = Field(default=0.0, ge=-0.003, le=0.006, description="pad bottom above the table [m]")
        slide_direction_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="toward the cloth interior (recovery)")
        slide_m: float = Field(default=0.015, ge=0.0, le=0.04, description="recovery inset [m]")
        min_width_m: float | None = Field(default=None, ge=0.0, le=0.02, description="empty below this opening [m]; default: the gripper calibration's held_width_min_m")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        min_width = float(p.min_width_m) if p.min_width_m is not None else _held_width_min(ctx, p.side)
        point = np.array([float(p.xyz[0]), float(p.xyz[1]), _pinch_height(ctx, p.pad_clearance_m)])
        approach = _reachable_height(ctx, p.side, point[0], point[1], point[2], 0.06, p.yaw)
        if approach is None:
            return self.result(SkillStatus.failure, f"pinch point {np.round(point[:2], 3).tolist()} is not reachable top-down by the {p.side} arm", verified=True, metrics={"stage": "approach", "reachable": False})
        above = point + [0, 0, approach]
        r = ctx.call("move_to", side=p.side, xyz=above.tolist(), yaw=p.yaw, gripper=1.0)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", verified=True, metrics={"stage": "approach", "robot_time_s": _robot_time(ctx, tick0)})
        slide = None if p.slide_direction_xy is None else np.asarray(p.slide_direction_xy, float)
        if slide is not None and np.linalg.norm(slide) > 1e-6:
            slide = slide / np.linalg.norm(slide)
        else:
            slide = None
        width = _pinch_once(ctx, p.side, point, p.yaw, slide_xy=None, slide_m=0.0)
        slid = False
        bad = lambda w: w is None or w < min_width or w > PINCH_MAX_WIDTH_M  # noqa: E731
        metrics = {"stage": "pinch", "pinch_width_m": width, "min_width_m": min_width}
        if bad(width) and slide is not None:
            # recovery: re-pinch RECOVERY_INSET_M further onto the cloth (flat, undisturbed); a sliding close bunches layers
            _set_grippers(ctx, {p.side: 1.0}, 0.3)
            ctx.call("move_to", side=p.side, xyz=(point + [0, 0, 0.03]).tolist(), yaw=p.yaw)
            point = point + np.array([slide[0], slide[1], 0.0]) * max(p.slide_m, RECOVERY_INSET_M)
            width = _pinch_once(ctx, p.side, point, p.yaw, slide_xy=None, slide_m=0.0)
            slid = True
            metrics.update({"pinch_width_m": width, "slid": True, "recovery_point": np.round(point, 4).tolist()})
        if bad(width):
            _set_grippers(ctx, {p.side: 1.0}, 0.3)
            ctx.call("move_to", side=p.side, xyz=above.tolist(), yaw=p.yaw)
            why = "bunched several layers" if (width is not None and width > PINCH_MAX_WIDTH_M) else "missed"
            return self.result(SkillStatus.failure, f"pinch {why} (opening {0 if width is None else round(width * 1000, 1)} mm)", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        metrics["stage"] = "test_lift"
        tip = ctx.tip(p.side)
        ctx.call("move_to", side=p.side, xyz=[tip[0], tip[1], point[2] + 0.02], yaw=p.yaw)
        width2 = ctx.gripper_width(p.side)
        metrics["width_after_test_lift_m"] = width2
        if width2 is None or width2 < min_width or width2 < PINCH_SLIP_RATIO * width:
            _set_grippers(ctx, {p.side: 1.0}, 0.3)
            ctx.call("move_to", side=p.side, xyz=above.tolist(), yaw=p.yaw)
            return self.result(SkillStatus.failure, f"pinch slipped during the test lift (opening {round(width * 1000, 1)} -> {0 if width2 is None else round(width2 * 1000, 1)} mm)", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": False})
        return self.result(SkillStatus.success, f"cloth pinched (opening {round(width2 * 1000, 1)} mm){' after a recovery slide' if slid else ''}", verified=True, metrics={**metrics, "slid": slid, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"held": True, "tip_pos": np.round(ctx.tip(p.side), 4).tolist(), "opening_width_m": width2})


RISE_TOLERANCE_M = 0.01  # a measured rise this much below the commanded lift is re-commanded once, then reported as short
APPROACH_TOLERANCE_M = 0.04  # the open-finger approach above the grasp points (the descent is the precise, corrected move)
#: lift_height is a MINIMUM: the commanded lift is this much higher (within reach) so the measured rise clears the request even
#: with the 3-5 mm the servos leave at the top (agents asking for exactly the height a task needs otherwise ended 3-4 mm short)
LIFT_OVERSHOOT_M = 0.015


def _wrist_camera_pos(ctx, side: str, qpos_arm) -> np.ndarray | None:
    """World position of ``side``'s wrist camera body for an arm joint solution (kinematic model FK, no scene truth); None when
    the model has no such body (a kinematic shadow without the camera)."""

    import mujoco

    ctrl = ctx.controller
    bid = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_camera_d405")
    if bid < 0 or not hasattr(ctrl, "_scratch") or not hasattr(ctrl, "_qpos"):
        return None
    scratch = ctrl._scratch
    scratch.qpos[:] = ctrl.data.qpos
    scratch.qpos[ctrl._qpos[side]] = np.asarray(qpos_arm, dtype=float)
    mujoco.mj_kinematics(ctrl.model, scratch)
    return np.asarray(scratch.xpos[bid], dtype=float).copy()


def _camera_away_yaws(ctx, points: dict[str, np.ndarray], yaws: dict[str, float], column_height: float) -> tuple[dict[str, float], dict[str, float | None]]:
    """Per side, the wrist yaw branch (yaw or yaw +/- pi: the same parallel grasp) whose wrist CAMERA ends farthest from the other
    arm's grasp point, among the branches on which the grasp column [z, z + column_height] is reachable.

    The wrist cameras hang ~10 cm beside the fingertips. With two grasps at yaw 0 about 20 cm apart side by side, the branch the
    IK picks from home puts BOTH cameras into the gap between the arms (measured on capx_two_arm_lift with the agents' inner
    handle targets: left camera at y = -0.14 for a tip at -0.007, right camera at -0.094 for a tip at -0.213 -> contact, tips
    2-4 cm off, corrections of one arm pushing the other). The flipped branches put the cameras on the outer sides and the same
    approach lands within 0.4 cm. Returns (yaws, camera clearance per side [m] or None when the model has no camera body)."""

    ctrl = ctx.controller
    chosen = dict(yaws)
    clearance: dict[str, float | None] = {s: None for s in SIDES}
    for s in SIDES:
        other = "right" if s == "left" else "left"
        best = None
        for branch in (float(yaws[s]), float(yaws[s]) + np.pi, float(yaws[s]) - np.pi):
            col = ctrl.solve_column(s, points[s][:2], branch, [float(points[s][2]), float(points[s][2]) + float(column_height)], symmetric=False)
            if col is None:
                continue
            cam = _wrist_camera_pos(ctx, s, col[0].qpos)
            if cam is None:
                return dict(yaws), clearance  # no camera body in this model: keep the requested yaws
            d = float(np.linalg.norm(cam[:2] - points[other][:2]))
            if best is None or d > best[1] + 1e-6:
                best = (branch, d)
        if best is not None:
            chosen[s] = float(best[0])
            clearance[s] = round(best[1], 4)
    return chosen, clearance


class BimanualLift(Skill):
    name = "bimanual_lift"
    version = "1.2.0"
    kind = "code"
    description = "Grasp ONE object at two points (left arm at `left_xyz`, right arm at `right_xyz`), verify both grasps, then lift both synchronously by AT LEAST `lift_height` keeping the object level. Wrist branches keep the wrist cameras clear of each other (same grasp; a camera may then face away from the object). The lift is MEASURED: targets = tips measured after closing + lift_height + 1.5 cm (up to the reach ceiling), the per-side rise is read back by FK, a shortfall is re-commanded once and the achieved rise is reported (metrics.rise_m). Verified by calibrated openings after closing and a 2 cm test lift."
    returns = "success if both grasps verified and both tips rose >= commanded lift - 1 cm, level (outputs.tips, rise_m); failure names side / stage (metrics.stage, sides, slipped, tilt_m, rise_m)"
    preconditions = ["grasp points top-down reachable, >= 12 cm apart"]
    effects = ["object lifted by the measured rise, held by both arms"]
    failure_conditions = ["a side closes on nothing", "slip / drift during the lift", "rise short after one retry"]
    effect_verification = "per-side calibrated openings + verify_grasp; relation monitor; per-side tip rise by FK"
    dependencies = ["move_both", "set_gripper", "verify_grasp"]
    known_limitations = ["top-down grasps only", "level-ness from fingertip FK, not the object"]
    timeout_s = 40.0

    class Params(Params):
        left_xyz: list[float] = Field(min_length=3, max_length=3)
        right_xyz: list[float] = Field(min_length=3, max_length=3)
        left_yaw: float = 0.0
        right_yaw: float = 0.0
        approach_height: float = Field(default=0.08, ge=0.03, le=0.2)
        lift_height: float = Field(default=0.10, ge=0.03, le=0.25)
        level_tolerance_m: float = Field(default=0.02, ge=0.005, le=0.1)
        target_object: str = Field(default="the object", max_length=120)
        use_model_verification: bool = True

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        L, R = np.asarray(p.left_xyz, float), np.asarray(p.right_xyz, float)
        if float(np.linalg.norm(L - R)) < 0.12:
            raise SkillError("grasp points closer than 12 cm: grippers interfere (object too short for a bimanual grasp)", error_type="precondition")
        metrics = {"stage": "approach", "sides": {}}
        # wrist branches with the wrist cameras on the OUTER sides (the same parallel grasp): two yaw-0 grasps ~20 cm apart put both
        # cameras into the gap between the arms and the arms pushed each other 2-4 cm off (see _camera_away_yaws)
        branch, clearance = _camera_away_yaws(ctx, {"left": L, "right": R}, {"left": float(p.left_yaw), "right": float(p.right_yaw)}, p.approach_height)
        yaws = dict(left_yaw=branch["left"], right_yaw=branch["right"], strict_yaw=True)
        metrics["branch_yaw"] = {s: round(branch[s], 3) for s in SIDES}
        metrics["camera_clearance_m"] = clearance
        appr = {s: _reachable_height(ctx, s, *xyz, p.approach_height, branch[s]) for s, xyz in (("left", L), ("right", R))}
        if any(v is None for v in appr.values()):
            return self.result(SkillStatus.failure, f"grasp point not reachable: {[s for s, v in appr.items() if v is None]}", verified=True, metrics=metrics)
        h = min(appr.values())
        # the approach only has to bring the OPEN fingers above the grasp points (the descent below is the corrected, precise move):
        # APPROACH_TOLERANCE_M instead of move_both's 2 cm default - grasp points near an arm's reach edge (the inner part of a
        # handle, formal_v6..v11 agents) settle 2.3-2.8 cm off at the approach height and were refused although the descent from
        # there lands within 1 cm
        r = ctx.call("move_both", left_xyz=(L + [0, 0, h]).tolist(), right_xyz=(R + [0, 0, h]).tolist(), left_gripper=1.0, right_gripper=1.0, tolerance_m=APPROACH_TOLERANCE_M, **yaws)
        metrics["approach_tip_error_m"] = r.metrics.get("tip_error_m")
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", verified=True, metrics=metrics)
        metrics["stage"] = "descend"
        r = ctx.call("move_both", left_xyz=L.tolist(), right_xyz=R.tolist(), tolerance_m=0.015, **yaws)
        metrics["descent_tip_error_m"] = r.metrics.get("tip_error_m")
        if r.status != SkillStatus.success:
            ctx.call("move_both", left_xyz=(L + [0, 0, h]).tolist(), right_xyz=(R + [0, 0, h]).tolist(), **yaws)
            return self.result(SkillStatus.failure, f"descent failed: {r.message}", verified=True, metrics=metrics)
        metrics["stage"] = "grasp"
        _set_grippers(ctx, {s: 0.0 for s in SIDES}, 0.5)
        held = {s: _held(ctx, s) for s in SIDES}
        metrics["sides"] = {s: {"held_after_close": held[s][0], "width_m": held[s][1]} for s in SIDES}
        bad = [s for s in SIDES if not held[s][0]]
        if bad:
            self._release_and_retreat(ctx, L, R, h, yaws)
            return self.result(SkillStatus.failure, f"{' and '.join(bad)} gripper closed on nothing", verified=True, metrics=metrics)
        if p.use_model_verification and "verify_grasp" in ctx.library:
            for s in SIDES:
                v = ctx.call("verify_grasp", side=s, target_object=p.target_object, context="closed both grippers on the object")
                metrics["sides"][s]["verify"] = (v.outputs or {}).get("status")
                if (v.outputs or {}).get("status") == "not_held":
                    self._release_and_retreat(ctx, L, R, h, yaws)
                    return self.result(SkillStatus.failure, f"verify_grasp reports {s} not_held", verified=True, metrics=metrics)
        # 1.2.0: every height from here on is relative to the tips MEASURED after closing (the descent may have ended up to 1.5 cm
        # above the nominal grasp points, which used to be silently subtracted from the lift)
        G = {s: ctx.tip(s).copy() for s in SIDES}
        metrics["tips_at_grasp"] = {s: np.round(G[s], 4).tolist() for s in SIDES}
        metrics["stage"] = "test_lift"
        r = ctx.call("move_both", left_xyz=(G["left"] + [0, 0, 0.02]).tolist(), right_xyz=(G["right"] + [0, 0, 0.02]).tolist(), keep_relation=True, relation_tolerance_m=p.level_tolerance_m, **yaws)
        held = {s: _held(ctx, s) for s in SIDES}
        slipped = [s for s in SIDES if not held[s][0]]
        if slipped or r.status != SkillStatus.success:
            metrics["slipped"] = slipped
            self._release_and_retreat(ctx, L, R, h, yaws)
            return self.result(SkillStatus.failure, f"test lift failed: {slipped or r.message}", verified=True, metrics=metrics)
        metrics["stage"] = "lift"
        # the true reach ceiling above the measured grasp tips (no joint-limit-margin back-off: the rise is verified below instead);
        # lift_height is a minimum, so the command overshoots it by LIFT_OVERSHOOT_M when the reach allows
        wanted = float(p.lift_height) + LIFT_OVERSHOOT_M
        lift = min(_reachable_height(ctx, "left", *G["left"], wanted, branch["left"], prefer_margin=False) or 0.02, _reachable_height(ctx, "right", *G["right"], wanted, branch["right"], prefer_margin=False) or 0.02)
        r = ctx.call("move_both", left_xyz=(G["left"] + [0, 0, lift]).tolist(), right_xyz=(G["right"] + [0, 0, lift]).tolist(), duration=max(0.8, lift / 0.06), keep_relation=True, relation_tolerance_m=p.level_tolerance_m, **yaws)
        held = {s: _held(ctx, s) for s in SIDES}
        rise = {s: float(ctx.tip(s)[2] - G[s][2]) for s in SIDES}
        retried = False
        shortfall = lift - min(rise.values())
        if r.status == SkillStatus.success and all(held[s][0] for s in SIDES) and shortfall > RISE_TOLERANCE_M:
            # the servos hold a ceiling pose 1-2 cm low: re-command the residual once (only what is still reachable)
            extra = min(shortfall, 0.03)
            tgt = {s: G[s] + [0, 0, lift + extra] for s in SIDES}
            if all(ctx.controller.solve_ik(s, tgt[s], branch[s], symmetric=False) is not None for s in SIDES):
                retried = True
                r = ctx.call("move_both", left_xyz=tgt["left"].tolist(), right_xyz=tgt["right"].tolist(), duration=0.8, keep_relation=True, relation_tolerance_m=p.level_tolerance_m, tolerance_m=0.03, **yaws)
                held = {s: _held(ctx, s) for s in SIDES}
                rise = {s: float(ctx.tip(s)[2] - G[s][2]) for s in SIDES}
        tips = {s: ctx.tip(s) for s in SIDES}
        tilt = float(abs(rise["left"] - rise["right"]))
        min_rise = min(rise.values())
        metrics.update({"lift_requested": float(p.lift_height), "lift_commanded": lift, "lift_height": lift, "rise_m": {s: round(v, 4) for s, v in rise.items()}, "retried_lift": retried, "tilt_m": round(tilt, 4), "widths_after_lift": {s: held[s][1] for s in SIDES}, "sync_abort": r.metrics.get("aborted_reason"), "robot_time_s": _robot_time(ctx, tick0)})
        # the request is a minimum: enough when the measured rise clears it (or the reach-limited command, within the tolerance)
        need = min(float(p.lift_height), lift - RISE_TOLERANCE_M)
        limited = f"; reach limited the command to {lift * 100:.1f} cm" if lift < p.lift_height - 1e-6 else ""
        cams = "" if all(v is None for v in clearance.values()) else " Wrist branches were chosen for arm-arm clearance: a wrist camera may face away from the object, so its view is not evidence about the grasp - the calibrated openings and the measured rise are."
        if not (r.status == SkillStatus.success and all(held[s][0] for s in SIDES) and tilt <= p.level_tolerance_m):
            return self.result(SkillStatus.failure, f"lift not level or a side slipped: {r.message}; tilt {tilt * 100:.1f} cm; rise left {rise['left'] * 100:.1f} / right {rise['right'] * 100:.1f} cm", verified=True, metrics=metrics)
        if min_rise < need:
            return self.result(SkillStatus.failure, f"both arms hold the object but it rose only {min_rise * 100:.1f} cm (left {rise['left'] * 100:.1f} / right {rise['right'] * 100:.1f}) of the {p.lift_height * 100:.0f} cm requested{limited}: the pose is at the reach ceiling; lower the grasp height or move the object closer", verified=True, metrics=metrics, outputs={"tips": {s: np.round(tips[s], 4).tolist() for s in SIDES}, "held": True, "rise_m": {s: round(v, 4) for s, v in rise.items()}})
        return self.result(SkillStatus.success, f"object lifted {min_rise * 100:.1f} cm by both arms, level (measured tip rise left {rise['left'] * 100:.1f} / right {rise['right'] * 100:.1f} cm for a {p.lift_height * 100:.0f} cm request{limited}; tilt {tilt * 100:.1f} cm; both grasps verified by the calibrated openings).{cams}", verified=True, metrics=metrics, outputs={"tips": {s: np.round(tips[s], 4).tolist() for s in SIDES}, "held": True, "rise_m": {s: round(v, 4) for s, v in rise.items()}})

    @staticmethod
    def _release_and_retreat(ctx, L, R, h, yaws) -> None:
        _set_grippers(ctx, {s: 1.0 for s in SIDES}, 0.3)
        ctx.call("move_both", left_xyz=(L + [0, 0, h]).tolist(), right_xyz=(R + [0, 0, h]).tolist(), **yaws)


class BimanualFold(Skill):
    name = "bimanual_fold"
    version = "1.3.0"
    kind = "code"
    description = "Two-point cloth fold: each arm pinches one point (e.g. two towel corners >= 12 cm apart; only x, y used, the pads touch the table), both lift, carry the points to `left_target_xy` / `right_target_xy`, lower, release. Pinches are verified from the calibrated opening after closing AND after a synchronised 2 cm test lift (a missed side is re-pinched once further onto the cloth). Lift and carry are straight Cartesian lines planned before lifting; if any fails or a pinch is lost, the cloth is released where it is (never dragged)."
    returns = "success if both pinches held through the carry; failure names side / stage. metrics: stage, pinch_widths_m, widths_after_test_lift_m, widths_after_lift_m, recovered_sides, carry_plan, max_relation_drift_m, robot_time_s"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "per-side opening at every stage; relation monitor"
    dependencies = ["move_both", "move_to", "set_gripper"]
    known_limitations = []
    timeout_s = 45.0

    class Params(Params):
        left_xyz: list[float] = Field(min_length=3, max_length=3, description="z ignored")
        right_xyz: list[float] = Field(min_length=3, max_length=3)
        left_target_xy: list[float] = Field(min_length=2, max_length=2)
        right_target_xy: list[float] = Field(min_length=2, max_length=2)
        yaw: float = 0.0
        lift_height: float = Field(default=0.10, ge=0.03, le=0.2)
        release_height: float = Field(default=0.03, ge=0.01, le=0.08, description="[m]")
        pinch_min_width_m: float | None = Field(default=None, ge=0.0, le=0.02, description="default: the gripper calibration's held_width_min_m")
        pad_clearance_m: float = Field(default=0.0, ge=-0.003, le=0.006)
        slack_m: float = Field(default=0.0, ge=0.0, le=0.03, description="[m]")
        centre_inset_m: float = Field(default=PINCH_CENTRE_INSET_M, ge=0.0, le=0.03, description="toward the other arm [m]")
        carry_speed_mps: float = Field(default=CLOTH_SPEED_MPS, ge=0.02, le=0.5, description="[m/s]")
        gather_m: float = Field(default=GATHER_M, ge=0.0, le=0.03, description="pre-close slide [m]")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        pz = _pinch_height(ctx, p.pad_clearance_m)
        P = {"left": np.array([float(p.left_xyz[0]), float(p.left_xyz[1]), pz]), "right": np.array([float(p.right_xyz[0]), float(p.right_xyz[1]), pz])}
        T = {"left": np.array([float(p.left_target_xy[0]), float(p.left_target_xy[1]), pz]), "right": np.array([float(p.right_target_xy[0]), float(p.right_target_xy[1]), pz])}
        if float(np.linalg.norm(P["left"] - P["right"])) < 0.12:
            raise SkillError("grasp points closer than 12 cm: grippers interfere", error_type="precondition")
        fold_dir = {}
        for s in SIDES:
            d = T[s][:2] - P[s][:2]
            fold_dir[s] = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else None
        across = P["right"][:2] - P["left"][:2]
        across = across / np.linalg.norm(across)
        inset = {"left": np.array([across[0], across[1], 0.0]), "right": -np.array([across[0], across[1], 0.0])}
        for s in SIDES:  # pads fully on the cloth: inset toward the other arm (and the targets by the same offset)
            P[s] = P[s] + inset[s] * p.centre_inset_m
            T[s] = T[s] + inset[s] * p.centre_inset_m
        metrics_inset = {s: np.round(P[s], 4).tolist() for s in SIDES}
        slack = {s: inset[s] * p.slack_m for s in SIDES}
        metrics: dict = {"stage": "approach", "pinch_points": metrics_inset}
        # wrist branch per side, chosen ONCE for the whole fold: the branch on which the pinch column and the straight
        # carry path to the target are both feasible (at the far corners one branch needs a wrist flip mid-path)
        yaws = self._choose_branches(ctx, P, T, p.yaw, p.lift_height)
        metrics["branch_yaw"] = {s: round(yaws[s], 3) for s in SIDES}
        appr = {s: _reachable_height(ctx, s, *P[s], 0.08, yaws[s]) for s in SIDES}
        if any(v is None for v in appr.values()):
            return self.result(SkillStatus.failure, f"grasp points not reachable at table height: {[s for s, v in appr.items() if v is None]}", verified=True, metrics=metrics)
        h = min(appr.values())
        both = dict(left_yaw=yaws["left"], right_yaw=yaws["right"], strict_yaw=True)
        r = ctx.call("move_both", left_xyz=(P["left"] + [0, 0, h]).tolist(), right_xyz=(P["right"] + [0, 0, h]).tolist(), left_gripper=1.0, right_gripper=1.0, **both)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)})
        metrics["stage"] = "pinch"
        ctx.call("move_both", left_xyz=P["left"].tolist(), right_xyz=P["right"].tolist(), tolerance_m=0.015, correction_passes=1, **both)
        if p.gather_m > 0 and all(fold_dir[s] is not None for s in SIDES):
            G = {s: P[s] + np.array([fold_dir[s][0], fold_dir[s][1], 0.0]) * p.gather_m for s in SIDES}
            ctx.call("move_both", left_xyz=G["left"].tolist(), right_xyz=G["right"].tolist(), tolerance_m=0.02, correction_passes=0, duration=0.5, **both)
            P = G
            # the pinched point is now gather_m further along the fold than the caller's grasp point; after the fold that
            # offset lies on the far side of the release point, so the tips release gather_m short of the caller's targets
            T = {s: T[s] - np.array([fold_dir[s][0], fold_dir[s][1], 0.0]) * p.gather_m for s in SIDES}
        _set_grippers(ctx, {s: 0.0 for s in SIDES}, PINCH_CLOSE_S)
        widths = {s: ctx.gripper_width(s) for s in SIDES}
        pinch_min = float(p.pinch_min_width_m) if p.pinch_min_width_m is not None else max(_held_width_min(ctx, s) for s in SIDES)
        bad = lambda w: w is None or w < pinch_min or w > PINCH_MAX_WIDTH_M  # noqa: E731
        missed = [s for s in SIDES if bad(widths[s])]
        recovered = []
        for s in missed:  # one-sided recovery while the other arm holds: re-pinch further onto the cloth (flat, undisturbed)
            _set_grippers(ctx, {s: 1.0}, 0.3)
            ctx.call("move_to", side=s, xyz=(P[s] + [0, 0, 0.03]).tolist(), yaw=yaws[s], strict_yaw=True)
            if fold_dir[s] is not None:
                P[s] = P[s] + np.array([fold_dir[s][0], fold_dir[s][1], 0.0]) * RECOVERY_INSET_M
            widths[s] = _pinch_once(ctx, s, P[s], yaws[s], slide_xy=None, slide_m=0.0, strict=True)
            if not bad(widths[s]):
                recovered.append(s)
        metrics.update({"pinch_widths_m": widths, "recovered_sides": recovered})
        still = [s for s in SIDES if bad(widths[s])]
        if still:
            self._release_and_retreat(ctx, P, h, both)
            return self.result(SkillStatus.failure, f"pinch missed on {still} (openings {[None if widths[s] is None else round(widths[s] * 1000, 1) for s in SIDES]} mm)", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)})
        metrics["stage"] = "test_lift"
        r = ctx.call("move_both", left_xyz=(P["left"] + [0, 0, 0.02]).tolist(), right_xyz=(P["right"] + [0, 0, 0.02]).tolist(), keep_relation=True, relation_tolerance_m=0.05, tolerance_m=0.015, duration=0.6, correction_passes=0, **both)
        w2 = {s: ctx.gripper_width(s) for s in SIDES}
        metrics["widths_after_test_lift_m"] = w2
        slipped = [s for s in SIDES if w2[s] is None or w2[s] < pinch_min or w2[s] < PINCH_SLIP_RATIO * widths[s]]
        if slipped or r.status != SkillStatus.success:
            self._release_and_retreat(ctx, P, h, both)
            return self.result(SkillStatus.failure, f"pinch slipped on {slipped} during the test lift" if slipped else f"test lift failed: {r.message}", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)})
        metrics["stage"] = "lift"
        # carry path: lift as high as the pinch points allow (less cloth drags on the table), end as high as the targets
        # allow; the straight line between them must be feasible for BOTH arms on their wrist branches (at the far
        # corners the highest heights are reachable only on a flipped wrist branch, which would force a joint-space
        # move that drifts 5 cm and aborts)
        h_start = min(_reachable_height(ctx, s, *P[s], p.lift_height, yaws[s]) or 0.0 for s in SIDES)
        h_end_max = min(_reachable_height(ctx, s, *T[s], p.lift_height, yaws[s]) or 0.0 for s in SIDES)
        pulled_back = 0.0
        while h_end_max < 0.03 and pulled_back < 0.04 - 1e-9:
            # far targets beyond the top-down reach at carry height: pull them back toward the pinch points a little
            # (the fold lands the corners that much short; callers give corner positions with a few cm of tolerance)
            pulled_back = round(pulled_back + 0.02, 3)
            for s in SIDES:
                if fold_dir[s] is not None:
                    T[s] = T[s] - np.array([fold_dir[s][0], fold_dir[s][1], 0.0]) * 0.02
            h_end_max = min(_reachable_height(ctx, s, *T[s], p.lift_height, yaws[s]) or 0.0 for s in SIDES)
        metrics["targets_pulled_back_m"] = pulled_back
        if h_start < 0.03 or h_end_max < 0.03:
            self._release_and_retreat(ctx, P, h, both)
            return self.result(SkillStatus.failure, "fold targets not reachable top-down; released without moving", verified=True, metrics={**metrics, "reachable_targets": False, "robot_time_s": _robot_time(ctx, tick0)})
        carry = None
        plan_reason = None
        h1 = round(h_start, 3)
        while h1 >= 0.03 - 1e-9 and carry is None:
            starts = {s: P[s] + [0, 0, h1] + slack[s] for s in SIDES}
            start_q = {s: ctx.controller.solve_ik(s, starts[s], yaws[s], symmetric=False) for s in SIDES}
            if all(q is not None for q in start_q.values()):
                h2 = round(min(h_end_max, h1), 3)
                while h2 >= 0.03 - 1e-9:
                    plan = ctx.controller.plan_cartesian_both({s: T[s] + [0, 0, h2] + slack[s] for s in SIDES}, yaws, starts=starts, start_qpos=start_q, symmetric=False)
                    if plan["feasible"]:
                        carry = (h1, h2)
                        metrics["carry_plan"] = {"path": "cartesian", "min_margin_rad": plan["min_margin_rad"]}
                        break
                    plan_reason = plan["reason"]
                    h2 = round(h2 - 0.01, 3)
            h1 = round(h1 - 0.02, 3)
        if carry is None:
            carry = (max(0.03, min(h_start, h_end_max)),) * 2
            metrics["carry_plan"] = {"path": "joint", "reason": plan_reason}
        lift, lift_end = carry
        metrics["carry_height"] = lift
        metrics["carry_end_height"] = lift_end
        r = ctx.call("move_both", left_xyz=(P["left"] + [0, 0, lift] + slack["left"]).tolist(), right_xyz=(P["right"] + [0, 0, lift] + slack["right"]).tolist(), keep_relation=True, relation_tolerance_m=0.05, duration=_cloth_duration(lift, speed=p.carry_speed_mps), correction_passes=0, **both)
        metrics["lift_drift_m"] = r.metrics.get("max_relation_drift_m")
        if r.status != SkillStatus.success:
            self._lower_release_here(ctx, p, pz, both)
            return self.result(SkillStatus.failure, f"lift failed: {r.message}; cloth released where it was", verified=True, metrics={**metrics, "lift_failure": r.message, "lift_abort_fraction": r.metrics.get("fraction"), "robot_time_s": _robot_time(ctx, tick0)})
        widths_lift = {s: ctx.gripper_width(s) for s in SIDES}
        metrics["widths_after_lift_m"] = widths_lift
        lost = [s for s in SIDES if widths_lift[s] is None or widths_lift[s] < p.pinch_min_width_m]
        if lost:
            # hand_v4 (cand_20260909_214510_0ef7c7): the lift result decides whether the fold move may start - a pinch lost
            # during the lift means the cloth is not held at both points; lower, release and retreat instead of dragging it
            self._lower_release_here(ctx, p, pz, both)
            return self.result(SkillStatus.failure, f"pinch lost on {lost} during the lift; cloth released where it was", verified=True, metrics={**metrics, "lift_failure": f"pinch lost on {lost} during the lift", "robot_time_s": _robot_time(ctx, tick0)})
        metrics["stage"] = "move"
        travel = max(float(np.linalg.norm(T[s][:2] - P[s][:2])) for s in SIDES)
        r = ctx.call("move_both", left_xyz=(T["left"] + [0, 0, lift_end] + slack["left"]).tolist(), right_xyz=(T["right"] + [0, 0, lift_end] + slack["right"]).tolist(), keep_relation=True, relation_tolerance_m=0.05, duration=_cloth_duration(travel, 1.5, speed=p.carry_speed_mps), correction_passes=0, **both)
        metrics["max_relation_drift_m"] = r.metrics.get("max_relation_drift_m")
        metrics["move_path"] = r.metrics.get("path")
        metrics["widths_after_move_m"] = {s: ctx.gripper_width(s) for s in SIDES}
        if r.status != SkillStatus.success:
            metrics["move_failure"] = r.message
            self._lower_release_here(ctx, p, pz, both)
            return self.result(SkillStatus.failure, f"fold move aborted: {r.message}; cloth released where it was", verified=True, metrics={**metrics, "robot_time_s": _robot_time(ctx, tick0)})
        metrics["stage"] = "lower"
        ctx.call("move_both", left_xyz=(T["left"] + [0, 0, p.release_height] + slack["left"]).tolist(), right_xyz=(T["right"] + [0, 0, p.release_height] + slack["right"]).tolist(), keep_relation=True, relation_tolerance_m=0.05, duration=0.8, correction_passes=0, **both)
        widths_end = {s: ctx.gripper_width(s) for s in SIDES}
        _set_grippers(ctx, {s: 1.0 for s in SIDES}, 0.4)
        ctx.call("move_both", left_xyz=(T["left"] + [0, 0, 0.12]).tolist(), right_xyz=(T["right"] + [0, 0, 0.12]).tolist(), **both)
        held_through = all(widths_end[s] is not None and widths_end[s] >= p.pinch_min_width_m for s in SIDES)
        metrics.update({"widths_before_release_m": widths_end, "robot_time_s": _robot_time(ctx, tick0)})
        return self.result(SkillStatus.success if held_through else SkillStatus.failure, "both points carried to their targets and released" if held_through else f"pinch lost before release on {[s for s in SIDES if widths_end[s] is None or widths_end[s] < p.pinch_min_width_m]}", verified=True, metrics=metrics)

    @staticmethod
    def _choose_branches(ctx, P, T, yaw: float, lift_height: float) -> dict[str, float]:
        """Per side, the wrist yaw branch (yaw, yaw+pi, yaw-pi) on which the pinch column AND a straight carry path
        to the target (at some height >= 3 cm) are feasible; falls back to the requested yaw."""

        ctrl = ctx.controller
        chosen = {}
        for s in SIDES:
            pick = None
            for b in (float(yaw), float(yaw) + np.pi, float(yaw) - np.pi):
                if ctrl.solve_column(s, P[s][:2], b, [P[s][2], P[s][2] + 0.03], symmetric=False) is None:
                    continue
                h_start = _reachable_height(ctx, s, *P[s], lift_height, b) or 0.03
                start = P[s] + [0, 0, h_start]
                q0 = ctrl.solve_ik(s, start, b, symmetric=False)
                h = max(0.03, min(float(lift_height), h_start))
                while h >= 0.03 - 1e-9 and pick is None and q0 is not None:
                    if ctrl.plan_cartesian_both({s: T[s] + [0, 0, h]}, {s: b}, starts={s: start}, start_qpos={s: q0}, symmetric=False)["feasible"]:
                        pick = b
                    h = round(h - 0.01, 3)
                if pick is not None:
                    break
            chosen[s] = float(yaw) if pick is None else pick
        return chosen

    @staticmethod
    def _release_and_retreat(ctx, P, h, both) -> None:
        _set_grippers(ctx, {s: 1.0 for s in SIDES}, 0.3)
        ctx.call("move_both", left_xyz=(P["left"] + [0, 0, h]).tolist(), right_xyz=(P["right"] + [0, 0, h]).tolist(), **both)

    @staticmethod
    def _lower_release_here(ctx, p, pz, both) -> None:
        """Lower both tips straight down to the release height where they are, open, retreat (no dragging)."""

        tips = {s: ctx.tip(s) for s in SIDES}
        low = {s: np.array([tips[s][0], tips[s][1], pz + p.release_height]) for s in SIDES}
        ctx.call("move_both", left_xyz=low["left"].tolist(), right_xyz=low["right"].tolist(), **both)
        _set_grippers(ctx, {s: 1.0 for s in SIDES}, 0.3)
        ctx.call("move_both", left_xyz=(low["left"] + [0, 0, 0.10]).tolist(), right_xyz=(low["right"] + [0, 0, 0.10]).tolist(), **both)


class HoldAndPush(Skill):
    name = "hold_and_push"
    version = "1.1.0"
    kind = "code"
    description = "Fixture-and-operate: `hold_side` presses closed fingertips onto `hold_xyz` (holding a container / base in place) while `push_side` pushes from behind `start_xyz` toward `goal_xy` at `contact_height` (fingertip height above the table) in four Cartesian segments. The holder's fingertip drift is checked between segments; more than `hold_tolerance_m` aborts the push. Durations scale with distance."
    returns = "success if the push stroke completed with the holder within tolerance (metrics.holder_drift_m); failure names the stage"
    preconditions = ["holder and pusher targets reachable by their arms and >= 6 cm apart"]
    failure_conditions = ["holder drift", "unreachable", "arm interference"]
    effect_verification = "fingertip FK of both arms; holder drift monitor"
    dependencies = ["move_to", "set_gripper"]
    known_limitations = ["object motion is not verified (use locate_objects / ask_runtime_model)"]
    timeout_s = 30.0

    class Params(Params):
        hold_side: Side
        hold_xyz: list[float] = Field(min_length=3, max_length=3)
        push_side: Side
        start_xyz: list[float] = Field(min_length=3, max_length=3, description="object point to push (centre of the face)")
        goal_xy: list[float] = Field(min_length=2, max_length=2)
        contact_height: float = Field(default=0.02, ge=0.005, le=0.15)
        standoff: float = Field(default=0.04, ge=0.02, le=0.1)
        hold_tolerance_m: float = Field(default=0.015, ge=0.005, le=0.05)

    def run(self, ctx, p):
        if p.hold_side == p.push_side:
            raise SkillError("hold_side and push_side must differ", error_type="invalid_params")
        H = np.asarray(p.hold_xyz, float)
        S = np.asarray(p.start_xyz, float)
        G = np.array([p.goal_xy[0], p.goal_xy[1]])
        d = G - S[:2]
        if np.linalg.norm(d) < 1e-3:
            raise SkillError("goal equals start", error_type="invalid_params")
        u = d / np.linalg.norm(d)
        behind = S[:2] - u * p.standoff
        z = ctx.table_z + p.contact_height
        _set_grippers(ctx, {p.hold_side: 0.0, p.push_side: 0.0}, 0.3)
        r = ctx.call("move_to", side=p.hold_side, xyz=[H[0], H[1], H[2] + 0.06], yaw=0.0)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"holder cannot reach: {r.message}", verified=True, metrics={"stage": "hold_approach"})
        r = ctx.call("move_to", side=p.hold_side, xyz=H.tolist(), yaw=0.0, tolerance_m=0.015)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"holder did not settle: {r.message}", verified=True, metrics={"stage": "hold"})
        hold0 = ctx.tip(p.hold_side)
        r = ctx.call("move_to", side=p.push_side, xyz=[behind[0], behind[1], z + 0.06], yaw=0.0)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"pusher cannot reach start: {r.message}", verified=True, metrics={"stage": "push_approach"})
        ctx.call("move_to", side=p.push_side, xyz=[behind[0], behind[1], z], yaw=0.0)
        # push stroke in 4 segments, checking holder drift between segments (observation-driven progress)
        drift = 0.0
        for k in range(1, 5):
            pt = behind + (G - behind) * k / 4
            r = ctx.call("move_to", side=p.push_side, xyz=[pt[0], pt[1], z], yaw=0.0, cartesian=True)
            drift = float(np.linalg.norm(ctx.tip(p.hold_side) - hold0))
            if drift > p.hold_tolerance_m:
                ctx.call("move_to", side=p.push_side, xyz=[pt[0], pt[1], z + 0.08], yaw=0.0)
                return self.result(SkillStatus.failure, f"holder drifted {drift * 100:.1f} cm; push aborted at segment {k}/4", verified=True, metrics={"stage": "push", "holder_drift_m": round(drift, 4), "segments": k})
        ctx.call("move_to", side=p.push_side, xyz=[G[0], G[1], z + 0.08], yaw=0.0)
        ctx.call("move_to", side=p.hold_side, xyz=[H[0], H[1], H[2] + 0.08], yaw=0.0)
        return self.result(SkillStatus.success, f"push stroke complete; holder drift {drift * 100:.1f} cm", verified=True, metrics={"stage": "done", "holder_drift_m": round(drift, 4)})
