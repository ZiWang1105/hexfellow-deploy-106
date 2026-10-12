"""Reach-edge accuracy and motion-time skills (library ``hand_v6``: unchanged from hand_v5; from candidate cand_20260909_221511_026ccd, proposals P6 + P9).

* ``move_to`` v1.2.0 - distance-scaled duration (``duration`` omitted), optional straight-line Cartesian path,
  correction passes on the MEASURED fingertip error after the move (re-solved from the settled configuration on the
  same wrist branch); reports the residual before/after correction, the joint-limit margin of the solution and the
  robot time spent.
* ``solve_ik`` v1.1.0 - reachability plus joint-limit margin / manipulability, and ``full_column=True`` to check the
  approach, grasp and lift heights of a descent on ONE wrist branch (a branch change between heights is what made
  reach-edge descents miss by centimetres).
* ``pull_toward_base`` - open-gripper rake: the far finger pad (4.75 cm beyond the fingertip along the closing axis)
  is placed just behind the object and dragged along a straight Cartesian line toward the arm base, bringing a far
  object into the zone where descents are accurate.  Deployable observations only; the object is not tracked - the
  caller re-localises (optional ``relocate_prompt`` / ``relocate_color`` does it inside the skill).

hand_v9@3.5.0 (parent hand_v8): correction of the hand_v7 integration defect (docs/evaluation_protocol.md amendment 9 addendum). This file is candidate cand_20260910_164848_9a670e's reach.py (hand_v6's + ``ResetArmsScaled`` = reset_arms 1.1.0, which hand_v7 / hand_v8 loaded from a suffixed copy reach_9a670e.py); pull_toward_base 1.0.0 unchanged - candidate cand_20260910_190626_d1856a (symmetric yaw branch in its reach checks) was NOT folded in: the motion core's solve_ik(symmetric=True) already evaluates yaw +/- pi, so the change cannot alter the abort decision (600/600 rake starts identical in sim) and its validation was net +1 over 30 pairs. The calibrated held / empty width rule of hand_v8 (``_held`` / ``_held_width_min`` from configs/gripper/<backend>_<side>.yaml) is kept everywhere; no width constant.

hand_v11@3.7.0 (parent hand_v10): cell-calibration plumbing (docs/evaluation_protocol.md, hand_v11 addendum). ``pull_toward_base`` 1.0.1 computes the contact height and the minimum approach height from ``ctx.cell_calibration.finger_pad_below_tip_m`` (was physapi.sim.control.FINGER_PAD_BELOW_TIP_M); move_to / solve_ik / reset_arms unchanged. The values come from configs/cell_calibration/<backend>.yaml through ``ctx.cell_calibration`` (physapi.calibration.cell), loaded per backend kind (sim -> sim.yaml = the constants this file used to hard-code, so simulation decisions are identical; real / dry -> real.yaml). No skill branches on the backend kind. No other behaviour change.

Candidate cand_20260911_085230_18233b (parent hand_v11; ported from cand_20260910_221908_2ce510): ``move_to`` 1.3.0 and ``solve_ik`` 1.2.0 accept ``pitch_deg`` - the wrist
pitched about the finger-closing axis (``physapi.sim.control`` ``pitch``: the approach axis tilts from straight down toward
``(-sin yaw, cos yaw)``). 0 (default) is the top-down wrist of before; nothing else changes. Needed because the top-down reach
ceiling (~0.87 m over the bin centre of put_bottles) is below a 0.96 m rim: pitched 60 deg both arms reach z = 1.02 there.
pull_toward_base 1.0.1 and reset_arms 1.1.0 are hand_v11's, unchanged.

Candidate cand_20260911_135918_55d735 (parent hand_v13@3.9.0): pull_toward_base 1.1.0 - the straight drag is planned before any motion (``plan_cartesian_both`` from the configuration the approach + descent will reach, on the descent's wrist branch and then on the other one); an infeasible drag fails immediately with 0 s of robot time (metrics.preflight) and points to a low-column pick. formal_v13 sort_cubes s104: the drag's wrist-branch flip was discovered after 4.7-6.2 s of approach and descent at the edge of the 60 s budget. move_to / solve_ik / reset_arms unchanged.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field

from physapi.schemas.core import SkillStatus
from physapi.sim.control import LOW_MARGIN_RAD, OPEN_FINGER_HALF_SPAN_M
from physapi.skills.base import Params, Skill, SkillError

Side = Literal["left", "right"]
REACH_EDGE_HINT = "target sits at the reach edge (low joint-limit margin); bring the object closer with pull_toward_base or use the other arm/yaw"


def _robot_time(ctx, tick0: int) -> float:
    return round((ctx.backend.control_step_index - tick0) / ctx.control_hz, 3)


class MoveToPrecise(Skill):
    name = "move_to"
    version = "1.3.0"
    kind = "primitive"
    description = "Move one arm so its fingertip grasp point reaches a world position, wrist top-down (no collision avoidance; reach ceiling ~z 0.90 m) or pitched by `pitch_deg` about the closing axis (60 deg reaches z ~1.05 m). Duration scales with distance unless given; `cartesian=true` = straight line. FK measures the error after the move; `correction_passes` re-solve the residual (> 1 cm left at low margin = reach edge: pull_toward_base)."
    returns = "success if the fingertip ends within `tolerance_m`; failure if unreachable or not reached (contact / joint limit). metrics: tip_error_m, correction_passes, limit_margin_rad (< 0.01 = on a limit), branch_yaw, robot_time_s; outputs: tip_pos"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK vs target after the correction passes"
    known_limitations = []
    timeout_s = 10.0

    class Params(Params):
        side: Side
        xyz: list[float] = Field(min_length=3, max_length=3, description="fingertip target [m]")
        yaw: float | None = Field(default=None, description="wrist yaw [rad]; 0 closes along x")
        gripper: float | None = Field(default=None, ge=0.0, le=1.0, description="aperture during the move")
        duration: float | None = Field(default=None, ge=0.2, le=8.0, description="[s]")
        tolerance_m: float = Field(default=0.03, ge=0.005, le=0.1)
        cartesian: bool = Field(default=False)
        correction_passes: int = Field(default=2, ge=0, le=4, description="0 = fast transit (~1 cm residual)")
        strict_yaw: bool = Field(default=False, description="exactly this yaw branch")
        fast: bool | None = Field(default=None, description="settle mode (default: correction_passes == 0)")
        pitch_deg: float = Field(default=0.0, ge=-90.0, le=90.0, description="wrist pitch [deg]; 0 = straight down")

    def run(self, ctx, p):
        target = np.asarray(p.xyz, dtype=float)
        if target[2] < ctx.table_z - 0.02:
            raise SkillError(f"target z={target[2]:.3f} is below the table (z={ctx.table_z:.3f})", error_type="precondition")
        tick0 = ctx.backend.control_step_index
        ctrl = ctx.controller
        fast = (p.correction_passes == 0) if p.fast is None else bool(p.fast)
        pitch = math.radians(float(p.pitch_deg))
        if p.cartesian:
            rep = ctrl.move_cartesian(p.side, target, p.yaw, duration=p.duration, gripper=p.gripper, correction_passes=p.correction_passes, symmetric=not p.strict_yaw, pitch=pitch)
        else:
            rep = ctrl.move_to_report(p.side, target, p.yaw, gripper=p.gripper, duration=p.duration, correction_passes=p.correction_passes, symmetric=not p.strict_yaw, fast=fast, pitch=pitch)
        if not rep.reachable:
            return self.result(SkillStatus.failure, f"unreachable target {np.round(target, 3).tolist()} for {p.side} arm" + (f" ({rep.aborted_reason})" if rep.aborted_reason and rep.aborted_reason != "unreachable" else ""), verified=True, metrics={"reachable": False, "robot_time_s": _robot_time(ctx, tick0)})
        tip = ctx.tip(p.side)
        err = float(np.linalg.norm(tip - target))
        ok = err <= p.tolerance_m and rep.aborted_reason is None
        metrics = rep.metrics()
        metrics.update({"tip_error_m": round(err, 4), "reachable": True, "robot_time_s": _robot_time(ctx, tick0)})
        msg = f"fingertip {'reached' if ok else 'missed'} target (error {err * 100:.1f} cm"
        if rep.correction_passes and not math.isnan(rep.tip_error_m_before_correction):
            msg += f", {rep.tip_error_m_before_correction * 100:.1f} cm before {rep.correction_passes} correction pass(es)"
        msg += ")"
        if rep.aborted_reason:
            msg += f"; aborted: {rep.aborted_reason}"
        if not ok and rep.solution is not None and rep.solution.limit_margin_rad < 0.03:
            msg += "; " + REACH_EDGE_HINT
        return self.result(SkillStatus.success if ok else SkillStatus.failure, msg, verified=True, metrics=metrics, outputs={"tip_pos": np.round(tip, 4).tolist(), "robot_time_s": metrics["robot_time_s"]})


class SolveIKMargin(Skill):
    name = "solve_ik"
    is_query = True
    version = "1.2.0"
    kind = "primitive"
    description = "Check whether a fingertip target with a top-down wrist (optional yaw) - or a wrist pitched `pitch_deg` about the closing axis (as move_to / pick_tall_object use) - is reachable for one arm, with the joint-limit margin and manipulability of the solution (low margin = reach edge = poor accuracy). `full_column=true` also checks that the heights `column_heights` above the target (default 0, +3, +10 cm: grasp, test lift, approach/lift) are reachable on ONE wrist branch and reports the reach ceiling above the point. Does not move the robot."
    returns = "success if reachable (outputs.joint_solution; metrics limit_margin_rad, manipulability, branch_yaw, low_margin; with full_column: metrics.column {reachable, limit_margin_rad, heights} and metrics.reach_ceiling_m (highest reachable z above the point, searched to +0.25 m)); failure otherwise"
    timeout_s = 2.0
    effect_verification = "n/a (no motion)"

    class Params(Params):
        side: Side = Field(description="arm")
        xyz: list[float] = Field(min_length=3, max_length=3, description="world target of the fingertip grasp point [m]")
        yaw: float | None = Field(default=None, description="wrist yaw about world z [rad]; 0 closes fingers along world x")
        full_column: bool = Field(default=False, description="also check the descent column (heights above the point) on one wrist branch and the reach ceiling")
        column_heights: list[float] = Field(default=[0.0, 0.03, 0.10], min_length=1, max_length=6, description="heights above xyz to check when full_column is true [m]")
        pitch_deg: float = Field(default=0.0, ge=-90.0, le=90.0, description="wrist pitch about the closing axis [deg]; 0 = top-down")

    def run(self, ctx, p):
        ctrl = ctx.controller
        target = np.asarray(p.xyz, dtype=float)
        pitch = math.radians(float(p.pitch_deg))
        sol = ctrl.solve_ik_detailed(p.side, target, p.yaw, pitch=pitch)
        if sol is None:
            wrist = "a top-down wrist" if abs(pitch) < 1e-9 else f"the wrist pitched {p.pitch_deg:.0f} deg"
            return self.result(SkillStatus.failure, f"{p.side} arm cannot reach {np.round(target, 3).tolist()} with {wrist}", verified=True, metrics={"reachable": False})
        metrics = {"reachable": True, **sol.metrics(), "low_margin": bool(sol.limit_margin_rad < LOW_MARGIN_RAD)}
        msg = f"reachable (joint-limit margin {sol.limit_margin_rad:.2f} rad" + (", ON a joint limit: expect poor accuracy" if sol.limit_margin_rad < LOW_MARGIN_RAD else "") + ")"
        if p.full_column:
            heights = [float(target[2]) + float(h) for h in p.column_heights]
            col = ctrl.column_margin(p.side, target[:2], p.yaw, heights, pitch=pitch)
            metrics["column"] = col
            ceiling = None
            for h in np.arange(0.0, 0.251, 0.01):
                if ctrl.solve_ik(p.side, [target[0], target[1], target[2] + h], p.yaw, pitch=pitch) is not None:
                    ceiling = round(float(target[2] + h), 3)
                elif ceiling is not None:
                    break
            metrics["reach_ceiling_m"] = ceiling
            if col["reachable"]:
                msg += f"; column {[round(h, 3) for h in heights]} reachable on one branch (margin {col['limit_margin_rad']:.2f} rad); reach ceiling {ceiling} m"
            else:
                msg += f"; the column above is NOT reachable on one branch (reach ceiling {ceiling} m): lower the approach/lift heights or move the object closer"
        return self.result(SkillStatus.success, msg, verified=True, metrics=metrics, outputs={"joint_solution": np.round(sol.qpos, 4).tolist()})


COLOR_WORDS = ("red", "blue", "green", "yellow", "orange", "purple", "cyan", "tan", "pink")


class PullTowardBase(Skill):
    name = "pull_toward_base"
    version = "1.1.0"
    kind = "code"
    description = "Rake an object at the reach edge toward the arm base: gripper open, fingertip on the base side of the object at contact height so the FAR finger pad (4.75 cm beyond the tip) sits behind it, then dragged `distance` (shortened to what is reachable) toward the base or `direction_xy`, then lifted. The drag is PLANNED before any motion: an infeasible drag fails at once with 0 s (metrics.preflight) - then pick with a LOW column (approach_height 0.04, lift_height 0.05) if solve_ik allows, or use the other arm / transfer. Object <= 7 cm tall, <= 8 cm along the pull; re-localise afterwards."
    returns = "success if the drag completed: outputs.expected_xyz, travel_m, object_xyz when re-localised; failure: start unreachable (beyond the rake), drag infeasible (preflight), too tall / wide, descent or drag blocked. metrics: travel_m, distance_used_m, preflight"
    preconditions = []
    effects = []
    failure_conditions = []
    effect_verification = "fingertip FK of descent and drag"
    observations_used = ["proprio", "rgb:top", "depth:top"]
    dependencies = ["move_to", "set_gripper", "segment_objects", "locate_objects"]
    known_limitations = []
    timeout_s = 15.0

    class Params(Params):
        side: Side
        object_xyz: list[float] = Field(min_length=3, max_length=3, description="object centre [m]")
        distance: float = Field(default=0.06, ge=0.02, le=0.15, description="[m]")
        object_half_width_m: float = Field(default=0.015, ge=0.005, le=0.04, description="[m]")
        object_top_z: float | None = Field(default=None)
        direction_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="default: toward the base")
        contact_clearance_m: float = Field(default=0.005, ge=0.0, le=0.02, description="pad clearance [m]")
        relocate_prompt: str | None = Field(default=None, max_length=80)
        relocate_color: str | None = Field(default=None)

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        ctrl = ctx.controller
        obj = np.asarray(p.object_xyz, dtype=float)
        table = ctx.table_z
        half = float(p.object_half_width_m)
        top = float(p.object_top_z) if p.object_top_z is not None else float(obj[2] + half)
        if top - table > 0.07:
            return self.result(SkillStatus.failure, f"object top {top - table:.3f} m above the table: objects taller than 7 cm may topple under a rake pull", verified=True, metrics={"stage": "precondition"})
        if p.direction_xy is not None:
            u = np.asarray(p.direction_xy, dtype=float)
        else:
            u = ctrl.base_position(p.side)[:2] - obj[:2]
        if float(np.linalg.norm(u)) < 1e-6:
            raise SkillError("pull direction is undefined (object at the arm base?)", error_type="invalid_params")
        u = u / np.linalg.norm(u)
        yaw = float(np.arctan2(u[1], u[0]))
        standoff = OPEN_FINGER_HALF_SPAN_M - half - 0.008
        if standoff < 0.0:
            return self.result(SkillStatus.failure, f"object half width {half:.3f} m does not fit inside the open fingers ({OPEN_FINGER_HALF_SPAN_M:.4f} m to the pad)", verified=True, metrics={"stage": "precondition"})
        pad_below_tip = float(ctx.cell_calibration.finger_pad_below_tip_m)  # hand_v11: the cell calibration, not a sim constant
        contact_z = table + pad_below_tip + float(p.contact_clearance_m)
        tip0 = np.array([obj[0] + standoff * u[0], obj[1] + standoff * u[1], contact_z])
        if ctrl.solve_ik(p.side, tip0, yaw) is None:
            # the far pad has to sit BEHIND the object (further from the base than the object itself): beyond the rake reach
            return self.result(SkillStatus.failure, f"the pull start {np.round(tip0[:2], 3).tolist()} (far pad behind the object) is not reachable top-down at contact height by the {p.side} arm: the object lies beyond this arm's rake reach - use the other arm, or transfer_between_arms after a pick by the arm that reaches it", verified=True, metrics={"stage": "reach", "reachable": False, "yaw": round(yaw, 3)})
        # the drag end: the requested distance, shortened in 1 cm steps to the longest reachable drag (>= 2 cm)
        distance = float(p.distance)
        tip1 = tip0 + np.array([distance * u[0], distance * u[1], 0.0])
        while ctrl.solve_ik(p.side, tip1, yaw) is None and distance - 0.01 >= 0.02 - 1e-9:
            distance = round(distance - 0.01, 3)
            tip1 = tip0 + np.array([distance * u[0], distance * u[1], 0.0])
        if ctrl.solve_ik(p.side, tip1, yaw) is None:
            return self.result(SkillStatus.failure, f"no reachable drag end along {np.round(u, 2).tolist()} from {np.round(tip0[:2], 3).tolist()} at contact height by the {p.side} arm (tried {p.distance * 100:.0f} down to 2 cm): use the default direction (toward the base) or the other arm", verified=True, metrics={"stage": "reach", "reachable": False, "yaw": round(yaw, 3)})
        # approach height: open pads must clear the object top; prefer 6 cm above contact, lower until reachable
        min_approach = top + pad_below_tip + 0.015
        approach_z = None
        z = max(contact_z + 0.06, min_approach)
        while z >= min_approach - 1e-9:
            if ctrl.solve_ik(p.side, [tip0[0], tip0[1], z], yaw) is not None:
                approach_z = z
                break
            z -= 0.01
        if approach_z is None:
            return self.result(SkillStatus.failure, f"no reachable approach height above {np.round(tip0[:2], 3).tolist()} that clears the object top", verified=True, metrics={"stage": "reach", "reachable": False})
        # 1.1.0 drag preflight (formal_v13 sort_cubes s104: the straight drag needed a wrist-branch flip and was refused only AFTER 4.7 s of
        # approach and descent, at the reach edge where every second of the 60 s budget counted): plan the Cartesian drag from the contact
        # pose before any motion - on the wrist branch the descent will use, else on the other branch (the drag then pins that branch)
        preflight: dict = {"feasible": False, "branch_yaw": None, "reason": None, "tried": []}
        yaw_used, strict = yaw, False
        for cand_yaw, symmetric in ((yaw, True), (yaw + math.pi, False)):
            # emulate the approach (joint move from the current pose) and the descent (seeded from the approach pose): the drag is planned
            # from the configuration the arm will actually be in at the contact pose - the branch the IK picks from the home pose differs
            s_app = ctrl.solve_ik_detailed(p.side, [tip0[0], tip0[1], approach_z], cand_yaw, symmetric=symmetric)
            s0 = None if s_app is None else ctrl.solve_ik_detailed(p.side, tip0, float(s_app.yaw), symmetric=False, seed_qpos=s_app.qpos, reference_qpos=s_app.qpos)
            if s0 is None:
                preflight["tried"].append({"yaw": round(float(cand_yaw), 3), "start": "unreachable"})
                continue
            plan = ctrl.plan_cartesian_both({p.side: tip1}, {p.side: float(s0.yaw)}, starts={p.side: tip0}, start_qpos={p.side: s0.qpos}, symmetric=True)
            preflight["tried"].append({"yaw": round(float(s0.yaw), 3), "start": "ok", "drag": "feasible" if plan["feasible"] else str(plan["reason"])})
            if plan["feasible"]:
                preflight.update({"feasible": True, "branch_yaw": round(float(s0.yaw), 3), "path_dq_rad": plan["path_dq_rad"], "min_margin_rad": plan["min_margin_rad"]})
                yaw_used = float(s0.yaw)
                strict = abs(math.atan2(math.sin(yaw_used - yaw), math.cos(yaw_used - yaw))) > 0.1 or not symmetric
                break
            preflight["reason"] = plan["reason"]
        if not preflight["feasible"]:
            return self.result(SkillStatus.failure, f"the {distance * 100:.0f} cm drag from {np.round(tip0[:2], 3).tolist()} along {np.round(u, 2).tolist()} is not a continuous straight path for the {p.side} arm on either wrist branch ({preflight['reason']}): the object lies where a rake cannot pull it - nothing moved. Pick it with a LOW column instead (pick with approach_height 0.04 and lift_height 0.05 if solve_ik(full_column, column_heights [0, 0.03, 0.04]) allows), or use the other arm / transfer_between_arms", verified=True, metrics={"stage": "preflight", "reachable": False, "preflight": preflight, "yaw": round(yaw, 3), "robot_time_s": _robot_time(ctx, tick0)})
        r = ctx.call("move_to", side=p.side, xyz=[tip0[0], tip0[1], approach_z], yaw=yaw_used, gripper=1.0, strict_yaw=strict)
        if r.status != SkillStatus.success and strict:
            r = ctx.call("move_to", side=p.side, xyz=[tip0[0], tip0[1], approach_z], yaw=yaw_used, gripper=1.0)
        if r.status != SkillStatus.success:
            return self.result(SkillStatus.failure, f"approach failed: {r.message}", verified=True, metrics={"stage": "approach", "preflight": preflight, "robot_time_s": _robot_time(ctx, tick0)})
        down = ctx.call("move_to", side=p.side, xyz=tip0.tolist(), yaw=yaw_used, tolerance_m=0.015, correction_passes=1, strict_yaw=strict)
        if down.status != SkillStatus.success:
            ctx.call("move_to", side=p.side, xyz=[tip0[0], tip0[1], approach_z], yaw=yaw_used)
            return self.result(SkillStatus.failure, f"descent to contact height failed (a finger may have landed on the object; check object_xyz): {down.message}", verified=True, metrics={"stage": "descend", "tip_error_m": down.metrics.get("tip_error_m"), "preflight": preflight, "robot_time_s": _robot_time(ctx, tick0)})
        start_tip = ctx.tip(p.side)
        drag = ctx.call("move_to", side=p.side, xyz=tip1.tolist(), yaw=yaw_used, cartesian=True, tolerance_m=0.02, correction_passes=1, strict_yaw=strict)
        yaw = yaw_used
        end_tip = ctx.tip(p.side)
        travel = float((end_tip[:2] - start_tip[:2]) @ u)
        ctx.call("move_to", side=p.side, xyz=[end_tip[0], end_tip[1], approach_z], yaw=yaw)
        expected = np.array([obj[0] + travel * u[0], obj[1] + travel * u[1], obj[2]])
        metrics = {"stage": "drag", "travel_m": round(travel, 4), "distance_used_m": distance, "drag_tip_error_m": drag.metrics.get("tip_error_m"), "yaw": round(yaw, 3), "direction_xy": np.round(u, 3).tolist(), "preflight": preflight, "robot_time_s": _robot_time(ctx, tick0)}
        outputs = {"expected_xyz": np.round(expected, 4).tolist(), "travel_m": round(travel, 4), "direction_xy": np.round(u, 3).tolist()}
        ok = drag.status == SkillStatus.success and travel >= 0.7 * distance
        relocated = None
        if ok and (p.relocate_prompt or p.relocate_color):
            relocated = self._relocate(ctx, p, expected)
            if relocated is not None:
                outputs["object_xyz"] = np.round(relocated, 4).tolist()
                metrics["relocated"] = True
                metrics["relocation_offset_m"] = round(float(np.linalg.norm(np.asarray(relocated)[:2] - expected[:2])), 4)
            else:
                metrics["relocated"] = False
        shortened = f" (drag shortened to {distance * 100:.0f} of the requested {p.distance * 100:.0f} cm: the reach edge)" if distance < p.distance - 1e-9 else ""
        msg = f"pulled {travel * 100:.1f} cm toward {'the base' if p.direction_xy is None else 'direction ' + str(np.round(u, 2).tolist())}{shortened}; expect the object near {np.round(expected[:2], 3).tolist()}" if ok else f"pull incomplete (travel {travel * 100:.1f} cm of {distance * 100:.0f}): {drag.message}"
        if relocated is not None:
            msg += f"; re-localised at {np.round(relocated[:2], 3).tolist()}"
        elif ok and (p.relocate_prompt or p.relocate_color):
            msg += "; re-localisation found nothing near the expected point - locate it before picking"
        return self.result(SkillStatus.success if ok else SkillStatus.failure, msg, verified=ok and relocated is not None, metrics=metrics, outputs=outputs)

    @staticmethod
    def _relocate(ctx, p, expected):
        dets = []
        if p.relocate_prompt and "segment_objects" in ctx.library:
            r = ctx.call("segment_objects", text_prompt=p.relocate_prompt, max_results=6)
            if r.status != SkillStatus.error:
                dets = list(r.outputs.get("detections") or [])
        if not dets and "locate_objects" in ctx.library:
            color = p.relocate_color or next((w for w in COLOR_WORDS if p.relocate_prompt and w in p.relocate_prompt.lower()), None)
            if color:
                r = ctx.call("locate_objects", color=color, object_height_m=max(0.01, 2 * (float(expected[2]) - ctx.table_z)))
                if r.status != SkillStatus.error:
                    dets = list(r.outputs.get("detections") or [])
        near = [d for d in dets if d.get("world_xyz") and float(np.linalg.norm(np.asarray(d["world_xyz"][:2]) - expected[:2])) < 0.06]
        if not near:
            return None
        best = min(near, key=lambda d: float(np.linalg.norm(np.asarray(d["world_xyz"][:2]) - expected[:2])))
        xyz = list(best["world_xyz"])
        xyz[2] = float(expected[2])  # keep the caller's grasp height; the detection z is a plane/half-height estimate
        return xyz


class ResetArmsScaled(Skill):
    """reset_arms 1.1.0 (candidate cand_20260910_164848_9a670e, hand_v6 follow-up 3): the requested ``duration`` is a MINIMUM.

    hand_v6's primitive drove the joints home over exactly the requested time and read the joint error right after the
    last commanded step: from a far pose ``reset_arms(duration=0.5)`` has to move ~1.2 rad in 0.5 s, the per-tick rate
    clamp (0.05 rad) and the position servos cannot follow, and the primitive reported "home joint error 1.16 rad"
    (agent abc_sort s0 at the budget edge). The duration is now raised to the rate floor of the largest joint
    displacement (``controller.rate_floor_s``: ~1.05 s per radian at 30 Hz / 0.05 rad per tick) and the arms settle
    briefly (joint velocities quiet, at most SETTLE_MAX_S) before the error is measured. Grippers open as before."""

    name = "reset_arms"
    version = "1.1.0"
    kind = "primitive"
    description = "Move both arms back to the home configuration (grippers open). `duration` is a minimum: it is raised to the shortest time the joint rate clamp allows for the largest joint displacement (~1.05 s per rad), and the arms settle (<= 0.3 s) before the home joint error is measured - a short duration from a far pose is executed at the rate floor instead of failing."
    returns = "success if every arm joint ends within 0.15 rad of home; metrics: max_joint_error_rad, duration_requested_s, duration_used_s, dq_max_rad, settle_s, robot_time_s"
    effects = ["both arms at home pose, grippers open"]
    effect_verification = "joint error to home pose after a settle"
    timeout_s = 8.0

    class Params(Params):
        duration: float = Field(default=2.0, ge=0.5, le=6.0, description="[s], minimum; raised to the rate floor of the move")

    ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    SETTLE_MAX_S = 0.3
    SETTLE_MIN_S = 0.1
    SETTLE_VEL_TOL = 0.08
    SETTLE_QUIET_TICKS = 2
    HOME_TOL_RAD = 0.15

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        backend = ctx.backend
        home = np.asarray(backend.home_action(), dtype=float)
        q0 = np.asarray(backend.state().joint_pos, dtype=float)
        dq_max = float(np.abs(home[self.ARM_JOINTS] - q0[self.ARM_JOINTS]).max())
        floor = float(ctx.controller.rate_floor_s(dq_max))
        duration = max(float(p.duration), floor)
        ctx.controller.reset_arms(duration=duration)
        # settle: hold the home command until the arm joints are quiet (the shoulder servos ring after a fast move)
        hz = ctx.control_hz
        quiet = 0
        ticks = 0
        for tick in range(1, int(round(self.SETTLE_MAX_S * hz)) + 1):
            st = backend.step(home)
            ticks = tick
            vel = getattr(st, "joint_vel", None)
            quiet = quiet + 1 if vel is None or float(np.abs(np.asarray(vel, dtype=float)[self.ARM_JOINTS]).max()) < self.SETTLE_VEL_TOL else 0
            if tick >= int(round(self.SETTLE_MIN_S * hz)) and quiet >= self.SETTLE_QUIET_TICKS:
                break
        err = float(np.abs(np.asarray(backend.state().joint_pos, dtype=float)[self.ARM_JOINTS] - home[self.ARM_JOINTS]).max())
        ok = err < self.HOME_TOL_RAD
        metrics = {"max_joint_error_rad": round(err, 4), "duration_requested_s": float(p.duration), "duration_used_s": round(duration, 3), "dq_max_rad": round(dq_max, 4), "settle_s": round(ticks / hz, 3), "robot_time_s": _robot_time(ctx, tick0)}
        msg = f"home joint error {err:.3f} rad" + (f" (duration raised {p.duration:.2f} -> {duration:.2f} s for a {dq_max:.2f} rad move)" if duration > float(p.duration) + 1e-9 else "")
        return self.result(SkillStatus.success if ok else SkillStatus.failure, msg, verified=True, metrics=metrics)
