"""Grasp-verified manipulation skills (library ``hand_v6``@3.2.0 = hand_v5@3.1.0 + candidate cand_20260910_121426_b48287: hand_v1_gv lineage;
hand_v4's status semantics (cand_20260909_214510_0ef7c7) and restricted re-localisation (cand_20260909_214243_dc0358)
plus the reach-edge / motion-time revision of cand_20260909_221511_026ccd, merged by hand).

Same motions as ``manipulation.py`` but with event-triggered model-backed grasp verification (``verify_grasp``): after
closing the gripper, after a small test lift (re-verified when the closure was not model-confirmed), before transport/place
and after a transfer regrasp.

Status semantics (hand_v4, unchanged): the proprioceptive evidence decides when the model cannot. A grasp that the
calibrated opening reports as held through the lifts is ``success`` even when the verifier was *unavailable* (budget
exhausted, model error, truncated answer) or stayed *uncertain*; such a result carries
``metrics.verification_status = "held_proprio_only"``, ``effect_verified = False`` and
``metrics.verifier in {"unavailable", "uncertain", "not_called"}``. ``pick`` never returns ``unknown``.

Candidate cand_20260910_121426_b48287 (truth-checked diagnoses of the remaining hand_v5 single-seed failures):
  * verifier conflict: when the calibrated opening reports an object between the pads and verify_grasp answers
    ``not_held``, pick no longer reopens on the model's word alone (every such verdict in the evidence - abc_sort s102/s104,
    sort_cubes s104, campaign abc_sort s0 / sort_cubes s3 - was a false negative with the block inside, and the reopen +
    retry cost 4-10 s of the 60 s relay budget). The 3 cm test lift decides: an object that follows the gripper (opening
    held, tip rose) is re-verified once; a second not_held against a held opening is ``held_proprio_only`` with
    ``metrics.verifier_conflict``. Nothing follows the gripper -> the old bounded recovery (reopen, re-acquire, one retry).
    ``place``'s pre-transport verification likewise keeps transporting a proprioceptively held object.
  * the 3 cm test lift is kept for every closure (skipping it after a model-confirmed closure was tried: the single 15 cm
    lift then falls into the fast mode's large-joint-move branch - two settles and a correction pass - and costs 0.5 s
    MORE than the two short moves); a model-confirmed closure is not re-verified after the test lift.
  * ``pick_and_place`` routes by reach: ``side='auto'`` picks the arm whose approach and release columns are both
    reachable; when no single arm reaches both (or the given arm cannot reach the release column) the object is picked by
    the arm that reaches it, handed over with ``transfer_between_arms`` and placed by the other - one call for a cross-side
    relay, decided by IK before any motion.
  * idle-arm clearance and approach lowering from ``manipulation.py``.
Budgeted: at most ``max_verifications`` model calls per pick.

hand_v8@3.4.0 (parent hand_v7): the held / empty decision is no longer a constant in this library. ``_held`` reads the reading's calibrated width rule (``GripperReading.held_by_width``: opening >= the backend/side ``held_width_min_m`` of configs/gripper/<backend>_<side>.yaml; ambiguous and empty count as not held) and ``_held_width_min`` exposes that threshold to the pinch / transfer checks. No other behaviour change.

hand_v9@3.5.0 (parent hand_v8): correction of the hand_v7 integration defect (docs/evaluation_protocol.md amendment 9 addendum). This file is hand_v8's (pick 1.5.0 held_by_width; the pre-transport verifier conflict reads held_by_width) with PlaceVerified 1.6.0: the transit_height >= 0.01 contract of candidate cand_20260910_164848_9a670e comes from the base Place in manipulation.py. The calibrated held / empty width rule of hand_v8 (``_held`` / ``_held_width_min`` from configs/gripper/<backend>_<side>.yaml) is kept everywhere; no width constant.

Candidate cand_20260911_003629_419496 (parent hand_v9@3.5.0): pick 1.6.0. (1) already_holding precondition (formal_v9 abc_insert s101: a pick /
extract issued as "recovery" for an already-held peg opened the fingers and discarded the grasp): a gripper commanded closed whose calibrated
width rule says held is refused without motion. (2) ``exclusive_of`` (formal_v9 abc_extract s104: the target stands in a container and a grasp
of target + rim was verified "held"): the verifier's question names what must NOT be between the pads; a model verdict held with
holding_target false is a wrong_object failure (released, no blind retry) instead of a held pick.

Candidate cand_20260911_135918_55d735 (parent hand_v13@3.9.0): pick 1.8.0 = 1.7.0 + the base module's ``pad_floor_z`` contract and measured pad gate (metrics.pad_gate). formal_v13 abc_extract s103: a descent accepted 0.6 cm low (1.5 cm tolerance) put the pads 3 mm below the cup rim and the fingers closed on peg AND cup.
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

_spec = importlib.util.spec_from_file_location(f"physapi_libmod_{hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]}_manipulation_base", Path(__file__).with_name("manipulation.py"))
if _spec.name in sys.modules:  # one base module per library: every skills file shares it (pydantic resolves the parent Params through sys.modules)
    _base = sys.modules[_spec.name]
else:
    _base = importlib.util.module_from_spec(_spec)
    assert _spec.loader
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
_held = _base._held
_reachable_height = _base._reachable_height
_descend = _base._descend
_descent_metrics = _base._descent_metrics
_edge_hint = _base._edge_hint
_robot_time = _base._robot_time
_clear_other_arm = _base._clear_other_arm
_approach = _base._approach
_xyz = _base._xyz
_held_width_min = _base._held_width_min
MIN_TRANSIT_M = _base.MIN_TRANSIT_M
Side = Literal["left", "right"]


RELOCALISE_RADIUS_M = 0.025
#: The 3 cm test lift must raise the fingertip by at least this much for the closure to count as "the object follows the gripper"
#: (a lower rise means the fingers hold something that cannot be lifted, or nothing).
TEST_LIFT_MIN_RISE_M = 0.01
TEST_LIFT_M = 0.03


def _relocalise(ctx, target_object: str, x: float, y: float, z: float) -> tuple[float, float]:
    """Restricted re-localisation for the re-grasp: requested colour (colour word in ``target_object``) within
    RELOCALISE_RADIUS_M of (x, y); falls back to any saturated blob in the same radius; keeps (x, y) otherwise."""

    from physapi.perception.sam3 import PROMPT_COLOR_WORDS_V2, prompt_color

    colour = prompt_color(target_object or "", PROMPT_COLOR_WORDS_V2)
    height = max(0.01, 2 * (z - ctx.table_z))
    for color in ([colour] if colour else []) + ["any"]:
        det = ctx.call("locate_objects", color=color, object_height_m=height, near_xy=[x, y], near_radius_m=RELOCALISE_RADIUS_M, return_geometry=False)
        near = [d for d in (det.outputs.get("detections") or []) if d.get("world_xyz")]
        if near:
            best = min(near, key=lambda d: (d["world_xyz"][0] - x) ** 2 + (d["world_xyz"][1] - y) ** 2)
            return float(best["world_xyz"][0]), float(best["world_xyz"][1])
    return x, y


#: Radius of the WIDER fresh localisation after the restricted one failed (an empty closure means the estimate was off by
#: more than the finger opening allows; a second blind descent on the same estimate is the identical failure).
REACQUIRE_RADIUS_M = 0.08
#: A re-acquired estimate must move the grasp point by at least this much to count as new (otherwise it is the same blind retry).
MIN_UPDATE_M = 0.003


def _observation_pose(ctx, side: str, x: float, y: float, z: float, approach: float) -> list[float] | None:
    """Reachable pose that takes the arm out of the top camera's line of sight over the object: toward the arm's own
    side and up (the camera looks along +x from the robots' side); falls back to a higher point above the object."""

    own = 0.12 if side == "left" else -0.12
    for cand in ([x, y + own, ctx.table_z + 0.15], [x - 0.06, y + own, ctx.table_z + 0.15], [x, y, z + approach + 0.05]):
        if ctx.controller.solve_ik(side, cand, 0.0) is not None:
            return [float(v) for v in cand]
    return None


def _reacquire(ctx, target_object: str, x: float, y: float, z: float, radius_m: float = REACQUIRE_RADIUS_M) -> tuple[tuple[float, float] | None, str]:
    """Wider fresh localisation of the requested object around the failed estimate: SAM3 instance (``target_object`` prompt)
    when a server is present, then colour blobs of the requested colour, both restricted to ``radius_m`` (never a blob of
    another colour: the unrestricted search re-targeted a different object in the failure analysis). Returns
    ((x, y), source) or (None, what was tried)."""

    from physapi.perception.sam3 import PROMPT_COLOR_WORDS_V2, prompt_color

    tried = []
    height = max(0.01, 2 * (z - ctx.table_z))
    if "segment_objects" in ctx.library and target_object and target_object != "the target object":
        r = ctx.call("segment_objects", text_prompt=target_object[:80], max_results=6, near_xy=[x, y], near_radius_m=radius_m, return_geometry=False)
        tried.append(f"segment_objects:{r.status.value}")
        if r.status != SkillStatus.error:
            near = [d for d in (r.outputs.get("detections") or []) if d.get("world_xyz")]
            if near:
                best = min(near, key=lambda d: (d["world_xyz"][0] - x) ** 2 + (d["world_xyz"][1] - y) ** 2)
                return (float(best["world_xyz"][0]), float(best["world_xyz"][1])), "segment_objects"
    colour = prompt_color(target_object or "", PROMPT_COLOR_WORDS_V2)
    if colour:
        r = ctx.call("locate_objects", color=colour, object_height_m=height, near_xy=[x, y], near_radius_m=radius_m, return_geometry=False)
        tried.append(f"locate_objects:{colour}:{r.status.value}")
        if r.status != SkillStatus.error:
            near = [d for d in (r.outputs.get("detections") or []) if d.get("world_xyz")]
            if near:
                best = min(near, key=lambda d: (d["world_xyz"][0] - x) ** 2 + (d["world_xyz"][1] - y) ** 2)
                return (float(best["world_xyz"][0]), float(best["world_xyz"][1])), f"locate_objects:{colour}"
    else:
        # no colour word (e.g. the receiver's regrasp inside transfer_between_arms): any saturated blob, but only when it is the
        # ONLY one within the radius - an ambiguous neighbourhood must not re-target a different object
        r = ctx.call("locate_objects", color="any", object_height_m=height, near_xy=[x, y], near_radius_m=radius_m, return_geometry=False)
        near = [d for d in (r.outputs.get("detections") or []) if d.get("world_xyz")] if r.status != SkillStatus.error else []
        tried.append(f"locate_objects:any:{r.status.value}:{len(near)} blob(s)")
        if len(near) == 1:
            return (float(near[0]["world_xyz"][0]), float(near[0]["world_xyz"][1])), "locate_objects:any(unambiguous)"
    return None, ", ".join(tried) or "no colour word in target_object and no SAM3"


def _verify(ctx, side: str, target: str, context: str, before: list[str]) -> dict:
    r = ctx.call("verify_grasp", side=side, target_object=target, context=context, before_frames=before)
    out = dict(r.outputs or {})
    out.setdefault("status", "uncertain")
    out["skill_status"] = r.status.value
    # availability: "ok" = the model (or the proprioceptive short-circuit) answered; "unavailable" = budget exhausted,
    # model error or truncated output (verify_grasp returns skill status `unknown` in that case)
    out["availability"] = out.get("availability") or ("ok" if r.status == SkillStatus.success else "unavailable")
    return out


def _verifier_kind(v: dict) -> str:
    """held | not_held | uncertain | unavailable from one verify_grasp output."""

    if v.get("availability") == "unavailable":
        return "unavailable"
    return str(v.get("status", "uncertain"))


def _verify_brief(v: dict) -> dict:
    """The part of a verify_grasp answer worth keeping in a conflict record."""

    return {k: v.get(k) for k in ("status", "holding_something", "holding_target", "confidence", "notes") if k in v}


#: pick 1.6.0 (cand_20260911_003629_419496): a gripper that is commanded closed AND holds by the calibrated width rule is not re-picked
#: (abc_insert s101: a pick / extract issued as "recovery" for an already-held peg opened the fingers and discarded the grasp).
ALREADY_HOLDING_HINT = ("pick starts a NEW grasp: it opens the fingers and would drop what is held. place / insert_into_container the held object first, "
                        "or call set_gripper(aperture=1) deliberately to discard the grasp")


def _holding(ctx, side: str) -> tuple[bool, float | None]:
    """(an object is between the fingers, opening width): commanded closed (< half open) AND the calibrated width rule says held."""

    commanded = float(ctx.backend.commanded_action()[6 if side == "left" else 13])
    held, width = _held(ctx, side)
    return (held and commanded < 0.5), width


class PickVerified(_base.Pick):
    version = "1.8.0"
    description = "Top-down pick; refused while the arm already holds something. Evidence: calibrated opening, 3 cm test lift, lift, verify_grasp (not_held against an object-sized opening: the test lift decides). `exclusive_of` (the container it stands in): only-the-target check, held-but-not-target = wrong_object (released). Empty pads -> one re-grasp at a fresh estimate. Pads never below the table or `pad_floor_z` (a rim; measured: pad_gate)."
    returns = "success when held: metrics.verification_status 'held' | 'held_proprio_only'. failure: already_holding, unreachable, descent missed, closed on nothing, wrong_object, lost in lift. outputs: grasp_xy, grasp_z"
    preconditions = []
    effects = []
    failure_conditions = []
    dependencies = list(_base.Pick.dependencies) + ["verify_grasp"]
    effect_verification = "calibrated opening (close, test lift, lift) + verify_grasp"
    known_limitations = []
    timeout_s = 30.0

    class Params(_base.Pick.Params):
        target_object: str = Field(default="the target object", max_length=120, description="e.g. 'red cube'")
        max_verifications: int = Field(default=3, ge=0, le=5)
        retry_on_not_held: bool = True
        exclusive_of: str = Field(default="", max_length=80, description="what must NOT be held too")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        x, y, z = float(p.xyz[0]), float(p.xyz[1]), float(p.xyz[2])
        if z < ctx.table_z - 0.01:
            raise SkillError(f"grasp point z={z:.3f} is below the table", error_type="precondition")
        holding, width_held = _holding(ctx, p.side)
        if holding:
            return self.result(SkillStatus.failure, f"the {p.side} gripper already holds an object (opening {0 if width_held is None else round(width_held * 100, 1)} cm, fingers commanded closed): nothing moved. {ALREADY_HOLDING_HINT}", verified=True, metrics={"stage": "precondition", "reason": "already_holding", "opening_width_m": width_held, "robot_time_s": 0.0}, outputs={"held": True, "opening_width_m": width_held})
        z, raised = _base._pad_floor(ctx, z, p.pad_floor_z)  # 1.7.0: pads never commanded into the table (thin objects: false "held" on a blocked close); 1.8.0: nor below the caller's floor (a container rim)
        pad_gate: dict | None = None
        wanted_approach = p.approach_height
        if p.object_top_z is not None:
            wanted_approach = max(wanted_approach, float(p.object_top_z) - z + 0.03)
        approach = _reachable_height(ctx, p.side, x, y, z, wanted_approach, p.yaw)
        if approach is None:
            return self.result(SkillStatus.failure, f"grasp point {np.round([x, y, z], 3).tolist()} is not reachable top-down by the {p.side} arm", verified=True, metrics={"stage": "approach", "reachable": False})
        cleared = _clear_other_arm(ctx, p.side, x, y)
        verifications = 0
        attempts = 0
        last_verify: dict = {}
        verifier = "not_called"  # held | not_held | uncertain | unavailable | not_called (last verify_grasp outcome)
        down = None
        retried = False
        poses_tried: list[list[float]] = []
        reacquired: dict | None = None
        conflict: dict | None = None
        test_lift: dict | None = None
        while True:
            attempts += 1
            above, approach = _approach(ctx, p.side, x, y, z, approach, p.yaw, p.open_aperture)
            if above.status != SkillStatus.success:
                return self.result(SkillStatus.failure, f"approach failed: {above.message}", verified=True, metrics={"stage": "approach", "approach_height": approach, "other_arm_cleared": cleared, "robot_time_s": _robot_time(ctx, tick0)})
            _, before_ref = ctx.capture(f"{p.side}_wrist", tag="pre_grasp")
            down, retried = _descend(ctx, p.side, (x, y, z), p.yaw, approach)
            if down.status != SkillStatus.success:
                ctx.call("move_to", side=p.side, xyz=[x, y, z + approach], yaw=p.yaw)
                return self.result(SkillStatus.failure, f"descent failed: {down.message}{_edge_hint(down)}", verified=True, metrics={"stage": "descend", "descent_retried": retried, **_descent_metrics(down), "robot_time_s": _robot_time(ctx, tick0)})
            if p.pad_floor_z is not None:
                # 1.8.0 measured pad gate (formal_v13 abc_extract s103: a descent accepted 0.6 cm low put the pads 3 mm below the cup rim and the
                # fingers closed on peg AND cup): the pads must clear the caller's floor before the fingers close
                pad_gate = _base._pad_gate(ctx, p.side, x, y, z, p.yaw, float(p.pad_floor_z))
            ctx.call("set_gripper", side=p.side, aperture=p.grasp_aperture, duration=0.5)
            held_now, width_now = _held(ctx, p.side)
            status = "not_held" if not held_now else "unchecked"
            if held_now and verifications < p.max_verifications:
                last_verify = _verify(ctx, p.side, p.target_object, "closed the gripper at the grasp point", [before_ref.path] if before_ref else [])
                verifications += 1
                verifier = _verifier_kind(last_verify)
                status = last_verify["status"]  # an unavailable verifier reports "uncertain": proprioception decides below
            if held_now and status == "not_held":
                # the calibrated opening reports an object between the pads while the model says not_held: never reopen on
                # the model's word alone (every such verdict in the truth-checked evidence was a false negative with the
                # block inside); the test lift decides whether something follows the gripper
                conflict = {"stage": "closure", "opening_width_m": width_now, "verdict": _verify_brief(last_verify), "resolved_by": None}
                status = "uncertain"
            if status in ("held", "unchecked", "uncertain"):
                # test lift, then recheck (this is where "object follows the gripper" becomes visible); a closure the model
                # confirmed as held is not re-verified, only checked by proprioception
                z_before = float(ctx.tip(p.side)[2])
                ctx.call("move_to", side=p.side, xyz=[x, y, z + TEST_LIFT_M], yaw=p.yaw, correction_passes=0)
                rise = float(ctx.tip(p.side)[2]) - z_before
                held_lift, width_lift = _held(ctx, p.side)
                test_lift = {"rise_m": round(rise, 4), "opening_width_m": width_lift, "held": bool(held_lift)}
                if not held_lift or rise < TEST_LIFT_MIN_RISE_M:
                    status = "not_held"  # nothing follows the gripper (or the fingers hold something that cannot be lifted)
                    if conflict is not None:
                        conflict["resolved_by"] = "test_lift:not_held"
                elif status == "held":
                    pass  # model-confirmed at the closure and followed the gripper: no second model call
                elif verifications < p.max_verifications:
                    last_verify = _verify(ctx, p.side, p.target_object, f"lifted {TEST_LIFT_M * 100:.0f} cm after closing", [before_ref.path] if before_ref else [])
                    verifications += 1
                    verifier = _verifier_kind(last_verify)
                    status = last_verify["status"]
                    if status == "not_held":
                        # contradicted a second time by an opening held through the lift: proprioception decides
                        conflict = {"stage": "test_lift", "opening_width_m": width_lift, "rise_m": round(rise, 4), "verdict": _verify_brief(last_verify), "resolved_by": "proprioception"}
                        status = "held_proprio_only"
                    elif conflict is not None:
                        conflict["resolved_by"] = f"test_lift:{status}"
                elif status == "unchecked":
                    status = "held_proprio_only"
                elif conflict is not None:
                    conflict["resolved_by"] = "proprioception (no verification budget left)"
                    status = "held_proprio_only"
            if status in ("held", "held_proprio_only", "uncertain"):
                break
            # not_held -> bounded recovery
            poses_tried.append([round(x, 4), round(y, 4), round(z, 4)])
            ctx.call("set_gripper", side=p.side, aperture=p.open_aperture, duration=0.3)
            ctx.call("move_to", side=p.side, xyz=[x, y, z + approach], yaw=p.yaw)
            common = {"attempts": attempts, "verifications": verifications, "last_verify": last_verify, "poses_tried": poses_tried, "reacquisition": reacquired, "test_lift": test_lift, **_descent_metrics(down)}
            if not p.retry_on_not_held or attempts >= 2:
                return self.result(SkillStatus.failure, f"grasp not confirmed after {attempts} attempt(s) at {poses_tried}: {last_verify.get('notes', 'fingers closed on nothing')}{_edge_hint(down)}", verified=True, metrics={"stage": "grasp", "reason": "grasp_not_confirmed", **common, "robot_time_s": _robot_time(ctx, tick0)})
            # re-acquire the object - never an identical blind retry (the v5 formal failures repeated the same stale point):
            # (1) restricted re-localisation, requested colour within RELOCALISE_RADIUS_M (the unrestricted 5 cm 'any' search
            #     re-targeted a different object 4.6 cm away, failure analysis P6);
            # (2) if that returns the same estimate: retract to an observation pose (arm out of the top camera's view) and
            #     search wider (REACQUIRE_RADIUS_M) for the requested object / colour;
            # the retry only happens with an UPDATED estimate whose approach column is reachable.
            nx, ny = _relocalise(ctx, p.target_object, x, y, z)
            source = f"locate_objects:near_{RELOCALISE_RADIUS_M}"
            if math.hypot(nx - x, ny - y) < MIN_UPDATE_M:
                obs = _observation_pose(ctx, p.side, x, y, z, approach)
                if obs is not None:
                    ctx.call("move_to", side=p.side, xyz=obs, yaw=p.yaw, correction_passes=0)
                found, source = _reacquire(ctx, p.target_object, x, y, z)
                if found is None or math.hypot(found[0] - x, found[1] - y) < MIN_UPDATE_M:
                    return self.result(SkillStatus.failure, f"fingers closed on nothing at {poses_tried[-1]} and the object could not be re-acquired (restricted {RELOCALISE_RADIUS_M * 100:.1f} cm and wider {REACQUIRE_RADIUS_M * 100:.0f} cm search: {source}); no blind retry - re-observe (segment_objects) and pick with an updated position", verified=True, metrics={"stage": "grasp", "reason": "reacquisition_failed", **common, "reacquisition": {"source": source, "found": None if found is None else [round(found[0], 4), round(found[1], 4)]}, "robot_time_s": _robot_time(ctx, tick0)})
                nx, ny = found
            new_approach = _reachable_height(ctx, p.side, nx, ny, z, wanted_approach, p.yaw)
            if new_approach is None:
                return self.result(SkillStatus.failure, f"fingers closed on nothing at {poses_tried[-1]}; the re-acquired estimate ({nx:.3f}, {ny:.3f}) is not reachable top-down by the {p.side} arm - use the other arm or pull_toward_base", verified=True, metrics={"stage": "grasp", "reason": "reacquisition_failed", **common, "reacquisition": {"source": source, "found": [round(nx, 4), round(ny, 4)], "reachable": False}, "robot_time_s": _robot_time(ctx, tick0)})
            reacquired = {"source": source, "from": [round(x, 4), round(y, 4)], "to": [round(nx, 4), round(ny, 4)], "shift_m": round(float(math.hypot(nx - x, ny - y)), 4)}
            x, y, approach = nx, ny, new_approach
            conflict, test_lift = None, None
        lift = _reachable_height(ctx, p.side, x, y, z, p.lift_height, p.yaw) or approach
        ctx.call("move_to", side=p.side, xyz=[x, y, z + lift], yaw=p.yaw, correction_passes=0)
        held_after, width_after = _held(ctx, p.side)
        if not held_after:
            return self.result(SkillStatus.failure, "object lost during lift", verified=True, metrics={"stage": "lift", "verifications": verifications, "verifier": verifier, "width_after_lift_m": width_after, "verifier_conflict": conflict, "robot_time_s": _robot_time(ctx, tick0)})
        # the calibrated opening says something is held through the lift: the model confirms ("held") or proprioception
        # alone decides ("held_proprio_only": verifier unavailable / uncertain / contradicted / not called) - never "unknown"
        if status == "held":
            final_status, reason = "held", "model-verified"
        else:
            final_status = "held_proprio_only"
            if conflict is not None and conflict.get("resolved_by") == "proprioception":
                reason = f"verifier said not_held twice, but the calibrated opening ({round(width_after * 100, 1) if width_after is not None else '?'} cm) held through the {TEST_LIFT_M * 100:.0f} cm test lift and the lift: proprioception decides"
            else:
                reason = {"unavailable": f"verifier unavailable: {last_verify.get('unavailable_reason', 'no model answer')}", "uncertain": f"verifier uncertain: {last_verify.get('notes') or 'no decisive view'}", "not_called": "verifier not called (max_verifications=0 or budget)"}.get(verifier, f"verifier {verifier}")
        msg = f"grasp held; lifted {lift:.2f} m (target confirmed: {last_verify.get('holding_target')})" if final_status == "held" else f"grasp held by proprioception only (opening {round(width_after * 100, 1) if width_after is not None else '?'} cm through the lift; {reason}); lifted {lift:.2f} m"
        if raised >= 0.005:  # a thin object (handle, nut); millimetre raises on ordinary blocks are not worth the agent's attention
            msg += f" (grasp height raised {raised * 100:.1f} cm to the pad floor: tip z {z:.3f})"
        return self.result(SkillStatus.success, msg, verified=final_status == "held", metrics={"stage": "lift", "verification_status": final_status, "verifier": verifier, "verifier_reason": reason, "verifier_conflict": conflict, "verifications": verifications, "attempts": attempts, "poses_tried": poses_tried, "reacquisition": reacquired, "test_lift": test_lift, "width_after_lift_m": width_after, "holding_target": last_verify.get("holding_target"), "verifier_truncated": bool(last_verify.get("truncated")), "lift_height": lift, "descent_retried": retried, "other_arm_cleared": cleared, "grasp_z_raised_m": raised, "pad_gate": pad_gate, **_descent_metrics(down), "robot_time_s": _robot_time(ctx, tick0)}, outputs={"lift_pos": np.round(ctx.tip(p.side), 4).tolist(), "held": True, "opening_width_m": width_after, "verification_status": final_status, "verification": last_verify, "grasp_xy": [round(x, 4), round(y, 4)], "grasp_z": round(z, 4)})


class PlaceVerified(_base.Place):
    version = "1.6.0"
    description = _base.Place.description + " Opening < 2 cm before transport: verify_grasp once; not_held aborts only with closed fingers (metrics.verifier_conflict)."
    dependencies = list(_base.Place.dependencies) + ["verify_grasp"]

    class Params(_base.Place.Params):
        target_object: str = Field(default="the held object", max_length=120)
        verify_before_transport: bool = True

    def run(self, ctx, p):
        conflict = None
        if p.verify_before_transport:
            is_held, width = _held(ctx, p.side)
            r = ctx.gripper_reading(p.side)
            weak = (not r.valid) or (width is not None and width < 0.02)
            if is_held and weak:
                v = _verify(ctx, p.side, p.target_object, "about to transport the object", [])
                if v.get("status") == "not_held":
                    now = ctx.gripper_reading(p.side)
                    w_now = now.opening_width_m
                    if now.valid and w_now is not None and now.held_by_width:
                        # the calibrated opening still reports an object between the pads (width rule of the calibration): the
                        # model's negative alone does not abort a transport (cand_20260910_100451_4ef66a; false negatives)
                        conflict = {"stage": "pre_transport", "opening_width_m": round(float(w_now), 4), "verdict": _verify_brief(v), "resolved_by": "proprioception"}
                        ctx.log(f"place: verify_grasp said not_held but the {p.side} opening is {w_now * 100:.1f} cm (valid reading): transporting")
                    else:
                        return self.result(SkillStatus.failure, f"verify_grasp reports not_held before transport and the fingers are closed (opening {0 if w_now is None else round(float(w_now) * 100, 1)} cm)", verified=True, metrics={"stage": "precondition", "verification": v})
        res = super().run(ctx, p)
        if conflict is not None:
            res.metrics["verifier_conflict"] = conflict
        return res


def _margin_at(ctx, side: str, xyz, yaw: float) -> float | None:
    sol = ctx.controller.solve_ik_detailed(side, xyz, yaw)
    return None if sol is None else round(float(sol.limit_margin_rad), 3)


def _route(side: str, reach: dict, allow_transfer: bool) -> tuple[str, str] | None:
    """(pick_side, place_side) from the per-arm reach table, or None when no route exists."""

    def score(s):
        return min(reach[s]["pick_margin"] or 0.0, reach[s]["place_margin"] or 0.0)

    both = [s for s in ("left", "right") if reach[s]["pick"] and reach[s]["place"]]
    if side != "auto":
        other = "right" if side == "left" else "left"
        if reach[side]["pick"] and reach[side]["place"]:
            return side, side
        if reach[side]["pick"] and allow_transfer and reach[other]["place"]:
            return side, other
        return None
    if both:
        best = max(both, key=score)
        return best, best
    pickers = [s for s in ("left", "right") if reach[s]["pick"]]
    placers = [s for s in ("left", "right") if reach[s]["place"]]
    if allow_transfer and pickers and placers:
        return max(pickers, key=lambda s: reach[s]["pick_margin"] or 0.0), max(placers, key=lambda s: reach[s]["place_margin"] or 0.0)
    return None


class PickAndPlaceVerified(_base.PickAndPlace):
    version = "1.4.0"
    description = "Pick at pick_xyz and place at place_xyz. side='auto' takes the arm whose approach and release columns are both reachable (larger joint-limit margin wins); when no single arm reaches both - or the given arm cannot reach the release column - and allow_transfer is true, the object is picked by the arm that reaches it, handed over with transfer_between_arms and placed by the other arm: one call for a cross-side relay. The route is decided by IK before any motion (a route-less call moves nothing)."
    returns = "success if pick reported a held object (model-verified or held_proprio_only) and place released it; metrics.route {pick_side, place_side, transfer}, reach (per arm), pick_verification, place metrics incl. landed_early, robot_time_s. failure: no arm reaches the grasp / release column (message names the arms that do), pick / transfer / place failed (outputs.giver_still_holding after a failed transfer)"
    dependencies = ["pick", "place", "transfer_between_arms"]

    class Params(_base.PickAndPlace.Params):
        side: Literal["left", "right", "auto"] = Field(description="arm, or 'auto' = chosen by reach")
        target_object: str = Field(default="the target object", max_length=120)
        allow_transfer: bool = Field(default=True, description="hand over via transfer_between_arms when the picking arm cannot reach the release column")

    def run(self, ctx, p):
        tick0 = ctx.backend.control_step_index
        px, py, pz = _xyz(p.pick_xyz)
        qx, qy, qz = _xyz(p.place_xyz)
        z_low = qz + p.release_height
        wanted_place = max(p.lift_height - p.release_height, 0.03)
        reach = {}
        for s in ("left", "right"):
            can_pick = _reachable_height(ctx, s, px, py, pz, 0.10, p.pick_yaw) is not None
            can_place = _reachable_height(ctx, s, qx, qy, z_low, wanted_place, p.place_yaw, minimum=MIN_TRANSIT_M) is not None
            reach[s] = {"pick": can_pick, "place": can_place, "pick_margin": _margin_at(ctx, s, [px, py, pz], p.pick_yaw), "place_margin": _margin_at(ctx, s, [qx, qy, z_low], p.place_yaw)}
        route = _route(p.side, reach, p.allow_transfer)
        if route is None:
            pickers = [s for s in ("left", "right") if reach[s]["pick"]]
            placers = [s for s in ("left", "right") if reach[s]["place"]]
            who = f"grasp column reachable by {pickers or 'neither arm'}, release column by {placers or 'neither arm'}"
            hint = "; pull_toward_base or a different landing point" if not pickers or not placers else "; use side='auto' or allow_transfer"
            return self.result(SkillStatus.failure, f"no route for side={p.side!r}: {who}{hint}", verified=True, metrics={"stage": "route", "reachable": False, "reach": reach, "route": None, "robot_time_s": 0.0})
        pick_side, place_side = route
        route_rec = {"pick_side": pick_side, "place_side": place_side, "transfer": pick_side != place_side}
        pick = ctx.call("pick", side=pick_side, xyz=p.pick_xyz, yaw=p.pick_yaw, lift_height=p.lift_height, target_object=p.target_object)
        if pick.status == SkillStatus.failure:
            return self.result(SkillStatus.failure, f"pick failed ({pick_side}): {pick.message}", verified=True, metrics={"stage": "pick", "route": route_rec, "reach": reach, **pick.metrics, "robot_time_s": _robot_time(ctx, tick0)})
        transfer = None
        if pick_side != place_side:
            from physapi.perception.sam3 import PROMPT_COLOR_WORDS_V2, prompt_color

            colour = prompt_color(p.target_object or "", PROMPT_COLOR_WORDS_V2)
            object_height = float(np.clip(2.0 * (pz - ctx.table_z), 0.01, 0.12))
            kw = {"relocate_color": colour} if colour else {}
            transfer = ctx.call("transfer_between_arms", giver=pick_side, receiver=place_side, object_height=round(object_height, 3), regrasp_yaw=0.0, **kw)
            if transfer.status != SkillStatus.success:
                return self.result(SkillStatus.failure, f"transfer {pick_side} -> {place_side} failed: {transfer.message}", verified=True, metrics={"stage": "transfer", "route": route_rec, "reach": reach, "pick_verification": pick.metrics.get("verification_status"), "transfer": transfer.metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs={"giver_still_holding": (transfer.outputs or {}).get("giver_still_holding"), "staging_xy": (transfer.outputs or {}).get("staging_xy")})
        place = ctx.call("place", side=place_side, xyz=p.place_xyz, yaw=p.place_yaw, transit_height=p.lift_height, release_height=p.release_height, target_object=p.target_object)
        ok = place.status == SkillStatus.success
        status = SkillStatus.success if ok and pick.status == SkillStatus.success else (SkillStatus.unknown if ok else SkillStatus.failure)
        return self.result(status, ("placed" if ok else f"place failed ({place_side}): {place.message}") + (f" [relay: picked {pick_side}, transferred, placed {place_side}]" if route_rec["transfer"] else ""), verified=pick.status == SkillStatus.success and ok, metrics={"stage": "place", "route": route_rec, "reach": reach, "pick_verification": pick.metrics.get("verification_status"), "transfer_robot_time_s": None if transfer is None else round(float(transfer.duration_s), 3), **place.metrics, "robot_time_s": _robot_time(ctx, tick0)}, outputs=place.outputs)


class TransferBetweenArmsVerified(_base.TransferBetweenArms):
    """Superseded by skills/transfer_between_arms.py (disabled in the manifest, kept for lineage)."""

    version = "1.1.0"
    description = _base.TransferBetweenArms.description + " The receiver's regrasp uses the verified pick; the giver releases only after placing (table-mediated), so no in-air release without a confirmed receiver grasp occurs."
    dependencies = list(_base.TransferBetweenArms.dependencies) + ["verify_grasp"]
