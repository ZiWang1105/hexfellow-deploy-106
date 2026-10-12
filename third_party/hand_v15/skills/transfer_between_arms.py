"""Dynamic reachable shared-zone selection and height clamping for ``transfer_between_arms`` (library ``hand_v5``).

Lineage: round1b candidate cand_20260909_074042_0c47c9 (dynamic zone, height clamping) -> hand_v3 fix
cand_20260909_210618_424b99 (the four call sites that read a non-existent ``SkillResult`` success attribute raised
AttributeError after the giver had already released; they test ``status == success``) -> candidate cand_20260909_221511_026ccd:
the shared zone must keep >= MIN_ZONE_MARGIN_RAD joint-limit margin for BOTH arms (a point on an arm's reach limit is
"reachable" for the IK but the servos land 3-5 cm off there, so the giver's release missed the zone), the margin enters
the zone score, and a giver whose place failed retreats to its own side (holding or not) instead of staying parked at the
midline where the receiver's approach would collide with it. The hand_v3 crash fix and the ``(outputs or {})`` guards are kept.

Candidate cand_20260910_121426_b48287 (this file, 1.4.0; integrated unchanged into library ``hand_v6``@3.2.0): robot time. The 4-object relays of sort_cubes / abc_sort ran at the
60 s budget edge even for the oracle (two transfers of 10-11 s each; sort_cubes s1 timed out holding the last cube). The
giver's carry height over the staging point drops from 12 to 8 cm (a ceiling pose is slow and 1-3 cm off), the staging
release from 2 to 1 cm above the object centre (pads ~0.8 cm above the table: the 2 cm drop bounced blocks into diagonal
regrasps), the receiver's approach / lift from 8 / 10 to 6 / 8 cm, and the giver's retreat runs at the distance-scaled
duration instead of a fixed 1.2 s. Together ~1.5-2 s per transfer; behaviour otherwise unchanged.

hand_v8@3.4.0 (parent hand_v7): the held / empty decision is no longer a constant in this library. ``_held`` reads the reading's calibrated width rule (``GripperReading.held_by_width``: opening >= the backend/side ``held_width_min_m`` of configs/gripper/<backend>_<side>.yaml; ambiguous and empty count as not held) and ``_held_width_min`` exposes that threshold to the pinch / transfer checks. No other behaviour change.
"""

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from physapi.schemas.core import SkillStatus
from physapi.skills.base import Params, Skill, SkillError

_spec = importlib.util.spec_from_file_location(f"physapi_libmod_{hashlib.sha1(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]}_manipulation_base", Path(__file__).with_name("manipulation.py"))
if _spec.name in sys.modules:  # one base module per library (see manipulation_gv.py)
    _base = sys.modules[_spec.name]
else:
    _base = importlib.util.module_from_spec(_spec)
    assert _spec.loader
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
_held = _base._held  # hand_v8: giver_still_holding from the calibrated width rule (was a 1 cm constant)

MIN_ZONE_MARGIN_RAD = 0.05
#: Carry / release heights of the giver's staging place and approach / lift heights of the receiver's regrasp [m]
TRANSIT_H_M = 0.08
RELEASE_H_M = 0.01
APPROACH_H_M = 0.06
LIFT_H_M = 0.08


def _check_ik(ctx, side: str, xyz: list[float], yaw: float) -> bool:
    if hasattr(ctx, "controller") and hasattr(ctx.controller, "solve_ik"):
        try:
            sol = ctx.controller.solve_ik(side, xyz, yaw)
            return sol is not None
        except Exception:
            pass
    try:
        res = ctx.call("solve_ik", side=side, xyz=xyz, yaw=yaw)
        return res.status == SkillStatus.success
    except Exception:
        return False


def _reachable_height(
    ctx, side: str, x: float, y: float, yaw: float, z_min: float
) -> Optional[float]:
    if not _check_ik(ctx, side, [x, y, z_min], yaw):
        return None
    highest = z_min
    max_z = min(z_min + 0.25, 0.95)
    z = z_min + 0.02
    while z <= max_z + 1e-4:
        if _check_ik(ctx, side, [x, y, z], yaw):
            highest = z
        else:
            break
        z += 0.02
    return highest


def _zone_margin(ctx, giver: str, receiver: str, x: float, y: float, z: float, regrasp_yaw: float) -> Optional[float]:
    """Smallest joint-limit margin of the giver's release pose and the receiver's grasp pose at a candidate zone."""

    ctrl = getattr(ctx, "controller", None)
    if ctrl is None or not hasattr(ctrl, "solve_ik_detailed"):
        return 0.1
    g = ctrl.solve_ik_detailed(giver, [x, y, z + 0.02], 0.0)
    r = ctrl.solve_ik_detailed(receiver, [x, y, z], regrasp_yaw)
    if g is None or r is None:
        return None
    return float(min(g.limit_margin_rad, r.limit_margin_rad))


class TransferBetweenArms(Skill):
    name = "transfer_between_arms"
    version = "1.5.0"
    kind = "code"
    is_query = False
    description = (
        "Move a held object from one arm to the other via the table: picks a shared staging point both arms reach top-down "
        "with joint-limit margin (or `staging_xy`), clamps transit / release / approach heights to each arm's reach ceiling, "
        "places the object with the giver, retreats the giver to its own side, and regrasps with the receiver's verified pick "
        "(optionally re-localised by `relocate_color`)."
    )
    returns = (
        "success: outputs.staging_xy, release_pos, lift_pos, grasp_aperture; failure: no shared zone with margin, giver place failed "
        "(giver retreated; outputs.giver_still_holding), receiver regrasp failed"
    )
    preconditions = ["giver holds an object", "giver != receiver"]
    effects = []
    failure_conditions = []
    effect_verification = "receiver's pick verification (calibrated opening + verify_grasp)"
    dependencies = ["place", "pick", "move_to", "locate_objects", "solve_ik", "wait"]
    known_limitations = ["table-mediated (no in-air handover with top-down wrists); ~8-10 s of robot time"]
    timeout_s = 30.0

    class Params(Params):
        giver: Literal["left", "right"] = Field(..., description="arm currently holding the object")
        receiver: Literal["left", "right"] = Field(..., description="arm to receive the object")
        object_height: float = Field(default=0.03, ge=0.005, le=0.3, description="object height [m] (release / regrasp heights)")
        regrasp_yaw: float = Field(default=0.0, description="wrist yaw for the receiver's grasp")
        relocate_color: Optional[str] = Field(default=None, description="re-localise the object by this colour before regrasping")
        staging_xy: Optional[list[float]] = Field(default=None, description="optional shared staging point [x, y]; default: searched")

    def run(self, ctx, p: Params):
        if p.giver == p.receiver:
            raise SkillError("giver and receiver must be different arms", error_type="precondition")

        if p.staging_xy is not None and len(p.staging_xy) != 2:
            raise SkillError("staging_xy must be a 2-element list [x, y]", error_type="parameter")

        table_z = getattr(ctx, "table_z", 0.75)
        z_center = table_z + p.object_height / 2.0

        candidates: list[list[float]] = []
        if p.staging_xy is not None:
            candidates.append([float(p.staging_xy[0]), float(p.staging_xy[1])])

        # Candidate shared staging points centered around the dual-arm overlap zone
        for x in [0.46, 0.48, 0.45, 0.47, 0.49, 0.50]:
            for y in [-0.04, -0.02, -0.05, -0.03, -0.06, -0.01, 0.00, -0.08]:
                pt = [round(x, 3), round(y, 3)]
                if pt not in candidates:
                    candidates.append(pt)

        best_pt: Optional[list[float]] = None
        best_g_ceiling: float = 0.0
        best_r_ceiling: float = 0.0
        best_score: float = -float("inf")

        for pt in candidates:
            cx, cy = pt[0], pt[1]
            g_ceil = _reachable_height(ctx, p.giver, cx, cy, yaw=0.0, z_min=z_center)
            if g_ceil is None:
                continue

            r_ceil = _reachable_height(ctx, p.receiver, cx, cy, yaw=p.regrasp_yaw, z_min=z_center)
            if r_ceil is None:
                continue

            g_clearance = g_ceil - z_center
            r_clearance = r_ceil - z_center

            # place requires transit_height >= 0.05; pick requires approach_height >= 0.03
            if g_clearance < 0.052 or r_clearance < 0.035:
                continue

            # both arms must have joint-limit margin at the zone: a point on an arm's reach limit is "reachable" for
            # the IK but the servos land 3-5 cm off there (the giver's release then misses the zone)
            margin = _zone_margin(ctx, p.giver, p.receiver, cx, cy, z_center, p.regrasp_yaw)
            if margin is None or margin < MIN_ZONE_MARGIN_RAD:
                continue

            # Prioritize caller-specified staging_xy if valid
            if p.staging_xy is not None and pt == [float(p.staging_xy[0]), float(p.staging_xy[1])]:
                best_pt = pt
                best_g_ceiling = g_ceil
                best_r_ceiling = r_ceil
                break

            # Score by minimum clearance and joint-limit margin for both arms with slight preference for central staging
            score = min(g_clearance, r_clearance) + 0.3 * margin - 0.02 * (abs(cx - 0.47) + abs(cy - (-0.04)))
            if score > best_score:
                best_score = score
                best_pt = pt
                best_g_ceiling = g_ceil
                best_r_ceiling = r_ceil

        if best_pt is None:
            raise SkillError(
                f"No shared staging coordinate found reachable by both arms with joint-limit margin for object_height={p.object_height:.3f}m",
                error_type="reachability",
            )

        staging_x, staging_y = best_pt[0], best_pt[1]

        # Clamp giver transit and release heights within its reach ceiling
        # carry 8 cm (not 12) over the staging point: a ceiling pose is slow and lands 1-3 cm off; release 1 cm (not 2) above
        # the object centre: the pads end ~0.8 cm above the table and the object is set down instead of dropped
        max_g_transit = best_g_ceiling - z_center - 0.005
        transit_h = max(0.05, min(TRANSIT_H_M, max_g_transit))

        max_g_release = best_g_ceiling - z_center - 0.005
        release_h = max(0.005, min(RELEASE_H_M, max_g_release))

        ctx.log(
            f"transfer_between_arms staging at ({staging_x:.3f}, {staging_y:.3f}) | "
            f"giver ceil: {best_g_ceiling:.3f}m, transit_h: {transit_h:.3f}m, release_h: {release_h:.3f}m"
        )

        # Place object with giver arm
        place_res = ctx.call(
            "place",
            side=p.giver,
            xyz=[staging_x, staging_y, z_center],
            yaw=0.0,
            transit_height=transit_h,
            release_height=release_h,
            verify_before_transport=True,
        )
        if place_res.status != SkillStatus.success:
            # do not leave the giver parked at the midline (the receiver's next approach would collide with it)
            tip = ctx.tip(p.giver)
            still_holding, width_now = _held(ctx, p.giver)
            ctx.call("move_to", side=p.giver, xyz=[float(tip[0]), float(tip[1]), min(float(tip[2]) + 0.08, table_z + 0.15)], yaw=0.0, tolerance_m=0.05, correction_passes=0)
            ctx.call("move_to", side=p.giver, xyz=[0.40, 0.22 if p.giver == "left" else -0.22, table_z + 0.15], yaw=0.0, tolerance_m=0.05, correction_passes=0)
            return self.result(
                SkillStatus.failure,
                f"Giver {p.giver} failed to place object in shared zone: {place_res.message}",
                verified=False,
                outputs={"staging_xy": [staging_x, staging_y], "giver_still_holding": bool(still_holding), "giver_opening_width_m": width_now},
            )

        # Retreat giver arm away from the shared staging zone to avoid arm collision
        retreat_y = 0.22 if p.giver == "left" else -0.22
        retreat_x = 0.38
        retreat_z = min(table_z + 0.15, 0.90)
        ctx.call(
            "move_to",
            side=p.giver,
            xyz=[retreat_x, retreat_y, retreat_z],
            yaw=0.0,
            tolerance_m=0.04,
            correction_passes=0,
        )
        ctx.call("wait", seconds=0.2)

        # Determine receiver pick target (optionally re-localised)
        pick_xy = [staging_x, staging_y]
        if p.relocate_color:
            loc_res = ctx.call(
                "locate_objects",
                color=p.relocate_color,
                object_height_m=p.object_height,
                max_results=5,
            )
            if loc_res.status == SkillStatus.success and loc_res.outputs and loc_res.outputs.get("detections"):
                dets = loc_res.outputs["detections"]
                best_d = float("inf")
                best_det_xy = None
                for d in dets:
                    d_xyz = d.get("world_xyz", [])
                    if len(d_xyz) >= 2:
                        dist = math.hypot(d_xyz[0] - staging_x, d_xyz[1] - staging_y)
                        if dist < 0.10 and dist < best_d:
                            best_d = dist
                            best_det_xy = [float(d_xyz[0]), float(d_xyz[1])]
                if best_det_xy is not None:
                    pick_xy = best_det_xy
                    ctx.log(f"Re-localised {p.relocate_color} object at {pick_xy}")

        # Clamp receiver approach and lift heights within reach ceiling at pick_xy
        r_ceil = _reachable_height(
            ctx, p.receiver, pick_xy[0], pick_xy[1], yaw=p.regrasp_yaw, z_min=z_center
        )
        if r_ceil is None:
            r_ceil = best_r_ceiling

        max_r_approach = r_ceil - z_center - 0.005
        approach_h = max(0.03, min(APPROACH_H_M, max_r_approach))

        max_r_lift = r_ceil - z_center - 0.005
        lift_h = max(0.03, min(LIFT_H_M, max_r_lift))

        # Regrasp with receiver arm using verified pick
        pick_kwargs = {"target_object": f"{p.relocate_color} object"} if p.relocate_color else {}  # lets pick's re-acquisition search by colour
        pick_res = ctx.call(
            "pick",
            side=p.receiver,
            xyz=[pick_xy[0], pick_xy[1], z_center],
            yaw=p.regrasp_yaw,
            approach_height=approach_h,
            lift_height=lift_h,
            object_top_z=table_z + p.object_height,
            retry_on_not_held=True,
            **pick_kwargs,
        )
        if pick_res.status != SkillStatus.success:
            return self.result(
                SkillStatus.failure,
                f"Receiver {p.receiver} failed to pick object at shared zone: {pick_res.message}",
                verified=False,
                outputs={"staging_xy": [staging_x, staging_y], "giver_still_holding": False},
            )

        return self.result(
            SkillStatus.success,
            f"Successfully transferred object from {p.giver} to {p.receiver} via shared zone at ({staging_x:.3f}, {staging_y:.3f})",
            verified=True,
            outputs={
                "staging_xy": [staging_x, staging_y],
                "release_pos": (place_res.outputs or {}).get("release_pos"),
                "lift_pos": (pick_res.outputs or {}).get("lift_pos"),
                "grasp_aperture": (pick_res.outputs or {}).get("grasp_aperture"),
            },
        )


def CANDIDATE_SMOKE(ctx) -> dict:
    table_z = getattr(ctx, "table_z", 0.75)
    z_center = table_z + 0.03 / 2.0
    candidates = [
        [0.47, -0.04],
        [0.46, -0.04],
        [0.48, -0.02],
        [0.45, -0.03],
        [0.49, -0.05],
    ]
    found = False
    valid_pt = None
    for cx, cy in candidates:
        g = _reachable_height(ctx, "left", cx, cy, yaw=0.0, z_min=z_center)
        r = _reachable_height(ctx, "right", cx, cy, yaw=0.0, z_min=z_center)
        if g is not None and r is not None and (g - z_center >= 0.052) and (r - z_center >= 0.035):
            found = True
            valid_pt = (cx, cy)
            break
    if not found:
        return {"ok": False, "notes": "No valid shared staging point found for both arms"}
    return {"ok": True, "notes": f"Shared staging candidate verified at {valid_pt}"}
