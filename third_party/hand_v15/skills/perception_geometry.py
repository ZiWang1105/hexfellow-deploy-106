"""Geometry-returning perception (``hand_v4``, from candidate cand_20260909_214243_dc0358; failure classes C3/C4, proposal P5).

Overrides the core primitives ``segment_objects``, ``locate_objects`` and ``pixel_to_world`` under the same names so
callers and traces stay compatible. Every first-generation output key is kept unchanged; the additions are:

* per detection: world-frame footprint (oriented box: centre, size, yaw), the four footprint corners, four faces
  (edge midpoints + outward normals), ``grasp_yaw``, ``top_z`` / ``estimated_height_m`` with an explicit
  ``height_source``, mask area and a ``quality`` block (see :func:`physapi.perception.geometry.mask_geometry`);
* ``segment_objects``: when SAM3 returns no mask the result says so explicitly and carries the colour / dark-on-light
  / depth fallback blobs in the SAME call (``status = unknown``, ``outputs.fallback_detections``), never a silent zero;
* ``locate_objects``: "brown", "dark"/"black", "dark_on_light" and "raised" (depth) modes; the plane height defaults
  to the depth image instead of a fixed 3 cm (``height_source`` reported);
* ``pixel_to_world``: batch form ``points=[[u, v], ...]`` (one call, one query unit), depth-based height when
  ``height_m`` is omitted, ``source`` reported per point;
* restricted re-localisation: ``near_xy`` / ``near_radius_m`` on both detectors keep only blobs near a previous estimate.
* near-square footprints (short/long >= :data:`NEAR_SQUARE_RATIO`): both grasp yaws are valid, so ``grasp_yaw`` is the one
  nearer 0 (yaw 0 is the most reachable wrist orientation; +-pi/2 is often unreachable at the far reach edge). Recorded
  follow-up of the candidate's validation (sort_cubes: the agent spent turns checking a +-1.5 rad yaw at x = 0.69).

Observation boundary: top RGB + top depth + camera intrinsics/extrinsics + table height only (deployable_v1).
Every perception call is still recorded by the executor trace, the recorder's image log and the SAM3 client log.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

import numpy as np
from pydantic import Field

from physapi.perception.geometry import PIXEL_CENTRE_OFFSET, dark_on_light_mask, depth_at, depth_blob_masks, deproject, wrap_half_pi
from physapi.perception.locator import COLOR_RANGES_V2, ColorLocator, pixel_to_world
from physapi.perception.sam3 import PROMPT_COLOR_WORDS_V2, Sam3Unavailable, prompt_color
from physapi.schemas.core import SkillStatus
from physapi.skills.base import Params, Skill, SkillError

COLOUR_NAMES = sorted(COLOR_RANGES_V2)
SPECIAL_MODES = ("any", "foreground", "dark_on_light", "raised")
GEOMETRY_KEYS = "footprint {centre_xy, size_xy [long, short], yaw (long axis from +x)}, corners_xy[4], faces[4] = [mid_x, mid_y, nx, ny] (edge midpoint + outward normal), grasp_yaw (across the short axis; squares: the one nearer 0, alt in grasp_yaw_alt), top_z, estimated_height_m, height_source, quality {rectangularity, depth_valid_frac, touches_border, z_spread_m, geometry_confidence}, polygon_xy (non-rectangular)"
#: Geometry is attached to at most this many detections per call, and stripped from the tail beyond
#: PAYLOAD_LIMIT_CHARS, so the JSON result stays under the agent's 6000-character tool-result cap (a detection with
#: geometry is ~700 characters; the SkillResult wrapper and message need ~400).
MAX_GEOMETRY_DETECTIONS = 8
PAYLOAD_LIMIT_CHARS = 5200
#: A footprint whose short/long side ratio is at least this is treated as square for the grasp yaw: both yaws are valid
#: grasps, and the one nearer 0 is reported (most reachable; see the module docstring).
NEAR_SQUARE_RATIO = 0.85


def _prefer_reachable_grasp_yaw(det: dict[str, Any]) -> dict[str, Any]:
    """For a near-square footprint report the grasp yaw nearer 0 (the two candidate yaws are grasp_yaw and grasp_yaw +- pi/2,
    both closing across a side of the same length); records the alternative in ``grasp_yaw_alt``. Elongated shapes keep the
    across-the-short-axis yaw and get no ``grasp_yaw_alt``."""

    fp = det.get("footprint")
    if not fp or det.get("grasp_yaw") is None:
        return det
    size = fp.get("size_xy") or []
    if len(size) != 2 or float(size[0]) <= 1e-6:
        return det
    ratio = float(size[1]) / float(size[0])
    if ratio < NEAR_SQUARE_RATIO:
        return det
    yaw = float(det["grasp_yaw"])
    alt = wrap_half_pi(yaw + math.pi / 2)
    if abs(alt) < abs(yaw):
        det["grasp_yaw"], det["grasp_yaw_alt"] = round(alt, 2), round(yaw, 2)
    else:
        det["grasp_yaw_alt"] = round(alt, 2)
    det["grasp_yaw_note"] = "near-square footprint: both yaws grasp; the one nearer 0 is reported (most reachable)"
    return det


def _v2_locator(ctx, camera: str) -> ColorLocator:
    """ColorLocator with the extended palette, cached per camera in ctx.scratch (extrinsics refreshed every call)."""

    key = f"perception_v2_locator:{camera}"
    loc = ctx.scratch.get(key)
    if loc is None:
        loc = ColorLocator(ctx.backend.camera_intrinsics(camera), ctx.backend.camera_to_world(camera), table_z=ctx.table_z, camera=camera, color_ranges=COLOR_RANGES_V2)
        ctx.scratch[key] = loc
    else:
        loc.cam_to_world = ctx.backend.camera_to_world(camera)
    return loc


def _strip_geometry(det: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in det.items() if k not in ("footprint", "corners_xy", "faces", "grasp_yaw", "grasp_yaw_alt", "grasp_yaw_note", "quality", "polygon_xy")}


def _jsonable(dets, *, geometry: bool, source: str | None = None, label: str | None = None) -> list[dict[str, Any]]:
    out = []
    for i, d in enumerate(dets):
        j = d.jsonable()
        j.setdefault("height_source", "depth" if j.get("top_z") is not None else "assumed")  # never silent about the height's origin
        if label is not None:
            j["label"] = label
        if source is not None:
            j["source"] = source
            j["match_verified"] = False
        if not geometry or i >= MAX_GEOMETRY_DETECTIONS:
            j = _strip_geometry(j)
        else:
            j = _prefer_reachable_grasp_yaw(j)
        out.append(j)
    return out


def _fit_payload(outputs: dict[str, Any], key: str, limit: int = PAYLOAD_LIMIT_CHARS) -> dict[str, Any]:
    """Strip geometry from the lowest-ranked detections until the JSON outputs fit the agent's tool-result cap
    (6000 characters, after which the runtime truncates the JSON mid-way). Reports how many were stripped."""

    dets = outputs.get(key) or []
    stripped = 0
    while dets and len(json.dumps(outputs, separators=(",", ":"), default=str)) > limit:
        idx = next((i for i in range(len(dets) - 1, -1, -1) if "footprint" in dets[i]), None)
        if idx is None:
            break
        dets[idx] = _strip_geometry(dets[idx])
        stripped += 1
    if stripped:
        outputs["geometry_truncated"] = stripped
    return outputs


def _near(params) -> tuple[tuple[float, float] | None, float]:
    near = None if params.near_xy is None else (float(params.near_xy[0]), float(params.near_xy[1]))
    return near, float(params.near_radius_m)


def _fallback(ctx, camera: str, rgb: np.ndarray, depth: np.ndarray, prompt: str, mode: str, *, max_results: int, near_xy, near_radius_m: float, geometry: bool) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Colour / dark-on-light / depth blobs when SAM3 returns nothing. Returns (detections, source, tried)."""

    loc = _v2_locator(ctx, camera)
    tried: list[str] = []
    common = dict(object_height_m=None, depth=depth, geometry=True, near_xy=near_xy, near_radius_m=near_radius_m, max_results=max_results)
    if mode in ("auto", "colour"):
        colour = prompt_color(prompt, PROMPT_COLOR_WORDS_V2)
        if colour is not None:
            tried.append(f"colour:{colour}")
            dets = loc.detect(rgb, colour, **common)
            if dets:
                return _jsonable(dets, geometry=geometry, source=f"fallback:colour:{colour}", label=prompt), f"colour:{colour}", tried
        tried.append("colour:dark_on_light")
        dets = loc.detect(rgb, "dark_on_light", mask=dark_on_light_mask(rgb), **common)
        if dets:
            return _jsonable(dets, geometry=geometry, source="fallback:dark_on_light", label=prompt), "dark_on_light", tried
    if mode in ("auto", "depth"):
        masks, info = depth_blob_masks(depth, loc.K, loc.cam_to_world, table_z=ctx.table_z)
        tried.append(f"depth:raised>={info.get('min_height_m', 0.006)}m")
        if masks:
            combined = np.zeros(depth.shape[:2], dtype=np.uint8)
            for m in masks:
                combined[m] = 255
            dets = loc.detect(rgb, "raised", mask=combined, **common)
            if dets:
                return _jsonable(dets, geometry=geometry, source="fallback:depth_raised", label=prompt), "depth_raised", tried
    return [], "none", tried


class SegmentObjectsGeo(Skill):
    name = "segment_objects"
    is_query = True
    version = "2.0.0"
    kind = "primitive"
    description = (f"SAM3 text-prompted localisation, top camera ('red cube', 'drawer'). Each instance: centroid world_xyz (table plane, half height) + world-frame geometry - {GEOMETRY_KEYS}. Use these, not pixel probes. "
                   "SAM3 miss: status unknown + unverified fallback blobs (colour/dark-on-light/depth) in outputs.fallback_detections, or failure listing what was tried.")
    returns = "outputs.detections (+ label, pixel, bbox, area_px, plane_z, confidence, sam3_score, color_agreement), sam3_found; miss: fallback_source / fallback_tried / fallback_detections (+source, match_verified=false); error 'blocked' if the server is down"
    observations_used = ["rgb:top", "depth:top"]
    effect_verification = "n/a (perception)"
    known_limitations = ["top-down footprint: overhang/occlusion -> quality", "fallback blobs unverified", "robot links may be segmented (height > 0.2)"]
    timeout_s = 4.0

    class Params(Params):
        text_prompt: str = Field(min_length=2, max_length=80)
        max_results: int = Field(default=6, ge=1, le=20)
        camera: Literal["top"] = "top"
        return_geometry: bool = Field(default=True, description="attach geometry")
        fallback: Literal["auto", "colour", "depth", "none"] = Field(default="auto", description="SAM3-miss fallback order")
        near_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="restrict to near_radius_m of this xy")
        near_radius_m: float = Field(default=0.025, ge=0.005, le=0.3)

    def run(self, ctx, params):
        image, ref = ctx.capture(params.camera, tag="segment")
        depth = ctx.capture_depth(params.camera)
        near_xy, radius = _near(params)
        locator = ctx.sam3_locator(params.camera)
        try:
            dets = locator.detect(image, params.text_prompt, depth=depth, max_results=params.max_results, geometry=params.return_geometry, near_xy=near_xy, near_radius_m=radius, color_ranges=COLOR_RANGES_V2)
        except Sam3Unavailable as exc:
            raise SkillError(f"SAM3 segmentation unavailable: {exc}", error_type="blocked") from exc
        calls = getattr(locator.client, "calls", None)
        n_masks = max(len(dets), int(calls[-1].get("n", 0))) if calls else len(dets)
        img = ref.model_dump() if ref else None
        if dets:
            outputs = _fit_payload({"detections": _jsonable(dets, geometry=params.return_geometry), "sam3_found": n_masks, "image": img}, "detections")
            return self.result(SkillStatus.success, f"{len(dets)} '{params.text_prompt}' instance(s) with geometry" + (f" within {radius * 100:.1f} cm of near_xy" if near_xy else ""), verified=True, outputs=outputs)
        if near_xy is not None and n_masks > 0:
            return self.result(SkillStatus.failure, f"SAM3 found {n_masks} '{params.text_prompt}' mask(s) but none within {radius * 100:.1f} cm of near_xy {list(np.round(near_xy, 3))}; widen near_radius_m or drop near_xy", verified=True, outputs={"detections": [], "sam3_found": n_masks, "rejected_far": n_masks, "image": img})
        fb, source, tried = ([], "none", []) if params.fallback == "none" else _fallback(ctx, params.camera, image, depth, params.text_prompt, params.fallback, max_results=params.max_results, near_xy=near_xy, near_radius_m=radius, geometry=params.return_geometry)
        base = f"SAM3 found no '{params.text_prompt}' (0 masks)."
        if fb:
            msg = f"{base} Fallback {source} found {len(fb)} blob(s) - UNVERIFIED match, see outputs.fallback_detections (world_xyz, footprint, corners). Use them if they fit the task, otherwise try a synonym or a colour word."
            outputs = _fit_payload({"detections": [], "sam3_found": 0, "fallback_source": source, "fallback_tried": tried, "fallback_detections": fb, "image": img}, "fallback_detections")
            return self.result(SkillStatus.unknown, msg, verified=True, outputs=outputs)
        msg = f"{base} Fallbacks tried without result: {', '.join(tried) if tried else 'none (fallback=none)'}. Try a synonym, a colour word ('brown patch'), or locate_objects(color='dark_on_light'|'raised')."
        return self.result(SkillStatus.failure, msg, verified=True, outputs={"detections": [], "sam3_found": 0, "fallback_source": "none", "fallback_tried": tried, "fallback_detections": [], "image": img})


class LocateObjectsGeo(Skill):
    name = "locate_objects"
    is_query = True
    version = "2.0.0"
    kind = "primitive"
    description = (f"RGB colour blobs on the top camera. color: {', '.join(COLOUR_NAMES)}, 'any' (saturated), 'dark_on_light' (darker than the table) or 'raised' (depth >= 6 mm above the table). "
                   f"Each blob: centroid world_xyz PLUS world-frame geometry - {GEOMETRY_KEYS}. Heights from depth unless object_height_m is given.")
    returns = "outputs.detections (+ label, pixel, bbox, area_px, plane_z, confidence, fill_ratio), outputs.mode; failure when nothing matches"
    observations_used = ["rgb:top", "depth:top"]
    effect_verification = "n/a (perception)"
    known_limitations = ["grey/textured objects: use segment_objects or 'raised'", "'dark_on_light'/'raised' include robot parts"]
    timeout_s = 3.0

    class Params(Params):
        color: str = Field(default="any", description="colour name, 'any', 'dark_on_light' or 'raised'")
        object_height_m: float | None = Field(default=None, ge=0.0, le=0.3, description="assumed height; omit -> per-blob depth")
        camera: Literal["top"] = "top"
        max_results: int = Field(default=12, ge=1, le=30)
        return_geometry: bool = Field(default=True, description="attach geometry (first 8 blobs)")
        near_xy: list[float] | None = Field(default=None, min_length=2, max_length=2, description="only blobs within near_radius_m of this world xy")
        near_radius_m: float = Field(default=0.025, ge=0.005, le=0.3)

    def run(self, ctx, params):
        image, ref = ctx.capture(params.camera, tag="locate")
        depth = ctx.capture_depth(params.camera)
        loc = _v2_locator(ctx, params.camera)
        near_xy, radius = _near(params)
        mode = params.color.lower().strip()
        common = dict(object_height_m=params.object_height_m, depth=depth, geometry=True, near_xy=near_xy, near_radius_m=radius, max_results=params.max_results)
        try:
            if mode == "dark_on_light":
                dets = loc.detect(image, mode, mask=dark_on_light_mask(image), **common)
            elif mode == "raised":
                masks, info = depth_blob_masks(depth, loc.K, loc.cam_to_world, table_z=ctx.table_z)
                combined = np.zeros(depth.shape[:2], dtype=np.uint8)
                for m in masks:
                    combined[m] = 255
                dets = loc.detect(image, mode, mask=combined, **common) if masks else []
            else:
                dets = loc.detect(image, mode, **common)
        except KeyError as exc:
            raise SkillError(f"unknown colour {params.color!r}: {exc}; known {COLOUR_NAMES} + {list(SPECIAL_MODES)}", error_type="invalid_params") from exc
        img = ref.model_dump() if ref else None
        if not dets:
            where = f" within {radius * 100:.1f} cm of near_xy" if near_xy else ""
            return self.result(SkillStatus.failure, f"no {params.color} blobs found{where}", verified=True, outputs={"detections": [], "mode": mode, "image": img})
        src = "depth" if params.object_height_m is None else "given"
        outputs = _fit_payload({"detections": _jsonable(dets, geometry=params.return_geometry), "mode": mode, "image": img}, "detections")
        return self.result(SkillStatus.success, f"{len(dets)} {params.color} blob(s) with geometry (heights: {src})", verified=True, outputs=outputs)


class PixelToWorldBatch(Skill):
    name = "pixel_to_world"
    is_query = True
    version = "2.0.0"
    kind = "primitive"
    description = ("Top-camera pixels -> world positions: a single (u, v) or a BATCH points=[[u, v], ...] (<= 16) in ONE call (one query unit). "
                   "Without height_m the depth image gives the surface point (source 'depth'); with height_m the ray meets z = table + height_m (source 'plane_given_height'); no valid depth -> table plane, flagged (source 'plane_table_fallback_no_depth'). Prefer segment_objects / locate_objects geometry (corners, faces) over probing edges pixel by pixel.")
    returns = "single: outputs.world_xyz, source, depth_m, surface_height_above_table_m; batch: outputs.points = [{u, v, world_xyz, source, depth_m, surface_height_above_table_m}], outputs.n, outputs.sources"
    observations_used = ["rgb:top", "depth:top"]
    effect_verification = "n/a"
    timeout_s = 1.0

    class Params(Params):
        u: float | None = Field(default=None, ge=0, description="pixel column (single)")
        v: float | None = Field(default=None, ge=0, description="pixel row (single)")
        points: list[list[float]] | None = Field(default=None, min_length=1, max_length=16, description="batch: [[u, v], ...] up to 16")
        height_m: float | None = Field(default=None, ge=0.0, le=0.4, description="plane height above the table; omit -> depth")
        use_depth: bool = Field(default=False, description="compatibility: same as omitting height_m")
        camera: Literal["top"] = "top"
        width: int = Field(default=640)
        height: int = Field(default=480)

    def _one(self, u: float, v: float, K, c2w, depth, table_z: float, height_m: float | None, width: int, height: int) -> dict[str, Any]:
        ui = float(np.clip(u, 0, width - 1))
        vi = float(np.clip(v, 0, height - 1))
        if height_m is not None:
            xyz = pixel_to_world(ui, vi, K, c2w, table_z + height_m)
            if xyz is None:
                return {"u": u, "v": v, "world_xyz": None, "source": "plane_miss"}
            return {"u": u, "v": v, "world_xyz": np.round(xyz, 4).tolist(), "source": "plane_given_height"}
        d = depth_at(depth, ui, vi, radius=2)
        if d is not None:
            xyz = deproject([round(ui) + 0.0], [round(vi) + 0.0], [d], K, c2w)[0]
            return {"u": u, "v": v, "world_xyz": np.round(xyz, 4).tolist(), "source": "depth", "depth_m": round(d, 4), "surface_height_above_table_m": round(float(xyz[2] - table_z), 4)}
        xyz = pixel_to_world(ui, vi, K, c2w, table_z)
        return {"u": u, "v": v, "world_xyz": None if xyz is None else np.round(xyz, 4).tolist(), "source": "plane_table_fallback_no_depth"}

    def run(self, ctx, params):
        if params.points is None and (params.u is None or params.v is None):
            raise SkillError("give u and v, or points=[[u, v], ...]", error_type="invalid_params")
        K = ctx.backend.camera_intrinsics(params.camera, width=params.width, height=params.height)
        c2w = ctx.backend.camera_to_world(params.camera)
        height_m = None if params.use_depth else params.height_m
        depth = ctx.capture_depth(params.camera, width=params.width, height=params.height) if height_m is None else None
        if params.points is not None:
            pts = []
            for p in params.points:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    raise SkillError(f"points entries must be [u, v]; got {p!r}", error_type="invalid_params")
                pts.append(self._one(float(p[0]), float(p[1]), K, c2w, depth, ctx.table_z, height_m, params.width, params.height))
            counts: dict[str, int] = {}
            for r in pts:
                counts[r["source"]] = counts.get(r["source"], 0) + 1
            ok = sum(1 for r in pts if r["world_xyz"] is not None)
            status = SkillStatus.success if ok == len(pts) else (SkillStatus.failure if ok == 0 else SkillStatus.unknown)
            return self.result(status, f"{ok}/{len(pts)} pixels converted ({', '.join(f'{k}: {v}' for k, v in counts.items())})", verified=True, outputs={"points": pts, "n": len(pts), "sources": counts})
        one = self._one(float(params.u), float(params.v), K, c2w, depth, ctx.table_z, height_m, params.width, params.height)
        if one["world_xyz"] is None:
            return self.result(SkillStatus.failure, "ray does not hit the plane" if one["source"] == "plane_miss" else "no valid depth and no plane hit", verified=True, outputs={"source": one["source"]})
        outputs = {k: v for k, v in one.items() if k not in ("u", "v")}
        return self.result(SkillStatus.success, f"ok ({one['source']})", verified=True, outputs=outputs)


def CANDIDATE_SMOKE(ctx) -> dict:
    """Kinematics-free check in the empty smoke scene: batch depth conversion of table pixels lands on the table."""

    r = ctx.call("pixel_to_world", points=[[320, 240], [300, 260], [340, 220], [280, 300]])
    pts = r.outputs.get("points", [])
    zs = [p["world_xyz"][2] for p in pts if p.get("world_xyz")]
    on_table = sum(1 for z in zs if abs(z - ctx.table_z) < 0.02)
    # the smoke scene is not guaranteed empty: most probed pixels must land on the table, all must come from depth
    ok = r.status == SkillStatus.success and len(zs) == 4 and on_table >= 3 and all(p.get("source") == "depth" for p in pts)
    r2 = ctx.call("locate_objects", color="raised")
    ok = ok and r2.status in (SkillStatus.success, SkillStatus.failure)
    return {"ok": bool(ok), "notes": f"batch pixel_to_world sources={r.outputs.get('sources')} z={np.round(zs, 4).tolist()} on_table={on_table}/4; locate_objects(raised) -> {r2.status.value}: {r2.message[:80]}"}
