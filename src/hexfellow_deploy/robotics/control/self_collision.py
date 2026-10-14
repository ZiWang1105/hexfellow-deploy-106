"""Approval-gated, broad-phase-culled self-collision barriers for Pink."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco

import hppfcl
import numpy as np
import pinocchio as pin
import yaml
from pink import Configuration
from pink.barriers import Barrier

try:
    from _hexfellow_collision_kernel import KERNEL_ABI_VERSION as _NATIVE_KERNEL_ABI
    from _hexfellow_collision_kernel import (
        NativeCollisionKernelV2 as _NativeCollisionKernelV2,
    )
    from _hexfellow_collision_kernel import (
        NativeFclUnionKernel as _NativeFclUnionKernel,
    )
    from _hexfellow_collision_kernel import (
        segment_union_distances as _native_segment_unions,
    )
    from _hexfellow_collision_kernel import witness_jacobians as _native_witness_jacobians
except ImportError:  # Pure-Python oracle remains the portable fail-safe path.
    _NATIVE_KERNEL_ABI = None
    _NativeCollisionKernelV2 = None
    _NativeFclUnionKernel = None
    _native_segment_unions = None
    _native_witness_jacobians = None

from ..paths import CONFIG_DIR, PROJECT_ROOT

DEFAULT_SELF_COLLISION_CONFIG = CONFIG_DIR / "control" / "self_collision_proxies.v2.yaml"

_EXPECTED_TIERS = {
    "high": {
        "maximum_overhang_m": 0.010,
        "maximum_undercoverage_area_fraction": 0.005,
        "maximum_undercoverage_depth_m": 0.003,
    },
    "medium": {
        "maximum_overhang_m": 0.010,
        "maximum_undercoverage_area_fraction": 0.010,
        "maximum_undercoverage_depth_m": 0.005,
    },
    "coarse": {
        "maximum_overhang_m": 0.010,
        "maximum_undercoverage_area_fraction": 0.030,
        "maximum_undercoverage_depth_m": 0.010,
    },
}
_REVIEW_STAGES = ("arms", "grippers", "other")


@dataclass(frozen=True)
class CollisionPairDiagnostic:
    """Minimum proxy distance for one configured logical-link pair."""

    first: str
    second: str
    distance_m: float
    minimum_distance_m: float
    exact: bool = True
    active: bool = True

    @property
    def residual_m(self) -> float:
        return self.distance_m - self.minimum_distance_m


@dataclass(frozen=True)
class OraclePairDiagnostic:
    """Detailed-MuJoCo distance for one logical-link pair."""

    first: str
    second: str
    distance_m: float
    minimum_distance_m: float
    exact: bool = True

    @property
    def residual_m(self) -> float:
        return self.distance_m - self.minimum_distance_m


@dataclass(frozen=True)
class BroadPhaseDiagnostics:
    logical_pair_count: int
    sphere_candidate_count: int
    aabb_candidate_count: int
    active_pair_count: int
    exact_distance_calls: int
    primitive_pair_count: int
    configured_pair_count: int = 0
    certified_inactive_pair_count: int = 0
    broadphase_ms: float = 0.0
    analytic_distance_ms: float = 0.0
    native_distance_ms: float = 0.0
    barrier_assembly_ms: float = 0.0


@dataclass(frozen=True)
class CollisionEnvelope:
    """Profile-bound dynamic margins layered over reviewed proxy geometry."""

    path: Path
    envelope_id: str
    profile_id: str
    profile_sha256: str
    geometry_payload_sha256: str
    reference_dt_s: float
    maximum_relative_step_scale: float
    worst_case_stopping_distance_m: float
    stopping_distance_m: float
    guard_distance_m: float
    requires_dynamic_travel: bool = False
    activation_only: bool = False
    reviewed_proxy_clearance_m: float | None = None


def pair_static_clearance(pair: Mapping[str, object]) -> float:
    """A total static budget, never a cap on geometry uncertainty.

    The optional floor is hash-bound as part of each pair. Legacy documents
    retain exactly their original sum when the floor is absent.
    """
    terms = [float(pair[key]) for key in ("first_undercoverage_m", "second_undercoverage_m",
                                          "calibration_error_m", "state_estimation_error_m")]
    floor = float(pair.get("static_clearance_floor_m", 0.0))
    if not np.all(np.isfinite([*terms, floor])) or min(*terms, floor) < 0:
        raise ValueError("static clearance components must be finite and nonnegative")
    return max(sum(terms), floor)


def load_collision_envelope(
    path: str | Path,
    *,
    profile_id: str,
    profile_sha256: str,
    geometry_payload_sha256: str,
) -> CollisionEnvelope:
    """Reject stale envelopes before any collision thresholds are applied."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"self-collision envelope does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") not in {1, 2, 3}:
        raise ValueError("unsupported self-collision envelope schema")
    stopping = raw.get("stopping_distance")
    if not isinstance(stopping, Mapping):
        raise ValueError("self-collision envelope stopping-distance block is missing")
    if raw.get("profile_id") != profile_id or raw.get("profile_sha256") != profile_sha256:
        raise ValueError("self-collision envelope is stale for the selected motion profile")
    if raw.get("geometry_payload_sha256") != geometry_payload_sha256:
        raise ValueError("self-collision envelope is stale for the reviewed proxy geometry")
    reference_dt = float(raw.get("reference_dt_s", np.nan))
    relative_scale = float(raw.get("maximum_relative_step_scale", np.nan))
    worst_case = float(stopping.get("worst_case_m", np.nan))
    applied = float(stopping.get("applied_m", np.nan))
    guard = float(raw.get("guard_distance_m", np.nan))
    values = (reference_dt, relative_scale, worst_case, applied, guard)
    dynamic = raw.get("schema_version") in {2, 3}
    activation_only = raw.get("schema_version") == 3
    if any(not np.isfinite(value) for value in values) or min(reference_dt, relative_scale, guard) <= 0:
        raise ValueError("self-collision envelope distances, scale, and period must be positive")
    if (dynamic and (worst_case != 0 or applied != 0)) or (
        not dynamic and min(worst_case, applied) <= 0
    ):
        raise ValueError("dynamic envelopes must use runtime travel, legacy envelopes require positive budgets")
    if relative_scale < 1.0 or applied + 1e-12 < worst_case:
        raise ValueError("self-collision envelope may not shrink its reference dynamic margins")
    method = str(stopping.get("method", ""))
    if dynamic:
        expected_method = "activation_only_jerk_v1" if activation_only else "measured_reference_jerk_v1"
        if method != expected_method:
            raise ValueError("dynamic envelope requires measured-reference jerk bounds")
    elif method == "ceil_1mm_max_8mm_1p25x_plus_1mm":
        expected = float(np.ceil(max(0.008, 1.25 * worst_case + 0.001) * 1000.0) / 1000.0)
        if not np.isclose(applied, expected, atol=1e-12):
            raise ValueError("self-collision stopping distance does not match its declared formula")
    elif method != "inherited_approved_v2":
        raise ValueError(f"unsupported self-collision stopping-distance method {method!r}")
    proxy_clearance = raw.get("reviewed_proxy_clearance_m")
    if proxy_clearance is not None:
        proxy_clearance = float(proxy_clearance)
        if (not activation_only or raw.get("clearance_basis") != "reviewed_proxy_v1"
                or not np.isfinite(proxy_clearance) or proxy_clearance <= 0):
            raise ValueError("reviewed proxy clearance requires a positive activation-only explicit policy")
    elif raw.get("clearance_basis") is not None:
        raise ValueError("clearance basis requires reviewed_proxy_clearance_m")
    return CollisionEnvelope(
        path=path,
        envelope_id=str(raw.get("envelope_id", "")),
        profile_id=profile_id,
        profile_sha256=profile_sha256,
        geometry_payload_sha256=geometry_payload_sha256,
        reference_dt_s=reference_dt,
        maximum_relative_step_scale=relative_scale,
        worst_case_stopping_distance_m=worst_case,
        stopping_distance_m=applied,
        guard_distance_m=guard,
        requires_dynamic_travel=dynamic,
        activation_only=activation_only,
        reviewed_proxy_clearance_m=proxy_clearance,
    )


@dataclass(frozen=True)
class _PrimitiveRecord:
    geometry_index: int
    parent_joint: int
    kind: str
    radius_m: float | None = None
    radii_m: tuple[float, float, float] | None = None
    half_length_m: float | None = None


@dataclass(frozen=True)
class _LogicalPair:
    first: str
    second: str
    minimum_distance_m: float
    activation_distance_m: float
    maximum_relative_step_m: float
    first_indices: tuple[int, ...]
    second_indices: tuple[int, ...]


@dataclass(frozen=True)
class _ActiveDistance:
    pair: _LogicalPair
    first_geometry_index: int
    second_geometry_index: int
    distance_m: float
    first_point: np.ndarray
    second_point: np.ndarray
    fallback_normal: np.ndarray


@dataclass(frozen=True)
class PairExclusionCertificate:
    """Hash-bound proof that selected logical pairs stay globally separated."""

    path: Path
    sha256: str
    geometry_payload_sha256: str
    stopping_distance_m: float
    certified_pairs: frozenset[tuple[str, str]]
    joint_indices: np.ndarray
    lower_position: np.ndarray
    upper_position: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pair_exclusion_certificate(
    path: str | Path,
    *,
    expected_sha256: str,
    geometry_document: Mapping[str, object],
    model: pin.Model,
) -> PairExclusionCertificate:
    """Validate a deterministic full-domain pair-separation certificate."""

    path = Path(path).resolve()
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("self-collision pair certificate is missing or has a stale SHA-256")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("unsupported self-collision pair certificate schema")
    geometry_hash = str(geometry_document.get("geometry_payload_sha256", ""))
    robot_description = geometry_document.get("robot_description")
    if (
        raw.get("robot") != "maver_x4"
        or raw.get("algorithm") != "global_lipschitz_branch_and_bound_v1"
        or raw.get("geometry_payload_sha256") != geometry_hash
        or not isinstance(robot_description, Mapping)
        or raw.get("robot_description_sha256") != robot_description.get("sha256")
        or not np.isclose(float(raw.get("numerical_margin_m", np.nan)), 0.0001, atol=1e-12)
    ):
        raise ValueError("self-collision pair certificate provenance is stale")

    bounds = raw.get("joint_bounds")
    if not isinstance(bounds, list) or not bounds:
        raise ValueError("self-collision pair certificate has no joint domain")
    joint_indices: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    seen_joints: set[str] = set()
    covered_model_joints = 0
    for bound in bounds:
        if not isinstance(bound, Mapping):
            raise ValueError("invalid certified joint bound")
        name = str(bound.get("name", ""))
        joint_id = model.getJointId(name)
        if not name or name in seen_joints:
            raise ValueError(f"unknown or duplicate certified joint {name!r}")
        lo = float(bound.get("lower", np.nan))
        hi = float(bound.get("upper", np.nan))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"invalid certified domain for joint {name!r}")
        seen_joints.add(name)
        # The certificate is generated from the complete URDF. Pink locks the
        # grippers into a reduced model, so a valid certificate can contain a
        # conservative superset of the runtime model's joints.
        if joint_id == model.njoints:
            continue
        if joint_id == 0:
            raise ValueError("the universe joint cannot appear in a pair certificate")
        joint = model.joints[joint_id]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError("pair certificates currently require scalar robot joints")
        if (
            model.lowerPositionLimit[joint.idx_q] < lo - 1e-9
            or model.upperPositionLimit[joint.idx_q] > hi + 1e-9
        ):
            raise ValueError("controller model exceeds the certified collision domain")
        covered_model_joints += 1
        joint_indices.append(joint.idx_q)
        lower.append(lo)
        upper.append(hi)
    root_joint_id = model.getJointId("root_joint")
    has_planar_root = (
        root_joint_id < model.njoints
        and model.parents[root_joint_id] == 0
        and model.joints[root_joint_id].shortname() == "JointModelComposite"
        and model.joints[root_joint_id].nv == 3
    )
    required_model_joints = model.njoints - 1 - int(has_planar_root)
    if covered_model_joints != required_model_joints:
        raise ValueError("pair certificate must bound every non-universe model joint")

    configured = {
        tuple(sorted((str(pair["first"]), str(pair["second"])))): pair
        for pair in geometry_document["pairs"]
    }
    certified: set[tuple[str, str]] = set()
    entries = raw.get("certified_pairs")
    if not isinstance(entries, list):
        raise ValueError("self-collision pair certificate has no proof entries")
    guard = float(geometry_document["broadphase"]["guard_distance_m"])
    certified_stopping_distance = float(raw.get("stopping_distance_m", np.nan))
    if not np.isfinite(certified_stopping_distance) or certified_stopping_distance < 0.0:
        raise ValueError("pair certificate has an invalid stopping-distance domain")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid pair proof entry")
        key = tuple(sorted((str(entry.get("first", "")), str(entry.get("second", "")))))
        pair = configured.get(key)
        if pair is None or key in certified:
            raise ValueError("pair certificate references an unknown or duplicate pair")
        required = (
            pair_static_clearance(pair)
            + certified_stopping_distance
            + guard
            + 0.0001
        )
        if (
            entry.get("status") != "certified_separated"
            or not np.isclose(float(entry.get("required_clearance_m", np.nan)), required, atol=1e-12)
            or float(entry.get("proven_lower_bound_m", -np.inf)) <= required
            or int(entry.get("evaluated_cells", 0)) < 1
        ):
            raise ValueError("pair certificate contains an invalid separation proof")
        certified.add(key)
    return PairExclusionCertificate(
        path=path,
        sha256=expected_sha256,
        geometry_payload_sha256=geometry_hash,
        stopping_distance_m=certified_stopping_distance,
        certified_pairs=frozenset(certified),
        joint_indices=np.asarray(joint_indices, dtype=int),
        lower_position=np.asarray(lower),
        upper_position=np.asarray(upper),
    )


def _canonical_geometry_payload(document: Mapping[str, object]) -> bytes:
    payload = {
        key: document[key]
        for key in (
            "schema_version",
            "robot",
            "robot_description",
            "scope",
            "precision_tiers",
            "surface_sampling_resolution_m",
            "broadphase",
            "links",
            "pairs",
            "excluded_pairs",
        )
        if key in document
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def geometry_payload_sha256(document: Mapping[str, object]) -> str:
    """Hash all geometry, pair, threshold, and broad-phase safety inputs."""

    return hashlib.sha256(_canonical_geometry_payload(document)).hexdigest()


def stage_geometry_sha256(document: Mapping[str, object], stage: str) -> str:
    """Hash the reviewed geometry for one human-review stage."""

    if stage not in _REVIEW_STAGES:
        raise ValueError(f"unknown self-collision review stage {stage!r}")
    links = document.get("links")
    if not isinstance(links, list):
        raise ValueError("self-collision links are missing")
    payload = {
        "schema_version": document.get("schema_version"),
        "robot": document.get("robot"),
        "precision_tiers": document.get("precision_tiers"),
        "surface_sampling_resolution_m": document.get("surface_sampling_resolution_m"),
        "stage": stage,
        "links": [link for link in links if isinstance(link, Mapping) and link.get("stage") == stage],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _review_accepts_undercoverage(
    document: Mapping[str, object], stage: str
) -> bool:
    """Whether a hash-bound human review accepted the documented small gaps."""

    reviews = document.get("stage_reviews")
    if not isinstance(reviews, Mapping):
        return False
    review = reviews.get(stage)
    return (
        isinstance(review, Mapping)
        and review.get("status") == "approved"
        and review.get("stage_geometry_sha256")
        == stage_geometry_sha256(document, stage)
        and review.get("accepted_omission_policy")
        == "joint_seams_corners_and_shell_edges"
    )


def _project_file(entry: Mapping[str, object], *, label: str) -> Path:
    relative = str(entry.get("path", ""))
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} leaves the project root: {path}") from error
    if not path.is_file() or _sha256(path) != entry.get("sha256"):
        raise ValueError(f"{label} hash mismatch: {relative!r}")
    return path


def _validate_precision_tiers(document: Mapping[str, object]) -> Mapping[str, object]:
    tiers = document.get("precision_tiers")
    if not isinstance(tiers, Mapping):
        raise ValueError("self-collision precision tiers are missing")
    for tier, expected_values in _EXPECTED_TIERS.items():
        values = tiers.get(tier)
        if not isinstance(values, Mapping):
            raise ValueError(f"self-collision precision tier {tier!r} is missing")
        for key, expected in expected_values.items():
            if not np.isclose(float(values.get(key, np.nan)), expected, atol=1e-12):
                raise ValueError(f"precision tier {tier!r} field {key!r} must be {expected:g}")
    return tiers


def _validate_primitive(primitive: Mapping[str, object]) -> None:
    kind = primitive.get("type")
    if kind == "sphere":
        center = np.asarray(primitive.get("center_m"), dtype=float)
        radius = float(primitive.get("radius_m", np.nan))
        if center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("collision sphere requires a finite center and positive radius")
        return
    if kind == "capsule":
        start = np.asarray(primitive.get("start_m"), dtype=float)
        end = np.asarray(primitive.get("end_m"), dtype=float)
        radius = float(primitive.get("radius_m", np.nan))
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or not np.all(np.isfinite((start, end)))
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("collision capsule requires finite endpoints and a positive radius")
        return
    if kind == "ellipsoid":
        center = np.asarray(primitive.get("center_m"), dtype=float)
        radii = np.asarray(primitive.get("radii_m"), dtype=float)
        quaternion = np.asarray(primitive.get("quaternion_wxyz"), dtype=float)
        if (
            center.shape != (3,)
            or radii.shape != (3,)
            or quaternion.shape != (4,)
            or not np.all(np.isfinite((center, radii)))
            or not np.all(np.isfinite(quaternion))
            or np.any(radii <= 0.0)
            or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6)
        ):
            raise ValueError("collision ellipsoid requires a finite center, positive radii, and unit quaternion")
        return
    raise ValueError(f"unsupported collision primitive type {kind!r}")


def load_self_collision_document(
    path: str | Path = DEFAULT_SELF_COLLISION_CONFIG,
    *,
    require_approved: bool,
) -> tuple[Path, dict[str, object]]:
    """Load a v2 proxy document and enforce provenance and approval gates."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"self-collision proxy configuration does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("self-collision proxy configuration must be a mapping")
    document = dict(raw)
    if document.get("schema_version") != 2 or document.get("robot") != "maver_x4":
        raise ValueError("unsupported self-collision proxy schema or robot")
    if document.get("status") not in {"candidate", "approved"}:
        raise ValueError("self-collision proxy status must be candidate or approved")
    if document.get("scope") not in _REVIEW_STAGES + ("full",):
        raise ValueError("self-collision scope must be arms, grippers, other, or full")
    resolution = float(document.get("surface_sampling_resolution_m", np.nan))
    if not np.isclose(resolution, 0.0005, atol=1e-12):
        raise ValueError("self-collision review resolution must be 0.5 mm")
    tiers = _validate_precision_tiers(document)
    robot_description = document.get("robot_description")
    if not isinstance(robot_description, Mapping):
        raise ValueError("self-collision robot-description provenance is missing")
    _project_file(robot_description, label="robot description")

    broadphase = document.get("broadphase")
    if not isinstance(broadphase, Mapping):
        raise ValueError("self-collision broad-phase parameters are missing")
    guard = float(broadphase.get("guard_distance_m", np.nan))
    hysteresis = float(broadphase.get("hysteresis_m", np.nan))
    reference_dt = float(broadphase.get("reference_dt_s", np.nan))
    if (
        not np.isclose(guard, 0.002, atol=1e-12)
        or not np.isclose(hysteresis, 0.005, atol=1e-12)
        or not np.isclose(reference_dt, 0.020, atol=1e-12)
    ):
        raise ValueError(
            "broad-phase guard, hysteresis, and reference timestep must be 2 mm, 5 mm, and 20 ms"
        )

    links = document.get("links")
    pairs = document.get("pairs")
    if not isinstance(links, list) or not links or not isinstance(pairs, list):
        raise ValueError("self-collision proxy configuration requires links and a pair list")
    names: set[str] = set()
    stages_present: set[str] = set()
    for link in links:
        if not isinstance(link, Mapping):
            raise ValueError("each self-collision link must be a mapping")
        name = str(link.get("name", ""))
        frame = str(link.get("frame", ""))
        tier = str(link.get("precision_tier", ""))
        stage = str(link.get("stage", ""))
        sources = link.get("sources")
        primitives = link.get("primitives")
        metrics = link.get("metrics")
        mujoco_bodies = link.get("mujoco_bodies")
        if (
            not name
            or not frame
            or name in names
            or tier not in _EXPECTED_TIERS
            or stage not in _REVIEW_STAGES
            or not isinstance(sources, list)
            or not sources
            or not isinstance(primitives, list)
            or not primitives
            or not isinstance(metrics, Mapping)
            or not isinstance(mujoco_bodies, list)
            or not mujoco_bodies
        ):
            raise ValueError(f"invalid or duplicate self-collision link {name!r}")
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError(f"self-collision link {name!r} has an invalid mesh source")
            _project_file(source, label=f"self-collision source for {name!r}")
        for primitive in primitives:
            if not isinstance(primitive, Mapping):
                raise ValueError(f"self-collision link {name!r} has an invalid primitive")
            _validate_primitive(primitive)
        enforce_geometry_gates = require_approved or document.get("status") == "approved"
        thresholds = tiers[tier]
        accepted_undercoverage = _review_accepts_undercoverage(document, stage)
        for metric, threshold_key in (
            ("maximum_overhang_m", "maximum_overhang_m"),
            ("undercoverage_area_fraction", "maximum_undercoverage_area_fraction"),
            ("maximum_undercoverage_depth_m", "maximum_undercoverage_depth_m"),
        ):
            value = float(metrics.get(metric, np.inf))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"self-collision link {name!r} has invalid metric {metric!r}")
            exception_applies = (
                metric != "maximum_overhang_m" and accepted_undercoverage
            )
            if (
                enforce_geometry_gates
                and not exception_applies
                and value > float(thresholds[threshold_key]) + 1e-12
            ):
                raise ValueError(f"self-collision link {name!r} exceeds {tier!r} {metric!r}")
        names.add(name)
        stages_present.add(stage)

    pair_keys: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError(f"invalid self-collision pair: {pair!r}")
        first = str(pair.get("first", ""))
        second = str(pair.get("second", ""))
        key = tuple(sorted((first, second)))
        if first not in names or second not in names or first == second or key in pair_keys:
            raise ValueError(f"invalid or duplicate self-collision pair: {pair!r}")
        minimum = pair_static_clearance(pair) + float(pair.get("stopping_distance_m", np.nan))
        relative_step = float(pair.get("maximum_relative_step_m", np.nan))
        activation = float(pair.get("activation_distance_m", np.nan))
        if not np.isfinite(minimum) or minimum < 0.0 or not np.isfinite(relative_step) or relative_step < 0.0:
            raise ValueError(f"invalid safety-distance components for pair {pair!r}")
        if not np.isclose(float(pair.get("minimum_distance_m", np.nan)), minimum, atol=1e-9):
            raise ValueError("pair minimum distance must equal geometry, calibration, state, and stopping terms")
        if not np.isclose(activation, minimum + relative_step + guard, atol=1e-9):
            raise ValueError("pair activation distance must include margin, relative step, and broad-phase guard")
        pair_keys.add(key)

    payload_hash = geometry_payload_sha256(document)
    if document.get("geometry_payload_sha256") != payload_hash:
        raise ValueError("self-collision geometry payload hash is stale")
    if require_approved:
        if document.get("scope") != "full":
            raise ValueError("production self-collision requires a full-body proxy document")
        if document.get("approval_blockers"):
            raise ValueError("self-collision candidate still has approval blockers")
        reviews = document.get("stage_reviews")
        if not isinstance(reviews, Mapping):
            raise ValueError("self-collision stage reviews are missing")
        for stage in _REVIEW_STAGES:
            review = reviews.get(stage)
            if stage not in stages_present or not isinstance(review, Mapping):
                raise ValueError(f"self-collision stage {stage!r} is missing")
            reviewed_hash = review.get("stage_geometry_sha256")
            if review.get("status") != "approved" or reviewed_hash != stage_geometry_sha256(
                document, stage
            ):
                raise ValueError(f"self-collision stage {stage!r} is not approved for this geometry")
            if not review.get("reviewer") or not review.get("reviewed_at"):
                raise ValueError(f"self-collision stage {stage!r} approval is incomplete")
        approval = document.get("approval")
        if document.get("status") != "approved" or not isinstance(approval, Mapping):
            raise ValueError("self-collision proxies are not explicitly approved")
        if approval.get("geometry_payload_sha256") != payload_hash:
            raise ValueError("self-collision final approval does not match the geometry payload")
        if not approval.get("reviewer") or not approval.get("approved_at"):
            raise ValueError("self-collision approval is missing reviewer or timestamp")
    return path, document


def _quaternion_rotation(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _primitive_placement(
    primitive: Mapping[str, object],
) -> tuple[hppfcl.CollisionGeometry, pin.SE3, str, float | None, tuple[float, float, float] | None, float | None]:
    kind = str(primitive["type"])
    if kind == "sphere":
        radius = float(primitive["radius_m"])
        center = np.asarray(primitive["center_m"], dtype=float)
        return hppfcl.Sphere(radius), pin.SE3(np.eye(3), center), kind, radius, None, None
    if kind == "ellipsoid":
        radii_array = np.asarray(primitive["radii_m"], dtype=float)
        radii = tuple(float(value) for value in radii_array)
        center = np.asarray(primitive["center_m"], dtype=float)
        rotation = _quaternion_rotation(primitive["quaternion_wxyz"])
        return hppfcl.Ellipsoid(radii_array), pin.SE3(rotation, center), kind, None, radii, None
    radius = float(primitive["radius_m"])
    start = np.asarray(primitive["start_m"], dtype=float)
    end = np.asarray(primitive["end_m"], dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:
        return hppfcl.Sphere(radius), pin.SE3(np.eye(3), 0.5 * (start + end)), "sphere", radius, None, None
    z_axis = axis / length
    seed = np.array((1.0, 0.0, 0.0)) if abs(z_axis[0]) < 0.8 else np.array((0.0, 1.0, 0.0))
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    half_length = 0.5 * length
    return (
        # HPP-FCL names the second constructor argument ``lz`` and stores
        # ``halfLength = lz / 2``. Pass the full segment length represented by
        # the reviewed start/end points.
        hppfcl.Capsule(radius, length),
        pin.SE3(rotation, 0.5 * (start + end)),
        kind,
        radius,
        None,
        half_length,
    )


class ActivePairCollisionBarrier(Barrier):
    """Barrier rows for the logical pairs activated on the current cycle."""

    def __init__(
        self,
        distances: tuple[_ActiveDistance, ...],
        geometry_model: pin.GeometryModel,
        geometry_parent_joints: np.ndarray | None = None,
    ) -> None:
        self._distance_objects: tuple[_ActiveDistance, ...] | None = distances
        self._logical_pairs = tuple(distance.pair for distance in distances)
        self.geometry_model = geometry_model
        count = len(distances)
        self._first_geometry_indices = np.empty(count, dtype=int)
        self._second_geometry_indices = np.empty(count, dtype=int)
        self._first_points = np.empty((count, 3), dtype=float)
        self._second_points = np.empty((count, 3), dtype=float)
        self._fallback_normals = np.empty((count, 3), dtype=float)
        self._residuals = np.empty(count, dtype=float)
        for index, distance in enumerate(distances):
            self._first_geometry_indices[index] = distance.first_geometry_index
            self._second_geometry_indices[index] = distance.second_geometry_index
            self._first_points[index] = distance.first_point
            self._second_points[index] = distance.second_point
            self._fallback_normals[index] = distance.fallback_normal
            self._residuals[index] = (
                distance.distance_m - distance.pair.minimum_distance_m
            )
        if geometry_parent_joints is None:
            geometry_parent_joints = np.fromiter(
                (
                    geometry.parentJoint
                    for geometry in geometry_model.geometryObjects
                ),
                dtype=int,
                count=len(geometry_model.geometryObjects),
            )
        self._first_joint_ids = geometry_parent_joints[
            self._first_geometry_indices
        ]
        self._second_joint_ids = geometry_parent_joints[
            self._second_geometry_indices
        ]
        super().__init__(dim=count, gain=1.0, safe_displacement_gain=0.0)

    @classmethod
    def from_compact_arrays(
        cls,
        *,
        pair_indices: np.ndarray,
        logical_pairs: tuple[_LogicalPair, ...],
        first_geometry_indices: np.ndarray,
        second_geometry_indices: np.ndarray,
        distances_m: np.ndarray,
        first_points: np.ndarray,
        second_points: np.ndarray,
        fallback_normals: np.ndarray,
        geometry_model: pin.GeometryModel,
        geometry_parent_joints: np.ndarray,
    ) -> ActivePairCollisionBarrier:
        """Build a current-cycle barrier without per-pair Python objects."""

        pair_indices = np.asarray(pair_indices, dtype=int)
        count = len(pair_indices)
        arrays = (
            np.asarray(first_geometry_indices, dtype=int),
            np.asarray(second_geometry_indices, dtype=int),
            np.asarray(distances_m, dtype=float),
            np.asarray(first_points, dtype=float),
            np.asarray(second_points, dtype=float),
            np.asarray(fallback_normals, dtype=float),
        )
        if (
            pair_indices.shape != (count,)
            or arrays[0].shape != (count,)
            or arrays[1].shape != (count,)
            or arrays[2].shape != (count,)
            or any(value.shape != (count, 3) for value in arrays[3:])
            or np.any(pair_indices < 0)
            or np.any(pair_indices >= len(logical_pairs))
            or not all(np.all(np.isfinite(value)) for value in arrays[2:])
        ):
            raise ValueError("invalid compact collision barrier arrays")
        self = cls.__new__(cls)
        self._distance_objects = None
        self._logical_pairs = tuple(logical_pairs[int(index)] for index in pair_indices)
        self.geometry_model = geometry_model
        self._first_geometry_indices = arrays[0]
        self._second_geometry_indices = arrays[1]
        self._first_points = arrays[3]
        self._second_points = arrays[4]
        self._fallback_normals = arrays[5]
        minimum = np.fromiter(
            (pair.minimum_distance_m for pair in self._logical_pairs),
            dtype=float,
            count=count,
        )
        self._residuals = arrays[2] - minimum
        self._first_joint_ids = geometry_parent_joints[self._first_geometry_indices]
        self._second_joint_ids = geometry_parent_joints[self._second_geometry_indices]
        Barrier.__init__(self, dim=count, gain=1.0, safe_displacement_gain=0.0)
        return self

    @property
    def distances(self) -> tuple[_ActiveDistance, ...]:
        if self._distance_objects is None:
            self._distance_objects = tuple(
                _ActiveDistance(
                    pair,
                    int(first_geometry),
                    int(second_geometry),
                    float(residual + pair.minimum_distance_m),
                    first_point.copy(),
                    second_point.copy(),
                    fallback.copy(),
                )
                for pair, first_geometry, second_geometry, residual, first_point,
                second_point, fallback in zip(
                    self._logical_pairs,
                    self._first_geometry_indices,
                    self._second_geometry_indices,
                    self._residuals,
                    self._first_points,
                    self._second_points,
                    self._fallback_normals,
                    strict=True,
                )
            )
        return self._distance_objects

    def pair_label(self, row: int, prefix: str) -> str:
        pair = self._logical_pairs[int(row)]
        return f"{prefix}:{pair.first}:{pair.second}"

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        return self._residuals.copy()

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        model = configuration.model
        data = configuration.data
        if _native_witness_jacobians is not None and self.dim:
            joint_ids, inverse = np.unique(
                np.concatenate((self._first_joint_ids, self._second_joint_ids)),
                return_inverse=True,
            )
            joint_jacobians = np.empty((len(joint_ids), 6, model.nv), dtype=float)
            joint_origins = np.empty((len(joint_ids), 3), dtype=float)
            for row, joint_id in enumerate(joint_ids):
                joint_jacobians[row] = pin.getJointJacobian(
                    model,
                    data,
                    int(joint_id),
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                joint_origins[row] = data.oMi[int(joint_id)].translation
            return np.asarray(_native_witness_jacobians(
                inverse[: self.dim],
                inverse[self.dim :],
                self._first_points,
                self._second_points,
                self._fallback_normals,
                joint_jacobians,
                joint_origins,
            ))
        if self.dim == 1:
            first_joint_id = int(self._first_joint_ids[0])
            second_joint_id = int(self._second_joint_ids[0])
            first_jacobian = pin.getJointJacobian(
                model,
                data,
                first_joint_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            second_jacobian = pin.getJointJacobian(
                model,
                data,
                second_joint_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            delta = self._first_points[0] - self._second_points[0]
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                delta = self._fallback_normals[0]
                norm = float(np.linalg.norm(delta))
            normal = delta / norm if norm > 1e-12 else np.zeros(3)
            first_offset = (
                self._first_points[0] - data.oMi[first_joint_id].translation
            )
            second_offset = (
                self._second_points[0] - data.oMi[second_joint_id].translation
            )
            row = (
                normal @ first_jacobian[:3]
                + np.cross(first_offset, normal) @ first_jacobian[3:]
                - normal @ second_jacobian[:3]
                - np.cross(second_offset, normal) @ second_jacobian[3:]
            )
            return row[np.newaxis]
        joint_ids, inverse = np.unique(
            np.concatenate((self._first_joint_ids, self._second_joint_ids)),
            return_inverse=True,
        )
        joint_jacobians = np.stack(
            [
                pin.getJointJacobian(
                    model,
                    data,
                    int(joint_id),
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                for joint_id in joint_ids
            ]
        )
        joint_origins = np.stack(
            [np.asarray(data.oMi[int(joint_id)].translation) for joint_id in joint_ids]
        )
        first_inverse = inverse[: self.dim]
        second_inverse = inverse[self.dim :]
        first_jacobians = joint_jacobians[first_inverse]
        second_jacobians = joint_jacobians[second_inverse]
        first_offsets = self._first_points - joint_origins[first_inverse]
        second_offsets = self._second_points - joint_origins[second_inverse]
        delta = self._first_points - self._second_points
        norms = np.linalg.norm(delta, axis=1)
        degenerate = norms <= 1e-12
        if np.any(degenerate):
            delta[degenerate] = self._fallback_normals[degenerate]
            norms[degenerate] = np.linalg.norm(delta[degenerate], axis=1)
        normals = np.zeros_like(delta)
        np.divide(
            delta,
            norms[:, np.newaxis],
            out=normals,
            where=norms[:, np.newaxis] > 1e-12,
        )
        first_angular = np.cross(first_offsets, normals)
        second_angular = np.cross(second_offsets, normals)
        jacobian = (
            np.einsum("ri,rij->rj", normals, first_jacobians[:, :3])
            + np.einsum("ri,rij->rj", first_angular, first_jacobians[:, 3:])
            - np.einsum("ri,rij->rj", normals, second_jacobians[:, :3])
            - np.einsum("ri,rij->rj", second_angular, second_jacobians[:, 3:])
        )
        return np.nan_to_num(jacobian)


def _aabb_distance(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    separation = np.maximum(np.maximum(second[0] - first[1], first[0] - second[1]), 0.0)
    return float(np.linalg.norm(separation))


def _aabb_signed_lower_bound(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> float:
    """Conservative signed distance between enclosing axis-aligned boxes."""

    separation = np.maximum(
        np.maximum(second[0] - first[1], first[0] - second[1]), 0.0
    )
    if np.any(separation > 0.0):
        return float(np.linalg.norm(separation))
    overlap = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    return -float(np.min(overlap))


class SelfCollisionSystem:
    """Conservative broad phase, exact active distances, and Pink barriers."""

    def __init__(
        self,
        model: pin.Model,
        config_path: str | Path = DEFAULT_SELF_COLLISION_CONFIG,
        *,
        require_approved: bool = True,
        envelope_config: str | Path | None = None,
        pair_certificate_config: str | Path | None = None,
        pair_certificate_sha256: str | None = None,
        motion_profile_id: str | None = None,
        motion_profile_sha256: str | None = None,
        runtime: str = "auto_compatible_v1",
        native_kernel_abi: int | None = None,
    ) -> None:
        if runtime not in {"auto_compatible_v1", "native_fused_v2"}:
            raise ValueError(f"unknown self-collision runtime {runtime!r}")
        if runtime == "native_fused_v2" and (
            native_kernel_abi not in {2, 3}
            or _NATIVE_KERNEL_ABI is None
            or _NATIVE_KERNEL_ABI < native_kernel_abi
            or _NativeCollisionKernelV2 is None
            or _native_witness_jacobians is None
        ):
            raise RuntimeError(
                "native_fused_v2 requires _hexfellow_collision_kernel ABI 2+"
            )
        self.runtime = runtime
        self.native_kernel_abi = native_kernel_abi
        self.config_path, document = load_self_collision_document(
            config_path, require_approved=require_approved
        )
        self._configured_pair_count = len(document["pairs"])
        if (pair_certificate_config is None) != (pair_certificate_sha256 is None):
            raise ValueError("pair certificate path and SHA-256 must be configured together")
        self.pair_certificate = (
            None
            if pair_certificate_config is None
            else load_pair_exclusion_certificate(
                pair_certificate_config,
                expected_sha256=str(pair_certificate_sha256),
                geometry_document=document,
                model=model,
            )
        )
        if self.pair_certificate is not None:
            certified = self.pair_certificate.certified_pairs
            document = dict(document)
            document["pairs"] = [
                pair
                for pair in document["pairs"]
                if tuple(sorted((str(pair["first"]), str(pair["second"]))))
                not in certified
            ]
        self._certified_inactive_pair_count = (
            0 if self.pair_certificate is None else len(self.pair_certificate.certified_pairs)
        )
        self.collision_envelope: CollisionEnvelope | None = None
        if envelope_config is not None:
            if not motion_profile_id or not motion_profile_sha256:
                raise ValueError("a collision envelope requires a motion-profile ID and SHA-256")
            self.collision_envelope = load_collision_envelope(
                envelope_config,
                profile_id=motion_profile_id,
                profile_sha256=motion_profile_sha256,
                geometry_payload_sha256=str(document["geometry_payload_sha256"]),
            )
            envelope = self.collision_envelope
            if envelope.reviewed_proxy_clearance_m is not None:
                # This policy changes the boundary definition, not the fitted
                # shapes. An unreviewed geometry bypass must never enable it.
                load_self_collision_document(config_path, require_approved=True)
                if self.pair_certificate is not None:
                    proof = yaml.safe_load(Path(pair_certificate_config).read_text())
                    required = (envelope.reviewed_proxy_clearance_m + envelope.stopping_distance_m
                                + envelope.guard_distance_m + 0.0001)
                    if any(not float(p["proven_lower_bound_m"]) > required for p in proof["certified_pairs"]):
                        raise ValueError("pair certificate does not cover reviewed proxy clearance")
            if (
                self.pair_certificate is not None
                and self.pair_certificate.stopping_distance_m
                + 1e-12
                < envelope.stopping_distance_m
            ):
                raise ValueError("pair certificate does not cover the active stopping distance")
            if not np.isclose(
                envelope.reference_dt_s,
                float(document["broadphase"]["reference_dt_s"]),
                atol=1e-12,
            ):
                raise ValueError("collision envelope period differs from reviewed geometry")
            if not np.isclose(
                envelope.guard_distance_m,
                float(document["broadphase"]["guard_distance_m"]),
                atol=1e-12,
            ):
                raise ValueError("collision envelope guard differs from reviewed geometry")
            active_pairs: list[dict[str, object]] = []
            for original in document["pairs"]:
                pair = dict(original)
                minimum = (pair_static_clearance(pair) if envelope.reviewed_proxy_clearance_m is None
                           else envelope.reviewed_proxy_clearance_m) + envelope.stopping_distance_m
                maximum_step = (
                    float(pair["maximum_relative_step_m"])
                    * envelope.maximum_relative_step_scale
                )
                pair.update(
                    stopping_distance_m=envelope.stopping_distance_m,
                    minimum_distance_m=minimum,
                    maximum_relative_step_m=maximum_step,
                    activation_distance_m=minimum + maximum_step + envelope.guard_distance_m,
                )
                active_pairs.append(pair)
            document = dict(document)
            document["pairs"] = active_pairs
            document["active_collision_envelope"] = {
                "id": envelope.envelope_id,
                "path": str(envelope.path),
                "motion_profile_id": envelope.profile_id,
                "motion_profile_sha256": envelope.profile_sha256,
            }
        elif self.pair_certificate is not None:
            raise ValueError("pair certificates require a profile-bound collision envelope")
        if runtime == "native_fused_v2" and (
            self.collision_envelope is None or not self.collision_envelope.activation_only
        ):
            raise ValueError("native_fused_v2 requires an activation-only collision envelope")
        self.document = document
        self._model = model
        self.geometry_model = pin.GeometryModel()
        self._records: list[_PrimitiveRecord] = []
        indices_by_link: dict[str, tuple[int, ...]] = {}
        self._frame_id_by_link: dict[str, int] = {}
        self._local_bounds_by_link: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._local_primitive_bounds_by_link: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._local_segments_by_link: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for link in document["links"]:
            frame_name = str(link["frame"])
            frame_id = model.getFrameId(frame_name)
            if frame_id >= model.nframes:
                raise ValueError(f"Pinocchio model is missing collision frame {frame_name!r}")
            frame = model.frames[frame_id]
            parent_joint = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
            indices: list[int] = []
            local_lower: list[np.ndarray] = []
            local_upper: list[np.ndarray] = []
            local_segment_starts: list[np.ndarray] = []
            local_segment_ends: list[np.ndarray] = []
            local_segment_radii: list[float] = []
            for primitive_index, primitive in enumerate(link["primitives"]):
                geometry, local, kind, radius, radii, half_length = _primitive_placement(primitive)
                name = f"proxy::{link['name']}::{primitive_index}"
                placement = frame.placement * local
                if hasattr(frame, "parentJoint"):
                    geometry_object = pin.GeometryObject(name, parent_joint, frame_id, placement, geometry)
                else:
                    geometry_object = pin.GeometryObject(name, frame_id, parent_joint, geometry, placement)
                geometry_index = self.geometry_model.addGeometryObject(geometry_object)
                indices.append(geometry_index)
                self._records.append(
                    _PrimitiveRecord(geometry_index, parent_joint, kind, radius, radii, half_length)
                )
                local_center = np.asarray(local.translation, dtype=float)
                if kind == "sphere":
                    local_extent = np.full(3, float(radius))
                elif kind == "ellipsoid":
                    local_extent = np.abs(np.asarray(local.rotation)) @ np.asarray(
                        radii, dtype=float
                    )
                else:
                    local_extent = (
                        np.abs(np.asarray(local.rotation)[:, 2]) * float(half_length)
                        + float(radius)
                    )
                local_lower.append(local_center - local_extent)
                local_upper.append(local_center + local_extent)
                if kind == "sphere":
                    local_segment_starts.append(local_center)
                    local_segment_ends.append(local_center)
                    local_segment_radii.append(float(radius))
                elif kind == "capsule":
                    local_offset = (
                        np.asarray(local.rotation)[:, 2] * float(half_length)
                    )
                    local_segment_starts.append(local_center - local_offset)
                    local_segment_ends.append(local_center + local_offset)
                    local_segment_radii.append(float(radius))
            link_name = str(link["name"])
            indices_by_link[link_name] = tuple(indices)
            self._frame_id_by_link[link_name] = frame_id
            self._local_bounds_by_link[link_name] = (
                np.min(np.stack(local_lower), axis=0),
                np.max(np.stack(local_upper), axis=0),
            )
            self._local_primitive_bounds_by_link[link_name] = (
                0.5 * (np.stack(local_lower) + np.stack(local_upper)),
                0.5 * (np.stack(local_upper) - np.stack(local_lower)),
            )
            if len(local_segment_starts) == len(indices):
                self._local_segments_by_link[link_name] = (
                    np.stack(local_segment_starts),
                    np.stack(local_segment_ends),
                    np.asarray(local_segment_radii),
                )
        self.geometry_data = pin.GeometryData(self.geometry_model)
        self._clearance_components = {
            (str(pair["first"]), str(pair["second"])): {
                "clearance_basis": (
                    "reviewed_proxy_v1" if self.collision_envelope is not None
                    and self.collision_envelope.reviewed_proxy_clearance_m is not None
                    else "geometry_error_sum"
                ),
                "raw_geometry_error_sum_m": pair_static_clearance(pair),
                **{key: float(pair[key]) for key in (
                    "first_undercoverage_m", "second_undercoverage_m", "calibration_error_m",
                    "state_estimation_error_m", "stopping_distance_m")},
                "static_clearance_floor_m": float(pair.get("static_clearance_floor_m", 0.0)),
                "minimum_distance_m": float(pair["minimum_distance_m"]),
                "activation_distance_m": float(pair["activation_distance_m"]),
            } for pair in document["pairs"]
        }
        self._pairs = tuple(
            _LogicalPair(
                first=str(pair["first"]),
                second=str(pair["second"]),
                minimum_distance_m=float(pair["minimum_distance_m"]),
                activation_distance_m=float(pair["activation_distance_m"]),
                maximum_relative_step_m=float(pair["maximum_relative_step_m"]),
                first_indices=indices_by_link[str(pair["first"])],
                second_indices=indices_by_link[str(pair["second"])],
            )
            for pair in document["pairs"]
        )
        self._hysteresis_m = float(document["broadphase"]["hysteresis_m"])
        self._guard_distance_m = float(document["broadphase"]["guard_distance_m"])
        self._reference_dt_s = float(document["broadphase"]["reference_dt_s"])
        self._link_names = tuple(indices_by_link)
        self._link_index = {
            link_name: index for index, link_name in enumerate(self._link_names)
        }
        self._frame_ids = np.asarray(
            [self._frame_id_by_link[link_name] for link_name in self._link_names],
            dtype=int,
        )
        link_frames = [model.frames[int(frame_id)] for frame_id in self._frame_ids]
        self._link_parent_joints = np.asarray(
            [
                frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
                for frame in link_frames
            ],
            dtype=int,
        )
        self._unique_link_parent_joints, self._link_parent_inverse = np.unique(
            self._link_parent_joints,
            return_inverse=True,
        )
        self._link_frame_rotations_in_joint = np.stack(
            [np.asarray(frame.placement.rotation) for frame in link_frames]
        )
        self._link_frame_translations_in_joint = np.stack(
            [np.asarray(frame.placement.translation) for frame in link_frames]
        )
        primitive_count = len(self._records)
        self._primitive_parent_joints = np.asarray(
            [record.parent_joint for record in self._records], dtype=int
        )
        self._primitive_link_indices = np.empty(primitive_count, dtype=int)
        self._analytic_primitive_mask = np.zeros(primitive_count, dtype=bool)
        self._local_segment_starts = np.zeros((primitive_count, 3), dtype=float)
        self._local_segment_ends = np.zeros((primitive_count, 3), dtype=float)
        self._segment_radii = np.zeros(primitive_count, dtype=float)
        self._local_primitive_centers = np.zeros((primitive_count, 3), dtype=float)
        self._local_primitive_extents = np.zeros((primitive_count, 3), dtype=float)
        for link_name, geometry_indices in indices_by_link.items():
            indices = np.asarray(geometry_indices, dtype=int)
            self._primitive_link_indices[indices] = self._link_index[link_name]
            local_centers, local_extents = self._local_primitive_bounds_by_link[
                link_name
            ]
            self._local_primitive_centers[indices] = local_centers
            self._local_primitive_extents[indices] = local_extents
            segments = self._local_segments_by_link.get(link_name)
            if segments is None:
                continue
            starts, ends, radii = segments
            self._analytic_primitive_mask[indices] = True
            self._local_segment_starts[indices] = starts
            self._local_segment_ends[indices] = ends
            self._segment_radii[indices] = radii
        self._local_link_centers = np.stack(
            [
                0.5
                * (
                    self._local_bounds_by_link[link_name][0]
                    + self._local_bounds_by_link[link_name][1]
                )
                for link_name in self._link_names
            ]
        )
        self._local_link_extents = np.stack(
            [
                0.5
                * (
                    self._local_bounds_by_link[link_name][1]
                    - self._local_bounds_by_link[link_name][0]
                )
                for link_name in self._link_names
            ]
        )
        self._joint_local_link_centers = (
            np.einsum(
                "nij,nj->ni",
                self._link_frame_rotations_in_joint,
                self._local_link_centers,
            )
            + self._link_frame_translations_in_joint
        )
        self._link_radii = np.asarray(
            [
                0.5
                * np.linalg.norm(
                    self._local_bounds_by_link[link_name][1]
                    - self._local_bounds_by_link[link_name][0]
                )
                for link_name in self._link_names
            ]
        )
        # Conservative point-travel bounds let an all-clear broad phase be
        # reused while the measured configuration has moved too little to
        # close its previous clearance.  Each coefficient bounds how far any
        # point on a proxy link can move per unit tangent displacement of an
        # ancestor joint.  Fixed joint offsets are accumulated with the
        # triangle inequality, so this remains conservative for articulated
        # chains while avoiding Pinocchio placement work on most clear cycles.
        self._link_motion_coefficients = np.zeros(
            (len(self._link_names), model.nv), dtype=float
        )
        for link_index, parent_joint in enumerate(self._link_parent_joints):
            joint_id = int(parent_joint)
            reach_m = float(
                np.linalg.norm(self._joint_local_link_centers[link_index])
                + self._link_radii[link_index]
            )
            while joint_id > 0:
                joint = model.joints[joint_id]
                shortname = joint.shortname()
                velocity_slice = slice(joint.idx_v, joint.idx_v + joint.nv)
                if shortname.startswith("JointModelP"):
                    self._link_motion_coefficients[link_index, velocity_slice] = 1.0
                elif shortname == "JointModelComposite" and joint.nv == 3:
                    # The controller's supported mobile root is planar x/y/yaw.
                    self._link_motion_coefficients[link_index, joint.idx_v : joint.idx_v + 2] = 1.0
                    self._link_motion_coefficients[link_index, joint.idx_v + 2] = reach_m
                else:
                    self._link_motion_coefficients[link_index, velocity_slice] = max(
                        reach_m, 1.0 if joint.nv > 1 else 0.0
                    )
                reach_m += float(np.linalg.norm(model.jointPlacements[joint_id].translation))
                joint_id = int(model.parents[joint_id])
        self._pair_first_link_indices = np.asarray(
            [self._link_index[pair.first] for pair in self._pairs], dtype=int
        )
        self._pair_second_link_indices = np.asarray(
            [self._link_index[pair.second] for pair in self._pairs], dtype=int
        )
        first_motion_coefficients = self._link_motion_coefficients[
            self._pair_first_link_indices
        ]
        second_motion_coefficients = self._link_motion_coefficients[
            self._pair_second_link_indices
        ]
        shared_ancestors = (first_motion_coefficients > 0.0) & (
            second_motion_coefficients > 0.0
        )
        # Motion of a common ancestor is one rigid transform applied to both
        # links and therefore cannot change their mutual distance.  Only the
        # joints after the two kinematic paths diverge contribute to relative
        # proxy travel.
        self._pair_motion_coefficients = np.where(
            shared_ancestors,
            0.0,
            first_motion_coefficients + second_motion_coefficients,
        )
        self._pair_minimum_distances = np.asarray(
            [pair.minimum_distance_m for pair in self._pairs]
        )
        self._pair_activation_distances = np.asarray(
            [pair.activation_distance_m for pair in self._pairs]
        )
        self._pair_relative_steps = np.asarray(
            [pair.maximum_relative_step_m for pair in self._pairs]
        )
        self._analytic_pair_mask = np.asarray(
            [
                bool(
                    np.all(self._analytic_primitive_mask[np.asarray(pair.first_indices)])
                    and np.all(self._analytic_primitive_mask[np.asarray(pair.second_indices)])
                )
                for pair in self._pairs
            ],
            dtype=bool,
        )
        analytic_combo_pairs: list[np.ndarray] = []
        analytic_combo_first: list[np.ndarray] = []
        analytic_combo_second: list[np.ndarray] = []
        primitive_combo_pairs: list[np.ndarray] = []
        primitive_combo_first: list[np.ndarray] = []
        primitive_combo_second: list[np.ndarray] = []
        for pair_index, pair in enumerate(self._pairs):
            first = np.asarray(pair.first_indices, dtype=int)
            second = np.asarray(pair.second_indices, dtype=int)
            combo_pairs = np.full(first.size * second.size, pair_index, dtype=int)
            combo_first = np.repeat(first, second.size)
            combo_second = np.tile(second, first.size)
            primitive_combo_pairs.append(combo_pairs)
            primitive_combo_first.append(combo_first)
            primitive_combo_second.append(combo_second)
            if self._analytic_pair_mask[pair_index]:
                analytic_combo_pairs.append(combo_pairs)
                analytic_combo_first.append(combo_first)
                analytic_combo_second.append(combo_second)
        self._primitive_combo_pair_indices = np.concatenate(primitive_combo_pairs)
        self._primitive_combo_first_indices = np.concatenate(primitive_combo_first)
        self._primitive_combo_second_indices = np.concatenate(primitive_combo_second)
        self._analytic_combo_pair_indices = (
            np.concatenate(analytic_combo_pairs)
            if analytic_combo_pairs
            else np.empty(0, dtype=int)
        )
        self._analytic_combo_first_indices = (
            np.concatenate(analytic_combo_first)
            if analytic_combo_first
            else np.empty(0, dtype=int)
        )
        self._analytic_combo_second_indices = (
            np.concatenate(analytic_combo_second)
            if analytic_combo_second
            else np.empty(0, dtype=int)
        )
        # Pinocchio owns reusable native distance requests/results. This avoids
        # rebuilding Python-to-FCL transforms for every ellipsoid primitive pair.
        self._native_distance_indices: dict[tuple[int, int], int] = {}
        if self.collision_envelope is not None and self.collision_envelope.activation_only:
            for pair_index, pair in enumerate(self._pairs):
                if self._analytic_pair_mask[pair_index]:
                    continue
                for first in pair.first_indices:
                    for second in pair.second_indices:
                        key = (first, second)
                        if key not in self._native_distance_indices:
                            self._native_distance_indices[key] = len(self.geometry_model.collisionPairs)
                            self.geometry_model.addCollisionPair(pin.CollisionPair(first, second))
            self.geometry_data = pin.GeometryData(self.geometry_model)
            for index in self._native_distance_indices.values():
                self.geometry_data.distanceRequests[index] = hppfcl.DistanceRequest(True)
        nonanalytic_combo_mask = ~self._analytic_pair_mask[
            self._primitive_combo_pair_indices
        ]
        self._native_combo_pair_indices = self._primitive_combo_pair_indices[
            nonanalytic_combo_mask
        ]
        self._native_combo_first_indices = self._primitive_combo_first_indices[
            nonanalytic_combo_mask
        ]
        self._native_combo_second_indices = self._primitive_combo_second_indices[
            nonanalytic_combo_mask
        ]
        self._native_combo_distance_indices = np.fromiter(
            (
                self._native_distance_indices.get((int(first), int(second)), -1)
                for first, second in zip(
                    self._native_combo_first_indices,
                    self._native_combo_second_indices,
                    strict=True,
                )
            ),
            dtype=int,
            count=len(self._native_combo_pair_indices),
        )
        kind_codes = {"sphere": 0, "capsule": 1, "ellipsoid": 2}
        primitive_kinds = np.asarray(
            [kind_codes[record.kind] for record in self._records], dtype=int
        )
        primitive_dimensions = np.zeros((len(self._records), 3), dtype=float)
        local_rotations = np.empty((len(self._records), 3, 3), dtype=float)
        local_translations = np.empty((len(self._records), 3), dtype=float)
        for record in self._records:
            geometry_index = record.geometry_index
            if record.kind == "sphere":
                primitive_dimensions[geometry_index, 0] = float(record.radius_m)
            elif record.kind == "capsule":
                primitive_dimensions[geometry_index, :2] = (
                    float(record.radius_m),
                    2.0 * float(record.half_length_m),
                )
            else:
                primitive_dimensions[geometry_index] = np.asarray(
                    record.radii_m, dtype=float
                )
            placement = self.geometry_model.geometryObjects[geometry_index].placement
            local_rotations[geometry_index] = np.asarray(placement.rotation)
            local_translations[geometry_index] = np.asarray(placement.translation)
        self._primitive_rotations_in_joint = local_rotations
        self._primitive_translations_in_joint = local_translations
        self._native_fcl_kernel = None
        if _NativeFclUnionKernel is not None and self._native_distance_indices:
            self._native_fcl_kernel = _NativeFclUnionKernel(
                primitive_kinds,
                primitive_dimensions,
                self._native_combo_pair_indices,
                self._native_combo_first_indices,
                self._native_combo_second_indices,
            )
        self._native_collision_kernel_v2 = (
            None
            if runtime != "native_fused_v2"
            else _NativeCollisionKernelV2(
                primitive_kinds,
                primitive_dimensions,
                self._primitive_combo_pair_indices,
                self._primitive_combo_first_indices,
                self._primitive_combo_second_indices,
                self._pair_first_link_indices,
                self._pair_second_link_indices,
                self._local_link_centers,
                self._local_link_extents,
                self._link_radii,
            )
        )
        self._native_exact_pair_indices = np.empty(0, dtype=int)
        self._native_exact_first_indices = np.empty(0, dtype=int)
        self._native_exact_second_indices = np.empty(0, dtype=int)
        self._native_exact_distances = np.empty(0, dtype=float)
        self._native_exact_first_points = np.empty((0, 3), dtype=float)
        self._native_exact_second_points = np.empty((0, 3), dtype=float)
        self._native_exact_fallback_normals = np.empty((0, 3), dtype=float)
        self._native_barrier_mask = np.empty(0, dtype=bool)
        self._active_pair_indices: set[int] = set()
        self._last_distances: dict[int, _ActiveDistance] = {}
        self._last_lower_bounds = np.full(len(self._pairs), np.inf)
        self._last_barrier: ActivePairCollisionBarrier | None = None
        self._last_configuration_q: np.ndarray | None = None
        self._current_segments: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._current_link_rotations: np.ndarray | None = None
        self._current_link_translations: np.ndarray | None = None
        self._current_link_parent_rotations: np.ndarray | None = None
        self._current_link_parent_translations: np.ndarray | None = None
        self._broadphase_diagnostics = BroadPhaseDiagnostics(
            logical_pair_count=len(self._pairs),
            sphere_candidate_count=0,
            aabb_candidate_count=0,
            active_pair_count=0,
            exact_distance_calls=0,
            primitive_pair_count=sum(
                len(pair.first_indices) * len(pair.second_indices) for pair in self._pairs
            ),
            configured_pair_count=self._configured_pair_count,
            certified_inactive_pair_count=self._certified_inactive_pair_count,
        )

    @classmethod
    def for_review(cls, model: pin.Model, config_path: str | Path) -> SelfCollisionSystem:
        """Load a candidate for review without weakening production defaults."""

        return cls(model, config_path, require_approved=False)

    @property
    def broadphase_diagnostics(self) -> BroadPhaseDiagnostics:
        return self._broadphase_diagnostics

    def certifies_tangent_motion(self, tangent: np.ndarray) -> bool:
        """Prove every proxy pair clear along a tangent-space chord.

        Distance is one-Lipschitz in relative point travel. The stored broad-
        phase values are conservative lower bounds, so subtracting the maximum
        link travel certifies the entire chord rather than sampling it.
        """

        tangent = np.abs(np.asarray(tangent, dtype=float))
        if (
            self._last_configuration_q is None
            or tangent.shape != (self._model.nv,)
            or not np.all(np.isfinite(tangent))
        ):
            return False
        relative_travel = self._pair_motion_coefficients @ tangent
        chord_lower_bounds = self._last_lower_bounds - relative_travel
        return bool(
            np.all(
                chord_lower_bounds
                > self._pair_minimum_distances + self._guard_distance_m
            )
        )

    def _ensure_current_link_placements(self) -> None:
        if (
            self._current_link_rotations is not None
            and self._current_link_translations is not None
        ):
            return
        if (
            self._current_link_parent_rotations is None
            or self._current_link_parent_translations is None
        ):
            raise RuntimeError("self-collision parent placements have not been updated")
        self._current_link_rotations = np.einsum(
            "nij,njk->nik",
            self._current_link_parent_rotations,
            self._link_frame_rotations_in_joint,
        )
        self._current_link_translations = (
            np.einsum(
                "nij,nj->ni",
                self._current_link_parent_rotations,
                self._link_frame_translations_in_joint,
            )
            + self._current_link_parent_translations
        )

    def _primitive_aabbs(
        self,
        geometry_indices: Sequence[int],
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for geometry_index in geometry_indices:
            record = self._records[int(geometry_index)]
            placement = self.geometry_data.oMg[record.geometry_index]
            center = np.asarray(placement.translation, dtype=float)
            rotation = np.asarray(placement.rotation, dtype=float)
            if record.kind == "sphere":
                extent = np.full(3, float(record.radius_m))
            elif record.kind == "ellipsoid":
                extent = np.abs(rotation) @ np.asarray(record.radii_m, dtype=float)
            else:
                extent = np.abs(rotation[:, 2]) * float(record.half_length_m) + float(record.radius_m)
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
                raise RuntimeError("self-collision broad phase produced non-finite geometry bounds")
            bounds[record.geometry_index] = (center - extent, center + extent)
        return bounds

    @staticmethod
    def _transform_aabb(
        placement: pin.SE3, local_bound: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        local_center = 0.5 * (local_bound[0] + local_bound[1])
        local_extent = 0.5 * (local_bound[1] - local_bound[0])
        world_center = placement.rotation @ local_center + placement.translation
        world_extent = np.abs(placement.rotation) @ local_extent
        return world_center - world_extent, world_center + world_extent

    def _exact_distance(
        self,
        pair: _LogicalPair,
        primitive_bounds: Mapping[int, tuple[np.ndarray, np.ndarray]] | None,
    ) -> tuple[_ActiveDistance, int]:
        if all(
            self._records[index].kind in {"sphere", "capsule"}
            for index in (*pair.first_indices, *pair.second_indices)
        ):
            return self._analytic_sphere_capsule_distance(pair)
        if self._native_distance_indices:
            indices = [self._native_distance_indices[first, second]
                       for first in pair.first_indices for second in pair.second_indices]
            index = min(indices, key=lambda value: self.geometry_data.distanceResults[value].min_distance)
            result = self.geometry_data.distanceResults[index]
            distance = float(result.min_distance)
            if not np.isfinite(distance):
                raise RuntimeError("exact self-collision distance is non-finite")
            native_pair = self.geometry_model.collisionPairs[index]
            first, second = int(native_pair.first), int(native_pair.second)
            return _ActiveDistance(
                pair, first, second, distance,
                np.asarray(result.getNearestPoint1(), dtype=float).copy(),
                np.asarray(result.getNearestPoint2(), dtype=float).copy(),
                np.asarray(self.geometry_data.oMg[first].translation
                           - self.geometry_data.oMg[second].translation).copy(),
            ), 0  # Counted once for the native batch in update().
        if primitive_bounds is None:
            raise RuntimeError("ellipsoid exact distance requires primitive bounds")
        candidates = sorted(
            (
                (
                    _aabb_signed_lower_bound(
                        primitive_bounds[first], primitive_bounds[second]
                    ),
                    first,
                    second,
                )
                for first in pair.first_indices
                for second in pair.second_indices
            ),
            key=lambda candidate: candidate[0],
        )
        request = hppfcl.DistanceRequest(True)
        best: _ActiveDistance | None = None
        calls = 0
        for lower_bound, first, second in candidates:
            if best is not None and lower_bound > best.distance_m:
                break
            first_object = self.geometry_model.geometryObjects[first]
            second_object = self.geometry_model.geometryObjects[second]
            first_placement = self.geometry_data.oMg[first]
            second_placement = self.geometry_data.oMg[second]
            result = hppfcl.DistanceResult()
            distance = float(
                hppfcl.distance(
                    first_object.geometry,
                    hppfcl.Transform3f(first_placement.rotation, first_placement.translation),
                    second_object.geometry,
                    hppfcl.Transform3f(second_placement.rotation, second_placement.translation),
                    request,
                    result,
                )
            )
            calls += 1
            if not np.isfinite(distance):
                raise RuntimeError("exact self-collision distance is non-finite")
            if best is None or distance < best.distance_m:
                best = _ActiveDistance(
                    pair,
                    first,
                    second,
                    distance,
                    np.asarray(result.getNearestPoint1(), dtype=float).copy(),
                    np.asarray(result.getNearestPoint2(), dtype=float).copy(),
                    np.asarray(first_placement.translation - second_placement.translation).copy(),
                )
        if best is None:
            raise RuntimeError("active self-collision pair has no primitive candidates")
        return best, calls

    def _batch_native_distances(
        self,
        pair_indices: Sequence[int],
    ) -> tuple[dict[int, _ActiveDistance], int]:
        """Evaluate active native unions with one grouped result reduction.

        Native HPP-FCL calls remain candidate-only. The grouping removes the
        former per-logical-pair list/lambda search while preserving the exact
        minimum primitive distance and its witness points.
        """

        if not pair_indices:
            return {}, 0
        if not self._native_distance_indices:
            raise RuntimeError("native collision batch is not configured")
        if self._native_fcl_kernel is not None:
            link_indices = self._primitive_link_indices
            parent_rotations = self._current_link_parent_rotations[link_indices]
            parent_translations = self._current_link_parent_translations[link_indices]
            world_rotations = np.einsum(
                "nij,njk->nik",
                parent_rotations,
                self._primitive_rotations_in_joint,
            )
            world_translations = (
                np.einsum(
                    "nij,nj->ni",
                    parent_rotations,
                    self._primitive_translations_in_joint,
                )
                + parent_translations
            )
            (
                native_pairs,
                native_first,
                native_second,
                native_distances,
                native_first_points,
                native_second_points,
                native_fallback_normals,
                exact_calls,
            ) = self._native_fcl_kernel.compute(
                np.asarray(pair_indices, dtype=int),
                world_rotations,
                world_translations,
            )
            results = {
                int(pair_index): _ActiveDistance(
                    self._pairs[int(pair_index)],
                    int(native_first[index]),
                    int(native_second[index]),
                    float(native_distances[index]),
                    native_first_points[index].copy(),
                    native_second_points[index].copy(),
                    native_fallback_normals[index].copy(),
                )
                for index, pair_index in enumerate(native_pairs)
            }
            return results, int(exact_calls)
        active_pair_mask = np.zeros(len(self._pairs), dtype=bool)
        active_pair_mask[np.asarray(pair_indices, dtype=int)] = True
        combo_mask = active_pair_mask[self._native_combo_pair_indices]
        combo_pair_indices = self._native_combo_pair_indices[combo_mask]
        first_geometry_indices = self._native_combo_first_indices[combo_mask]
        second_geometry_indices = self._native_combo_second_indices[combo_mask]
        distance_indices = self._native_combo_distance_indices[combo_mask]
        if distance_indices.size == 0 or np.any(distance_indices < 0):
            raise RuntimeError("active native collision pairs are not registered")

        for distance_index in distance_indices:
            pin.computeDistance(
                self.geometry_model,
                self.geometry_data,
                int(distance_index),
            )
        distances = np.fromiter(
            (
                self.geometry_data.distanceResults[int(distance_index)].min_distance
                for distance_index in distance_indices
            ),
            dtype=float,
            count=len(distance_indices),
        )
        if not np.all(np.isfinite(distances)):
            raise RuntimeError("exact self-collision distance is non-finite")
        pair_starts = np.flatnonzero(
            np.concatenate(
                (
                    np.array((True,)),
                    combo_pair_indices[1:] != combo_pair_indices[:-1],
                )
            )
        )
        pair_ends = np.append(pair_starts[1:], len(combo_pair_indices))
        results: dict[int, _ActiveDistance] = {}
        for start, end in zip(pair_starts, pair_ends, strict=True):
            best = int(start + np.argmin(distances[start:end]))
            pair_index = int(combo_pair_indices[best])
            first = int(first_geometry_indices[best])
            second = int(second_geometry_indices[best])
            result = self.geometry_data.distanceResults[int(distance_indices[best])]
            results[pair_index] = _ActiveDistance(
                self._pairs[pair_index],
                first,
                second,
                float(distances[best]),
                np.asarray(result.getNearestPoint1(), dtype=float).copy(),
                np.asarray(result.getNearestPoint2(), dtype=float).copy(),
                np.asarray(
                    self.geometry_data.oMg[first].translation
                    - self.geometry_data.oMg[second].translation
                ).copy(),
            )
        return results, len(distance_indices)

    def _batch_analytic_sphere_capsule_distances(
        self,
        pair_indices: Sequence[int],
    ) -> tuple[dict[int, _ActiveDistance], int]:
        """Evaluate all active ball/capsule unions in one vectorized batch."""

        if not pair_indices:
            return {}, 0
        active_pair_mask = np.zeros(len(self._pairs), dtype=bool)
        active_pair_mask[np.asarray(pair_indices, dtype=int)] = True
        combo_mask = active_pair_mask[self._analytic_combo_pair_indices]
        combo_pair_indices = self._analytic_combo_pair_indices[combo_mask]
        first_geometry_indices = self._analytic_combo_first_indices[combo_mask]
        second_geometry_indices = self._analytic_combo_second_indices[combo_mask]
        if combo_pair_indices.size == 0:
            raise RuntimeError("active analytic collision pairs have no primitive combinations")

        primitive_indices = np.unique(
            np.concatenate((first_geometry_indices, second_geometry_indices))
        )
        link_indices = self._primitive_link_indices[primitive_indices]
        primitive_rotations = np.einsum(
            "nij,njk->nik",
            self._current_link_parent_rotations[link_indices],
            self._link_frame_rotations_in_joint[link_indices],
        )
        primitive_translations = (
            np.einsum(
                "nij,nj->ni",
                self._current_link_parent_rotations[link_indices],
                self._link_frame_translations_in_joint[link_indices],
            )
            + self._current_link_parent_translations[link_indices]
        )
        world_starts = np.empty((len(self._records), 3), dtype=float)
        world_ends = np.empty((len(self._records), 3), dtype=float)
        world_starts[primitive_indices] = (
            np.einsum(
                "nij,nj->ni",
                primitive_rotations,
                self._local_segment_starts[primitive_indices],
            )
            + primitive_translations
        )
        world_ends[primitive_indices] = (
            np.einsum(
                "nij,nj->ni",
                primitive_rotations,
                self._local_segment_ends[primitive_indices],
            )
            + primitive_translations
        )

        if _native_segment_unions is not None:
            (
                native_pair_indices,
                native_combination_indices,
                native_distances,
                native_first_points,
                native_second_points,
                native_fallback_normals,
            ) = _native_segment_unions(
                combo_pair_indices,
                first_geometry_indices,
                second_geometry_indices,
                world_starts,
                world_ends,
                self._segment_radii,
            )
            results: dict[int, _ActiveDistance] = {}
            for output_index, pair_index_value in enumerate(native_pair_indices):
                combination_index = int(native_combination_indices[output_index])
                pair_index = int(pair_index_value)
                first_geometry_index = int(first_geometry_indices[combination_index])
                second_geometry_index = int(second_geometry_indices[combination_index])
                results[pair_index] = _ActiveDistance(
                    self._pairs[pair_index],
                    first_geometry_index,
                    second_geometry_index,
                    float(native_distances[output_index]),
                    native_first_points[output_index].copy(),
                    native_second_points[output_index].copy(),
                    native_fallback_normals[output_index].copy(),
                )
            return results, int(combo_pair_indices.size)

        first_start = world_starts[first_geometry_indices]
        first_end = world_ends[first_geometry_indices]
        second_start = world_starts[second_geometry_indices]
        second_end = world_ends[second_geometry_indices]
        first_axis = first_end - first_start
        second_axis = second_end - second_start
        offset = first_start - second_start
        first_length_squared = np.einsum("ij,ij->i", first_axis, first_axis)
        second_length_squared = np.einsum("ij,ij->i", second_axis, second_axis)
        axes_dot = np.einsum("ij,ij->i", first_axis, second_axis)
        first_offset_dot = np.einsum("ij,ij->i", first_axis, offset)
        second_offset_dot = np.einsum("ij,ij->i", second_axis, offset)
        denominator = (
            first_length_squared * second_length_squared - axes_dot * axes_dot
        )
        combination_count = combo_pair_indices.size
        s_values = np.zeros((5, combination_count), dtype=float)
        t_values = np.zeros((5, combination_count), dtype=float)
        valid = np.ones((5, combination_count), dtype=bool)
        nonparallel = denominator > 1e-15
        np.divide(
            axes_dot * second_offset_dot
            - second_length_squared * first_offset_dot,
            denominator,
            out=s_values[0],
            where=nonparallel,
        )
        np.divide(
            first_length_squared * second_offset_dot
            - axes_dot * first_offset_dot,
            denominator,
            out=t_values[0],
            where=nonparallel,
        )
        valid[0] = (
            nonparallel
            & (s_values[0] >= 0.0)
            & (s_values[0] <= 1.0)
            & (t_values[0] >= 0.0)
            & (t_values[0] <= 1.0)
        )

        def safe_ratio(
            numerator: np.ndarray,
            divisor: np.ndarray,
        ) -> np.ndarray:
            result = np.zeros(combination_count, dtype=float)
            np.divide(numerator, divisor, out=result, where=divisor > 1e-15)
            return np.clip(result, 0.0, 1.0)

        s_values[1] = 0.0
        s_values[2] = 1.0
        t_values[1] = safe_ratio(second_offset_dot, second_length_squared)
        t_values[2] = safe_ratio(
            second_offset_dot + axes_dot,
            second_length_squared,
        )
        t_values[3] = 0.0
        t_values[4] = 1.0
        s_values[3] = safe_ratio(-first_offset_dot, first_length_squared)
        s_values[4] = safe_ratio(
            axes_dot - first_offset_dot,
            first_length_squared,
        )

        first_points = (
            first_start[np.newaxis]
            + s_values[..., np.newaxis] * first_axis[np.newaxis]
        )
        second_points = (
            second_start[np.newaxis]
            + t_values[..., np.newaxis] * second_axis[np.newaxis]
        )
        squared_distances = np.sum(
            np.square(first_points - second_points),
            axis=2,
        )
        squared_distances[~valid] = np.inf
        radius_sum = (
            self._segment_radii[first_geometry_indices]
            + self._segment_radii[second_geometry_indices]
        )
        signed_distances = np.sqrt(squared_distances) - radius_sum[np.newaxis]
        candidate_indices = np.argmin(signed_distances, axis=0)
        combination_indices = np.arange(combination_count)
        combination_distances = signed_distances[
            candidate_indices,
            combination_indices,
        ]
        first_axis_points = first_points[candidate_indices, combination_indices]
        second_axis_points = second_points[candidate_indices, combination_indices]

        pair_starts = np.flatnonzero(
            np.concatenate(
                (
                    np.array((True,)),
                    combo_pair_indices[1:] != combo_pair_indices[:-1],
                )
            )
        )
        pair_ends = np.concatenate((pair_starts[1:], np.array((combination_count,))))
        results: dict[int, _ActiveDistance] = {}
        for start, end in zip(pair_starts, pair_ends, strict=True):
            combination_index = int(
                start + np.argmin(combination_distances[start:end])
            )
            pair_index = int(combo_pair_indices[combination_index])
            first_geometry_index = int(first_geometry_indices[combination_index])
            second_geometry_index = int(second_geometry_indices[combination_index])
            first_axis_point = first_axis_points[combination_index]
            second_axis_point = second_axis_points[combination_index]
            delta = first_axis_point - second_axis_point
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-12:
                delta = 0.5 * (
                    first_start[combination_index]
                    + first_end[combination_index]
                    - second_start[combination_index]
                    - second_end[combination_index]
                )
                norm = float(np.linalg.norm(delta))
            normal = (
                delta / norm
                if norm > 1e-12
                else np.array((1.0, 0.0, 0.0))
            )
            first_point = (
                first_axis_point
                - normal * self._segment_radii[first_geometry_index]
            )
            second_point = (
                second_axis_point
                + normal * self._segment_radii[second_geometry_index]
            )
            results[pair_index] = _ActiveDistance(
                self._pairs[pair_index],
                first_geometry_index,
                second_geometry_index,
                float(combination_distances[combination_index]),
                first_point,
                second_point,
                delta,
            )
        return results, int(combination_count)

    def _analytic_sphere_capsule_distance(
        self, pair: _LogicalPair
    ) -> tuple[_ActiveDistance, int]:
        """Vectorized exact distance between two unions of balls/capsules."""

        def segments(link_name: str, indices: tuple[int, ...]):
            cached = self._current_segments.get(link_name)
            if cached is not None:
                return cached
            local_start, local_end, radii = self._local_segments_by_link[link_name]
            self._ensure_current_link_placements()
            if self._current_link_rotations is None or self._current_link_translations is None:
                raise RuntimeError("self-collision link placements have not been resolved")
            link_index = self._link_index[link_name]
            rotation = self._current_link_rotations[link_index]
            translation = self._current_link_translations[link_index]
            resolved = (
                local_start @ rotation.T + translation,
                local_end @ rotation.T + translation,
                radii,
            )
            self._current_segments[link_name] = resolved
            return resolved

        first_start, first_end, first_radius = segments(pair.first, pair.first_indices)
        second_start, second_end, second_radius = segments(pair.second, pair.second_indices)
        first_start = first_start[:, np.newaxis, :]
        first_end = first_end[:, np.newaxis, :]
        second_start = second_start[np.newaxis, :, :]
        second_end = second_end[np.newaxis, :, :]
        first_axis = first_end - first_start
        second_axis = second_end - second_start
        offset = first_start - second_start
        first_length_squared = np.sum(first_axis * first_axis, axis=2)
        second_length_squared = np.sum(second_axis * second_axis, axis=2)
        axes_dot = np.sum(first_axis * second_axis, axis=2)
        first_offset_dot = np.sum(first_axis * offset, axis=2)
        second_offset_dot = np.sum(second_axis * offset, axis=2)
        denominator = (
            first_length_squared * second_length_squared - axes_dot * axes_dot
        )

        shape = denominator.shape
        candidate_s: list[np.ndarray] = []
        candidate_t: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        interior_s = np.zeros(shape)
        interior_t = np.zeros(shape)
        nonparallel = denominator > 1e-15
        np.divide(
            axes_dot * second_offset_dot
            - second_length_squared * first_offset_dot,
            denominator,
            out=interior_s,
            where=nonparallel,
        )
        np.divide(
            first_length_squared * second_offset_dot
            - axes_dot * first_offset_dot,
            denominator,
            out=interior_t,
            where=nonparallel,
        )
        candidate_s.append(interior_s)
        candidate_t.append(interior_t)
        valid.append(
            nonparallel
            & (interior_s >= 0.0)
            & (interior_s <= 1.0)
            & (interior_t >= 0.0)
            & (interior_t <= 1.0)
        )

        def safe_ratio(numerator: np.ndarray, divisor: np.ndarray) -> np.ndarray:
            result = np.zeros(shape)
            np.divide(numerator, divisor, out=result, where=divisor > 1e-15)
            return np.clip(result, 0.0, 1.0)

        candidate_s.extend((np.zeros(shape), np.ones(shape)))
        candidate_t.extend(
            (
                safe_ratio(second_offset_dot, second_length_squared),
                safe_ratio(second_offset_dot + axes_dot, second_length_squared),
            )
        )
        valid.extend((np.ones(shape, dtype=bool), np.ones(shape, dtype=bool)))
        candidate_t.extend((np.zeros(shape), np.ones(shape)))
        candidate_s.extend(
            (
                safe_ratio(-first_offset_dot, first_length_squared),
                safe_ratio(axes_dot - first_offset_dot, first_length_squared),
            )
        )
        valid.extend((np.ones(shape, dtype=bool), np.ones(shape, dtype=bool)))

        s_values = np.stack(candidate_s)
        t_values = np.stack(candidate_t)
        first_points = first_start[np.newaxis] + s_values[..., np.newaxis] * first_axis[
            np.newaxis
        ]
        second_points = second_start[np.newaxis] + t_values[..., np.newaxis] * second_axis[
            np.newaxis
        ]
        squared_distances = np.sum(np.square(first_points - second_points), axis=3)
        squared_distances[~np.stack(valid)] = np.inf
        radius_sum = first_radius[:, np.newaxis] + second_radius[np.newaxis, :]
        signed_distances = np.sqrt(squared_distances) - radius_sum[np.newaxis]
        candidate_index, first_index, second_index = np.unravel_index(
            int(np.argmin(signed_distances)), signed_distances.shape
        )
        first_axis_point = first_points[candidate_index, first_index, second_index]
        second_axis_point = second_points[candidate_index, first_index, second_index]
        delta = first_axis_point - second_axis_point
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            delta = (
                0.5 * (first_start[first_index, 0] + first_end[first_index, 0])
                - 0.5 * (second_start[0, second_index] + second_end[0, second_index])
            )
            norm = float(np.linalg.norm(delta))
        normal = delta / norm if norm > 1e-12 else np.array((1.0, 0.0, 0.0))
        first_point = first_axis_point - normal * first_radius[first_index]
        second_point = second_axis_point + normal * second_radius[second_index]
        result = _ActiveDistance(
            pair,
            pair.first_indices[first_index],
            pair.second_indices[second_index],
            float(signed_distances[candidate_index, first_index, second_index]),
            first_point,
            second_point,
            delta,
        )
        return result, len(pair.first_indices) * len(pair.second_indices)

    def _update_native_v2(
        self,
        configuration: Configuration,
        thresholds: np.ndarray,
        cached_lower_bounds: np.ndarray | None,
    ) -> tuple[Barrier, ...]:
        """Run the v2 broad phase, exact distances, and witness reduction natively."""

        if self._native_collision_kernel_v2 is None:
            raise RuntimeError("native_fused_v2 kernel was not initialized")
        parent_placements = [
            configuration.data.oMi[int(joint_id)]
            for joint_id in self._unique_link_parent_joints
        ]
        parent_rotations = np.stack(
            [np.asarray(placement.rotation) for placement in parent_placements]
        )
        parent_translations = np.stack(
            [np.asarray(placement.translation) for placement in parent_placements]
        )
        link_parent_rotations = parent_rotations[self._link_parent_inverse]
        link_parent_translations = parent_translations[self._link_parent_inverse]
        link_rotations = np.einsum(
            "nij,njk->nik", link_parent_rotations, self._link_frame_rotations_in_joint
        )
        link_translations = (
            np.einsum(
                "nij,nj->ni",
                link_parent_rotations,
                self._link_frame_translations_in_joint,
            )
            + link_parent_translations
        )
        primitive_links = self._primitive_link_indices
        primitive_parent_rotations = link_parent_rotations[primitive_links]
        primitive_rotations = np.einsum(
            "nij,njk->nik",
            primitive_parent_rotations,
            self._primitive_rotations_in_joint,
        )
        primitive_translations = (
            np.einsum(
                "nij,nj->ni",
                primitive_parent_rotations,
                self._primitive_translations_in_joint,
            )
            + link_parent_translations[primitive_links]
        )
        cached = (
            np.zeros(len(self._pairs), dtype=float)
            if cached_lower_bounds is None
            else cached_lower_bounds
        )
        assembly_started = time.perf_counter()
        (
            pair_indices,
            first_indices,
            second_indices,
            distances,
            first_points,
            second_points,
            fallback_normals,
            barrier_mask,
            lower_bounds,
            sphere_count,
            aabb_count,
            exact_calls,
            _analytic_calls,
            _fcl_calls,
            broadphase_ms,
            exact_ms,
        ) = self._native_collision_kernel_v2.compute(
            thresholds,
            cached,
            link_rotations,
            link_translations,
            primitive_rotations,
            primitive_translations,
        )
        self._native_exact_pair_indices = np.asarray(pair_indices, dtype=int)
        self._native_exact_first_indices = np.asarray(first_indices, dtype=int)
        self._native_exact_second_indices = np.asarray(second_indices, dtype=int)
        self._native_exact_distances = np.asarray(distances, dtype=float)
        self._native_exact_first_points = np.asarray(first_points, dtype=float)
        self._native_exact_second_points = np.asarray(second_points, dtype=float)
        self._native_exact_fallback_normals = np.asarray(fallback_normals, dtype=float)
        self._native_barrier_mask = np.asarray(barrier_mask, dtype=bool)
        if (
            self._native_exact_pair_indices.shape != self._native_barrier_mask.shape
            or np.asarray(lower_bounds).shape != (len(self._pairs),)
            or not np.all(np.isfinite(lower_bounds))
        ):
            raise RuntimeError("native_fused_v2 returned invalid collision arrays")
        barrier_pairs = self._native_exact_pair_indices[self._native_barrier_mask]
        self._active_pair_indices = {int(index) for index in barrier_pairs}
        self._last_configuration_q = configuration.q.copy()
        self._last_lower_bounds = np.asarray(lower_bounds, dtype=float)
        self._last_distances = {}
        self._current_segments = {}
        self._current_link_rotations = link_rotations
        self._current_link_translations = link_translations
        self._current_link_parent_rotations = link_parent_rotations
        self._current_link_parent_translations = link_parent_translations
        self._last_barrier = (
            None
            if not len(barrier_pairs)
            else ActivePairCollisionBarrier.from_compact_arrays(
                pair_indices=barrier_pairs,
                logical_pairs=self._pairs,
                first_geometry_indices=self._native_exact_first_indices[
                    self._native_barrier_mask
                ],
                second_geometry_indices=self._native_exact_second_indices[
                    self._native_barrier_mask
                ],
                distances_m=self._native_exact_distances[self._native_barrier_mask],
                first_points=self._native_exact_first_points[self._native_barrier_mask],
                second_points=self._native_exact_second_points[self._native_barrier_mask],
                fallback_normals=self._native_exact_fallback_normals[
                    self._native_barrier_mask
                ],
                geometry_model=self.geometry_model,
                geometry_parent_joints=self._primitive_parent_joints,
            )
        )
        self._broadphase_diagnostics = BroadPhaseDiagnostics(
            logical_pair_count=len(self._pairs),
            sphere_candidate_count=int(sphere_count),
            aabb_candidate_count=int(aabb_count),
            active_pair_count=len(barrier_pairs),
            exact_distance_calls=int(exact_calls),
            primitive_pair_count=self._broadphase_diagnostics.primitive_pair_count,
            configured_pair_count=self._configured_pair_count,
            certified_inactive_pair_count=self._certified_inactive_pair_count,
            broadphase_ms=float(broadphase_ms),
            native_distance_ms=float(exact_ms),
            barrier_assembly_ms=1000.0 * (time.perf_counter() - assembly_started)
            - float(broadphase_ms)
            - float(exact_ms),
        )
        return () if self._last_barrier is None else (self._last_barrier,)

    def update(
        self,
        configuration: Configuration,
        dt: float,
        *,
        joint_safety_travel: np.ndarray | None = None,
        joint_activation_travel: np.ndarray | None = None,
        joint_position_uncertainty: np.ndarray | None = None,
    ) -> tuple[Barrier, ...]:
        """Update conservative activation and return current-cycle barrier rows."""

        broadphase_started = time.perf_counter()

        if self.pair_certificate is not None:
            certificate = self.pair_certificate
            certified_q = configuration.q[certificate.joint_indices]
            if np.any(certified_q < certificate.lower_position - 1e-9) or np.any(
                certified_q > certificate.upper_position + 1e-9
            ):
                raise ValueError("measured configuration is outside the certified collision domain")

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("self-collision update timestep must be finite and positive")
        if configuration.model is not self._model:
            raise ValueError("self-collision system and Pink configuration use different models")
        if (self.collision_envelope is not None
                and self.collision_envelope.requires_dynamic_travel
                and joint_safety_travel is None):
            raise ValueError("MIT collision envelope requires per-cycle joint travel bounds")
        thresholds = self._pair_activation_distances
        uncertainty = np.zeros(len(self._pairs))
        if joint_position_uncertainty is not None:
            value = np.asarray(joint_position_uncertainty, dtype=float)
            if value.shape != (self._model.nv,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError("invalid joint position uncertainty")
            uncertainty = self._pair_motion_coefficients @ value
        if joint_activation_travel is not None:
            activation_travel = np.asarray(joint_activation_travel, dtype=float)
            if (
                activation_travel.shape != (self._model.nv,)
                or not np.all(np.isfinite(activation_travel))
                or np.any(activation_travel < 0.0)
            ):
                raise ValueError(
                    "joint activation travel must be a finite nonnegative tangent vector"
                )
            thresholds = (
                self._pair_minimum_distances
                + self._pair_motion_coefficients @ activation_travel
                + self._guard_distance_m
            )
        elif dt != self._reference_dt_s:
            thresholds = (
                self._pair_minimum_distances
                + self._pair_relative_steps * (dt / self._reference_dt_s)
                + self._guard_distance_m
            )
        pair_reserve = np.zeros(len(self._pairs))
        if joint_safety_travel is not None:
            travel = np.asarray(joint_safety_travel, dtype=float)
            if (travel.shape != (self._model.nv,) or not np.all(np.isfinite(travel))
                    or np.any(travel < 0)):
                raise ValueError("joint safety travel must be a finite nonnegative tangent vector")
            pair_reserve = self._pair_motion_coefficients @ travel
            thresholds = thresholds + pair_reserve
        thresholds = thresholds + uncertainty
        if self._active_pair_indices:
            thresholds = thresholds.copy()
            thresholds[np.fromiter(self._active_pair_indices, dtype=int)] += (
                self._hysteresis_m
            )

        # Reuse a previous all-clear result only while a conservative bound on
        # actual proxy travel cannot consume its clearance.  Unlike periodic
        # collision checking, this has no unchecked time window: large motion
        # immediately falls through to the full broad phase.
        cached_lower_bounds: np.ndarray | None = None
        if self._last_configuration_q is not None:
            tangent_delta = np.abs(
                pin.difference(
                    self._model,
                    self._last_configuration_q,
                    configuration.q,
                )
            )
            pair_travel = self._pair_motion_coefficients @ tangent_delta
            cached_lower_bounds = np.maximum(
                self._last_lower_bounds - pair_travel,
                0.0,
            )
        if self.runtime == "native_fused_v2":
            barriers = self._update_native_v2(configuration, thresholds, cached_lower_bounds)
            return self._with_position_uncertainty(barriers, uncertainty)
        if cached_lower_bounds is not None and np.all(cached_lower_bounds > thresholds):
            self._last_configuration_q = configuration.q.copy()
            self._active_pair_indices = set()
            self._last_lower_bounds = cached_lower_bounds
            self._last_distances = {}
            self._last_barrier = None
            self._current_segments = {}
            self._broadphase_diagnostics = BroadPhaseDiagnostics(
                logical_pair_count=len(self._pairs),
                sphere_candidate_count=0,
                aabb_candidate_count=0,
                active_pair_count=0,
                exact_distance_calls=0,
                primitive_pair_count=self._broadphase_diagnostics.primitive_pair_count,
                configured_pair_count=self._configured_pair_count,
                certified_inactive_pair_count=self._certified_inactive_pair_count,
            )
            return ()

        parent_placements = [
            configuration.data.oMi[int(joint_id)]
            for joint_id in self._unique_link_parent_joints
        ]
        parent_rotations = np.stack(
            [np.asarray(placement.rotation) for placement in parent_placements]
        )
        parent_translations = np.stack(
            [np.asarray(placement.translation) for placement in parent_placements]
        )
        self._current_link_parent_rotations = parent_rotations[
            self._link_parent_inverse
        ]
        self._current_link_parent_translations = parent_translations[
            self._link_parent_inverse
        ]
        self._current_link_rotations = None
        self._current_link_translations = None

        world_link_centers = (
            np.einsum(
                "nij,nj->ni",
                self._current_link_parent_rotations,
                self._joint_local_link_centers,
            )
            + self._current_link_parent_translations
        )
        center_deltas = (
            world_link_centers[self._pair_first_link_indices]
            - world_link_centers[self._pair_second_link_indices]
        )
        lower_bounds = np.maximum(
            np.linalg.norm(center_deltas, axis=1)
            - self._link_radii[self._pair_first_link_indices]
            - self._link_radii[self._pair_second_link_indices],
            0.0,
        )
        if cached_lower_bounds is not None:
            # Every stored value is a conservative lower bound. Distance is
            # one-Lipschitz in the accumulated relative proxy travel, so this
            # per-pair certificate remains valid even while other pairs are
            # active and is safely combined with the fresh sphere bound.
            lower_bounds = np.maximum(lower_bounds, cached_lower_bounds)
        sphere_candidates = np.flatnonzero(lower_bounds <= thresholds)

        # A sphere is the cheapest conservative bound. Tighten all sphere hits
        # in one vectorized union-AABB pass before inspecting primitive unions.
        sphere_first = self._pair_first_link_indices[sphere_candidates]
        sphere_second = self._pair_second_link_indices[sphere_candidates]
        sphere_link_indices = np.unique(
            np.concatenate((sphere_first, sphere_second))
        )
        sphere_link_rotations = np.einsum(
            "nij,njk->nik",
            self._current_link_parent_rotations[sphere_link_indices],
            self._link_frame_rotations_in_joint[sphere_link_indices],
        )
        sphere_link_extents = np.einsum(
            "nij,nj->ni",
            np.abs(sphere_link_rotations),
            self._local_link_extents[sphere_link_indices],
        )
        world_link_lower = np.empty_like(world_link_centers)
        world_link_upper = np.empty_like(world_link_centers)
        world_link_lower[sphere_link_indices] = (
            world_link_centers[sphere_link_indices] - sphere_link_extents
        )
        world_link_upper[sphere_link_indices] = (
            world_link_centers[sphere_link_indices] + sphere_link_extents
        )
        union_separation = np.maximum(
            np.maximum(
                world_link_lower[sphere_second]
                - world_link_upper[sphere_first],
                world_link_lower[sphere_first]
                - world_link_upper[sphere_second],
            ),
            0.0,
        )
        lower_bounds[sphere_candidates] = np.maximum(
            lower_bounds[sphere_candidates],
            np.linalg.norm(union_separation, axis=1),
        )
        aabb_candidates = sphere_candidates[
            lower_bounds[sphere_candidates] <= thresholds[sphere_candidates]
        ]
        if aabb_candidates.size == 0:
            self._last_configuration_q = configuration.q.copy()
            self._active_pair_indices = set()
            self._last_distances = {}
            self._last_lower_bounds = lower_bounds
            self._last_barrier = None
            self._current_segments = {}
            self._broadphase_diagnostics = BroadPhaseDiagnostics(
                logical_pair_count=len(self._pairs),
                sphere_candidate_count=len(sphere_candidates),
                aabb_candidate_count=0,
                active_pair_count=0,
                exact_distance_calls=0,
                primitive_pair_count=self._broadphase_diagnostics.primitive_pair_count,
                configured_pair_count=self._configured_pair_count,
                certified_inactive_pair_count=self._certified_inactive_pair_count,
            )
            return ()

        # Tighten remaining hits against all primitive AABBs in one flattened
        # pass. Primitive combinations are contiguous by logical pair, so a
        # reduce-at returns the same minimum as the previous per-pair loop.
        world_primitive_lower: np.ndarray | None = None
        world_primitive_upper: np.ndarray | None = None
        active_candidates: list[int] = []
        if aabb_candidates.size:
            candidate_pair_mask = np.zeros(len(self._pairs), dtype=bool)
            candidate_pair_mask[aabb_candidates] = True
            combo_mask = candidate_pair_mask[self._primitive_combo_pair_indices]
            combo_pair_indices = self._primitive_combo_pair_indices[combo_mask]
            combo_first = self._primitive_combo_first_indices[combo_mask]
            combo_second = self._primitive_combo_second_indices[combo_mask]
            primitive_indices = np.unique(
                np.concatenate((combo_first, combo_second))
            )
            link_indices = self._primitive_link_indices[primitive_indices]
            primitive_rotations = np.einsum(
                "nij,njk->nik",
                self._current_link_parent_rotations[link_indices],
                self._link_frame_rotations_in_joint[link_indices],
            )
            primitive_translations = (
                np.einsum(
                    "nij,nj->ni",
                    self._current_link_parent_rotations[link_indices],
                    self._link_frame_translations_in_joint[link_indices],
                )
                + self._current_link_parent_translations[link_indices]
            )
            world_primitive_centers = np.empty((len(self._records), 3), dtype=float)
            world_primitive_extents = np.empty((len(self._records), 3), dtype=float)
            world_primitive_centers[primitive_indices] = (
                np.einsum(
                    "nij,nj->ni",
                    primitive_rotations,
                    self._local_primitive_centers[primitive_indices],
                )
                + primitive_translations
            )
            world_primitive_extents[primitive_indices] = np.einsum(
                "nij,nj->ni",
                np.abs(primitive_rotations),
                self._local_primitive_extents[primitive_indices],
            )
            world_primitive_lower = np.empty_like(world_primitive_centers)
            world_primitive_upper = np.empty_like(world_primitive_centers)
            world_primitive_lower[primitive_indices] = (
                world_primitive_centers[primitive_indices]
                - world_primitive_extents[primitive_indices]
            )
            world_primitive_upper[primitive_indices] = (
                world_primitive_centers[primitive_indices]
                + world_primitive_extents[primitive_indices]
            )
            separation = np.maximum(
                np.maximum(
                    world_primitive_lower[combo_second]
                    - world_primitive_upper[combo_first],
                    world_primitive_lower[combo_first]
                    - world_primitive_upper[combo_second],
                ),
                0.0,
            )
            combo_lower_bounds = np.linalg.norm(separation, axis=1)
            pair_starts = np.flatnonzero(
                np.concatenate(
                    (
                        np.array((True,)),
                        combo_pair_indices[1:] != combo_pair_indices[:-1],
                    )
                )
            )
            candidate_pair_indices = combo_pair_indices[pair_starts]
            primitive_lower_bounds = np.minimum.reduceat(
                combo_lower_bounds,
                pair_starts,
            )
            lower_bounds[candidate_pair_indices] = np.maximum(
                lower_bounds[candidate_pair_indices],
                primitive_lower_bounds,
            )
            if not np.all(np.isfinite(lower_bounds[candidate_pair_indices])):
                raise RuntimeError("self-collision broad-phase lower bound is non-finite")
            active_candidates = [
                int(pair_index)
                for pair_index in candidate_pair_indices[
                    lower_bounds[candidate_pair_indices]
                    <= thresholds[candidate_pair_indices]
                ]
            ]
        next_active = set(active_candidates)
        broadphase_ms = 1000.0 * (time.perf_counter() - broadphase_started)
        analytic_candidates = [
            pair_index
            for pair_index in active_candidates
            if self._analytic_pair_mask[pair_index]
        ]
        nonanalytic_candidates = [
            pair_index
            for pair_index in active_candidates
            if not self._analytic_pair_mask[pair_index]
        ]
        needs_primitive_bounds = bool(nonanalytic_candidates)
        native_started = time.perf_counter()
        if needs_primitive_bounds:
            if world_primitive_lower is None or world_primitive_upper is None:
                raise RuntimeError("active ellipsoid pairs have no primitive bounds")
            if self._native_fcl_kernel is None or not self._native_distance_indices:
                pin.updateGeometryPlacements(
                    configuration.model,
                    configuration.data,
                    self.geometry_model,
                    self.geometry_data,
                )
            if self._native_distance_indices:
                native_distances, native_calls = self._batch_native_distances(
                    nonanalytic_candidates
                )
                primitive_bounds = None
            else:
                primitive_indices = sorted(
                    {
                        geometry_index
                        for pair_index in nonanalytic_candidates
                        for geometry_index in (
                            *self._pairs[pair_index].first_indices,
                            *self._pairs[pair_index].second_indices,
                        )
                    }
                )
                primitive_bounds = {
                    geometry_index: (
                        world_primitive_lower[geometry_index],
                        world_primitive_upper[geometry_index],
                    )
                    for geometry_index in primitive_indices
                }
                native_distances = {}
                native_calls = 0
        else:
            primitive_bounds = None
            native_distances = {}
            native_calls = 0
        native_distance_ms = 1000.0 * (time.perf_counter() - native_started)
        self._current_segments = {}
        analytic_started = time.perf_counter()
        next_distances, exact_calls = self._batch_analytic_sphere_capsule_distances(
            analytic_candidates
        )
        analytic_distance_ms = 1000.0 * (time.perf_counter() - analytic_started)
        next_distances.update(native_distances)
        exact_calls += native_calls
        if not self._native_distance_indices:
            for pair_index in nonanalytic_candidates:
                exact, calls = self._exact_distance(
                    self._pairs[pair_index], primitive_bounds
                )
                exact_calls += calls
                next_distances[pair_index] = exact
        assembly_started = time.perf_counter()
        if next_distances:
            exact_pair_indices = np.fromiter(next_distances, dtype=int)
            exact_pair_distances = np.fromiter(
                (distance.distance_m for distance in next_distances.values()),
                dtype=float,
            )
            lower_bounds[exact_pair_indices] = np.maximum(
                lower_bounds[exact_pair_indices],
                exact_pair_distances,
            )
        barrier_indices = tuple(
            pair_index
            for pair_index, distance in next_distances.items()
            if distance.distance_m <= thresholds[pair_index]
        )
        next_active = set(barrier_indices)
        self._active_pair_indices = next_active
        self._last_configuration_q = configuration.q.copy()
        self._last_distances = next_distances
        self._last_lower_bounds = lower_bounds
        activation_only = self.collision_envelope is not None and self.collision_envelope.activation_only
        active_distances = tuple(
            next_distances[index] if activation_only else replace(
                next_distances[index],
                pair=replace(
                    next_distances[index].pair,
                    minimum_distance_m=next_distances[index].pair.minimum_distance_m + pair_reserve[index],
                ),
            )
            for index in sorted(barrier_indices)
        )
        self._last_barrier = (
            None
            if not active_distances
            else ActivePairCollisionBarrier(
                active_distances,
                self.geometry_model,
                self._primitive_parent_joints,
            )
        )
        self._broadphase_diagnostics = BroadPhaseDiagnostics(
            logical_pair_count=len(self._pairs),
            sphere_candidate_count=len(sphere_candidates),
            aabb_candidate_count=len(aabb_candidates),
            active_pair_count=len(active_distances),
            exact_distance_calls=exact_calls,
            primitive_pair_count=self._broadphase_diagnostics.primitive_pair_count,
            configured_pair_count=self._configured_pair_count,
            certified_inactive_pair_count=self._certified_inactive_pair_count,
            broadphase_ms=broadphase_ms,
            analytic_distance_ms=analytic_distance_ms,
            native_distance_ms=native_distance_ms,
            barrier_assembly_ms=max(
                0.0,
                1000.0 * (time.perf_counter() - assembly_started),
            ),
        )
        barriers = () if self._last_barrier is None else (self._last_barrier,)
        return self._with_position_uncertainty(barriers, uncertainty)

    def _with_position_uncertainty(self, barriers, uncertainty):
        if not np.any(uncertainty):
            return barriers
        lookup = {(p.first, p.second): uncertainty[i] for i, p in enumerate(self._pairs)}
        for barrier in barriers:
            reserve = np.array([lookup[(p.first, p.second)] for p in barrier._logical_pairs])
            barrier._residuals -= reserve
            barrier._logical_pairs = tuple(
                replace(p, minimum_distance_m=p.minimum_distance_m + r)
                for p, r in zip(barrier._logical_pairs, reserve, strict=True))
            barrier._distance_objects = None
        return barriers

    def clearance_breakdown(self, first: str, second: str) -> dict[str, object]:
        """Explain static minimum versus activation, outside the control hot path."""
        key = (first, second) if (first, second) in self._clearance_components else (second, first)
        return dict(self._clearance_components[key])

    def clearance_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Owned distance/minimum/exact arrays in configured logical-pair order.

        Avoid constructing hundreds of diagnostic dataclasses in a realtime
        consumer. Minimums are static; callers account for position uncertainty
        separately, just as with diagnostics(all_pairs=True).
        """
        if self.runtime == "native_fused_v2":
            distances = self._last_lower_bounds.copy()
            exact = np.zeros(len(self._pairs), dtype=bool)
            indices = self._native_exact_pair_indices
            distances[indices] = self._native_exact_distances
            exact[indices] = True
            return distances, self._pair_minimum_distances.copy(), exact
        pairs = self.diagnostics(all_pairs=True)
        if [(p.first, p.second) for p in pairs] != [(p.first, p.second) for p in self._pairs]:
            raise ValueError("collision diagnostic ordering changed")
        return (np.array([p.distance_m for p in pairs]), np.array([p.minimum_distance_m for p in pairs]),
                np.array([p.exact for p in pairs], dtype=bool))

    def diagnostics(
        self,
        *,
        all_pairs: bool = False,
        max_pairs: int | None = None,
    ) -> tuple[CollisionPairDiagnostic, ...]:
        if max_pairs is not None and max_pairs < 1:
            raise ValueError("maximum collision diagnostics must be positive")
        if self.runtime == "native_fused_v2":
            exact_by_pair = np.full(len(self._pairs), np.nan)
            if len(self._native_exact_pair_indices):
                exact_by_pair[self._native_exact_pair_indices] = self._native_exact_distances
            residuals = np.where(
                np.isfinite(exact_by_pair),
                exact_by_pair,
                self._last_lower_bounds,
            ) - self._pair_minimum_distances
            if all_pairs:
                reported_indices = np.arange(len(self._pairs), dtype=int)
            else:
                exact_indices = self._native_exact_pair_indices
                inactive = np.flatnonzero(~np.isfinite(exact_by_pair))
                closest = np.empty(0, dtype=int)
                if len(inactive):
                    closest = inactive[[int(np.argmin(residuals[inactive]))]]
                reported_indices = np.unique(np.concatenate((exact_indices, closest)))
            if max_pairs is not None and len(reported_indices) > max_pairs:
                order = np.argsort(residuals[reported_indices], kind="stable")[:max_pairs]
                reported_indices = reported_indices[order]
            return tuple(
                CollisionPairDiagnostic(
                    self._pairs[int(index)].first,
                    self._pairs[int(index)].second,
                    float(
                        exact_by_pair[int(index)]
                        if np.isfinite(exact_by_pair[int(index)])
                        else self._last_lower_bounds[int(index)]
                    ),
                    self._pairs[int(index)].minimum_distance_m,
                    exact=bool(np.isfinite(exact_by_pair[int(index)])),
                    active=bool(np.isfinite(exact_by_pair[int(index)])),
                )
                for index in reported_indices
            )
        diagnostics: list[CollisionPairDiagnostic] = []
        closest_inactive: int | None = None
        if len(self._last_distances) < len(self._pairs):
            inactive_residuals = (
                self._last_lower_bounds - self._pair_minimum_distances
            ).copy()
            if self._last_distances:
                inactive_residuals[
                    np.fromiter(self._last_distances, dtype=int)
                ] = np.inf
            closest_inactive = int(np.argmin(inactive_residuals))
        reported_indices = (
            list(range(len(self._pairs)))
            if all_pairs
            else sorted(
                (*self._last_distances.keys(),)
                + (() if closest_inactive is None else (closest_inactive,))
            )
        )
        if max_pairs is not None and len(reported_indices) > max_pairs:
            reported_indices = sorted(
                reported_indices,
                key=lambda index: (
                    self._last_distances[index].distance_m
                    if index in self._last_distances
                    else self._last_lower_bounds[index]
                )
                - self._pair_minimum_distances[index],
            )[:max_pairs]
        for pair_index in reported_indices:
            pair = self._pairs[pair_index]
            exact = self._last_distances.get(pair_index)
            if exact is None:
                diagnostics.append(
                    CollisionPairDiagnostic(
                        pair.first,
                        pair.second,
                        float(self._last_lower_bounds[pair_index]),
                        pair.minimum_distance_m,
                        exact=False,
                        active=False,
                    )
                )
            else:
                diagnostics.append(
                    CollisionPairDiagnostic(
                        pair.first,
                        pair.second,
                        exact.distance_m,
                        pair.minimum_distance_m,
                    )
                )
        return tuple(diagnostics)


class MuJoCoSelfCollisionOracle:
    """Read detailed CAD distances independently of proxy barriers."""

    def __init__(
        self,
        model: mujoco.MjModel,
        document: Mapping[str, object],
        *,
        distance_max_m: float = 1.0,
    ) -> None:
        import mujoco

        if not np.isfinite(distance_max_m) or distance_max_m <= 0.0:
            raise ValueError("MuJoCo oracle distance maximum must be finite and positive")
        links = document.get("links")
        pairs = document.get("pairs")
        if not isinstance(links, list) or not isinstance(pairs, list):
            raise ValueError("MuJoCo oracle requires self-collision links and pairs")

        def configured_bodies(link: Mapping[str, object]) -> list[str]:
            bodies = link.get("mujoco_bodies")
            if not isinstance(bodies, list):
                bodies = [link.get("mujoco_body", link.get("frame", ""))]
            return [str(body) for body in bodies if str(body)]

        body_ids_by_side: dict[str, set[int]] = {"left": set(), "right": set()}
        for link in links:
            if not isinstance(link, Mapping):
                raise ValueError("MuJoCo oracle link must be a mapping")
            link_name = str(link.get("name", ""))
            side = next(
                (candidate for candidate in ("left", "right") if link_name.startswith(f"{candidate}_")),
                None,
            )
            if side is None:
                continue
            for body_name in configured_bodies(link):
                body_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name,
                )
                if body_id >= 0:
                    body_ids_by_side[side].add(body_id)

        def mesh_path(mesh_id: int) -> str:
            address = int(model.mesh_pathadr[mesh_id])
            if address < 0:
                return ""
            end = model.paths.find(b"\0", address)
            if end < 0:
                end = len(model.paths)
            return model.paths[address:end].decode(errors="surrogateescape").replace("\\", "/")

        mesh_paths = tuple(mesh_path(mesh_id) for mesh_id in range(model.nmesh))
        mesh_geoms: dict[int, list[int]] = {}
        for geom_id in range(model.ngeom):
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_geoms.setdefault(int(model.geom_dataid[geom_id]), []).append(geom_id)

        def source_geometries(link: Mapping[str, object]) -> set[int]:
            sources = link.get("sources")
            if not isinstance(sources, list):
                return set()
            configured_paths = [
                str(source.get("path", "")).replace("\\", "/")
                for source in sources
                if isinstance(source, Mapping) and source.get("path")
            ]
            matched_meshes = {
                mesh_id
                for mesh_id, path in enumerate(mesh_paths)
                if any(path.endswith(configured_path) for configured_path in configured_paths)
            }
            if not matched_meshes:
                configured_names = {Path(path).name for path in configured_paths}
                matched_meshes = {
                    mesh_id
                    for mesh_id, path in enumerate(mesh_paths)
                    if Path(path).name in configured_names
                }
            return {
                geom_id
                for mesh_id in matched_meshes
                for geom_id in mesh_geoms.get(mesh_id, ())
            }

        def is_ancestor(ancestor: int, descendant: int) -> bool:
            current = int(descendant)
            while current > 0:
                if current == ancestor:
                    return True
                current = int(model.body_parentid[current])
            return ancestor == 0

        def same_kinematic_branch(body_id: int, anchors: set[int]) -> bool:
            return any(
                is_ancestor(body_id, anchor) or is_ancestor(anchor, body_id)
                for anchor in anchors
            )

        geom_indices_by_link: dict[str, tuple[int, ...]] = {}
        for link in links:
            if not isinstance(link, Mapping):
                raise ValueError("MuJoCo oracle link must be a mapping")
            link_name = str(link.get("name", ""))
            body_ids: list[int] = []
            missing_bodies: list[str] = []
            for body_name in configured_bodies(link):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
                if body_id < 0:
                    missing_bodies.append(body_name)
                else:
                    body_ids.append(body_id)
            body_geometries = {
                int(geom_id)
                for geom_id, geom_body_id in enumerate(model.geom_bodyid)
                if int(geom_body_id) in body_ids
            }
            source_matches = source_geometries(link)
            if source_matches and body_geometries:
                geom_indices = source_matches & body_geometries
            elif source_matches:
                geom_indices = source_matches
            else:
                geom_indices = body_geometries
            side = next(
                (candidate for candidate in ("left", "right") if link_name.startswith(f"{candidate}_")),
                None,
            )
            if side is not None and len(geom_indices) > 1:
                branch_matches = {
                    geom_id
                    for geom_id in geom_indices
                    if same_kinematic_branch(
                        int(model.geom_bodyid[geom_id]),
                        body_ids_by_side[side],
                    )
                }
                if branch_matches:
                    geom_indices = branch_matches
            if not geom_indices:
                detail = ""
                if missing_bodies:
                    detail = f"; missing bodies: {missing_bodies}"
                raise ValueError(
                    f"MuJoCo oracle link {link_name!r} has no exact mesh geoms{detail}"
                )
            geom_indices_by_link[link_name] = tuple(sorted(geom_indices))

        resolved: list[tuple[str, str, float, tuple[tuple[int, int], ...]]] = []
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise ValueError("MuJoCo oracle pair must be a mapping")
            first = str(pair.get("first", ""))
            second = str(pair.get("second", ""))
            if first not in geom_indices_by_link or second not in geom_indices_by_link:
                raise ValueError(f"MuJoCo oracle pair references an unknown link: {pair!r}")
            geom_pairs = tuple(
                (first_geom, second_geom)
                for first_geom in geom_indices_by_link[first]
                for second_geom in geom_indices_by_link[second]
            )
            resolved.append((first, second, float(pair["minimum_distance_m"]), geom_pairs))
        self.model = model
        self.distance_max_m = float(distance_max_m)
        self.geom_indices_by_link = dict(geom_indices_by_link)
        self._pairs = tuple(resolved)
        self._convex_meshes: dict[int, tuple[np.ndarray, object]] = {}

    @classmethod
    def from_config(
        cls,
        model: mujoco.MjModel,
        config_path: str | Path = DEFAULT_SELF_COLLISION_CONFIG,
        *,
        require_approved: bool = True,
        distance_max_m: float = 1.0,
    ) -> MuJoCoSelfCollisionOracle:
        _path, document = load_self_collision_document(
            config_path, require_approved=require_approved
        )
        return cls(model, document, distance_max_m=distance_max_m)

    def motion_closing_bounds(
        self, initial: mujoco.MjData, current: mujoco.MjData
    ) -> dict[tuple[str, str], float]:
        """Bound pair closing by maximum rigid-mesh point displacement.

        Both data objects must have up-to-date kinematics for this model.
        Bounding spheres enclose the local mesh AABBs. The sum of the two
        bodies' point-travel bounds bounds any reduction in their separation.
        Unlike differencing distance queries, this does not mix a narrow-phase
        distance with an AABB lower-bound fallback.
        """
        travel: dict[int, float] = {}
        result = {}
        for first, second, _minimum, geom_pairs in self._pairs:
            for pair in geom_pairs:
                for geom in pair:
                    if geom in travel:
                        continue
                    initial_rotation = initial.geom_xmat[geom].reshape(3, 3)
                    current_rotation = current.geom_xmat[geom].reshape(3, 3)
                    local_aabb = self.model.geom_aabb[geom]
                    center, extent = local_aabb[:3], local_aabb[3:]
                    initial_center = initial.geom_xpos[geom] + initial_rotation @ center
                    current_center = current.geom_xpos[geom] + current_rotation @ center
                    cosine = np.clip(
                        (np.trace(initial_rotation.T @ current_rotation) - 1.0) / 2.0, -1.0, 1.0
                    )
                    rotation_travel = np.linalg.norm(extent) * np.sqrt(2.0 * (1.0 - cosine))
                    travel[geom] = float(
                        np.linalg.norm(current_center - initial_center) + rotation_travel
                    )
            result[(first, second)] = max(travel[a] + travel[b] for a, b in geom_pairs)
        return result


    def _mesh_separation_certificate(self, data, first_geom: int, second_geom: int) -> float | None:
        """Certify separation using all mesh vertices, not a second solver's verdict.

        A convex query only proposes an axis. Positive separation is accepted
        only after projecting every original MuJoCo mesh vertex onto that axis.
        This oracle-only fallback never supplies geometry to the controller.
        """
        import mujoco

        if any(self.model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH
               for index in (first_geom, second_geom)):
            return None
        meshes, placements = [], []
        for geom in (first_geom, second_geom):
            mesh = int(self.model.geom_dataid[geom])
            if mesh not in self._convex_meshes:
                start = int(self.model.mesh_vertadr[mesh])
                count = int(self.model.mesh_vertnum[mesh])
                vertices = np.array(self.model.mesh_vert[start:start + count], dtype=float, copy=True)
                points = hppfcl.StdVec_Vec3f()
                for point in vertices:
                    points.append(point)
                self._convex_meshes[mesh] = (vertices, hppfcl.Convex.convexHull(points, True, "Qt"))
            meshes.append(self._convex_meshes[mesh])
            placements.append(hppfcl.Transform3f(
                np.asarray(data.geom_xmat[geom]).reshape(3, 3), data.geom_xpos[geom],
            ))
        result = hppfcl.DistanceResult()
        distance = float(hppfcl.distance(
            meshes[0][1], placements[0], meshes[1][1], placements[1],
            hppfcl.DistanceRequest(True), result,
        ))
        if not np.isfinite(distance) or distance <= 1e-9:
            return None
        axis = np.asarray(result.getNearestPoint2()) - np.asarray(result.getNearestPoint1())
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        axis = axis / norm
        projections = [
            vertices @ (np.asarray(data.geom_xmat[geom]).reshape(3, 3).T @ axis)
            + float(data.geom_xpos[geom] @ axis)
            for geom, (vertices, _shape) in zip((first_geom, second_geom), meshes, strict=True)
        ]
        gap = float(np.min(projections[1]) - np.max(projections[0]))
        return gap if np.isfinite(gap) and gap > 1e-9 else None

    def sample(self, data: mujoco.MjData) -> tuple[OraclePairDiagnostic, ...]:
        import mujoco

        diagnostics: list[OraclePairDiagnostic] = []

        def geom_aabb(geom_id: int) -> tuple[np.ndarray, np.ndarray]:
            rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            local = np.asarray(self.model.geom_aabb[geom_id])
            center = data.geom_xpos[geom_id] + rotation @ local[:3]
            extent = np.abs(rotation) @ local[3:]
            return center - extent, center + extent

        aabbs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        bounded_types = {
            mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH,
        }
        for first, second, minimum, geom_pairs in self._pairs:
            distances: list[float] = []
            exact = True
            for first_geom, second_geom in geom_pairs:
                for geom in (first_geom, second_geom):
                    if geom not in aabbs:
                        aabbs[geom] = geom_aabb(geom)
                aabb_gap = _aabb_distance(aabbs[first_geom], aabbs[second_geom])
                if (aabb_gap > 1e-9
                        and self.model.geom_type[first_geom] in bounded_types
                        and self.model.geom_type[second_geom] in bounded_types):
                    # An independent separating-box certificate. Report its
                    # conservative lower bound, not a purported exact CAD gap.
                    # This also avoids unreliable signed narrow-phase results
                    # for demonstrably separated mesh pairs.
                    distances.append(aabb_gap)
                    exact = False
                    continue
                distance = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        data,
                        first_geom,
                        second_geom,
                        self.distance_max_m,
                        None,
                    )
                )
                if distance <= 0.0:
                    certificate = self._mesh_separation_certificate(data, first_geom, second_geom)
                    if certificate is not None:
                        distance = certificate
                        exact = False
                distances.append(distance)
            distance = min(distances)
            diagnostics.append(OraclePairDiagnostic(first, second, distance, minimum, exact))
        return tuple(diagnostics)
