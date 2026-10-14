"""Measured Wi-Fi/WebXR quality gates for Quest-driven IK."""

from __future__ import annotations

import dataclasses
import threading
from collections import deque

import numpy as np

from ..contract import ARM_ORDER
from .mapping import quaternion_xyzw_to_matrix, rotation_distance
from .protocol import QuestPoseSample


@dataclasses.dataclass(frozen=True)
class QuestQualityReport:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, object]
    warnings: tuple[str, ...] = ()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _pose_stability(samples: list[QuestPoseSample], side: str) -> dict[str, float | None]:
    poses = [sample.controllers[side] for sample in samples if sample.controllers[side].present]
    if len(poses) < 10:
        return {"jitter_position_p95_m": None, "jitter_rotation_p95_rad": None, "drift_m": None, "drift_rad": None}
    positions = np.stack([pose.position_m for pose in poses])
    center = np.median(positions, axis=0)
    position_jitter = np.linalg.norm(positions - center, axis=1)
    quaternions = np.stack([pose.orientation_xyzw for pose in poses])
    quaternions[quaternions @ quaternions[0] < 0.0] *= -1.0
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    mean_rotation = quaternion_xyzw_to_matrix(mean_quaternion)
    rotations = [quaternion_xyzw_to_matrix(value) for value in quaternions]
    rotation_jitter = [rotation_distance(mean_rotation, value) for value in rotations]
    window = max(3, len(poses) // 5)
    first_position = np.median(positions[:window], axis=0)
    last_position = np.median(positions[-window:], axis=0)
    first_quaternion = np.mean(quaternions[:window], axis=0)
    last_quaternion = np.mean(quaternions[-window:], axis=0)
    return {
        "jitter_position_p95_m": float(np.percentile(position_jitter, 95)),
        "jitter_rotation_p95_rad": float(np.percentile(rotation_jitter, 95)),
        "drift_m": float(np.linalg.norm(last_position - first_position)),
        "drift_rad": rotation_distance(
            quaternion_xyzw_to_matrix(first_quaternion), quaternion_xyzw_to_matrix(last_quaternion)
        ),
    }


class QuestLinkMonitor:
    """Bounded raw-link telemetry shared by the network and control threads."""

    def __init__(self, *, capacity: int = 4096) -> None:
        self._samples: deque[QuestPoseSample] = deque(maxlen=capacity)
        self._ik_ages_ms: deque[tuple[int, float]] = deque(maxlen=capacity)
        self._effective_ages_ms: deque[tuple[int, float]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._rejected = {"duplicate": 0, "out_of_order": 0}

    def record_rejection(self, reason):
        with self._lock:
            self._rejected[reason] = self._rejected.get(reason, 0) + 1

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._ik_ages_ms.clear()
            self._effective_ages_ms.clear()

    def record(self, sample: QuestPoseSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def record_ik_age(self, *, timestamp_ns: int, age_ms: float, effective_age_ms: float | None = None) -> None:
        if np.isfinite(age_ms):
            with self._lock:
                self._ik_ages_ms.append((int(timestamp_ns), float(age_ms)))
                if effective_age_ms is not None:
                    self._effective_ages_ms.append((int(timestamp_ns), float(effective_age_ms)))

    def report(self, *, window_s: float = 10.0, require_static: bool = True,
               minimum_receive_hz=None) -> QuestQualityReport:
        with self._lock:
            samples = list(self._samples)
            ages = list(self._ik_ages_ms)
            effective_ages = list(self._effective_ages_ms)
            rejected = dict(self._rejected)
        if not samples:
            return QuestQualityReport(False, ("no_pose_samples",), {"sample_count": 0})
        cutoff_ns = samples[-1].host_received_ns - round(window_s * 1e9)
        samples = [sample for sample in samples if sample.host_received_ns >= cutoff_ns]
        ages_ms = [age for timestamp, age in ages if timestamp >= cutoff_ns]
        effective_ms = [age for timestamp, age in effective_ages if timestamp >= cutoff_ns]
        effective_valid = bool(effective_ms) and bool(np.isfinite(effective_ms).all())
        callbacks = [
            max(0.0, latency)
            for sample in samples
            if (latency := sample.callback_to_host_ms) is not None and latency >= -5.0
        ]
        uncertainties = [sample.clock_uncertainty_ms for sample in samples if sample.clock_uncertainty_ms is not None]
        intervals = [
            (right.host_received_ns - left.host_received_ns) / 1e6
            for left, right in zip(samples[:-1], samples[1:], strict=True)
            if right.host_received_ns >= left.host_received_ns
        ]
        missing = 0
        for left, right in zip(samples[:-1], samples[1:], strict=True):
            difference = (right.sequence - left.sequence) & 0xFFFFFFFF
            if 1 < difference < 0x80000000:
                missing += difference - 1
        received = len(samples)
        sequence_total = received + missing
        gap_ratio = missing / sequence_total if sequence_total else 1.0
        duration_s = max(0.0, (samples[-1].host_received_ns - samples[0].host_received_ns) / 1e9)
        frame_rates = [sample.frame_rate_hz for sample in samples]
        median_frame_rate = float(np.median(frame_rates))
        expected = duration_s * median_frame_rate
        delivered_ratio = min(1.0, (received - 1) / expected) if expected > 0.0 else 0.0
        stability = {side: (_pose_stability(samples, side) if require_static else {}) for side in ARM_ORDER}
        metrics: dict[str, object] = {
            "sample_count": received,
            "window_s": duration_s,
            "frame_rate_hz_median": median_frame_rate,
            "delivered_ratio": delivered_ratio,
            "sequence_gap_ratio": gap_ratio,
            "interarrival_p99_ms": _percentile(intervals, 99),
            "clock_uncertainty_p95_ms": _percentile([float(value) for value in uncertainties], 95),
            "callback_to_host_p95_ms": _percentile(callbacks, 95),
            "callback_to_host_p99_ms": _percentile(callbacks, 99),
            "ik_sample_age_p95_ms": _percentile(ages_ms, 95),
            "ik_sample_age_p99_ms": _percentile(ages_ms, 99),
            "effective_age_p95_ms": _percentile(effective_ms, 95) if effective_valid else None,
            "effective_age_p99_ms": _percentile(effective_ms, 99) if effective_valid else None,
            "stability": stability,
            "receive_hz": (received - 1) / duration_s if duration_s > 0 else 0.,
            "source_hz": ((received - 1) * 1000 / (samples[-1].sampled_time_ms - samples[0].sampled_time_ms)
                          if samples[-1].sampled_time_ms is not None and samples[0].sampled_time_ms is not None
                          and samples[-1].sampled_time_ms > samples[0].sampled_time_ms else None),
            "interarrival_mean_ms": float(np.mean(intervals)) if intervals else None,
            "interarrival_p95_ms": _percentile(intervals, 95),
            "interarrival_max_ms": max(intervals) if intervals else None,
            "sequence_missing_or_sender_skipped": missing,
            "rejected_cumulative": rejected,
        }
        reasons: list[str] = []
        warnings: list[str] = []

        def maximum(target: list[str], key: str, limit: float) -> None:
            value = metrics[key]
            if value is None or float(value) > limit:
                target.append(f"{key}>{limit}")

        if duration_s < min(window_s * 0.9, window_s - 0.25):
            reasons.append("preflight_window_too_short")
        if minimum_receive_hz is not None and metrics['receive_hz'] < minimum_receive_hz:
            reasons.append(f"receive_hz_below_{minimum_receive_hz:g}")
        elif minimum_receive_hz is None and median_frame_rate < 40.0:
            reasons.append("frame_rate_below_40hz")
        if delivered_ratio < 0.65:
            reasons.append("delivered_ratio_below_0.65")
        if gap_ratio >= 0.10:
            reasons.append("sequence_gap_ratio_at_or_above_0.10")
        # Quest Browser can deliver a complete, gap-free WebXR stream in
        # bursts. As in the proven G1 gateway, transport cadence and actual
        # controller samples are treated separately: burst latency is visible
        # in diagnostics, but only a near-disconnection blocks this MuJoCo
        # preflight. Live control consumes the newest value and holds targets
        # during shorter gaps without extrapolating.
        maximum(reasons, "interarrival_p99_ms", 500.0)
        maximum(reasons, "clock_uncertainty_p95_ms", 50.0)
        maximum(reasons, "callback_to_host_p95_ms", 1000.0)
        maximum(reasons, "callback_to_host_p99_ms", 1500.0)
        maximum(reasons, "ik_sample_age_p95_ms", 500.0)
        maximum(reasons, "ik_sample_age_p99_ms", 750.0)

        # Preserve the laboratory-grade thresholds as diagnostics, but do not
        # make normal hand tremor or a single Wi-Fi scheduling stall prevent a
        # safe interactive session from ever starting.
        # The browser intentionally transports poses at 50 Hz to match the
        # control loop while continuing to render XR at the headset refresh
        # rate. Warn only when the advertised transport rate falls below the
        # range needed to sustain that loop.
        if median_frame_rate < 48.0:
            warnings.append("frame_rate_below_48hz")
        if delivered_ratio < 0.90:
            warnings.append("delivered_ratio_below_0.90")
        if gap_ratio >= 0.01:
            warnings.append("sequence_gap_ratio_at_or_above_0.01")
        maximum(warnings, "interarrival_p99_ms", 40.0)
        maximum(warnings, "clock_uncertainty_p95_ms", 5.0)
        maximum(warnings, "callback_to_host_p95_ms", 15.0)
        maximum(warnings, "callback_to_host_p99_ms", 30.0)
        maximum(warnings, "ik_sample_age_p95_ms", 30.0)
        maximum(warnings, "ik_sample_age_p99_ms", 50.0)
        for side in ARM_ORDER:
            values = stability[side]
            if require_static:
                for key, limit in (
                    ("jitter_position_p95_m", 0.002),
                    ("jitter_rotation_p95_rad", np.deg2rad(0.5)),
                    ("drift_m", 0.003),
                    ("drift_rad", np.deg2rad(0.5)),
                ):
                    value = values[key]
                    if value is None or float(value) > limit:
                        warnings.append(f"{side}_{key}>{limit}")
                for key, limit in (
                    ("jitter_position_p95_m", 0.030),
                    ("jitter_rotation_p95_rad", np.deg2rad(10.0)),
                    ("drift_m", 0.050),
                    ("drift_rad", np.deg2rad(15.0)),
                ):
                    value = values[key]
                    if value is None or float(value) > limit:
                        reasons.append(f"{side}_{key}>{limit}")
            invalid = [
                sample
                for sample in samples
                if not sample.controllers[side].present or sample.controllers[side].emulated_position
            ]
            if invalid:
                reasons.append(f"{side}_tracking_invalid")
        return QuestQualityReport(not reasons, tuple(reasons), metrics, tuple(warnings))
