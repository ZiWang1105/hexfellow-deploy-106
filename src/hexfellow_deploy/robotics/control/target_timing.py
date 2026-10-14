"""Causal mapped-target replay clocks shared by simulation and hardware.

Legacy reconstruction is explicitly estimated: schema 5 did not record clock
sync offsets at every control tick. No interpolation consumes future samples.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecordingTimeline:
    times: np.ndarray
    quality: str

    @classmethod
    def load(cls, data, indices, *, mode="recorded"):
        if mode not in {"recorded", "legacy_fixed_20ms"}:
            raise ValueError("unknown replay clock mode")
        count = len(data["original_tick"])
        if mode == "legacy_fixed_20ms":
            values, quality = np.arange(count) * .02, "legacy_fixed_20ms"
        elif "control_time_ns" in data:
            raw = np.asarray(data["control_time_ns"])
            if (raw.shape != (count,) or not np.issubdtype(raw.dtype, np.integer)
                    or np.any(raw[indices] < 0) or np.any(raw[indices] > np.iinfo(np.int64).max)):
                raise ValueError("recorded control clock must be finite and strictly increasing")
            # Subtract integer nanoseconds before converting: subtracting large
            # floating-point host times can spuriously delay an exact tick.
            values = (raw.astype(np.int64) - int(raw[indices[0]])).astype(float) / 1e9
            quality = "recorded_host_monotonic"
        elif "raw_sampled_time_ms" in data and "sample_age_ms" in data:
            values = (np.asarray(data["raw_sampled_time_ms"], dtype=float)
                      + np.asarray(data["sample_age_ms"], dtype=float)) / 1000
            quality = "estimated_sample_plus_age_unknown_clock_offset"
        else:
            raise ValueError("recording has no usable wall clock; select legacy_fixed_20ms explicitly")
        if (values.shape != (count,) or not np.isfinite(values[indices]).all()
                or np.any(np.diff(values[indices]) <= 0)):
            raise ValueError("recorded control clock must be finite and strictly increasing")
        selected = values[indices].copy()
        selected -= selected[0]
        selected.setflags(write=False)
        return cls(selected, quality)

    @property
    def duration(self):
        return float(self.times[-1])

    def index(self, elapsed, speed=1.):
        if not np.isfinite(elapsed) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("replay time and positive speed must be finite")
        return int(np.clip(np.searchsorted(self.times, elapsed * speed, side="right") - 1,
                           0, len(self.times) - 1))


@dataclass(frozen=True)
class TargetSample:
    """Source time is used only for causal derivatives, never pose prediction."""
    time_s: float
    sequence: int
    epoch: int = 0
    quality: str = "controller_sample"

    def __post_init__(self):
        if not np.isfinite(self.time_s):
            raise ValueError("target sample time must be finite")

    @classmethod
    def from_quest(cls, sample):
        # Relative derivatives need no clock offset. Host receipt remains the
        # fallback when a source timestamp is unavailable.
        source = sample.sampled_time_ms
        return cls(sample.host_received_ns / 1e9 if source is None else source / 1000,
                   sample.sequence, sample.reference_epoch,
                   "host_receipt" if source is None else "source_monotonic")
