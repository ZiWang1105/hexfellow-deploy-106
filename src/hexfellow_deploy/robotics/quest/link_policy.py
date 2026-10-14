"""Explicit input-age budgets, independent of robot feedback and motor limits."""
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QuestLinkPolicy:
    name: str
    max_age_ms: float
    max_clock_uncertainty_ms: float
    include_clock_margin: bool = False
    stream_config: object = None

    def report(self):
        result = asdict(self)
        result.pop('stream_config')
        if self.stream_config is not None:
            result['stream_config'] = self.stream_config.report()
        return result

    def effective_age(self, sample, now_ns):
        age = sample.age_ms(now_ns)
        if not self.include_clock_margin:
            return age
        # A receive timestamp alone cannot establish the age of a delayed target.
        if sample.sampled_time_ms is None or sample.clock_offset_ms is None:
            return math.inf
        uncertainty = sample.clock_uncertainty_ms
        if uncertainty is None or not math.isfinite(uncertainty) or uncertainty < 0:
            return math.inf
        return now_ns / 1e6 - sample.sampled_time_ms - sample.clock_offset_ms + uncertainty

    def timing_reason(self, sample, now_ns):
        age = sample.age_ms(now_ns)
        if not math.isfinite(age) or age < 0:
            return "invalid Quest clock"
        if self.effective_age(sample, now_ns) > self.max_age_ms:
            return f"Quest pose older than {self.max_age_ms:g} ms"
        uncertainty = sample.clock_uncertainty_ms
        if (uncertainty is None or not math.isfinite(uncertainty) or uncertainty < 0
                or uncertainty > self.max_clock_uncertainty_ms):
            return f"Quest clock uncertainty exceeds {self.max_clock_uncertainty_ms:g} ms"
        return None

    def preflight_reasons(self, report):
        if not self.include_clock_margin:
            prefixes = ("interarrival_", "clock_uncertainty_", "callback_to_host_", "ik_sample_age_",
                        "frame_rate_", "delivered_", "sequence_gap_")
            return tuple(dict.fromkeys((*report.reasons, *(r for r in report.warnings if r.startswith(prefixes)))))
        reasons = list(report.reasons)
        for key, limit in (("clock_uncertainty_p95_ms", self.max_clock_uncertainty_ms),
                           ("effective_age_p95_ms", 100. if self.stream_config is None else self.stream_config.age_stale_ms),
                           ("effective_age_p99_ms", self.max_age_ms)):
            value = report.metrics.get(key)
            if value is None or not math.isfinite(value) or value > limit:
                reasons.append(f"{key}>{limit:g}")
        return tuple(dict.fromkeys(reasons))


POLICIES = {
    "strict": QuestLinkPolicy("strict", 100., 5.),
    "target-200ms": QuestLinkPolicy("target-200ms", 200., 10., True),
    "buffered": None,  # Loaded from the versioned stream configuration.
}


def link_policy(name="strict", stream_config=None):
    if name == 'buffered':
        from .stream_config import load_stream_config
        config = load_stream_config(stream_config)
        return QuestLinkPolicy(name, config.age_timeout_ms, config.clock_uncertainty_ms, True, config)
    return POLICIES[name]
