"""Bounded all-cycle latency distribution; 10us bins, conservative percentiles."""
import math

PERFORMANCE_TARGETS = {"cycle_to_udp_p99_ms": 8., "source_to_udp_p99_ms": 10.}
PERFORMANCE_LIMITS = {"cycle_to_udp_p99_ms": 8., "source_to_udp_p99_ms": 15.}


def qualify_performance(stats, expected_count):
    """All-cycle coverage is required even under the operator-approved 15ms limit."""
    def meets(limits):
        return bool(stats) and all(
            values.get(metric, {}).get('count', 0) >= max(1, expected_count)
            and values[metric]['p50_p95_p99_max_ms'][2] <= limit
            for values in stats.values()
            for metric, limit in (("cycle_to_udp", limits['cycle_to_udp_p99_ms']),
                                  ("source_to_udp", limits['source_to_udp_p99_ms'])))
    accepted, target = meets(PERFORMANCE_LIMITS), meets(PERFORMANCE_TARGETS)
    return {"performance_passed": accepted, "performance_target_met": target,
            "performance_grade": "target" if target else "relaxed" if accepted else "failed",
            "performance_targets": PERFORMANCE_TARGETS, "performance_acceptance_limits": PERFORMANCE_LIMITS}


class LatencyStats:
    def __init__(self):
        self.metrics = {}

    def add(self, name, seconds):
        row = self.metrics.get(name)
        if row is None:
            row = self.metrics[name] = {'count': 0, 'max_ms': 0., 'bins': [0] * 20001}
        ms = max(0., seconds * 1000)
        row['count'] += 1
        row['max_ms'] = max(row['max_ms'], ms)
        row['bins'][min(20000, math.ceil(ms * 100))] += 1

    def report(self):
        result = {}
        for name, row in self.metrics.items():
            percentiles = []
            for fraction in (.5, .95, .99):
                target = math.ceil(row['count'] * fraction)
                count = 0
                for index, value in enumerate(row['bins']):
                    count += value
                    if count >= target:
                        percentiles.append(index / 100 if index < 20000 else row['max_ms'])
                        break
            result[name] = dict(count=row['count'], p50_p95_p99_max_ms=[*percentiles, row['max_ms']],
                                bin_width_ms=.01)
        return result
