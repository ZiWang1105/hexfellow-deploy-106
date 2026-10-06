"""Cycle budgets are warnings, independent from device health decisions."""
import math
import time


class CycleBudget:
    def __init__(self, period_s, budget_s, *, clock=time.monotonic):
        if not all(math.isfinite(v) and v > 0 for v in (period_s, budget_s)):
            raise ValueError("Cycle period and budget must be positive and finite")
        if budget_s > period_s:
            raise ValueError("Compute budget must fit the configured period")
        self.period, self.budget, self.clock = period_s, budget_s, clock
        self.overruns = 0

    def finish(self, started):
        elapsed = self.clock() - started
        overrun = elapsed > self.budget
        self.overruns += int(overrun)
        return dict(elapsed_ms=1000 * elapsed, budget_overrun=overrun,
                    sleep_s=max(0., self.period - elapsed), overruns=self.overruns)
