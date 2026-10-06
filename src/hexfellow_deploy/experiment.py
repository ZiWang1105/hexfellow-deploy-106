"""Finalization of explicitly time-bounded validation experiments."""
from dataclasses import dataclass


@dataclass
class FinalHold:
    duration_s: float
    convergence_timeout_s: float = 15.
    requested_at: float | None = None
    stationary_since: float | None = None

    def update(self, now, status):
        if self.requested_at is None:
            self.requested_at = now
        status = status or {}
        diagnostic = status.get('diagnostic', {})
        acknowledged = bool(status.get('final_hold'))
        stationary = acknowledged and diagnostic.get('ik_status') == 'HOLD' and diagnostic.get('base_stopped', False)
        if stationary:
            if self.stationary_since is None:
                self.stationary_since = now
        else:
            self.stationary_since = None
        if self.stationary_since is not None and now-self.stationary_since >= self.duration_s:
            return 'completed_budget'
        if now-self.requested_at > self.convergence_timeout_s+self.duration_s:
            return 'final_hold_failed'
        return 'waiting' if acknowledged else 'request'
