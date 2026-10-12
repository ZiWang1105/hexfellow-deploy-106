"""Measured startup positioning, completed before the fixed IK reference is seeded."""
import math


def validate_target(target):
    if target is not None and (not math.isfinite(target) or not .002 <= target <= .400):
        raise ValueError('Startup lift target must be finite and within [0.002, 0.400] m')
    return target


class LiftArrival:
    """Require fresh, stationary, in-tolerance measurements over a continuous dwell."""
    def __init__(self, target, started):
        self.target = validate_target(target)
        self.started = started
        self.stable_since = None
        self.last_received = None
        self.report = dict(target_m=target, phase='positioning', started_s=started)

    def update(self, height, velocity, received, now, *, stopped=False):
        if stopped:
            raise RuntimeError('Startup lift positioning cancelled')
        if not all(math.isfinite(v) for v in (height, velocity, received, now)) or not 0 <= now-received <= .04:
            raise RuntimeError('Startup lift requires fresh finite feedback within 40 ms')
        if now-self.started >= 25.:
            raise RuntimeError('Startup lift did not settle before the 25-second deadline')
        if self.last_received is not None and received < self.last_received:
            raise RuntimeError('Startup lift feedback timestamp regressed')
        self.report.update(height_m=height, velocity_m_s=velocity, error_m=self.target-height,
                           elapsed_s=now-self.started)
        if self.last_received == received:
            return False
        self.last_received = received
        settled = abs(height-self.target) <= .001 and abs(velocity) <= .002
        self.stable_since = (received if self.stable_since is None else self.stable_since) if settled else None
        done = self.stable_since is not None and received-self.stable_since >= .2
        self.report.update(stable_s=0. if self.stable_since is None else received-self.stable_since,
                           phase='arrived' if done else 'positioning')
        return done
