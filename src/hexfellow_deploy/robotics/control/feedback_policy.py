"""Per-device receive-time policy. No synthetic timestamps or pose prediction."""

import numpy as np

THETA_POLICY = "theta-80ms-first-trial"
THETA_RECORD_ONLY = "theta-record-only"
THETA_BOUNDED = "theta-bounded"
FEEDBACK_POLICIES = ("strict", THETA_POLICY, THETA_RECORD_ONLY, THETA_BOUNDED)
GROUPS = {"base": slice(0, 3), "theta": slice(3, 4), "left": slice(4, 11), "right": slice(11, 18)}


def coupled_groups(policy):
    if policy not in FEEDBACK_POLICIES:
        raise ValueError("invalid feedback policy")
    return tuple(GROUPS) if policy == "strict" else ("base", "left", "right")


def received_deadline(received, policy):
    """Deadline of the original measurements, never renewed by later feedback."""
    deadline = min(received[n] + .04 for n in coupled_groups(policy))
    if policy in {THETA_POLICY, THETA_BOUNDED}:
        deadline = min(deadline, received["theta"] + .08)
    return deadline


class FeedbackSkewError(ValueError):
    """Age and clock checks passed, but the group is temporarily incoherent."""


class FeedbackAgeError(ValueError):
    """Valid clocks but feedback is too old for new movement."""


def validate_received(received, now, policy="strict", *, check_skew=True, check_age=True):
    if policy not in FEEDBACK_POLICIES or set(received) != set(GROUPS):
        raise ValueError("invalid feedback policy or incomplete device timestamps")
    stamps = np.array([received[n] for n in GROUPS], dtype=float)
    if not np.isfinite(now) or not np.all(np.isfinite(stamps)):
        raise ValueError("invalid feedback clock")
    ages = now - stamps
    limits = np.array([.04, .08 if policy in {THETA_POLICY, THETA_BOUNDED} else .04, .04, .04])
    if policy == THETA_RECORD_ONLY:
        limits[1] = np.inf
    if np.any(ages < -1e-12):
        raise ValueError("feedback clock is in the future")
    if check_age and np.any(ages > limits + 1e-12):
        raise FeedbackAgeError(f"stale robot feedback, ages_ms={dict(zip(GROUPS, np.round(ages * 1000, 2), strict=True))}")
    coupled = stamps if policy == "strict" else stamps[[0, 2, 3]]
    if check_skew and np.ptp(coupled) > .02 + 1e-12:
        raise FeedbackSkewError("robot feedback skew exceeds 20 ms")


def theta_uncertainty(received, now, policy=THETA_POLICY, *, lift_speed=None):
    if policy == THETA_RECORD_ONLY:
        return 0.0  # Age is diagnostic only in this explicitly selected policy.
    age = now - received["theta"]
    if policy == THETA_BOUNDED:
        if lift_speed is None or not np.isfinite(lift_speed) or lift_speed <= 0:
            raise ValueError("bounded Theta uncertainty requires an effective lift speed")
        if not np.isfinite(age) or age < -1e-12:
            raise ValueError("invalid Theta feedback age")
        return .002 + lift_speed * max(0., age)
    # Only temporary: regular <=20ms delivery has no extra static clearance.
    return .002 + .02 * age if age > .02 else 0.0


def validate_group_progress(previous_received, received, old_q, q, old_v, v, *, lift_speed=.02, theta_dt_s=None, base_dofs=3):
    if base_dofs not in (0, 3):
        raise ValueError("feedback progress supports fixed or planar models")
    if previous_received is None:
        return
    for name, part in GROUPS.items():
        if base_dofs == 0:
            if name == "base":
                if received[name] < previous_received[name]:
                    raise ValueError("base feedback timestamp regressed")
                continue
            part = slice(part.start - 3, part.stop - 3)
        old, new = previous_received[name], received[name]
        if new < old:
            raise ValueError(f"{name} feedback timestamp regressed")
        if new == old:
            if not np.array_equal(old_q[part], q[part]) or not np.array_equal(old_v[part], v[part]):
                raise ValueError(f"{name} repeated timestamp has changed feedback")
        elif name == "theta" and abs(float(q[base_dofs] - old_q[base_dofs])) > lift_speed * ((new - old) if theta_dt_s is None else theta_dt_s) + .002:
            raise ValueError("Theta recovery displacement exceeds first-trial model")
