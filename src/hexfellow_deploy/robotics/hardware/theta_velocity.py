"""Theta derivatives belong to raw device reports, never to polling/arrival clocks."""

import math


def device_nanoseconds(stamp):
    if not isinstance(stamp, dict):
        raise ValueError("Theta missing raw device monotonic timestamp")
    seconds, nanos = stamp.get("seconds"), stamp.get("nanoseconds")
    if (type(seconds) is not int or type(nanos) is not int
            or seconds < 0 or not 0 <= nanos < 1_000_000_000):
        raise ValueError("Theta invalid raw device monotonic timestamp")
    return seconds * 1_000_000_000 + nanos


def estimate_frame(current, previous):
    position = current["pos"]
    if not math.isfinite(position):
        raise ValueError("Theta nonfinite position")
    dt = None if previous is None else current["device_time_ns"] - previous["device_time_ns"]
    host_dt = None if previous is None else current["received"] - previous["received"]
    if dt is not None and dt <= 0:
        raise ValueError("Theta device timestamp did not advance")
    # A gap frame is the first sample of a new pair, not a valid zero velocity.
    valid = dt is not None and dt <= 40_000_000
    return {"device_dt_ns": dt, "receive_dt_s": host_dt, "velocity_valid": valid,
            "velocity": (position - previous["pos"]) * 1e9 / dt if valid else None}
