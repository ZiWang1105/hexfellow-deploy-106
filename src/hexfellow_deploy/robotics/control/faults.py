"""Structured controller fault evidence; classification does not relax control."""

from __future__ import annotations


def describe_control_fault(error: Exception) -> tuple[str, dict]:
    name = type(error).__name__
    details = {"exception": name, "message": str(error)}
    if name == "ControlDeadlineExceeded":
        kind = "feedback_expired" if "dispatch_state_expired" in str(error) else "control_deadline"
        for key in ("stage", "elapsed_s", "limit_s", "feedback_age_s"):
            details[key] = getattr(error, key, None)
        elapsed, limit = details["elapsed_s"], details["limit_s"]
        if elapsed is not None and limit is not None:
            actual = elapsed + (details["feedback_age_s"] or 0.0) if kind == "feedback_expired" else elapsed
            details["exceeded_by_s"] = max(0.0, actual - limit)
        return kind, details
    if name == "NoSolutionFound":
        details["qp_feasibility"] = "unknown_requires_offline_check"
        return "qp_no_solution", details
    if name == "BoundaryStateUnrecoverable":
        details.update(getattr(error, "evidence", {}))
        return "boundary_state_unrecoverable", details
    if "feedback" in str(error).lower() and ("age" in str(error).lower() or "expired" in str(error).lower()):
        return "feedback_expired", details
    return "other_control_fault", details
