"""Export failure evidence after gateway cleanup, without masking the first fault."""
import json
import numpy as np


def save_failure_evidence(writer, report, core, failed_cycle=None):
    from .ik_runner import jsonable

    snapshot = getattr(core, "last_qp_failure", None)
    capture_error = getattr(core, "last_qp_capture_error", None)
    if not report.get("error") and not (
            not report.get("passed") and (failed_cycle or snapshot is not None or capture_error)):
        return
    errors = []
    if failed_cycle is not None:
        try:
            path = writer.directory / "failed_cycle.json"
            path.write_text(json.dumps(jsonable(failed_cycle), allow_nan=False, indent=2) + "\n")
            report["failed_cycle_file"] = path.name
        except Exception as error:
            errors.append(f"failed_cycle: {type(error).__name__}: {error}")
    if snapshot is not None:
        try:
            path = writer.directory / "failed_qp.npz"
            np.savez_compressed(path, **snapshot.arrays)
            report["failed_qp_file"] = path.name
        except Exception as error:
            errors.append(f"failed_qp: {type(error).__name__}: {error}")
    if capture_error:
        report["qp_capture_error"] = capture_error
    if errors:
        report["failure_evidence_error"] = "; ".join(errors)
