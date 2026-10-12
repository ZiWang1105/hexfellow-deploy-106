"""Explain saved PinkIK scalar constraint conflicts without running devices."""
import argparse
import json
from pathlib import Path

import numpy as np

from hexfellow_deploy.workspace import activate


def analyze(path):
    """Return interval contradictions in original joint velocity units."""
    from policy_sim.control.qp_diagnostics import QPFailureSnapshot

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"G", "h", "q", "row_labels", "control_dt_s", "joint_names",
                "reference_q", "measured_q", "reference_velocity",
                "measured_velocity", "reference_acceleration"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Incomplete QP failure capture: {sorted(missing)}")
    snapshot = QPFailureSnapshot(arrays)
    conflicts = []
    for conflict in snapshot.scalar_conflicts():
        index = conflict["index"]
        state = {key: float(arrays[key][index]) for key in
                 ("reference_q", "measured_q", "reference_velocity",
                  "measured_velocity", "reference_acceleration")}
        conflicts.append(dict(conflict, state=state,
            reference_minus_measured=state["reference_q"]-state["measured_q"],
            interval_gap_rad_s=conflict["minimum_velocity"]-conflict["maximum_velocity"]))
    return dict(kind="offline_QP_constraint_analysis", source=str(Path(path).resolve()),
                control_dt_s=float(arrays["control_dt_s"]),
                feedback_age_s=float(arrays.get("feedback_age_s", 0)),
                inequalities=len(arrays["h"]), scalar_conflicts=conflicts,
                conclusion=("Contradictory scalar intervals prove this QP infeasible."
                            if conflicts else
                            "No scalar contradiction found; coupled constraints may still be infeasible."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capture = args.capture.resolve()
    output = None if args.output is None else args.output.resolve()
    activate(args.workspace)
    from hexfellow_deploy.robot.upstream import load_umi
    load_umi()
    result = analyze(capture)
    encoded = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if output is not None:
        output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
