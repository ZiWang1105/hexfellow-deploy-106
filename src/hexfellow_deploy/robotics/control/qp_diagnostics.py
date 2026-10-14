"""Failure-only QP evidence; no solvers, simulator calls, or file I/O."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPFailureSnapshot:
    """Owned arrays in Pink displacement units, safe to export without pickle."""

    arrays: Mapping[str, np.ndarray]

    def scalar_conflicts(self):
        """Prove contradictory one-axis rows without another runtime solve."""
        g, h = self.arrays["G"], self.arrays["h"]
        size = len(self.arrays["q"])
        lower, upper = np.full(size, -np.inf), np.full(size, np.inf)
        rows = np.full((2, size), -1, dtype=int)
        for i, row in enumerate(g):
            indices = np.flatnonzero(row != 0.)
            if len(indices) != 1:
                continue
            j = int(indices[0])
            bound = h[i] / row[j]
            if row[j] > 0 and bound < upper[j]:
                upper[j], rows[1, j] = bound, i
            if row[j] < 0 and bound > lower[j]:
                lower[j], rows[0, j] = bound, i
        dt = float(self.arrays.get("control_dt_s", 1.))
        active = self.arrays.get("active_indices", np.arange(size))
        names = self.arrays.get("joint_names")
        return [dict(index=int(active[j]), joint=None if names is None else str(names[active[j]]),
                     minimum_velocity=float(lower[j]/dt), maximum_velocity=float(upper[j]/dt),
                     lower_row=int(rows[0,j]), upper_row=int(rows[1,j]),
                     lower_constraint=str(self.arrays["row_labels"][rows[0,j]]),
                     upper_constraint=str(self.arrays["row_labels"][rows[1,j]]))
                for j in np.flatnonzero(lower > upper + 1e-10)]

    @classmethod
    def capture(cls, problem, labels, **state):
        size = len(problem.q)
        arrays = {}
        for name in ("P", "q", "G", "h", "A", "b", "lb", "ub"):
            value = getattr(problem, name, None)
            if value is None:
                value = np.empty((0, size)) if name in ("G", "A") else np.empty(0)
            if hasattr(value, "toarray"):
                value = value.toarray()
            arrays[name] = np.array(value, dtype=float, copy=True)
        if len(labels) != len(arrays["h"]):
            raise ValueError("QP row labels do not match inequalities")
        arrays["row_labels"] = np.asarray(labels, dtype=str)
        arrays.update({
            name: np.array([] if value is None else value, copy=True)
            for name, value in state.items()
        })
        for array in arrays.values():
            array.setflags(write=False)
        return cls(arrays)
