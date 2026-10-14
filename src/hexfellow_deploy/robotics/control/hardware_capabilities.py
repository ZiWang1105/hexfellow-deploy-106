"""Manufacturer ratings, distinct from operating limits and firmware evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from ..paths import CONFIG_DIR

CAPABILITIES_PATH = CONFIG_DIR / "hardware" / "hexfellow_capabilities.yaml"


@dataclass(frozen=True)
class HardwareCapabilities:
    sha256: str
    velocity_rad_s: tuple[float, ...]
    peak_torque_nm: tuple[float, ...]
    continuous_torque_nm: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    lift_velocity_m_s: float
    base_linear_speed_m_s: float


@lru_cache(maxsize=4)
def _parse(raw: bytes) -> HardwareCapabilities:
    data = yaml.safe_load(raw)
    if (data.get("schema_version") != 1 or data.get("reference_frame") != "joint_output"
            or data.get("arm_joint_order") != list(range(1, 8))):
        raise ValueError("invalid manufacturer hardware capability contract")
    values = {}
    for key in ("velocity_rad_s", "peak_torque_nm", "continuous_torque_nm", "kp", "kd"):
        v = np.asarray(data[key], dtype=float)
        if v.shape != (7,) or not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError(f"manufacturer {key} must contain seven positive finite values")
        values[key] = tuple(v) * 2  # Model order: left J1..7, then right J1..7.
    if np.any(np.asarray(values["continuous_torque_nm"]) > values["peak_torque_nm"]):
        raise ValueError("continuous torque cannot exceed peak torque")
    for key in ("lift_velocity_m_s", "base_linear_speed_m_s"):
        v = float(data[key])
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"invalid manufacturer {key}")
        values[key] = v
    return HardwareCapabilities(hashlib.sha256(raw).hexdigest(), **values)


def load_hardware_capabilities(reference=None) -> HardwareCapabilities:
    result = _parse(CAPABILITIES_PATH.read_bytes())
    if reference is not None and (
        reference.get("path") != "config/hardware/hexfellow_capabilities.yaml"
        or reference.get("sha256") != result.sha256
    ):
        raise ValueError("manufacturer hardware capability hash mismatch")
    return result


def effort_breakdown(command, q, v, bounds):
    """Sampling-time estimate; not a firmware torque readback or future bound."""
    position = command.kp * (command.position - np.asarray(q))
    velocity = command.kd * (command.velocity - np.asarray(v))
    pd = position + velocity
    total = pd + command.torque_ff
    return {"position_pd_nm": position, "velocity_pd_nm": velocity,
            "pd_nm": pd, "feedforward_nm": command.torque_ff, "requested_nm": total,
            "remaining_nm": np.asarray(bounds) - np.abs(total), "budget_nm": np.asarray(bounds)}
