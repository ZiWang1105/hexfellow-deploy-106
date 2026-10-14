"""Direct Quest experiment checks and a per-connection SDK downlink gate."""

import threading

import numpy as np

from .x7 import HardwareSafetyError


def validate_scope(args, mapping, profile):
    selected = getattr(args, "direct_quest_experiment", False)
    if not selected and not mapping.direct_quest:
        return
    mapping.validate_experiment_options()
    if not (selected and mapping.direct_quest and args.mode in {"probe", "shadow", "live"} and args.input == "quest"
            and args.stage == "whole-body" and profile.profile_id == "x7-mit-highspeed-v10"
            and profile.sha256 == mapping.data["commissioning"]["profile_sha256"]
            and getattr(args, "manual_test", None) is None
            and not getattr(args, "allow_candidate_geometry", False)):
        raise HardwareSafetyError("direct Quest requires its dedicated config, quest/whole-body and exact v10")


def hardware_review_report(mapping):
    review = mapping.effective_review
    names = ("mapping_checked", "same_tool_empty", "mit_units_gains_checked", "drive_protection_checked",
             "disconnect_stop_checked", "sdk_startup_checked", "initialization_stop_checked",
             "theta_stop_checked", "base_axes_checked")
    checks = {name: {"passed": review.get(name) is True, "source": "operator/config evidence"} for name in names}
    if mapping.verification is not None:
        for name in mapping.verification["checks"]:
            checks[name]["source"] = mapping.verification["report"]
    for name, size in (("hardware_effort_limits_nm", 14), ("hardware_base_velocity_limits", 3)):
        try:
            value = np.asarray(mapping.data.get(name), dtype=float)
        except (TypeError, ValueError):
            value = np.asarray([])
        checks[name] = {"passed": value.shape == (size,) and bool(np.all(np.isfinite(value) & (value > 0))),
                        "source": mapping.data.get("hardware_limits_source",
                                                   "reviewed hardware config; simulation values are not substitutes")}
    from ..control.hardware_capabilities import load_hardware_capabilities

    caps = load_hardware_capabilities(mapping.data.get("hardware_capabilities"))
    # Informational ratings do not imply verified firmware protection or a yaw rating.
    ratings = {"source": "manufacturer via operator, 2026-09-12", "sha256": caps.sha256,
               "base_linear_speed_m_s": caps.base_linear_speed_m_s, "base_angular_speed_rad_s": None,
               "lift_velocity_m_s": caps.lift_velocity_m_s,
               "continuous_torque_nm": caps.continuous_torque_nm}
    checks["evidence"] = {"passed": bool(str(review.get("evidence", "")).strip()),
                          "source": review.get("evidence", "")}
    missing = [k for k, v in checks.items() if not v["passed"]]
    accepted = []
    if mapping.verification is not None and "drive_protection_checked" in missing:
        accepted.append("drive_protection_checked")
    if mapping.demo_authorized:
        accepted.extend(k for k in missing if k in {
            "drive_protection_checked", "disconnect_stop_checked",
            "sdk_startup_checked", "initialization_stop_checked"})
    mapping.validate_experiment_options()
    if mapping.allow_unverified_hardware_limits:
        accepted = [k for k in missing if k == "drive_protection_checked" or (
            k in {"hardware_effort_limits_nm", "hardware_base_velocity_limits"} and mapping.data.get(k) is None)]
    if mapping.runtime_session:
        ratings["base_documented_limits"] = mapping.data.get("base_documented_limits")
    return {"checks": checks, "missing": missing, "experimental_unknowns_accepted": accepted,
            "verification": mapping.verification,
            "blocking": [k for k in missing if k not in accepted], "hardware_qualified": False,
            "manufacturer_ratings": ratings}


def check_power(state, *, acquiring, ignore_low_battery=False, allow_charging_readonly=False):
    if acquiring and allow_charging_readonly:
        raise HardwareSafetyError("read-only charging exception cannot authorize acquisition")
    battery = state.health["base"].get("battery")
    if not isinstance(battery, dict):
        raise HardwareSafetyError("battery telemetry unavailable")
    voltage, soc = battery.get("voltage"), battery.get("thousandth")
    if (not isinstance(voltage, (int, float)) or not np.isfinite(voltage) or voltage <= 0
            or not isinstance(soc, (int, float)) or not np.isfinite(soc) or not 0 <= soc <= 1000):
        raise HardwareSafetyError("invalid battery telemetry")
    if battery.get("charging") not in (None, False) and not allow_charging_readonly:
        raise HardwareSafetyError("charging: disconnect charger before whole-body control")
    # Operator-selected software threshold (2026-10-08), in SDK thousandths.
    # Apply equally during acquisition and operation; this is not a firmware rating.
    minimum = 150
    if (soc < minimum and not ignore_low_battery) or state.health["base"].get("warning") is not None:
        raise HardwareSafetyError(f"battery below experimental {minimum / 10:g}% threshold or base warning")
    # None means unavailable, never inferred False. CLI separately requires the
    # physical charger-disconnected confirmation even when telemetry says False.
    return {"percent": soc / 10, "voltage": voltage, "charging": battery.get("charging"),
            "source": "SDK BaseStatus, not a certified power interlock"}


def check_start_position(state, *, allow_lift_retreat=False):
    """Reject endpoint starts before acquisition; never auto-move away from a limit."""
    theta = state.health["theta"]
    low, high = theta["range"]
    at_upper = low <= theta["pos"] < low + .002
    if not (low + .002 <= theta["pos"] <= high - .002 or (allow_lift_retreat and at_upper)):
        raise HardwareSafetyError("Theta starts within 2 mm of an endpoint; move away with remote, then retry")
    for side in ("left", "right"):
        row = state.health[side]
        position, limits = np.asarray(row["pos"]), np.asarray(row["joint_limits"])
        if (position.shape != (7,) or limits.ndim != 2 or limits.shape[0] != 7 or limits.shape[1] < 2
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(limits))):
            raise HardwareSafetyError(f"invalid {side} initial joint bounds")
        if np.any(position < limits[:, 0] + .012) or np.any(position > limits[:, 1] - .012):
            raise HardwareSafetyError(f"{side} starts inside joint endpoint margin; reposition before acquisition")


class DirectDownlinkGate:
    def __init__(self, name):
        self.name = name
        self.phase = "readonly"
        self.initialized = False
        self.fault = None
        self.blocked = 0
        self.lock = threading.RLock()

    def transition(self, phase):
        with self.lock:
            if phase != "stopping" and (self.fault or self.phase == "stopping"):
                raise HardwareSafetyError("cannot rearm direct Quest downlink")
            self.phase = phase

    def filter(self, data):
        with self.lock:
            kind = data.WhichOneof("down")
            if kind in {"enable_kcp", "set_report_frequency", "placeholder_message"}:
                return data
            allowed = False
            expected = "arm_command" if self.name in {"left", "right"} else (
                "linear_lift_command" if self.name == "theta" else "base_command")
            phase = "stopping" if self.fault else self.phase
            if kind == expected and phase != "readonly":
                command = getattr(data, kind)
                if expected == "arm_command":
                    if command.WhichOneof("command") != "arm_exclusive_command":
                        return self.reject()
                    command = command.arm_exclusive_command
                    field = command.WhichOneof("exclusive_command")
                else:
                    field = command.WhichOneof("command")
                if field == "api_control_initialize":
                    if not command.api_control_initialize:
                        allowed = phase == "stopping"
                    elif phase == "initializing" and not self.initialized:
                        self.initialized = True
                        allowed = True
                elif expected == "arm_command":
                    if field == "calibrate":
                        allowed = phase == "initializing" and self.initialized
                    elif field == "arm_api_control_command":
                        control = command.arm_api_control_command
                        if control.WhichOneof("command") == "motor_targets":
                            modes = [t.WhichOneof("target") for t in control.motor_targets.targets]
                            allowed = (modes == ["brake"] * 7 or (not modes and phase != "stopping")
                                       or (phase == "active" and modes == ["mit_target"] * 7))
                elif expected == "linear_lift_command":
                    allowed = field == "brake" or (field == "set_speed" and phase == "initializing") or (
                        field == "target_pos" and phase == "active")
                elif field == "simple_move_command":
                    move = command.simple_move_command
                    if move.WhichOneof("command") == "xyz_speed":
                        allowed = phase == "active" or all(value == 0 for _, value in move.xyz_speed.ListFields())
                    elif move.WhichOneof("command") == "brake":
                        allowed = True
            if allowed:
                return data
            return self.reject()

    def reject(self):
        self.blocked += 1
        if self.phase not in {"readonly", "stopping"}:
            self.fault = "unexpected SDK command, automatic rearm or clear-stop blocked"
        return None


def gated_api_type(base, gate):
    class DirectApi(base):
        async def _send_down_message(self, data):
            allowed = gate.filter(data)
            if allowed is not None:
                import time

                self._policy_gate = gate
                started = time.monotonic()
                sequence = getattr(gate, "sent_sequence", None)
                try:
                    if getattr(gate, "checked_transport", False) and gate.phase != "readonly":
                        from .mit_wire import send_checked

                        result = await send_checked(self, allowed)
                    else:
                        result = await super()._send_down_message(allowed)
                except Exception as error:
                    gate.fault = f"SDK send failed: {error}"
                    raise
                if hasattr(gate, "record_dispatch"):
                    gate.record_dispatch(allowed, started, time.monotonic(), sequence=sequence)
                return result
    return DirectApi
