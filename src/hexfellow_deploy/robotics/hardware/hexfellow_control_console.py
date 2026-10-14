"""Safety-gated interactive control for both X7 arms, a Theta lift, and a Maver base."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .x7 import ARM_DOF, SDK_VERSION, HardwareSafetyError
from .x7_joint_console import (
    POSITION_MARGIN_RAD,
    AbsoluteJointMove,
    JointConsoleConfig,
    MovePreview,
    X7JointControlSession,
    _safe_bounds,
    _suppress_known_hands_timeout_log,
    build_scoped_websocket_url,
    build_websocket_url,
    parse_angle,
)
from .x7_sdk_smoke import _device_identity, _StateTracker, probe_x7_device

DEVICE_NAMES = ("right", "left", "theta", "base")
ARM_NAMES = frozenset(("right", "left"))
DEFAULT_CONTROLLER_IDS = {
    "right": "14ed72e8740b3715",
    "left": "14ed72e8740b3715",
    "theta": "ed657740a4e1442b",
    "base": "c13c4d53b9b681a6",
}
_BENIGN_IDLE_SDK_MESSAGES = frozenset(
    {
        "Arm: Waiting to get the control of the arm...",
        "Chassis: You are not the session holder, please use start() method to get the control of the chassis...",
    }
)

THETA_CONTROL_HZ = 50.0
THETA_COMMAND_DT_S = 1.0 / THETA_CONTROL_HZ
THETA_DEFAULT_SPEED_M_S = 0.010
THETA_MAX_SPEED_M_S = 0.030
THETA_MAX_STEP_M = 0.100
THETA_END_MARGIN_M = 0.002
THETA_TARGET_TOLERANCE_M = 0.001
THETA_START_CHANGE_TOLERANCE_M = 0.001
THETA_SETTLED_SAMPLES = 3
THETA_COMMAND_OVERHEAD_S = 2.0

BASE_CONTROL_HZ = 50.0
BASE_COMMAND_DT_S = 1.0 / BASE_CONTROL_HZ
BASE_ZERO_HOLD_S = 0.25
BASE_STATE_STALE_S = 0.100
BASE_MAX_LINEAR_M_S = 0.050
BASE_MAX_ANGULAR_RAD_S = 0.100
BASE_MAX_DURATION_S = 2.0


class VendorLift(Protocol):
    robot_type: int

    def has_new_data(self) -> bool: ...

    def get_state(self) -> str: ...

    def get_pos_range(self) -> tuple[float, float]: ...

    def get_motor_positions(self) -> object: ...

    def get_move_speed(self) -> float: ...

    def get_max_move_speed(self) -> float: ...

    def get_pulse_per_meter(self) -> float: ...

    def get_parking_stop_detail(self) -> object: ...

    def set_move_speed(self, speed: object) -> None: ...

    def motor_command(self, command_type: object, values: object) -> None: ...


class VendorBase(Protocol):
    robot_type: int
    motor_count: int

    def has_new_data(self) -> bool: ...

    def get_base_state(self) -> str: ...

    def get_battery_info(self) -> Mapping[str, object]: ...

    def get_vehicle_speed(self, pop: bool = True) -> object: ...

    def get_vehicle_position(self, pop: bool = True) -> object: ...

    def get_simple_motor_status(self, pop: bool = True) -> Mapping[str, object] | None: ...

    def get_motor_states(self) -> Sequence[str] | None: ...

    def get_parking_stop_detail(self) -> object: ...

    def get_warning(self) -> object: ...

    def get_session_holder(self) -> int: ...

    def get_my_session_id(self) -> int: ...

    def is_api_control_initialized(self) -> bool: ...

    def set_vehicle_speed(self, speed_x: float, speed_y: float, speed_z: float) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class _InteractiveIdleLogFilter(logging.Filter):
    """Keep background SDK idle chatter from overwriting the interactive prompt."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in _BENIGN_IDLE_SDK_MESSAGES


@contextmanager
def _suppress_interactive_idle_logs() -> Iterator[None]:
    logger = logging.getLogger("hex_device")
    log_filter = _InteractiveIdleLogFilter()
    handlers = tuple(logger.handlers)
    for handler in handlers:
        handler.addFilter(log_filter)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(log_filter)


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    controller_id: str
    url: str
    expected_robot_type: str
    control_hz: int


@dataclass(frozen=True)
class UnifiedConsoleConfig:
    endpoints: tuple[EndpointSpec, ...]
    timeout_s: float = 5.0
    execute: bool = False
    allowed_motion: frozenset[str] = frozenset()
    arm_zero_checked: bool = False
    workspace_clear: bool = False
    estop_ready: bool = False
    theta_clear: bool = False
    base_clear: bool = False
    base_untethered: bool = False

    def validate(self) -> None:
        names = tuple(endpoint.name for endpoint in self.endpoints)
        if names != DEVICE_NAMES:
            raise ValueError(f"endpoints must be ordered as {DEVICE_NAMES}")
        if len({endpoint.url for endpoint in self.endpoints}) != len(DEVICE_NAMES):
            raise ValueError("each device endpoint must have a unique WebSocket URL")
        for endpoint in self.endpoints:
            if not endpoint.controller_id or not endpoint.url:
                raise ValueError(f"{endpoint.name} controller ID and URL are required")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("connection timeout must be finite and positive")
        unknown = set(self.allowed_motion) - set(DEVICE_NAMES)
        if unknown:
            raise ValueError(f"unknown motion allowlist entries: {sorted(unknown)}")
        if not self.execute or not self.allowed_motion:
            return

        missing: list[str] = []
        if self.allowed_motion & ARM_NAMES and not self.arm_zero_checked:
            missing.append("--confirm-arm-zero-checked")
        if not self.workspace_clear:
            missing.append("--confirm-workspace-clear")
        if not self.estop_ready:
            missing.append("--confirm-estop-ready")
        if "theta" in self.allowed_motion and not self.theta_clear:
            missing.append("--confirm-theta-clear")
        if "base" in self.allowed_motion:
            if not self.base_clear:
                missing.append("--confirm-base-clear")
            if not self.base_untethered:
                missing.append("--confirm-base-untethered")
        if missing:
            raise HardwareSafetyError(
                "live motion allowlist requires explicit operator confirmations: " + ", ".join(missing)
            )


@dataclass(frozen=True)
class ThetaMovePreview:
    start_m: float
    target_m: float
    delta_m: float
    speed_m_s: float
    estimated_total_s: float


@dataclass(frozen=True)
class BaseDrivePreview:
    velocity_x_m_s: float
    velocity_y_m_s: float
    angular_z_rad_s: float
    duration_s: float

    def validate(self) -> None:
        values = (
            self.velocity_x_m_s,
            self.velocity_y_m_s,
            self.angular_z_rad_s,
            self.duration_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("base drive values must be finite")
        if self.duration_s <= 0.0 or self.duration_s > BASE_MAX_DURATION_S:
            raise ValueError(f"base duration must be in (0, {BASE_MAX_DURATION_S:g}] seconds")
        if abs(self.velocity_x_m_s) > BASE_MAX_LINEAR_M_S:
            raise ValueError(f"base |vx| must not exceed {BASE_MAX_LINEAR_M_S:g} m/s")
        if abs(self.velocity_y_m_s) > BASE_MAX_LINEAR_M_S:
            raise ValueError(f"base |vy| must not exceed {BASE_MAX_LINEAR_M_S:g} m/s")
        if abs(self.angular_z_rad_s) > BASE_MAX_ANGULAR_RAD_S:
            raise ValueError(f"base |wz| must not exceed {BASE_MAX_ANGULAR_RAD_S:g} rad/s")
        if self.velocity_x_m_s == self.velocity_y_m_s == self.angular_z_rad_s == 0.0:
            raise ValueError("base drive command must contain a non-zero velocity")


def build_endpoint_specs(
    arm_controller_ip: str,
    theta_controller_ip: str,
    base_controller_ip: str,
    interface: str | None,
    *,
    controller_ids: Mapping[str, str] = DEFAULT_CONTROLLER_IDS,
) -> tuple[EndpointSpec, ...]:
    """Build the four known endpoints; the two arms share one controller."""

    return (
        EndpointSpec(
            "right",
            controller_ids["right"],
            build_websocket_url(arm_controller_ip, "right", interface),
            "RtArmArcherX7h1",
            250,
        ),
        EndpointSpec(
            "left",
            controller_ids["left"],
            build_websocket_url(arm_controller_ip, "left", interface),
            "RtArmArcherX7h1",
            250,
        ),
        EndpointSpec(
            "theta",
            controller_ids["theta"],
            build_scoped_websocket_url(theta_controller_ip, 8439, interface),
            "RtTheta2",
            50,
        ),
        EndpointSpec(
            "base",
            controller_ids["base"],
            build_scoped_websocket_url(base_controller_ip, 8439, interface),
            "RtMaverX4H1",
            50,
        ),
    )


def parse_length(value: str) -> float:
    """Parse an explicitly suffixed lift length and return meters."""

    normalized = value.strip().lower()
    if normalized.endswith("mm"):
        number = normalized[:-2].strip()
        scale = 0.001
    elif normalized.endswith("m"):
        number = normalized[:-1].strip()
        scale = 1.0
    else:
        raise ValueError("Theta length requires an explicit 'mm' or 'm' suffix")
    try:
        result = float(number) * scale
    except ValueError as error:
        raise ValueError("Theta length must be numeric") from error
    if not math.isfinite(result):
        raise ValueError("Theta length must be finite")
    return result


def _scalar(value: object, *, label: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1 or not np.all(np.isfinite(array)):
        raise HardwareSafetyError(f"{label} must contain one finite value")
    return float(array.reshape(-1)[0])


def _vector(value: object, size: int, *, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise HardwareSafetyError(f"{label} must contain {size} finite values")
    return tuple(float(item) for item in array)


def _parking_stop_report(detail: object) -> dict[str, object] | None:
    if detail is None:
        return None
    byte_size = getattr(detail, "ByteSize", None)
    if callable(byte_size) and not byte_size():
        return None
    if not callable(byte_size) and not detail:
        return None
    return {
        "reason": str(getattr(detail, "reason", detail)),
        "category": int(getattr(detail, "category", -1)),
        "is_remotely_clearable": bool(getattr(detail, "is_remotely_clearable", False)),
    }


def _private_bool(device: object, attribute: str, *, label: str) -> bool:
    lock = getattr(device, "_status_lock", None)
    if lock is None or not hasattr(device, attribute):
        raise HardwareSafetyError(f"hex_device does not expose the {label} field required for safe control")
    with lock:
        return bool(getattr(device, attribute))


class MotionCancelled(HardwareSafetyError):
    """Operator stop or lost viewer heartbeat; cleanup must still run."""


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise MotionCancelled("viewer stopped or heartbeat expired")


def _wait_for_new_data(
    device: object,
    timeout_s: float,
    *,
    label: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    cancelled=None,
) -> None:
    deadline = clock() + timeout_s
    while clock() < deadline:
        _check_cancelled(cancelled)
        if bool(device.has_new_data()):
            return
        sleep(0.01)
    raise HardwareSafetyError(f"timed out waiting for fresh {label} telemetry")


def _timestamp_key(status: Mapping[str, object]) -> str:
    return json.dumps(status.get("ts"), sort_keys=True, default=str)


class ArmConsoleDevice:
    kind = "arm"

    def __init__(
        self,
        spec: EndpointSpec,
        device: Any,
        command_type: object,
        timeout_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.device = device
        self.command_type = command_type
        self.timeout_s = timeout_s
        self.clock = clock
        self.sleep = sleep
        self.session: X7JointControlSession | None = None

    def probe(self) -> dict[str, object]:
        result = probe_x7_device(self.device, clock=self.clock)
        result.update(
            {
                "controller_id": self.spec.controller_id,
                "robot_type": self.spec.expected_robot_type,
                "api_control_initialized": _private_bool(
                    self.device, "_api_control_initialized", label="arm API initialization"
                ),
                "calibrated": _private_bool(self.device, "_calibrated", label="arm calibration"),
                "session_holder": int(self.device.get_session_holder()),
            }
        )
        return result

    def acquire(self) -> None:
        if self.session is not None:
            return
        config = JointConsoleConfig(
            url=self.spec.url,
            arm=self.spec.name,
            timeout_s=self.timeout_s,
            execute=True,
            zero_checked=True,
            workspace_clear=True,
            estop_ready=True,
        )
        session = X7JointControlSession(
            self.device,
            self.command_type,
            config,
            clock=self.clock,
            sleep=self.sleep,
        )
        try:
            session.acquire()
        except BaseException:
            session.close()
            raise
        self.session = session

    def status(self) -> dict[str, object]:
        if self.session is None:
            return self.probe()
        result = self.session.status()
        result.update(
            {
                "controller_id": self.spec.controller_id,
                "robot_type": self.spec.expected_robot_type,
                "api_control_initialized": _private_bool(
                    self.device, "_api_control_initialized", label="arm API initialization"
                ),
                "calibrated": _private_bool(self.device, "_calibrated", label="arm calibration"),
            }
        )
        return result

    def limits(self) -> dict[str, object]:
        if self.session is not None:
            return self.session.limits()
        names, lower, upper = _device_identity(self.device)
        safe_lower, safe_upper = _safe_bounds(lower, upper)
        return {
            "joint_names": list(names),
            "sdk_lower_deg": np.degrees(lower).tolist(),
            "sdk_upper_deg": np.degrees(upper).tolist(),
            "safe_lower_deg": np.degrees(safe_lower).tolist(),
            "safe_upper_deg": np.degrees(safe_upper).tolist(),
            "safety_margin_rad": POSITION_MARGIN_RAD,
        }

    def preview_move(self, joint_number: int, target_rad: float) -> MovePreview:
        if self.session is None:
            raise HardwareSafetyError("arm motion is not enabled and acquired")
        return self.session.preview_move(AbsoluteJointMove(joint_number, target_rad))

    def preview_jog(self, joint_number: int, delta_rad: float) -> MovePreview:
        if self.session is None:
            raise HardwareSafetyError("arm motion is not enabled and acquired")
        return self.session.preview_jog(joint_number, delta_rad)

    def execute(self, preview: MovePreview) -> dict[str, object]:
        if self.session is None:
            raise HardwareSafetyError("arm motion is not enabled and acquired")
        return self.session.execute_move(preview)

    def stop(self) -> None:
        if self.session is not None:
            self.session.brake()

    def release(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            session.close()


class ThetaConsoleDevice:
    kind = "theta"

    def __init__(
        self,
        spec: EndpointSpec,
        device: VendorLift,
        command_type: object,
        timeout_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.device = device
        self.command_type = command_type
        self.timeout_s = timeout_s
        self.clock = clock
        self.sleep = sleep
        self.acquired = False
        self.home_position_m: float | None = None
        self.position_commands_sent = 0
        self.brake_commands_sent = 0

    def _snapshot(self, *, cancelled=None) -> dict[str, object]:
        _wait_for_new_data(
            self.device,
            self.timeout_s,
            label="Theta",
            clock=self.clock,
            sleep=self.sleep,
            cancelled=cancelled,
        )
        position = _scalar(self.device.get_motor_positions(), label="Theta position")
        lower, upper = (float(value) for value in self.device.get_pos_range())
        if not all(math.isfinite(value) for value in (lower, upper)) or lower >= upper:
            raise HardwareSafetyError("Theta position range is unavailable or invalid")
        if position < lower - THETA_TARGET_TOLERANCE_M or position > upper + THETA_TARGET_TOLERANCE_M:
            raise HardwareSafetyError("Theta measured position is outside its reported range")
        pulse_per_meter = float(self.device.get_pulse_per_meter())
        max_speed_pulses_s = float(self.device.get_max_move_speed())
        if not math.isfinite(pulse_per_meter) or pulse_per_meter <= 0.0:
            raise HardwareSafetyError("Theta pulse-per-meter calibration is invalid")
        return {
            "controller_id": self.spec.controller_id,
            "robot_type": self.spec.expected_robot_type,
            "state": self.device.get_state(),
            "calibrated": _private_bool(self.device, "_calibrated", label="Theta calibration"),
            "position_m": position,
            "position_range_m": [lower, upper],
            "configured_speed_pulses_s": float(self.device.get_move_speed()),
            "max_speed_pulses_s": max_speed_pulses_s,
            "pulse_per_meter": pulse_per_meter,
            "parking_stop": _parking_stop_report(self.device.get_parking_stop_detail()),
            "terminal_state": "idle_braked" if self.acquired else "read_only",
        }

    @staticmethod
    def _require_healthy(snapshot: Mapping[str, object]) -> None:
        if not snapshot["calibrated"]:
            raise HardwareSafetyError("Theta is not calibrated; this console will not initiate calibration")
        if snapshot["parking_stop"] is not None:
            raise HardwareSafetyError("Theta reports an active parking/emergency stop")
        if snapshot["state"] not in {"LsBrake", "LsAlgorithmControl"}:
            raise HardwareSafetyError(f"Theta is not controllable in state {snapshot['state']}")

    def probe(self) -> dict[str, object]:
        return self._snapshot()

    def acquire(self, *, cancelled=None) -> None:
        _check_cancelled(cancelled)
        if self.acquired:
            return
        snapshot = self._snapshot(cancelled=cancelled)
        self._require_healthy(snapshot)
        self.home_position_m = float(snapshot["position_m"])
        self.acquired = True
        try:
            self.brake()
        except BaseException:
            self.acquired = False
            raise

    def status(self) -> dict[str, object]:
        return self._snapshot()

    def limits(self) -> dict[str, object]:
        snapshot = self._snapshot()
        lower, upper = (float(value) for value in snapshot["position_range_m"])
        return {
            "physical_lower_m": lower,
            "physical_upper_m": upper,
            "normal_safe_lower_m": lower + THETA_END_MARGIN_M,
            "normal_safe_upper_m": upper - THETA_END_MARGIN_M,
            "maximum_step_m": THETA_MAX_STEP_M,
            "default_speed_m_s": THETA_DEFAULT_SPEED_M_S,
            "hard_max_speed_m_s": THETA_MAX_SPEED_M_S,
            "home_return_exception_m": self.home_position_m,
        }

    def _preview(self, target_m: float) -> ThetaMovePreview:
        if not self.acquired:
            raise HardwareSafetyError("Theta motion is not enabled and acquired")
        snapshot = self._snapshot()
        self._require_healthy(snapshot)
        start = float(snapshot["position_m"])
        delta = target_m - start
        if not math.isfinite(target_m) or abs(delta) < 1e-9:
            raise ValueError("Theta target must be finite and different from its current position")
        if abs(delta) > THETA_MAX_STEP_M + 1e-12:
            raise ValueError(f"Theta step must not exceed {THETA_MAX_STEP_M * 1000:g} mm")
        lower, upper = (float(value) for value in snapshot["position_range_m"])
        in_safe_range = lower + THETA_END_MARGIN_M <= target_m <= upper - THETA_END_MARGIN_M
        returning_home = self.home_position_m is not None and abs(target_m - self.home_position_m) <= 1e-9
        if not in_safe_range and not returning_home:
            raise ValueError("Theta target enters the 2 mm endpoint safety margin")
        return ThetaMovePreview(
            start_m=start,
            target_m=target_m,
            delta_m=delta,
            speed_m_s=THETA_DEFAULT_SPEED_M_S,
            estimated_total_s=abs(delta) / THETA_DEFAULT_SPEED_M_S + THETA_COMMAND_OVERHEAD_S,
        )

    def preview_move(self, target_m: float) -> ThetaMovePreview:
        return self._preview(target_m)

    def preview_jog(self, delta_m: float) -> ThetaMovePreview:
        if not self.acquired:
            raise HardwareSafetyError("Theta motion is not enabled and acquired")
        snapshot = self._snapshot()
        return self._preview(float(snapshot["position_m"]) + delta_m)

    def execute(self, preview: ThetaMovePreview, *, cancelled=None, progress=None) -> dict[str, object]:
        if not self.acquired:
            raise HardwareSafetyError("Theta motion is not enabled and acquired")
        active_error: BaseException | None = None
        commands_before = self.position_commands_sent
        try:
            snapshot = self._snapshot(cancelled=cancelled)
            self._require_healthy(snapshot)
            if abs(float(snapshot["position_m"]) - preview.start_m) > THETA_START_CHANGE_TOLERANCE_M:
                raise HardwareSafetyError("Theta position changed after preview; enter the move again")
            pulse_per_meter = float(snapshot["pulse_per_meter"])
            max_speed_m_s = float(snapshot["max_speed_pulses_s"]) / pulse_per_meter
            if max_speed_m_s <= 0.0:
                raise HardwareSafetyError("Theta reports an invalid maximum speed")
            speed_m_s = min(preview.speed_m_s, THETA_MAX_SPEED_M_S, max_speed_m_s)
            speed_pulses_s = int(round(speed_m_s * pulse_per_meter))
            if speed_pulses_s <= 0:
                raise HardwareSafetyError("Theta speed converts to zero encoder pulses per second")
            self.device.set_move_speed(speed_pulses_s)
            self.device.motor_command(self.command_type.POSITION, float(preview.target_m))
            self.position_commands_sent += 1

            deadline = self.clock() + preview.estimated_total_s
            settled = 0
            last_position = float(snapshot["position_m"])
            while self.clock() < deadline:
                current = self._snapshot(cancelled=cancelled)
                self._require_healthy(current)
                if progress is not None:
                    progress(current)
                last_position = float(current["position_m"])
                if abs(last_position - preview.target_m) <= THETA_TARGET_TOLERANCE_M:
                    settled += 1
                    if settled >= THETA_SETTLED_SAMPLES:
                        return {
                            "start_m": preview.start_m,
                            "target_m": preview.target_m,
                            "delta_m": preview.delta_m,
                            "speed_m_s": speed_m_s,
                            "final_position_m": last_position,
                            "final_target_error_m": abs(last_position - preview.target_m),
                            "position_commands_sent": self.position_commands_sent - commands_before,
                            "terminal_state": "idle_braked",
                        }
                else:
                    settled = 0
                self.sleep(THETA_COMMAND_DT_S)
            raise HardwareSafetyError(
                f"Theta did not settle at its target before timeout (last={last_position:.4f} m)"
            )
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                self.brake()
            except Exception as brake_error:
                if active_error is not None:
                    raise HardwareSafetyError(
                        f"Theta move failed ({active_error}) and terminal BRAKE failed: {brake_error}"
                    ) from brake_error
                raise HardwareSafetyError(f"Theta terminal BRAKE failed: {brake_error}") from brake_error

    def brake(self) -> None:
        if not self.acquired:
            return
        self.device.motor_command(self.command_type.BRAKE, True)
        self.brake_commands_sent += 1
        self.sleep(THETA_COMMAND_DT_S)

    def stop(self) -> None:
        self.brake()

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.brake()
        finally:
            self.acquired = False


class BaseConsoleDevice:
    kind = "base"

    def __init__(
        self,
        spec: EndpointSpec,
        device: VendorBase,
        timeout_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.device = device
        self.timeout_s = timeout_s
        self.clock = clock
        self.sleep = sleep
        self.tracker = _StateTracker()
        self.started = False
        self.acquired = False
        self.velocity_commands_sent = 0
        self.zero_commands_sent = 0

    def _snapshot(self, *, cancelled=None) -> dict[str, object]:
        _wait_for_new_data(
            self.device,
            self.timeout_s,
            label="base",
            clock=self.clock,
            sleep=self.sleep,
            cancelled=cancelled,
        )
        status = self.device.get_simple_motor_status(pop=False)
        if status is None:
            raise HardwareSafetyError("base motor telemetry is unavailable")
        now = self.clock()
        timestamp_key = _timestamp_key(status)
        if timestamp_key != self.tracker.timestamp_key:
            self.tracker.timestamp_key = timestamp_key
            self.tracker.received_monotonic_s = now
        if self.tracker.received_monotonic_s is None:
            self.tracker.received_monotonic_s = now
        age = now - self.tracker.received_monotonic_s
        if age < -1e-6 or age > BASE_STATE_STALE_S:
            raise HardwareSafetyError(f"base telemetry is stale by {age:.3f}s")
        count = int(self.device.motor_count)
        positions = _vector(status.get("pos"), count, label="base motor position")
        velocities = _vector(status.get("vel"), count, label="base motor velocity")
        motor_states = self.device.get_motor_states()
        if motor_states is None or len(motor_states) != count:
            raise HardwareSafetyError("base motor health is unavailable")
        battery = dict(self.device.get_battery_info())
        return {
            "controller_id": self.spec.controller_id,
            "robot_type": self.spec.expected_robot_type,
            "state": self.device.get_base_state(),
            "api_control_initialized": bool(self.device.is_api_control_initialized()),
            "session_holder": int(self.device.get_session_holder()),
            "my_session_id": int(self.device.get_my_session_id()),
            "motor_states": list(motor_states),
            "motor_positions": list(positions),
            "motor_velocities": list(velocities),
            "battery": battery,
            "vehicle_speed": self.device.get_vehicle_speed(pop=False),
            "vehicle_position": self.device.get_vehicle_position(pop=False),
            "parking_stop": _parking_stop_report(self.device.get_parking_stop_detail()),
            "warning": self.device.get_warning(),
            "terminal_state": "control_acquired" if self.acquired else "read_only",
        }

    @staticmethod
    def _require_healthy(snapshot: Mapping[str, object], *, require_parked: bool = False) -> None:
        if any(state != "normal" for state in snapshot["motor_states"]):
            raise HardwareSafetyError("base reports a motor error")
        if snapshot["parking_stop"] is not None:
            raise HardwareSafetyError("base reports an active parking/emergency stop")
        if snapshot["warning"] is not None:
            raise HardwareSafetyError(f"base reports warning {snapshot['warning']}")
        if require_parked and snapshot["state"] != "BsParked":
            raise HardwareSafetyError(f"base must be parked before control; state={snapshot['state']}")

    def probe(self) -> dict[str, object]:
        return self._snapshot()

    def acquire(self, *, cancelled=None) -> None:
        _check_cancelled(cancelled)
        if self.acquired:
            return
        snapshot = self._snapshot(cancelled=cancelled)
        self._require_healthy(snapshot, require_parked=True)
        self._send_zero()
        self.device.start()
        self.started = True
        deadline = self.clock() + self.timeout_s
        while self.clock() < deadline:
            current = self._snapshot(cancelled=cancelled)
            self._require_healthy(current)
            mine = int(current["my_session_id"])
            if (
                mine != 0
                and int(current["session_holder"]) == mine
                and bool(current["api_control_initialized"])
                and current["state"] == "BsAlgorithmControl"
            ):
                self.acquired = True
                return
            self._send_zero()
            self.sleep(BASE_COMMAND_DT_S)
        self.release()
        raise HardwareSafetyError("timed out waiting for the base API control session")

    def status(self) -> dict[str, object]:
        return self._snapshot()

    def limits(self) -> dict[str, object]:
        return {
            "maximum_abs_vx_m_s": BASE_MAX_LINEAR_M_S,
            "maximum_abs_vy_m_s": BASE_MAX_LINEAR_M_S,
            "maximum_abs_wz_rad_s": BASE_MAX_ANGULAR_RAD_S,
            "maximum_duration_s": BASE_MAX_DURATION_S,
            "zero_hold_s": BASE_ZERO_HOLD_S,
        }

    def _send_zero(self) -> None:
        self.device.set_vehicle_speed(0.0, 0.0, 0.0)
        self.zero_commands_sent += 1

    def execute(self, preview: BaseDrivePreview, *, cancelled=None, progress=None) -> dict[str, object]:
        preview.validate()
        if not self.acquired:
            raise HardwareSafetyError("base motion is not enabled and acquired")
        active_error: BaseException | None = None
        commands_before = self.velocity_commands_sent
        started_at = self.clock()
        try:
            while self.clock() - started_at < preview.duration_s:
                snapshot = self._snapshot(cancelled=cancelled)
                self._require_healthy(snapshot)
                if progress is not None:
                    progress(snapshot)
                _check_cancelled(cancelled)
                mine = int(snapshot["my_session_id"])
                if mine == 0 or int(snapshot["session_holder"]) != mine:
                    raise HardwareSafetyError("base API control session was lost")
                if not snapshot["api_control_initialized"] or snapshot["state"] != "BsAlgorithmControl":
                    raise HardwareSafetyError("base left API algorithm-control state")
                self.device.set_vehicle_speed(
                    preview.velocity_x_m_s,
                    preview.velocity_y_m_s,
                    preview.angular_z_rad_s,
                )
                self.velocity_commands_sent += 1
                self.sleep(BASE_COMMAND_DT_S)
            return {
                "velocity_x_m_s": preview.velocity_x_m_s,
                "velocity_y_m_s": preview.velocity_y_m_s,
                "angular_z_rad_s": preview.angular_z_rad_s,
                "duration_s": preview.duration_s,
                "velocity_commands_sent": self.velocity_commands_sent - commands_before,
                "terminal_state": "parked_released",
            }
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                self.release()
            except Exception as release_error:
                if active_error is not None:
                    raise HardwareSafetyError(
                        f"base drive failed ({active_error}) and stop/release also failed: {release_error}"
                    ) from release_error
                raise

    def stop(self) -> None:
        self.release()

    def release(self) -> None:
        if not self.started:
            self.acquired = False
            return
        failures: list[str] = []
        try:
            samples = max(1, math.ceil(BASE_ZERO_HOLD_S * BASE_CONTROL_HZ))
            for _ in range(samples):
                self._send_zero()
                self.sleep(BASE_COMMAND_DT_S)
        except Exception as error:
            failures.append(f"zero-speed hold failed: {error}")
        try:
            self.device.stop()
        except Exception as error:
            failures.append(f"stop request failed: {error}")
        self.acquired = False
        self.started = False

        try:
            deadline = self.clock() + self.timeout_s
            while self.clock() < deadline:
                snapshot = self._snapshot()
                if snapshot["state"] == "BsParked" and int(snapshot["session_holder"]) == 0:
                    break
                self.sleep(BASE_COMMAND_DT_S)
            else:
                failures.append("timed out waiting for parked/session_holder=0")
        except Exception as error:
            failures.append(f"terminal-state check failed: {error}")
        if failures:
            raise HardwareSafetyError("; ".join(failures))


ConsoleDevice = ArmConsoleDevice | ThetaConsoleDevice | BaseConsoleDevice


class UnifiedRobotConsole:
    def __init__(
        self,
        devices: Mapping[str, ConsoleDevice],
        config: UnifiedConsoleConfig,
    ) -> None:
        config.validate()
        if set(devices) != set(DEVICE_NAMES):
            raise ValueError(f"devices must contain exactly {DEVICE_NAMES}")
        self.devices = dict(devices)
        self.config = config
        self.selected: str | None = None
        self.completed: list[dict[str, object]] = []

    def motion_enabled(self, name: str) -> bool:
        return self.config.execute and name in self.config.allowed_motion

    def devices_report(self) -> list[dict[str, object]]:
        return [
            {
                "name": spec.name,
                "controller_id": spec.controller_id,
                "url": spec.url,
                "expected_robot_type": spec.expected_robot_type,
                "motion_enabled": self.motion_enabled(spec.name),
                "selected": self.selected == spec.name,
            }
            for spec in self.config.endpoints
        ]

    def status(self, name: str | None = None) -> dict[str, object]:
        selected = name or self.selected
        if selected is None:
            raise ValueError("select a device with 'use' or enter 'status all'")
        return self.devices[selected].status()

    def status_all(self) -> dict[str, dict[str, object]]:
        return {name: self.devices[name].status() for name in DEVICE_NAMES}

    def use(self, name: str) -> None:
        if name not in self.devices:
            raise ValueError(f"device must be one of: {', '.join(DEVICE_NAMES)}")
        if self.selected == name:
            return
        self.release()
        self.selected = name
        if self.motion_enabled(name):
            try:
                self.devices[name].acquire()
            except BaseException:
                self.selected = None
                raise

    def limits(self) -> dict[str, object]:
        if self.selected is None:
            raise ValueError("select a device before requesting limits")
        return self.devices[self.selected].limits()

    def require_motion(self) -> ConsoleDevice:
        if self.selected is None:
            raise ValueError("select a device before entering a motion command")
        if not self.motion_enabled(self.selected):
            raise HardwareSafetyError(
                f"motion for {self.selected} is disabled; --execute and --allow-motion {self.selected} are required"
            )
        return self.devices[self.selected]

    def stop(self) -> None:
        if self.selected is None:
            raise ValueError("no device is selected")
        self.devices[self.selected].stop()

    def release(self) -> None:
        selected, self.selected = self.selected, None
        if selected is not None:
            self.devices[selected].release()

    def close(self) -> None:
        failures: list[str] = []
        selected, self.selected = self.selected, None
        ordered = ([selected] if selected is not None else []) + [
            name for name in DEVICE_NAMES if name != selected
        ]
        for name in ordered:
            try:
                self.devices[name].release()
            except Exception as error:
                failures.append(f"{name}: {error}")
        if failures:
            raise HardwareSafetyError("failed to close every device safely: " + "; ".join(failures))


_CONSOLE_HELP = """Commands:
  devices                         list endpoints, controller IDs, and motion gates
  use right|left|theta|base       select one device; releases the previous device
  status                          show the selected device status
  status all                      show all device statuses
  limits                          show conservative command limits
  move JOINT ANGLE                arm absolute move (degrees default, or deg/rad)
  jog JOINT DELTA                 arm relative move (degrees default, or deg/rad)
  move HEIGHT                     Theta absolute move; mm or m suffix is required
  jog DELTA                       Theta relative move; mm or m suffix is required
  drive VX VY WZ DURATION         base velocity command; m/s, m/s, rad/s, seconds
  stop                            brake/zero and stop the selected active session
  release                         safely release and deselect the current device
  help                            show this help
  quit                            safely release everything and exit"""


def _parse_joint_motion(line: str) -> tuple[int, float]:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"usage: {parts[0]} JOINT ANGLE")
    try:
        joint_number = int(parts[1])
    except ValueError as error:
        raise ValueError("joint number must be an integer from 1 to 7") from error
    if not 1 <= joint_number <= ARM_DOF:
        raise ValueError(f"joint number must be in [1, {ARM_DOF}]")
    return joint_number, parse_angle(parts[2])


def _parse_theta_motion(line: str) -> float:
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"usage: {parts[0]} LENGTH, for example '{parts[0]} 10mm'")
    return parse_length(parts[1])


def parse_drive_command(line: str) -> BaseDrivePreview:
    parts = line.split()
    if len(parts) != 5 or parts[0].lower() != "drive":
        raise ValueError("usage: drive VX VY WZ DURATION")
    try:
        preview = BaseDrivePreview(*(float(value) for value in parts[1:]))
    except ValueError as error:
        raise ValueError("base drive values must be numeric") from error
    preview.validate()
    return preview


def _format_motion_preview(device_name: str, preview: object) -> str:
    if isinstance(preview, MovePreview):
        index = preview.move.joint_index
        return (
            f"{device_name} J{preview.move.joint_number}: "
            f"{math.degrees(preview.start_positions_rad[index]):.2f} deg -> "
            f"{math.degrees(preview.move.target_rad):.2f} deg "
            f"(delta {math.degrees(preview.delta_rad):+.2f} deg, "
            f"estimated {preview.estimated_total_s:.2f} s)"
        )
    if isinstance(preview, ThetaMovePreview):
        return (
            f"theta: {preview.start_m * 1000:.2f} mm -> {preview.target_m * 1000:.2f} mm "
            f"(delta {preview.delta_m * 1000:+.2f} mm, speed {preview.speed_m_s * 1000:.1f} mm/s, "
            f"timeout {preview.estimated_total_s:.2f} s)"
        )
    if isinstance(preview, BaseDrivePreview):
        return (
            f"base: vx={preview.velocity_x_m_s:+.3f} m/s, vy={preview.velocity_y_m_s:+.3f} m/s, "
            f"wz={preview.angular_z_rad_s:+.3f} rad/s for {preview.duration_s:.2f} s"
        )
    raise TypeError(f"unsupported preview type: {type(preview)!r}")


def _preview_selected_motion(console: UnifiedRobotConsole, line: str) -> object:
    command = line.split()[0].lower()
    device = console.require_motion()
    if isinstance(device, ArmConsoleDevice):
        joint_number, angle = _parse_joint_motion(line)
        return device.preview_move(joint_number, angle) if command == "move" else device.preview_jog(
            joint_number, angle
        )
    if isinstance(device, ThetaConsoleDevice):
        length = _parse_theta_motion(line)
        return device.preview_move(length) if command == "move" else device.preview_jog(length)
    raise ValueError("move/jog commands apply only to an arm or Theta")


def run_interactive_console(
    console: UnifiedRobotConsole,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    output_fn(_CONSOLE_HELP)
    while True:
        prompt_name = console.selected or "none"
        try:
            line = input_fn(f"hexfellow[{prompt_name}]> ").strip()
        except EOFError:
            line = "quit"
        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()
        try:
            if command in {"quit", "exit"}:
                return console.completed
            if command == "help" and len(parts) == 1:
                output_fn(_CONSOLE_HELP)
                continue
            if command == "devices" and len(parts) == 1:
                output_fn(json.dumps(console.devices_report(), indent=2, sort_keys=True))
                continue
            if command == "use" and len(parts) == 2:
                console.use(parts[1].lower())
                mode = "motion acquired" if console.motion_enabled(parts[1].lower()) else "read-only"
                output_fn(f"Selected {parts[1].lower()} ({mode}).")
                continue
            if command == "status":
                if len(parts) == 2 and parts[1].lower() == "all":
                    output_fn(json.dumps(console.status_all(), indent=2, sort_keys=True, default=str))
                elif len(parts) == 1:
                    output_fn(json.dumps(console.status(), indent=2, sort_keys=True, default=str))
                else:
                    raise ValueError("usage: status or status all")
                continue
            if command == "limits" and len(parts) == 1:
                output_fn(json.dumps(console.limits(), indent=2, sort_keys=True))
                continue
            if command == "stop" and len(parts) == 1:
                console.stop()
                output_fn("Safe stop requested for the selected device.")
                continue
            if command == "release" and len(parts) == 1:
                console.release()
                output_fn("Selected device safely released.")
                continue
            if command in {"move", "jog"}:
                preview = _preview_selected_motion(console, line)
                device_name = console.selected
            elif command == "drive":
                device = console.require_motion()
                if not isinstance(device, BaseConsoleDevice):
                    raise ValueError("drive applies only to the base")
                preview = parse_drive_command(line)
                device_name = console.selected
            else:
                raise ValueError("unknown command; enter 'help' for the command list")

            output_fn(_format_motion_preview(str(device_name), preview))
            try:
                confirmation = input_fn("Execute this motion? Type y to continue [y/N]: ").strip().lower()
            except EOFError:
                confirmation = ""
            if confirmation != "y":
                output_fn("Motion cancelled; no motion command was sent.")
                continue
            device = console.require_motion()
            report = device.execute(preview)  # type: ignore[arg-type]
            report = {"device": device_name, **report}
            console.completed.append(report)
            output_fn(json.dumps(report, indent=2, sort_keys=True))
            if device_name == "base":
                console.release()
        except ValueError as error:
            output_fn(f"Input rejected: {error}")
        except HardwareSafetyError as error:
            try:
                console.release()
            except Exception as release_error:
                output_fn(f"SAFETY STOP: {error}; release also failed: {release_error}")
            else:
                output_fn(f"SAFETY STOP: {error}; selected device released.")
        except Exception as error:
            try:
                console.release()
            except Exception as release_error:
                raise HardwareSafetyError(
                    f"SDK/transport failure ({error}); release also failed: {release_error}"
                ) from release_error
            raise HardwareSafetyError(f"SDK/transport failure; selected device released: {error}") from error


def _load_vendor_sdk() -> tuple[Any, object, dict[str, int]]:
    try:
        import hex_device
        from hex_device import HexDeviceApi
        from hex_device.generated import public_api_types_pb2
        from hex_device.motor_base import CommandType
    except ImportError as error:
        raise RuntimeError(
            f"unified console requires hex_device=={SDK_VERSION}; install the 'hardware' extra"
        ) from error
    if getattr(hex_device, "__version__", None) != SDK_VERSION:
        raise RuntimeError(
            f"unified console requires hex_device=={SDK_VERSION}, found {hex_device.__version__}"
        )
    robot_types = {
        name: int(getattr(public_api_types_pb2.RobotType, name))
        for name in ("RtArmArcherX7h1", "RtTheta2", "RtMaverX4H1")
    }
    return HexDeviceApi, CommandType, robot_types


def _wait_for_device(
    api: object,
    robot_type: int,
    timeout_s: float,
    *,
    endpoint_name: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    deadline = clock() + timeout_s
    while clock() < deadline:
        if api.is_api_exit():
            raise ConnectionError(f"vendor API exited before {endpoint_name} was discovered")
        device = api.find_device_by_robot_type(robot_type)
        if device is not None and device.has_new_data():
            return device
        sleep(0.01)
    raise TimeoutError(f"timed out waiting for {endpoint_name} at its configured endpoint")


def _run_loaded_console(
    config: UnifiedConsoleConfig,
    api_type: Any,
    command_type: object,
    robot_types: Mapping[str, int],
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    interaction_fn=None,
) -> dict[str, object]:
    apis: list[object] = []
    devices: dict[str, ConsoleDevice] = {}
    console: UnifiedRobotConsole | None = None
    cleanup_error: BaseException | None = None
    report: dict[str, object] = {
        "schema_version": 1,
        "sdk_version": SDK_VERSION,
        "mode": "interactive-live" if config.execute else "interactive-read-only",
        "allowed_motion": sorted(config.allowed_motion) if config.execute else [],
    }
    try:
        for spec in config.endpoints:
            api = api_type(
                ws_url=spec.url,
                control_hz=spec.control_hz,
                enable_kcp=True,
                local_port=0,
            )
            apis.append(api)
            device = _wait_for_device(
                api,
                robot_types[spec.expected_robot_type],
                config.timeout_s,
                endpoint_name=spec.name,
            )
            if spec.name in ARM_NAMES:
                devices[spec.name] = ArmConsoleDevice(
                    spec, device, command_type, config.timeout_s
                )
            elif spec.name == "theta":
                devices[spec.name] = ThetaConsoleDevice(
                    spec, device, command_type, config.timeout_s
                )
            else:
                devices[spec.name] = BaseConsoleDevice(spec, device, config.timeout_s)

        console = UnifiedRobotConsole(devices, config)
        initial_status = console.status_all()
        report["devices"] = console.devices_report()
        report["initial_status"] = initial_status
        output_fn("Connected to both X7 arms, Theta, and the base; initial status reads succeeded.")
        output_fn(json.dumps(console.devices_report(), indent=2, sort_keys=True))
        report["motions"] = (interaction_fn or run_interactive_console)(
            console,
            input_fn=input_fn,
            output_fn=output_fn,
        )
        return report
    finally:
        if console is not None:
            try:
                console.close()
                report["terminal_state"] = "all_sessions_released"
            except BaseException as error:
                cleanup_error = error
        for api in reversed(apis):
            try:
                api.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise HardwareSafetyError(f"console cleanup failed: {cleanup_error}") from cleanup_error


def run_unified_console(
    config: UnifiedConsoleConfig,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    interaction_fn=None,
) -> dict[str, object]:
    """Connect all endpoints and run one interactive, single-active-device console."""

    config.validate()
    api_type, command_type, robot_types = _load_vendor_sdk()
    with _suppress_known_hands_timeout_log(), _suppress_interactive_idle_logs():
        return _run_loaded_console(
            config,
            api_type,
            command_type,
            robot_types,
            input_fn=input_fn,
            output_fn=output_fn,
            interaction_fn=interaction_fn,
        )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-controller-ip", required=True, help="bare IPv4 or IPv6 address")
    parser.add_argument("--theta-controller-ip", required=True, help="bare IPv4 or IPv6 address")
    parser.add_argument("--base-controller-ip", required=True, help="bare IPv4 or IPv6 address")
    parser.add_argument("--interface", help="IPv6 interface name or numeric scope, e.g. eno2 or 2")
    parser.add_argument("--timeout", type=_positive_float, default=5.0)
    parser.add_argument("--execute", action="store_true", help="allow explicitly allowlisted motion")
    parser.add_argument(
        "--allow-motion",
        action="append",
        choices=DEVICE_NAMES,
        default=[],
        help="repeat once per device that may move",
    )
    parser.add_argument("--confirm-arm-zero-checked", action="store_true")
    parser.add_argument("--confirm-workspace-clear", action="store_true")
    parser.add_argument("--confirm-estop-ready", action="store_true")
    parser.add_argument("--confirm-theta-clear", action="store_true")
    parser.add_argument("--confirm-base-clear", action="store_true")
    parser.add_argument("--confirm-base-untethered", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        endpoints = build_endpoint_specs(
            args.arm_controller_ip,
            args.theta_controller_ip,
            args.base_controller_ip,
            args.interface,
        )
        config = UnifiedConsoleConfig(
            endpoints=endpoints,
            timeout_s=args.timeout,
            execute=args.execute,
            allowed_motion=frozenset(args.allow_motion),
            arm_zero_checked=args.confirm_arm_zero_checked,
            workspace_clear=args.confirm_workspace_clear,
            estop_ready=args.confirm_estop_ready,
            theta_clear=args.confirm_theta_clear,
            base_clear=args.confirm_base_clear,
            base_untethered=args.confirm_base_untethered,
        )
        report = run_unified_console(config)
    except KeyboardInterrupt:
        print("Interrupted; BRAKE/zero, stop, release, and API close were requested.", file=sys.stderr)
        return 130
    except (ConnectionError, HardwareSafetyError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
