"""Interactive, safety-gated single-joint control for one HexFellow Archer X7 arm."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import re
import socket
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from .x7 import ARM_DOF, SDK_VERSION, HardwareSafetyError
from .x7_sdk_smoke import (
    BRAKE_SETTLE_S,
    COMMAND_DT_S,
    COMMAND_HZ,
    MAX_FOLLOWING_ERROR_RAD,
    MAX_START_VELOCITY_RAD_S,
    POSITION_MARGIN_RAD,
    VendorArm,
    _device_identity,
    _load_vendor_sdk,
    _read_state,
    _StateTracker,
    _wait_for_control_session,
    _wait_for_x7,
    probe_x7_device,
)

ARM_PORTS = {"right": 8439, "left": 9439}
DEFAULT_MAX_SPEED_DEG_S = 10.0
HARD_MAX_SPEED_DEG_S = 10.0
DEFAULT_MAX_STEP_DEG = 30.0
MIN_MOVE_DURATION_S = 1.0
DEFAULT_SETTLE_S = 0.5
TARGET_TOLERANCE_RAD = 0.010
STATE_CHANGE_TOLERANCE_RAD = 0.010
_INTERFACE_RE = re.compile(r"^[A-Za-z0-9_.:@-]+$")
_KNOWN_HANDS_TIMEOUT_ERROR = "Hands periodic task exception: unsupported operand type(s) for -: 'float' and 'NoneType'"


class _KnownHandsTimeoutFilter(logging.Filter):
    """Hide only the hex_device 1.4.10 Hands timeout bug, not other SDK errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _KNOWN_HANDS_TIMEOUT_ERROR


@contextmanager
def _suppress_known_hands_timeout_log() -> Iterator[None]:
    logger = logging.getLogger("hex_device")
    log_filter = _KnownHandsTimeoutFilter()
    handlers = tuple(logger.handlers)
    for handler in handlers:
        handler.addFilter(log_filter)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(log_filter)


@dataclass(frozen=True)
class JointConsoleConfig:
    """Connection and non-bypassable motion limits for one console session."""

    url: str
    arm: str
    timeout_s: float = 5.0
    execute: bool = False
    zero_checked: bool = False
    workspace_clear: bool = False
    estop_ready: bool = False
    max_speed_deg_s: float = DEFAULT_MAX_SPEED_DEG_S
    max_step_deg: float = DEFAULT_MAX_STEP_DEG
    settle_s: float = DEFAULT_SETTLE_S

    @property
    def max_speed_rad_s(self) -> float:
        return math.radians(self.max_speed_deg_s)

    @property
    def max_step_rad(self) -> float:
        return math.radians(self.max_step_deg)

    def validate(self) -> None:
        if not self.url:
            raise ValueError("a WebSocket URL is required")
        if self.arm not in ARM_PORTS:
            raise ValueError("arm must be 'right' or 'left'")
        for label, value in (
            ("connection timeout", self.timeout_s),
            ("maximum speed", self.max_speed_deg_s),
            ("maximum step", self.max_step_deg),
            ("settle duration", self.settle_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if self.max_speed_deg_s > HARD_MAX_SPEED_DEG_S:
            raise ValueError(f"maximum speed must not exceed {HARD_MAX_SPEED_DEG_S:g} deg/s")
        if self.max_step_deg > DEFAULT_MAX_STEP_DEG:
            raise ValueError(f"maximum step must not exceed {DEFAULT_MAX_STEP_DEG:g} deg")
        if self.execute:
            missing = [
                option
                for enabled, option in (
                    (self.zero_checked, "--confirm-zero-checked"),
                    (self.workspace_clear, "--confirm-workspace-clear"),
                    (self.estop_ready, "--confirm-estop-ready"),
                )
                if not enabled
            ]
            if missing:
                raise HardwareSafetyError(
                    "live motion requires explicit operator confirmations: " + ", ".join(missing)
                )


@dataclass(frozen=True)
class AbsoluteJointMove:
    """An absolute target for a one-based X7 joint number."""

    joint_number: int
    target_rad: float

    @property
    def joint_index(self) -> int:
        return self.joint_number - 1

    def validate(self) -> None:
        if not 1 <= self.joint_number <= ARM_DOF:
            raise ValueError(f"joint number must be in [1, {ARM_DOF}]")
        if not math.isfinite(self.target_rad):
            raise ValueError("target angle must be finite")


@dataclass(frozen=True)
class MovePreview:
    """A checked move proposal shown to the operator before confirmation."""

    move: AbsoluteJointMove
    start_positions_rad: tuple[float, ...]
    target_positions_rad: tuple[float, ...]
    delta_rad: float
    move_duration_s: float
    estimated_total_s: float


def resolve_interface_scope(interface: str) -> str:
    """Return the numeric IPv6 scope accepted by ``hex_device==1.4.10``."""

    value = interface.strip()
    if not value or _INTERFACE_RE.fullmatch(value) is None:
        raise ValueError("invalid IPv6 interface name")
    if value.isdecimal():
        if int(value) <= 0:
            raise ValueError("IPv6 interface index must be positive")
        return str(int(value))
    try:
        return str(socket.if_nametoindex(value))
    except OSError as error:
        raise ValueError(f"network interface does not exist: {value}") from error


def build_scoped_websocket_url(
    controller_ip: str,
    port: int,
    interface: str | None = None,
) -> str:
    """Build an SDK URL, translating an IPv6 interface name to a numeric scope."""

    if not 1 <= port <= 65535:
        raise ValueError("WebSocket port must be in [1, 65535]")
    address_text = controller_ip.strip()
    if not address_text or "://" in address_text or "[" in address_text or "]" in address_text:
        raise ValueError("--controller-ip must be a bare IPv4 or IPv6 address")
    embedded_zone: str | None = None
    if "%" in address_text:
        address_text, embedded_zone = address_text.rsplit("%", 1)
    try:
        address = ipaddress.ip_address(address_text)
    except ValueError as error:
        raise ValueError(f"invalid controller IP address: {controller_ip}") from error
    if isinstance(address, ipaddress.IPv4Address):
        if embedded_zone is not None or interface is not None:
            raise ValueError("--interface is only valid with IPv6")
        return f"ws://{address}:{port}"

    interface_zone = resolve_interface_scope(interface) if interface is not None else None
    embedded_scope = resolve_interface_scope(embedded_zone) if embedded_zone is not None else None
    if interface_zone is not None and embedded_scope is not None and interface_zone != embedded_scope:
        raise ValueError("IPv6 zone and --interface disagree")
    zone = interface_zone or embedded_scope
    if address.is_link_local and zone is None:
        raise ValueError("link-local IPv6 requires --interface")
    scoped = f"{address}%{zone}" if zone is not None else str(address)
    return f"ws://[{scoped}]:{port}"


def build_websocket_url(controller_ip: str, arm: str, interface: str | None = None) -> str:
    """Build an Archer X7 URL using its right/left controller port mapping."""

    if arm not in ARM_PORTS:
        raise ValueError("arm must be 'right' or 'left'")
    return build_scoped_websocket_url(controller_ip, ARM_PORTS[arm], interface)


def parse_angle(value: str) -> float:
    """Parse degrees by default, with explicit ``deg`` and ``rad`` suffixes."""

    normalized = value.strip().lower()
    scale = math.pi / 180.0
    if normalized.endswith("rad"):
        normalized = normalized[:-3].strip()
        scale = 1.0
    elif normalized.endswith("deg"):
        normalized = normalized[:-3].strip()
    try:
        parsed = float(normalized)
    except ValueError as error:
        raise ValueError("angle must be a number in degrees or have a 'rad' suffix") from error
    result = parsed * scale
    if not math.isfinite(result):
        raise ValueError("angle must be finite")
    return result


def parse_move_command(line: str) -> AbsoluteJointMove:
    """Parse ``move JOINT ANGLE`` from the interactive console."""

    parts = line.split()
    if len(parts) != 3 or parts[0].lower() != "move":
        raise ValueError("usage: move JOINT ANGLE, for example 'move 3 30' or 'move 3 0.52rad'")
    try:
        joint_number = int(parts[1])
    except ValueError as error:
        raise ValueError("joint number must be an integer from 1 to 7") from error
    move = AbsoluteJointMove(joint_number=joint_number, target_rad=parse_angle(parts[2]))
    move.validate()
    return move


def _safe_bounds(lower: Sequence[float], upper: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    safe_lower = np.asarray(lower, dtype=float) + POSITION_MARGIN_RAD
    safe_upper = np.asarray(upper, dtype=float) - POSITION_MARGIN_RAD
    if safe_lower.shape != (ARM_DOF,) or safe_upper.shape != (ARM_DOF,):
        raise HardwareSafetyError("Archer X7 limits must contain seven values")
    if np.any(safe_lower >= safe_upper):
        raise HardwareSafetyError("Archer X7 limits leave no safe position margin")
    return safe_lower, safe_upper


def _validate_measured_pose(positions: np.ndarray, safe_lower: np.ndarray, safe_upper: np.ndarray) -> None:
    if np.any(positions < safe_lower) or np.any(positions > safe_upper):
        raise HardwareSafetyError("measured Archer X7 pose is inside the configured joint-limit margin")


def _validate_stationary(velocities: np.ndarray) -> None:
    if np.any(np.abs(velocities) > MAX_START_VELOCITY_RAD_S):
        raise HardwareSafetyError("Archer X7 must be stationary before starting a move")


def _move_duration_s(delta_rad: float, max_speed_rad_s: float) -> float:
    # The maximum derivative of 10t^3 - 15t^4 + 6t^5 is 1.875.
    return max(MIN_MOVE_DURATION_S, 1.875 * abs(delta_rad) / max_speed_rad_s)


def build_absolute_joint_trajectory(
    preview: MovePreview,
    *,
    command_hz: float = COMMAND_HZ,
) -> tuple[np.ndarray, ...]:
    """Build a 50 Hz minimum-jerk trajectory for a previously checked preview."""

    if not math.isfinite(command_hz) or command_hz <= 0.0:
        raise ValueError("command rate must be finite and positive")
    start = np.asarray(preview.start_positions_rad, dtype=float)
    target = np.asarray(preview.target_positions_rad, dtype=float)
    if start.shape != (ARM_DOF,) or target.shape != (ARM_DOF,):
        raise ValueError("move preview must contain seven-joint start and target vectors")
    samples = max(1, math.ceil(preview.move_duration_s * command_hz))
    targets = []
    for index in range(1, samples + 1):
        phase = index / samples
        progress = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        targets.append(start + progress * (target - start))
    return tuple(targets)


class X7JointControlSession:
    """Own one SDK control session and brake between confirmed moves."""

    def __init__(
        self,
        device: VendorArm,
        command_type: object,
        config: JointConsoleConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()
        if not config.execute:
            raise HardwareSafetyError("a control session requires execute=True")
        self.device = device
        self.command_type = command_type
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.tracker = _StateTracker()
        self.joint_names: tuple[str, ...] = ()
        self.lower = np.empty(0)
        self.upper = np.empty(0)
        self.safe_lower = np.empty(0)
        self.safe_upper = np.empty(0)
        self.started = False
        self.acquired = False
        self.closed = False
        self.position_commands_sent = 0
        self.brake_commands_sent = 0

    def acquire(self) -> None:
        if self.started:
            raise RuntimeError("Archer X7 control session was already started")
        self.joint_names, self.lower, self.upper = _device_identity(self.device)
        self.safe_lower, self.safe_upper = _safe_bounds(self.lower, self.upper)
        self.device.start()
        self.started = True
        _wait_for_control_session(
            self.device,
            self.tracker,
            self.config,
            clock=self.clock,
            sleep=self.sleep,
        )
        self.acquired = True
        positions, velocities = self._checked_state(require_stationary=True)
        _validate_measured_pose(positions, self.safe_lower, self.safe_upper)
        _validate_stationary(velocities)
        self.brake()

    def _ensure_control(self) -> None:
        session_id = int(self.device.get_my_session_id())
        holder = int(self.device.get_session_holder())
        if session_id == 0 or holder != session_id:
            raise HardwareSafetyError(
                f"Archer X7 API control session was lost (mine={session_id}, holder={holder})"
            )

    def _checked_state(self, *, require_stationary: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not self.acquired:
            raise HardwareSafetyError("Archer X7 API control session is not acquired")
        self._ensure_control()
        positions, velocities = _read_state(self.device, self.tracker, clock=self.clock)
        _validate_measured_pose(positions, self.safe_lower, self.safe_upper)
        if require_stationary:
            _validate_stationary(velocities)
        return positions, velocities

    def brake(self) -> None:
        if not self.acquired:
            return
        self.device.motor_command(self.command_type.BRAKE, [True] * ARM_DOF)
        self.brake_commands_sent += 1
        self.sleep(BRAKE_SETTLE_S)

    def status(self) -> dict[str, object]:
        positions, velocities = self._checked_state()
        return {
            "arm": self.config.arm,
            "joint_names": list(self.joint_names),
            "positions_rad": positions.tolist(),
            "positions_deg": np.degrees(positions).tolist(),
            "velocities_rad_s": velocities.tolist(),
            "session_holder": int(self.device.get_session_holder()),
            "terminal_state": "idle_braked",
        }

    def limits(self) -> dict[str, object]:
        return {
            "joint_names": list(self.joint_names),
            "sdk_lower_deg": np.degrees(self.lower).tolist(),
            "sdk_upper_deg": np.degrees(self.upper).tolist(),
            "safe_lower_deg": np.degrees(self.safe_lower).tolist(),
            "safe_upper_deg": np.degrees(self.safe_upper).tolist(),
            "safety_margin_rad": POSITION_MARGIN_RAD,
        }

    def preview_move(self, move: AbsoluteJointMove) -> MovePreview:
        move.validate()
        positions, _velocities = self._checked_state(require_stationary=True)
        return self._preview_from_positions(move, positions)

    def preview_jog(self, joint_number: int, delta_rad: float) -> MovePreview:
        """Preview a relative single-joint target from the latest checked pose."""

        if not math.isfinite(delta_rad) or abs(delta_rad) < 1e-9:
            raise ValueError("relative joint displacement must be finite and non-zero")
        positions, _velocities = self._checked_state(require_stationary=True)
        if not 1 <= joint_number <= ARM_DOF:
            raise ValueError(f"joint number must be in [1, {ARM_DOF}]")
        move = AbsoluteJointMove(joint_number, float(positions[joint_number - 1] + delta_rad))
        return self._preview_from_positions(move, positions)

    def _preview_from_positions(
        self,
        move: AbsoluteJointMove,
        positions: np.ndarray,
    ) -> MovePreview:
        move.validate()
        delta = float(move.target_rad - positions[move.joint_index])
        if abs(delta) < 1e-9:
            raise ValueError("selected joint is already at the requested target")
        if abs(delta) > self.config.max_step_rad + 1e-12:
            raise ValueError(
                f"requested step is {math.degrees(abs(delta)):.2f} deg; "
                f"the limit is {self.config.max_step_deg:.2f} deg"
            )
        if not self.safe_lower[move.joint_index] <= move.target_rad <= self.safe_upper[move.joint_index]:
            raise ValueError("requested target enters the configured joint-limit margin")
        target = positions.copy()
        target[move.joint_index] = move.target_rad
        duration_s = _move_duration_s(delta, self.config.max_speed_rad_s)
        return MovePreview(
            move=move,
            start_positions_rad=tuple(float(value) for value in positions),
            target_positions_rad=tuple(float(value) for value in target),
            delta_rad=delta,
            move_duration_s=duration_s,
            estimated_total_s=duration_s + self.config.settle_s,
        )

    def execute_move(self, preview: MovePreview) -> dict[str, object]:
        """Execute one confirmed preview, then always return to idle braking."""

        active_error: BaseException | None = None
        position_commands_before = self.position_commands_sent
        maximum_following_error = 0.0
        try:
            positions, _velocities = self._checked_state(require_stationary=True)
            preview_start = np.asarray(preview.start_positions_rad)
            if float(np.max(np.abs(positions - preview_start))) > STATE_CHANGE_TOLERANCE_RAD:
                raise HardwareSafetyError("arm pose changed after preview; inspect it and enter the move again")
            targets = build_absolute_joint_trajectory(preview)
            previous_target = preview_start.copy()
            target = np.asarray(preview.target_positions_rad)
            settle_samples = max(1, math.ceil(self.config.settle_s * COMMAND_HZ))

            for commanded_target in (*targets, *(target.copy() for _ in range(settle_samples))):
                tick_started = self.clock()
                measured, _velocities = self._checked_state()
                following_error = float(np.max(np.abs(measured - previous_target)))
                maximum_following_error = max(maximum_following_error, following_error)
                if following_error > MAX_FOLLOWING_ERROR_RAD:
                    raise HardwareSafetyError(f"Archer X7 following error is {following_error:.3f} rad")
                self.device.motor_command(self.command_type.POSITION, commanded_target.tolist())
                self.position_commands_sent += 1
                previous_target = commanded_target
                remaining = COMMAND_DT_S - (self.clock() - tick_started)
                if remaining > 0.0:
                    self.sleep(remaining)

            final_positions, final_velocities = self._checked_state()
            selected_error = abs(final_positions[preview.move.joint_index] - preview.move.target_rad)
            non_selected_error = float(
                np.max(np.abs(np.delete(final_positions - preview_start, preview.move.joint_index)))
            )
            if selected_error > TARGET_TOLERANCE_RAD:
                raise HardwareSafetyError(
                    f"selected joint target error is {selected_error:.3f} rad after settling"
                )
            if non_selected_error > TARGET_TOLERANCE_RAD:
                raise HardwareSafetyError(
                    f"a non-selected joint moved by {non_selected_error:.3f} rad after settling"
                )
            _validate_stationary(final_velocities)
            return {
                "joint_number": preview.move.joint_number,
                "start_rad": preview.start_positions_rad[preview.move.joint_index],
                "target_rad": preview.move.target_rad,
                "delta_rad": preview.delta_rad,
                "move_duration_s": preview.move_duration_s,
                "maximum_following_error_rad": maximum_following_error,
                "final_target_error_rad": float(selected_error),
                "position_commands_sent": self.position_commands_sent - position_commands_before,
                "terminal_state": "idle_braked",
            }
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                self.brake()
            except Exception as brake_error:
                if active_error is not None:
                    raise HardwareSafetyError(
                        f"move failed ({active_error}) and the terminal brake also failed: {brake_error}"
                    ) from brake_error
                raise HardwareSafetyError(f"failed to brake after the move: {brake_error}") from brake_error

    def close(self) -> None:
        if self.closed:
            return
        failures: list[str] = []
        if self.started:
            try:
                self.device.motor_command(self.command_type.BRAKE, [True] * ARM_DOF)
                self.brake_commands_sent += 1
                self.sleep(BRAKE_SETTLE_S)
            except Exception as error:
                failures.append(f"brake request failed: {error}")
        if self.started:
            try:
                self.device.stop()
                self.sleep(BRAKE_SETTLE_S)
            except Exception as error:
                failures.append(f"control release failed: {error}")
        self.acquired = False
        self.started = False
        self.closed = True
        if failures:
            raise HardwareSafetyError("; ".join(failures))


_CONSOLE_HELP = """Commands:
  status                 show current joint positions and velocities
  limits                 show SDK and safety-margin position limits
  move JOINT ANGLE       set one absolute target; degrees by default, e.g. move 3 30
  move JOINT ANGLErad    use radians explicitly, e.g. move 3 0.52rad
  stop                   re-issue BRAKE and remain in the console
  help                   show this help
  quit                   BRAKE, release the session, and exit"""


def _format_preview(preview: MovePreview) -> str:
    return (
        f"J{preview.move.joint_number}: "
        f"{math.degrees(preview.start_positions_rad[preview.move.joint_index]):.2f} deg -> "
        f"{math.degrees(preview.move.target_rad):.2f} deg "
        f"(delta {math.degrees(preview.delta_rad):+.2f} deg, "
        f"estimated {preview.estimated_total_s:.2f} s)"
    )


def run_interactive_console(
    session: X7JointControlSession,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Run the blocking terminal REPL while the arm remains braked between moves."""

    completed: list[dict[str, object]] = []
    output_fn(_CONSOLE_HELP)
    while True:
        try:
            line = input_fn(f"x7[{session.config.arm}]> ").strip()
        except EOFError:
            line = "quit"
        if not line:
            continue
        command = line.lower()
        if command in {"quit", "exit"}:
            return completed
        if command == "help":
            output_fn(_CONSOLE_HELP)
            continue
        if command == "status":
            output_fn(json.dumps(session.status(), indent=2, sort_keys=True))
            continue
        if command == "limits":
            output_fn(json.dumps(session.limits(), indent=2, sort_keys=True))
            continue
        if command == "stop":
            session.brake()
            output_fn("BRAKE requested; control session remains open.")
            continue
        if command.startswith("move"):
            try:
                preview = session.preview_move(parse_move_command(line))
            except ValueError as error:
                output_fn(f"Input rejected: {error}")
                continue
            output_fn(_format_preview(preview))
            try:
                confirmation = input_fn("Execute this move? Type y to continue [y/N]: ").strip().lower()
            except EOFError:
                confirmation = ""
            if confirmation != "y":
                output_fn("Move cancelled; no position command was sent.")
                continue
            report = session.execute_move(preview)
            completed.append(report)
            output_fn(
                f"Move complete; target error {math.degrees(report['final_target_error_rad']):.3f} deg. "
                "BRAKE requested."
            )
            continue
        output_fn("Unknown command. Type 'help' for the command list.")


def run_sdk_joint_console(
    config: JointConsoleConfig,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, object]:
    """Probe by default; acquire control and start the REPL only with all live gates."""

    config.validate()
    api_type, command_type, robot_type = _load_vendor_sdk()
    with _suppress_known_hands_timeout_log():
        return _run_loaded_sdk_joint_console(
            config,
            api_type,
            command_type,
            robot_type,
            input_fn=input_fn,
            output_fn=output_fn,
        )


def _run_loaded_sdk_joint_console(
    config: JointConsoleConfig,
    api_type: Any,
    command_type: object,
    robot_type: int,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> dict[str, object]:
    api = api_type(ws_url=config.url, control_hz=250, enable_kcp=True, local_port=0)
    session: X7JointControlSession | None = None
    report: dict[str, Any] | None = None
    try:
        device = _wait_for_x7(api, robot_type, config.timeout_s, clock=time.monotonic, sleep=time.sleep)
        probe = probe_x7_device(device)
        probe["session_holder"] = int(device.get_session_holder())
        report = {
            "schema_version": 1,
            "arm": config.arm,
            "url": config.url,
            "sdk_version": SDK_VERSION,
            "mode": "interactive-live" if config.execute else "read-only-probe",
            "probe": probe,
        }
        if not config.execute:
            return report
        session = X7JointControlSession(device, command_type, config)
        session.acquire()
        output_fn(
            f"Connected to the {config.arm} Archer X7. The arm is braked between confirmed moves."
        )
        report["moves"] = run_interactive_console(session, input_fn=input_fn, output_fn=output_fn)
        return report
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if report is not None and session is not None:
                report["position_commands_sent"] = session.position_commands_sent
                report["brake_commands_sent"] = session.brake_commands_sent
                report["terminal_state"] = "braked_stopped_closed"
            api.close()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_PORTS), required=True)
    parser.add_argument("--controller-ip", required=True, help="bare controller IPv4 or IPv6 address")
    parser.add_argument("--interface", help="network interface required for link-local IPv6, for example eno2")
    parser.add_argument("--timeout", type=_positive_float, default=5.0)
    parser.add_argument(
        "--max-speed-deg-s",
        type=_positive_float,
        default=DEFAULT_MAX_SPEED_DEG_S,
        help=f"maximum {HARD_MAX_SPEED_DEG_S:g} (default: %(default)s)",
    )
    parser.add_argument("--execute", action="store_true", help="acquire control and enable confirmed moves")
    parser.add_argument("--confirm-zero-checked", action="store_true")
    parser.add_argument("--confirm-workspace-clear", action="store_true")
    parser.add_argument("--confirm-estop-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        url = build_websocket_url(args.controller_ip, args.arm, args.interface)
        config = JointConsoleConfig(
            url=url,
            arm=args.arm,
            timeout_s=args.timeout,
            execute=args.execute,
            zero_checked=args.confirm_zero_checked,
            workspace_clear=args.confirm_workspace_clear,
            estop_ready=args.confirm_estop_ready,
            max_speed_deg_s=args.max_speed_deg_s,
        )
        report = run_sdk_joint_console(config)
    except KeyboardInterrupt:
        print("Interrupted; BRAKE, stop, and API close were requested.", file=sys.stderr)
        return 130
    except (ConnectionError, HardwareSafetyError, RuntimeError, TimeoutError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
