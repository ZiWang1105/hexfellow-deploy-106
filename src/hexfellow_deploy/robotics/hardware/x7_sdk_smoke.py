"""Conservative single-joint Archer X7 bring-up using the vendor SDK directly."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .x7 import ARM_DOF, ARM_SERIES, SDK_VERSION, STATE_STALE_S, HardwareSafetyError

COMMAND_HZ = 50.0
COMMAND_DT_S = 1.0 / COMMAND_HZ
DEFAULT_JOINT_NUMBER = 7
DEFAULT_DELTA_RAD = 0.020
MAX_DELTA_RAD = 0.050
DEFAULT_MOVE_DURATION_S = 2.0
DEFAULT_HOLD_DURATION_S = 0.5
POSITION_MARGIN_RAD = 0.050
MAX_START_VELOCITY_RAD_S = 0.050
MAX_FOLLOWING_ERROR_RAD = 0.050
PEAK_TOLERANCE_RAD = 0.005
RETURN_TOLERANCE_RAD = 0.010
BRAKE_SETTLE_S = 0.050


class VendorArm(Protocol):
    motor_count: int
    _api_control_initialized: bool
    _calibrated: bool

    def has_new_data(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_arm_series(self) -> int: ...

    def get_joint_names(self) -> Sequence[str] | None: ...

    def get_joint_limits(self) -> Sequence[Sequence[float]] | None: ...

    def get_simple_motor_status(self, pop: bool = False) -> Mapping[str, object] | None: ...

    def get_motor_states(self) -> Sequence[str] | None: ...

    def get_parking_stop_detail(self) -> object: ...

    def get_session_holder(self) -> int: ...

    def get_my_session_id(self) -> int: ...

    def motor_command(self, command_type: object, values: list[object]) -> None: ...


@dataclass(frozen=True)
class SingleJointJogConfig:
    url: str
    joint_number: int = DEFAULT_JOINT_NUMBER
    delta_rad: float = DEFAULT_DELTA_RAD
    move_duration_s: float = DEFAULT_MOVE_DURATION_S
    peak_hold_s: float = DEFAULT_HOLD_DURATION_S
    initial_hold_s: float = DEFAULT_HOLD_DURATION_S
    final_hold_s: float = DEFAULT_HOLD_DURATION_S
    timeout_s: float = 5.0
    execute: bool = False
    zero_checked: bool = False
    workspace_clear: bool = False
    estop_ready: bool = False

    @property
    def joint_index(self) -> int:
        return self.joint_number - 1

    def validate(self) -> None:
        if not self.url:
            raise ValueError("a WebSocket URL is required")
        if not 1 <= self.joint_number <= ARM_DOF:
            raise ValueError(f"joint number must be in [1, {ARM_DOF}]")
        if not math.isfinite(self.delta_rad) or not 0.0 < abs(self.delta_rad) <= MAX_DELTA_RAD:
            raise ValueError(f"delta must be finite, non-zero, and at most {MAX_DELTA_RAD:.3f} rad")
        for label, value in (
            ("move duration", self.move_duration_s),
            ("peak hold", self.peak_hold_s),
            ("initial hold", self.initial_hold_s),
            ("final hold", self.final_hold_s),
            ("connection timeout", self.timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if self.move_duration_s < 1.0:
            raise ValueError("move duration must be at least 1.0 s for a commissioning jog")
        if self.execute:
            missing = [
                label
                for enabled, label in (
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


@dataclass
class _StateTracker:
    timestamp_key: str | None = None
    received_monotonic_s: float | None = None


def _joint_vector(value: object, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (ARM_DOF,) or not np.all(np.isfinite(result)):
        raise HardwareSafetyError(f"{label} must contain seven finite values")
    return result


def _parking_stop_active(detail: object) -> bool:
    byte_size = getattr(detail, "ByteSize", None)
    if callable(byte_size):
        return bool(byte_size())
    return bool(detail)


def _read_state(
    device: VendorArm,
    tracker: _StateTracker,
    *,
    clock: Callable[[], float],
) -> tuple[np.ndarray, np.ndarray]:
    status = device.get_simple_motor_status(pop=False)
    if status is None:
        raise HardwareSafetyError("Archer X7 has no motor telemetry")
    now = clock()
    timestamp_key = json.dumps(status.get("ts"), sort_keys=True, default=str)
    if timestamp_key != tracker.timestamp_key:
        tracker.timestamp_key = timestamp_key
        tracker.received_monotonic_s = now
    if tracker.received_monotonic_s is None:
        tracker.received_monotonic_s = now
    state_age = now - tracker.received_monotonic_s
    if state_age < -1e-6 or state_age > STATE_STALE_S:
        raise HardwareSafetyError(f"Archer X7 telemetry is stale by {state_age:.3f}s")
    motor_states = device.get_motor_states()
    if motor_states is None or len(motor_states) != ARM_DOF:
        raise HardwareSafetyError("Archer X7 motor health is unavailable")
    if any(state != "normal" for state in motor_states):
        raise HardwareSafetyError("Archer X7 reports a motor error")
    if _parking_stop_active(device.get_parking_stop_detail()):
        raise HardwareSafetyError("Archer X7 reports an active parking/emergency stop")
    return (
        _joint_vector(status.get("pos"), label="SDK position"),
        _joint_vector(status.get("vel"), label="SDK velocity"),
    )


def _device_identity(device: VendorArm) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if device.get_arm_series() != ARM_SERIES or device.motor_count != ARM_DOF:
        raise HardwareSafetyError("endpoint is not an Archer X7 H1 with seven joints")
    names = device.get_joint_names()
    limits = device.get_joint_limits()
    if names is None or len(names) != ARM_DOF or limits is None or len(limits) != ARM_DOF:
        raise HardwareSafetyError("Archer X7 joint names or limits are unavailable")
    limits_array = np.asarray(limits, dtype=float)
    if limits_array.ndim != 2 or limits_array.shape[0] != ARM_DOF or limits_array.shape[1] < 2:
        raise HardwareSafetyError("Archer X7 joint limits have an unexpected shape")
    lower = limits_array[:, 0]
    upper = limits_array[:, 1]
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
        raise HardwareSafetyError("Archer X7 joint position limits are invalid")
    return tuple(str(name) for name in names), lower, upper


def probe_x7_device(
    device: VendorArm,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Read identity and fresh telemetry without acquiring control or sending commands."""

    names, lower, upper = _device_identity(device)
    positions, velocities = _read_state(device, _StateTracker(), clock=clock)
    return {
        "arm_series": int(device.get_arm_series()),
        "joint_names": list(names),
        "position_lower_rad": lower.tolist(),
        "position_upper_rad": upper.tolist(),
        "positions_rad": positions.tolist(),
        "velocities_rad_s": velocities.tolist(),
        "motion_commands_sent": 0,
    }


def _quintic_progress(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def build_jog_targets(
    start: Sequence[float],
    config: SingleJointJogConfig,
) -> tuple[np.ndarray, ...]:
    """Build the immutable 50 Hz hold/outbound/hold/return/settle command sequence."""

    config.validate()
    q0 = _joint_vector(start, label="initial position")
    peak = q0.copy()
    peak[config.joint_index] += config.delta_rad

    def hold(target: np.ndarray, duration_s: float) -> list[np.ndarray]:
        return [target.copy() for _ in range(max(1, math.ceil(duration_s * COMMAND_HZ)))]

    def move(begin: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
        samples = max(1, math.ceil(config.move_duration_s * COMMAND_HZ))
        return [
            begin + _quintic_progress(index / samples) * (end - begin)
            for index in range(1, samples + 1)
        ]

    return tuple(
        hold(q0, config.initial_hold_s)
        + move(q0, peak)
        + hold(peak, config.peak_hold_s)
        + move(peak, q0)
        + hold(q0, config.final_hold_s)
    )


def _wait_for_control_session(
    device: VendorArm,
    tracker: _StateTracker,
    config: SingleJointJogConfig,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    deadline = clock() + config.timeout_s
    last_ready = (False, False)
    while clock() < deadline:
        _read_state(device, tracker, clock=clock)
        session_id = int(device.get_my_session_id())
        if session_id != 0 and int(device.get_session_holder()) == session_id:
            status_lock = getattr(device, "_status_lock", None)
            if status_lock is None or not hasattr(device, "_api_control_initialized") or not hasattr(
                device, "_calibrated"
            ):
                raise HardwareSafetyError(
                    "hex_device does not expose the arm readiness fields required for safe control"
                )
            with status_lock:
                last_ready = (
                    bool(device._api_control_initialized),
                    bool(device._calibrated),
                )
            if all(last_ready):
                return
        sleep(0.01)
    raise HardwareSafetyError(
        "timed out waiting for the Archer X7 API session, initialization, and calibration "
        f"(api_initialized={last_ready[0]}, calibrated={last_ready[1]})"
    )


def _preflight_motion(
    q0: np.ndarray,
    dq0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: SingleJointJogConfig,
) -> np.ndarray:
    if np.any(np.abs(dq0) > MAX_START_VELOCITY_RAD_S):
        raise HardwareSafetyError("Archer X7 must be stationary before the commissioning jog")
    safe_lower = lower + POSITION_MARGIN_RAD
    safe_upper = upper - POSITION_MARGIN_RAD
    if np.any(safe_lower >= safe_upper):
        raise HardwareSafetyError("Archer X7 limits leave no commissioning position margin")
    if np.any(q0 < safe_lower) or np.any(q0 > safe_upper):
        raise HardwareSafetyError("initial Archer X7 pose is inside the commissioning limit margin")
    peak = q0.copy()
    peak[config.joint_index] += config.delta_rad
    if peak[config.joint_index] < safe_lower[config.joint_index] or peak[config.joint_index] > safe_upper[
        config.joint_index
    ]:
        raise HardwareSafetyError("requested joint jog enters the commissioning limit margin")
    return peak


def _brake_and_release(
    device: VendorArm,
    command_type: object,
    *,
    sleep: Callable[[float], None],
) -> None:
    failures: list[str] = []
    try:
        device.motor_command(command_type.BRAKE, [True] * ARM_DOF)
    except Exception as error:
        failures.append(f"brake request failed: {error}")
    sleep(BRAKE_SETTLE_S)
    try:
        device.stop()
    except Exception as error:
        failures.append(f"control release failed: {error}")
    sleep(BRAKE_SETTLE_S)
    if failures:
        raise HardwareSafetyError("; ".join(failures))


def execute_single_joint_jog(
    device: VendorArm,
    command_type: object,
    config: SingleJointJogConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Acquire one X7, execute the bounded jog, and always request a brake."""

    config.validate()
    if not config.execute:
        raise HardwareSafetyError("execute_single_joint_jog requires execute=True")
    _, lower, upper = _device_identity(device)
    tracker = _StateTracker()
    started = False
    active_error: BaseException | None = None
    position_commands = 0
    try:
        device.start()
        started = True
        _wait_for_control_session(device, tracker, config, clock=clock, sleep=sleep)
        q0, dq0 = _read_state(device, tracker, clock=clock)
        peak_target = _preflight_motion(q0, dq0, lower, upper, config)
        targets = build_jog_targets(q0, config)
        previous_target = q0.copy()
        direction = math.copysign(1.0, config.delta_rad)
        maximum_directed_displacement = 0.0
        maximum_following_error = 0.0

        for target in targets:
            positions, _velocities = _read_state(device, tracker, clock=clock)
            following_error = float(np.max(np.abs(positions - previous_target)))
            maximum_following_error = max(maximum_following_error, following_error)
            if following_error > MAX_FOLLOWING_ERROR_RAD:
                raise HardwareSafetyError(f"Archer X7 following error is {following_error:.3f} rad")
            directed_displacement = direction * (positions[config.joint_index] - q0[config.joint_index])
            maximum_directed_displacement = max(maximum_directed_displacement, float(directed_displacement))
            device.motor_command(command_type.POSITION, target.tolist())
            position_commands += 1
            previous_target = target
            sleep(COMMAND_DT_S)

        final_positions, final_velocities = _read_state(device, tracker, clock=clock)
        final_following_error = float(np.max(np.abs(final_positions - previous_target)))
        maximum_following_error = max(maximum_following_error, final_following_error)
        maximum_directed_displacement = max(
            maximum_directed_displacement,
            float(direction * (final_positions[config.joint_index] - q0[config.joint_index])),
        )
        if maximum_directed_displacement < abs(config.delta_rad) - PEAK_TOLERANCE_RAD:
            raise HardwareSafetyError(
                "selected joint did not reach the commissioning target: "
                f"observed {maximum_directed_displacement:.3f} rad"
            )
        return_error = np.abs(final_positions - q0)
        if float(return_error[config.joint_index]) > RETURN_TOLERANCE_RAD:
            raise HardwareSafetyError("selected joint did not return to its initial position")
        if float(np.max(np.delete(return_error, config.joint_index))) > RETURN_TOLERANCE_RAD:
            raise HardwareSafetyError("a non-selected joint moved outside the return tolerance")
        if np.any(np.abs(final_velocities) > MAX_START_VELOCITY_RAD_S):
            raise HardwareSafetyError("Archer X7 did not settle before terminal braking")
        return {
            "joint_number": config.joint_number,
            "requested_delta_rad": config.delta_rad,
            "initial_positions_rad": q0.tolist(),
            "peak_target_rad": peak_target.tolist(),
            "maximum_observed_displacement_rad": maximum_directed_displacement,
            "final_positions_rad": final_positions.tolist(),
            "maximum_return_error_rad": float(np.max(return_error)),
            "maximum_following_error_rad": maximum_following_error,
            "position_commands_sent": position_commands,
            "brake_commands_requested": 1,
            "terminal_state": "brake_requested",
        }
    except BaseException as error:
        active_error = error
        raise
    finally:
        if started:
            try:
                _brake_and_release(device, command_type, sleep=sleep)
            except Exception as brake_error:
                if active_error is not None:
                    raise HardwareSafetyError(
                        f"motion failed ({active_error}) and terminal brake/release also failed: {brake_error}"
                    ) from brake_error
                raise HardwareSafetyError(f"failed to brake and release Archer X7: {brake_error}") from brake_error


def _load_vendor_sdk() -> tuple[Any, Any, int]:
    try:
        import hex_device
        from hex_device import CommandType, HexDeviceApi, public_api_types_pb2
    except ImportError as error:
        raise RuntimeError("install the pinned hardware extra: python -m pip install -e '.[hardware]'") from error
    if hex_device.__version__ != SDK_VERSION:
        raise RuntimeError(
            f"single-joint smoke test requires hex_device=={SDK_VERSION}, found {hex_device.__version__}"
        )
    return HexDeviceApi, CommandType, public_api_types_pb2.RobotType.RtArmArcherX7h1


def _wait_for_x7(
    api: object,
    robot_type: int,
    timeout_s: float,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> VendorArm:
    deadline = clock() + timeout_s
    while clock() < deadline:
        if api.is_api_exit():
            raise ConnectionError("vendor API exited before an Archer X7 was discovered")
        device = api.find_device_by_robot_type(robot_type)
        if device is not None and device.has_new_data():
            return device
        sleep(0.01)
    raise TimeoutError("timed out waiting for an Archer X7 H1")


def run_sdk_smoke(config: SingleJointJogConfig) -> dict[str, object]:
    """Connect through hex_device, probe by default, and jog only when explicitly armed."""

    config.validate()
    api_type, command_type, robot_type = _load_vendor_sdk()
    api = api_type(ws_url=config.url, control_hz=250, enable_kcp=True, local_port=0)
    try:
        device = _wait_for_x7(api, robot_type, config.timeout_s, clock=time.monotonic, sleep=time.sleep)
        probe = probe_x7_device(device)
        report: dict[str, object] = {
            "schema_version": 1,
            "url": config.url,
            "sdk_version": SDK_VERSION,
            "mode": "live-jog" if config.execute else "read-only-probe",
            "probe": probe,
        }
        if config.execute:
            report["jog"] = execute_single_joint_jog(device, command_type, config)
        return report
    finally:
        api.close()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="quoted WebSocket URL, including the IPv6 zone identifier")
    parser.add_argument("--joint", type=int, choices=range(1, ARM_DOF + 1), default=DEFAULT_JOINT_NUMBER)
    parser.add_argument("--delta-rad", type=float, default=DEFAULT_DELTA_RAD)
    parser.add_argument("--move-duration", type=_positive_float, default=DEFAULT_MOVE_DURATION_S)
    parser.add_argument("--timeout", type=_positive_float, default=5.0)
    parser.add_argument("--execute", action="store_true", help="send the bounded position trajectory")
    parser.add_argument("--confirm-zero-checked", action="store_true")
    parser.add_argument("--confirm-workspace-clear", action="store_true")
    parser.add_argument("--confirm-estop-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = SingleJointJogConfig(
        url=args.url,
        joint_number=args.joint,
        delta_rad=args.delta_rad,
        move_duration_s=args.move_duration,
        timeout_s=args.timeout,
        execute=args.execute,
        zero_checked=args.confirm_zero_checked,
        workspace_clear=args.confirm_workspace_clear,
        estop_ready=args.confirm_estop_ready,
    )
    try:
        report = run_sdk_smoke(config)
    except KeyboardInterrupt:
        print("Interrupted; terminal brake was requested.", file=sys.stderr)
        return 130
    except (ConnectionError, HardwareSafetyError, RuntimeError, TimeoutError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
