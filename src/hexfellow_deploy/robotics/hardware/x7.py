"""Dual Archer X7 arm adapter with non-bypassable local watchdogs."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..control.motion_profile import MotionProfile

SDK_VERSION = "1.4.10"
ARM_SERIES = 29
ARM_DOF = 7
COMMAND_HZ = 50.0
COMMAND_DT_S = 1.0 / COMMAND_HZ
STATE_STALE_S = 0.040
COMMAND_STALE_S = 0.100
FOLLOWING_ERROR_SOFT_RAD = 0.050
FOLLOWING_ERROR_SOFT_HOLD_S = 0.100
FOLLOWING_ERROR_HARD_RAD = 0.100


class HardwareSafetyError(RuntimeError):
    """A local invariant stopped or prevented hardware motion."""


class ControlDelay(HardwareSafetyError):
    """Recoverable timing failure only; strict callers still stop on this error."""


class FeedbackSkewSafetyError(ControlDelay):
    """Otherwise valid feedback has not yet formed a coherent group."""


class FeedbackUnavailable(ControlDelay):
    """Transient IPC contention; never permission to substitute an old frame."""


@dataclass(frozen=True)
class ArmIdentity:
    side: str
    url: str
    arm_series: int
    joint_names: tuple[str, ...]
    position_lower_rad: tuple[float, ...]
    position_upper_rad: tuple[float, ...]


@dataclass(frozen=True)
class ArmState:
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    received_monotonic_s: float
    device_timestamp: Mapping[str, object] | None = None


class ArmEndpoint(Protocol):
    identity: ArmIdentity

    def read_state(self) -> ArmState: ...

    def command_position(self, positions_rad: tuple[float, ...]) -> None: ...

    def brake(self) -> None: ...

    def close(self) -> None: ...


def authorize_live_profile(profile: MotionProfile, supplied_sha256: str | None) -> None:
    """Require the exact immutable profile hash until hardware commissioning passes."""

    if profile.hardware_commissioned:
        return
    if supplied_sha256 != profile.sha256:
        raise HardwareSafetyError(
            f"motion profile {profile.profile_id!r} is uncommissioned; live motion requires "
            f"--allow-uncommissioned-profile-sha256 {profile.sha256}"
        )


def _joint_vector(value: object, *, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (ARM_DOF,) or not np.all(np.isfinite(array)):
        raise HardwareSafetyError(f"{label} must contain seven finite joint values")
    return tuple(float(item) for item in array)


def load_joint_trajectory(path: str | Path) -> tuple[dict[str, tuple[float, ...]], ...]:
    """Read immutable 50 Hz JSONL commands with left/right seven-joint arrays."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"joint trajectory does not exist: {path}")
    commands: list[dict[str, tuple[float, ...]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on trajectory line {line_number}: {error}") from error
        if not isinstance(row, Mapping) or set(row) != {"left", "right"}:
            raise ValueError(f"trajectory line {line_number} must contain only left and right")
        commands.append(
            {
                side: _joint_vector(row[side], label=f"line {line_number} {side}")
                for side in ("left", "right")
            }
        )
    if not commands:
        raise ValueError("joint trajectory is empty")
    return tuple(commands)


def _profile_command_derivatives(
    profile: MotionProfile,
    current: Mapping[str, np.ndarray],
    previous: Mapping[str, np.ndarray],
    previous_velocity: Mapping[str, np.ndarray],
    previous_acceleration: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    velocities: dict[str, np.ndarray] = {}
    accelerations: dict[str, np.ndarray] = {}
    for side, offset in (("left", 0), ("right", ARM_DOF)):
        velocity = (current[side] - previous[side]) / COMMAND_DT_S
        acceleration = (velocity - previous_velocity[side]) / COMMAND_DT_S
        jerk = (acceleration - previous_acceleration[side]) / COMMAND_DT_S
        velocity_limit = np.asarray(profile.arm_velocity_rad_s[offset : offset + ARM_DOF])
        acceleration_limit = np.asarray(
            profile.arm_acceleration_rad_s2[offset : offset + ARM_DOF]
        )
        jerk_limit = np.asarray(profile.arm_jerk_rad_s3[offset : offset + ARM_DOF])
        if np.any(np.abs(velocity) > velocity_limit + 1e-9):
            raise HardwareSafetyError(f"{side} command exceeds the profile velocity limit")
        if np.any(np.abs(acceleration) > acceleration_limit + 1e-9):
            raise HardwareSafetyError(f"{side} command exceeds the profile acceleration limit")
        if np.any(np.abs(jerk) > jerk_limit + 1e-9):
            raise HardwareSafetyError(f"{side} command exceeds the profile jerk limit")
        velocities[side] = velocity
        accelerations[side] = acceleration
    return velocities, accelerations


def validate_profile_trajectory(
    profile: MotionProfile,
    commands: Iterable[Mapping[str, object]],
) -> int:
    """Validate a file without importing or connecting to the vendor SDK."""

    previous: dict[str, np.ndarray] | None = None
    previous_velocity = {side: np.zeros(ARM_DOF) for side in ("left", "right")}
    previous_acceleration = {side: np.zeros(ARM_DOF) for side in ("left", "right")}
    count = 0
    for command in commands:
        if set(command) != {"left", "right"}:
            raise HardwareSafetyError("dual-arm command must contain exactly left and right")
        current: dict[str, np.ndarray] = {}
        for side, offset in (("left", 0), ("right", ARM_DOF)):
            target = np.asarray(_joint_vector(command[side], label=f"{side} command"))
            lower = np.asarray(profile.urdf_position_lower_rad[offset : offset + ARM_DOF])
            upper = np.asarray(profile.urdf_position_upper_rad[offset : offset + ARM_DOF])
            margin = profile.arm_position_margin_rad
            if np.any(target < lower + margin) or np.any(target > upper - margin):
                raise HardwareSafetyError(f"{side} command exceeds the profile planning bounds")
            current[side] = target
        if previous is not None:
            previous_velocity, previous_acceleration = _profile_command_derivatives(
                profile,
                current,
                previous,
                previous_velocity,
                previous_acceleration,
            )
        previous = current
        count += 1
    if count == 0:
        raise HardwareSafetyError("joint trajectory is empty")
    return count


class DualX7SafetyController:
    """Coordinate two endpoints so a fault on either side brakes both arms."""

    def __init__(
        self,
        left: ArmEndpoint,
        right: ArmEndpoint,
        profile: MotionProfile,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.endpoints = {"left": left, "right": right}
        self.profile = profile
        self.clock = clock
        self._last_commands: dict[str, np.ndarray] | None = None
        self._last_command_velocities: dict[str, np.ndarray] | None = None
        self._last_command_accelerations: dict[str, np.ndarray] | None = None
        self._last_command_time: float | None = None
        self._soft_error_since: dict[str, float | None] = {"left": None, "right": None}
        self._braked = False
        try:
            for side, endpoint in self.endpoints.items():
                identity = endpoint.identity
                if identity.side != side or identity.arm_series != ARM_SERIES:
                    raise HardwareSafetyError(f"{side} endpoint is not the expected Archer X7 H1")
                if (
                    len(identity.joint_names) != ARM_DOF
                    or len(identity.position_lower_rad) != ARM_DOF
                    or len(identity.position_upper_rad) != ARM_DOF
                ):
                    raise HardwareSafetyError(f"{side} endpoint does not expose seven bounded joints")
        except HardwareSafetyError:
            self.brake_both()
            raise

    def __enter__(self) -> DualX7SafetyController:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.brake_both()
        for endpoint in self.endpoints.values():
            endpoint.close()

    def brake_both(self) -> None:
        self._braked = True
        for endpoint in self.endpoints.values():
            try:
                endpoint.brake()
            except Exception:
                pass

    def _fail(self, message: str) -> None:
        self.brake_both()
        raise HardwareSafetyError(message)

    def states(self, *, now: float | None = None) -> dict[str, ArmState]:
        now = self.clock() if now is None else float(now)
        states: dict[str, ArmState] = {}
        try:
            for side, endpoint in self.endpoints.items():
                state = endpoint.read_state()
                _joint_vector(state.positions_rad, label=f"{side} state position")
                _joint_vector(state.velocities_rad_s, label=f"{side} state velocity")
                age = now - float(state.received_monotonic_s)
                if age < -1e-6 or age > STATE_STALE_S:
                    self._fail(f"{side} state is stale by {age:.3f}s")
                states[side] = state
        except HardwareSafetyError:
            raise
        except Exception as error:
            self._fail(f"failed to read dual-arm state: {error}")
        return states

    def _bounds(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        identity = self.endpoints[side].identity
        profile_offset = 0 if side == "left" else ARM_DOF
        profile_lower = np.asarray(
            self.profile.urdf_position_lower_rad[profile_offset : profile_offset + ARM_DOF]
        )
        profile_upper = np.asarray(
            self.profile.urdf_position_upper_rad[profile_offset : profile_offset + ARM_DOF]
        )
        lower = np.maximum(profile_lower, np.asarray(identity.position_lower_rad))
        upper = np.minimum(profile_upper, np.asarray(identity.position_upper_rad))
        margin = self.profile.arm_position_margin_rad
        if np.any(lower + margin >= upper - margin):
            raise HardwareSafetyError(f"{side} profile and device bounds have no safe overlap")
        return lower + margin, upper - margin

    def validate_commands(
        self,
        targets: Mapping[str, object],
        *,
        previous: Mapping[str, object] | None = None,
    ) -> dict[str, np.ndarray]:
        if set(targets) != {"left", "right"}:
            raise HardwareSafetyError("dual-arm command must contain exactly left and right")
        result: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            target = np.asarray(_joint_vector(targets[side], label=f"{side} command"))
            lower, upper = self._bounds(side)
            if np.any(target < lower) or np.any(target > upper):
                raise HardwareSafetyError(f"{side} command exceeds the intersected URDF/SDK position bounds")
            if previous is not None:
                prior = np.asarray(_joint_vector(previous[side], label=f"previous {side} command"))
                offset = 0 if side == "left" else ARM_DOF
                velocity = np.asarray(
                    self.profile.arm_velocity_rad_s[offset : offset + ARM_DOF]
                )
                if np.any(np.abs(target - prior) > velocity * COMMAND_DT_S + 1e-9):
                    raise HardwareSafetyError(f"{side} command exceeds the profile velocity limit")
            result[side] = target
        return result

    def validate_trajectory(
        self, commands: Iterable[Mapping[str, object]]
    ) -> int:
        previous: Mapping[str, object] | None = None
        previous_velocity = {side: np.zeros(ARM_DOF) for side in ("left", "right")}
        previous_acceleration = {side: np.zeros(ARM_DOF) for side in ("left", "right")}
        count = 0
        for command in commands:
            validated = self.validate_commands(command, previous=previous)
            if previous is not None:
                previous_arrays = {
                    side: np.asarray(previous[side], dtype=float) for side in ("left", "right")
                }
                previous_velocity, previous_acceleration = _profile_command_derivatives(
                    self.profile,
                    validated,
                    previous_arrays,
                    previous_velocity,
                    previous_acceleration,
                )
            previous = validated
            count += 1
        if count == 0:
            raise HardwareSafetyError("joint trajectory is empty")
        return count

    def command(
        self,
        targets: Mapping[str, object],
        *,
        source_timestamp_s: float,
    ) -> dict[str, float]:
        now = self.clock()
        if self._braked:
            raise HardwareSafetyError("dual-arm controller is braked")
        if now - float(source_timestamp_s) > COMMAND_STALE_S:
            self._fail("joint command source is stale")
        states = self.states(now=now)
        previous: Mapping[str, object] = (
            {side: state.positions_rad for side, state in states.items()}
            if self._last_commands is None
            else self._last_commands
        )
        previous_velocity = (
            {
                side: np.asarray(state.velocities_rad_s, dtype=float)
                for side, state in states.items()
            }
            if self._last_command_velocities is None
            else self._last_command_velocities
        )
        previous_acceleration = (
            {side: np.zeros(ARM_DOF) for side in ("left", "right")}
            if self._last_command_accelerations is None
            else self._last_command_accelerations
        )
        try:
            validated = self.validate_commands(targets, previous=previous)
            previous_arrays = {
                side: np.asarray(previous[side], dtype=float) for side in ("left", "right")
            }
            velocities, accelerations = _profile_command_derivatives(
                self.profile,
                validated,
                previous_arrays,
                previous_velocity,
                previous_acceleration,
            )
        except HardwareSafetyError as error:
            self._fail(str(error))
        maximum_errors: dict[str, float] = {}
        for side in ("left", "right"):
            following_reference = (
                np.asarray(states[side].positions_rad)
                if self._last_commands is None
                else self._last_commands[side]
            )
            error = float(
                np.max(np.abs(np.asarray(states[side].positions_rad) - following_reference))
            )
            maximum_errors[side] = error
            if error > FOLLOWING_ERROR_HARD_RAD:
                self._fail(f"{side} hard following error {error:.3f}rad")
            if error > FOLLOWING_ERROR_SOFT_RAD:
                since = self._soft_error_since[side]
                self._soft_error_since[side] = now if since is None else since
                if since is not None and now - since >= FOLLOWING_ERROR_SOFT_HOLD_S:
                    self._fail(f"{side} sustained following error {error:.3f}rad")
            else:
                self._soft_error_since[side] = None
        try:
            for side in ("left", "right"):
                self.endpoints[side].command_position(tuple(validated[side]))
        except Exception as error:
            self._fail(f"dual-arm position command failed: {error}")
        self._last_commands = validated
        self._last_command_velocities = velocities
        self._last_command_accelerations = accelerations
        self._last_command_time = now
        return maximum_errors

    def check_command_watchdog(self, *, now: float | None = None) -> None:
        now = self.clock() if now is None else float(now)
        if self._last_command_time is not None and now - self._last_command_time > COMMAND_STALE_S:
            self._fail("dual-arm command watchdog expired")


class HexArmEndpoint:
    """Lazy wrapper around hex_device 1.4.10; importing Policy Sim stays optional."""

    def __init__(
        self,
        side: str,
        url: str,
        *,
        timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            import hex_device
            from hex_device import CommandType, HexDeviceApi, public_api_types_pb2
        except ImportError as error:
            raise RuntimeError(
                "hardware support requires `pip install -e '.[hardware]'` with hex_device 1.4.10"
            ) from error
        if hex_device.__version__ != SDK_VERSION:
            raise RuntimeError(
                f"hardware adapter requires hex_device=={SDK_VERSION}, found {hex_device.__version__}"
            )
        self._clock = clock
        self._command_type = CommandType
        self._api = HexDeviceApi(ws_url=url, control_hz=250, enable_kcp=True, local_port=0)
        deadline = clock() + timeout_s
        robot_type = public_api_types_pb2.RobotType.RtArmArcherX7h1
        device = None
        while clock() < deadline:
            device = self._api.find_device_by_robot_type(robot_type)
            if device is not None and device.has_new_data():
                break
            sleep(0.01)
        if device is None or not device.has_new_data():
            self._api.close()
            raise TimeoutError(f"timed out waiting for an Archer X7 H1 at {url}")
        device.start()
        limits = device.get_joint_limits()
        names = device.get_joint_names()
        if device.get_arm_series() != ARM_SERIES or limits is None or names is None:
            self._api.close()
            raise HardwareSafetyError(f"unexpected arm identity at {url}")
        self._device = device
        self._last_timestamp_key: str | None = None
        self._last_received = clock()
        self.identity = ArmIdentity(
            side=side,
            url=url,
            arm_series=int(device.get_arm_series()),
            joint_names=tuple(str(name) for name in names),
            position_lower_rad=tuple(float(limit[0]) for limit in limits),
            position_upper_rad=tuple(float(limit[1]) for limit in limits),
        )

    def read_state(self) -> ArmState:
        status = self._device.get_simple_motor_status(pop=False)
        if status is None:
            raise HardwareSafetyError(f"{self.identity.side} arm has no state")
        timestamp = status.get("ts")
        key = json.dumps(timestamp, sort_keys=True)
        if key != self._last_timestamp_key:
            self._last_timestamp_key = key
            self._last_received = self._clock()
        states = self._device.get_motor_states()
        if states is not None and any(state != "normal" for state in states):
            raise HardwareSafetyError(f"{self.identity.side} arm reports a motor error")
        return ArmState(
            positions_rad=_joint_vector(status.get("pos"), label="SDK position"),
            velocities_rad_s=_joint_vector(status.get("vel"), label="SDK velocity"),
            received_monotonic_s=self._last_received,
            device_timestamp=timestamp,
        )

    def command_position(self, positions_rad: tuple[float, ...]) -> None:
        self._device.motor_command(self._command_type.POSITION, list(positions_rad))

    def brake(self) -> None:
        self._device.motor_command(self._command_type.BRAKE, [True] * ARM_DOF)

    def close(self) -> None:
        self._api.close()


def connect_dual_x7(
    left_url: str,
    right_url: str,
    profile: MotionProfile,
    *,
    timeout_s: float = 5.0,
) -> DualX7SafetyController:
    """Connect both arms, braking/closing the first if the second fails."""

    left = HexArmEndpoint("left", left_url, timeout_s=timeout_s)
    try:
        right = HexArmEndpoint("right", right_url, timeout_s=timeout_s)
    except Exception:
        left.brake()
        left.close()
        raise
    try:
        return DualX7SafetyController(left, right, profile)
    except Exception:
        left.close()
        right.close()
        raise
