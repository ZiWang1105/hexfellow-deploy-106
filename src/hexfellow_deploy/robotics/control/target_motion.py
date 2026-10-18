"""C2 Cartesian reference shaping for discontinuous IK targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

try:
    from _hexfellow_collision_kernel import (
        KERNEL_ABI_VERSION as _NATIVE_KERNEL_ABI,
    )
    from _hexfellow_collision_kernel import (
        quintic_limits_are_satisfied as _native_quintic_limits,
    )
    from _hexfellow_collision_kernel import (
        quintic_peak_norms as _native_quintic_peak_norms,
    )
except ImportError:  # Source checkouts retain an equivalent NumPy fallback.
    _NATIVE_KERNEL_ABI = 0
    _native_quintic_peak_norms = None
    _native_quintic_limits = None


try:
    from _hexfellow_collision_kernel import quintic_coefficients as _native_coefficients
except ImportError:
    _native_coefficients = None


class TargetTrajectoryError(RuntimeError):
    """A valid requested pose has no bounded trajectory in the current search."""


@dataclass(frozen=True)
class CartesianMotionLimits:
    linear_velocity_m_s: float
    linear_acceleration_m_s2: float
    linear_jerk_m_s3: float
    angular_velocity_rad_s: float
    angular_acceleration_rad_s2: float
    angular_jerk_rad_s3: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Cartesian motion limits must be finite and positive")


@dataclass(frozen=True)
class CartesianMotion:
    pose: pin.SE3
    body_twist: np.ndarray
    body_acceleration: np.ndarray
    body_jerk: np.ndarray

    def copy(self) -> CartesianMotion:
        return CartesianMotion(
            self.pose.copy(),
            self.body_twist.copy(),
            self.body_acceleration.copy(),
            self.body_jerk.copy(),
        )


@dataclass(frozen=True)
class CartesianMotionUtilization:
    linear_velocity: float
    linear_acceleration: float
    linear_jerk: float
    angular_velocity: float
    angular_acceleration: float
    angular_jerk: float


@dataclass
class _QuinticSegment:
    translation_coefficients: np.ndarray
    rotation_coefficients: np.ndarray
    start_rotation: np.ndarray
    duration_s: float
    elapsed_s: float
    request: CartesianMotion
    destination: CartesianMotion
    motion_limited: bool = False


@dataclass
class _PlannerSnapshot:
    motion: CartesianMotion | None
    segment: _QuinticSegment | None
    previous_request: pin.SE3 | None
    previous_secant_world: np.ndarray
    previous_velocity_world: np.ndarray
    elapsed_since_distinct_request_s: float
    previous_distinct_interval_s: float | None
    request_is_moving: bool


def _validate_vector(value: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite 6-vector")
    return vector.copy()


def _validate_pose(pose: pin.SE3) -> pin.SE3:
    try:
        translation = np.asarray(pose.translation, dtype=float)
        rotation = np.asarray(pose.rotation, dtype=float)
    except AttributeError as error:
        raise ValueError("Cartesian target must be a Pinocchio SE3 pose") from error
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("Cartesian target translation must be finite")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Cartesian target rotation must be finite")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("Cartesian target rotation is not in SO(3)")
    return pose.copy()


def _quintic_coefficients(
    start: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end: np.ndarray,
    end_velocity: np.ndarray,
    end_acceleration: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Return ascending-power coefficients for a vector quintic."""

    duration_s = float(duration_s)
    c0 = np.asarray(start, dtype=float)
    c1 = np.asarray(start_velocity, dtype=float)
    c2 = 0.5 * np.asarray(start_acceleration, dtype=float)
    residual_position = np.asarray(end, dtype=float) - (c0 + c1 * duration_s + c2 * duration_s**2)
    residual_velocity = np.asarray(end_velocity, dtype=float) - (c1 + 2.0 * c2 * duration_s)
    residual_acceleration = np.asarray(end_acceleration, dtype=float) - 2.0 * c2
    duration_squared = duration_s * duration_s
    c3 = (
        10.0 * residual_position
        - 4.0 * duration_s * residual_velocity
        + 0.5 * duration_squared * residual_acceleration
    ) / duration_s**3
    c4 = (
        -15.0 * residual_position
        + 7.0 * duration_s * residual_velocity
        - duration_squared * residual_acceleration
    ) / duration_s**4
    c5 = (
        6.0 * residual_position
        - 3.0 * duration_s * residual_velocity
        + 0.5 * duration_squared * residual_acceleration
    ) / duration_s**5
    return np.vstack((c0, c1, c2, c3, c4, c5))


def _evaluate_polynomial(coefficients: np.ndarray, time_s: float) -> tuple[np.ndarray, ...]:
    time_s = float(time_s)
    if len(coefficients) == 8:
        derivatives = []
        for order in range(4):
            result = np.zeros(3)
            for k in range(7, order - 1, -1):
                multiplier = 1
                for factor in range(order):
                    multiplier *= k - factor
                result = result * time_s + multiplier * coefficients[k]
            derivatives.append(result)
        return tuple(derivatives)
    powers = np.array((1.0, time_s, time_s**2, time_s**3, time_s**4, time_s**5))
    value = powers @ coefficients
    velocity = (
        np.array((0.0, 1.0, 2.0 * time_s, 3.0 * time_s**2, 4.0 * time_s**3, 5.0 * time_s**4))
        @ coefficients
    )
    acceleration = (
        np.array((0.0, 0.0, 2.0, 6.0 * time_s, 12.0 * time_s**2, 20.0 * time_s**3)) @ coefficients
    )
    jerk = np.array((0.0, 0.0, 0.0, 6.0, 24.0 * time_s, 60.0 * time_s**2)) @ coefficients
    return value, velocity, acceleration, jerk


def _jexp_directional_derivative(rotation_vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    scale = max(1.0, float(np.linalg.norm(direction)))
    epsilon = 1e-6 / scale
    return (
        pin.Jexp3(rotation_vector + epsilon * direction)
        - pin.Jexp3(rotation_vector - epsilon * direction)
    ) / (2.0 * epsilon)


def _rotation_motion(
    rotation_vector: np.ndarray,
    rotation_velocity: np.ndarray,
    rotation_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = pin.Jexp3(rotation_vector)
    angular_velocity = jacobian @ rotation_velocity
    jacobian_rate = _jexp_directional_derivative(rotation_vector, rotation_velocity)
    angular_acceleration = jacobian @ rotation_acceleration + jacobian_rate @ rotation_velocity
    return angular_velocity, angular_acceleration


def _coordinate_rotation_boundary(
    rotation_vector: np.ndarray,
    angular_velocity: np.ndarray,
    angular_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(angular_velocity) and not np.any(angular_acceleration):
        return np.zeros(3), np.zeros(3)
    jacobian = pin.Jexp3(rotation_vector)
    coordinate_velocity = np.linalg.solve(jacobian, angular_velocity)
    jacobian_rate = _jexp_directional_derivative(rotation_vector, coordinate_velocity)
    coordinate_acceleration = np.linalg.solve(
        jacobian,
        angular_acceleration - jacobian_rate @ coordinate_velocity,
    )
    return coordinate_velocity, coordinate_acceleration


class C2TargetMotionPlanner:
    """Online quintic reference governor with finite-time, zero-speed arrival."""

    _REQUEST_TOLERANCE = 1e-10
    _DIRECT_POSITION_TOLERANCE_M = 2e-4
    _DIRECT_ORIENTATION_TOLERANCE_RAD = 2e-3
    _SAMPLE_COUNT = 33

    def __init__(
        self,
        limits: CartesianMotionLimits,
        *,
        continuous_tracking: bool = False,
        translation_deadband_m: float = 0.0,
        orientation_deadband_rad: float = 0.0,
        stationary_dwell_s: float = 0.0,
        limit_check_runtime: str = "auto_compatible_v1",
    ) -> None:
        self._boundary_cache = None
        self.diagnostics = {"replans": 0, "limit_checks": 0, "boundary_builds": 0, "coefficient_builds": 0}
        self.limits = limits
        self._limit_vector = np.array(
            (
                limits.linear_velocity_m_s,
                limits.linear_acceleration_m_s2,
                limits.linear_jerk_m_s3,
                limits.angular_velocity_rad_s,
                limits.angular_acceleration_rad_s2,
                limits.angular_jerk_rad_s3,
            )
        )
        self.continuous_tracking = continuous_tracking
        self.translation_deadband_m = float(translation_deadband_m)
        self.orientation_deadband_rad = float(orientation_deadband_rad)
        self.stationary_dwell_s = float(stationary_dwell_s)
        self.limit_check_runtime = str(limit_check_runtime)
        if self.limit_check_runtime not in {"auto_compatible_v1", "native_full_so3_v1"}:
            raise ValueError("unknown target limit-check runtime")
        if self.limit_check_runtime == "native_full_so3_v1" and (
            _NATIVE_KERNEL_ABI < 3 or _native_quintic_peak_norms is None
        ):
            raise RuntimeError("native_full_so3_v1 requires collision kernel ABI 3")
        if (
            not np.isfinite(self.translation_deadband_m)
            or self.translation_deadband_m < 0.0
            or not np.isfinite(self.orientation_deadband_rad)
            or self.orientation_deadband_rad < 0.0
            or not np.isfinite(self.stationary_dwell_s)
            or self.stationary_dwell_s < 0.0
        ):
            raise ValueError("Cartesian request deadbands must be finite and non-negative")
        self._motion: CartesianMotion | None = None
        self._segment: _QuinticSegment | None = None
        self._previous_request: pin.SE3 | None = None
        self._previous_secant_world = np.zeros(6)
        self._previous_velocity_world = np.zeros(6)
        self._elapsed_since_distinct_request_s = 0.0
        self._previous_distinct_interval_s: float | None = None
        self.request_is_moving = False

    @property
    def initialized(self) -> bool:
        return self._motion is not None

    @property
    def settled(self) -> bool:
        return bool(
            self._segment is None
            and self._motion is not None
            and np.linalg.norm(self._motion.body_twist) <= 1e-8
            and np.linalg.norm(self._motion.body_acceleration) <= 1e-8
        )

    def reset(
        self,
        pose: pin.SE3,
        body_twist: np.ndarray | None = None,
        body_acceleration: np.ndarray | None = None,
    ) -> None:
        twist = np.zeros(6) if body_twist is None else _validate_vector(body_twist, "target twist")
        acceleration = (
            np.zeros(6)
            if body_acceleration is None
            else _validate_vector(body_acceleration, "target acceleration")
        )
        self._motion = CartesianMotion(
            _validate_pose(pose),
            twist,
            acceleration,
            np.zeros(6),
        )
        self._segment = None
        self.request_is_moving = False
        self._previous_request = pose.copy()
        self._previous_secant_world = np.zeros(6)
        self._previous_velocity_world = np.zeros(6)
        self._elapsed_since_distinct_request_s = 0.0
        self._previous_distinct_interval_s = None

    @staticmethod
    def _copy_segment(segment: _QuinticSegment | None) -> _QuinticSegment | None:
        if segment is None:
            return None
        return _QuinticSegment(
            segment.translation_coefficients.copy(),
            segment.rotation_coefficients.copy(),
            segment.start_rotation.copy(),
            segment.duration_s,
            segment.elapsed_s,
            segment.request.copy(),
            segment.destination.copy(),
            segment.motion_limited,
        )

    def motion(self) -> CartesianMotion:
        if self._motion is None:
            raise RuntimeError("reset the Cartesian planner before reading its state")
        return self._motion.copy()

    def snapshot(self) -> _PlannerSnapshot:
        return _PlannerSnapshot(
            None if self._motion is None else self._motion.copy(),
            self._copy_segment(self._segment),
            None if self._previous_request is None else self._previous_request.copy(),
            self._previous_secant_world.copy(),
            self._previous_velocity_world.copy(),
            self._elapsed_since_distinct_request_s,
            self._previous_distinct_interval_s,
            self.request_is_moving,
        )

    def restore(self, snapshot: _PlannerSnapshot) -> None:
        self._motion = None if snapshot.motion is None else snapshot.motion.copy()
        self._segment = self._copy_segment(snapshot.segment)
        self._previous_request = (
            None if snapshot.previous_request is None else snapshot.previous_request.copy()
        )
        self._previous_secant_world = snapshot.previous_secant_world.copy()
        self._previous_velocity_world = snapshot.previous_velocity_world.copy()
        self._elapsed_since_distinct_request_s = snapshot.elapsed_since_distinct_request_s
        self._previous_distinct_interval_s = snapshot.previous_distinct_interval_s
        self.request_is_moving = snapshot.request_is_moving

    def _estimate_request(self, pose: pin.SE3, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Estimate derivatives of sampled poses, never of the filtered output.

        A backward quadratic estimate removes the half-tick lag of a secant.
        Jumps and stationary handles request a stop; the trajectory generator,
        not this estimator, is responsible for bounded physical transitions.
        """
        self._elapsed_since_distinct_request_s += dt
        previous = self._previous_request
        if previous is None:
            self._previous_request = pose.copy()
            self._elapsed_since_distinct_request_s = 0.0
            return np.zeros(6), np.zeros(6)
        displacement = np.concatenate(
            (
                pose.translation - previous.translation,
                previous.rotation @ pin.log3(previous.rotation.T @ pose.rotation),
            )
        )
        linear, angular = np.linalg.norm(displacement[:3]), np.linalg.norm(displacement[3:])
        if (
            linear <= max(1e-8, self.translation_deadband_m)
            and angular <= max(1e-8, self.orientation_deadband_rad)
        ):
            if self._elapsed_since_distinct_request_s <= self.stationary_dwell_s:
                return (
                    np.concatenate(
                        (
                            pose.rotation.T @ self._previous_velocity_world[:3],
                            pose.rotation.T @ self._previous_velocity_world[3:],
                        )
                    ),
                    np.zeros(6),
                )
            return np.zeros(6), np.zeros(6)
        interval = self._elapsed_since_distinct_request_s
        jumped = (
            linear > 2.0 * self.limits.linear_velocity_m_s * interval
            or angular > 2.0 * self.limits.angular_velocity_rad_s * interval
        )
        self._previous_request = pose.copy()
        self._elapsed_since_distinct_request_s = 0.0
        if jumped:
            self._previous_secant_world = np.zeros(6)
            self._previous_velocity_world = np.zeros(6)
            self._previous_distinct_interval_s = None
            return np.zeros(6), np.zeros(6)
        secant = displacement / interval
        previous_interval = self._previous_distinct_interval_s
        first_sample = previous_interval is None
        velocity = (
            secant
            if first_sample
            else ((2.0 * interval + previous_interval) / (interval + previous_interval))
            * secant
            - (interval / (interval + previous_interval)) * self._previous_secant_world
        )
        acceleration = (
            np.zeros(6)
            if first_sample
            else 2.0 * (secant - self._previous_secant_world) / (
                interval + previous_interval
            )
        )
        for segment, vmax, amax in (
            (slice(0, 3), self.limits.linear_velocity_m_s, self.limits.linear_acceleration_m_s2),
            (
                slice(3, 6),
                self.limits.angular_velocity_rad_s,
                self.limits.angular_acceleration_rad_s2,
            ),
        ):
            velocity[segment] *= min(1.0, vmax / max(1e-12, np.linalg.norm(velocity[segment])))
            acceleration[segment] *= min(
                1.0, amax / max(1e-12, np.linalg.norm(acceleration[segment]))
            )
        self._previous_secant_world = secant
        self._previous_velocity_world = velocity
        self._previous_distinct_interval_s = interval
        linear_velocity = pose.rotation.T @ velocity[:3]
        angular_velocity = pose.rotation.T @ velocity[3:]
        return (
            np.concatenate((linear_velocity, angular_velocity)),
            np.concatenate(
                (
                    pose.rotation.T @ acceleration[:3]
                    - np.cross(angular_velocity, linear_velocity),
                    pose.rotation.T @ acceleration[3:],
                )
            ),
        )

    def utilization(self) -> CartesianMotionUtilization:
        motion = self.motion()
        return CartesianMotionUtilization(
            linear_velocity=float(
                np.linalg.norm(motion.body_twist[:3]) / self.limits.linear_velocity_m_s
            ),
            linear_acceleration=float(
                np.linalg.norm(motion.body_acceleration[:3]) / self.limits.linear_acceleration_m_s2
            ),
            linear_jerk=float(np.linalg.norm(motion.body_jerk[:3]) / self.limits.linear_jerk_m_s3),
            angular_velocity=float(
                np.linalg.norm(motion.body_twist[3:]) / self.limits.angular_velocity_rad_s
            ),
            angular_acceleration=float(
                np.linalg.norm(motion.body_acceleration[3:])
                / self.limits.angular_acceleration_rad_s2
            ),
            angular_jerk=float(
                np.linalg.norm(motion.body_jerk[3:]) / self.limits.angular_jerk_rad_s3
            ),
        )

    @staticmethod
    def _request_motion(
        pose: pin.SE3,
        body_twist: np.ndarray | None,
        body_acceleration: np.ndarray | None,
    ) -> CartesianMotion:
        return CartesianMotion(
            _validate_pose(pose),
            np.zeros(6) if body_twist is None else _validate_vector(body_twist, "target twist"),
            (
                np.zeros(6)
                if body_acceleration is None
                else _validate_vector(body_acceleration, "target acceleration")
            ),
            np.zeros(6),
        )

    @classmethod
    def _same_request(cls, first: CartesianMotion, second: CartesianMotion) -> bool:
        return bool(
            np.linalg.norm(first.pose.translation - second.pose.translation)
            <= cls._REQUEST_TOLERANCE
            and np.linalg.norm(pin.log3(first.pose.rotation.T @ second.pose.rotation))
            <= cls._REQUEST_TOLERANCE
            and np.linalg.norm(first.body_twist - second.body_twist) <= cls._REQUEST_TOLERANCE
            and np.linalg.norm(first.body_acceleration - second.body_acceleration)
            <= cls._REQUEST_TOLERANCE
        )

    def _direct_transition_is_valid(self, request: CartesianMotion, dt: float) -> bool:
        assert self._motion is not None
        current = self._motion
        relative = current.pose.actInv(request.pose)
        displacement = pin.log(relative).vector
        average_twist = 0.5 * (current.body_twist + request.body_twist)
        predicted_displacement = average_twist * dt
        jerk = (request.body_acceleration - current.body_acceleration) / dt
        if self.continuous_tracking:
            # A velocity jump must not bypass acceleration limits simply
            # because both callers labelled its acceleration as zero.
            current_world = np.concatenate(
                (
                    current.pose.rotation @ current.body_twist[:3],
                    current.pose.rotation @ current.body_twist[3:],
                )
            )
            request_world = np.concatenate(
                (
                    request.pose.rotation @ request.body_twist[:3],
                    request.pose.rotation @ request.body_twist[3:],
                )
            )
            acceleration = (request_world - current_world) / dt
            if (
                np.linalg.norm(acceleration[:3]) > self.limits.linear_acceleration_m_s2 + 1e-9
                or np.linalg.norm(acceleration[3:]) > self.limits.angular_acceleration_rad_s2 + 1e-9
            ):
                return False
        return bool(
            np.linalg.norm(displacement[:3] - predicted_displacement[:3])
            <= self._DIRECT_POSITION_TOLERANCE_M
            and np.linalg.norm(displacement[3:] - predicted_displacement[3:])
            <= self._DIRECT_ORIENTATION_TOLERANCE_RAD
            and np.linalg.norm(request.body_twist[:3]) <= self.limits.linear_velocity_m_s + 1e-9
            and np.linalg.norm(request.body_twist[3:]) <= self.limits.angular_velocity_rad_s + 1e-9
            and np.linalg.norm(request.body_acceleration[:3])
            <= self.limits.linear_acceleration_m_s2 + 1e-9
            and np.linalg.norm(request.body_acceleration[3:])
            <= self.limits.angular_acceleration_rad_s2 + 1e-9
            and np.linalg.norm(jerk[:3]) <= self.limits.linear_jerk_m_s3 + 1e-9
            and np.linalg.norm(jerk[3:]) <= self.limits.angular_jerk_rad_s3 + 1e-9
        )

    def _build_coefficients(
        self,
        destination: CartesianMotion,
        duration_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._motion is not None
        self.diagnostics["coefficient_builds"] += 1
        cached = self._boundary_cache
        if cached is not None and cached[0] is self._motion and cached[1] is destination:
            boundaries, start_rotation = cached[2:]
        else:
            self.diagnostics["boundary_builds"] += 1
            start = self._motion
            start_rotation = start.pose.rotation.copy()
            end_rotation_vector = pin.log3(start_rotation.T @ destination.pose.rotation)
            end_velocity, end_acceleration = _coordinate_rotation_boundary(
                end_rotation_vector, destination.body_twist[3:], destination.body_acceleration[3:])
            boundaries = np.empty((6, 6))
            boundaries[0] = np.r_[start.pose.translation, np.zeros(3)]
            boundaries[1] = np.r_[start_rotation @ start.body_twist[:3], start.body_twist[3:]]
            boundaries[2] = np.r_[start_rotation @ (
                start.body_acceleration[:3] + np.cross(start.body_twist[3:], start.body_twist[:3])),
                start.body_acceleration[3:]]
            boundaries[3] = np.r_[destination.pose.translation, end_rotation_vector]
            boundaries[4] = np.r_[destination.pose.rotation @ destination.body_twist[:3], end_velocity]
            boundaries[5] = np.r_[destination.pose.rotation @ (
                destination.body_acceleration[:3]
                + np.cross(destination.body_twist[3:], destination.body_twist[:3])), end_acceleration]
            self._boundary_cache = (self._motion, destination, boundaries, start_rotation)
        coefficients = (_native_coefficients(boundaries, duration_s) if _native_coefficients is not None
                        else _quintic_coefficients(*boundaries, duration_s))
        return np.ascontiguousarray(coefficients[:, :3]), np.ascontiguousarray(coefficients[:, 3:]), start_rotation

    def _within_limits(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        duration_s: float,
    ) -> bool:
        self.diagnostics["limit_checks"] += 1
        if _native_quintic_limits is not None and _NATIVE_KERNEL_ABI >= 3:
            return bool(_native_quintic_limits(
                translation, rotation, duration_s, self._limit_vector, self._SAMPLE_COUNT))
        if _native_quintic_peak_norms is not None:
            peaks = np.asarray(
                _native_quintic_peak_norms(
                    translation, rotation, duration_s, self._SAMPLE_COUNT
                ),
                dtype=float,
            )
            if peaks.shape != (6,) or not np.all(np.isfinite(peaks)):
                raise RuntimeError("native quintic kernel returned invalid peak norms")
            return bool(np.all(peaks <= self._limit_vector * (1.0 + 1e-6)))
        if self.limit_check_runtime == "native_full_so3_v1":
            raise RuntimeError("native_full_so3_v1 kernel became unavailable")
        if _native_quintic_limits is not None:
            native_result = _native_quintic_limits(
                translation,
                rotation,
                duration_s,
                self._limit_vector,
                self._SAMPLE_COUNT,
            )
            if native_result is not None:
                return bool(native_result)
        times = np.linspace(0.0, duration_s, self._SAMPLE_COUNT)
        t = times[:, None]
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        linear_velocity = (
            translation[1]
            + 2.0 * t * translation[2]
            + 3.0 * t2 * translation[3]
            + 4.0 * t3 * translation[4]
            + 5.0 * t4 * translation[5]
        )
        linear_acceleration = (
            2.0 * translation[2]
            + 6.0 * t * translation[3]
            + 12.0 * t2 * translation[4]
            + 20.0 * t3 * translation[5]
        )
        linear_jerk = (
            6.0 * translation[3] + 24.0 * t * translation[4] + 60.0 * t2 * translation[5]
        )
        axis = rotation[int(np.argmax(np.linalg.norm(rotation, axis=1)))]
        collinear = np.max(np.linalg.norm(np.cross(rotation, axis), axis=1)) <= 1e-12 * max(
            1.0, np.linalg.norm(axis) ** 2
        )
        if np.max(np.abs(rotation)) <= 1e-14:
            angular_velocity = np.zeros_like(linear_velocity)
            angular_acceleration = np.zeros_like(linear_acceleration)
            angular_jerk = np.zeros_like(linear_jerk)
        elif collinear:
            # Fixed-axis rotations commute: SO(3) derivatives are exactly the
            # polynomial derivatives, so no per-sample Jacobians are needed.
            angular_velocity = (
                rotation[1]
                + 2.0 * t * rotation[2]
                + 3.0 * t2 * rotation[3]
                + 4.0 * t3 * rotation[4]
                + 5.0 * t4 * rotation[5]
            )
            angular_acceleration = (
                2.0 * rotation[2]
                + 6.0 * t * rotation[3]
                + 12.0 * t2 * rotation[4]
                + 20.0 * t3 * rotation[5]
            )
            angular_jerk = 6.0 * rotation[3] + 24.0 * t * rotation[4] + 60.0 * t2 * rotation[5]
        else:
            angular_velocity_values: list[np.ndarray] = []
            angular_acceleration_values: list[np.ndarray] = []
            for sample_time in times:
                rotation_vector, rotation_velocity, rotation_acceleration, _ = _evaluate_polynomial(
                    rotation, sample_time
                )
                omega, alpha = _rotation_motion(
                    rotation_vector,
                    rotation_velocity,
                    rotation_acceleration,
                )
                angular_velocity_values.append(omega)
                angular_acceleration_values.append(alpha)
            angular_velocity = np.asarray(angular_velocity_values)
            angular_acceleration = np.asarray(angular_acceleration_values)
            angular_jerk = np.gradient(
                angular_acceleration,
                times,
                axis=0,
                edge_order=2,
            )
        checks = (
            (linear_velocity, self.limits.linear_velocity_m_s),
            (linear_acceleration, self.limits.linear_acceleration_m_s2),
            (linear_jerk, self.limits.linear_jerk_m_s3),
            (angular_velocity, self.limits.angular_velocity_rad_s),
            (angular_acceleration, self.limits.angular_acceleration_rad_s2),
            (angular_jerk, self.limits.angular_jerk_rad_s3),
        )
        return all(
            float(np.max(np.linalg.norm(np.asarray(values), axis=1)))
            <= limit * (1.0 + 1e-6)
            for values, limit in checks
        )

    def _new_segment(self, destination: CartesianMotion, dt: float) -> _QuinticSegment:
        self.diagnostics["replans"] += 1
        request = destination
        moving = (
            np.linalg.norm(request.body_twist) > 1e-12
            or np.linalg.norm(request.body_acceleration) > 1e-12
        )
        if self.continuous_tracking and moving:
            # The stored reference is at t-dt, whereas the incoming sample is at
            # t. Search an interception duration using that exact time origin.
            # Rest-to-rest duration bounds are inappropriate for a moving state.
            assert self._motion is not None
            delta_twist = request.body_twist - self._motion.body_twist
            duration = max(
                dt,
                np.sqrt(2.0 * np.linalg.norm(delta_twist[:3]) / self.limits.linear_jerk_m_s3),
                np.sqrt(2.0 * np.linalg.norm(delta_twist[3:]) / self.limits.angular_jerk_rad_s3),
            )
            for _ in range(12):
                if duration > 0.35:
                    break
                destination = self._predict_tracking_destination(request, max(0.0, duration - dt))
                translation, rotation, start_rotation = self._build_coefficients(
                    destination, duration
                )
                if self._within_limits(translation, rotation, duration):
                    return _QuinticSegment(
                        translation,
                        rotation,
                        start_rotation,
                        duration,
                        0.0,
                        request.copy(),
                        destination.copy(),
                    )
                duration *= 1.25
            # An aggressive/noisy derivative can make interception inadmissible.
            # Project to a bounded stop at the requested pose, preserving the
            # current C2 state. This is reference limiting, not an IK fallback.
            stopped = CartesianMotion(request.pose.copy(), np.zeros(6), np.zeros(6), np.zeros(6))
            destination, duration, translation, rotation, start_rotation = self._bounded_segment(
                stopped, dt
            )
            return _QuinticSegment(
                translation,
                rotation,
                start_rotation,
                duration,
                0.0,
                request.copy(),
                destination.copy(),
                motion_limited=True,
            )
        if (
            np.linalg.norm(request.body_twist) > 1e-12
            or np.linalg.norm(request.body_acceleration) > 1e-12
        ):
            # A continuously moving reference would otherwise stay one planning
            # horizon behind forever. Aim at its constant-acceleration lookahead,
            # then recompute a bounded segment to that interception point.
            horizon = self._minimum_duration(request, dt)
            for _ in range(3):
                destination = self._predict_destination(request, min(horizon, 0.75))
                horizon = self._minimum_duration(destination, dt)
        destination, duration, translation, rotation, start_rotation = self._bounded_segment(
            destination, dt
        )
        return _QuinticSegment(
            translation,
            rotation,
            start_rotation,
            duration,
            0.0,
            request.copy(),
            destination.copy(),
        )

    def _predict_tracking_destination(
        self, request: CartesianMotion, horizon: float
    ) -> CartesianMotion:
        """Extrapolate pose AND derivatives in a common world frame."""
        rotation = request.pose.rotation
        velocity = rotation @ request.body_twist[:3]
        acceleration = rotation @ (
            request.body_acceleration[:3] + np.cross(request.body_twist[3:], request.body_twist[:3])
        )
        angular = request.body_twist[3:]
        alpha = request.body_acceleration[3:]
        predicted_rotation = rotation @ pin.exp3(angular * horizon + 0.5 * alpha * horizon**2)
        predicted_velocity = velocity + acceleration * horizon
        predicted_angular = angular + alpha * horizon
        for value, limit in (
            (predicted_velocity, self.limits.linear_velocity_m_s),
            (predicted_angular, self.limits.angular_velocity_rad_s),
        ):
            value *= min(1.0, limit / max(1e-12, np.linalg.norm(value)))
        body_velocity = predicted_rotation.T @ predicted_velocity
        return CartesianMotion(
            pin.SE3(
                predicted_rotation,
                request.pose.translation + velocity * horizon + 0.5 * acceleration * horizon**2,
            ),
            np.concatenate((body_velocity, predicted_angular)),
            np.concatenate(
                (
                    predicted_rotation.T @ acceleration
                    - np.cross(predicted_angular, body_velocity),
                    alpha,
                )
            ),
            np.zeros(6),
        )

    @staticmethod
    def _predict_destination(request: CartesianMotion, horizon_s: float) -> CartesianMotion:
        horizon_s = float(horizon_s)
        body_displacement = (
            request.body_twist * horizon_s + 0.5 * request.body_acceleration * horizon_s**2
        )
        translation = request.pose.translation + request.pose.rotation @ body_displacement[:3]
        rotation = request.pose.rotation @ pin.exp3(body_displacement[3:])
        return CartesianMotion(
            pin.SE3(rotation, translation),
            request.body_twist.copy(),
            request.body_acceleration.copy(),
            np.zeros(6),
        )

    def _bounded_segment(
        self,
        destination: CartesianMotion,
        dt: float,
    ) -> tuple[CartesianMotion, float, np.ndarray, np.ndarray, np.ndarray]:
        duration = self._minimum_duration(destination, dt)
        lower = 0.0
        upper = duration
        accepted = None
        for _ in range(16):
            translation, rotation, start_rotation = self._build_coefficients(destination, upper)
            if self._within_limits(translation, rotation, upper):
                accepted = (translation, rotation, start_rotation)
                break
            lower = upper
            upper *= 1.25
        else:
            raise TargetTrajectoryError("could not construct a bounded Cartesian target trajectory")
        # The analytical rest-to-rest lower bound is already time-optimal for
        # ordinary viewer steps. Only refine after a moving-boundary expansion;
        # this keeps the first safety-shaped IK tick well inside its 20 ms budget.
        if lower > 0.0:
            for _ in range(6):
                candidate = 0.5 * (lower + upper)
                translation, rotation, start_rotation = self._build_coefficients(
                    destination, candidate
                )
                if self._within_limits(translation, rotation, candidate):
                    upper = candidate
                    accepted = (translation, rotation, start_rotation)
                else:
                    lower = candidate
        assert accepted is not None
        return destination, upper, *accepted

    def _minimum_duration(self, destination: CartesianMotion, dt: float) -> float:
        assert self._motion is not None
        distance = float(
            np.linalg.norm(destination.pose.translation - self._motion.pose.translation)
        )
        angle = float(
            np.linalg.norm(pin.log3(self._motion.pose.rotation.T @ destination.pose.rotation))
        )
        return max(
            dt,
            1.875 * distance / self.limits.linear_velocity_m_s,
            np.sqrt(5.7735026919 * distance / self.limits.linear_acceleration_m_s2),
            np.cbrt(60.0 * distance / self.limits.linear_jerk_m_s3),
            1.875 * angle / self.limits.angular_velocity_rad_s,
            np.sqrt(5.7735026919 * angle / self.limits.angular_acceleration_rad_s2),
            np.cbrt(60.0 * angle / self.limits.angular_jerk_rad_s3),
        )

    @staticmethod
    def _sample_segment(segment: _QuinticSegment, time_s: float) -> CartesianMotion:
        translation, linear_velocity_world, linear_acceleration_world, linear_jerk_world = (
            _evaluate_polynomial(segment.translation_coefficients, time_s)
        )
        rotation_vector, rotation_velocity, rotation_acceleration, rotation_jerk = (
            _evaluate_polynomial(segment.rotation_coefficients, time_s)
        )
        rotation = segment.start_rotation @ pin.exp3(rotation_vector)
        angular_velocity, angular_acceleration = _rotation_motion(
            rotation_vector,
            rotation_velocity,
            rotation_acceleration,
        )
        body_linear_velocity = rotation.T @ linear_velocity_world
        body_linear_acceleration = rotation.T @ linear_acceleration_world - np.cross(
            angular_velocity, body_linear_velocity
        )
        body_linear_jerk = rotation.T @ linear_jerk_world
        return CartesianMotion(
            pin.SE3(rotation, translation),
            np.concatenate((body_linear_velocity, angular_velocity)),
            np.concatenate((body_linear_acceleration, angular_acceleration)),
            np.concatenate((body_linear_jerk, rotation_jerk)),
        )

    def step(
        self,
        desired_pose: pin.SE3,
        dt: float,
        *,
        desired_body_twist: np.ndarray | None = None,
        desired_body_acceleration: np.ndarray | None = None,
    ) -> tuple[CartesianMotion, bool]:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Cartesian planner timestep must be finite and positive")
        request = self._request_motion(
            desired_pose,
            desired_body_twist,
            desired_body_acceleration,
        )
        if self.continuous_tracking:
            # Keep sampled-pose history current even while the caller supplies
            # derivatives, so switching back to pose-only input cannot see a jump.
            twist, acceleration = self._estimate_request(request.pose, dt)
            if desired_body_twist is None and desired_body_acceleration is None:
                # Coalesce sub-deadband device jitter around the most recent
                # distinct pose. The bounded spatial error is smaller than the
                # profile accuracy tolerance and avoids issuing real joint
                # motion for every noisy input sample.
                accepted_pose = (
                    request.pose.copy()
                    if self._previous_request is None
                    else self._previous_request.copy()
                )
                request = CartesianMotion(
                    accepted_pose, twist, acceleration, np.zeros(6)
                )
        self.request_is_moving = bool(
            np.linalg.norm(request.body_twist) > 1e-12
            or np.linalg.norm(request.body_acceleration) > 1e-12
        )
        if self._motion is None:
            self.reset(request.pose, request.body_twist, request.body_acceleration)
            return self.motion(), False

        if (self._segment is None or self.continuous_tracking) and self._direct_transition_is_valid(
            request, dt
        ):
            previous_acceleration = self._motion.body_acceleration
            self._motion = CartesianMotion(
                request.pose,
                request.body_twist,
                request.body_acceleration,
                (request.body_acceleration - previous_acceleration) / dt,
            )
            self._segment = None
            return self.motion(), False

        if self._segment is None or not self._same_request(self._segment.request, request):
            self._segment = self._new_segment(request, dt)

        self._segment.elapsed_s = min(
            self._segment.duration_s,
            self._segment.elapsed_s + dt,
        )
        motion_limited = self._segment.motion_limited
        if self._segment.elapsed_s >= self._segment.duration_s - 1e-12:
            self._motion = self._segment.destination.copy()
            self._segment = None
        else:
            self._motion = self._sample_segment(self._segment, self._segment.elapsed_s)
        moving_request = bool(
            np.linalg.norm(request.body_twist) > 1e-12
            or np.linalg.norm(request.body_acceleration) > 1e-12
        )
        if moving_request:
            position_gap = float(
                np.linalg.norm(self._motion.pose.translation - request.pose.translation)
            )
            orientation_gap = float(
                np.linalg.norm(pin.log3(request.pose.rotation.T @ self._motion.pose.rotation))
            )
            materially_limited = bool(
                position_gap > self.limits.linear_velocity_m_s * dt + 1e-9
                or orientation_gap > self.limits.angular_velocity_rad_s * dt + 1e-9
            )
        else:
            materially_limited = True
        return self.motion(), materially_limited or motion_limited


def _tracking_jerk_with_braking_reserve(jerk, acceleration, velocity, jmax, amax, vmax, dt):
    """Limit a whole discrete step, including the future acceleration-to-zero stop.

    Clamping next acceleration against *current* speed misses the velocity
    gained during this tick. Once that consumes the braking reserve, even a
    small fresh target can leave no feasible next segment. Blend toward a
    bounded acceleration brake before entering that state; preserve C2 history.
    """
    jcap, acap, vcap = .95*jmax, .95*amax, .95*vmax
    next_a = acceleration + jerk*dt
    next_a *= min(1., acap/max(1e-12, np.linalg.norm(next_a)))
    proposed = (next_a-acceleration)/dt
    proposed *= min(1., jcap/max(1e-12, np.linalg.norm(proposed)))

    def viable(candidate):
        end_a = acceleration + candidate*dt
        end_v = velocity + acceleration*dt + .5*candidate*dt**2
        # Norm bound on additional speed while removing acceleration at jcap.
        return (np.linalg.norm(end_a) <= acap+1e-12
                and np.linalg.norm(end_v) + np.dot(end_a, end_a)/(2*jcap) <= vcap+1e-12)

    if viable(proposed):
        return proposed
    brake = -acceleration/dt
    brake *= min(1., jcap/max(1e-12, np.linalg.norm(brake)))
    if not viable(brake):
        # A supplied initial state may already lack this conservative reserve.
        # The caller's full continuous-time limit check remains mandatory.
        return brake
    low, high = 0., 1.
    for _ in range(20):
        middle = .5*(low+high)
        if viable(brake + middle*(proposed-brake)):
            low = middle
        else:
            high = middle
    return brake + low*(proposed-brake)


class ContinuousTargetMotionPlanner(C2TargetMotionPlanner):
    """V4: causal source secants and one-tick C2 reference integration.

    Moving requests drive bounded jerk without extrapolating the input pose.
    Stationary requests use exact-arrival segments. Infeasible reference bounds
    invoke the core's constrained recovery, never a silent zero-speed fallback.
    The default v4 bandwidth is preserved. Alternate bandwidths are explicit
    profile-bound tuning; the 50 Hz sampled loop is restricted to 2--20 s^-1.
    Low-acceleration profiles need lower bandwidth.
    """

    TRACKING_BANDWIDTH_S = 12.0

    def __init__(self, *args, tracking_bandwidth_s_inv=12.0, source_velocity_gain=1.0, continuous_arrival=False,
                 source_acceleration_gain=0.0, source_acceleration_tau_s=0.06, **kwargs):
        if not np.isfinite(tracking_bandwidth_s_inv) or not 2.0 <= tracking_bandwidth_s_inv <= 20.0:
            raise ValueError("tracking bandwidth must be in [2, 20] s^-1")
        self.TRACKING_BANDWIDTH_S = float(tracking_bandwidth_s_inv)
        if not np.isfinite(source_velocity_gain) or not 0 <= source_velocity_gain <= 1:
            raise ValueError("source velocity gain must be in [0, 1]")
        self.source_velocity_gain = float(source_velocity_gain)
        self.continuous_arrival = bool(continuous_arrival)
        if not np.isfinite(source_acceleration_gain) or not 0 <= source_acceleration_gain <= 1:
            raise ValueError("source acceleration gain must be in [0, 1]")
        if not np.isfinite(source_acceleration_tau_s) or not 0.01 <= source_acceleration_tau_s <= 0.5:
            raise ValueError("source acceleration filter must be in [0.01, 0.5] s")
        self.source_acceleration_gain = float(source_acceleration_gain)
        self.source_acceleration_tau_s = float(source_acceleration_tau_s)
        self._has_explicit_acceleration = False
        super().__init__(*args, **kwargs)
        self._sample = self._last_sample = None
        self.input_event = "initial"
        self._duplicate_elapsed = 0.
        self._reset_source_acceleration()

    def _reset_source_acceleration(self):
        # World linear velocity/acceleration and body angular velocity/acceleration of the source.
        self._source_velocity = None
        self._source_acceleration = np.zeros(6)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._sample = self._last_sample = None
        self.input_event = "initial"
        self._duplicate_elapsed = 0.
        self._reset_source_acceleration()

    def snapshot(self):
        return (super().snapshot(), self._last_sample, self.input_event, self._duplicate_elapsed,
                None if self._source_velocity is None else self._source_velocity.copy(),
                self._source_acceleration.copy())

    def restore(self, snapshot):
        super().restore(snapshot[0])
        (self._last_sample, self.input_event, self._duplicate_elapsed,
         self._source_velocity, self._source_acceleration) = snapshot[1:]
        self._sample = None

    def _update_source_acceleration(self, pose, twist, dt):
        """Filtered world-frame source acceleration, updated once per control tick.

        Zero whenever the request is stationary or its history restarts.
        """
        if not self.source_acceleration_gain:
            return
        moving = np.linalg.norm(twist) > 1e-12
        velocity = np.r_[pose.rotation @ twist[:3], pose.rotation @ twist[3:]]
        if not moving or self._source_velocity is None:
            self._source_velocity = velocity if moving else None
            self._source_acceleration = np.zeros(6)
            return
        raw = (velocity - self._source_velocity) / dt
        blend = -np.expm1(-dt / self.source_acceleration_tau_s)
        self._source_acceleration = self._source_acceleration + blend * (raw - self._source_acceleration)
        self._source_velocity = velocity

    def step(self, desired_pose, dt, *, sample=None, **kwargs):
        self._sample = sample
        self._tick_dt = dt
        # Explicit trajectory acceleration is already a derivative of the
        # timestamped target. Do not replace it with a filtered pose secant.
        # The parent validates the vectors before constructing any segment.
        self._has_explicit_acceleration = kwargs.get('desired_body_acceleration') is not None
        try:
            return super().step(desired_pose, dt, **kwargs)
        finally:
            self._has_explicit_acceleration = False

    def _estimate_request(self, pose, dt):
        twist, acceleration = self._estimate_request_v4(pose, dt)
        self._update_source_acceleration(pose, twist, getattr(self, "_tick_dt", dt))
        return twist, acceleration

    def _estimate_request_v4(self, pose, dt):
        sample, previous = self._sample, self._last_sample
        if sample is not None:
            if (previous is not None and sample.epoch == previous.epoch
                    and sample.quality == previous.quality):
                if sample.sequence == previous.sequence:
                    self._duplicate_elapsed += dt
                    self.input_event = "repeated_packet"
                    if self._duplicate_elapsed > .1:
                        self.input_event = "stale_packet_braking"
                        return np.zeros(6), np.zeros(6)
                    # Continue the existing bounded segment. Duplicate packets
                    # do not invent a new zero-velocity observation.
                    return (np.concatenate((pose.rotation.T @ self._previous_velocity_world[:3],
                                            pose.rotation.T @ self._previous_velocity_world[3:])),
                            np.zeros(6))
                if sample.sequence < previous.sequence or sample.time_s <= previous.time_s:
                    raise TargetTrajectoryError("out_of_order_target_sample")
                dt = sample.time_s - previous.time_s
            else:
                self._previous_request = pose.copy()
                self._previous_velocity_world = np.zeros(6)
                self._elapsed_since_distinct_request_s = 0.
                self._previous_distinct_interval_s = None
                self._last_sample = sample
                self._duplicate_elapsed = 0.
                self.input_event = "initial" if previous is None else "epoch_reset"
                return np.zeros(6), np.zeros(6)
            self._last_sample = sample
            self._duplicate_elapsed = 0.
        else:
            self._last_sample = None
        # Secants are less sensitive to source jitter than backward quadratic
        # acceleration. The trajectory itself provides bounded acceleration.
        self._previous_distinct_interval_s = None
        twist, _ = super()._estimate_request(pose, dt)
        self.input_event = "moving" if np.linalg.norm(twist) > 1e-12 else "stationary"
        if self.input_event == "stationary":
            self._previous_velocity_world = np.zeros(6)
        return twist, np.zeros(6)

    def _new_segment(self, destination, dt):
        self.diagnostics["replans"] += 1
        assert self._motion is not None
        if self.continuous_arrival and not self.request_is_moving:
            try:
                return self._endpoint_segment(destination, dt)
            except TargetTrajectoryError:
                # Advance a bounded C2 segment instead of retrying identical history.
                # This is observable and still has to pass the complete SO(3) check.
                key = "arrival_continuations"
                self.diagnostics[key] = self.diagnostics.get(key, 0) + 1
        if self.request_is_moving or self.continuous_arrival:
            # One-tick C2 reference integration. Position error drives jerk;
            # source velocity supplies causal feedforward. No future input pose
            # is synthesized. Full native SO(3) bounds validate the segment.
            current = self._motion
            rotation = current.pose.rotation
            relative = rotation.T @ destination.pose.rotation
            world_v = rotation @ current.body_twist[:3]
            world_a = rotation @ (current.body_acceleration[:3]
                                 + np.cross(current.body_twist[3:], current.body_twist[:3]))
            position_error = destination.pose.translation - current.pose.translation
            rotation_error = pin.log3(relative)
            omega = self.TRACKING_BANDWIDTH_S
            source_linear = destination.pose.rotation @ destination.body_twist[:3]
            # Optional filtered source-acceleration feedforward (0 keeps the v4 law).
            source_acceleration = self._source_acceleration
            gain = self.source_acceleration_gain
            if gain and self._has_explicit_acceleration:
                twist, acceleration = destination.body_twist, destination.body_acceleration
                source_acceleration = np.r_[
                    destination.pose.rotation @ (acceleration[:3] + np.cross(twist[3:], twist[:3])),
                    destination.pose.rotation @ acceleration[3:]]
            linear_jerk = (omega**3 * position_error
                           + 3 * omega**2 * (self.source_velocity_gain * source_linear - world_v)
                           - 3 * omega * (world_a - gain * source_acceleration[:3]))
            angular_jerk = (omega**3 * rotation_error
                            + 3 * omega**2 * (self.source_velocity_gain * (relative @ destination.body_twist[3:])
                                              - current.body_twist[3:])
                            - 3 * omega * (current.body_acceleration[3:]
                                           - gain * rotation.T @ source_acceleration[3:]))
            for jerk, acceleration, velocity, jmax, amax, vmax in (
                (linear_jerk, world_a, world_v, self.limits.linear_jerk_m_s3,
                 self.limits.linear_acceleration_m_s2, self.limits.linear_velocity_m_s),
                (angular_jerk, current.body_acceleration[3:], current.body_twist[3:],
                 self.limits.angular_jerk_rad_s3, self.limits.angular_acceleration_rad_s2,
                 self.limits.angular_velocity_rad_s)):
                jerk[:] = _tracking_jerk_with_braking_reserve(
                    jerk, acceleration, velocity, jmax, amax, vmax, dt)
            translation = np.zeros((6, 3))
            translation[:4] = [current.pose.translation, world_v, world_a / 2, linear_jerk / 6]
            angular = np.zeros((6, 3))
            angular[:4] = [np.zeros(3), current.body_twist[3:], current.body_acceleration[3:] / 2,
                           angular_jerk / 6]
            if self._within_limits(translation, angular, dt):
                segment = _QuinticSegment(translation, angular, rotation.copy(), dt, 0.,
                                           destination.copy(), destination.copy())
                segment.destination = self._sample_segment(segment, dt)
                return segment
            raise TargetTrajectoryError("continuous_reference_bounds_exhausted")
        return self._endpoint_segment(destination, dt)

    def _endpoint_segment(self, destination, dt):
        # Stationary requests retain bounded arrival and exact final position.
        duration = dt
        for _ in range(25):
            translation, rotation, start_rotation = self._build_coefficients(destination, duration)
            if self._within_limits(translation, rotation, duration):
                return _QuinticSegment(translation, rotation, start_rotation, duration, 0.,
                                       destination.copy(), destination.copy())
            duration *= 1.25
        raise TargetTrajectoryError("stationary_endpoint_search_exhausted")


class GuardedTargetMotionPlanner(ContinuousTargetMotionPlanner):
    """V5: distinguish fresh stationary observations from duplicated packets.

    A pose jump outside the source velocity envelope is an endpoint request,
    not evidence of a sustained velocity. Command derivatives remain C2 bounded.
    """

    def _stop_duration_lower_bound(self, v, a, w, alpha):
        # Integrating the acceleration/jerk bounds gives necessary conditions
        # for reaching zero velocity and acceleration. Skip only durations that
        # cannot satisfy those conditions; retain the original geometric grid
        # and full native SO(3) test for every remaining candidate.
        bounds = []
        for velocity, acceleration, amax, jmax in (
            (v, a, self.limits.linear_acceleration_m_s2, self.limits.linear_jerk_m_s3),
            (w, alpha, self.limits.angular_acceleration_rad_s2, self.limits.angular_jerk_rad_s3),
        ):
            speed, accel = np.linalg.norm(velocity), np.linalg.norm(acceleration)
            bounds.extend((speed / amax, accel / jmax,
                           2 * speed / (np.sqrt(accel * accel + 2 * jmax * speed) + accel)
                           if speed else 0.))
        return max(bounds) / (1 + 1e-6)

    def _estimate_request(self, pose, dt):
        sample, previous = self._sample, self._last_sample
        fresh = sample is None or previous is None or sample.sequence != previous.sequence
        old_pose = None if self._previous_request is None else self._previous_request.copy()
        interval = dt
        if sample is not None and previous is not None and sample.epoch == previous.epoch:
            interval = sample.time_s - previous.time_s
        twist, acceleration = super()._estimate_request(pose, dt)
        if not fresh or old_pose is None or interval <= 0:
            return twist, acceleration
        linear = float(np.linalg.norm(pose.translation - old_pose.translation))
        angular = float(np.linalg.norm(pin.log3(old_pose.rotation.T @ pose.rotation)))
        # Sub-deadband motion is not a fresh zero-velocity observation: the
        # parent's dwell/coalescing logic must accumulate slow source motion.
        stationary = linear <= 1e-8 and angular <= 1e-8
        jumped = (linear > self.limits.linear_velocity_m_s * interval
                  or angular > self.limits.angular_velocity_rad_s * interval)
        if stationary or jumped:
            self._previous_velocity_world = np.zeros(6)
            self._previous_secant_world = np.zeros(6)
            self._previous_distinct_interval_s = None
            self.input_event = "endpoint_jump" if jumped else "fresh_stationary"
            return np.zeros(6), np.zeros(6)
        return twist, acceleration

    def _new_segment(self, destination, dt):
        if not self.request_is_moving and self._motion is not None:
            current = self._motion
            rotation = current.pose.rotation
            v = rotation @ current.body_twist[:3]
            a = rotation @ (current.body_acceleration[:3]
                            + np.cross(current.body_twist[3:], current.body_twist[:3]))
            w, alpha = current.body_twist[3:], current.body_acceleration[3:]
            if np.linalg.norm(v) + np.linalg.norm(w) > 1e-6:
                duration = dt
                lower_bound = self._stop_duration_lower_bound(v, a, w, alpha)
                for _ in range(60):
                    if duration < lower_bound:
                        duration *= 1.1
                        continue
                    # A minimum-duration feasible stop, independent of the
                    # eventual target. Forcing a quintic to return to a target
                    # inside its stopping distance can needlessly overshoot.
                    stop = self._request_motion(pin.SE3(
                        rotation @ pin.exp3(.5 * w * duration + alpha * duration**2 / 12),
                        current.pose.translation + .5 * v * duration + a * duration**2 / 12), None, None)
                    translation, angular, start = self._build_coefficients(stop, duration)
                    if self._within_limits(translation, angular, duration):
                        beyond_linear = np.dot(stop.pose.translation - destination.pose.translation, v) > 1e-9
                        beyond_angular = np.dot(pin.log3(destination.pose.rotation.T @ stop.pose.rotation), w) > 1e-9
                        if beyond_linear or beyond_angular:
                            self.diagnostics['replans'] += 1
                            return _QuinticSegment(translation, angular, start, duration, 0.,
                                                   destination.copy(), stop)
                        break
                    duration *= 1.1
        return super()._new_segment(destination, dt)


class PrecisionTargetMotionPlanner(GuardedTargetMotionPlanner):
    """V6: geometric endpoint horizon; bounded work independent of target distance.

    The old 25 probes starting at dt stopped searching at 4.24 seconds at 50 Hz.
    Large, perfectly valid rotations consequently froze the emitted reference.
    Starting at the analytic motion lower bound also avoids repeated tiny probes.
    """

    def _endpoint_segment(self, destination, dt):
        current = self._motion
        duration = max(dt, self._minimum_duration(destination, dt),
                       self._stop_duration_lower_bound(
                           current.body_twist[:3], current.body_acceleration[:3],
                           current.body_twist[3:], current.body_acceleration[3:]))
        for _ in range(32):
            translation, rotation, start = self._build_coefficients(destination, duration)
            if self._within_limits(translation, rotation, duration):
                return _QuinticSegment(translation, rotation, start, duration, 0.,
                                       destination.copy(), destination.copy())
            duration *= 1.25
        # A moving initial state close to the speed envelope need not admit a
        # single endpoint quintic for ANY duration. Stop continuously first,
        # retaining the requested endpoint so the next segment still reaches it.
        # This is bounded reference shaping, not a controller-history reset.
        rotation = current.pose.rotation
        v = rotation @ current.body_twist[:3]
        a = rotation @ (current.body_acceleration[:3]
                        + np.cross(current.body_twist[3:], current.body_twist[:3]))
        w, alpha = current.body_twist[3:], current.body_acceleration[3:]
        duration = max(dt, self._stop_duration_lower_bound(v, a, w, alpha))
        for _ in range(40):
            stop = self._request_motion(pin.SE3(
                rotation @ pin.exp3(.5*w*duration + alpha*duration**2/12),
                current.pose.translation + .5*v*duration + a*duration**2/12), None, None)
            translation, angular, start = self._build_coefficients(stop, duration)
            if self._within_limits(translation, angular, duration):
                self.diagnostics['endpoint_brakes'] = self.diagnostics.get('endpoint_brakes', 0)+1
                return _QuinticSegment(translation, angular, start, duration, 0., destination.copy(), stop)
            duration *= 1.1
        # A single quintic is not a complete stopping family: with outward
        # acceleration close to vmax it may initially accelerate too long at
        # every duration. Spend one tick removing acceleration before retrying.
        # This preserves pose/velocity/acceleration, and validates the entire
        # emitted SO(3) segment with the same hard bounds.
        coefficients = []
        for position, velocity, acceleration, jmax in (
            (current.pose.translation, v, a, self.limits.linear_jerk_m_s3),
            (np.zeros(3), w, alpha, self.limits.angular_jerk_rad_s3),
        ):
            jerk = -acceleration / dt
            jerk *= min(1., .95*jmax/max(1e-12, np.linalg.norm(jerk)))
            polynomial = np.zeros((6, 3))
            polynomial[:4] = [position, velocity, acceleration/2, jerk/6]
            coefficients.append(polynomial)
        if self._within_limits(*coefficients, dt):
            self.diagnostics['acceleration_brake_ticks'] = self.diagnostics.get('acceleration_brake_ticks', 0)+1
            segment = _QuinticSegment(*coefficients, rotation.copy(), dt, 0.,
                                     destination.copy(), destination.copy())
            segment.destination = self._sample_segment(segment, dt)
            return segment
        raise TargetTrajectoryError("precision_endpoint_bounds_exhausted")

    def _estimate_request(self, pose, dt):
        # V5 compared accumulated sub-deadband motion with ONE packet interval.
        # Compare it with its actual accumulation interval instead.
        previous = self._last_sample
        interval = dt
        if self._sample is not None and previous is not None and self._sample.epoch == previous.epoch:
            interval = self._sample.time_s - previous.time_s
        old_pose = None if self._previous_request is None else self._previous_request.copy()
        elapsed = self._elapsed_since_distinct_request_s + max(0., interval)
        twist, acceleration = ContinuousTargetMotionPlanner._estimate_request(self, pose, dt)
        fresh = self._sample is None or previous is None or self._sample.sequence != previous.sequence
        if not fresh or old_pose is None or elapsed <= 0:
            return twist, acceleration
        linear = np.linalg.norm(pose.translation - old_pose.translation)
        angular = np.linalg.norm(pin.log3(old_pose.rotation.T @ pose.rotation))
        stationary = linear <= 1e-8 and angular <= 1e-8
        jumped = (linear > self.limits.linear_velocity_m_s * elapsed
                  or angular > self.limits.angular_velocity_rad_s * elapsed)
        if stationary or jumped:
            self._previous_velocity_world = np.zeros(6)
            self._previous_secant_world = np.zeros(6)
            self._previous_distinct_interval_s = None
            self.input_event = "endpoint_jump" if jumped else "fresh_stationary"
            return np.zeros(6), np.zeros(6)
        return twist, acceleration
