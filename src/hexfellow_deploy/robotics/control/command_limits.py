"""Limits on persistent joint references, expressed in Pink displacement units."""

from __future__ import annotations

import numpy as np
from pink.limits import Limit

try:
    from _hexfellow_collision_kernel import reference_future_bounds as _native_future_bounds
except ImportError:
    _native_future_bounds = None


class JerkSafeVelocityLimit(Limit):
    """Keep the next reference inside the discrete velocity viability region.

    After choosing v_next, the fastest way to remove positive acceleration
    decreases it by j*dt each tick. Every prefix of that ramp must stay below
    vmax: (k+1)*v_next <= vmax + k*v_previous + j*dt²*k*(k+1)/2.
    The mirrored inequalities protect -vmax. These are reference guarantees,
    not actuator stopping guarantees; the ordinary acceleration/jerk rows stay.
    """

    def __init__(self, model, jerk_limit, acceleration_limits, dt, reference_velocity_fraction=1.):
        self.vmax = np.asarray(model.velocityLimit).copy() * reference_velocity_fraction
        self.history = jerk_limit
        self.dt = float(dt)
        self.matrix = np.vstack((np.eye(model.nv), -np.eye(model.nv)))
        maximum_k = int(np.ceil(np.max(np.asarray(acceleration_limits) / (jerk_limit.jerk_limit * dt))))
        k = np.arange(maximum_k + 1, dtype=float)[:, None]
        self.weight = k / (k + 1)
        self.offset = self.vmax / (k + 1) + jerk_limit.jerk_limit * dt**2 * k / 2

    def compute_qp_inequalities(self, configuration, dt):
        if not np.isclose(dt, self.dt, atol=1e-12, rtol=0):
            raise ValueError("jerk-safe velocity bound requires the configured control period")
        previous = self.history.previous_velocity
        upper = np.min(self.offset + self.weight * previous, axis=0)
        negative = np.min(self.offset - self.weight * previous, axis=0)
        return self.matrix, dt * np.r_[upper, negative]


class PlanarResultantVelocityLimit(Limit):
    """Inscribed 16-gon plus projected jerk viability; QP variable is displacement."""

    def __init__(self, model, history, acceleration_limits, dt, fraction):
        angles = np.arange(16) * (2 * np.pi / 16)
        self.normals = np.c_[np.cos(angles), np.sin(angles)]
        self.matrix = np.zeros((16, model.nv))
        self.matrix[:, :2] = self.normals
        self.history, self.dt = history, dt
        radius = min(model.velocityLimit[:2]) * fraction
        self.bound = radius * np.cos(np.pi / 16)
        # A projected jerk guaranteed by the component box, with the same
        # conservative projection of acceleration. No post-solve clipping.
        j = np.min(history.jerk_limit[:2])
        a = np.sum(np.abs(self.normals) * np.asarray(acceleration_limits)[:2], axis=1)
        k = np.arange(int(np.ceil(max(a) / (j * dt))) + 1)[:, None]
        self.weight = k / (k + 1)
        self.offset = self.bound / (k + 1) + j * dt**2 * k / 2

    def compute_qp_inequalities(self, configuration, dt):
        if not np.isclose(dt, self.dt, atol=1e-12, rtol=0):
            raise ValueError("planar velocity bound requires configured period")
        previous = self.normals @ self.history.previous_velocity[:2]
        return self.matrix, dt * np.min(self.offset + self.weight * previous, axis=0)


def jerk_braking_step(velocity, acceleration, amax, jmax, dt):
    """Approach zero velocity while reserving room to unwind braking acceleration.

    Greedily aiming at zero and then clipping jerk overshoots zero repeatedly.
    These discrete ramp prefixes select the smallest speed from which braking
    acceleration can return to zero without reversing. Existing infeasible
    initial states and external QP constraints can still require a reversal.
    """
    direction = np.where(np.abs(velocity) > 1e-12, np.sign(velocity), np.sign(acceleration))
    k = np.arange(int(np.ceil(np.max(amax / (jmax * dt)))) + 1)[:, None]
    speed_goal = np.max(k / (k + 1) * np.abs(velocity) - jmax * dt**2 * k / 2, axis=0)
    desired = direction * (speed_goal - np.abs(velocity)) / dt
    desired = np.clip(desired, -amax, amax)
    next_acceleration = np.clip(desired, acceleration - jmax * dt, acceleration + jmax * dt)
    return velocity + next_acceleration * dt, next_acceleration


class MeasuredCollisionLimit(Limit):
    """Express measured-configuration barriers in reference-displacement units.

    The supplied barrier owns measured witness points and must use the matching
    measured Jacobian. The QP variable advances the persistent reference, so
    its offset from the measurement must also be included.
    """

    def __init__(self, barrier, measured_configuration, reference_offset, *,
                 method="reference_velocity_v1"):
        self.barrier = barrier
        self.measured_configuration = measured_configuration
        self.reference_offset = np.array(reference_offset, dtype=float, copy=True)
        if method not in {"reference_velocity_v1", "reference_endpoint_v2"}:
            raise ValueError("unknown measured-reference collision method")
        self.method = method
        self._endpoint_geometry = None

    def freeze_endpoint(self, jacobian, clearance, *, copy=True, qp_matrix=None):
        """Own current-cycle geometry shared by the hard endpoint and warning."""
        if self.method != "reference_endpoint_v2":
            raise ValueError("only endpoint constraints can freeze geometry")
        jacobian = np.array(jacobian, dtype=float, copy=copy)
        clearance = np.array(clearance, dtype=float, copy=copy)
        if (jacobian.shape != (len(clearance), len(self.reference_offset))
                or not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(clearance))):
            raise ValueError("invalid endpoint geometry")
        matrix = None if qp_matrix is None else np.array(qp_matrix, dtype=float, copy=copy)
        if matrix is not None and (
            matrix.shape != jacobian.shape or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("invalid endpoint QP matrix")
        self._endpoint_geometry = jacobian, clearance, matrix

    def compute_qp_inequalities(self, configuration, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("collision constraint timestep must be finite and positive")
        if self._endpoint_geometry is not None:
            jacobian, clearance, matrix = self._endpoint_geometry
            return (
                -jacobian / dt if matrix is None else matrix,
                (clearance + jacobian @ self.reference_offset) / dt,
            )
        matrix, bound = self.barrier.compute_qp_inequalities(self.measured_configuration, dt)
        if self.method == "reference_endpoint_v2":
            # Pink's velocity CBF has G=-J/dt and b=gain*h. Offsetting its
            # displacement variable silently demands that a persistent servo
            # tracking error be recovered in one period. Instead constrain
            # the linearized reference endpoint: h + J*(offset+dq) >= 0.
            # This protects reference geometry, NOT physical stopping.
            bound = self.barrier.compute_barrier(self.measured_configuration) / dt
        return matrix, bound - matrix @ self.reference_offset



class DirectionalVelocityLimit(Limit):
    """Early slowdown in displacement units, separate from hard geometry.

    The delay extrapolation is an engineering governor, not a physical
    stopping certificate. Only closing velocity consumes reaction clearance.
    """

    def __init__(self, jacobian, clearance, measured_velocity, reaction_s, *,
                 gain_s_inv=1.0, labels=(), guard_distance=0.0,
                 maximum_closing_speed=None, copy_geometry=True,
                 label_factory=None, qp_matrix=None):
        self.jacobian = np.array(jacobian, dtype=float, copy=copy_geometry)
        self.clearance = np.array(clearance, dtype=float, copy=copy_geometry)
        self._qp_matrix = (
            None if qp_matrix is None else np.array(qp_matrix, dtype=float, copy=copy_geometry)
        )
        velocity = np.asarray(measured_velocity, dtype=float)
        if (self.jacobian.ndim != 2 or self.clearance.shape != (len(self.jacobian),)
                or velocity.shape != (self.jacobian.shape[1],)
                or not all(np.all(np.isfinite(value)) for value in
                           (self.jacobian, self.clearance, velocity))
                or not np.isfinite(reaction_s) or reaction_s < 0
                or not np.isfinite(gain_s_inv) or gain_s_inv <= 0
                or not np.isfinite(guard_distance) or guard_distance < 0):
            raise ValueError("invalid directional boundary state")
        if self._qp_matrix is not None and (
            self._qp_matrix.shape != self.jacobian.shape
            or not np.all(np.isfinite(self._qp_matrix))
        ):
            raise ValueError("invalid directional QP matrix")
        self.closing_speed = np.maximum(0.0, -self.jacobian @ velocity)
        self.reaction_clearance = self.clearance - self.closing_speed * reaction_s
        self.available = np.maximum(0.0, self.reaction_clearance - guard_distance)
        if maximum_closing_speed is None:
            self.bound = gain_s_inv * self.available
        else:
            bound = np.asarray(maximum_closing_speed, dtype=float)
            if bound.shape != self.clearance.shape or not np.all(np.isfinite(bound)) or np.any(bound < 0):
                raise ValueError("directional closing-speed bound is invalid")
            self.bound = bound.copy()
        self.labels = tuple(labels)
        self._label_factory = label_factory
        if label_factory is not None and self.labels:
            raise ValueError("boundary labels and label factory are mutually exclusive")
        if label_factory is None and len(self.labels) != len(self.clearance):
            raise ValueError("boundary labels must match constraint rows")
        if label_factory is not None and not callable(label_factory):
            raise ValueError("boundary label factory must be callable")

    @property
    def reaction_clearance_exhausted(self):
        return bool(np.any((self.reaction_clearance <= 0.0) & (self.closing_speed > 1e-6)))

    def compute_qp_inequalities(self, configuration, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("boundary timestep must be positive")
        if not len(self.clearance):
            return None
        return (-self.jacobian / dt if self._qp_matrix is None else self._qp_matrix), self.bound

    def binding_labels(self, velocity, tolerance=1e-6):
        residual = self.bound + self.jacobian @ velocity
        if self._label_factory is not None:
            return tuple(
                self._label_factory(index)
                for index in np.flatnonzero(residual <= tolerance)
            )
        return tuple(label for label, value in zip(self.labels, residual, strict=True)
                     if value <= tolerance)


def joint_boundary_damper(model, indices, measured_q, reference_q, measured_velocity,
                          activation_travel, reaction_s, gain_s_inv=1.0):
    """Activate directed joint warnings without replacing position/jerk limits."""
    indices = np.asarray(indices, dtype=int)
    projection = np.eye(model.nv)[indices]
    clearance = np.concatenate((
        np.minimum(measured_q[indices], reference_q[indices]) - model.lowerPositionLimit[indices],
        model.upperPositionLimit[indices] - np.maximum(measured_q[indices], reference_q[indices]),
    ))
    jacobian = np.vstack((projection, -projection))
    # At activation the warning cap must be at least the ordinary speed cap.
    # A stopping-travel-only switch introduces a discontinuous speed limit.
    influence = np.maximum(
        np.asarray(activation_travel)[indices],
        np.asarray(model.velocityLimit)[indices] * (1.0 / gain_s_inv + reaction_s),
    )
    active = clearance <= np.tile(influence, 2)
    labels = tuple(f"joint_{side}:{index}" for side in ("lower", "upper") for index in indices)
    return DirectionalVelocityLimit(
        jacobian[active], clearance[active], measured_velocity, reaction_s,
        gain_s_inv=gain_s_inv, labels=tuple(label for label, keep in zip(labels, active, strict=True) if keep),
    )


class JointBoundaryGovernor:
    """Jerk-stoppable joint warnings for one measured or reference state.

    The ordinary configuration and persistent-reference limits remain hard.
    This governor activates continuously before their acceleration/jerk rows
    become incompatible, without imposing the old ``speed <= clearance / s``
    ceiling throughout narrow wrist ranges.
    """

    def __init__(self, model, indices, acceleration_limits, jerk_limits):
        self.model = model
        self.indices = np.asarray(indices, dtype=int)
        if self.indices.ndim != 1 or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("joint boundary indices must be a unique vector")
        self.projection = np.eye(model.nv)[self.indices]
        self.jacobian = np.vstack((self.projection, -self.projection))
        self.vmax = np.asarray(model.velocityLimit, dtype=float)[self.indices]
        self.amax = np.asarray(acceleration_limits, dtype=float)[self.indices]
        self.jmax = np.asarray(jerk_limits, dtype=float)[self.indices]
        if any(value.shape != self.indices.shape for value in (self.vmax, self.amax, self.jmax)):
            raise ValueError("joint boundary limits have incompatible shapes")
        self.row_vmax = np.tile(self.vmax, 2)
        self.row_amax = np.tile(self.amax, 2)
        self.row_jmax = np.tile(self.jmax, 2)
        self.labels = tuple(
            f"joint_{side}:{index}"
            for side in ("lower", "upper")
            for index in self.indices
        )

    def limit(self, q, velocity, reaction_s, position_uncertainty=None):
        q = np.asarray(q, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if (
            q.shape != (self.model.nq,)
            or velocity.shape != (self.model.nv,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(velocity))
            or not np.isfinite(reaction_s)
            or reaction_s <= 0.0
        ):
            raise ValueError("invalid jerk-stoppable joint boundary state")
        clearance = np.concatenate((
            q[self.indices] - self.model.lowerPositionLimit[self.indices],
            self.model.upperPositionLimit[self.indices] - q[self.indices],
        ))
        if position_uncertainty is not None:
            reserve = np.asarray(position_uncertainty, dtype=float)
            if reserve.shape != (self.model.nv,) or not np.all(np.isfinite(reserve)) or np.any(reserve < 0):
                raise ValueError("invalid joint boundary position uncertainty")
            clearance -= np.tile(reserve[self.indices], 2)
        safe_speed = delayed_stoppable_speed(
            clearance,
            self.row_vmax,
            self.row_amax,
            self.row_jmax,
            reaction_s,
        )
        active = safe_speed < self.row_vmax - 1e-12
        return DirectionalVelocityLimit(
            self.jacobian[active],
            clearance[active],
            velocity,
            reaction_s,
            labels=tuple(
                label for label, keep in zip(self.labels, active, strict=True) if keep
            ),
            maximum_closing_speed=safe_speed[active],
        )


def stopping_distance(velocity, acceleration, acceleration_limit, jerk_limit, delay_s=0.0):
    """Conservative directed travel under jerk-limited braking.

    Velocity is nonnegative; acceleration is signed along the travel direction.
    During a declared delay, acceleration is conservatively bounded by +amax.
    After the delay jerk is -jmax until -amax, followed by constant braking.
    This is a reference-trajectory bound, not an actuator stopping certificate.
    """
    v, a, amax, jmax = np.broadcast_arrays(
        np.asarray(velocity, dtype=float), np.asarray(acceleration, dtype=float),
        np.asarray(acceleration_limit, dtype=float), np.asarray(jerk_limit, dtype=float),
    )
    if (not all(np.all(np.isfinite(x)) for x in (v, a, amax, jmax))
            or np.any(v < 0) or np.any(amax <= 0) or np.any(jmax <= 0)
            or not np.isfinite(delay_s) or delay_s < 0):
        raise ValueError("invalid braking state or limits")
    a = np.clip(a, -amax, amax)
    delayed_travel = v * delay_s + 0.5 * amax * delay_s**2
    if delay_s:
        v = v + amax * delay_s
        a = amax
    ramp = (a + amax) / jmax
    stop = (a + np.sqrt(a * a + 2.0 * jmax * v)) / jmax
    time = np.minimum(ramp, stop)
    travel = v * time + 0.5 * a * time**2 - jmax * time**3 / 6.0
    remaining_v = np.maximum(0.0, v + a * time - 0.5 * jmax * time**2)
    return delayed_travel + np.maximum(0.0, travel) + remaining_v**2 / (2.0 * amax)


def jerk_reaction_stopping_distance(
    velocity, acceleration, acceleration_limit, jerk_limit, reaction_s
):
    """Causal travel from the observed acceleration, then jerk-limited braking.

    This is the causal reference counterpart to :func:`stopping_distance`,
    whose delay branch deliberately assumes an instantaneous jump to ``amax``.
    The reference advances with the current outward acceleration during the
    reaction window, then follows the same jerk-limited braking envelope.
    Future acceleration commands are checked again before they are emitted.
    """
    velocity, acceleration, amax, jmax = np.broadcast_arrays(
        np.asarray(velocity, dtype=float),
        np.asarray(acceleration, dtype=float),
        np.asarray(acceleration_limit, dtype=float),
        np.asarray(jerk_limit, dtype=float),
    )
    if (
        not all(np.all(np.isfinite(value)) for value in (velocity, acceleration, amax, jmax))
        or np.any(velocity < 0.0)
        or np.any(amax <= 0.0)
        or np.any(jmax <= 0.0)
        or not np.isfinite(reaction_s)
        or reaction_s < 0.0
    ):
        raise ValueError("invalid jerk-reaction stopping state")
    # Negative measured acceleration may cease before the next observation;
    # crediting it would understate forward travel. Zero is the conservative
    # causal floor while still avoiding an instantaneous jump to +amax.
    acceleration = np.clip(acceleration, 0.0, amax)
    reaction_travel = velocity * reaction_s + 0.5 * acceleration * reaction_s**2
    braking_velocity = velocity + acceleration * reaction_s
    return reaction_travel + stopping_distance(
        braking_velocity, acceleration, amax, jmax
    )


def stoppable_speed(distance, acceleration, vmax, amax, jmax, delay_s):
    """Invert the monotonic braking envelope without allocating per-joint QPs."""
    distance = np.maximum(np.asarray(distance, dtype=float), 0.0)
    low = np.zeros_like(distance)
    high = np.asarray(vmax, dtype=float).copy()
    fits_maximum = stopping_distance(high, acceleration, amax, jmax, delay_s) <= distance
    if np.all(fits_maximum):
        return high
    for _ in range(24):
        middle = 0.5 * (low + high)
        fits = stopping_distance(middle, acceleration, amax, jmax, delay_s) <= distance
        low = np.where(fits, middle, low)
        high = np.where(fits, high, middle)
    return np.where(fits_maximum, np.asarray(vmax), low)



def delayed_stoppable_speed(distance, vmax, amax, jmax, delay_s):
    """Exact inverse of stopping_distance for its positive-delay branch.

    That branch assumes +amax during the delay and starts braking at +amax.
    With ramp=2*amax/jmax, travel is v**2/(2*amax) + v*(2*delay+ramp)
    + amax*delay**2 + amax*delay*ramp + 2*amax**3/(3*jmax**2).
    This is the SAME reference envelope, not an actuator stopping bound.
    """
    distance, vmax, amax, jmax = np.broadcast_arrays(
        np.asarray(distance, dtype=float), np.asarray(vmax, dtype=float),
        np.asarray(amax, dtype=float), np.asarray(jmax, dtype=float),
    )
    if (not all(np.all(np.isfinite(value)) for value in (distance, vmax, amax, jmax))
            or np.any(vmax < 0) or np.any(amax <= 0) or np.any(jmax <= 0)
            or not np.isfinite(delay_s) or delay_s <= 0):
        raise ValueError("closed-form reference envelope requires positive delay and limits")
    ramp = 2.0 * amax / jmax
    constant = amax * delay_s**2 + amax * delay_s * ramp + 2.0 * amax**3 / (3.0 * jmax**2)
    linear = 2.0 * delay_s + ramp
    remaining = np.maximum(distance - constant, 0.0)
    # Rationalized quadratic root avoids cancellation close to zero.
    speed = 2.0 * remaining / (np.sqrt(linear**2 + 2.0 * remaining / amax) + linear)
    return np.minimum(vmax, np.nextafter(speed, 0.0))


class JointReferenceLimit(Limit):
    """Constrain integrated MIT references and predicted total joint effort.

    The ordinary Pink limits protect measured-state IK. These additional rows
    protect a persistent reference and keep PD feedforward from winding it up
    against a stalled or saturated arm. Each controller owns its own history.
    """

    def __init__(self, model, arm_start, servo, acceleration_limits, jerk_limits, *, closed_form=False,
                 planning_effort_fraction=1.0):
        self.model = model
        if not np.isfinite(planning_effort_fraction) or not 0 < planning_effort_fraction <= 1:
            raise ValueError("planning effort fraction must be in (0, 1]")
        self.planning_effort_fraction = planning_effort_fraction
        self.closed_form = closed_form
        self.indices = np.arange(arm_start, model.nv)
        self.projection = np.eye(model.nv)[self.indices]
        self.servo = servo
        self.amax = np.asarray(acceleration_limits)[self.indices]
        self.jmax = np.asarray(jerk_limits)[self.indices]
        self.vmax = np.asarray(model.velocityLimit)[self.indices]
        self._future_dt = None
        self.reference = None

    def update(self, reference, acceleration, measured_q, measured_velocity, gravity, dt, reference_velocity=None):
        self.reference = np.asarray(reference)[self.indices].copy()
        self.measured_q = np.asarray(measured_q)[self.indices].copy()
        self.acceleration = np.asarray(acceleration)[self.indices].copy()
        self.measured_velocity = np.asarray(measured_velocity)[self.indices].copy()
        self.gravity = np.asarray(gravity)[self.indices].copy()
        self.reference_velocity = np.asarray(
            measured_velocity if reference_velocity is None else reference_velocity)[self.indices].copy()
        if self._future_dt != dt:
            horizon = int(np.ceil(np.max(self.vmax / self.amax + 3 * self.amax / self.jmax) / dt)) + 2
            self._future_k = np.arange(horizon, dtype=float)[:, None]
            self._future_dt = dt

    def _future_speed_bounds(self, distances, previous_velocity, maximum, dt):
        """Tighten today's envelope so a jerk ramp can respect future envelopes.

        This is a vectorized scalar reference calculation, not a physical robot
        prediction. It uses a fixed reference reserve; the measured-state
        envelope remains an independent restriction. Every future prefix
        retains the existing worst-case stopping envelope.
        """
        if _native_future_bounds is not None:
            return np.asarray(_native_future_bounds(
                distances, previous_velocity, maximum, self.amax, self.jmax, dt, len(self._future_k)))
        amax, jmax = np.tile(self.amax, 2), np.tile(self.jmax, 2)
        ramp = 2 * amax / jmax
        constant = amax * dt**2 + amax * dt * ramp + 2 * amax**3 / (3 * jmax**2)
        linear = 2 * dt + ramp
        def needed(candidate):
            future_a = np.maximum((candidate-previous_velocity)/dt - self._future_k*jmax*dt, -amax)
            future_a[0] = 0.
            future_v = np.maximum(candidate + dt*np.cumsum(future_a, axis=0), 0.)
            travel = dt*(np.cumsum(future_v, axis=0)-future_v)
            envelope = future_v**2/(2*amax) + linear*future_v + constant
            return np.max(travel + np.where(future_v > 1e-12, envelope, 0.), axis=0)
        # A 2 mrad reference reserve accommodates small changes in tracking offset.
        distance = np.maximum(distances - .002, 0.)
        fits = needed(maximum) <= distance
        if np.all(fits):
            return maximum
        active = ~fits
        amax, jmax = amax[active], jmax[active]
        constant, linear = constant[active], linear[active]
        previous_velocity, distance = previous_velocity[active], distance[active]
        low, high = np.zeros(np.count_nonzero(active)), maximum[active].copy()
        for _ in range(36):
            middle = (low + high) * .5
            feasible = needed(middle) <= distance
            low = np.where(feasible, middle, low)
            high = np.where(feasible, high, middle)
        result = maximum.copy()
        result[active] = low
        return result

    def compute_qp_inequalities(self, configuration, dt):
        if self.reference is None:
            raise RuntimeError("reference limit requires a measured state update")
        q = self.measured_q
        lower = self.model.lowerPositionLimit[self.indices]
        upper = self.model.upperPositionLimit[self.indices]
        # Account for both the physical configuration and command position.
        positive_distance = upper - np.maximum(q, self.reference)
        negative_distance = np.minimum(q, self.reference) - lower
        positive_acc = np.minimum(self.amax, self.acceleration + self.jmax * dt)
        negative_acc = np.minimum(self.amax, -self.acceleration + self.jmax * dt)
        if self.closed_form:
            positive = delayed_stoppable_speed(positive_distance, self.vmax, self.amax, self.jmax, dt)
            negative = delayed_stoppable_speed(negative_distance, self.vmax, self.amax, self.jmax, dt)
            # Only investigate candidates reachable in THIS tick under the
            # unchanged acceleration/jerk bounds, avoiding irrelevant lookahead.
            positive = np.minimum(positive, np.maximum(0., self.reference_velocity + positive_acc * dt))
            negative = np.minimum(negative, np.maximum(0., -self.reference_velocity + negative_acc * dt))
            positive, negative = np.split(self._future_speed_bounds(
                # A fixed reserve around the reference, rather than a moving
                # reserve around measured tracking noise. Today's measured
                # envelope remains an independent hard restriction above.
                np.r_[upper - self.reference, self.reference - lower],
                np.r_[self.reference_velocity, -self.reference_velocity], np.r_[positive, negative], dt), 2)
        else:
            positive = stoppable_speed(positive_distance, positive_acc, self.vmax,
                                       self.amax, self.jmax, dt)
            negative = stoppable_speed(negative_distance, negative_acc, self.vmax,
                                       self.amax, self.jmax, dt)
        kp, kd = np.asarray(self.servo.kp), np.asarray(self.servo.kd)
        effort_gain = kp + kd / dt
        effort_offset = kp * (self.reference - q) - kd * self.measured_velocity + self.gravity
        # Reserve feedback-change headroom without changing the publisher's bound
        # or the gains/torque fields sent to the drive.
        effort_bound = np.asarray(self.servo.torque_limit_nm) * self.planning_effort_fraction
        matrix = np.vstack((self.projection, -self.projection,
                            effort_gain[:, None] * self.projection,
                            -effort_gain[:, None] * self.projection))
        bound = np.concatenate((np.minimum(positive * dt, upper - self.reference),
                                np.minimum(negative * dt, self.reference - lower),
                                effort_bound - effort_offset, effort_bound + effort_offset))
        return matrix, bound


class FiniteBaseBoundaryGovernor(JointBoundaryGovernor):
    """Coordinate finite base ranges with the next acceleration/jerk interval.

    Uses scalar monotone searches, not another QP. It checks future discrete
    braking prefixes against the same delayed envelope used for activation.
    This is a reference governor; plant stopping remains an empirical claim.
    """

    def limits(self, measured_q, reference_q, measured_velocity, previous_velocity,
               previous_acceleration, reaction_s, uncertainty, dt):
        states = (measured_q, reference_q, measured_velocity, previous_velocity, previous_acceleration, uncertainty)
        if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(reaction_s) or reaction_s <= 0
                or any(np.asarray(value).shape != (self.model.nv,) or not np.all(np.isfinite(value))
                       for value in states) or np.any(np.asarray(uncertainty) < 0)):
            raise ValueError("invalid finite base governor state")
        i = self.indices
        if not len(i):
            return ()
        qlo = np.minimum(measured_q[i], reference_q[i])
        qhi = np.maximum(measured_q[i], reference_q[i])
        distance = np.r_[qlo - self.model.lowerPositionLimit[i],
                         self.model.upperPositionLimit[i] - qhi]
        distance -= np.tile(uncertainty[i], 2)
        # Jacobian rows point away from each boundary; closing is its negative.
        previous = np.r_[-previous_velocity[i], previous_velocity[i]]
        acceleration = np.r_[-previous_acceleration[i], previous_acceleration[i]]
        measured = np.r_[-measured_velocity[i], measured_velocity[i]]
        # Reserve actual closing travel beyond the command's closing component.
        # An inward command with a stationary (or slower inward) plant must not
        # consume outward clearance: the signed speed difference is tracking
        # lag, not motion toward this boundary. Both positions are already
        # reserved by qlo/qhi, and reference braking history stays unchanged.
        distance -= np.maximum(0., np.maximum(measured, 0.) - np.maximum(previous, 0.)) * reaction_s
        maximum = delayed_stoppable_speed(np.maximum(distance, 0.), self.row_vmax,
                                          self.row_amax, self.row_jmax, reaction_s)
        horizon = int(np.ceil(np.max(self.vmax / self.amax + 3*self.amax/self.jmax) / dt)) + 2
        k = np.arange(horizon)[:, None]
        # The positive-delay stopping envelope is a quadratic in speed (same
        # identity used by delayed_stoppable_speed). Hoist its validated
        # coefficients out of the 36 bisection iterations; do not recompute
        # broadcast/validation/jerk-ramp arrays for every candidate prefix.
        ramp = 2*self.row_amax/self.row_jmax
        constant = (self.row_amax*reaction_s**2 + self.row_amax*reaction_s*ramp
                    + 2*self.row_amax**3/(3*self.row_jmax**2))
        linear = 2*reaction_s + ramp

        def required(candidate):
            a = np.maximum((candidate - previous) / dt - k*self.row_jmax*dt, -self.row_amax)
            a[0] = 0.
            v = np.maximum(candidate + dt*np.cumsum(a, axis=0), 0.)
            travel = dt*np.cumsum(v, axis=0)
            envelope = v**2/(2*self.row_amax) + linear*v + constant
            required_distance = np.max(travel + np.where(v > 1e-12, envelope, 0.), axis=0)
            return np.where(np.max(v, axis=0) <= self.row_vmax + 1e-10, required_distance, np.inf)

        low, high = np.zeros_like(maximum), maximum.copy()
        fits = required(maximum) <= distance
        for _ in range(36):
            if np.all(fits):
                break
            mid = (low + high) * .5
            feasible = required(mid) <= distance
            low = np.where(feasible, mid, low)
            high = np.where(feasible, high, mid)
        cap = np.where(fits, maximum, low)
        # A pre-existing state outside the viability region cannot be repaired by
        # resetting history or projecting its measured position into the range.
        minimum_acceleration = np.maximum(-self.row_amax, acceleration - self.row_jmax*dt)
        minimum_velocity = previous + minimum_acceleration*dt
        violated = np.flatnonzero((minimum_velocity > cap + 1e-7) | (distance < -1e-7))
        if len(violated):
            raise BoundaryStateUnrecoverable(
                "finite base state is outside the braking envelope",
                {"reaction_s": float(reaction_s), "dt_s": float(dt), "boundaries": [
                    {"label": self.labels[row], "available_distance": float(distance[row]),
                     "minimum_closing_velocity": float(minimum_velocity[row]),
                     "allowed_closing_velocity": float(cap[row]),
                     "previous_closing_velocity": float(previous[row])}
                    for row in violated]},
            )
        active = cap < self.row_vmax - 1e-12
        return (DirectionalVelocityLimit(
            self.jacobian[active], distance[active], measured_velocity, reaction_s,
            labels=tuple(label for label, keep in zip(self.labels, active, strict=True) if keep),
            maximum_closing_speed=cap[active],
        ),)


class BoundaryStateUnrecoverable(RuntimeError):
    """The recorded state/history cannot meet the finite-base braking envelope."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


class JerkEffortPreviewLimit(Limit):
    """Necessary scalar braking-prefix effort bounds, not a plant certificate.

    For each signed effort boundary, maximum opposing jerk minimizes future
    signed reference effort. Every prefix must fit, under both frozen feedback
    and constant measured velocity. Opposite boundaries use different extremal
    trajectories: these rows anticipate conflict but do not assert a shared
    feasible future trajectory, especially when real feedback changes.
    """

    def __init__(self, reference_limit):
        self.limit = reference_limit
        self.bounds = None
        self._dt = None

    def _prepare(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("effort preview needs a positive finite timestep")
        limit = self.limit
        if limit.reference is None:
            raise RuntimeError("effort preview requires current feedback")
        # Include feedback overspeed admitted by the existing recovery policy.
        n = (
            int(
                np.ceil(
                    np.max(
                        (limit.vmax + np.maximum(limit.vmax, np.abs(limit.measured_velocity))) / limit.amax
                        + 2 * limit.amax / limit.jmax
                    )
                    / dt
                )
            )
            + 3
        )
        if self._dt != dt or getattr(self, "_n", None) != n:
            # Cover reversal of the largest reference/feedback relative speed
            # after traversing the full acceleration range.
            self._n = n
            self.k = np.arange(n, dtype=float)[:, None]
            self._dt = dt
        self.sign = np.r_[np.ones(len(limit.indices)), -np.ones(len(limit.indices))]

    def peaks(self, signed_velocity, dt):
        limit, k = self.limit, self.k

        def tile(x):
            return np.tile(x, 2)

        s = self.sign
        previous = s * tile(limit.reference_velocity)
        a = (signed_velocity - previous) / dt
        acceleration = np.clip(a[None, :] - k * tile(limit.jmax) * dt, -tile(limit.amax), tile(limit.amax))
        acceleration[0] = 0.0  # first candidate is already integrated once
        v = signed_velocity + dt * np.cumsum(acceleration, axis=0)
        travel = dt * np.cumsum(v, axis=0)
        gap = s * tile(limit.reference - limit.measured_q)
        measured_v = s * tile(limit.measured_velocity)
        base = tile(limit.servo.kp) * (gap + travel) + tile(limit.servo.kd) * (v - measured_v) + s * tile(limit.gravity)
        moving = base - tile(limit.servo.kp) * measured_v * k * dt
        return np.maximum(base.max(axis=0), moving.max(axis=0))

    def compute_qp_inequalities(self, configuration, dt):
        self._prepare(dt)
        limit = self.limit
        # Within the existing acceleration box, future opposing acceleration is
        # max((w-w_previous)/dt - i*j*dt, -amax). Its effort contribution is a
        # convex piecewise affine function of signed candidate velocity w.
        # Unsaturated terms always form a prefix of length m. Enumerating those
        # prefixes gives the EXACT maximum of affine pieces, without root search.
        k = self.k[:, None, :]
        maximum_m = int(np.ceil(np.max(2 * limit.amax / (limit.jmax * dt))))
        m = np.minimum(k, np.arange(maximum_m + 1, dtype=float)[None, :, None])
        kp, kd = np.tile(limit.servo.kp, 2), np.tile(limit.servo.kd, 2)
        amax, jmax = np.tile(limit.amax, 2), np.tile(limit.jmax, 2)
        previous = self.sign * np.tile(limit.reference_velocity, 2)
        measured = self.sign * np.tile(limit.measured_velocity, 2)
        gap = self.sign * np.tile(limit.reference - limit.measured_q, 2)
        sum_i = m * (m + 1) / 2
        sum_i2 = m * (m + 1) * (2 * m + 1) / 6
        prefix = kp * dt**2 * (m * (k + 1) - sum_i) + kd * dt * m
        weighted = kp * dt**2 * ((k + 1) * sum_i - sum_i2) + kd * dt * sum_i
        total = kp * dt**2 * k * (k + 1) / 2 + kd * dt * k
        gain = kp * dt * (k + 1) + kd + prefix / dt
        offset = (
            kp * gap
            - kd * measured
            + self.sign * np.tile(limit.gravity, 2)
            - prefix * previous / dt
            - weighted * jmax * dt
            - (total - prefix) * amax
            - np.minimum(0.0, kp * measured * k * dt)
        )
        budget = np.tile(np.asarray(limit.servo.torque_limit_nm) * limit.planning_effort_fraction, 2)
        cap = np.minimum(np.tile(limit.vmax, 2), np.min((budget - offset) / gain, axis=(0, 1)))
        self.bounds = np.stack((-cap[len(limit.indices) :], cap[: len(limit.indices)]))
        return np.vstack((limit.projection, -limit.projection)), dt * cap

    def report(self, velocity, dt):
        v = np.asarray(velocity)[self.limit.indices]
        peak = self.peaks(self.sign * np.tile(v, 2), dt)
        n = len(v)
        return dict(
            mode="jerk_prefix_v1",
            indices=self.limit.indices.copy(),
            minimum_velocity=self.bounds[0].copy(),
            maximum_velocity=self.bounds[1].copy(),
            predicted_signed_peak_nm=np.stack((peak[:n], peak[n:])),
            horizon_s=float(self.k[-1, 0] * dt),
        )
