"""Versioned, dimensionless precision tasks; shared by physical and simulated IK."""

import numpy as np
import pinocchio as pin
from pink.tasks import FrameTask


class PrecisionFrameTask(FrameTask):
    """World position and local SO(3) error, with independent feedback gains.

    Costs are residual multipliers (squared in the objective). A unit multiplier
    means one unit of cost at 5 mm / 2 degrees, respectively. Translation is
    deliberately not the translation component of a coupled SE(3) logarithm.
    """

    def __init__(self, *args, orientation_gain, **kwargs):
        super().__init__(*args, **kwargs)
        self.orientation_gain = orientation_gain
        self._feedforward_displacement = np.zeros(6)

    def set_target(self, transform_target_to_world):
        super().set_target(transform_target_to_world)
        self._feedforward_displacement = np.zeros(6)

    def set_target_motion(self, target, error_rate, dt):
        self.set_target(target)
        self._feedforward_displacement = np.asarray(error_rate) * dt

    def compute_error(self, configuration):
        current = configuration.get_transform_frame_to_world(self.frame)
        target = self.transform_target_to_world
        return np.r_[target.translation - current.translation,
                     pin.log3(current.rotation.T @ target.rotation)]

    compute_geometric_error = compute_error

    def compute_jacobian(self, configuration):
        current = configuration.get_transform_frame_to_world(self.frame)
        relative = current.rotation.T @ self.transform_target_to_world.rotation
        body = configuration.get_frame_jacobian(self.frame)
        return np.vstack((-current.rotation @ body[:3], -pin.Jlog3(relative.T) @ body[3:]))

    def compute_qp_objective(self, configuration):
        scales = np.r_[np.full(3, .005), np.full(3, np.deg2rad(2))]
        weight = np.asarray(self.cost) / scales
        gains = np.r_[np.full(3, self.gain), np.full(3, self.orientation_gain)]
        desired = weight * (gains * self.compute_error(configuration) + self._feedforward_displacement)
        jacobian = weight[:, None] * self.compute_jacobian(configuration)
        damping = self.lm_damping * float(desired @ desired)
        return jacobian.T @ jacobian + damping * np.eye(configuration.model.nv), desired @ jacobian


class EffortHeadroomTask:
    """Discourage PD windup before the hard effort envelope becomes binding.

    This is a soft objective, not proof of future physical torque. The independent
    position, jerk, acceleration and predicted-total-effort rows remain hard.
    """

    def __init__(self, reference_limit, cost, lookahead_s):
        self.limit, self.cost, self.lookahead_s = reference_limit, cost, lookahead_s

    def compute_qp_objective(self, configuration):
        limit = self.limit
        n = configuration.model.nv
        if limit.reference is None or not self.cost:
            return np.zeros((n, n)), np.zeros(n)
        kp, kd = np.asarray(limit.servo.kp), np.asarray(limit.servo.kd)
        bounds = np.asarray(limit.servo.torque_limit_nm) * limit.planning_effort_fraction
        pd = kp * (limit.reference - limit.measured_q) - kd * limit.measured_velocity
        current = pd + kd * limit.reference_velocity + limit.gravity
        # Look ahead through the bounded feedback/dispatch interval. Predict
        # reference motion with its retained acceleration, and feedback with
        # measured velocity. This early soft penalty complements, not replaces,
        # the hard reference/effort constraints and stopping-distance bounds.
        horizon = self.lookahead_s
        predicted = current + kp * ((limit.reference_velocity-limit.measured_velocity)*horizon
                                    + .5*limit.acceleration*horizon*horizon) + kd*limit.acceleration*horizon
        reserve = np.maximum(.1*bounds, bounds-np.maximum(np.abs(current), np.abs(predicted)))
        weight = self.cost / reserve
        jacobian = weight[:, None] * (kp + kd / limit._future_dt)[:, None] * limit.projection
        residual = weight * pd
        return jacobian.T @ jacobian, residual @ jacobian


from pink.tasks import LowAccelerationTask


class NormalizedAccelerationTask(LowAccelerationTask):
    """Dimensionless reference acceleration residual: (dq-dq_prev)/(dt² amax)."""

    def __init__(self, cost, acceleration_limits, dt):
        super().__init__(cost)
        self.scale = 1. / (np.asarray(acceleration_limits, dtype=float) * dt**2)

    def compute_error(self, configuration):
        return self.scale * super().compute_error(configuration)

    def compute_jacobian(self, configuration):
        return self.scale[:, None] * super().compute_jacobian(configuration)
