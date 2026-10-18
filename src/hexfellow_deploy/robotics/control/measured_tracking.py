"""Resolved-rate Cartesian feedback from measured joints to the latest goal.

Only velocity/acceleration requests have history. There is no independently
integrated Cartesian pose reference and no following-error speed multiplier.
"""
import numpy as np
import pinocchio as pin

from .target_motion import _tracking_jerk_with_braking_reserve, _validate_pose


class MeasuredPoseTracking:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.velocity = {}
        self.acceleration = {}
        self.observations = {}
        self.last_sample = None
        self.repeat_s = 0.
        self.progress_error = float('inf')
        self.progress_targets = {}

    def check_sample(self, sample, dt):
        if sample is None:
            return
        old = self.last_sample
        if old is not None and sample.epoch == old.epoch:
            if sample.sequence < old.sequence or sample.time_s < old.time_s:
                raise ValueError('out_of_order measured tracking target')
            if sample.sequence == old.sequence:
                if sample.time_s != old.time_s:
                    raise ValueError('inconsistent measured tracking target timestamp')
                self.repeat_s += dt
                if self.repeat_s > .1:
                    raise ValueError('stale measured tracking target')
                return
            if sample.time_s <= old.time_s:
                raise ValueError('out_of_order measured tracking target clock')
        elif old is not None and sample.epoch < old.epoch:
            raise ValueError('out_of_order measured tracking target epoch')
        self.last_sample, self.repeat_s = sample, 0.

    def making_progress(self, error, targets):
        changed = not self.progress_targets
        for side, target in targets.items():
            if side not in self.progress_targets:
                continue
            previous = self.progress_targets[side]
            changed |= (np.linalg.norm(target.translation-previous.translation) > self.config.target_translation_deadband
                        or np.linalg.norm(pin.log3(previous.rotation.T @ target.rotation)) > self.config.target_orientation_deadband)
        if changed or error < self.progress_error-.02:
            self.progress_error = error
            self.progress_targets = {s: t.copy() for s,t in targets.items()}
            return True
        return False

    def objective(self, side, task, configuration, target, dt, *,
                  target_twist=None, target_acceleration=None, measured_joint_velocity=None):
        target = _validate_pose(target)
        current = configuration.get_transform_frame_to_world(task.frame)
        error = np.r_[target.translation-current.translation,
                      pin.log3(target.rotation @ current.rotation.T)]
        desired = self.config.frame_gain / dt * error
        if np.linalg.norm(error[:3]) <= self.config.target_translation_deadband:
            desired[:3] = 0.
        if np.linalg.norm(error[3:]) <= self.config.target_orientation_deadband:
            desired[3:] = 0.
        source_acceleration = np.zeros(6)
        source_velocity = np.zeros(6)
        if target_twist is not None:
            twist = np.asarray(target_twist, dtype=float)
            if twist.shape != (6,) or not np.isfinite(twist).all():
                raise ValueError('Measured tracking target twist must be a finite 6-vector')
            source_velocity = np.r_[target.rotation @ twist[:3], target.rotation @ twist[3:]]
            desired += source_velocity
            if target_acceleration is not None:
                acceleration = np.asarray(target_acceleration, dtype=float)
                if acceleration.shape != (6,) or not np.isfinite(acceleration).all():
                    raise ValueError('Measured tracking target acceleration must be a finite 6-vector')
                source_acceleration = np.r_[
                    target.rotation @ (acceleration[:3] + np.cross(twist[3:], twist[:3])),
                    target.rotation @ acceleration[3:]]
        elif target_acceleration is not None:
            raise ValueError('Measured tracking acceleration requires a target twist')
        # LOCAL_WORLD_ALIGNED twist: measured TCP origin, world-aligned axes.
        jacobian = configuration.get_frame_jacobian(task.frame).copy()
        jacobian[:3] = current.rotation @ jacobian[:3]
        jacobian[3:] = current.rotation @ jacobian[3:]
        error_rate = np.zeros(6)
        if self.config.measured_error_derivative_gain:
            measured = np.asarray(measured_joint_velocity, dtype=float)
            if measured.shape != (configuration.model.nv,) or not np.isfinite(measured).all():
                raise ValueError('Measured error derivative requires finite measured joint velocity')
            measured_velocity = jacobian @ measured
            rotation_error = target.rotation @ current.rotation.T
            error_rate[:3] = source_velocity[:3] - measured_velocity[:3]
            error_rate[3:] = pin.Jlog3(rotation_error) @ (
                rotation_error.T @ source_velocity[3:] - measured_velocity[3:])
        # The derivative of v_target + K * pose_error includes K * error_rate.
        # Omitting it leaves a lightly damped third-order position loop when K
        # is raised. This opt-in term uses encoder velocities, never plant truth.
        desired_acceleration = source_acceleration + (
            self.config.measured_error_derivative_gain * self.config.frame_gain / dt * error_rate)
        velocity = self.velocity.get(side, np.zeros(6))
        acceleration = self.acceleration.get(side, np.zeros(6))
        jerk = np.zeros(6)
        cfg = self.config
        limits = ((slice(0, 3), cfg.target_linear_velocity_limit,
                   cfg.target_linear_acceleration_limit, cfg.target_linear_jerk_limit),
                  (slice(3, 6), cfg.target_angular_velocity_limit,
                   cfg.target_angular_acceleration_limit, cfg.target_angular_jerk_limit))
        for part, vmax, amax, jmax in limits:
            desired[part] *= min(1., .95*vmax/max(1e-12, np.linalg.norm(desired[part])))
            bandwidth = cfg.target_tracking_bandwidth_s_inv
            proposed = (bandwidth**2 * (desired[part]-velocity[part])
                        + 2*bandwidth*(desired_acceleration[part]-acceleration[part]))
            jerk[part] = _tracking_jerk_with_braking_reserve(
                proposed, acceleration[part], velocity[part], jmax, amax, vmax, dt)
        next_velocity = velocity + acceleration*dt + .5*jerk*dt**2
        next_acceleration = acceleration + jerk*dt
        self.velocity[side], self.acceleration[side] = next_velocity, next_acceleration
        weight = np.asarray(task.cost)
        weighted_jacobian = weight[:, None] * jacobian
        desired_displacement = weight * next_velocity * dt
        damping = task.lm_damping * float(desired_displacement @ desired_displacement)
        hessian = weighted_jacobian.T @ weighted_jacobian + damping*np.eye(configuration.model.nv)
        linear = -weighted_jacobian.T @ desired_displacement
        self.observations[side] = dict(measured_pose=current.homogeneous.copy(),
            target_pose=target.homogeneous.copy(), error_world=error.copy(),
            error_rate_world=error_rate.copy(),
            velocity_world=next_velocity.copy(), acceleration_world=next_acceleration.copy(),
            jerk_world=jerk.copy())
        return hessian, linear

    @property
    def settled(self):
        return bool(self.observations) and all(
            np.linalg.norm(o['error_world'][:3]) <= self.config.target_translation_deadband
            and np.linalg.norm(o['error_world'][3:]) <= self.config.target_orientation_deadband
            and np.linalg.norm(o['velocity_world'][:3]) <= self.config.settling_linear_velocity_m_s
            and np.linalg.norm(o['velocity_world'][3:]) <= self.config.settling_angular_velocity_rad_s
            for o in self.observations.values())
