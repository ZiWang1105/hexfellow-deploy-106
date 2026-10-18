"""Analytic C2 targets with explicit derivatives and a stationary final hold."""
from dataclasses import dataclass

import numpy as np
import pinocchio as pin


def quintic(u):
    """Unit displacement and its first two derivatives in normalized time."""
    if u <= 0:
        return 0., 0., 0.
    if u >= 1:
        return 1., 0., 0.
    return (10*u**3-15*u**4+6*u**5,
            30*u**2-60*u**3+30*u**4, 60*u-180*u**2+120*u**3)


def validate_scale(speed, distance):
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('Speed must be finite and positive')
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('Distance must be finite and positive')


@dataclass
class Trajectory:
    initial: dict
    kind: str
    speed: float
    dual: bool = False
    distance: float | None = None
    settle_s: float = 1.
    final_hold_s: float = 1.5

    def __post_init__(self):
        if self.kind not in ('vertical', 'lateral', 'forward', 'circle', 'figure8', 'rotate', 'rounded_rectangle'):
            raise ValueError(f'Unknown trajectory {self.kind}')
        if self.distance is None:
            self.distance = dict(vertical=.4, lateral=.25, forward=.18,
                                 circle=.08, figure8=.07, rotate=.20, rounded_rectangle=.12)[self.kind]
        validate_scale(self.speed, self.distance)
        self.curve = self.kind in ('circle', 'figure8', 'rounded_rectangle')
        if self.kind == 'rounded_rectangle':
            from scipy.interpolate import CubicSpline
            corners = np.array([[0, 0, 0], [0, 1, 0], [0, 1.5, .5], [0, 1.5, 1.5],
                                [0, 1, 2], [0, 0, 2], [0, -.5, 1.5], [0, -.5, .5], [0, 0, 0]])
            self.spline = CubicSpline(np.linspace(0, 2*np.pi, 9), corners*self.distance, bc_type='periodic')
        if self.curve:
            # Determine time scaling from the analytic unit-duration derivative.
            unit_peak = max(np.linalg.norm(self._curve(u)[1]) for u in np.linspace(0, 1, 2001))
            self.travel_s = unit_peak/self.speed
            self.motion_end_s = self.settle_s+self.travel_s
        else:
            self.travel_s = 1.875*self.distance/self.speed
            self.motion_end_s = self.settle_s+2*self.travel_s+.25
        self.duration_s = self.motion_end_s+self.final_hold_s

    def _curve(self, u):
        h, dh, ddh = quintic(u)
        a, w, aw = 4*np.pi*h, 4*np.pi*dh, 4*np.pi*ddh
        r = self.distance
        if self.kind == 'rounded_rectangle':
            p, da, dda = (self.spline(a, order) for order in range(3))
        elif self.kind == 'circle':
            p = r*np.array([0., np.sin(a), 1-np.cos(a)])
            da = r*np.array([0., np.cos(a), np.sin(a)])
            dda = r*np.array([0., -np.sin(a), np.cos(a)])
        else:
            p = r*np.array([0., np.sin(a), .5*np.sin(2*a)])
            da = r*np.array([0., np.cos(a), np.cos(2*a)])
            dda = r*np.array([0., -np.sin(a), -2*np.sin(2*a)])
        return p, da*w, dda*w*w+da*aw

    def sample(self, time_s):
        elapsed = time_s-self.settle_s
        if self.curve:
            delta, velocity, acceleration = self._curve(elapsed/self.travel_s)
            velocity, acceleration = velocity/self.travel_s, acceleration/self.travel_s**2
            angle = angular_velocity = angular_acceleration = 0.
        else:
            reverse = elapsed > self.travel_s+.25
            u = ((elapsed-self.travel_s-.25) if reverse else elapsed)/self.travel_s
            h, dh, ddh = quintic(u)
            if reverse:
                h, dh, ddh = 1-h, -dh, -ddh
            axis = dict(vertical=2, lateral=1, forward=0, rotate=2)[self.kind]
            delta, velocity, acceleration = (np.zeros(3) for _ in range(3))
            delta[axis] = self.distance*h
            velocity[axis] = self.distance*dh/self.travel_s
            acceleration[axis] = self.distance*ddh/self.travel_s**2
            angle, angular_velocity, angular_acceleration = (
                (.35*h, .35*dh/self.travel_s, .35*ddh/self.travel_s**2)
                if self.kind == 'rotate' else (0., 0., 0.))
        poses, twists, accelerations = {}, {}, {}
        for side, initial in self.initial.items():
            poses[side] = initial.copy()
            twists[side], accelerations[side] = np.zeros(6), np.zeros(6)
            if side == 'left' and not self.dual:
                continue
            mirror = np.array([1., -1. if side == 'left' else 1., 1.])
            rotation = initial.rotation @ pin.exp3(np.array([angle, 0., 0.]))
            poses[side] = pin.SE3(rotation, initial.translation+mirror*delta)
            linear = rotation.T@(mirror*velocity)
            angular = np.array([angular_velocity, 0., 0.])
            twists[side] = np.r_[linear, angular]
            accelerations[side] = np.r_[rotation.T@(mirror*acceleration)-np.cross(angular, linear),
                                        angular_acceleration, 0., 0.]
        return poses, twists, accelerations

    def motion_mask(self, times):
        return (times >= self.settle_s) & (times < self.motion_end_s)


class ScrewTrajectory(Trajectory):
    """Cartesian rigid-arm swing, generated from an X7 joint screw offline.

    The controller receives only poses and their derivatives, not the joint
    path. This supplies a reachable translation-plus-rotation test independent
    of fixed-orientation wrist workspace limits.
    """
    def __init__(self, initial, screws, speed, dual=False, distance=.8):
        validate_scale(speed, distance)
        if any(np.asarray(v).shape != (6,) or not np.isfinite(v).all() for v in screws.values()):
            raise ValueError('Joint screws must be finite 6-vectors')
        if np.linalg.norm(screws['right'][:3]) < 1e-8:
            raise ValueError('The right screw must move the TCP position')
        self.initial, self.screws = initial, screws
        self.kind, self.speed, self.dual, self.distance = 'swing', speed, dual, distance
        self.settle_s, self.final_hold_s = 1., 1.5
        self.travel_s = 1.875*distance*np.linalg.norm(screws['right'][:3])/speed
        self.motion_end_s = self.settle_s+2*self.travel_s+.25
        self.duration_s = self.motion_end_s+self.final_hold_s

    def sample(self, time_s):
        elapsed = time_s-self.settle_s
        reverse = elapsed > self.travel_s+.25
        u = ((elapsed-self.travel_s-.25) if reverse else elapsed)/self.travel_s
        h, dh, ddh = quintic(u)
        if reverse:
            h, dh, ddh = 1-h, -dh, -ddh
        targets, twists, accelerations = {}, {}, {}
        for side in self.initial:
            screw = self.screws[side] if self.dual or side == 'right' else np.zeros(6)
            targets[side] = self.initial[side]*pin.exp6(screw*self.distance*h)
            twists[side] = screw*self.distance*dh/self.travel_s
            accelerations[side] = screw*self.distance*ddh/self.travel_s**2
        return targets, twists, accelerations


class JointPathTrajectory(Trajectory):
    """An independently specified multi-joint path, exposed only as Cartesian targets.

    Unlike the single-axis screw, every arm joint moves. The path is an offline
    target generator: its joint states never enter the controller's feedback,
    reference history or optimization. Body derivatives use J and its time
    derivative, with the same definitions as the analytic Cartesian paths.
    """
    def __init__(self, model, initial_q, speed, dual=False, distance=1.):
        validate_scale(speed, distance)
        self.model, self.data = model, model.createData()
        self.q0 = np.asarray(initial_q, dtype=float).copy()
        self.kind, self.speed, self.dual, self.distance = 'joint_curve', speed, dual, distance
        self.frame_ids = {s: model.getFrameId(s+'_tcp') for s in ('left', 'right')}
        self.delta = np.zeros(model.nv)
        self.delta[8:] = distance*np.array([.8, -.3, .25, -.25, .2, .15, .1])
        if dual:
            self.delta[1:8] = self.delta[8:]*np.array([-1., 1., -1., 1., -1., 1., -1.])
        self.settle_s, self.final_hold_s = 1., 1.5
        peak = 0.
        for u in np.linspace(0., 1., 2001):
            h, dh, _ = quintic(u)
            jac = pin.computeFrameJacobian(model, self.data, self.q0+self.delta*h,
                                           self.frame_ids['right'], pin.LOCAL)
            peak = max(peak, np.linalg.norm((jac@self.delta*dh)[:3]))
        self.travel_s = peak/speed
        self.motion_end_s = self.settle_s+2*self.travel_s+.25
        self.duration_s = self.motion_end_s+self.final_hold_s

    def sample(self, time_s):
        elapsed = time_s-self.settle_s
        reverse = elapsed > self.travel_s+.25
        u = ((elapsed-self.travel_s-.25) if reverse else elapsed)/self.travel_s
        h, dh, ddh = quintic(u)
        if reverse:
            h, dh, ddh = 1-h, -dh, -ddh
        q = self.q0+self.delta*h
        velocity, acceleration = self.delta*dh/self.travel_s, self.delta*ddh/self.travel_s**2
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, velocity)
        pin.updateFramePlacements(self.model, self.data)
        targets, twists, derivatives = {}, {}, {}
        for side, frame in self.frame_ids.items():
            targets[side] = self.data.oMf[frame].copy()
            jac = pin.getFrameJacobian(self.model, self.data, frame, pin.LOCAL)
            jac_dot = pin.getFrameJacobianTimeVariation(self.model, self.data, frame, pin.LOCAL)
            twists[side] = jac@velocity
            derivatives[side] = jac@acceleration+jac_dot@velocity
        return targets, twists, derivatives


def envelope_report(trajectory, profile, samples=4001):
    """Dense request screening, not a proof of joint/dynamic feasibility."""
    times = np.linspace(0., trajectory.duration_s, samples)
    linear_velocity, angular_velocity, linear_acceleration, angular_acceleration = [], [], [], []
    moving = getattr(trajectory, 'moving_arms', ['right'])
    for t in times:
        target, twist, derivative = trajectory.sample(t)
        linear_velocity.append([target[s].rotation@twist[s][:3] for s in moving])
        angular_velocity.append([target[s].rotation@twist[s][3:] for s in moving])
        linear_acceleration.append([target[s].rotation@(derivative[s][:3]+np.cross(twist[s][3:], twist[s][:3])) for s in moving])
        angular_acceleration.append([target[s].rotation@derivative[s][3:] for s in moving])
    values = dict(linear_velocity=np.asarray(linear_velocity), angular_velocity=np.asarray(angular_velocity),
                  linear_acceleration=np.asarray(linear_acceleration), angular_acceleration=np.asarray(angular_acceleration))
    for part in ('linear', 'angular'):
        values[part+'_jerk'] = np.gradient(values[part+'_acceleration'], times, axis=0)
    maxima = {key: float(np.max(np.linalg.norm(value, axis=-1))) for key, value in values.items()}
    limits = dict(linear_velocity=profile.target_linear_velocity_m_s,
                  linear_acceleration=profile.target_linear_acceleration_m_s2,
                  linear_jerk=profile.target_linear_jerk_m_s3,
                  angular_velocity=profile.target_angular_velocity_rad_s,
                  angular_acceleration=profile.target_angular_acceleration_rad_s2,
                  angular_jerk=profile.target_angular_jerk_rad_s3)
    ratios = {key: maxima[key]/limits[key] for key in maxima}
    return dict(maxima=maxima, ratios=ratios, exceeds=[key for key, ratio in ratios.items() if ratio > 1.001],
                samples=samples, limitation='Dense target sampling; jerk is differentiated acceleration. '
                'Passing does not establish joint-space reachability or dynamic feasibility.')
