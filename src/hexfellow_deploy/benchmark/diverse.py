"""Frozen path specifications for generalization tests, not controller inputs.

Joint-space constructions provide independently reachable Cartesian requests.
Only FK poses and local body derivatives are exposed to the public IK adapter.
Cartesian constructions deliberately test fixed-orientation workspace limits.
"""
import numpy as np
import pinocchio as pin

from .trajectories import Trajectory, quintic, validate_scale


class DiverseTrajectory(Trajectory):
    def __init__(self, model, initial_q, spec, speed):
        validate_scale(speed, 1.)
        self.model, self.data = model, model.createData()
        self.q0 = np.asarray(initial_q, dtype=float).copy()
        self.spec = dict(spec)
        self.moving_arms = list(spec.get('arms', ['right']))
        if not self.moving_arms or len(set(self.moving_arms)) != len(self.moving_arms) or any(
                s not in ('left', 'right') for s in self.moving_arms):
            raise ValueError('arms must be a nonempty unique subset of left/right')
        self.dual = len(self.moving_arms) == 2
        self.kind, self.speed = spec['name'], speed
        self.settle_s, self.final_hold_s = 1., 1.5
        self.frame_ids = {s: model.getFrameId(s+'_tcp') for s in ('left', 'right')}
        pin.framesForwardKinematics(model, self.data, self.q0)
        self.initial = {s: self.data.oMf[f].copy() for s, f in self.frame_ids.items()}
        self.joint_path = spec['type'] == 'joint'
        if self.joint_path:
            self.delta = np.asarray(spec['delta'], dtype=float)
            self.waves = np.asarray(spec.get('waves', [[0.]*7]), dtype=float)
            if self.delta.shape != (7,) or self.waves.ndim != 2 or self.waves.shape[1] != 7 or not (
                    np.isfinite(self.delta).all() and np.isfinite(self.waves).all()):
                raise ValueError('Finite seven-joint path coefficients required')
            margin = np.inf
            for u in np.linspace(0, 1, 1001):
                q, _, _ = self.joint_state(u)
                margin = min(margin, np.min(q-model.lowerPositionLimit), np.min(model.upperPositionLimit-q))
            if margin < .012:
                raise ValueError(f'Offline path violates joint position margin: {margin}')
            self.geometry_margin_rad = float(margin)
            self.travel_s = 1.
            peak = max(np.linalg.norm(v[:3]) for u in np.linspace(0, 1, 2001)
                       for s, v in self.sample(self.settle_s+u)[1].items() if s in self.moving_arms)
            if peak < 1e-8:
                raise ValueError('Path must move the TCP')
            self.travel_s = peak/speed
            self.motion_end_s = self.settle_s+2*self.travel_s+.25
        elif spec['type'] == 'cartesian':
            self.base = Trajectory(self.initial, spec['kind'], speed, True, spec['distance'])
            self.basis = np.asarray(spec.get('basis', np.eye(3)), dtype=float)
            if self.basis.shape != (3, 3) or not np.isfinite(self.basis).all() or not np.allclose(
                    self.basis.T@self.basis, np.eye(3), atol=1e-10) or np.linalg.det(self.basis) < .999999:
                raise ValueError('Cartesian basis must be a proper rotation')
            self.travel_s, self.motion_end_s = self.base.travel_s, self.base.motion_end_s
        else:
            raise ValueError('Path type must be joint or cartesian')
        self.duration_s = self.motion_end_s+self.final_hold_s

    def joint_state(self, u):
        h, dh, ddh = quintic(u)
        displacement, first, second = self.delta*h, self.delta.copy(), np.zeros(7)
        for k, wave in enumerate(self.waves, 1):
            a = k*np.pi
            displacement += wave*np.sin(a*h)
            first += wave*a*np.cos(a*h)
            second -= wave*a*a*np.sin(a*h)
        q, v, acc = self.q0.copy(), np.zeros(self.model.nv), np.zeros(self.model.nv)
        for side in self.moving_arms:
            part = slice(1, 8) if side == 'left' else slice(8, 15)
            signs = np.array([-1, 1, -1, 1, -1, 1, -1]) if side == 'left' else np.ones(7)
            q[part] += signs*displacement
            v[part] = signs*first*dh
            acc[part] = signs*(second*dh**2+first*ddh)
        return q, v, acc

    def sample(self, time_s):
        if self.joint_path:
            elapsed = time_s-self.settle_s
            reverse = elapsed > self.travel_s+.25
            u = ((elapsed-self.travel_s-.25) if reverse else elapsed)/self.travel_s
            q, v, acc = self.joint_state(1-u if reverse else u)
            v *= (-1 if reverse else 1)/self.travel_s
            acc /= self.travel_s**2
            pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)
            pin.updateFramePlacements(self.model, self.data)
            poses, twists, derivatives = {}, {}, {}
            for side, frame in self.frame_ids.items():
                poses[side] = self.data.oMf[frame].copy()
                J = pin.getFrameJacobian(self.model, self.data, frame, pin.LOCAL)
                dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, frame, pin.LOCAL)
                twists[side], derivatives[side] = J@v, J@acc+dJ@v
            return poses, twists, derivatives
        poses, twists, derivatives = self.base.sample(time_s)
        for side in poses:
            if side not in self.moving_arms:
                poses[side], twists[side], derivatives[side] = self.initial[side].copy(), np.zeros(6), np.zeros(6)
                continue
            R, v, a = poses[side].rotation, twists[side], derivatives[side]
            world_v = self.basis@R@v[:3]
            world_a = self.basis@R@(a[:3]+np.cross(v[3:], v[:3]))
            poses[side].translation = self.initial[side].translation+self.basis@(
                poses[side].translation-self.initial[side].translation)
            twists[side] = np.r_[R.T@world_v, v[3:]]
            derivatives[side] = np.r_[R.T@world_a-np.cross(v[3:], R.T@world_v), a[3:]]
        return poses, twists, derivatives
