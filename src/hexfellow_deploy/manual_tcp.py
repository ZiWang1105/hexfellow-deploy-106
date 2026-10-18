"""Leased operator TCP goals, enforced by the control owner rather than the GUI."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .interfaces.contract import Action


class ManualTCPControl:
    def __init__(self, executor, period_s):
        config = executor.ik.core.config
        self.executor, self.dt = executor, period_s
        self.linear_speed = config.target_linear_velocity_limit
        self.angular_speed = config.target_angular_velocity_limit
        if not all(math.isfinite(v) and v > 0 for v in
                   (period_s, self.linear_speed, self.angular_speed)):
            raise ValueError('Manual TCP requires finite positive profile rates and period')
        self.session = self.deadline = None
        self.targets = self.goals = None
        self.last_created = -math.inf
        self.reason = 'disabled'

    def stop(self, measured, reason='operator HOLD'):
        self.executor.hold(measured)
        self.executor.gripper = dict(left=None, right=None)
        self.session = self.deadline = None
        self.targets = self.goals = None
        self.reason = reason

    def receive(self, request, measured, now):
        try:
            operation = request['operation']
            if operation in ('hold', 'final_hold'):
                self.stop(measured)
                return
            created = float(request['created_s'])
            if not math.isfinite(created) or not 0 <= now-created < .25 or created <= self.last_created:
                raise ValueError('Operator input expired or out of order; press E to enable again')
            self.last_created = created
            if operation == 'interactive_enable':
                self.stop(measured)
                self.executor.kinematics.update(measured['joint_positions'])
                self.targets = {s: self.executor.kinematics.pose(s).copy() for s in ('left', 'right')}
                self.session = request['id']
                self.reason = 'enabled; drag a handle'
                return
            if operation != 'interactive_target' or self.session is None or request.get('session') != self.session:
                raise ValueError('Interactive target requires a fresh E enable')
            if self.deadline is not None and now >= self.deadline:
                raise ValueError('Operator lease expired; press E to enable again')
            expires = float(request['expires_s'])
            if not math.isfinite(expires) or not created < expires <= created+.25 or now >= expires:
                raise ValueError('Invalid operator lease')
            action = Action.from_dict(request['action'])
            if action.navigation is not None or action.skill is not None:
                raise ValueError('Manual TCP input cannot command navigation or skills')
            self.executor.kinematics.update(measured['joint_positions'])
            goals = {}
            for side in ('left', 'right'):
                pose = getattr(action, side)
                current = self.executor.kinematics.pose(side)
                goal = current.copy()
                goal.translation = np.asarray(pose.position_m)
                goal.rotation = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
                goals[side] = goal
            self.goals, self.deadline = goals, expires
            self.reason = 'dragging'
        except (ValueError, KeyError, TypeError) as error:
            self.stop(measured, str(error))
            raise ValueError(str(error)) from error

    def step(self, measured, now):
        if self.deadline is None:
            return
        if now >= self.deadline:
            self.stop(measured, 'Operator lease expired; press E to enable again')
            return
        for side, goal in self.goals.items():
            target = self.targets[side]
            delta = goal.translation-target.translation
            target.translation += delta*min(1., self.linear_speed*self.dt/max(np.linalg.norm(delta), 1e-12))
            delta_r = Rotation.from_matrix(target.rotation.T @ goal.rotation).as_rotvec()
            target.rotation = target.rotation @ Rotation.from_rotvec(
                delta_r*min(1., self.angular_speed*self.dt/max(np.linalg.norm(delta_r), 1e-12))).as_matrix()
        # Ordinary IK shaping, joint/effort limits and MIT validation still apply.
        # No Action.submit: the manual interface never changes jaws or base.
        self.executor.ik.track({s: (p.translation.copy(), p.rotation.copy()) for s,p in self.targets.items()})
        self.executor.gripper = dict(left=None, right=None)

    def report(self):
        return dict(enabled=self.session is not None, session=self.session,
                    expires_s=self.deadline, reason=self.reason,
                    target_speed_m_s=self.linear_speed, target_speed_rad_s=self.angular_speed)
