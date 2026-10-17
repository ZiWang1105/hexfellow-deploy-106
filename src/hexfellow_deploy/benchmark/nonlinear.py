"""AgileX-style nonlinear pose objective adapted to the owned X7 model.

Reference: AgileX QuestArmTeleop ArmIK, revision
145b80360cf2c5f2e63179086d0aab2cb563826c. The objective is squared weighted
SE(3) log error plus zero-posture and measured-posture regularization. This is
an independent implementation, not the original NERO solver or its geometry.
CasADi symbolic FK is constructed from the owned Pinocchio model, avoiding a
second Pinocchio installation or a dependency on the agilex-deploy checkout.
The returned position solution is not an executable motor command. The shared
reference constraints and MIT synthesis must still process it.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


def symbolic_fk(model):
    import casadi as ca
    q = ca.SX.sym('q', model.nq)
    data = model.createData()
    neutral = pin.neutral(model)
    transforms = [ca.SX.eye(4)]
    for jid in range(1, model.njoints):
        joint = model.joints[jid]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError('Benchmark symbolic FK supports scalar arm/lift joints only')
        axis = pin.computeJointJacobian(model, data, neutral, jid)[:, joint.idx_v]
        motion = ca.SX.eye(4)
        if np.linalg.norm(axis[3:]) > .5:
            if np.linalg.norm(axis[:3]) > 1e-10:
                raise ValueError('Helical joints are not supported')
            skew = ca.DM(pin.skew(axis[3:]))
            angle = q[joint.idx_q]
            motion[:3, :3] = ca.SX.eye(3)+ca.sin(angle)*skew+(1-ca.cos(angle))*(skew@skew)
        else:
            motion[:3, 3] = ca.DM(axis[:3])*q[joint.idx_q]
        transforms.append(transforms[model.parents[jid]] @ ca.DM(model.jointPlacements[jid].homogeneous) @ motion)
    frames = {side: transforms[model.frames[model.getFrameId(side+'_tcp')].parent]
              @ ca.DM(model.frames[model.getFrameId(side+'_tcp')].placement.homogeneous)
              for side in ('left', 'right')}
    return q, frames


def se3_log(matrix):
    """SE(3) log with finite identity derivatives; targets must avoid the pi cut."""
    import casadi as ca
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    vee = .5*ca.vertcat(rotation[2, 1]-rotation[1, 2],
                       rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1])
    sine = ca.sqrt(ca.dot(vee, vee)+1e-24)
    cosine = (ca.trace(rotation)-1)/2
    theta = ca.atan2(sine, cosine)
    omega = theta/sine*vee
    skew = ca.skew(omega)
    theta2 = ca.dot(omega, omega)
    safe_theta = ca.sqrt(theta2+1e-16)
    coefficient = ca.if_else(theta2 < 1e-6, 1/12+theta2/720+theta2**2/30240,
                            (1-safe_theta/(2*ca.tan(safe_theta/2)))/(theta2+1e-16))
    linear = (ca.SX.eye(3)-.5*skew+coefficient*(skew@skew))@translation
    return ca.vertcat(linear, omega)


class NonlinearPoseIK:
    """Warm-started IPOPT endpoint IK, retaining X7 position bounds."""
    def __init__(self, model, initial, *, w_pos=40., w_ori=4., w_reg=.01,
                 w_smooth=.1, max_iter=50):
        import casadi as ca
        self.model, self.data = model, model.createData()
        q, frames = symbolic_fk(model)
        self.fk = ca.Function('x7_fk', [q], [frames['left'], frames['right']])
        self.parameters = dict(w_pos=w_pos, w_ori=w_ori, w_reg=w_reg,
                               w_smooth=w_smooth, max_iter=max_iter)
        self.initial = np.asarray(initial).copy()
        # Lift is a fixed measured/session coordinate, never a decision variable.
        variable = ca.SX.sym('arm_q', model.nq-1)
        previous = ca.SX.sym('previous', model.nq-1)
        left, right = ca.SX.sym('left', 4, 4), ca.SX.sym('right', 4, 4)
        full = ca.vertcat(float(initial[0]), variable)
        cost = w_reg*ca.sumsqr(variable)+w_smooth*ca.sumsqr(variable-previous)
        for side, target in (('left', left), ('right', right)):
            frame = ca.substitute(frames[side], q, full)
            relative = ca.SX.eye(4)
            relative[:3, :3] = frame[:3, :3].T@target[:3, :3]
            relative[:3, 3] = frame[:3, :3].T@(target[:3, 3]-frame[:3, 3])
            error = se3_log(relative)
            cost += ca.sumsqr(w_pos*error[:3])+ca.sumsqr(w_ori*error[3:])
        self.solver = ca.nlpsol('x7_ipopt', 'ipopt',
            {'x': variable, 'p': ca.vertcat(previous, ca.vec(left), ca.vec(right)), 'f': cost},
            {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': False,
             'ipopt.max_iter': max_iter, 'ipopt.tol': 1e-4})
        self.last_status = None
        self.solve({s: pin.SE3(np.asarray(t)) for s, t in zip(('left', 'right'), self.fk(initial))}, initial)

    def solve(self, targets, measured):
        measured = np.asarray(measured, dtype=float)
        pin.framesForwardKinematics(self.model, self.data, measured)
        for side in ('left', 'right'):
            current = self.data.oMf[self.model.getFrameId(side+'_tcp')]
            if np.linalg.norm(pin.log3(current.rotation.T@targets[side].rotation)) > np.pi-1e-4:
                raise ValueError('Nonlinear benchmark target is on the SO(3) pi branch cut')
        parameters = np.r_[measured[1:], targets['left'].homogeneous.ravel(order='F'),
                           targets['right'].homogeneous.ravel(order='F')]
        solution = self.solver(x0=measured[1:], p=parameters,
                              lbx=self.model.lowerPositionLimit[1:], ubx=self.model.upperPositionLimit[1:])
        status = self.solver.stats()
        self.last_status = status['return_status']
        if not status['success']:
            raise RuntimeError(f'Nonlinear IK failed: {self.last_status}')
        result = np.r_[self.initial[0], np.asarray(solution['x']).reshape(-1)]
        if not np.all(np.isfinite(result)):
            raise ValueError('Nonlinear IK returned non-finite joints')
        return result

    def velocity(self, q, twists):
        """Local continuous derivative of a pose solution, not a speed override."""
        rows, desired = [], []
        for side in ('left', 'right'):
            rows.append(pin.computeFrameJacobian(self.model, self.data, q,
                        self.model.getFrameId(side+'_tcp'), pin.LOCAL)[:, 1:])
            desired.append(twists[side])
        jac = np.vstack(rows)
        return np.r_[0., np.linalg.lstsq(jac, np.concatenate(desired), rcond=1e-5)[0]]
