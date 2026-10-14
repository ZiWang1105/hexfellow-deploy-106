"""Eye-to-hand calibration: X C_i = G_i Y, transforms map source to destination.

X = mount_from_camera_optical; Y = hand_from_board;
G_i = mount_from_hand (measured FK); C_i = camera_optical_from_board (PnP).
For a moving camera mount: base_from_camera(q) = base_from_mount(q) @ X.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def transform(rotation, translation):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, np.asarray(translation).reshape(3)
    return t


def inverse(t):
    return transform(t[:3, :3].T, -t[:3, :3].T @ t[:3, 3])


def validate(transforms):
    a = np.asarray(transforms, dtype=float)
    if a.ndim != 3 or a.shape[1:] != (4, 4) or not np.isfinite(a).all():
        raise ValueError('Expected finite N x 4 x 4 transforms')
    r = a[:, :3, :3]
    if (not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-6)
        or not np.allclose(np.linalg.det(r), 1, atol=1e-6)):
        raise ValueError('Invalid rigid transform')
    return a


def residual(g, c, x, y, position_sigma=.002, angle_sigma=.01):
    left, right = x @ c, g @ y
    dp = (left[:, :3, 3] - right[:, :3, 3]) / position_sigma
    dr = Rotation.from_matrix(right[:, :3, :3].transpose(0, 2, 1) @ left[:, :3, :3]).as_rotvec() / angle_sigma
    return np.concatenate((dp, dr), axis=1).ravel()


def retract(t, delta):
    # Local SO(3) update plus translation in destination axes; rotations remain proper.
    return transform(Rotation.from_rotvec(delta[3:]).as_matrix() @ t[:3, :3], t[:3, 3]+delta[:3])


def jacobian(g, c, x, y, position_sigma=.002, angle_sigma=.01):
    eps = 1e-6
    columns = []
    for j in range(12):
        d = np.zeros(6); d[j % 6] = eps
        xp, xm = (retract(x, d), retract(x, -d)) if j < 6 else (x, x)
        yp, ym = (y, y) if j < 6 else (retract(y, d), retract(y, -d))
        columns.append((residual(g, c, xp, yp, position_sigma, angle_sigma)-residual(g, c, xm, ym, position_sigma, angle_sigma))/(2*eps))
    return np.column_stack(columns)


def summarize(g, c, x, y):
    r = residual(g, c, x, y, 1, 1).reshape(-1, 6)
    p, a = np.linalg.norm(r[:, :3], axis=1)*1000, np.rad2deg(np.linalg.norm(r[:, 3:], axis=1))
    return {'count': len(g), 'position_mm': p.tolist(), 'angle_deg': a.tolist(),
            'position_rms_mm': float(np.sqrt(np.mean(p*p))),
            'position_max_mm': float(p.max()), 'angle_rms_deg': float(np.sqrt(np.mean(a*a)))}


def solve(g, c, *, max_iterations=2000, position_sigma=.002, angle_sigma=.01):
    """Park initialization, then joint robust preconditioned gradient descent.

    Central-difference Jacobian, pseudo-Huber loss, diagonal metric and Armijo
    backtracking. No unconstrained 4x4 matrix optimization. Returned convergence
    and observability must be inspected; a fitted matrix is not qualification.
    """
    import cv2
    g, c = validate(g), validate(c)
    if len(g) != len(c) or len(g) < 6:
        raise ValueError('At least six paired, diverse poses are required')
    if min(position_sigma, angle_sigma) <= 0 or max_iterations < 1:
        raise ValueError('Positive scales and iteration limit required')
    relative = Rotation.from_matrix(g[0, :3, :3].T @ g[:, :3, :3]).as_rotvec()
    sv = np.linalg.svd(relative, compute_uv=False)
    if sv[1] < .03:
        raise ValueError('Insufficient rotations about two independent axes')
    gi = np.array([inverse(t) for t in g])
    rx, tx = cv2.calibrateHandEye(list(gi[:, :3, :3]), list(gi[:, :3, 3]),
        list(c[:, :3, :3]), list(c[:, :3, 3]), method=cv2.CALIB_HAND_EYE_PARK)
    x = validate([transform(rx, tx)])[0]
    estimates = gi @ x @ c
    y = transform(Rotation.from_matrix(estimates[:, :3, :3]).mean().as_matrix(), np.median(estimates[:, :3, 3], axis=0))
    def loss(r): return float(np.sum(4*(np.sqrt(1+(r/2)**2)-1)))
    history = []; converged = False; reason = 'iteration_limit'
    for iteration in range(max_iterations):
        r = residual(g, c, x, y, position_sigma, angle_sigma)
        value = loss(r); history.append(value)
        j = jacobian(g, c, x, y, position_sigma, angle_sigma)
        w = 1/np.sqrt(1+(r/2)**2)
        gradient = j.T @ (w*r)
        diagonal = np.maximum(np.sum(w[:, None]*j*j, axis=0), 1e-8)
        direction = -gradient/diagonal
        if np.linalg.norm(gradient/np.sqrt(diagonal), np.inf) < 1e-6:
            converged = True; reason = 'gradient_tolerance'; break
        slope = gradient @ direction
        alpha = 1.
        for _ in range(30):
            xn, yn = retract(x, alpha*direction[:6]), retract(y, alpha*direction[6:])
            new_value = loss(residual(g, c, xn, yn, position_sigma, angle_sigma))
            if new_value <= value+1e-4*alpha*slope: break
            alpha *= .5
        else:
            reason = 'line_search_failed'; break
        x, y = xn, yn
        if abs(value-new_value) < 1e-10*(1+value):
            converged = True; reason = 'objective_tolerance'; break
    singular = np.linalg.svd(jacobian(g, c, x, y, position_sigma, angle_sigma), compute_uv=False)
    if singular[-1]/singular[0] < 1e-6:
        raise ValueError('Unobservable or severely ill-conditioned pose set')
    return {'T_mount_from_camera': x.tolist(), 'T_hand_from_board': y.tolist(),
            'method': 'Park seed + joint pseudo-Huber gradient descent with Armijo search',
            'converged': converged, 'termination': reason, 'iterations': iteration+1,
            'initial_loss': history[0], 'final_loss': loss(residual(g,c,x,y,position_sigma,angle_sigma)),
            'jacobian_singular_values': singular.tolist(), 'training': summarize(g,c,x,y),
            'position_sigma_m': position_sigma, 'angle_sigma_rad': angle_sigma,
            'hardware_qualified': False}
