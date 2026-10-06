#!/usr/bin/env python3
"""Offline 3D-model / rectified-LEFT 2D landmark camera fit. No robot access.

Input observations need p_mount (meters), real_uv (pixels), and pose_id.
T_mount_from_camera maps LEFT optical coordinates into the moving lift frame.
The caller must use synchronized measured joints to calculate p_mount.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def camera_from_parameters(parameters):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:]
    return result


def project(points, mount_from_camera, k):
    camera_from_mount = np.linalg.inv(mount_from_camera)
    camera_points = np.asarray(points) @ camera_from_mount[:3, :3].T + camera_from_mount[:3, 3]
    pixels = camera_points @ np.asarray(k).T
    return pixels[:, :2] / pixels[:, 2:3], camera_points[:, 2]


def fit(points, pixels, k, threshold_px=6.0, refine_all=False):
    points, pixels, k = np.asarray(points, float), np.asarray(pixels, float), np.asarray(k, float)
    if points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2) or len(points) < 6:
        raise ValueError('Need at least six 3D-to-2D landmark pairs')
    if k.shape != (3, 3) or not all(np.isfinite(a).all() for a in (points, pixels, k)):
        raise ValueError('Expected finite points and 3x3 intrinsics')
    if not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Expected rectified pinhole intrinsics')
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError('Reprojection threshold must be positive')
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-8) < 2:
        raise ValueError('Collinear 3D points cannot determine camera pose')
    cv2.setRNGSeed(730)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(points), np.ascontiguousarray(pixels), k, np.zeros(5),
        iterationsCount=3000, reprojectionError=threshold_px, confidence=.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError('Fewer than six RANSAC inliers; check landmark matches')
    # Approximate manual landmarks can have bounded errors rather than a clean
    # inlier/outlier split. Optionally refine ALL pairs with a robust loss after
    # RANSAC initialization; the 6px classification remains a separate report.
    selected = np.arange(len(points)) if refine_all else inliers.ravel()
    initial = np.r_[rvec.ravel(), tvec.ravel()]

    def residual(parameters):
        camera_from_mount = camera_from_parameters(parameters)
        p = points[selected] @ camera_from_mount[:3, :3].T + camera_from_mount[:3, 3]
        # Penalize crossing behind the camera rather than accepting mirrored poses.
        h = p @ k.T
        uv = h[:, :2] / np.maximum(h[:, 2:3], 1e-6)
        return (uv - pixels[selected]).ravel()

    result = least_squares(residual, initial, loss='soft_l1', f_scale=3.0,
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
    if not result.success:
        raise ValueError(f'Reprojection refinement failed: {result.message}')
    mount_from_camera = np.linalg.inv(camera_from_parameters(result.x))
    predicted, depths = project(points, mount_from_camera, k)
    errors = np.linalg.norm(predicted - pixels, axis=1)
    if np.any(depths[selected] <= 0):
        raise ValueError('Fitted landmarks lie behind the camera')
    inlier_mask = (errors <= threshold_px) & (depths > 0)
    if inlier_mask.sum() < 6:
        raise ValueError('Refined camera has fewer than six inliers')
    return dict(T_mount_from_camera=mount_from_camera.tolist(),
                refinement='all pairs, soft_l1' if refine_all else 'RANSAC inliers, soft_l1',
                robust_loss_scale_px=3.0, reprojection_threshold_px=float(threshold_px),
                errors_px=errors.tolist(), inliers=inlier_mask.tolist(),
                inlier_rms_px=float(np.sqrt(np.mean(errors[inlier_mask] ** 2))),
                all_rms_px=float(np.sqrt(np.mean(errors ** 2))),
                point_spread_singular_values_m=np.linalg.svd(points-points.mean(axis=0), compute_uv=False).tolist())


def evaluate(observations, k, threshold_px=6, refine_all=False):
    points = np.asarray([row['p_mount'] for row in observations], float)
    pixels = np.asarray([row['real_uv'] for row in observations], float)
    pose_ids = np.asarray([row['pose_id'] for row in observations])
    output = fit(points, pixels, k, threshold_px, refine_all)
    output['per_pose'] = []
    output['leave_one_pose_out'] = []
    for pose_id in np.unique(pose_ids):
        mask = pose_ids == pose_id
        errors = np.asarray(output['errors_px'])[mask]
        output['per_pose'].append(dict(pose_id=int(pose_id), count=int(mask.sum()),
            inliers=int(np.asarray(output['inliers'])[mask].sum()),
            rms_px=float(np.sqrt(np.mean(errors**2))), max_px=float(errors.max())))
        try:
            other = fit(points[~mask], pixels[~mask], k, threshold_px, refine_all)
            predicted, depths = project(points[mask], np.asarray(other['T_mount_from_camera']), k)
            errors = np.linalg.norm(predicted - pixels[mask], axis=1)
            output['leave_one_pose_out'].append(dict(held_out_pose=int(pose_id),
                training_inliers=sum(other['inliers']),
                held_out_rms_px=float(np.sqrt(np.mean(errors**2))),
                held_out_max_px=float(errors.max()), held_out_positive_depth=bool(np.all(depths > 0)),
                T_mount_from_camera=other['T_mount_from_camera']))
        except ValueError as error:
            output['leave_one_pose_out'].append(dict(held_out_pose=int(pose_id), error=str(error)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold-px', type=float, default=6)
    parser.add_argument('--refine-all', action='store_true',
                        help='Robustly refine all pairs after RANSAC initialization')
    args = parser.parse_args()
    document = json.loads(args.points.read_text())
    if document.get('camera') != 'rectified LEFT optical':
        raise ValueError('Landmarks must use the rectified LEFT view and matching K')
    report = evaluate(document['observations'], document['K'], args.threshold_px, args.refine_all)
    report.update(status='provisional landmark fit; physical matches require review',
                  camera_frame='rectified LEFT optical; x right, y down, z forward',
                  mount_frame='maver_x4_asm_URDF_link_1',
                  point_file=str(args.points.resolve()), offline_only=True, hardware_commands_sent=0,
                  active_configuration_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('inlier_rms_px', 'all_rms_px', 'per_pose')}, indent=2))


if __name__ == '__main__':
    main()
