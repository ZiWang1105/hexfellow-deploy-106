#!/usr/bin/env python3
"""Offline wrist-camera fit from measured FK and stationary-board PnP pairs.

Input JSON contains observations with pose_id, eligible (or accepted),
T_base_from_wrist and T_camera_from_board.  Camera metadata lives in camera
{K, distortion, serial, ...} or intrinsics {matrix, distortion}.  Ineligible
poses are retained in the report but excluded from fitting.  Every fifth
eligible observation is held out, with at least 12 eligible poses required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from policy_sim.calibration.wrist_handeye import camera_from_board, solve, validate


def _eligible(row):
    accepted = row.get('eligible', row.get('accepted')) is True
    return accepted and all(row.get(name, True) is not False
                            for name in ('stationary', 'synchronized', 'reachable'))


def _camera_metadata(data):
    camera = dict(data.get('camera', {}))
    intrinsics = data.get('intrinsics', {})
    if isinstance(intrinsics, dict):
        camera.setdefault('K', intrinsics.get('matrix', intrinsics.get('K')))
        camera.setdefault('distortion', intrinsics.get('distortion'))
    for name in ('K', 'distortion', 'serial', 'camera_frame', 'distortion_model'):
        if camera.get(name) is None and name in data:
            camera[name] = data[name]
    return camera


def reprojection_report(rows, g, x, y, camera):
    """Score raw image corners using fitted X,Y rather than stored PnP poses."""
    import cv2

    matrix = camera.get('K')
    if matrix is None:
        return {'available': False, 'reason': 'Camera K was not supplied'}
    k = np.asarray(matrix, dtype=float)
    distortion = np.asarray(camera.get('distortion') or [0] * 5, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Expected finite positive-focal-length 3x3 camera K')
    if not np.isfinite(distortion).all():
        raise ValueError('Expected finite camera distortion coefficients')
    # RealSense inverse-Brown coefficients are not OpenCV forward-Brown values.
    model = str(camera.get('distortion_model', '')).lower()
    if 'inverse' in model and np.any(distortion != 0):
        return {'available': False, 'reason': 'Nonzero inverse-Brown color distortion needs SDK rectification'}
    predictions = camera_from_board(g, x, y)
    all_errors, per_pose = [], []
    for row, c in zip(rows, predictions):
        object_points = row.get('object_points', row.get('board_object_points'))
        image_points = row.get('image_points', row.get('board_image_points'))
        if object_points is None or image_points is None:
            per_pose.append({'pose_id': row['pose_id'], 'available': False,
                             'reason': 'Raw board corners were not supplied'})
            continue
        objects = np.asarray(object_points, dtype=float).reshape(-1, 3)
        images = np.asarray(image_points, dtype=float).reshape(-1, 2)
        if len(objects) != len(images) or not len(objects) or not all(
                np.isfinite(a).all() for a in (objects, images)):
            raise ValueError(f"Invalid paired raw corners for pose {row['pose_id']}")
        points_camera = objects @ c[:3, :3].T + c[:3, 3]
        if np.any(points_camera[:, 2] <= 0):
            per_pose.append({'pose_id': row['pose_id'], 'available': False,
                             'reason': 'Predicted board corners lie behind the camera'})
            continue
        projected, _ = cv2.projectPoints(objects, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], k, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2) - images, axis=1)
        all_errors.extend(errors.tolist())
        per_pose.append({'pose_id': row['pose_id'], 'available': True, 'count': len(errors),
                         'rms_px': float(np.sqrt(np.mean(errors ** 2))),
                         'max_px': float(errors.max())})
    if not all_errors:
        return {'available': False, 'reason': 'No valid raw corner pairs', 'per_pose': per_pose}
    errors = np.asarray(all_errors)
    return {'available': True, 'count': len(errors), 'per_pose': per_pose,
            'rms_px': float(np.sqrt(np.mean(errors ** 2))), 'max_px': float(errors.max()),
            'uses_stored_pnp_transform_for_projection': False}


def fit_dataset(data, *, max_iterations=2000):
    rows = data.get('observations')
    if not isinstance(rows, list):
        raise ValueError('Dataset must contain observations')
    if any('pose_id' not in row for row in rows):
        raise ValueError('Every observation requires pose_id')
    if len({row['pose_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate pose IDs')
    eligible = [row for row in rows if _eligible(row)]
    if len(eligible) < 12:
        raise ValueError('Need at least 12 eligible distinct stationary observations for fitting and holdout')
    g = validate([row['T_base_from_wrist'] for row in eligible])
    c = validate([row['T_camera_from_board'] for row in eligible])
    held = np.arange(len(eligible)) % 5 == 4
    camera = _camera_metadata(data)
    result = solve(g[~held], c[~held],
                   parent_frame=data.get('parent_frame', data.get('wrist_frame', 'maver_x4_asm_mount2')),
                   base_frame=data.get('base_frame', 'maver_x4_asm_maver_x4_base_link'),
                   camera_serial=camera.get('serial', data.get('camera_serial', '243222074562')),
                   camera_frame=camera.get('camera_frame', data.get('camera_frame')),
                   validation_g=g[held], validation_c=c[held], max_iterations=max_iterations)
    x, y = np.asarray(result['T_wrist_from_camera']), np.asarray(result['T_base_from_board'])
    training = [row for row, h in zip(eligible, held) if not h]
    validation = [row for row, h in zip(eligible, held) if h]
    result.update({
        'camera': camera,
        'metadata': {key: value for key, value in data.items() if key != 'observations'},
        'training_pose_ids': [row['pose_id'] for row in training],
        'validation_pose_ids': [row['pose_id'] for row in validation],
        'excluded_poses': [{'pose_id': row['pose_id'], 'reason': row.get('reason', 'not eligible')}
                           for row in rows if not _eligible(row)],
        'eligible_observation_count': len(eligible),
        'total_observation_count': len(rows),
        'holdout_policy': 'every fifth eligible observation, never used to fit either matrix',
        'training_reprojection': reprojection_report(training, g[~held], x, y, camera),
        'validation_reprojection': reprojection_report(validation, g[held], x, y, camera),
        'hardware_commands_sent': 0,
        'active_configuration_changed': False,
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=2000)
    args = parser.parse_args(argv)
    data = json.loads(args.dataset.read_text())
    result = fit_dataset(data, max_iterations=args.max_iterations)
    result['source_dataset'] = str(args.dataset.resolve())
    result['source_dataset_sha256'] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps({key: result[key] for key in ('converged', 'termination', 'training',
                                                 'validation', 'validation_reprojection')}, indent=2))
    return 0 if result['converged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
