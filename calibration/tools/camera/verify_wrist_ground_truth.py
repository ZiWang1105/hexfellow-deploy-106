#!/usr/bin/env python3
"""Offline end-to-end wrist calibration comparison against simulator truth.

Recorded image pixels and measured joints enter the calibration solver.
MuJoCo camera/body transforms and nominal CAD transforms are used for scoring
only, never as a seed or observation in the fitted dataset.  No robot SDK,
camera device, network, hardware endpoint, or motion command is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from policy_sim.calibration.handeye import inverse, transform
from policy_sim.calibration.wrist_handeye import solve
from policy_sim.control.robot import BASE_FRAME, BASE_JOINTS, CONTROLLED_JOINTS
from solve_wrist_calibration import fit_dataset


def error(estimate, truth):
    delta = inverse(np.asarray(truth)) @ np.asarray(estimate)
    return {'translation_mm': float(np.linalg.norm(delta[:3, 3])*1000),
            'rotation_deg': float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))}


def body_world(model, data, name):
    index = model.body(name).id
    return transform(data.xmat[index].reshape(3, 3).copy(), data.xpos[index].copy())


def evaluate(directory, *, minimum_poses=30):
    directory = Path(directory)
    dataset_file, plan_file = directory/'dataset.json', directory/'wrist-plan.json'
    dataset_bytes = dataset_file.read_bytes()
    dataset, plan = json.loads(dataset_bytes), json.loads(plan_file.read_text())
    if dataset.get('simulation_only') is not True:
        raise ValueError('Ground-truth comparison only accepts explicit simulation datasets')
    rows = [row for row in dataset['observations'] if row.get('accepted') is True]
    if len(rows) < minimum_poses:
        raise ValueError(f'Need {minimum_poses} eligible simulation observations; currently {len(rows)}')
    if any(row.get('stationary') is not True or row.get('synchronized') is not True for row in rows):
        raise ValueError('Every scored observation must be stationary and paired with measured feedback')
    scene_file = directory/'geometry/scene.xml'
    urdf_file = ROOT/'urdf/maver_x4_gp80_zed_40deg.urdf'
    model, pm = mujoco.MjModel.from_xml_path(str(scene_file)), pin.buildModelFromUrdf(str(urdf_file))
    data, pd = mujoco.MjData(model), pm.createData()
    mi = [model.joint(name).qposadr[0] for name in CONTROLLED_JOINTS]
    pi = [pm.joints[pm.getJointId(name)].idx_q for name in CONTROLLED_JOINTS]
    camera_id = model.camera('wrist_calibration_view').id
    k = np.asarray(dataset['camera']['K'])
    distortion = np.asarray(dataset['camera']['distortion'])
    base_frame, parent_frame = dataset['base_frame'], dataset['parent_frame']
    if base_frame != BASE_FRAME or parent_frame != plan['parent_frame']:
        raise ValueError('Dataset frame identities do not match the simulated URDF')
    gs, cs, truth_gs, truth_cs, truth_xs, truth_ys, per_pose = [], [], [], [], [], [], []
    for row in rows:
        q = np.asarray(row['q'])
        data.qpos[mi] = q[3:]
        for name, value in zip(BASE_JOINTS, q[:3]):
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        world_base = body_world(model, data, base_frame)
        world_wrist = body_world(model, data, parent_frame)
        world_board = body_world(model, data, 'stationary_calibration_board')
        # Camera's local OpenGL axes are X right, Y up, Z backwards.
        # Convert independently from actual MuJoCo camera state into optical.
        world_camera = transform(data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.]),
                                 data.cam_xpos[camera_id])
        truth_g = inverse(world_base) @ world_wrist
        truth_c = inverse(world_camera) @ world_board
        truth_x = inverse(world_wrist) @ world_camera
        truth_y = inverse(world_base) @ world_board
        pin_q = pin.neutral(pm)
        pin_q[pi] = q[3:]
        pin.framesForwardKinematics(pm, pd, pin_q)
        fk_g = (pd.oMf[pm.getFrameId(base_frame)].inverse()
                * pd.oMf[pm.getFrameId(parent_frame)]).homogeneous.copy()
        recorded_g = np.asarray(row['T_base_from_wrist'])
        recorded_c = np.asarray(row['T_camera_from_board'])
        objects, images = np.asarray(row['object_points']), np.asarray(row['image_points'])
        projection = cv2.projectPoints(objects, cv2.Rodrigues(truth_c[:3, :3])[0],
                                       truth_c[:3, 3], k, distortion)[0].reshape(-1, 2)
        pixel_errors = np.linalg.norm(projection-images, axis=1)
        per_pose.append({'pose_id': row['pose_id'], 'corners': len(objects),
                         'recorded_vs_recomputed_fk': error(recorded_g, fk_g),
                         'pinocchio_vs_mujoco_fk': error(fk_g, truth_g),
                         'pnp_vs_geometry': error(recorded_c, truth_c),
                         'geometry_corner_rms_px': float(np.sqrt(np.mean(pixel_errors**2))),
                         'geometry_corner_max_px': float(pixel_errors.max()),
                         'stored_pnp_reprojection_rms_px': row['reprojection_rms_px'],
                         'image_joint_skew_s': row['pairing']['image_joint_skew_s'],
                         'maximum_capture_target_offset_deg': row['capture_offset']['maximum_active_offset_deg']})
        gs.append(recorded_g); cs.append(recorded_c)
        truth_gs.append(truth_g); truth_cs.append(truth_c)
        truth_xs.append(truth_x); truth_ys.append(truth_y)
    gs, cs, truth_gs, truth_cs = map(np.asarray, (gs, cs, truth_gs, truth_cs))
    x_truth, y_truth = truth_xs[0], truth_ys[0]
    x_spread = max(np.max(np.abs(t-x_truth)) for t in truth_xs)
    y_spread = max(np.max(np.abs(t-y_truth)) for t in truth_ys)
    constancy = {
        'camera_mount': {key: max(error(t, x_truth)[key] for t in truth_xs)
                         for key in ('translation_mm', 'rotation_deg')},
        'board_base': {key: max(error(t, y_truth)[key] for t in truth_ys)
                       for key in ('translation_mm', 'rotation_deg')},
    }
    held = np.arange(len(rows)) % 5 == 4
    rendered = fit_dataset(dataset)
    exact = solve(truth_gs[~held], truth_cs[~held], parent_frame=parent_frame,
                  camera_serial=dataset['camera']['serial'], base_frame=base_frame,
                  validation_g=truth_gs[held], validation_c=truth_cs[held])
    report = {
        'simulation_only': True,
        'physical_calibration_validated': False,
        'hardware_commands_sent': 0,
        'eligible_poses': len(rows),
        'scoring_truth': 'MuJoCo actual camera/world body poses at recorded measured joints, converted to RGB optical axes',
        'solver_receives_ground_truth': False,
        'fit_observations': 'image detected corners -> PnP, synchronized measured joint q -> Pinocchio wrist FK',
        'stationary_pairing_limitation': 'Camera renders request q, while FK uses the stationary time-paired measured q; finite permitted drift is included in the residual',
        'base_frame': base_frame,
        'parent_frame': parent_frame,
        'camera_frame': dataset['camera_frame'],
        'truth': {'T_wrist_from_camera': x_truth.tolist(), 'T_base_from_board': y_truth.tolist()},
        'truth_constancy_max_matrix_difference': {'camera_mount': float(x_spread), 'board_base': float(y_spread)},
        'truth_constancy': constancy,
        'nominal_vs_actual_camera': error(plan['T_parent_from_optical_nominal'], x_truth),
        'nominal_vs_actual_board': error(plan['T_base_from_board_nominal'], y_truth),
        'per_pose': per_pose,
        'thresholds': {'exact': {'translation_mm': 1e-5, 'rotation_deg': 1e-5},
                       'rendered': {'translation_mm': 1., 'rotation_deg': .2},
                       'recorded_fk_max_translation_mm': 1e-5,
                       'recorded_fk_max_rotation_deg': 1e-5,
                       'pinocchio_mujoco_fk_max_translation_mm': .001,
                       'pinocchio_mujoco_fk_max_rotation_deg': .0001,
                       'truth_constancy_translation_mm': .001,
                       'truth_constancy_rotation_deg': .0001,
                       'held_out_corner_rms_px': 1.},
        'inputs_sha256': {str(path): hashlib.sha256(dataset_bytes if path == dataset_file else path.read_bytes()).hexdigest()
                          for path in (dataset_file, plan_file, scene_file, urdf_file,
                                       ROOT/'src/policy_sim/calibration/wrist_handeye.py',
                                       ROOT/'src/policy_sim/calibration/handeye.py', Path(__file__))},
    }
    for label, fit in (('exact', exact), ('rendered', rendered)):
        fit['camera_ground_truth_error'] = error(fit['T_wrist_from_camera'], x_truth)
        fit['board_ground_truth_error'] = error(fit['T_base_from_board'], y_truth)
        fit['passed_ground_truth_recovery'] = bool(fit['converged'] and all(
            fit[name][key] < threshold
            for name in ('camera_ground_truth_error', 'board_ground_truth_error')
            for key, threshold in report['thresholds'][label].items()))
        report[label] = fit
    recorded_fk_ok = all(row['recorded_vs_recomputed_fk']['translation_mm'] < 1e-5
                         and row['recorded_vs_recomputed_fk']['rotation_deg'] < 1e-5 for row in per_pose)
    independent_fk_ok = all(row['pinocchio_vs_mujoco_fk']['translation_mm'] < .001
                            and row['pinocchio_vs_mujoco_fk']['rotation_deg'] < .0001 for row in per_pose)
    reprojection = rendered['validation_reprojection']
    constant_truth_ok = all(row['translation_mm'] < .001 and row['rotation_deg'] < .0001
                            for row in constancy.values())
    report['passed'] = bool(exact['passed_ground_truth_recovery']
                            and rendered['passed_ground_truth_recovery']
                            and recorded_fk_ok and independent_fk_ok and constant_truth_ok
                            and reprojection['available'] and reprojection['rms_px'] < 1.)
    report['recorded_measured_fk_passed'] = recorded_fk_ok
    report['independent_fk_crosscheck_passed'] = independent_fk_ok
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-poses', type=int, default=30)
    args = parser.parse_args(argv)
    report = evaluate(args.directory, minimum_poses=args.minimum_poses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({
        'eligible_poses': report['eligible_poses'],
        'passed': report['passed'],
        'recorded_measured_fk_passed': report['recorded_measured_fk_passed'],
        'camera_error': report['rendered']['camera_ground_truth_error'],
        'board_error': report['rendered']['board_ground_truth_error'],
        'held_out_reprojection_rms_px': report['rendered']['validation_reprojection']['rms_px'],
    }, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
