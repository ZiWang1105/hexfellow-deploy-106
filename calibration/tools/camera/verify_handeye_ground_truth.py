#!/usr/bin/env python3
"""Offline, reproducible hand-eye recovery check against MuJoCo world transforms.

Ground truth is used for exact-data generation and final scoring only. Neither
the rendered-data solver nor its initializer receives camera/board ground truth.
"""
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
sys.path.insert(0, str(ROOT / 'src'))
from policy_sim.calibration.charuco import detect
from policy_sim.calibration.handeye import inverse, solve, summarize, transform
from policy_sim.control.robot import CONTROLLED_JOINTS


def error(estimate, truth):
    delta = inverse(truth) @ estimate
    return {
        'translation_mm': float(np.linalg.norm(delta[:3, 3]) * 1000),
        'rotation_deg': float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(delta[:3, :3]).as_rotvec()))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', type=Path,
                        default=ROOT / 'runs/camera-calibration-40deg/route')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/camera-calibration-40deg/ground-truth-verification')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = np.load(args.route / 'physics_trace.npz')
    physics = json.loads((args.route / 'physics_report.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(args.route / 'scene.xml'))
    data = mujoco.MjData(model)
    pm = pin.buildModelFromUrdf(str(ROOT / 'urdf/maver_x4_gp80_zed_40deg.urdf'))
    pd = pm.createData()
    joint_indices = [pm.joints[pm.getJointId(n)].idx_q for n in CONTROLLED_JOINTS]
    mount_name, hand_name = 'maver_x4_asm_URDF_link_1', 'maver_x4_asm_mount2'
    k = np.array(json.loads((ROOT / 'config/camera/zed_view_study.json').read_text())['K'])
    board = json.loads((ROOT / 'config/camera/charuco_87mm.json').read_text())
    opt = mujoco.MjvOption()
    opt.geomgroup[3:5] = 1
    mx, my = np.meshgrid(np.arange(1920, dtype=np.float32) + 1000 - k[0, 2],
                         np.arange(1080, dtype=np.float32) + 580 - k[1, 2])
    camera_id = model.camera('calibration_view').id

    def body_world(name):
        bid = model.body(name).id
        return transform(data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy())

    gs, exact_gs, cs, exact_cs, truths_x, truths_y, detections = [], [], [], [], [], [], []
    with mujoco.Renderer(model, height=1160, width=2000) as renderer:
        for stop in physics['dwells']:
            idx = int(np.argmin(abs(trace['time'] - stop['time_s'])))
            data.qpos[:] = trace['qpos'][idx]
            mujoco.mj_forward(model, data)
            wm, wh, wb = (body_world(n) for n in (mount_name, hand_name, 'calibration_board'))
            # Use actual renderer camera pose, converting OpenGL to OpenCV axes.
            wc = transform(data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1]),
                           data.cam_xpos[camera_id])
            truths_x.append(inverse(wm) @ wc)
            truths_y.append(inverse(wh) @ wb)
            exact_gs.append(inverse(wm) @ wh)
            exact_cs.append(inverse(wc) @ wb)
            q = pin.neutral(pm)
            q[joint_indices] = trace['q'][idx]
            pin.framesForwardKinematics(pm, pd, q)
            gs.append((pd.oMf[pm.getFrameId(mount_name)].inverse() *
                       pd.oMf[pm.getFrameId(hand_name)]).homogeneous.copy())
            renderer.update_scene(data, camera='calibration_view', scene_option=opt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            rgb = cv2.remap(renderer.render(), mx, my, cv2.INTER_LINEAR)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            result, annotated = detect(bgr, board, k, np.zeros(5))
            result['pose_id'] = stop['pose_id']
            if not result['accepted']:
                raise RuntimeError(f"Detection failed at pose {stop['pose_id']}")
            cv2.imwrite(str(args.output / f"pose_{stop['pose_id']:02}_raw.png"), bgr)
            cs.append(result['T_camera_from_board'])
            detections.append(result)
    gs, exact_gs, cs, exact_cs = map(np.array, (gs, exact_gs, cs, exact_cs))
    x_truth, y_truth = truths_x[0], truths_y[0]
    np.testing.assert_allclose(truths_x, np.broadcast_to(x_truth, (len(gs), 4, 4)), atol=1e-10)
    np.testing.assert_allclose(truths_y, np.broadcast_to(y_truth, (len(gs), 4, 4)), atol=1e-10)
    fit = np.arange(len(gs)) % 5 != 4
    report = {
        'simulation_only': True, 'physical_calibration': False,
        'ground_truth_source': 'MuJoCo body world poses and renderer camera world pose',
        'solver_receives_ground_truth': False,
        'training_pose_ids': [d['pose_id'] for d, f in zip(detections, fit) if f],
        'held_out_pose_ids': [d['pose_id'] for d, f in zip(detections, fit) if not f],
        'truth': {'T_mount_from_camera': x_truth.tolist(), 'T_hand_from_board': y_truth.tolist()},
        'fk_crosscheck_max': {key: max(error(g, e)[key] for g, e in zip(gs, exact_gs))
                             for key in ('translation_mm', 'rotation_deg')},
        'thresholds': {'exact': {'translation_mm': 1e-5, 'rotation_deg': 1e-5},
                       'rendered': {'translation_mm': 1.0, 'rotation_deg': 0.2}},
        'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            args.route / 'scene.xml', args.route / 'physics_trace.npz',
            ROOT / 'src/policy_sim/calibration/handeye.py', Path(__file__))},
    }
    for label, g, c in [('exact', exact_gs, exact_cs), ('rendered', gs, cs)]:
        result = solve(g[fit], c[fit])
        x, y = (np.array(result[key]) for key in ('T_mount_from_camera', 'T_hand_from_board'))
        result['camera_error'] = error(x, x_truth)
        result['board_error'] = error(y, y_truth)
        result['held_out'] = summarize(g[~fit], c[~fit], x, y)
        result['passed'] = bool(result['converged'] and all(
            result[name][key] < threshold
            for name in ('camera_error', 'board_error')
            for key, threshold in report['thresholds'][label].items()))
        report[label] = result
        print(label, json.dumps({key: result[key] for key in (
            'converged', 'iterations', 'camera_error', 'board_error', 'passed')}), flush=True)
    report['passed'] = all(report[label]['passed'] for label in ('exact', 'rendered'))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output / 'detections.json').write_text(json.dumps(detections, indent=2, allow_nan=False))
    np.savez_compressed(args.output / 'observations.npz', g=gs, c=cs, exact_g=exact_gs,
                        exact_c=exact_cs, fit=fit, truth_x=x_truth, truth_y=y_truth)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
