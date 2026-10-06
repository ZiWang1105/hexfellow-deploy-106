"""Independent wrist transform recovery, convention, and held-out image checks."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from policy_sim.calibration.handeye import inverse, retract, transform
from policy_sim.calibration.wrist_handeye import camera_from_board, solve, summarize

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('solve_wrist_calibration',
                                             ROOT / 'tools/camera/solve_wrist_calibration.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def fixture(noise=0):
    rng = np.random.default_rng(9321)
    x = transform(Rotation.from_euler('xyz', [.1, np.deg2rad(33), .2]).as_matrix(), [.045, -.02, .032])
    y = transform(Rotation.from_euler('xyz', [.05, .02, -.1]).as_matrix(), [.015, .045, .72])
    g = np.array([transform(Rotation.from_rotvec(rng.normal(0, .22, 3)).as_matrix(),
                            rng.uniform(-.13, .13, 3)) for _ in range(30)])
    # Independent generation from the physical eye-in-hand equation.
    c = np.array([inverse(x) @ inverse(t) @ y for t in g])
    for i in range(len(c)):
        c[i] = retract(c[i], rng.normal(0, noise, 6))
    return g, c, x, y


def transform_error(estimate, truth):
    estimate = np.asarray(estimate)
    return (np.linalg.norm(estimate[:3, 3] - truth[:3, 3]),
            Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).magnitude())


def test_exact_recovery_and_runtime_direction():
    g, c, x, y = fixture()
    result = solve(g, c)
    assert result['converged']
    np.testing.assert_allclose(result['T_wrist_from_camera'], x, atol=1e-7)
    np.testing.assert_allclose(result['T_parent_from_optical'], x, atol=1e-7)
    np.testing.assert_allclose(result['T_base_from_board'], y, atol=1e-7)
    assert 'T_mount_from_camera' not in result and 'T_hand_from_board' not in result
    assert result['head_extrinsics_used_for_fit'] is False
    assert result['hardware_qualified'] is False
    assert result['camera_frame'] == 'realsense_243222074562_color_optical'
    for wrist, board_in_camera in zip(g, c):
        base_from_camera = wrist @ result['T_parent_from_optical']
        np.testing.assert_allclose(base_from_camera @ board_in_camera, y, atol=1e-7)
    # The saved X is camera-to-wrist, not its inverse and not a fixed base pose.
    assert np.linalg.norm(inverse(x) - result['T_wrist_from_camera']) > .5
    assert not np.allclose(g[0] @ x, g[1] @ x)


def test_noisy_fit_and_validation_do_not_reuse_holdout():
    g, c, x, y = fixture(.0004)
    held = np.arange(len(g)) % 5 == 4
    result = solve(g[~held], c[~held], validation_g=g[held], validation_c=c[held])
    assert result['final_loss'] <= result['initial_loss']
    assert result['converged']
    assert transform_error(result['T_wrist_from_camera'], x)[0] < .002
    assert transform_error(result['T_wrist_from_camera'], x)[1] < np.deg2rad(.2)
    assert transform_error(result['T_base_from_board'], y)[0] < .002
    assert result['validation']['position_rms_mm'] < 2
    assert result['validation']['angle_rms_deg'] < .2
    perturbed = c[held].copy()
    perturbed[:, :3, 3] += [.08, 0, 0]
    other = solve(g[~held], c[~held], validation_g=g[held], validation_c=perturbed)
    np.testing.assert_array_equal(other['T_wrist_from_camera'], result['T_wrist_from_camera'])
    assert other['validation']['position_rms_mm'] > 79


def test_summary_is_physical_base_closure_and_predicted_pnp():
    g, c, x, y = fixture()
    np.testing.assert_allclose(camera_from_board(g, x, y), c, atol=1e-12)
    report = summarize(g, c, x, y)
    assert report['position_rms_mm'] < 1e-9
    assert report['angle_rms_deg'] < 1e-9
    wrong = summarize(g, c, inverse(x), y)
    assert wrong['angle_rms_deg'] > 30


def test_rejects_degenerate_invalid_and_unpaired_data():
    g, c, _, _ = fixture()
    with pytest.raises(ValueError, match='paired'):
        solve(g[:-1], c)
    with pytest.raises(ValueError, match='six'):
        solve(g[:5], c[:5])
    g[:, :3, :3] = np.eye(3)
    with pytest.raises(ValueError, match='independent axes'):
        solve(g, c)
    g, c, _, _ = fixture()
    c[0, 0, 0] = 10
    with pytest.raises(ValueError, match='rigid transform'):
        solve(g, c)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_recovers_ground_truth_with_actual_robot_forward_kinematics(side):
    import pinocchio as pin
    from policy_sim.control.robot import BASE_FRAME, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
    joints = LEFT_ARM_JOINTS if side == 'left' else RIGHT_ARM_JOINTS
    parent = 'maver_x4_asm_mount1' if side == 'left' else 'maver_x4_asm_mount2'
    serial = '230422271628' if side == 'left' else '243222074562'

    model = pin.buildModelFromUrdf(str(ROOT / 'urdf/maver_x4_gp80_zed_40deg.urdf'))
    data = model.createData()
    rng = np.random.default_rng(3108)
    g = []
    for _ in range(30):
        q = pin.neutral(model)
        q[model.joints[model.getJointId('maver_x4_asm_URDF_joint_1')].idx_q] = .3
        for name in joints:
            index = model.joints[model.getJointId(name)].idx_q
            lo, hi = max(model.lowerPositionLimit[index], -.6), min(model.upperPositionLimit[index], .6)
            q[index] = rng.uniform(lo, hi)
        pin.framesForwardKinematics(model, data, q)
        g.append((data.oMf[model.getFrameId(BASE_FRAME)].inverse()
                  * data.oMf[model.getFrameId(parent)]).homogeneous.copy())
    g = np.asarray(g)
    _, _, x, y = fixture()
    c = np.asarray([inverse(x) @ inverse(wrist) @ y for wrist in g])
    result = solve(g, c, parent_frame=parent, camera_serial=serial)
    np.testing.assert_allclose(result['T_wrist_from_camera'], x, atol=1e-7)
    np.testing.assert_allclose(result['T_base_from_board'], y, atol=1e-7)
    assert result['parent_frame'] == parent
    assert result['camera_frame'] == f'realsense_{serial}_color_optical'


def image_dataset():
    g, c, _, _ = fixture()
    k = np.array([[606.6, 0, 322.66], [0, 606.52, 248.74], [0, 0, 1.]])
    xx, yy = np.meshgrid(np.arange(1, 5) * .0174, np.arange(1, 5) * .0174)
    objects = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(16)])
    observations = []
    for i, (wrist, camera) in enumerate(zip(g, c)):
        pixels = cv2.projectPoints(objects, cv2.Rodrigues(camera[:3, :3])[0], camera[:3, 3], k, np.zeros(5))[0]
        # Recover each C by real PnP rather than feeding the ground-truth C.
        ok, rv, tv = cv2.solvePnP(objects, pixels, k, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        assert ok
        estimated = transform(cv2.Rodrigues(rv)[0], tv)
        observations.append({'pose_id': i + 1, 'accepted': True, 'stationary': True,
                             'synchronized': True, 'T_base_from_wrist': wrist.tolist(),
                             'T_camera_from_board': estimated.tolist(),
                             'object_points': objects.tolist(), 'image_points': pixels.reshape(-1, 2).tolist()})
    return {'camera': {'K': k.tolist(), 'distortion': [0] * 5, 'serial': '243222074562'},
            'parent_frame': 'maver_x4_asm_mount2', 'observations': observations}


def test_dataset_skips_invalid_and_predicts_heldout_corners():
    data = image_dataset()
    data['observations'].append({'pose_id': 31, 'eligible': False, 'reason': 'board not visible'})
    fit = CLI.fit_dataset(data)
    assert fit['eligible_observation_count'] == 30
    assert fit['total_observation_count'] == 31
    assert fit['training_pose_ids'] == [i for i in range(1, 31) if i % 5]
    assert fit['validation_pose_ids'] == [5, 10, 15, 20, 25, 30]
    assert fit['excluded_poses'] == [{'pose_id': 31, 'reason': 'board not visible'}]
    assert fit['validation_reprojection']['count'] == 96
    assert fit['validation_reprojection']['rms_px'] < 1e-3
    assert fit['training_reprojection']['rms_px'] < 1e-3
    assert fit['validation_reprojection']['uses_stored_pnp_transform_for_projection'] is False


def test_dataset_requires_minimum_and_excludes_unstable_or_duplicate_ids():
    data = image_dataset()
    for row in data['observations'][10:]:
        row['stationary'] = False
    with pytest.raises(ValueError, match='12 eligible'):
        CLI.fit_dataset(data)
    data = image_dataset()
    data['observations'][0]['pose_id'] = data['observations'][1]['pose_id']
    with pytest.raises(ValueError, match='Duplicate'):
        CLI.fit_dataset(data)


def test_inverse_brown_is_not_silently_treated_as_opencv_distortion():
    data = image_dataset()
    data['camera'].update(distortion_model='distortion.inverse_brown_conrady',
                          distortion=[-.05, .06, 0, 0, -.02])
    fit = CLI.fit_dataset(data)
    assert fit['validation_reprojection']['available'] is False
    assert 'SDK rectification' in fit['validation_reprojection']['reason']
