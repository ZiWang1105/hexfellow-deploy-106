"""Independent synthetic camera pose and outlier checks; no robot or renderer."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('landmark_pnp',
    Path(__file__).resolve().parents[2]/'tools/camera/fit_landmark_extrinsics.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def scene():
    rng = np.random.default_rng(123)
    k = np.array([[1055.,0,980.],[0,1055.,559.],[0,0,1.]])
    # Construct pixels independently from known camera-space points.
    camera_points = rng.uniform([-.18,-.12,.40],[.18,.15,.90],(36,3))
    camera_from_mount = np.eye(4)
    camera_from_mount[:3,:3] = Rotation.from_euler('xyz',[.5,-.2,.7]).as_matrix()
    camera_from_mount[:3,3] = [.14,-.09,.28]
    truth = np.linalg.inv(camera_from_mount)
    points = (camera_points-camera_from_mount[:3,3])@camera_from_mount[:3,:3]
    uvw = camera_points@k.T
    return points, uvw[:,:2]/uvw[:,2:3], k, truth


@pytest.mark.parametrize('refine_all',[False,True])
def test_recovers_known_camera_to_mount_direction(refine_all):
    points, pixels, k, truth = scene()
    fit = solver.fit(points,pixels,k,refine_all=refine_all)
    recovered = np.asarray(fit['T_mount_from_camera'])
    np.testing.assert_allclose(recovered,truth,atol=1e-7)
    assert fit['inlier_rms_px'] < 1e-5


def test_ransac_rejects_bad_matches():
    points, pixels, k, truth = scene()
    pixels[:6] += [85,-60]
    fit = solver.fit(points,pixels,k)
    assert not any(fit['inliers'][:6])
    assert all(fit['inliers'][6:])
    np.testing.assert_allclose(fit['T_mount_from_camera'],truth,atol=1e-7)


@pytest.mark.parametrize('refine_all',[False,True])
def test_leave_one_pose_out_is_independent(refine_all):
    points, pixels, k, truth = scene()
    observations = [dict(p_mount=p.tolist(),real_uv=u.tolist(),pose_id=1+i//12)
                    for i,(p,u) in enumerate(zip(points,pixels))]
    fit = solver.evaluate(observations,k,refine_all=refine_all)
    assert len(fit['leave_one_pose_out']) == 3
    assert all(r['held_out_rms_px']<1e-5 for r in fit['leave_one_pose_out'])
    np.testing.assert_allclose(fit['T_mount_from_camera'],truth,atol=1e-7)


def test_rejects_insufficient_or_collinear_points():
    points, pixels, k, _ = scene()
    with pytest.raises(ValueError,match='six'):
        solver.fit(points[:5],pixels[:5],k)
    with pytest.raises(ValueError,match='Collinear'):
        solver.fit(np.column_stack([np.arange(6),np.zeros((6,2))]),pixels[:6],k)


def test_rejects_nonfinite_input():
    points, pixels, k, _ = scene()
    points[0,0] = np.nan
    with pytest.raises(ValueError,match='finite'):
        solver.fit(points,pixels,k)


def test_all_pair_refinement_with_pixel_noise():
    points,pixels,k,truth = scene()
    pixels += np.random.default_rng(14).normal(0,.5,pixels.shape)
    fit = solver.fit(points,pixels,k,refine_all=True)
    recovered = np.asarray(fit['T_mount_from_camera'])
    assert np.linalg.norm(recovered[:3,3]-truth[:3,3]) < .002
    assert Rotation.from_matrix(recovered[:3,:3].T@truth[:3,:3]).magnitude() < np.deg2rad(.2)
    assert fit['all_rms_px'] < 1.2
