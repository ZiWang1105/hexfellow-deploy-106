"""Offline consistency of published physical camera evidence and conventions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from solve_wrist_calibration import reprojection_report


def read(path):
    return json.loads((ROOT/path).read_text())


def parameters():
    return read('config/camera/camera_parameters_20261006.json')


@pytest.mark.parametrize('camera',['head','right'])
def test_saved_camera_transform_is_proper_se3_and_matches_fit(camera):
    c=parameters()['cameras'][camera]
    path=ROOT/c['candidate']['path'];raw=json.loads(path.read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==c['candidate']['sha256']
    t=np.asarray(c['T_parent_from_optical'])
    np.testing.assert_allclose(t,np.asarray(raw[c['legacy_matrix_key']]),atol=0,rtol=0)
    np.testing.assert_allclose(t[3],[0,0,0,1],atol=0,rtol=0)
    np.testing.assert_allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-10,rtol=0)
    assert np.linalg.det(t[:3,:3])==pytest.approx(1.,abs=1e-10)
    assert c['camera_frame']!=c['parent_frame']


def test_no_left_camera_calibration_is_fabricated():
    c=parameters()['cameras']['left']
    assert c['serial']=='230422271628'
    assert c['parent_frame']=='maver_x4_asm_mount1'
    assert c['status']=='uncalibrated'
    assert c['T_parent_from_optical'] is None and c['intrinsics'] is None and c['candidate'] is None


def test_projection_from_fitted_constants_and_measured_fk_reproduces_all_heldout_errors():
    c=parameters()['cameras']['right'];fit=read(c['candidate']['path']);data=read(c['evidence']['path'])
    assert hashlib.sha256((ROOT/c['evidence']['path']).read_bytes()).hexdigest()==c['evidence']['sha256']
    held=fit['validation_pose_ids'];train=fit['training_pose_ids']
    assert held==[6,12,17,23,28] and not set(held)&set(train)
    assert len(train)==22 and len(data['observations'])==30
    rows=[r for r in data['observations'] if r['pose_id'] in held]
    assert all(r['accepted'] and r['stationary'] and r['synchronized'] for r in rows)
    g=np.asarray([r['T_base_from_wrist'] for r in rows])
    computed=reprojection_report(rows,g,np.asarray(c['T_parent_from_optical']),
                                np.asarray(c['T_base_from_board']),c['intrinsics'])
    assert computed['count']==80
    assert computed['rms_px']==pytest.approx(11.38241799965472,abs=1e-8)
    assert computed['uses_stored_pnp_transform_for_projection'] is False
    for got,want in zip(computed['per_pose'],fit['validation_reprojection']['per_pose']):
        assert got['pose_id']==want['pose_id']
        assert got['rms_px']==pytest.approx(want['rms_px'],abs=1e-8)
    # Reverse-direction camera/wrist data must not silently reproduce this score.
    wrong=reprojection_report(rows,g,np.linalg.inv(np.asarray(c['T_parent_from_optical'])),
                              np.asarray(c['T_base_from_board']),c['intrinsics'])
    assert not wrong['available'] or wrong['rms_px']>computed['rms_px']*10


def test_stream_and_board_units_remain_explicit_and_camera_specific():
    p=parameters();h=p['cameras']['head'];r=p['cameras']['right']
    assert p['base_frame']=='maver_x4_asm_maver_x4_base_link'
    assert p['translation_units']=='metres'
    assert h['parent_frame']=='maver_x4_asm_URDF_link_1'
    assert r['parent_frame']=='maver_x4_asm_mount2'
    assert [h['intrinsics']['width'],h['intrinsics']['height']]==[1920,1080]
    assert [r['intrinsics']['width'],r['intrinsics']['height']]==[1280,720]
    assert h['intrinsics']['K']!=r['intrinsics']['K']
    assert p['board']['square_length_m']==pytest.approx(.0174)
    assert h['calibration_accepted'] is False
    assert r['independent_physical_validation_performed'] is False


@pytest.mark.parametrize('side,serial,parent', [
    ('left', '230422271628', 'maver_x4_asm_mount1'),
    ('right', '243222074562', 'maver_x4_asm_mount2'),
])
def test_current_physical_wrist_export_preserves_identity_transform_and_candidate(side, serial, parent):
    snapshot = read('config/camera/camera_parameters_20261008.json')
    camera = snapshot['cameras'][side]
    registry = read('config/camera/cameras.json')
    fit = read(camera['candidate']['path'])
    assert registry['parameter_snapshot'] == 'config/camera/camera_parameters_20261008.json'
    assert registry['cameras'][side]['candidate'] == camera['candidate']['path']
    assert camera['serial'] == fit['camera_serial'] == serial
    assert camera['parent_frame'] == fit['parent_frame'] == parent
    assert fit['simulation_only'] is False
    assert camera['status'] == fit['calibration_status'] == 'candidate_requires_physical_validation'
    assert camera['independent_physical_validation_performed'] is False
    assert fit['hardware_qualified'] is False
    for field in ('candidate', 'evidence', 'capture_summary'):
        path = ROOT / camera[field]['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == camera[field]['sha256']
    transform = np.asarray(camera['T_parent_from_optical'])
    np.testing.assert_array_equal(transform, fit['T_wrist_from_camera'])
    np.testing.assert_array_equal(transform, fit['T_parent_from_optical'])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1., abs=1e-10)
    np.testing.assert_array_equal(transform[3], [0., 0., 0., 1.])


@pytest.mark.parametrize('side,expected_rms,held', [
    ('left', 7.921881469344646, [5, 10, 15, 20, 29]),
    ('right', 13.623685792294532, [9, 14, 19, 24, 29]),
])
def test_portable_current_wrist_data_reproduce_heldout_projection_without_pnp_pose(side, expected_rms, held):
    camera = read('config/camera/camera_parameters_20261008.json')['cameras'][side]
    fit = read(camera['candidate']['path'])
    data = read(camera['evidence']['path'])
    assert fit['validation_pose_ids'] == held
    assert not set(held) & set(fit['training_pose_ids'])
    assert len(fit['training_pose_ids']) == 21
    assert len(data['observations']) == 30
    assert sum(row.get('accepted') is True for row in data['observations']) == 26
    assert all('image' not in row and 'measured_geometry' not in row for row in data['observations'])
    rows = [row for row in data['observations'] if row['pose_id'] in held]
    assert all(row['stationary'] and row['synchronized'] for row in rows)
    measured_fk = np.asarray([row['T_base_from_wrist'] for row in rows])
    computed = reprojection_report(rows, measured_fk, np.asarray(camera['T_parent_from_optical']),
                                   np.asarray(camera['T_base_from_board']), camera['intrinsics'])
    assert computed['rms_px'] == pytest.approx(expected_rms, abs=1e-8)
    assert computed['uses_stored_pnp_transform_for_projection'] is False
    for got, expected in zip(computed['per_pose'], fit['validation_reprojection']['per_pose']):
        assert got['pose_id'] == expected['pose_id']
        assert got['rms_px'] == pytest.approx(expected['rms_px'], abs=1e-8)
    assert camera['quality']['heldout_reprojection_rms_px'] == pytest.approx(expected_rms, abs=1e-8)


def test_current_sdk_profile_keeps_nonzero_d405_native_distortion_and_separate_pinhole_pixels():
    cameras = read('config/camera/camera_parameters_20261008.json')['cameras']
    left, right = cameras['left']['intrinsics'], cameras['right']['intrinsics']
    for camera in (left, right):
        assert (camera['width'], camera['height'], camera['fps']) == (1280, 720, 30)
        assert camera['distortion_model'] == 'none'
        assert camera['distortion'] == [0.] * 5
    assert left['corner_processing'] == 'SDK-deprojected pinhole pixels'
    assert any(value != 0. for value in left['native_distortion'])
    assert right['native_distortion'] == [0.] * 5
    assert left['K'] != right['K']
    assert cameras['right']['session_passed'] is False
    assert cameras['left']['session_passed'] is True
