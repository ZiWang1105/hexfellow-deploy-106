"""Left planning must preserve the measured base/lift/right and use D405 optics."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from wrist_calibration_geometry import LocalWristChecker, installed_camera_transforms, load_plan
from plan_left_wrist_calibration import PLAN


def test_left_checked_capture_keeps_actual_other_arm_and_lift():
    checker=object.__new__(LocalWristChecker)
    target=np.zeros(18);target[4:11]=.4
    checker.route={'capture_side':'left','poses':[{'q':target.tolist()}]}
    checker.study=SimpleNamespace(clearance=lambda q:(.025,None),visibility=lambda q:{})
    measured=np.zeros(18);measured[:4]=[.1,-.1,.05,.30];measured[11:]=.2
    reference=measured.copy();reference[4]=.1
    bounds=[np.full(18,-3.),np.full(18,3.)]
    bounds[0][3]=.002;bounds[1][3]=.417
    result=checker.plan(measured,1,bounds,reference_origin=reference)
    q=np.asarray(result['targets'][0])
    np.testing.assert_array_equal(q[:4],measured[:4])
    np.testing.assert_array_equal(q[11:],measured[11:])
    np.testing.assert_array_equal(q[4:11],target[4:11])
    assert result['path_origins_checked']==['measured','commanded_reference']


def test_d405_uses_its_own_pupil_not_the_d435i_offset():
    _,body,optical=installed_camera_transforms('left')
    local=np.linalg.inv(body)@optical
    np.testing.assert_allclose(local[:3,3],[0,-.009,.0192],atol=1e-12)
    np.testing.assert_allclose(local[:3,:3],[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-12)


def test_left_plan_has_30_fixed_lift_camera_visible_targets():
    plan=load_plan(PLAN)
    assert plan['serial']=='230422271628' and plan['parent_frame']=='maver_x4_asm_mount1'
    assert plan['simulation_only'] and not plan['hardware_execution_authorized']
    assert plan['intrinsics_provenance']['measured']
    np.testing.assert_allclose(plan['K'],[[651.4862670898438,0,643.0418701171875],[0,650.7327880859375,362.6119689941406],[0,0,1]],atol=1e-12)
    assert plan['simulation_validation']['minimum_path_clearance_m']>=.015
    start=np.asarray(plan['initial_q'])
    nodes=[row['q'] for row in plan['poses']]+plan['approach_waypoints']+plan['return_path']['targets']
    for node in nodes:
        np.testing.assert_array_equal(np.asarray(node)[:4],start[:4])
        np.testing.assert_array_equal(np.asarray(node)[11:],start[11:])
    assert len(plan['poses'])==30
    assert all(row['nominal_detection']['accepted'] and row['nominal_detection']['corners']>=12 for row in plan['poses'])


def test_left_cannot_be_silently_relabelled_as_right(tmp_path):
    plan=json.loads(PLAN.read_text());plan['serial']='243222074562'
    path=tmp_path/'wrong.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='mount1 and D405'):load_plan(path)
