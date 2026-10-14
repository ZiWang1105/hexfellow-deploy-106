"""Protect measured-state paths and wrist FK conventions without robot access."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from wrist_calibration_geometry import LocalWristChecker, load_plan, DEFAULT_PLAN, geometry_payload_hash
from hexfellow_deploy.robotics.control.robot import CONTROLLED_JOINTS, BASE_FRAME


def checker():
    c=object.__new__(LocalWristChecker)
    q=np.zeros(18);q[3]=.3
    pose=q.copy();pose[-7:]=.4
    c.route={'poses':[{'q':pose.tolist()}]}
    c.study=SimpleNamespace(clearance=lambda q:(.025,None),visibility=lambda q:{'simulation_only':True})
    bounds=[np.full(18,-3.).tolist(),np.full(18,3.).tolist()]
    bounds[0][3]=.002;bounds[1][3]=.417
    return c,q,bounds


def test_capture_preserves_actual_base_lift_and_left_and_reference():
    c,q,bounds=checker();q[:3]=[.1,-.2,.01];q[3]=.301;q[4:11]=.015
    reference=q.copy();reference[-7]+=.02
    payload=c.plan(q,1,bounds,reference_origin=reference)
    np.testing.assert_array_equal(np.asarray(payload['targets'][-1])[:11],q[:11])
    np.testing.assert_array_equal(payload['reference_origin'],reference)
    assert payload['path_origins_checked']==['measured','commanded_reference']
    assert payload['reference_path_minimum_clearance_m']>=.015


def test_upper_lift_retreat_is_explicit_before_full_default():
    c,q,bounds=checker();q[3]=.422
    default=q.copy();default[3]=.300
    payload=c.plan(q,-2,bounds,default_target=default)
    assert [v[3] for v in payload['targets']]==[.409,.300]
    # An ordinary capture cannot silently reuse the mechanical-endpoint exception.
    with pytest.raises(ValueError,match='outside controller bounds'):c.plan(q,1,bounds)


def test_unsafe_actual_pose_is_not_an_invalid_target_to_skip():
    c,q,bounds=checker();c.study.clearance=lambda q:(.009,['forearm','table'])
    with pytest.raises(ValueError,match='below 15 mm') as error:c.plan(q,1,bounds)
    from calibration_outcomes import InvalidCalibrationPose
    assert not isinstance(error.value,InvalidCalibrationPose)


def test_default_return_takes_a_checked_detour_around_obstacle():
    c,q,bounds=checker();default=q.copy();q[11:13]=[1,1]
    bridge=default.copy();bridge[11:13]=[0,1]
    c.route={'poses':[{'q':bridge.tolist()},{'q':q.tolist()}]}
    c.study.clearance=lambda v:(.005 if np.linalg.norm(np.asarray(v)[11:13]-.5)<.2 else .025,None)
    with pytest.raises(ValueError):c.edge(q,default,bounds)
    payload=c.plan(q,-1,bounds,default_target=default,reference_origin=q)
    assert len(payload['targets'])==2
    np.testing.assert_array_equal(payload['targets'][0],bridge)
    np.testing.assert_array_equal(payload['targets'][1],default)
    assert payload['minimum_sampled_clearance_m']>=.015


def test_edited_simulated_targets_or_optics_require_revalidation(tmp_path):
    plan=load_plan(DEFAULT_PLAN)
    plan['poses'][0]['q'][11]+=.1
    path=tmp_path/'changed.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='simulated targets changed'):load_plan(path)
    plan=load_plan(DEFAULT_PLAN);plan['K'][0][0]*=.95;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='simulated targets changed'):load_plan(path)


def test_all_wrist_planning_frames_match_gp80_controller_urdf():
    plan=load_plan(DEFAULT_PLAN)
    models=[pin.buildModelFromUrdf(str(ROOT/'urdf'/name)) for name in ('maver_x4_gp80_zed_40deg.urdf','maver_x4_gp80.urdf')]
    data=[m.createData() for m in models]
    indices=[[m.joints[m.getJointId(n)].idx_q for n in CONTROLLED_JOINTS] for m in models]
    for row in plan['poses']:
        transforms=[]
        for m,d,idx in zip(models,data,indices):
            q=pin.neutral(m);q[idx]=np.asarray(row['q'])[3:]
            pin.framesForwardKinematics(m,d,q)
            transforms.append(np.linalg.inv(d.oMf[m.getFrameId(BASE_FRAME)].homogeneous)@d.oMf[m.getFrameId(plan['parent_frame'])].homogeneous)
        np.testing.assert_allclose(transforms[0],transforms[1],atol=1e-12,rtol=0)
        np.testing.assert_allclose(transforms[0],row['nominal_view']['T_base_from_parent'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(np.asarray(row['q'])[:11],np.asarray(plan['initial_q'])[:11])


def test_complete_mount_rotates_to_outside_at_default_and_preserves_fixed_pitch():
    from wrist_calibration_geometry import DESIGN, installed_camera_transforms
    installation,housing,optical=installed_camera_transforms()
    transform=np.asarray(installation['T_parent_from_design'])
    np.testing.assert_array_equal(transform,np.diag([-1,-1,1,1]))
    assert np.linalg.det(transform[:3,:3])==1
    original=np.asarray(json.loads((DESIGN/'design_and_validation.json').read_text())['camera_transform_mm'])
    original[:3,3]*=.001
    np.testing.assert_allclose(housing,transform@original,atol=1e-12,rtol=0)
    local=np.eye(4);local[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];local[:3,3]=[0,-.0325,.0207547]
    from wrist_calibration_geometry import optical_roll
    np.testing.assert_allclose(optical,optical_roll(housing@local, installation.get('optical_axis_roll_deg',0.)),atol=1e-12,rtol=0)
    plan=load_plan(DEFAULT_PLAN)
    model=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80_zed_40deg.urdf'));data=model.createData()
    q=pin.neutral(model)
    q[[model.joints[model.getJointId(n)].idx_q for n in CONTROLLED_JOINTS]]=np.asarray(plan['initial_q'])[3:]
    pin.framesForwardKinematics(model,data,q)
    wrist=np.linalg.inv(data.oMf[model.getFrameId(BASE_FRAME)].homogeneous)@data.oMf[model.getFrameId(plan['parent_frame'])].homogeneous
    assert (wrist@original@local)[1,3]>wrist[1,3]  # Historical installation was inward.
    assert (wrist@optical)[1,3]<wrist[1,3]         # Current installation is outward.
    np.testing.assert_allclose(plan['T_parent_from_optical_nominal'],optical,atol=1e-12,rtol=0)


def test_mount_edit_cannot_silently_reuse_validated_poses(tmp_path):
    plan=load_plan(DEFAULT_PLAN)
    plan['mount_installation']['T_parent_from_design'][0][0]=1
    path=tmp_path/'wrong-side.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='installation or optical transform changed'):load_plan(path)


def test_optical_roll_preserves_pupil_and_forward_axis_but_rotates_projected_pixels():
    from wrist_calibration_geometry import optical_roll
    pose=np.eye(4);pose[:3,:3]=pin.rpy.rpyToMatrix(.2,-.4,.3);pose[:3,3]=[.1,-.2,.8]
    rolled=optical_roll(pose,180.)
    np.testing.assert_array_equal(rolled[:3,3],pose[:3,3])
    np.testing.assert_array_equal(rolled[:3,2],pose[:3,2])
    np.testing.assert_allclose(rolled[:3,:2],-pose[:3,:2],atol=1e-14)
    point=pose@np.array([.04,-.03,.4,1.])
    k=np.array([[900.,0.,644.],[0.,910.,373.],[0.,0.,1.]])
    def project(t):
        p=k@(np.linalg.inv(t)@point)[:3]
        return p[:2]/p[2]
    np.testing.assert_allclose(project(rolled),2*k[:2,2]-project(pose),atol=1e-12)
    np.testing.assert_allclose(optical_roll(pose,0.),pose,atol=0.)


def test_right_planning_ray_avoids_gripper_after_sensor_roll():
    from plan_wrist_calibration import image_target_center
    plan=dict(capture_side='right',width=1280,height=720,mount_installation={})
    np.testing.assert_allclose(image_target_center(plan),[640,170])
    plan['mount_installation']['optical_axis_roll_deg']=180.
    np.testing.assert_allclose(image_target_center(plan),[640,518.4])
    plan['capture_side']='left'
    np.testing.assert_allclose(image_target_center(plan),[640,180])
