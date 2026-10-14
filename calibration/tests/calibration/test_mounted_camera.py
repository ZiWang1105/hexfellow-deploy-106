"""Camera attachment must follow lift FK without changing arm coordinates."""
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import pinocchio as pin
import pytest
from hexfellow_deploy.robotics.control.robot_model import load_robot_model
from hexfellow_deploy.robotics.scene.builder import build_tabletop_scene

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from mounted_view import frustum_corners


@pytest.mark.parametrize('angle',[15,32,40,42,55])
def test_optical_frame_follows_lift_and_matches_mujoco(tmp_path,angle):
    original=pin.buildModelFromUrdf(str(ROOT/'urdf/maver_x4_gp80.urdf'))
    suffix='' if angle==15 else f'_{angle}deg'
    mounted=pin.buildModelFromUrdf(str(ROOT/f'urdf/maver_x4_gp80_zed{suffix}.urdf'))
    assert list(original.names)==list(mounted.names)
    np.testing.assert_array_equal(original.lowerPositionLimit,mounted.lowerPositionLimit)
    np.testing.assert_array_equal(original.upperPositionLimit,mounted.upperPositionLimit)
    name='maver-x4-gp80-zed-view-v1.json' if angle==15 else f'maver-x4-gp80-zed-{angle}deg-view-v1.json'
    spec=ROOT/'config/control/robot_models'/name
    descriptor=load_robot_model({'path':str(spec),'sha256':hashlib.sha256(spec.read_bytes()).hexdigest()})
    scene=tmp_path/'robot.xml'
    build_tabletop_scene(output_path=scene,object_ids=(),include_table=False,include_box=False,
                        include_trashcan=False,base_mode='planar',preserve_fixed_inertias=True,
                        robot_model=descriptor,appearance_seed=0,layout_seed=0)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    frame='zed_left_camera_frame_optical';body=model.body(frame).id
    previous=None
    for lift in [.1,.3]:
        q=pin.neutral(mounted);q[mounted.joints[mounted.getJointId('maver_x4_asm_URDF_joint_1')].idx_q]=lift
        # Exercise nonzero arm coordinates without altering the sensor frame.
        q[2:]=.15
        pdata=mounted.createData();pin.framesForwardKinematics(mounted,pdata,q)
        base_data=original.createData();pin.framesForwardKinematics(original,base_data,q)
        for hand in ['maver_x4_asm_mount1','maver_x4_asm_mount2']:
            np.testing.assert_allclose(pdata.oMf[mounted.getFrameId(hand)].homogeneous,
                                       base_data.oMf[original.getFrameId(hand)].homogeneous,atol=1e-12)
        for name in mounted.names[1:]:
            data.qpos[model.jnt_qposadr[model.joint(name).id]]=q[mounted.joints[mounted.getJointId(name)].idx_q]
        mujoco.mj_forward(model,data)
        pose=pdata.oMf[mounted.getFrameId(frame)]
        np.testing.assert_allclose(data.xpos[body],pose.translation,atol=3e-6)
        np.testing.assert_allclose(data.xmat[body].reshape(3,3),pose.rotation,atol=3e-6)
        right=pdata.oMf[mounted.getFrameId('zed_right_camera_frame_optical')]
        np.testing.assert_allclose((pose.inverse()*right).translation,[.12,0,0],atol=1e-12)
        if previous is not None:np.testing.assert_allclose(pose.translation-previous,[0,0,.2],atol=1e-12)
        previous=pose.translation.copy()


def test_frustum_edges_project_to_exact_image_boundaries():
    k=np.array(json.loads((ROOT/'config/camera/zed_view_study.json').read_text())['K'])
    rotation=pin.rpy.rpyToMatrix(.2,-.4,.3);position=np.array([.1,-.3,1.5])
    corners=frustum_corners(position,rotation,k,1.7)
    optical=(corners-position)@rotation
    homogeneous=optical@k.T
    np.testing.assert_allclose(homogeneous[:,:2]/homogeneous[:,2:3],
                               [[0,0],[1920,0],[1920,1080],[0,1080]],atol=1e-9)


def test_short_v5_geometry_and_camera_datum():
    import xml.etree.ElementTree as ET
    urdf=ET.parse(ROOT/'urdf/maver_x4_gp80_zed.urdf')
    mount=urdf.find("joint[@name='zed_camera_mount']/origin")
    pitch=np.deg2rad(15)
    np.testing.assert_allclose(np.fromstring(mount.get('xyz'),sep=' '),
                               [.003*np.sin(pitch),0,.065+.003*np.cos(pitch)],atol=1e-12)
    np.testing.assert_allclose(np.fromstring(mount.get('rpy'),sep=' '),[0,pitch,0],atol=1e-12)
    for geometry in urdf.findall("link[@name='zed_mount']//mesh"):
        assert Path(geometry.get('filename')).name=='zed_short_triangular_v5_15deg_mm.stl'
        np.testing.assert_allclose(np.fromstring(geometry.get('scale'),sep=' '),[.001]*3)
        # Binary STL coordinates are mm, unlike the older metre-based asset.
        dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
        vertices=np.fromfile(ROOT/'urdf'/geometry.get('filename'),dtype=dtype,offset=84)['v'].reshape(-1,3)
        np.testing.assert_allclose(np.ptp(vertices,axis=0),[65,60,76.3093948364],atol=1e-5)


@pytest.mark.parametrize('angle',[32,40,42,55])
def test_angle_variants_keep_short_shelf_datum_and_base_attachment(angle):
    import xml.etree.ElementTree as ET
    root=ET.parse(ROOT/f'urdf/maver_x4_gp80_zed_{angle}deg.urdf')
    mount=root.find("joint[@name='zed_camera_mount']/origin")
    a=np.deg2rad(angle)
    np.testing.assert_allclose(np.fromstring(mount.get('xyz'),sep=' '),[.003*np.sin(a),0,.065+.003*np.cos(a)],atol=1e-12)
    np.testing.assert_allclose(np.fromstring(mount.get('rpy'),sep=' '),[0,a,0],atol=1e-12)
    parent=root.find("joint[@name='zed_bracket_mount']/origin")
    np.testing.assert_allclose(np.fromstring(parent.get('xyz'),sep=' '),[.04,0,.08],atol=1e-12)
