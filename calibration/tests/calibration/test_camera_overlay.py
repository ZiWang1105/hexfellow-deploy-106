"""Independent pinhole/raster checks for fitted-camera joint replay."""
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
from validate_camera_overlay import camera_remap, inject_camera, mask_metrics, CAMERA


def test_rendered_fiducials_match_pinhole_and_follow_lift(tmp_path):
    k=np.array([[420.,0,240.3],[0,390.,168.7],[0,0,1.]])
    x=np.eye(4);x[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.1]).as_matrix()
    x[:3,3]=[.02,-.04,.07]
    mount=np.array([.1,.2,.3])
    optical=np.array([[-.16,-.10,.7],[.2,.14,.9],[0,0,.6]])
    world=optical@x[:3,:3].T+mount+[0,0,.15]+x[:3,3]
    root=ET.fromstring('<mujoco><worldbody><body name="maver_x4_asm_URDF_link_1" pos=".1 .2 .3"><joint type="slide" axis="0 0 1"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>')
    for i,p in enumerate(world):
        ET.SubElement(root.find('worldbody'),'geom',name=f'point{i}',type='sphere',
            pos=' '.join(map(str,p)),size='.004',rgba='1 0 0 1')
    scene=tmp_path/'source.xml';ET.ElementTree(root).write(scene)
    rendered=tmp_path/'render.xml'
    w,h=inject_camera(scene,rendered,x,k,500,350)
    m=mujoco.MjModel.from_xml_path(str(rendered));d=mujoco.MjData(m)
    mx,my=camera_remap(k,500,350,w,h)
    with mujoco.Renderer(m,height=h,width=w) as renderer:
        renderer.enable_segmentation_rendering()
        for lift in [.15,.23]:
            d.qpos[0]=lift;mujoco.mj_forward(m,d)
            camera_position=mount+[0,0,lift]+x[:3,3]
            np.testing.assert_allclose(d.cam_xpos[m.camera(CAMERA).id],camera_position,atol=1e-12)
            np.testing.assert_allclose(d.cam_xmat[m.camera(CAMERA).id].reshape(3,3),
                x[:3,:3]@np.diag([1,-1,-1]),atol=1e-12)
            points=(world-camera_position)@x[:3,:3]
            uv=points@k.T;uv=uv[:,:2]/uv[:,2,None]
            renderer.update_scene(d,camera=CAMERA)
            seg=cv2.remap(renderer.render(),mx,my,cv2.INTER_NEAREST)
            for i,expected in enumerate(uv):
                ys,xs=np.where((seg[:,:,0]==m.geom(f'point{i}').id)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)))
                assert len(xs)>5
                np.testing.assert_allclose([xs.mean(),ys.mean()],expected,atol=.8)


def test_overlap_detects_misalignment_and_rejects_empty_masks():
    a=np.zeros((100,100),bool);b=a.copy()
    a[20:60,20:60]=True;b[20:60,25:65]=True
    assert mask_metrics(a,a)==dict(iou=1.,boundary_mean_px=0.,boundary_p95_px=0.)
    m=mask_metrics(a,b)
    assert m['iou']==pytest.approx(35/45)
    assert 0<m['boundary_mean_px']<5
    assert m['boundary_p95_px']==pytest.approx(5.)
    with pytest.raises(ValueError):mask_metrics(a,np.zeros_like(a))
    with pytest.raises(ValueError):mask_metrics(a,b[:10])


def test_intrinsics_and_overscan_are_checked():
    with pytest.raises(ValueError):camera_remap(np.eye(3),1920,1080,2000,1160)
    k=np.array([[1000,2,960],[0,1000,540],[0,0,1.]])
    with pytest.raises(ValueError):camera_remap(k,1920,1080,2000,1160)
