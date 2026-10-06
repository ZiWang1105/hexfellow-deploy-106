"""Sensor geometry tests with no live cameras or robot connections."""
import copy
from pathlib import Path
import time

import numpy as np
import pytest

from hexfellow_deploy.workspace import activate

ROOT = activate(Path(__file__).resolve().parents[3])
from hexfellow_deploy.camera import CameraAvailability, NativeCamera, resize_registered_rgbd, simulation_kinematics
from mobile_manipulator.hardware.camera_stream import CameraFrame
from mobile_manipulator.sim.scene_config import SceneConfig, load_scene


def test_native_resize_preserves_metric_depth_and_pixel_center_projection():
    intrinsics = dict(fx=600., fy=620., cx=3.1, cy=2.6, width=8, height=6)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    depth = np.tile(np.array([0., 0., 1.25, 1.25, 2.5, 2.5, 7., 7.], dtype=np.float32), (6, 1))
    frame = CameraFrame(rgb, depth, 1., 2., intrinsics)
    resized, metric, matrix = resize_registered_rgbd(frame, 4, 3)
    assert resized.shape == (3, 4, 3) and resized.dtype == np.uint8
    assert metric.shape == (3, 4) and metric.dtype == np.float32
    np.testing.assert_array_equal(metric[0], [0., 1.25, 2.5, 7.])
    k = np.asarray(matrix)
    ray = np.array([.001, -.002, 1.])
    source_pixel = np.array([intrinsics['fx']*ray[0]+intrinsics['cx'],
                             intrinsics['fy']*ray[1]+intrinsics['cy']])
    np.testing.assert_allclose((k@ray)[:2], (source_pixel+.5)*.5-.5)
    assert frame.intrinsics == intrinsics


def test_resize_rejects_unregistered_or_mislabeled_sensor_pixels():
    frame = CameraFrame(np.zeros((6, 8, 3), dtype=np.uint8), np.zeros((6, 8), dtype=np.float32),
                        1., 1., dict(fx=10., fy=10., cx=3., cy=2., width=10, height=6))
    with pytest.raises(ValueError, match='intrinsics resolution'):
        resize_registered_rgbd(frame, 4, 3)
    frame.intrinsics['width'] = 8
    frame.depth_m = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match='registered'):
        resize_registered_rgbd(frame, 4, 3)


def test_camera_health_allows_encoder_startup_then_reports_missing_publication():
    monitor = CameraAvailability(.25,30.,started_s=10.)
    health = dict(error=None,running=True,age_s=.05)
    monitor.check(health,39.)
    with pytest.raises(RuntimeError,match='no frame matched with encoder feedback'):
        monitor.check(health,40.1)
    monitor.published(40.1)
    monitor.check(health,40.2)
    with pytest.raises(RuntimeError,match='no matched frame published'):
        monitor.check(health,40.36)


@pytest.mark.parametrize('mode,reason', [('no_frames','no captured frame'), ('blocked','RGB-D capture expired')])
def test_native_camera_surfaces_missing_and_blocked_capture_without_devices(monkeypatch, mode, reason):
    from mobile_manipulator.hardware import camera_stream, config, observations
    from mobile_manipulator.robot import assembly
    class Source:
        def __init__(self):
            self.reads = 0
            self.closed = False
        def open(self):
            pass
        def read(self):
            self.reads += 1
            if mode == 'no_frames':
                return None
            if self.reads > 1:
                # Model a SDK read stalled beyond the configured sensor age.
                time.sleep(.3)
                return None
            return CameraFrame(np.zeros((24,32,3),dtype=np.uint8),np.ones((24,32),dtype=np.float32),
                time.monotonic(),1.,dict(fx=30.,fy=30.,cx=16.,cy=12.,width=32,height=24))
        def close(self):
            self.closed = True
    source = Source()
    monkeypatch.setattr(camera_stream,'make_source',lambda settings: source)
    monkeypatch.setattr(config,'load_deployment',lambda *args,**kwargs: dict(
        robot=dict(policy_camera='head',urdf='',assembly='',camera_max_age_s=.08,camera_encoder_max_skew_s=.08),
        camera_configs={'head': {}},calibrations={}))
    monkeypatch.setattr(assembly,'load_assembly',lambda path: None)
    class Kinematics:
        def __init__(self,*args):
            pass
        def update(self,joints):
            pass
        def packet(self):
            return {}
    monkeypatch.setattr(observations,'HardwareKinematics',Kinematics)
    class Feedback:
        name = 'feedback'
        def get(self):
            return dict(measured=dict(timestamp_s=time.monotonic(),joint_positions={}))
    class Images:
        name,width,height = 'images',32,24
        def __init__(self):
            self.count = 0
        def put(self,*args):
            self.count += 1
            return True
    images = Images()
    camera = NativeCamera('camera',Feedback(),images,str(ROOT),'unused.yaml',fps=120,startup_timeout_s=.15)
    camera.initialize()
    started = time.monotonic()
    with pytest.raises(RuntimeError,match=reason):
        camera.run()
    assert time.monotonic()-started < 1.5
    assert source.closed
    assert camera.ready == (mode == 'blocked')
    assert bool(images.count) == (mode == 'blocked')


@pytest.mark.parametrize('pose_source', ['robot_optical_frame', 'calibration'])
def test_simulated_fk_matches_camera_and_excludes_world_pose(tmp_path, pose_source):
    import mujoco
    from mobile_manipulator.robot.joints import BASE
    from mobile_manipulator.sim.scene import build_scene
    original = load_scene(ROOT/'configs/scenes/apple_on_table.yaml')
    data = copy.deepcopy(original.data)
    data['camera']['pose_source'] = pose_source
    data['robot']['initial_base_xyyaw'] = [.4, -.3, .7]
    config = SceneConfig(original.path, data)
    kin = simulation_kinematics(config)
    kin.update(data['robot']['initial_joints'])
    expected = kin.packet()
    assert set(expected) == {'tcp_poses', 'camera_T_base_from_optical'}
    scene = build_scene(config, tmp_path/'camera.xml')
    mujoco.mj_forward(scene.model, scene.data)
    camera = scene.model.camera('policy_rgb').id
    base = scene.model.body(BASE).id
    rotation = scene.data.xmat[base].reshape(3, 3)
    relative = np.eye(4)
    relative[:3, :3] = rotation.T @ scene.data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
    relative[:3, 3] = rotation.T @ (scene.data.cam_xpos[camera]-scene.data.xpos[base])
    np.testing.assert_allclose(expected['camera_T_base_from_optical'], relative, atol=1e-6)
    # Neither initial chassis pose nor object placement changes encoder-only FK.
    data['robot']['initial_base_xyyaw'] = [1., 2., -.6]
    data['entities'][0]['position_m'][0] += 1.
    other = simulation_kinematics(SceneConfig(original.path, data))
    other.update(data['robot']['initial_joints'])
    assert other.packet() == expected
