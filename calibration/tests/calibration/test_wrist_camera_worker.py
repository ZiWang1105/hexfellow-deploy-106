"""Wrist RGB clock rejection and independent optical projection checks."""
import sys
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/camera'))
from wrist_camera_worker import ExposureClock, SimulationRGB, signed_counter_delta_us


def camera_scene(tmp_path):
    from policy_sim.control.robot import CONTROLLED_JOINTS
    root = ET.fromstring('<mujoco><visual><global offwidth="1400" offheight="800"/></visual><worldbody/></mujoco>')
    world = root.find('worldbody')
    ET.SubElement(world, 'camera', name='wrist_calibration_view', quat='0 1 0 0')
    parent = world
    for i, name in enumerate(CONTROLLED_JOINTS):
        parent = ET.SubElement(parent, 'body', name=f'joint{i}', pos='10 0 0' if i == 0 else '0 0 .1')
        ET.SubElement(parent, 'joint', name=name, type='hinge', axis='0 0 1')
        ET.SubElement(parent, 'inertial', pos='0 0 0', mass='1', diaginertia='1 1 1')
    points = np.array([[-.12, -.10, .7], [.20, .16, .9], [.02, .01, .6]])
    for i, point in enumerate(points):
        ET.SubElement(world, 'geom', name=f'point{i}', type='sphere',
                      pos=' '.join(map(str, point)), size='.004')
    scene = tmp_path/'scene.xml'
    ET.ElementTree(root).write(scene)
    return scene, points


def frame(clock, i, *, domain='timestamp_domain.global_time', delay=.014, sdk_jitter=0.):
    t = i/30
    metadata = dict(frame_timestamp=int(1e6+t*1e6), sensor_timestamp=int(992000+t*1e6))
    bridge = dict(realtime_s=1700000000.+t+delay,
                  monotonic_s=100.+t+delay, uncertainty_s=.000001)
    return clock.map((1700000000.+t+.008+sdk_jitter)*1000,
                     domain, metadata, 100.+t+delay, bridge, i)


def test_global_metadata_maps_exposure_not_host_receipt():
    clock = ExposureClock()
    first = frame(clock, 0)
    assert not first['timing_accepted']
    for i in range(1, 91):
        result = frame(clock, i)
    assert result['timing_accepted'], result
    assert result['image_time'] == pytest.approx(103., abs=1e-6)
    assert result['host_receipt_monotonic_s']-result['image_time'] == pytest.approx(.014, abs=1e-6)
    assert .002 <= result['clock_mapping_uncertainty_s'] < .005
    assert result['uncertainty_is_estimate']
    assert 'no hardware exposure trigger' in result['synchronization']


@pytest.mark.parametrize('domain', ['timestamp_domain.system_time', 'timestamp_domain.hardware_clock'])
def test_unmapped_or_receipt_fallback_is_not_synchronized(domain):
    result = frame(ExposureClock(), 1, domain=domain)
    assert not result['timing_accepted']
    assert result['clock_mapping_uncertainty_s'] > .005
    assert 'fallback' in result['timing_reason']


def test_missing_exposure_metadata_is_rejected():
    result = ExposureClock().map(1700000000000., 'global_time', {}, 100.,
                                dict(realtime_s=1700000000., monotonic_s=100., uncertainty_s=0.), 1)
    assert not result['timing_accepted']
    assert 'metadata unavailable' in result['timing_reason']


def test_mapping_jump_and_usb_delayed_frames_are_rejected():
    clock = ExposureClock()
    for i in range(91):
        frame(clock, i)
    assert not frame(clock, 91, sdk_jitter=.020)['timing_accepted']
    delayed = ExposureClock()
    for i in range(91):
        result = frame(delayed, i, delay=.300)
    assert not result['timing_accepted']
    assert '150 ms old' in result['timing_reason']


def test_metadata_counter_wrap_preserves_small_exposure_difference():
    assert signed_counter_delta_us(2**32-5000, 3000) == -8000
    assert signed_counter_delta_us(3000, 2**32-5000) == 8000


def test_simulation_camera_uses_offcentre_principal_point_and_unequal_focal_lengths(tmp_path):
    import mujoco
    scene, points = camera_scene(tmp_path)
    k = np.array([[930., 0., 617.3], [0., 900., 389.7], [0., 0., 1.]])
    camera = SimulationRGB(scene, 'test', k)
    try:
        camera.renderer.enable_segmentation_rendering()
        mujoco.mj_forward(camera.model, camera.data)
        camera.renderer.update_scene(camera.data, camera=camera.camera_name)
        segmentation = cv2.remap(camera.renderer.render(), camera.mapx, camera.mapy, cv2.INTER_NEAREST)
        projected = points@k.T
        projected = projected[:, :2]/projected[:, 2, None]
        for i, expected in enumerate(projected):
            ys, xs = np.where((segmentation[:, :, 0] == camera.model.geom(f'point{i}').id)
                              & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
            assert len(xs) > 10
            np.testing.assert_allclose([xs.mean(), ys.mean()], expected, atol=.6)
    finally:
        camera.close()


def test_simulation_worker_cli_first_capture_and_rejected_vision_continue(tmp_path):
    """Exercise the actual process/JSON/image boundary, including timing merge."""
    scene, _ = camera_scene(tmp_path)
    directory = tmp_path/'camera-ipc'
    directory.mkdir()
    script = Path(__file__).resolve().parents[2]/'tools/camera/wrist_camera_worker.py'
    env = dict(os.environ, MUJOCO_GL='egl')
    process = subprocess.Popen([sys.executable, str(script), '--mode', 'sim',
                                '--scene', str(scene), '--directory', str(directory)],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def wait_for(path):
        deadline = time.monotonic()+15
        while not path.exists():
            if process.poll() is not None:
                pytest.fail('Camera worker exited: '+process.communicate()[0])
            if time.monotonic() > deadline:
                pytest.fail('Camera worker IPC timeout: '+str(path))
            time.sleep(.02)
        return json.loads(path.read_text())

    try:
        ready = wait_for(directory/'ready.json')
        assert ready['mode'] == 'sim'
        assert ready['width'] == 1280 and ready['height'] == 720
        for ident in (1, 2):
            request = dict(id=ident, after=time.monotonic(), q=[0.]*18)
            temporary = directory/'request.tmp'
            temporary.write_text(json.dumps(request))
            temporary.replace(directory/'request.json')
            result = wait_for(directory/str(ident)/'result.json')
            assert result['simulation_only'] and result['timing_accepted']
            # This independent fixture has no board. A bad vision observation
            # must still produce a raw image and leave the worker alive.
            assert not result['accepted']
            image = cv2.imread(str(directory/str(ident)/'left.png'))
            assert image.shape == (720, 1280, 3)
            assert (directory/str(ident)/'metadata.json').exists()
            assert (directory/'preview.jpg').exists()
            assert process.poll() is None
    finally:
        (directory/'stop').touch()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    assert process.returncode == 0, output
