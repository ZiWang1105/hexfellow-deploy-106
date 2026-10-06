"""Independent rendering component; privileged plant topic ends here."""
from pathlib import Path
import json
import math
import time
from deploy_anything.components.interface.base_interface import DABaseInterface
from .workspace import activate


def simulation_kinematics(config):
    """Reconstruct camera/TCP poses from static robot files and encoders only."""
    from mobile_manipulator.hardware.observations import HardwareKinematics
    from mobile_manipulator.robot.assembly import load_assembly
    from mobile_manipulator.robot.kinematics import Kinematics
    robot, camera = config.data['robot'], config.data['camera']
    urdf = config.resolve(robot['source_checkout']) / robot['urdf']
    assembly = load_assembly(config.resolve(robot['assembly']))
    if camera.get('pose_source', 'calibration') == 'robot_optical_frame':
        return Kinematics(urdf, assembly)
    calibration = json.loads(config.resolve(camera['calibration']).read_text())
    transform = dict(parent_frame=calibration['mount_frame'],
                     T_parent_from_optical=calibration['T_mount_from_camera'])
    return HardwareKinematics(urdf, assembly, {'primary': transform}, 'primary')


def resize_registered_rgbd(frame, width, height):
    """Resize registered sensor pixels without interpolating metric depth edges."""
    import cv2
    import numpy as np
    from mobile_manipulator.hardware.observations import camera_intrinsics
    old_height, old_width = frame.rgb.shape[:2]
    if frame.depth_m.shape != (old_height, old_width):
        raise ValueError('Depth must be registered to the RGB pixel grid')
    for key, expected in [('width', old_width), ('height', old_height)]:
        if key in frame.intrinsics and frame.intrinsics[key] != expected:
            raise ValueError('Camera intrinsics resolution does not match sensor pixels')
    k = np.asarray(camera_intrinsics(frame), dtype=float)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera pinhole intrinsics')
    sx, sy = width/old_width, height/old_height
    # Both the MuJoCo frustum and OpenCV resize use pixel-center coordinates.
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2]+.5)*sx-.5
    k[1, 2] = (k[1, 2]+.5)*sy-.5
    shape = (width, height)
    rgb = cv2.resize(frame.rgb, shape, interpolation=cv2.INTER_AREA)
    depth = cv2.resize(frame.depth_m, shape, interpolation=cv2.INTER_NEAREST_EXACT)
    return rgb, depth, k.tolist()


class CameraAvailability:
    """Observe capture and encoder matching without touching the control clock."""
    def __init__(self, max_age_s, startup_timeout_s, *, started_s):
        if not all(math.isfinite(value) and value > 0 for value in (max_age_s, startup_timeout_s)):
            raise ValueError('Camera health timeouts must be finite and positive')
        self.max_age = max_age_s
        self.startup_timeout = startup_timeout_s
        self.started = started_s
        self.last_publication = None

    def check(self, health, now):
        if health.get('error'):
            raise RuntimeError(health['error'])
        if not health.get('running'):
            raise RuntimeError('RGB-D capture thread stopped unexpectedly')
        age = health.get('age_s')
        if age is not None and (not math.isfinite(age) or age < 0):
            raise RuntimeError('RGB-D capture timestamp is invalid or future-dated')
        if age is not None and age > self.max_age:
            raise RuntimeError(f'RGB-D capture expired: age {age:.3f}s exceeds {self.max_age:.3f}s')
        if self.last_publication is None:
            if now-self.started > self.startup_timeout:
                detail = 'no captured frame' if age is None else 'no frame matched with encoder feedback'
                raise RuntimeError(f'RGB-D initialization timed out after {self.startup_timeout:.3f}s: {detail}')
        elif now-self.last_publication > self.max_age:
            raise RuntimeError(f'Registered RGB-D publication expired: no matched frame published for '
                               f'{now-self.last_publication:.3f}s')

    def published(self, now):
        self.last_publication = now


class SimulationCamera(DABaseInterface):
    def __init__(self, name, plant_state, images, workspace, scene, output, fps=15, time_budget=.04):
        super().__init__(name, [plant_state, images], time_budget)
        self.source, self.images = plant_state, images
        self.workspace, self.scene, self.output, self.fps = workspace, scene, output, fps

    def initialize(self):
        if not math.isfinite(self.fps) or not 0 < self.fps <= 120:
            raise ValueError("Camera frequency must be finite and in (0, 120] Hz")
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        activate(self.workspace)
        import cv2
        import mujoco
        from mobile_manipulator.sim.scene import (build_scene, policy_renderer, policy_camera_maps,
                                                   update_policy_camera, read_policy_depth)
        from mobile_manipulator.sim.scene_config import load_scene
        output = Path(self.output)/'camera'
        output.mkdir(parents=True, exist_ok=True)
        scene = build_scene(load_scene(self.scene), output/'scene.xml')
        kin = simulation_kinematics(scene.config)
        renderer = policy_renderer(scene)
        maps = policy_camera_maps(scene)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        self._ready.value = True
        try:
            while not self.stopping:
                started = time.monotonic()
                row = self.source.get()
                if row:
                    scene.data.qpos[:] = row['qpos']
                    scene.data.qvel[:] = row['qvel']
                    scene.data.time = row['sim_s']
                    mujoco.mj_forward(scene.model, scene.data)
                    update_policy_camera(renderer, scene, option)
                    rgb = cv2.remap(renderer.render(), *maps, interpolation=cv2.INTER_LINEAR)
                    depth = read_policy_depth(renderer, scene, maps)
                    kin.update(row['measured']['joint_positions'])
                    self.images.put(row['measured'], rgb, depth, scene.camera_k.tolist(), kin.packet())
                time.sleep(max(0., 1/self.fps-(time.monotonic()-started)))
        finally:
            renderer.close()


class NativeCamera(DABaseInterface):
    """Physical RGB-D capture; only matched physical encoders accompany images."""
    def __init__(self,name,feedback,images,workspace,deployment_config,fps=15,time_budget=.04,
                 startup_timeout_s=None):
        super().__init__(name,[feedback,images],time_budget)
        self.feedback,self.images = feedback,images
        self.workspace,self.config,self.fps = workspace,deployment_config,fps
        self.startup_timeout = startup_timeout_s

    def initialize(self):
        if not math.isfinite(self.fps) or not 0 < self.fps <= 120:
            raise ValueError("Camera frequency must be finite and in (0, 120] Hz")
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        root = activate(self.workspace)
        from mobile_manipulator.hardware.config import load_deployment
        from mobile_manipulator.hardware.camera_stream import CameraStream,make_source
        from mobile_manipulator.hardware.observations import HardwareKinematics
        from mobile_manipulator.robot.assembly import load_assembly
        config = load_deployment(root/self.config,execute=True,mock=False,research=True)
        primary = config['robot']['policy_camera']
        cfg = config['camera_configs'][primary]
        kin = HardwareKinematics(config['robot']['urdf'],load_assembly(config['robot']['assembly']),
                                  config['calibrations'],primary)
        # Device open and initial encoder availability can take seconds. Once
        # images exist, use the configured sensor freshness interval instead.
        startup_timeout = self.startup_timeout if self.startup_timeout is not None else cfg.get('startup_timeout_s', 30.)
        availability = CameraAvailability(config['robot']['camera_max_age_s'], startup_timeout,
                                          started_s=time.monotonic())
        stream = CameraStream(lambda: make_source(cfg),fps=self.fps).start()
        try:
            while not self.stopping:
                started = time.monotonic()
                health,frame = stream.snapshot()
                row = self.feedback.get()
                availability.check(health,time.monotonic())
                if frame is not None and row is not None:
                    measured = row['measured']
                    if (0 <= time.monotonic()-frame.received_s < config['robot']['camera_max_age_s']
                            and abs(frame.received_s-measured['timestamp_s']) < config['robot']['camera_encoder_max_skew_s']):
                        rgb, depth, k = resize_registered_rgbd(frame, self.images.width, self.images.height)
                        kin.update(measured['joint_positions'])
                        if self.images.put(measured,rgb,depth,k,kin.packet()):
                            availability.published(time.monotonic())
                            self._ready.value = True
                time.sleep(max(0.,started+1/self.fps-time.monotonic()))
        finally:
            stream.close()
