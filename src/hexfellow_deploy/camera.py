"""Independent rendering component; privileged plant topic ends here."""
from pathlib import Path
from collections import Counter, deque
import json
import math
import time
from deploy_anything.components.interface.base_interface import DABaseInterface
from .workspace import activate


def simulation_kinematics(config):
    """Reconstruct camera/TCP poses from static robot files and encoders only."""
    from hexfellow_deploy.hardware.observations import HardwareKinematics
    from hexfellow_deploy.robot.assembly import load_assembly
    from hexfellow_deploy.robot.kinematics import Kinematics
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
    from hexfellow_deploy.hardware.observations import camera_intrinsics
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
        self.check_capture(health, now)
        self.check_publication(health, now)

    def check_capture(self, health, now):
        if health.get('error'):
            raise RuntimeError(health['error'])
        if not health.get('running'):
            raise RuntimeError('RGB-D capture thread stopped unexpectedly')
        age = health.get('age_s')
        if age is not None and (not math.isfinite(age) or age < 0):
            raise RuntimeError('RGB-D capture timestamp is invalid or future-dated')
        if age is not None and age > self.max_age:
            raise RuntimeError(f'RGB-D capture expired: age {age:.3f}s exceeds {self.max_age:.3f}s')

    def check_publication(self, health, now):
        age = health.get('age_s')
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
        from hexfellow_deploy.sim.scene import (build_scene, policy_renderer, policy_camera_maps,
                                                   update_policy_camera, read_policy_depth)
        from hexfellow_deploy.sim.scene_config import load_scene
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
                 startup_timeout_s=None, output=None, feedback_hz=250):
        super().__init__(name,[feedback,images],time_budget)
        self.feedback,self.images = feedback,images
        self.workspace,self.config,self.fps = workspace,deployment_config,fps
        self.startup_timeout = startup_timeout_s
        self.output, self.feedback_hz = output, feedback_hz

    def initialize(self):
        if not math.isfinite(self.fps) or not 0 < self.fps <= 120:
            raise ValueError("Camera frequency must be finite and in (0, 120] Hz")
        if not math.isfinite(self.feedback_hz) or not 0 < self.feedback_hz <= 1000:
            raise ValueError('Camera feedback frequency must be finite and in (0, 1000] Hz')
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        root = activate(self.workspace)
        from hexfellow_deploy.hardware.config import load_deployment
        from hexfellow_deploy.hardware.camera_stream import CameraStream,make_source
        from hexfellow_deploy.hardware.observations import HardwareKinematics
        from hexfellow_deploy.robot.assembly import load_assembly
        from .camera_sync import EncoderHistory, FeedbackSampler
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
        max_skew = config['robot']['camera_encoder_max_skew_s']
        history = EncoderHistory(availability.max_age+max_skew)
        sampler = FeedbackSampler(self.feedback, history, 1/self.feedback_hz).start()
        counts, events = Counter(), deque(maxlen=256)
        last_frame = None
        last_report = 0.
        failure = None
        health = {}

        def report(now):
            nonlocal last_report
            if self.output is not None:
                path = Path(self.output)/'camera_health.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(
                    observed_s=now, error=failure, capture=health, counts=dict(counts),
                    last_publication_s=availability.last_publication, last_frame_received_s=last_frame,
                    feedback_history=history.stats(), feedback_reads=sampler.reads,
                    feedback_read_misses=sampler.misses, feedback_error=sampler.error,
                    max_age_s=availability.max_age, max_skew_s=max_skew,
                    recent_attempts=list(events)), allow_nan=False))
                temporary.replace(path)
            last_report = now

        stream = CameraStream(lambda: make_source(cfg),fps=self.fps).start()
        try:
            while not self.stopping:
                started = time.monotonic()
                if sampler.error:
                    raise RuntimeError(f'Camera encoder sampler failed: {sampler.error}')
                health = stream.health()
                availability.check_capture(health, time.monotonic())
                # Poll only metadata between captures. Retry SHM contention at
                # the feedback cadence, rather than losing a full camera period.
                if health.get('received_s') is not None and health['received_s'] != last_frame:
                    health, frame = stream.snapshot()
                    now = time.monotonic()
                    attempt = dict(started_s=started, capture_sequence=health.get('sequence'),
                                   snapshot_ms=(now-started)*1000)
                    if frame is None:
                        reason = 'no_frame'
                    else:
                        attempt.update(frame_received_s=frame.received_s,
                                       device_timestamp_s=frame.device_timestamp_s,
                                       frame_age_s=now-frame.received_s)
                        row = history.closest(frame.received_s, now, max_skew)
                        reason = 'no_encoder_match' if row is None else 'matched'
                        if row is not None:
                            measured = row['measured']
                            attempt.update(encoder_timestamp_s=measured['timestamp_s'],
                                           encoder_skew_s=measured['timestamp_s']-frame.received_s)
                    if reason == 'matched':
                        rgb, depth, k = resize_registered_rgbd(frame, self.images.width, self.images.height)
                        kin.update(measured['joint_positions'])
                        attempt['prepare_ms'] = (time.monotonic()-now)*1000
                        publish_started = time.monotonic()
                        if not 0 <= publish_started-frame.received_s <= availability.max_age:
                            reason = 'expired_before_publication'
                        elif self.images.put(measured,rgb,depth,k,kin.packet()):
                            completed = time.monotonic()
                            availability.published(completed)
                            last_frame = frame.received_s
                            reason = 'published'
                            self._ready.value = True
                        else:
                            reason = 'image_topic_busy'
                        attempt['publish_ms'] = (time.monotonic()-publish_started)*1000
                    counts[reason] += 1
                    events.append(dict(attempt, result=reason, completed_s=time.monotonic()))
                # A valid recovered frame must get a publication attempt before
                # the previous publication gap is evaluated.
                health = stream.health()
                now = time.monotonic()
                availability.check(health, now)
                if now-last_report >= 1.:
                    report(now)
                time.sleep(max(0.,started+1/self.feedback_hz-time.monotonic()))
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
            report(time.monotonic())
            raise
        finally:
            try:
                stream.close()
            finally:
                sampler.close()
                report(time.monotonic())
