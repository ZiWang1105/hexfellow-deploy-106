"""MuJoCo runs in a separate process; only sensor packets cross to control."""
import multiprocessing as mp
from pathlib import Path
import traceback


def check_numerics(data):
    """Fast object rotation is valid dynamics, not an episode stop signal."""
    import numpy as np
    if not all(np.isfinite(getattr(data, name)).all() for name in ("qpos", "qvel", "qacc")):
        raise RuntimeError("Simulation produced non-finite state")
    # MuJoCo may auto-reset invalid state to finite values. Its warning counters
    # must therefore be checked as well, immediately after each physics step.
    if np.any(data.warning.number):
        raise RuntimeError("MuJoCo reported a numerical or solver warning")


def _plant(connection, scene_path, directory, record, gripper_travel_time_s, images=None):
    import mujoco
    import numpy as np
    from .scene import build_scene
    from .scene_config import load_scene
    from .recording import SceneRecorder
    from .sensors import EncoderReader
    from ..control.gripper import BinaryGripperServo, validate_gripper
    from ..robot.assembly import JOINTS, BASE_JOINTS
    from ..hardware.scheduling import runtime_cpu
    scheduling = runtime_cpu("gateway")
    scheduling.__enter__()
    recorder = None
    streams = None
    recorder_closed = False
    try:
        directory = Path(directory)
        scene = build_scene(load_scene(scene_path), directory / "scene.xml")
        model, data = scene.model, scene.data
        if images is not None:
            from .streaming import PlantStreams
            streams = PlantStreams(scene, scene_path, directory, images, record)
        recorder = SceneRecorder(scene, directory, video=record and images is None, sensors=images is None)
        encoders = EncoderReader(model)
        gripper_servos = { (side, jaw): BinaryGripperServo(0.,
            float(model.actuator_ctrlrange[model.actuator(f'position_{side}_jaw_{jaw}').id, 1]),
            float(data.sensordata[encoders.jaws[side][i]]), gripper_travel_time_s)
            for side in ('left', 'right') for i, jaw in enumerate(('positive', 'negative')) }
        def read(image=False):
            result = encoders.read(data.sensordata, data.time)
            if image:
                result["rgb"], result["depth_m"] = recorder.rgbd()
            return result
        connection.send({"ok": True})
        while True:
            operation, payload = connection.recv()
            if operation == "read":
                connection.send(read(payload))
            elif operation == "advance":
                command, dt = payload
                if set(command['gripper_open']) != {'left', 'right'}:
                    raise ValueError('Gripper commands require both hands')
                for value in command['gripper_open'].values():
                    validate_gripper(value, allow_hold=True)
                for name, value in command["joint_positions"].items():
                    data.ctrl[model.actuator("position_"+name).id] = value
                # Actuator emulation: body-frame twist to the planar plant's
                # world-axis actuators. This plant-internal orientation never
                # crosses the observation boundary or enters the controller.
                yaw = data.qpos[model.joint(BASE_JOINTS[2]).qposadr[0]]
                vx, vy, wz = command["base_velocity_body"]
                world_twist = [np.cos(yaw)*vx-np.sin(yaw)*vy, np.sin(yaw)*vx+np.cos(yaw)*vy, wz]
                for name, value in zip(BASE_JOINTS, world_twist):
                    data.ctrl[model.actuator("velocity_"+name).id] = value
                for side in ("left", "right"):
                    for jaw in ("positive", "negative"):
                        actuator = model.actuator(f"position_{side}_jaw_{jaw}").id
                        data.ctrl[actuator] = gripper_servos[side, jaw].step(command['gripper_open'][side], dt)
                for _ in range(round(dt/model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    check_numerics(data)
                mujoco.mj_forward(model, data)
                recorder.step()
                if streams is not None:
                    streams.publish(data)
                connection.send(read())
            elif operation == "close":
                recorder.close()
                recorder_closed = True
                if streams is not None:
                    streams.close()
                    streams = None
                    video_path = directory / "video/pickup.mp4"
                    if video_path.exists():
                        video_path.replace(directory / "pickup.mp4")
                connection.send({"ok": True})
                break
            else:
                raise ValueError("Unsupported sensor/actuator operation")
    except BaseException:
        error = traceback.format_exc()
        try:
            connection.send({"error": error})
        except (OSError, EOFError):
            pass
    finally:
        try:
            if recorder is not None and not recorder_closed:
                recorder.close()
            if streams is not None:
                streams.close()
        finally:
            connection.close()
            scheduling.__exit__(None, None, None)


class MujocoTransport:
    def __init__(self, scene_path, directory, *, record=True, gripper_travel_time_s=1., images=None):
        self.failed = False
        self.closed = False
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_plant, args=(child, str(scene_path), str(directory), record, gripper_travel_time_s, images))
        self.process.start()
        self._receive()

    def _receive(self):
        try:
            if not self.connection.poll(60):
                raise TimeoutError("Simulation transport timed out")
            value = self.connection.recv()
        except (OSError, EOFError, TimeoutError):
            self.failed = True
            raise
        if "error" in value:
            self.failed = True
            raise RuntimeError(value["error"])
        return value

    def read(self, *, image):
        self.connection.send(("read", image))
        return self._receive()

    def advance(self, command, dt):
        self.connection.send(("advance", (command, dt)))
        return self._receive()

    def close(self):
        if self.closed:
            return
        try:
            if self.process.is_alive() and not self.failed:
                self.connection.send(("close", None))
                self._receive()
        finally:
            # A worker reporting an error is already leaving its receive loop.
            # Let it finalize recordings instead of writing to its closed pipe.
            self.process.join(timeout=35)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.connection.close()
            self.closed = True
