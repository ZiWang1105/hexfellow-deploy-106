"""Independent MuJoCo plant; only physical encoders enter the controller.

This backend uses the scene's position actuators. It validates control timing
and IK integration, not physical MIT gains, UDP delivery, or firmware stopping.
"""
import threading
import time
from pathlib import Path
import numpy as np

from .reference import Receipt


class SimulationPlant:
    def __init__(self, scene_path, output, *, submit_delay_s=0.):
        import mujoco
        from mobile_manipulator.sim.scene import build_scene
        from mobile_manipulator.sim.scene_config import load_scene
        from mobile_manipulator.sim.sensors import EncoderReader
        from mobile_manipulator.robot.assembly import JOINTS, BASE_JOINTS
        from mobile_manipulator.control.gripper import BinaryGripperServo
        self.mj = mujoco
        Path(output).mkdir(parents=True, exist_ok=True)
        self.scene = build_scene(load_scene(scene_path), Path(output) / "plant.xml")
        self.model, self.data = self.scene.model, self.scene.data
        self.sensor = EncoderReader(self.model)
        self.joints, self.base_joints = JOINTS, BASE_JOINTS
        self.stop_event = threading.Event()
        self.pending = None
        self.receipt = None
        self.error = None
        self.submit_delay = submit_delay_s
        self.grippers = {(s, jaw): BinaryGripperServo(0.,
            float(self.model.actuator_ctrlrange[self.model.actuator(f"position_{s}_jaw_{jaw}").id, 1]),
            float(self.data.sensordata[self.sensor.jaws[s][i]]), 1.)
            for s in ("left", "right") for i, jaw in enumerate(("positive", "negative"))}
        self.gripper_request = dict(left=None, right=None)
        self.base_request = np.zeros(3)
        self._capture()
        self.thread = threading.Thread(target=self._run, name="mujoco-plant", daemon=True)
        self.thread.start()

    def _capture(self):
        measured = self.sensor.read(self.data.sensordata, time.monotonic())
        # Plant coordinates go only to the renderer, never the policy topic.
        self.snapshot = dict(measured=measured, sim_s=float(self.data.time),
                             qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist())

    def _run(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                pending = self.pending
                if pending is not None and started >= pending[2]:
                    sequence, command, _ = pending
                    for name, q in command["joint_positions"].items():
                        self.data.ctrl[self.model.actuator("position_" + name).id] = q
                    self.base_request = np.asarray(command["base_velocity_body"])
                    self.gripper_request = dict(command["gripper_open"])
                    self.pending = None
                    self.receipt = Receipt(sequence, time.monotonic(), "simulator_applied")
                yaw = self.data.qpos[self.model.joint(self.base_joints[2]).qposadr[0]]
                vx, vy, wz = self.base_request
                twist = (np.cos(yaw)*vx-np.sin(yaw)*vy, np.sin(yaw)*vx+np.cos(yaw)*vy, wz)
                for name, value in zip(self.base_joints, twist):
                    self.data.ctrl[self.model.actuator("velocity_" + name).id] = value
                for (side, jaw), servo in self.grippers.items():
                    self.data.ctrl[self.model.actuator(f"position_{side}_jaw_{jaw}").id] = servo.step(
                        self.gripper_request[side], self.model.opt.timestep)
                self.mj.mj_step(self.model, self.data)
                self.mj.mj_forward(self.model, self.data)
                from mobile_manipulator.sim.transport import check_numerics
                check_numerics(self.data)
                self._capture()
                self.stop_event.wait(max(0., self.model.opt.timestep - (time.monotonic()-started)))
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"

    def read(self):
        if self.error:
            raise RuntimeError(self.error)
        return self.snapshot["measured"]

    def submit(self, sequence, command, mit):
        if self.pending is not None:
            raise RuntimeError("An actuator reference is already pending")
        if mit is None:
            raise ValueError("PinkIK must provide its MIT reference even for position simulation")
        self.pending = (sequence, command, time.monotonic() + self.submit_delay)

    def acknowledgement(self, sequence):
        receipt = self.receipt
        return receipt if receipt is not None and receipt.sequence == sequence else None

    def close(self):
        self.stop_event.set()
        self.thread.join(2)
        if self.thread.is_alive():
            raise RuntimeError("Simulation plant did not close")
