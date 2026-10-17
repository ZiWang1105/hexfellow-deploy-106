"""MuJoCo leaf driver for the shared MIT actuator protocol.

Arm effort is kp*(q_ref-q)+kd*(v_ref-v)+tau_ff, saturated at the same
configured device effort ceilings. Base, lift and jaws model SDK servo inputs.
"""
import threading
import time
from pathlib import Path
import numpy as np

from .reference import Receipt


class LiftPositionServo:
    """Model the lift SDK's position servo with a bounded velocity demand.

    PI velocity feedback rejects unmodeled carried loads. Damping is evaluated
    implicitly by MuJoCo; this is a motor model, never a joint-state clamp.
    Gains model an inner servo and are not measurements of the device firmware.
    """
    def __init__(self, model, data, mujoco, joint_name, speed_limit):
        if not np.isfinite(speed_limit) or speed_limit <= 0:
            raise ValueError('Lift speed limit must be finite and positive')
        joint = model.joint(joint_name)
        self.qpos = int(joint.qposadr[0])
        self.dof = int(joint.dofadr[0])
        self.actuator = model.actuator('position_'+joint_name).id
        self.speed_limit = float(speed_limit)
        self.position_gain = 100.
        self.velocity_gain = 50000.
        self.integral_gain = 250000.
        self.integral_force = 0.
        self.target = float(data.qpos[self.qpos])
        self.force_limits = model.actuator_forcerange[self.actuator].copy()
        # The source URDF's placeholder 10 N joint clamp must agree with the
        # explicit 1800 N assembly motor model, including gravity support.
        model.jnt_actfrclimited[joint.id] = True
        model.jnt_actfrcrange[joint.id] = self.force_limits
        model.actuator_gaintype[self.actuator] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[self.actuator] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[self.actuator] = 0.
        model.actuator_gainprm[self.actuator, 0] = 1.
        model.actuator_biasprm[self.actuator] = 0.
        model.actuator_biasprm[self.actuator, 2] = -self.velocity_gain
        model.actuator_ctrllimited[self.actuator] = False
        model.actuator_forcelimited[self.actuator] = True

    def step(self, data, dt):
        position = float(data.qpos[self.qpos])
        velocity = float(data.qvel[self.dof])
        desired_velocity = float(np.clip(self.position_gain*(self.target-position),
                                         -self.speed_limit, self.speed_limit))
        error = desired_velocity-velocity
        gravity = float(data.qfrc_bias[self.dof])
        force = gravity+self.velocity_gain*error+self.integral_force
        # Conditional integration avoids winding up against the motor effort cap.
        unconstrained_drive = desired_velocity+self.integral_force/self.velocity_gain
        drive_unsaturated = abs(unconstrained_drive) < self.speed_limit or error*unconstrained_drive < 0
        if drive_unsaturated and (self.force_limits[0] <= force <= self.force_limits[1] or error*force < 0):
            self.integral_force = float(np.clip(self.integral_force+self.integral_gain*error*dt,
                                               self.force_limits[0], self.force_limits[1]))
        drive_velocity = float(np.clip(desired_velocity+self.integral_force/self.velocity_gain,
                                       -self.speed_limit, self.speed_limit))
        data.ctrl[self.actuator] = gravity+self.velocity_gain*drive_velocity

    def metadata(self):
        return dict(mode='position_to_bounded_velocity_PI_force', speed_limit_m_s=self.speed_limit,
                    position_gain_s_inverse=self.position_gain, velocity_gain_ns_m=self.velocity_gain,
                    integral_gain_n_m=self.integral_gain, force_limits_n=self.force_limits.tolist(),
                    firmware_identified=False)


class SimulationPlant:
    def __init__(self, scene_path, output, *, submit_delay_s=0., effort_limits=None, profile=None,
                 lift_speed_limit_m_s=.025):
        import mujoco
        from hexfellow_deploy.sim.scene import build_scene
        from hexfellow_deploy.sim.scene_config import load_scene
        from hexfellow_deploy.sim.sensors import EncoderReader
        from hexfellow_deploy.robot.assembly import JOINTS, BASE_JOINTS
        from hexfellow_deploy.robot.joints import LEFT, RIGHT
        from hexfellow_deploy.control.gripper import BinaryGripperServo
        self.mj = mujoco
        Path(output).mkdir(parents=True, exist_ok=True)
        self.scene = build_scene(load_scene(scene_path), Path(output) / "plant.xml")
        self.model, self.data = self.scene.model, self.scene.data
        self.sensor = EncoderReader(self.model)
        self.arm_names = LEFT+RIGHT
        if effort_limits is None or profile is None:
            raise ValueError('MIT simulation requires the shared deployment profile and effort limits')
        self.arm_ids = np.array([self.model.actuator('position_'+n).id for n in self.arm_names])
        self.model.body_gravcomp[:] = 0.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.actuator_gaintype[self.arm_ids] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[self.arm_ids] = mujoco.mjtBias.mjBIAS_AFFINE
        self.model.actuator_gainprm[self.arm_ids] = 0.
        self.model.actuator_gainprm[self.arm_ids,0] = 1.
        self.model.actuator_biasprm[self.arm_ids] = 0.
        self.model.actuator_ctrllimited[self.arm_ids] = False
        self.model.actuator_forcelimited[self.arm_ids] = True
        self.model.actuator_forcerange[self.arm_ids] = np.column_stack((-np.asarray(effort_limits), effort_limits))
        self.lift_servo = LiftPositionServo(self.model, self.data, mujoco, JOINTS[0], lift_speed_limit_m_s)
        servo_profile = getattr(profile, 'mit_servo', None)
        if servo_profile is not None:
            self.model.actuator_biasprm[self.arm_ids, 1] = -np.asarray(servo_profile.kp)
            self.model.actuator_biasprm[self.arm_ids, 2] = -np.asarray(servo_profile.kd)
        # Persist the configured motor model, not build_scene's position-servo
        # template. Initial controls are motor inputs for a measured-pose hold;
        # the actual MIT frame replaces them before physics is enabled.
        arm_qpos = [int(self.model.joint(n).qposadr[0]) for n in self.arm_names]
        arm_dofs = [int(self.model.joint(n).dofadr[0]) for n in self.arm_names]
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[self.arm_ids] = (-self.model.actuator_biasprm[self.arm_ids, 1]*self.data.qpos[arm_qpos]
            -self.model.actuator_biasprm[self.arm_ids, 2]*self.data.qvel[arm_dofs]+self.data.qfrc_bias[arm_dofs])
        self.data.ctrl[self.lift_servo.actuator] = self.data.qfrc_bias[self.lift_servo.dof]
        self.model.key_ctrl[self.model.key('initial').id] = self.data.ctrl
        mujoco.mj_saveLastXML(str(Path(output)/'plant.xml'), self.model)
        # mj_saveLastXML only copies a subset of runtime model attributes back
        # into its source specification. Write the changed flags and force
        # limits explicitly, and retain the full compiled model as well.
        import xml.etree.ElementTree as ET
        tree = ET.parse(Path(output)/'plant.xml')
        for body in tree.findall('.//worldbody//body'):
            body.set('gravcomp', '0')
        for name in JOINTS:
            element = tree.find(f"./actuator/general[@name='position_{name}']")
            aid = self.model.actuator('position_'+name).id
            element.set('ctrllimited', 'false')
            element.set('forcelimited', 'true')
            element.set('forcerange', ' '.join(format(x, '.17g') for x in self.model.actuator_forcerange[aid]))
            element.set('gainprm', ' '.join(format(x, '.17g') for x in self.model.actuator_gainprm[aid]))
            element.set('biasprm', ' '.join(format(x, '.17g') for x in self.model.actuator_biasprm[aid]))
        lift_element = tree.find(f".//worldbody//joint[@name='{JOINTS[0]}']")
        lift_element.set('actuatorfrclimited', 'true')
        lift_element.set('actuatorfrcrange', ' '.join(format(x, '.17g') for x in self.lift_servo.force_limits))
        tree.write(Path(output)/'plant.xml', encoding='unicode')
        mujoco.mj_saveModel(self.model, str(Path(output)/'plant.mjb'))
        self.enabled = False
        self.have_reference = False
        from hexfellow_deploy.sim.recording import SceneRecorder
        self.recorder = SceneRecorder(self.scene, output, video=False, sensors=False)
        from .physics_audit import PhysicsAudit
        self.physics_audit = PhysicsAudit(self.model, self.data)
        self.max_torque = np.zeros(14)
        self.max_lift_speed = 0.
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
        self.snapshot = dict(measured=measured, received={n: measured["timestamp_s"] for n in ("left","right","theta","base")}, sim_s=float(self.data.time),
                             qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist())

    def _run(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                pending = self.pending
                if pending is not None and started >= pending[2]:
                    sequence, command, _, mit = pending
                    selected = mit.select(self.arm_names)
                    self.model.actuator_biasprm[self.arm_ids,1] = -selected.kp
                    self.model.actuator_biasprm[self.arm_ids,2] = -selected.kd
                    self.data.ctrl[self.arm_ids] = selected.motor_input()
                    from hexfellow_deploy.robot.joints import LIFT
                    self.lift_servo.target = command['joint_positions'][LIFT]
                    self.have_reference = True
                    self.base_request = np.asarray(command["base_velocity_body"])
                    self.gripper_request = dict(command["gripper_open"])
                    for (side, jaw), servo in self.grippers.items():
                        self.data.ctrl[self.model.actuator(f"position_{side}_jaw_{jaw}").id] = servo.step(
                            self.gripper_request[side], .02)
                    self.pending = None
                    self.receipt = Receipt(sequence, time.monotonic(), "simulator_applied")
                yaw = self.data.qpos[self.model.joint(self.base_joints[2]).qposadr[0]]
                vx, vy, wz = self.base_request
                twist = (np.cos(yaw)*vx-np.sin(yaw)*vy, np.sin(yaw)*vx+np.cos(yaw)*vy, wz)
                for name, value in zip(self.base_joints, twist):
                    self.data.ctrl[self.model.actuator("velocity_" + name).id] = value
                if not self.enabled or not self.have_reference:
                    self._capture()
                    self.stop_event.wait(.002)
                    continue
                self.lift_servo.step(self.data, self.model.opt.timestep)
                self.mj.mj_step(self.model, self.data)
                self.mj.mj_forward(self.model, self.data)
                from hexfellow_deploy.sim.transport import check_numerics
                check_numerics(self.data)
                self.physics_audit.step(self.data)
                self.max_torque = np.maximum(self.max_torque, np.abs(self.data.actuator_force[self.arm_ids]))
                self.max_lift_speed = max(self.max_lift_speed, abs(float(self.data.qvel[self.lift_servo.dof])))
                self.recorder.step()
                self._capture()
                self.stop_event.wait(max(0., self.model.opt.timestep - (time.monotonic()-started)))
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"

    def acquire(self, *, startup_lift_m=None, progress=lambda report: None, stopped=lambda: False):
        from .startup_lift import LiftArrival, validate_target
        from hexfellow_deploy.robot.joints import LIFT
        validate_target(startup_lift_m)
        self.enabled = True
        self.startup_lift_report = None
        if startup_lift_m is None:
            return
        arrival = LiftArrival(startup_lift_m, time.monotonic())
        arrival.report['initial_height_m'] = self.snapshot['measured']['joint_positions'][LIFT]
        self.startup_lift_report = arrival.report
        # Same bounded lift servo as normal actuation; arms retain the initial
        # measured-pose motor hold. Never teleport simulated joints.
        self.lift_servo.target = startup_lift_m
        self.have_reference = True
        while True:
            self.check()
            measured = self.snapshot['measured']
            if arrival.update(measured['joint_positions'][LIFT], measured['joint_velocities'][LIFT],
                              measured['timestamp_s'], time.monotonic(), stopped=stopped()):
                self.lift_servo.target = measured['joint_positions'][LIFT]
                arrival.report.update(phase='complete', locked_height_m=self.lift_servo.target)
                return
            progress(dict(arrival.report))
            time.sleep(.004)

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def read(self):
        self.check()
        return self.snapshot["measured"]

    def submit(self, sequence, command, mit, *, created_s=None, received=None, measured=None, reference_velocity=None):
        if self.pending is not None:
            raise RuntimeError("An actuator reference is already pending")
        if mit is None:
            raise ValueError("PinkIK must provide its MIT reference for MIT simulation")
        self.pending = (sequence, command, time.monotonic() + self.submit_delay, mit)

    def acknowledgement(self, sequence):
        receipt = self.receipt
        return receipt if receipt is not None and receipt.sequence == sequence else None

    def close(self):
        self.stop_event.set()
        self.thread.join(2)
        if self.thread.is_alive():
            raise RuntimeError("Simulation plant did not close")
        self.recorder.close()
        self.physics_audit.save(self.recorder.directory)
        import json
        (self.recorder.directory/'actuation.json').write_text(json.dumps(dict(
            mode='MIT', profile_torque_limits_nm=self.model.actuator_forcerange[self.arm_ids].tolist(),
            maximum_observed_torque_nm=self.max_torque.tolist(),
            maximum_observed_lift_speed_m_s=self.max_lift_speed,
            gravity_compensation='arm_MIT_feedforward_and_lift_inner_servo',
            lift_servo=self.lift_servo.metadata()), indent=2))
