"""Upstream UMI arm/lift IK and bounded body-frame base commands.

There is no localization input or inferred chassis pose. A policy-estimated
object XY point or relative yaw can request ONE pulse, then reobservation.
The same executor receives physical encoder packets from either transport.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from ..interfaces.contract import validate_sensor_packet
from ..robot.joints import JOINTS, SIDES
from .base_motion import BaseMotionConfig
from .umi_ik import DEFAULT_CONFIG, UMIPinkIK


class PoseExecutor:
    def __init__(self, kinematics, dt=.02, *, controller_config=DEFAULT_CONFIG, base_motion=None,
                 runtime_effort_limits_nm=None, runtime_velocity_bounds=None):
        self.kinematics = kinematics
        self.base_motion = BaseMotionConfig.load() if base_motion is None else base_motion
        if min(self.base_motion.ramp_s, self.base_motion.settle_s) < dt:
            raise ValueError("Base ramps and settle time must be at least one controller tick")
        self.ik = UMIPinkIK(kinematics, controller_config, runtime_effort_limits_nm=runtime_effort_limits_nm,
                            runtime_velocity_bounds=runtime_velocity_bounds)
        if not np.isclose(dt, self.ik.dt, atol=1e-12, rtol=0):
            raise ValueError("Executor cadence must match the UMI motion profile")
        self.dt = self.ik.dt
        self.target = None
        self.startup = None
        # None is transport-only HOLD; policy actions always use 0 or 1.
        self.gripper = {side: None for side in SIDES}
        self.needs_observation = False
        self.tracking_stalled = False
        self.navigation = None
        self.requires_base = False
        self._base_reference = np.zeros(3)

    def start_carry(self, preset, measured):
        from .startup_pose import CarryStartup
        self.hold(measured)
        self.startup = CarryStartup(self.ik, preset, measured)

    def submit(self, action, measured):
        if self.startup is not None and not self.startup.complete:
            raise ValueError("Complete carrying startup before submitting policy actions")
        if action.skill is not None:
            raise ValueError('Skills require the shared deployment SkillExecutor')
        self.startup = None
        validate_sensor_packet(measured)
        # Base-relative pose from the latest image. Chassis motion consumes it;
        # there is no world/odometry anchor and no integrated travel estimate.
        self.target = {side: (np.asarray(getattr(action, side).position_m),
                             Rotation.from_quat(getattr(action, side).quaternion_xyzw).as_matrix())
                       for side in SIDES}
        self.arm_targets = {side: (position.copy(), rotation.copy())
                            for side, (position, rotation) in self.target.items()}
        self.pulse_started_s = None
        self.pulse_velocity = np.zeros(3)
        self.needs_observation = False
        self.tracking_stalled = False
        self.navigation = None
        self.requires_base = False
        if action.navigation is not None:
            xy = action.navigation.object_xy_m
            distance = None if xy is None else float(np.linalg.norm(xy))
            yaw = action.navigation.yaw_rad
            turn = abs(yaw) > self.base_motion.yaw_tolerance_rad
            approach = distance is not None and distance > (self.base_motion.approach_distance_m
                                                            + self.base_motion.distance_tolerance_m)
            self.requires_base = turn or approach
            self.navigation = {
                "object_xy_m": None if xy is None else list(xy), "estimated_distance_m": distance,
                "yaw_rad": yaw, "motion": "turn" if turn else ("approach" if approach else "none"),
                "approach_distance_m": self.base_motion.approach_distance_m,
                "distance_tolerance_m": self.base_motion.distance_tolerance_m,
                "source": "policy_rgbd_estimate", "requires_base": self.requires_base}
            if self.requires_base:
                area = self.base_motion.pulse_s - self.base_motion.ramp_s
                if turn:
                    # Relative turn request, not a measured heading target. Consume
                    # this action and reobserve before translating in a new frame.
                    rate = min(self.base_motion.max_yaw_rate_rad_s, abs(yaw)/area)
                    self.pulse_velocity[2] = np.copysign(rate, yaw)
                else:
                    remaining = distance - self.base_motion.approach_distance_m
                    speed = min(self.base_motion.max_speed_m_s, .5*remaining/area)
                    self.pulse_velocity[:2] = speed*np.asarray(xy)/distance
                # Preserve arm/lift reference history and existing gripper commands.
                # Navigation never attempts a grasp/preparation pose first.
                self.ik.hold()
                return
        self.gripper = {side: getattr(action, side).gripper_open for side in SIDES}
        self.ik.track(self.arm_targets)
        self.best_arm_error = float("inf")
        self.last_progress_s = measured["timestamp_s"]

    def hold(self, measured):
        validate_sensor_packet(measured)
        if self.startup is not None:
            self.startup.close()
            self.startup = None
        # Preserve the upstream reference history and use constrained braking.
        # Resetting to measured q here would bypass its acceleration/jerk limits.
        self.ik.hold()
        self.target = None
        self.needs_observation = False
        self.navigation = None
        self.requires_base = False

    def tick(self, measured):
        command, diagnostic = self._tick(measured)
        requested = np.asarray(command['base_velocity_body'], dtype=float)
        # Sensor receipt intervals jitter independently of the 50 Hz reference
        # clock. Bound consecutive commands even when a ramp crosses a slot.
        step = requested-self._base_reference
        linear_step = self.base_motion.max_speed_m_s/self.base_motion.ramp_s*self.dt
        norm = np.linalg.norm(step[:2])
        if norm > linear_step:
            step[:2] *= linear_step/norm
        angular_step = self.base_motion.max_yaw_rate_rad_s/self.base_motion.ramp_s*self.dt
        step[2] = np.clip(step[2],-angular_step,angular_step)
        self._base_reference += step
        command['base_velocity_body'] = self._base_reference.tolist()
        diagnostic['base_stopped'] = bool(np.max(np.abs(self._base_reference)) < 1e-9)
        if not diagnostic['base_stopped'] and np.max(np.abs(requested)) < 1e-9:
            diagnostic.update(mode='base_braking', needs_observation=False, arrived=False)
        if 'navigation' in diagnostic:
            diagnostic['navigation']['commanded_velocity_body'] = self._base_reference.tolist()
        return command, diagnostic

    def _tick(self, measured):
        validate_sensor_packet(measured)
        positions, ik_diagnostic = self.ik.step(measured)
        command = {"joint_positions": positions, "base_velocity_body": [0., 0., 0.],
                   "gripper_open": dict(self.gripper)}
        diagnostic = {"arrived": False, "mode": "hold", "needs_observation": self.needs_observation,
                      "ik_status": ik_diagnostic.status}
        if self.startup is not None:
            diagnostic.update(self.startup.update(measured))
            return command, diagnostic
        if self.navigation is not None:
            diagnostic["navigation"] = dict(self.navigation)
        if self.target is None or self.needs_observation:
            if self.needs_observation:
                diagnostic["mode"] = ("tracking_stalled" if self.tracking_stalled else "reobserve")
            return command, diagnostic
        if self.requires_base:
            # Constrained braking must finish, but TCP arrival is irrelevant.
            # Start against the current tick, never the old image timestamp.
            if self.pulse_started_s is None:
                if self.ik.core.control_state != "HOLD":
                    diagnostic["mode"] = "navigation_hold"
                    return command, diagnostic
                self.pulse_started_s = measured["timestamp_s"]
            elapsed = measured["timestamp_s"] - self.pulse_started_s
            envelope = max(0., min(1., elapsed/self.base_motion.ramp_s,
                                  (self.base_motion.pulse_s-elapsed)/self.base_motion.ramp_s))
            command["base_velocity_body"] = (self.pulse_velocity*envelope).tolist()
            diagnostic["mode"] = "base_pulse" if elapsed < self.base_motion.pulse_s else "brake"
            diagnostic["navigation"].update(
                pulse_elapsed_s=elapsed, commanded_velocity_body=command["base_velocity_body"])
            if elapsed >= self.base_motion.pulse_s+self.base_motion.settle_s:
                self.needs_observation = True
                diagnostic.update(mode="reobserve", needs_observation=True)
            # No travel sensor: the estimate is never decremented by commanded
            # motion, and the deferred TCP/gripper action is never resumed.
            return command, diagnostic
        joints = measured["joint_positions"]
        self.kinematics.update(joints)
        errors = {}
        for side, (position, rotation) in self.arm_targets.items():
            pose = self.kinematics.pose(side)
            errors[side] = {"position_m": float(np.linalg.norm(position-pose.translation)),
                "orientation_rad": float(np.linalg.norm(Rotation.from_matrix(rotation@pose.rotation.T).as_rotvec()))}
        pos_error = max(e["position_m"] for e in errors.values())
        rot_error = max(e["orientation_rad"] for e in errors.values())
        arm_arrived = pos_error < .007 and rot_error < .06
        # Include orientation progress and lift motion: both are part of UMI IK.
        error = pos_error + .1*rot_error
        if error < self.best_arm_error-.001:
            self.best_arm_error = error
            self.last_progress_s = measured["timestamp_s"]
        measured_speed = np.linalg.norm([measured["joint_velocities"][n] for n in JOINTS])
        if (not arm_arrived and measured_speed < .02
                and measured["timestamp_s"]-self.last_progress_s > 2.):
            self.tracking_stalled = True
            self.needs_observation = True
            self.ik.hold()
            diagnostic.update(mode="tracking_stalled", needs_observation=True,
                              arm_errors=errors)
            return command, diagnostic
        diagnostic.update(arrived=arm_arrived, mode="arm",
                          needs_observation=self.needs_observation,
                          arm_errors=errors,
                          arm_position_error_m=pos_error, arm_orientation_error_rad=rot_error)
        return command, diagnostic
