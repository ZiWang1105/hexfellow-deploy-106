"""Cartesian manual-pose planning with the editable, unmodified UMI IK core.

Planning is device-free. Measured encoders seed the first IK attempt; a bounded
model-home seed retry can escape a local minimum when solving preset endpoints.
The preset contains TCP and elbow frame poses only. The numerical retry seed is
never an executable waypoint. Native keyboard POSITION execution retains its existing
ownership, speed, following-error and key-release checks. This is not a live MIT
publisher, a policy path, or a hardware/dynamic validation of an offline solution.
"""
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ..control.lift_limits import validate_lift_target
from ..robot.joints import JOINTS, LEFT, RIGHT, LIFT
from ..robot.upstream import load_umi


class KeyboardIKNonconvergence(RuntimeError):
    """Endpoint optimization stalled, distinct from a core fault or cancellation."""


def load_ik_pose(path):
    value = yaml.safe_load(Path(path).read_text())
    if (not isinstance(value, dict) or set(value) !=
            {'schema_version', 'frame', 'lift_m', 'motion_profile', 'stages'}
            or value['schema_version'] != 1 or value['frame'] != 'robot_base'
            or type(value['lift_m']) not in (int, float) or not np.isfinite(value['lift_m'])
            or not isinstance(value['motion_profile'], str)
            or not isinstance(value['stages'], list) or not 1 <= len(value['stages']) <= 8):
        raise ValueError('Invalid Cartesian keyboard preset')
    validate_lift_target(value['lift_m'], 'Cartesian keyboard preset')
    for stage in value['stages']:
        if not isinstance(stage, dict) or set(stage) != {'left', 'right'}:
            raise ValueError('Cartesian stage requires both arms')
        for targets in stage.values():
            if not isinstance(targets, dict) or set(targets) != {'tcp', 'elbow'}:
                raise ValueError('Cartesian arm requires TCP and elbow frames')
            for pose in targets.values():
                if not isinstance(pose, dict) or set(pose) != {'position_m', 'quaternion_xyzw'}:
                    raise ValueError('Cartesian frames require position and quaternion')
                p, q = np.asarray(pose['position_m'], float), np.asarray(pose['quaternion_xyzw'], float)
                if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all()
                        or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-6):
                    raise ValueError('Invalid Cartesian frame pose')
    return value


class KeyboardIKPlanner:
    def __init__(self, preset, arm_speed):
        load_umi()
        # Preserve UMI's qpsolvers/Pinocchio import order.
        from policy_sim.control.ik_core import PinkIKCore
        from policy_sim.control.motion_profile import load_motion_profile, controller_config_for_profile
        from pink.tasks import FrameTask
        import pinocchio as pin
        self.pin, self.preset = pin, preset
        self.core = PinkIKCore(config=controller_config_for_profile(load_motion_profile(preset['motion_profile'])),
            base_mode='fixed', locked_joint_names=(LIFT,), measured_pose_tracking=False,
            runtime_velocity_bounds={n: .025 if n == LIFT else arm_speed for n in JOINTS})
        self.indices = {n: self.core.model.joints[self.core.model.getJointId(n)].idx_q for n in JOINTS}
        self.data = self.core.model.createData()
        self.planning_diagnostics = []
        self.elbows = {}
        self.elbow_tasks = {}
        for side, names in [('left', LEFT), ('right', RIGHT)]:
            # The JOINT frame at hinge 4 defines the forearm's bending plane.
            frame = names[3]
            self.elbows[side] = self.core.model.getFrameId(frame)
            # The elbow frame selects the folding branch. A weak orientation
            # objective lets a distant TCP pull the shoulder toward its limits
            # and trap the solve, even though the preset itself is reachable.
            task = FrameTask(frame, position_cost=20., orientation_cost=100., gain=.5, lm_damping=1e-4)
            self.elbow_tasks[side] = task
            self.core.tasks.append(task)

    def transform(self, value):
        return self.pin.SE3(Rotation.from_quat(value['quaternion_xyzw']).as_matrix(),
                            np.asarray(value['position_m'], float))

    def completion_errors(self, sdk_positions, lift_m, mapping, stage_index=-1):
        """Robot-relative FK errors from measured encoders, never plant truth."""
        q = np.zeros(self.core.model.nq)
        q[self.indices[LIFT]] = lift_m
        for side, names in [('left', LEFT), ('right', RIGHT)]:
            angles, _ = mapping.arm_state(side, sdk_positions[side], [0.]*7)
            for name, angle in zip(names, angles):
                q[self.indices[name]] = angle
        self.pin.framesForwardKinematics(self.core.model, self.data, q)
        result = {}
        for side in ('left', 'right'):
            actual = self.data.oMf[self.core.model.getFrameId(side+'_tcp')]
            # Compare policy pad centers, rather than the upstream fingertip.
            actual = actual*self.pin.SE3(np.eye(3), np.array([0.,0.,-.017]))
            target = self.transform(self.preset['stages'][stage_index][side]['tcp'])
            result[side] = dict(position_error_m=float(np.linalg.norm(actual.translation-target.translation)),
                orientation_error_rad=float(np.linalg.norm(self.pin.log3(actual.rotation.T@target.rotation))))
        return result

    def plan(self, measured_joints, cancelled=lambda: False):
        self.planning_diagnostics = []
        try:
            return self._plan_from_seed(measured_joints, cancelled, seed_name='measured')
        except KeyboardIKNonconvergence:
            # Offline endpoint solving is not a live control-history reset.
            # Recompute every stage from a deterministic second seed, keeping
            # partial solutions private until the complete preset converges.
            # Execution still interpolates from the actual measured encoders;
            # the model-home seed is never returned as a motion target.
            seed = {name: float(self.core.home_q[index]) for name, index in self.indices.items()}
            return self._plan_from_seed(seed, cancelled, seed_name='model_home')

    def _plan_from_seed(self, seed_joints, cancelled, *, seed_name):
        from policy_sim.hardware.hexfellow_control_console import MotionCancelled
        if cancelled():
            raise MotionCancelled('Keyboard released during IK planning')
        q = np.empty(self.core.model.nq)
        for name, index in self.indices.items():
            q[index] = seed_joints[name]
        # Solve endpoint arm geometry at the requested final lift height. The
        # keyboard executor owns the real arm/lift trajectory from live state.
        q[self.indices[LIFT]] = self.preset['lift_m']
        velocity = np.zeros(self.core.model.nv)
        self.pin.framesForwardKinematics(self.core.model, self.data, q)
        initial = {s: self.data.oMf[self.core.model.getFrameId(s+'_tcp')].copy() for s in ('left', 'right')}
        self.core.reset(q, initial, measured_velocity=velocity, state_timestamp=0.)
        stamp, solutions = 0., []
        for stage_index, stage in enumerate(self.preset['stages']):
            targets, elbows = {}, {}
            for side in ('left', 'right'):
                # Policy pad-center -> upstream fingertip, exactly as UMIPinkIK.
                targets[side] = self.transform(stage[side]['tcp']) * self.pin.SE3(np.eye(3), np.array([0., 0., .017]))
                elbows[side] = self.transform(stage[side]['elbow'])
                self.elbow_tasks[side].set_target(elbows[side])
            # This is virtual offline integration, not a hardware timeout.
            # Large posture changes can need more than 30 virtual seconds at
            # the retained 10 deg/s joint limit. Keep planning bounded.
            for iteration in range(3000):
                if cancelled():
                    raise MotionCancelled('Keyboard released during IK planning')
                stamp += self.core.config.control_dt
                result = self.core.step(q, targets, measured_velocity=velocity,
                    state_timestamp=stamp, control_timestamp=stamp)
                if not result.executable or result.diagnostics.fault:
                    raise RuntimeError(f'Keyboard UMI IK stopped: {result.diagnostics.fault or result.diagnostics.status}')
                q, velocity = result.positions.copy(), result.velocity.copy()
                self.pin.framesForwardKinematics(self.core.model, self.data, q)
                errors, residuals = [], {}
                for side in ('left', 'right'):
                    residuals[side] = {}
                    for name, frame, target in [('tcp', self.core.model.getFrameId(side+'_tcp'), targets[side]),
                                                ('elbow', self.elbows[side], elbows[side])]:
                        actual = self.data.oMf[frame]
                        position_error = float(np.linalg.norm(actual.translation-target.translation))
                        orientation_error = float(np.linalg.norm(self.pin.log3(actual.rotation.T @ target.rotation)))
                        residuals[side][name] = dict(position_error_m=position_error,
                                                     orientation_error_rad=orientation_error)
                        errors.extend((position_error / .001, orientation_error / .005))
                if max(errors) < 1 and np.max(abs(velocity)) < .002:
                    break
            else:
                self.planning_diagnostics.append(dict(seed=seed_name, stage=stage_index+1, iterations=iteration+1,
                    converged=False, errors=residuals, max_velocity_rad_s=float(np.max(abs(velocity)))))
                detail = '; '.join(f'{side} {name}: {error["position_error_m"]*1000:.2f} mm, '
                                  f'{np.degrees(error["orientation_error_rad"]):.2f} deg'
                                  for side, frames in residuals.items() for name, error in frames.items())
                raise KeyboardIKNonconvergence(f'Cartesian keyboard pose did not converge with {seed_name} seed '
                                   f'at stage {stage_index+1} '
                                   f'after {iteration+1} iterations ({detail}); no arm command sent')
            self.planning_diagnostics.append(dict(seed=seed_name, stage=stage_index+1, iterations=iteration+1,
                converged=True, errors=residuals, max_velocity_rad_s=float(np.max(abs(velocity)))))
            solutions.append({n: float(q[i]) for n, i in self.indices.items()})
        return solutions
