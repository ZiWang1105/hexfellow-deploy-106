"""Task-independent Cartesian carrying initialization through the live UMI core."""
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from ..hardware.keyboard_ik import load_ik_pose
from ..robot.joints import LEFT, RIGHT, LIFT
from ..robot.upstream import PROJECT_ROOT

DEFAULT_STARTUP_POSE = PROJECT_ROOT / 'configs/hardware/keyboard_carry_ik.yaml'


class CarryStartup:
    """Temporary elbow constraints; preserve MIT reference and braking history."""
    def __init__(self, ik, preset, measured):
        from pink.tasks import FrameTask
        self.ik = ik
        self.preset = copy.deepcopy(preset)
        # Preserve the carrying arm geometry at the session's fixed lift height.
        # Shift both TCP and elbow frames; never alter policy-selected targets.
        lift = float(measured['joint_positions'][LIFT])
        lift_delta = lift - self.preset['lift_m']
        self.preset['lift_m'] = lift
        for stage in self.preset['stages']:
            for poses in stage.values():
                for pose in poses.values():
                    pose['position_m'][2] += lift_delta
        self.started = measured['timestamp_s']
        self.stage = 0
        self.stable_since = None
        self.transitioning = False
        self.finishing = False
        self.complete = False
        self.tasks = {}
        self.frames = {}
        for side, names in [('left', LEFT), ('right', RIGHT)]:
            self.frames[side] = ik.core.model.getFrameId(names[3])
            self.tasks[side] = FrameTask(names[3], position_cost=20., orientation_cost=5.,
                                         gain=.5, lm_damping=1e-4)
        # An already-carried robot must not first lower its forearms again.
        errors = self.errors(measured, self.preset['stages'][-1])
        if self.arrived(errors):
            self.stage = len(preset['stages']) - 1
        ik.core.tasks.extend(self.tasks.values())
        self.set_targets()

    def set_targets(self):
        stage = self.preset['stages'][self.stage]
        targets = {}
        for side, poses in stage.items():
            tcp, elbow = poses['tcp'], poses['elbow']
            targets[side] = (np.array(tcp['position_m']),
                            Rotation.from_quat(tcp['quaternion_xyzw']).as_matrix())
            self.tasks[side].set_target(self.ik.pin.SE3(
                Rotation.from_quat(elbow['quaternion_xyzw']).as_matrix(),
                np.array(elbow['position_m'])))
        self.ik.track(targets)

    def errors(self, measured, stage):
        ik = self.ik
        actual_tcp = ik.poses(measured['joint_positions'])
        errors = {'lift_m': abs(measured['joint_positions'][LIFT] - self.preset['lift_m'])}
        for side in self.frames:
            actual = {'tcp': actual_tcp[side] * ik.policy_T_umi[side].inverse(),
                      'elbow': ik.data.oMf[self.frames[side]].copy()}
            for frame, pose in actual.items():
                desired = stage[side][frame]
                errors[f'{side}_{frame}_position_m'] = float(np.linalg.norm(
                    np.array(desired['position_m']) - pose.translation))
                errors[f'{side}_{frame}_orientation_rad'] = float(np.linalg.norm(
                    Rotation.from_matrix(Rotation.from_quat(desired['quaternion_xyzw']).as_matrix()
                                         @ pose.rotation.T).as_rotvec()))
        return errors

    @staticmethod
    def arrived(errors):
        return all(value < (.015 if name == 'lift_m' else
                            .025 if name.endswith('_m') else np.deg2rad(3.))
                   for name, value in errors.items())

    def close(self):
        for task in self.tasks.values():
            if task in self.ik.core.tasks:
                self.ik.core.tasks.remove(task)
        self.ik.hold()

    def update(self, measured):
        errors = self.errors(measured, self.preset['stages'][self.stage])
        now = measured['timestamp_s']
        if not self.complete and now - self.started > 120.:
            self.close()
            raise RuntimeError(f'Carrying startup did not settle: {errors}')
        if self.finishing:
            self.complete = self.ik.core.control_state == 'HOLD'
            if self.complete and not self.arrived(errors):
                raise RuntimeError(f'Carrying posture changed during final braking: {errors}')
        elif self.transitioning:
            if self.ik.core.control_state == 'HOLD':
                self.stage += 1
                self.transitioning = False
                self.set_targets()
        elif self.arrived(errors) and max(abs(v) for v in measured['joint_velocities'].values()) < .03:
            if self.stable_since is None:
                self.stable_since = now
            if now - self.stable_since >= .3:
                self.stable_since = None
                if self.stage + 1 < len(self.preset['stages']):
                    # End the prior Cartesian trajectory before changing the
                    # elbow bending direction; preserve joint braking history.
                    self.ik.hold()
                    self.transitioning = True
                else:
                    self.close()
                    self.finishing = True
        else:
            self.stable_since = None
        return dict(arrived=self.complete, needs_observation=False,
                    mode='startup_complete' if self.complete else 'startup_carry',
                    startup_stage=self.stage + 1, startup_stages=len(self.preset['stages']),
                    startup_elapsed_s=now-self.started, startup_errors=errors)


def prepare_carry(robot, tick=None, *, stop=None, progress=None):
    """Wait outside the control owner; startup never requests policy inference."""
    preset = load_ik_pose(DEFAULT_STARTUP_POSE)
    robot.start_carry(preset)
    tick = robot.tick if tick is None else tick
    last_stage = None
    while True:
        if stop is not None and stop.is_set():
            robot.hold()
            raise InterruptedError('Carrying startup cancelled')
        diagnostic = tick()
        if diagnostic.get('mode') not in ('startup_carry', 'startup_complete'):
            continue
        stage = diagnostic['startup_stage']
        if stage != last_stage:
            message = f'Carrying startup: stage {stage}/{diagnostic["startup_stages"]}'
            print(message, flush=True)
            if progress:
                progress(message)
            last_stage = stage
        if diagnostic['mode'] == 'startup_complete':
            print('Carrying startup complete; starting policy with both elbows bent.', flush=True)
            return diagnostic
