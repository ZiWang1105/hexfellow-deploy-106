"""Skill sequencing above the shared PoseExecutor, outside robot IK/actuation.

The deployment owner calls tick only after the preceding frame was accepted.
Each stage consequently consumes measured feedback and preserves the underlying
IK reference history. A failed stage never opens, retries, or advances the plan.
"""
from copy import deepcopy
import uuid
import numpy as np

from scipy.spatial.transform import Rotation

from ..control.executor import PoseExecutor
from ..interfaces.contract import Action, validate_sensor_packet
from .hand_v15 import compile_skill, validate_skill
from .workflows import ACQUIRE, CARRY, CHECKPOINTS
from .geometry import check_pad_floor


class SkillExecutor(PoseExecutor):
    stage_timeout_s = 30.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skill = None
        self._terminal = None
        self._checkpoint = None
        self._verifications = []

    def submit(self, action, measured):
        if self.startup is not None and not self.startup.complete:
            raise ValueError('Complete carrying startup before submitting a skill or policy action')
        if action.skill is None:
            super().submit(action, measured)
            self._skill = self._terminal = None
            self._checkpoint = None
            return
        validate_sensor_packet(measured)
        if action.navigation is not None:
            raise ValueError('Skills cannot span a base pulse; navigate and reobserve first')
        call = validate_skill(action.skill)
        if call.get('resume') is not None:
            self._resume(call, measured)
            return
        steps = compile_skill(call)
        # Check planned pad floors before sending even an opening command.
        last_waypoint = None
        for step in steps:
            if step.waypoint is not None:
                last_waypoint = step.waypoint
            if step.pad_floor_m is not None:
                check_pad_floor(last_waypoint, step.arms or call['arms'], step.pad_floor_m,
                                self.kinematics.gripper_geometry)
        # Do not silently open an existing grasp at the start of another pick.
        if call['name'] in ACQUIRE and any(self.gripper[s] == 0 for s in call['arms']):
            raise ValueError('Pick would open a commanded-closed hand; inspect and explicitly open first')
        if call['name'] in CARRY and any(self.gripper[s] != 0 for s in call['arms']):
            raise ValueError('Place requires an existing close command; retention must be visually checked')
        if call['name'] in ('transfer_between_arms', 'hold_and_push') and self.gripper[call['arms'][0]] != 0:
            raise ValueError('The giver/holder must have an existing close command')
        if call['name'] == 'transfer_between_arms' and self.gripper[call['arms'][1]] == 0:
            raise ValueError('Receiver is commanded closed; inspect and explicitly open before transfer')
        super().hold(measured)
        self._checkpoint, self._verifications = None, []
        self._skill, self._steps = call, steps
        self._index, self._entered, self._terminal = 0, False, None
        self._reference_poses = None

    def hold(self, measured):
        super().hold(measured)
        self._skill = self._terminal = None
        self._checkpoint = None

    def observation_hold(self, measured):
        """Internal reobservation preserves only a live visual checkpoint.

        An ordinary operator/policy HOLD cancels the plan. The deployment agent
        uses this operation solely after awaiting_verification, never after a
        horizon timeout or failed stage.
        """
        if self._terminal and self._terminal.get('status') == 'awaiting_verification':
            super().hold(measured)
        else:
            self.hold(measured)

    def _resume(self, call, measured):
        response = call['resume']
        checkpoint = getattr(self, '_checkpoint', None)
        if (not checkpoint or not self._terminal or self._terminal.get('status') != 'awaiting_verification'
                or response['token'] != checkpoint['token'] or call['name'] != self._skill['name']
                or call['arms'] != self._skill['arms']):
            raise ValueError('No matching live skill checkpoint; cancelled or consumed tokens cannot resume')
        if measured['timestamp_s'] <= checkpoint['observation_after_s']:
            raise ValueError('Checkpoint requires a newer sensor observation')
        if self.gripper != checkpoint['grippers']:
            raise ValueError('Gripper command changed since checkpoint; replan')
        assessment = dict(checkpoint=checkpoint['checkpoint'], verdict=response['verdict'],
                          evidence=response['evidence'], observed_s=measured['timestamp_s'])
        self._verifications.append(assessment)
        self._checkpoint = None  # consume once, including uncertain/failed assessments
        if response['verdict'] != 'confirmed':
            self._finish(measured, 'aborted', 'Visual assessment did not confirm checkpoint; remaining steps cancelled')
            return
        super().hold(measured)
        self._terminal, self._entered = None, False
        self._index += 1
        if self._index == len(self._steps):
            self._finish(measured, 'completed_unverified', 'Plan ended after model visual assessment; physical effect is not certified')

    def _clock(self, measured):
        return self.ik.control_clock() if self.ik.control_clock else measured['timestamp_s']

    def _begin(self, measured, now):
        step = self._steps[self._index]
        self._stage_started, self._dwell_started = now, None
        self._entered = True
        if step.kind != 'move':
            super().hold(measured)
            return
        # Anchor an unselected arm to the reference FK, not a measured TCP that
        # may differ under load. It remains a paired IK task, not a joint lock.
        # Capture again after every stage, including a giver retreat.
        result = self.ik.last_result
        joints = measured['joint_positions'] if result is None else {
            n: float(result.positions[i]) for n, i in self.ik.indices.items()}
        self._reference_poses = {s: p * self.ik.policy_T_umi[s].inverse()
                                 for s, p in self.ik.poses(joints).items()}
        targets = deepcopy(step.waypoint)
        for s in ('left', 'right'):
            if s not in (step.arms or self._skill['arms']):
                pose = self._reference_poses[s]
                targets[s] = dict(position_m=pose.translation.tolist(),
                                  quaternion_xyzw=Rotation.from_matrix(pose.rotation).as_quat().tolist())
            targets[s]['gripper_open'] = 1  # replaced before any command is emitted
        if self._skill['name'] == 'pull_toward_base':
            for side in self._skill['arms']:
                if np.linalg.norm(targets[side]['position_m'][:2]) >= np.linalg.norm(self._reference_poses[side].translation[:2]):
                    raise ValueError('Pull target must be horizontally nearer the base than the current reference')
        previous = dict(self.gripper)
        super().submit(Action.from_dict(targets), measured)
        self.gripper = previous

    def _finish(self, measured, status, reason):
        super().hold(measured)
        self._terminal = dict(status=status, reason=reason)
        if status != 'awaiting_verification':
            self._checkpoint = None

    def tick(self, measured):
        if self._skill is None:
            return super().tick(measured)
        now = self._clock(measured)
        if self._terminal is None and not self._entered:
            try:
                self._begin(measured, now)
            except ValueError as error:
                self._finish(measured, 'aborted', str(error))
        command, diagnostic = super().tick(measured)
        if self._terminal is None:
            step = self._steps[self._index]
            complete = False
            if diagnostic.get('tracking_stalled') or diagnostic.get('needs_observation'):
                self._finish(measured, 'blocked_unverified' if step.contact else 'aborted',
                             'Motion stalled; obstruction is not verified object contact or task success')
            elif now-self._stage_started > self.stage_timeout_s:
                self._finish(measured, 'aborted', 'Stage deadline exceeded; remaining steps cancelled')
            elif step.kind == 'move':
                complete = diagnostic.get('arrived') is True
            elif diagnostic.get('ik_status') == 'HOLD' and diagnostic.get('base_stopped', True):
                if step.kind == 'verify':
                    self._checkpoint = dict(token=uuid.uuid4().hex, checkpoint=step.checkpoint,
                        observation_after_s=measured['timestamp_s'], grippers=dict(self.gripper))
                    self._finish(measured, 'awaiting_verification', CHECKPOINTS[step.checkpoint])
                elif self._dwell_started is None:
                    if step.kind == 'gripper':
                        if step.pad_floor_m is not None:
                            self.kinematics.update(measured['joint_positions'])
                            try:
                                check_pad_floor(self.kinematics.packet()['tcp_poses'],
                                    step.arms or self._skill['arms'], step.pad_floor_m, self.kinematics.gripper_geometry)
                            except ValueError as error:
                                self._finish(measured, 'aborted', str(error))
                        for side in (() if self._terminal else (step.arms or self._skill['arms'])):
                            self.gripper[side] = step.gripper
                        command['gripper_open'] = dict(self.gripper)
                    self._dwell_started = now
                complete = self._terminal is None and self._dwell_started is not None and now-self._dwell_started >= step.duration_s
            if complete:
                self._index += 1
                self._entered = False
                if self._index == len(self._steps):
                    self._finish(measured, 'completed_unverified', 'Sequence ended; inspect fresh images to verify the effect')
        diagnostic['skill'] = dict(name=self._skill['name'], arms=self._skill['arms'],
                                   visual_assessments=len(getattr(self, '_verifications', [])), stage=min(self._index+1, len(self._steps)),
                                   stages=len(self._steps), status='running', effect_verified=False)
        # Internal stage arrival must not terminate the outer policy action.
        diagnostic.update(arrived=False, needs_observation=False)
        if self._terminal is not None:
            diagnostic['skill'].update(self._terminal)
            if self._checkpoint is not None:
                diagnostic['skill'].update({k:self._checkpoint[k] for k in ('token','checkpoint','observation_after_s')})
            if diagnostic.get('ik_status') == 'HOLD' and diagnostic.get('base_stopped', True):
                diagnostic.update(arrived=self._terminal['status'] == 'completed_unverified',
                                  needs_observation=True)
        return command, diagnostic
