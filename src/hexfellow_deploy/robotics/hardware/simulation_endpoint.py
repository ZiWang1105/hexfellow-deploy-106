"""Network-free device substitute for the production GatewaySupervisor.

Physics advances independently of controller publications. This adapter executes
the production packet conversion, then models SDK scheduling and actuator delay.
It is not a firmware emulator: local MIT PD and brake behavior remain assumptions.
"""
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .hexfellow_endpoint import HexWholeBodyEndpoint
from .whole_body import SLICES


class SimulatedDevice:
    def __init__(self):
        self.value = None

    def motor_command(self, kind, value):
        self.value = value

    def set_vehicle_speed(self, x, y, yaw):
        self.value = np.array([x, y, yaw])


class SimulationEndpoint:
    """Same supervisor and packet builder as live; no SDK construction or sockets."""

    _prepare_dispatch = HexWholeBodyEndpoint._prepare_dispatch
    _publish_packets = HexWholeBodyEndpoint._publish_packets
    initialize_lift_retreat = HexWholeBodyEndpoint.initialize_lift_retreat

    def __init__(self, plant, mapping, profile, *, dispatch_delay_s=.006,
                 actuator_delay_s=0., feedback_delay_s=0., feedback_noise_rad=0.,
                 body_delay_s=0., velocity_feedback_tau_s=None):
        self.plant, self.mapping, self.profile = plant, mapping, profile
        self.dispatch_delay_s = dispatch_delay_s
        self.actuator_delay_s = actuator_delay_s
        self.body_delay_s = body_delay_s
        if velocity_feedback_tau_s is None:
            velocity_feedback_tau_s = mapping.data.get('simulation', {}).get('velocity_feedback_tau_s', 0.)
        if not np.isfinite(velocity_feedback_tau_s) or velocity_feedback_tau_s < 0:
            raise ValueError('velocity feedback time constant must be finite and nonnegative')
        self.velocity_feedback_tau_s = velocity_feedback_tau_s
        self.filtered_arm_velocity = {s: plant.state().v[SLICES[s]].copy() for s in ('left','right')}
        self.feedback_delay_s = feedback_delay_s
        self.feedback_noise_rad = feedback_noise_rad
        self.clock = lambda: float(plant.data.time)
        self.devices = {name: SimulatedDevice() for name in SLICES}
        self.direct_gates = {name: SimpleNamespace(delay_reason=None) for name in SLICES}
        self.command_type = SimpleNamespace(MIT='MIT', POSITION='POSITION')
        self.mit_type = SimpleNamespace
        self.write_count = 0
        self.dispatch = None
        self.queue = []
        self.active = None
        self.holding = False
        self.epoch = 0
        self.lease_deadline = float("inf")
        self.expired_packets = 0
        self.applied_packets = 0
        self.noise = np.random.default_rng(20260924)
        self.feedback = {name: deque(maxlen=1024) for name in SLICES}
        self.periods = dict(left=.004, right=.004, theta=.02, base=.02)
        self.next_feedback = dict.fromkeys(SLICES, self.clock())
        self.health = {side: dict(joint_limits=np.column_stack((
            profile.sdk_position_lower_rad[i:i+7],
            profile.sdk_position_upper_rad[i:i+7])).tolist())
            for side, i in (('left', 0), ('right', 7))}
        self.health['theta'] = dict(range=[0., .422], pos=.422-plant.state().q[3])
        self.health['base'] = {}
        from ..control.mit_command import MitCommand
        from ..control.robot import ARM_JOINTS
        q = plant.state().q.copy()
        gravity = plant.core.gravity_model.evaluate(q, np.array([]))[4:]
        mit = MitCommand(ARM_JOINTS, q[4:].copy(), np.zeros(14), gravity,
                         np.asarray(profile.mit_servo.kp), np.asarray(profile.mit_servo.kd), self.clock())
        self._apply(SimpleNamespace(executable=True, positions=q, velocity=np.zeros(18), mit=mit,
            base=SimpleNamespace(velocity_world=np.zeros(3), velocity_body=np.zeros(3))))
        self._sample_feedback()

    def _sample_feedback(self):
        now = self.clock()
        state = self.plant.state()
        for name, part in SLICES.items():
            if now + 1e-9 < self.next_feedback[name]:
                continue
            q, v = state.q[part].copy(), state.v[part].copy()
            if name in ('left', 'right'):
                entry = self.mapping.data['arms'][name]
                order, signs = entry['sdk_indices'], np.asarray(entry['signs'])
                raw_q, raw_v = np.empty(7), np.empty(7)
                raw_q[order] = (q-entry['offsets_rad'])*signs
                raw_v[order] = v*signs
                raw_q += self.noise.normal(0., self.feedback_noise_rad, 7)
                q, v = self.mapping.arm_state(name, raw_q, raw_v)
                if self.velocity_feedback_tau_s:
                    alpha = -np.expm1(-self.periods[name] / self.velocity_feedback_tau_s)
                    self.filtered_arm_velocity[name] += alpha * (v - self.filtered_arm_velocity[name])
                    v = self.filtered_arm_velocity[name].copy()
            self.feedback[name].append((now, q, v))
            self.next_feedback[name] += self.periods[name]

    def state(self):
        state = self.plant.state()
        q, v, received, sequences = state.q.copy(), state.v.copy(), {}, {}
        for name, part in SLICES.items():
            history = self.feedback[name]
            stamp, qp, vp = next((row for row in reversed(history)
                                 if row[0] <= self.clock()-self.feedback_delay_s+1e-9), history[0])
            q[part], v[part] = qp, vp
            received[name] = stamp
            sequences[name] = round(stamp/self.periods[name])
        health = {**self.health, 'theta': {**self.health['theta'], 'pos': .422-q[3]}}
        return replace(state, q=q, v=v, received=received, sequences=sequences, health=health,
                       feedback_policy=self.mapping.feedback_policy, latency_policy=self.mapping.latency_policy)

    def check_health(self, state, stage, *, check_timing=True):
        if check_timing:
            state.validate(self.clock())

    def acquire(self, stage, profile):
        state = self.state()
        if self.holding:
            # A new supervisor may acquire only after the restart hold settled.
            # This is a new command history, never an in-flight delay resume.
            from ..control.reference_initialization import stationary_reference_velocity
            stationary_reference_velocity(self.plant.state().v)
            stationary_reference_velocity(state.v)
            self.epoch = 0
            self.resume_delay(self.epoch)
        self.initialize_lift_retreat(state)
        return state

    def publish_validated(self, command, stage, state):
        state.validate(self.clock())
        self._publish_packets(command, stage, state)
        # MitWireGate renews its publisher lease at authorization, not at the
        # later first SDK send. The queued packet retains its own send expiry.
        from ..control.feedback_policy import received_deadline
        from .whole_body import COMMAND_WATCHDOG
        self.lease_deadline = min(command.created+COMMAND_WATCHDOG,
            received_deadline(command.state_received, self.mapping.feedback_policy))
        self.queue.append((self.clock()+self.dispatch_delay_s, command, dict(
            (name, device.value) for name, device in self.devices.items()), False))

    def _decode(self, command, values):
        q, v = command.positions.copy(), command.velocity.copy()
        torque, kp, kd = np.zeros((3, 14))
        for side, start in (('left', 0), ('right', 7)):
            wire = values[side]
            qp, vp = self.mapping.arm_state(side, [x.position for x in wire], [x.speed for x in wire])
            q[4+start:11+start], v[4+start:11+start] = qp, vp
            entry = self.mapping.data['arms'][side]
            torque[start:start+7] = np.array([x.torque for x in wire])[entry['sdk_indices']]*entry['signs']
            kp[start:start+7] = np.array([x.kp for x in wire])[entry['sdk_indices']]
            kd[start:start+7] = np.array([x.kd for x in wire])[entry['sdk_indices']]
        q[3] = self.mapping.lift_position(values['theta'], self.health['theta']['range'])
        sdk_base = values['base']
        body = np.r_[self.mapping.rotation() @ sdk_base[:2], sdk_base[2]]
        c, s = np.cos(command.measured_q[2]), np.sin(command.measured_q[2])
        world = np.r_[np.array([[c, -s], [s, c]]) @ body[:2], body[2]]
        v[:3] = world
        mit = replace(command.mit, position=q[4:], velocity=v[4:], torque_ff=torque, kp=kp, kd=kd)
        return SimpleNamespace(executable=True, positions=q, velocity=v, mit=mit,
                               base=SimpleNamespace(velocity_world=world, velocity_body=body))

    def _apply(self, command):
        self.active = command
        self.plant.core._gravity_effort[:] = self.plant.core.gravity_model.evaluate(self.plant.state().q, np.array([]))
        self.plant.core.apply_command(self.plant.data, command)
        self.plant.last_command = command

    def advance_to(self, deadline):
        while self.clock()+1e-9 < deadline:
            if not self.holding and self.clock() >= self.lease_deadline:
                self.delay_hold('simulated SDK source lease expired', self.epoch+1)
            due, pending = [], []
            for row in self.queue:
                (due if row[0] <= self.clock()+1e-9 else pending).append(row)
            self.queue = pending
            for _, command, values, transmitted in due:
                if self.holding:
                    break
                if not transmitted:
                    if self.clock() >= command.expires:
                        self.expired_packets += 1
                        self.delay_hold('simulated SDK rejected expired command', self.epoch+1)
                        continue
                    self.queue.append((self.clock()+self.actuator_delay_s, command, values, 'arms'))
                    self.queue.append((self.clock()+self.body_delay_s, command, values, 'body'))
                else:
                    decoded = self._decode(command, values)
                    if transmitted == 'arms':
                        decoded.positions[:4] = self.active.positions[:4]
                        decoded.velocity[:4] = self.active.velocity[:4]
                        decoded.base = self.active.base
                    else:
                        decoded.positions[4:] = self.active.positions[4:]
                        decoded.velocity[4:] = self.active.velocity[4:]
                        decoded.mit = self.active.mit
                    self._apply(decoded)
                    self.applied_packets += 1
            before_substep = getattr(self.plant, '_before_substep', None)
            if before_substep is not None:
                before_substep()
            self.plant.mujoco.mj_step(self.plant.model, self.plant.data)
            self.plant._observe_effort()
            observer = getattr(self.plant, 'physics_observer', None)
            if observer:
                observer(self.plant)
            self._sample_feedback()

    def delay_hold(self, reason, epoch):
        self.queue.clear()
        self.holding, self.epoch = True, epoch
        for gate in self.direct_gates.values():
            gate.delay_reason = reason
        self.maintain_delay_hold()

    def maintain_delay_hold(self):
        if self.active is None or getattr(self, '_brake_applied', False):
            return
        state = self.plant.state()
        gravity = self.plant.core.gravity_model.evaluate(state.q, np.array([]))[4:]
        command = SimpleNamespace(executable=True, positions=state.q.copy(), velocity=np.zeros(18),
            mit=replace(self.active.mit, position=state.q[4:].copy(), velocity=np.zeros(14), torque_ff=gravity),
            base=SimpleNamespace(velocity_world=np.zeros(3), velocity_body=np.zeros(3)))
        self._apply(command)
        self._brake_applied = True

    def delay_drained(self, epoch):
        return not self.queue and self.holding and epoch == self.epoch

    def resume_delay(self, epoch):
        self.lease_deadline = self.clock()+.04
        self.holding = self._brake_applied = False
        for gate in self.direct_gates.values():
            gate.delay_reason = None

    def stop(self):
        self.delay_hold('stop', self.epoch+1)
        return []

    def settle_for_restart(self, *, timeout_s=3., stable_s=.2):
        """Advance the stopped physics until measured speeds stay stationary.

        Never write qpos/qvel or loosen IK limits to make a restart feasible.
        The same stationarity check as live reference initialization applies.
        """
        from ..control.reference_initialization import stationary_reference_velocity

        self.stop()
        began, stable_since = self.clock(), None
        settled = False
        while self.clock() - began < timeout_s:
            self.advance_to(self.clock() + .02)
            try:
                stationary_reference_velocity(self.plant.state().v)
                stationary_reference_velocity(self.state().v)
            except ValueError:
                stable_since = None
                continue
            if stable_since is None:
                stable_since = self.clock()
            if self.clock() - stable_since >= stable_s - 1e-9:
                settled = True
                break
        return dict(settled=settled, elapsed_s=self.clock()-began,
                    measured_velocity=self.plant.state().v.tolist())

    def diagnostic_window(self):
        return {}

    def report(self):
        return dict(sdk_dispatch_delay_s=self.dispatch_delay_s, actuator_delay_s=self.actuator_delay_s,
                    velocity_feedback_tau_s=self.velocity_feedback_tau_s,
                    feedback_delay_s=self.feedback_delay_s, feedback_periods_s=self.periods,
                    expired_packets=self.expired_packets, applied_packets=self.applied_packets,
                    hardware_commands_sent=0, firmware_emulated=False,
                    brake_model='stationary local PD with nominal gravity; not firmware brake qualification')
