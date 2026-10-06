"""Public joint observations and direct TCP tasks through the real UMI controller.

No nominal position, velocity or legacy effort is sent directly to a motor.
Direct Cartesian mode consumes TCP poses/body twists without a joint path.
The compatibility entry can interpret nominal joints as TCP tasks. Posture mode
uses UMI's native constrained joint objective, including explicit lift motion.
UMI replaces the Cartesian objective in posture mode; these modes are explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .portable_controller import UmiController, _transform, _upstream, _vector

JOINT_NAMES = (
    "lift_joint",
    *(f"{side}_joint_{i}" for side in ("left", "right") for i in range(1, 8)),
)
ARM_NAMES = JOINT_NAMES[1:]


@dataclass(frozen=True)
class UmiExecutionResult:
    q: dict[str, float]
    dq: dict[str, float]
    tau: dict[str, float]
    native_command: object
    native_packet: object
    diagnostics: object


class UmiExecutionFault(RuntimeError):
    """A failed upstream computation/guard; no executable transport output."""

    def __init__(self, message: str, *, command=None):
        super().__init__(message)
        self.command = command
        self.diagnostics = None if command is None else command.diagnostics


class UmiExecution:
    """One measured-feedback Pink solve and publisher guard per 20 ms tick.

    ``reset`` is an explicit lifecycle boundary, also required after a separate
    chassis segment changes the base frame. Mode changes retain controller and
    guard history. No automatic retry, reference rebasing or fault clearing is
    performed. Times are seconds in a single monotonic observation clock.

    Base pose/velocity default to zero for a stationary, base-relative frame.
    Callers with odometry can supply world pose and world velocity consistently.
    A real transport must supply its commissioned guard via ``guard_factory``;
    the default uses the same stock UMI guard with a simulation model binding.
    """

    def __init__(self, controller: UmiController, *, guard_factory=None):
        self.controller = controller
        self.control_dt = controller.control_dt
        self.guard_factory = guard_factory or controller.simulation_guard
        self.guard = None
        self.fault: UmiExecutionFault | None = None
        self.sequence = 0
        self._nominal_q = None
        self._kinematic_data = controller.core.model.createData()
        self._warmup_available = True
        self._last_mode = None

    def warmup(self, cycles: int = 4) -> dict:
        """Warm native kernels and the stock guard before acquiring any plant.

        Uses explicitly synthetic home observations with an offline clock. Both
        native objective paths run, but no transport output leaves this method.
        A subsequent reset from real public feedback is mandatory. This cannot
        be called after an external reset or an already active execution session.
        """
        if not self._warmup_available or self.guard is not None:
            raise RuntimeError("UMI warmup is allowed only before the first measured reset")
        if type(cycles) is not int or cycles < 2:
            raise ValueError("UMI warmup requires at least two native cycles")
        start = perf_counter()
        home = self.controller.core.home_q.copy()
        q = dict(zip(JOINT_NAMES, map(float, home[3:]), strict=True))
        dq = dict.fromkeys(JOINT_NAMES, 0.)
        timings = []
        try:
            self.reset(q, dq, 0., base_pose=home[:3])
            for index in range(cycles):
                mode = "posture" if index % 2 == 0 else "cartesian"
                before = perf_counter()
                self.step(q, dq, q, dq, index * self.control_dt,
                          mode=mode, base_pose=home[:3])
                timings.append(dict(mode=mode, wall_ms=1000 * (perf_counter() - before)))
            return dict(source="offline_native_home_warmup", cycles=cycles,
                        total_wall_ms=1000 * (perf_counter() - start), ticks=timings)
        finally:
            # Warm data/code caches survive; no synthetic reference, publisher
            # history or fault state may serve as the next real session's reset.
            self.guard = None
            self._nominal_q = None
            self.sequence = 0
            self.fault = None
            self._warmup_available = False
            self._last_mode = None

    def _joints(self, values, *, baseline=None, label="joint state"):
        if not isinstance(values, Mapping):
            raise ValueError(f"{label} must be a joint-name mapping")
        result = (np.full(15, np.nan) if baseline is None
                  else _vector(baseline, 15, label))
        for i, (short, native) in enumerate(
            zip(JOINT_NAMES, self.controller.coordinate_names[3:], strict=True)
        ):
            available = [values[name] for name in (short, native) if name in values]
            if not available:
                if baseline is None:
                    raise ValueError(f"{label} is missing {short}")
                continue
            if any(isinstance(value, (bool, np.bool_)) for value in available):
                raise ValueError(f"{label} has a boolean coordinate: {short}")
            if len(available) == 2 and available[0] != available[1]:
                raise ValueError(f"{label} aliases disagree for {short}")
            result[i] = float(available[0])
        return _vector(result, 15, label)

    def coordinates(self, q, dq, *, base_pose=(0., 0., 0.), base_velocity=(0., 0., 0.)):
        """Map the 15 measured short or native joint names to upstream 18-vectors."""
        return (
            np.r_[_vector(base_pose, 3, "base pose"), self._joints(q, label="measured q")],
            np.r_[_vector(base_velocity, 3, "base velocity"), self._joints(dq, label="measured dq")],
        )

    def reset(
        self, measured_q, measured_dq, timestamp: float, *,
        base_pose=(0., 0., 0.), base_velocity=(0., 0., 0.),
        stationary_reference=False,
    ) -> None:
        """Initialize core and guard with the same upstream acquisition policy.

        Stationary acquisition keeps actual measured dq for every physical
        check; only new command history uses the source's stationary policy.
        It is never applied by step or by a posture/Cartesian mode change.
        """
        if type(stationary_reference) is not bool:
            raise ValueError("stationary_reference must be an explicit boolean")
        self._warmup_available = False
        q, v = self.coordinates(measured_q, measured_dq,
                                base_pose=base_pose, base_velocity=base_velocity)
        state = self.controller.state(q, v, timestamp)
        state.validate(timestamp)
        self.controller.reset(q, v, timestamp, stationary_reference=stationary_reference)
        # Keep the historical custom guard-factory signature when the caller
        # uses the source default. A stationary custom gateway must opt in to
        # the same policy explicitly; silently mixing the histories is unsafe.
        self.guard = (self.guard_factory(state, stationary_reference=True)
                      if stationary_reference else self.guard_factory(state))
        self._nominal_q = q.copy()
        self.sequence = 0
        self.fault = None
        self._last_mode = None

    def _targets(self, q, dq):
        import pinocchio as pin

        model, data = self.controller.core.model, self._kinematic_data
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        targets, twists = {}, {}
        for side in ("left", "right"):
            frame = model.getFrameId(f"{side}_tcp")
            targets[side] = data.oMf[frame].homogeneous.copy()
            jacobian = pin.getFrameJacobian(model, data, frame, pin.ReferenceFrame.LOCAL)
            twists[side] = jacobian @ dq
        return targets, twists

    def step(
        self, measured_q, measured_dq, desired_q, desired_dq, timestamp: float, *,
        mode: str = "cartesian", control_phase: str = "tracking",
        control_timestamp: float | None = None,
        dispatch_timestamp: float | Callable[[], float] | None = None,
        base_pose=(0., 0., 0.), base_velocity=(0., 0., 0.), feedback_received=None,
        cartesian_targets=None, cartesian_twists=None, cartesian_accelerations=None,
    ) -> UmiExecutionResult:
        """Return only native, stock-guard-validated q/dq/arm-feedforward outputs.

        Call once per ``control_dt``; the caller selects the 50 Hz observations.
        ``control_timestamp`` is the actual generation time and defaults to the
        observation time for synchronous simulation. ``dispatch_timestamp`` lets
        a transport reject compute/queue delays with UMI's 20 ms packet lease.
        A callable is evaluated after the native solve, immediately before the
        guard; it can read the current transport clock without double-validation.
        The actual transport must also validate expiry when applying the packet.
        Desired joint dictionaries may be partial; unmentioned positions hold
        the last nominal request and unmentioned velocities are zero.
        """
        if self.fault is not None:
            raise self.fault
        if self.guard is None:
            raise RuntimeError("reset measured feedback before the first UMI execution step")
        command = None
        try:
            if mode not in {"cartesian", "posture"}:
                raise ValueError("UMI execution mode must be cartesian or posture")
            if control_phase != "tracking" and mode != "posture":
                raise ValueError("UMI preparation/lift-retreat requires explicit posture mode")
            q, v = self.coordinates(measured_q, measured_dq,
                                    base_pose=base_pose, base_velocity=base_velocity)
            nominal_q = np.r_[q[:3], self._joints(
                {} if desired_q is None else desired_q,
                baseline=self._nominal_q[3:], label="nominal q",
            )]
            nominal_v = np.r_[np.zeros(3), self._joints(
                {} if desired_dq is None else desired_dq,
                baseline=np.zeros(15), label="nominal dq",
            )]
            # These are source waypoints, not a second servo. A malformed or
            # impossible nominal configuration must not be hidden by FK.
            low, high = (self.controller.core.model.lowerPositionLimit,
                         self.controller.core.model.upperPositionLimit)
            if np.any(nominal_q < low) or np.any(nominal_q > high):
                raise ValueError("nominal q is outside the upstream planning margins")
            direct = cartesian_targets is not None
            if direct:
                if mode != 'cartesian':
                    raise ValueError('Direct TCP targets require Cartesian mode')
                targets = self._cartesian_mapping(cartesian_targets, 'targets', pose=True)
                twists = self._cartesian_mapping(cartesian_twists, 'twists')
                accelerations = (None if cartesian_accelerations is None else
                                 self._cartesian_mapping(cartesian_accelerations, 'accelerations'))
            else:
                if cartesian_twists is not None or cartesian_accelerations is not None:
                    raise ValueError('Direct twists require direct TCP targets')
                if mode == 'posture':
                    # TCP arguments are required by the source interface, but
                    # its posture objective replaces the Cartesian task. Never
                    # convert posture targets through nominal-joint FK.
                    targets, twists = self.controller.tcp_poses(q), None
                else:
                    targets, twists = self._targets(nominal_q, nominal_v)
                accelerations = None
            now = timestamp if control_timestamp is None else control_timestamp
            state = self.controller.state(q, v, timestamp, received=feedback_received)
            state.validate(now)
            entering_cartesian = mode == 'cartesian' and self._last_mode == 'posture'
            explicit_posture = direct and desired_q is not None
            changed_posture = explicit_posture and (
                self._last_mode is None or not np.array_equal(nominal_q[3:], self._nominal_q[3:]))
            if entering_cartesian or changed_posture:
                # The preceding checked posture is the task's redundancy goal.
                # Supply it through UMI's public task-input API; the profile's
                # human-blend algorithm, costs and every hard limit stay intact.
                # Otherwise a small Cartesian correction can blend toward the
                # generic human pose and leave the checked deployment family.
                posture = nominal_q if explicit_posture else self._nominal_q
                pose = _upstream("control.posture").NullPose(
                    float(posture[3]), tuple(posture[4:11]), tuple(posture[11:18]),
                )
                self.controller.core.set_nominal_pose(pose)
            if entering_cartesian or changed_posture:
                # This is the same transition used by upstream SessionController
                # after preparation. It updates only stationary soft-task/TCP
                # anchors, keeping MIT integration and publisher history intact.
                # Every changed direct checked posture is a stationary task
                # entry: set_nominal_pose does not rebuild the cached blend.
                # Repeated samples of the same task never reanchor.
                # Its native stationary-reference and measured-speed gate stays
                # authoritative: a moving transition fails instead of rebasing.
                self.controller.core.reanchor_tracking_tasks(q, v)
            sample = _upstream("control.target_timing").TargetSample(now, self.sequence)
            command = self.controller.step(
                q, v, targets, state_timestamp=state.timestamp, control_timestamp=now,
                feedback_received=state.received, target_sample=sample,
                target_twists=twists if mode == "cartesian" else None,
                target_accelerations=accelerations,
                posture_target=nominal_q if mode == "posture" else None,
                posture_velocity=nominal_v if mode == "posture" else None,
                control_phase=control_phase,
            )
            packet = self.controller.packet(
                command, state, self.sequence, created=now, control_phase=control_phase,
            )
            dispatch = (now if dispatch_timestamp is None else
                        dispatch_timestamp() if callable(dispatch_timestamp) else dispatch_timestamp)
            self.guard.validate(packet, state, dispatch)
            result = UmiExecutionResult(
                q=dict(zip(JOINT_NAMES, map(float, packet.positions[3:]), strict=True)),
                dq=dict(zip(JOINT_NAMES, map(float, packet.velocity[3:]), strict=True)),
                tau=dict(zip(ARM_NAMES, map(float, packet.mit.torque_ff), strict=True)),
                native_command=command, native_packet=packet, diagnostics=command.diagnostics,
            )
            self._nominal_q = nominal_q.copy()
            self._last_mode = mode
            self.sequence += 1
            return result
        except Exception as error:
            self.fault = UmiExecutionFault(str(error), command=command)
            raise self.fault from error

    @staticmethod
    def _cartesian_mapping(values, label, *, pose=False):
        if not isinstance(values, Mapping) or set(values) != {'left', 'right'}:
            raise ValueError(f'Direct Cartesian {label} require exactly left and right')
        return {side: (_transform(getattr(values[side], 'homogeneous', values[side]))
                       .homogeneous.copy() if pose else
                       _vector(values[side], 6, f'{side} TCP {label}'))
                for side in ('left', 'right')}

    def step_cartesian(
        self, measured_q, measured_dq, targets, timestamp: float, *, target_twists,
        target_accelerations=None, nominal_posture=None, control_timestamp=None,
        dispatch_timestamp=None, base_pose=(0., 0., 0.), base_velocity=(0., 0., 0.),
        feedback_received=None,
    ) -> UmiExecutionResult:
        """Send direct native-world TCP targets and LOCAL TCP [linear, angular] twists.

        No nominal joint path, FK target conversion or application compensator is
        used. Optional posture intent updates UMI's public arm NullPose task;
        source Cartesian lift anchoring remains authoritative. Explicit lift
        positioning uses posture mode, which replaces the Cartesian objective.
        Every native feedback, transition, packet and publisher guard remains
        the same as ``step`` and faults remain latched.
        """
        return self.step(
            measured_q, measured_dq, nominal_posture, None, timestamp,
            mode='cartesian', control_timestamp=control_timestamp,
            dispatch_timestamp=dispatch_timestamp, base_pose=base_pose,
            base_velocity=base_velocity, feedback_received=feedback_received,
            cartesian_targets={} if targets is None else targets, cartesian_twists=target_twists,
            cartesian_accelerations=target_accelerations,
        )

    def step_posture(
        self, measured_q, measured_dq, desired_q, desired_dq, timestamp: float, *,
        control_phase='tracking', control_timestamp=None, dispatch_timestamp=None,
        base_pose=(0., 0., 0.), base_velocity=(0., 0., 0.), feedback_received=None,
    ) -> UmiExecutionResult:
        """Run native posture task inputs, including explicit lift positioning.

        Positions/velocities are goals for the source solver and governor, never
        motor references. No FK of desired joints or legacy trajectory compiler
        is used. Mode changes retain the live command and guard histories.
        """
        return self.step(
            measured_q, measured_dq, desired_q, desired_dq, timestamp,
            mode='posture', control_phase=control_phase, control_timestamp=control_timestamp,
            dispatch_timestamp=dispatch_timestamp, base_pose=base_pose,
            base_velocity=base_velocity, feedback_received=feedback_received,
        )
