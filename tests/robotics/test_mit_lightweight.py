"""Lightweight MIT boundary semantics; no actuator stopping claims."""
from types import SimpleNamespace

import numpy as np
import pytest

import hexfellow_deploy.robotics.control.ik_core as core_module
from hexfellow_deploy.robotics.control.command_limits import DirectionalVelocityLimit
from hexfellow_deploy.robotics.control.ik_core import TCP_FRAMES, PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile


def make_core():
    core = PinkIKCore(config=controller_config_for_profile("x7-mit-highspeed-v10"))
    q = core.home_q.copy()
    core.configuration.update(q)
    targets = {side: core.configuration.get_transform_frame_to_world(frame).copy()
               for side, frame in TCP_FRAMES.items()}
    core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    return core, q, targets


@pytest.mark.parametrize("dt", [.01, .02, .04])
@pytest.mark.parametrize("speed", [-.2, 0., .2])
def test_damper_uses_signed_closing_speed_and_displacement_units(dt, speed):
    limit = DirectionalVelocityLimit([[1., 0.]], [.05], [speed, 4.], .06, labels=("pair",))
    matrix, bound = limit.compute_qp_inequalities(None, dt)
    available = max(0., .05 - max(0., -speed) * .06)
    np.testing.assert_allclose(bound, [available])
    # Tangential and away movement must remain possible.
    assert (matrix @ (np.array([.1, 2.]) * dt) <= bound).all()
    np.testing.assert_allclose(matrix @ (np.array([-.2, 0.]) * dt), [.2])


def test_empty_damper_and_exhausted_reaction_clearance():
    empty = DirectionalVelocityLimit(np.empty((0, 2)), [], [0., 0.], .06)
    assert empty.compute_qp_inequalities(None, .02) is None
    assert not empty.reaction_clearance_exhausted
    closing = DirectionalVelocityLimit([[1.]], [.01], [-1.], .06, labels=("joint",))
    assert closing.reaction_clearance_exhausted
    assert closing.bound[0] == 0.




def test_activation_travel_does_not_inflate_v3_hard_geometry():
    core, _q, _ = make_core()
    system = core.self_collision_system
    thresholds = system._pair_minimum_distances.copy()
    barriers = system.update(core.configuration, .02,
                             joint_safety_travel=np.ones(core.model.nv))
    assert barriers  # Force broad activation, including otherwise distant pairs.
    expected = {(pair.first, pair.second): pair.minimum_distance_m for pair in system._pairs}
    for barrier in barriers:
        for item in barrier.distances:
            assert item.pair.minimum_distance_m == expected[(item.pair.first, item.pair.second)]
    np.testing.assert_array_equal(system._pair_minimum_distances, thresholds)
    assert system.collision_envelope.activation_only


def test_normal_cycle_never_builds_a_future_backup(monkeypatch):
    import builtins

    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if any(part in name for part in ("mit_backup", "mit_prediction")):
            raise AssertionError("archived predictor entered the live controller")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    core, q, targets = make_core()
    result = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    assert result.executable, result.diagnostics.fault
    assert result.diagnostics.braking_retries == 0
    assert result.diagnostics.timings.gravity_ms > 0
    assert result.diagnostics.timings.output_checks_ms > 0


def test_v4_repeated_feedback_recovers_without_latching_hold():
    core = PinkIKCore(config=controller_config_for_profile("x7-mit-highspeed-v10"))
    q = core.home_q.copy()
    core.configuration.update(q)
    targets = {
        side: core.configuration.get_transform_frame_to_world(frame).copy()
        for side, frame in TCP_FRAMES.items()
    }
    velocity = np.zeros(core.model.nv)
    core.reset(q, targets, measured_velocity=velocity, state_timestamp=0.0)
    first = core.step(
        q, targets, measured_velocity=velocity, state_timestamp=0.0, control_timestamp=0.0
    )
    repeated = core.step(
        q, targets, measured_velocity=velocity, state_timestamp=0.0, control_timestamp=0.02
    )
    assert first.executable and repeated.executable
    assert repeated.diagnostics.recovery_reason == "repeated_feedback"
    assert core.control_state == "TRACKING"
    assert repeated.diagnostics.stop_reason is None


@pytest.mark.parametrize("both_fail", [False, True])
def test_retry_preserves_every_hard_row_and_is_bounded(monkeypatch, both_fail):
    core, q, targets = make_core()
    def warning(model, *args, **kwargs):
        return DirectionalVelocityLimit(np.zeros((1, model.nv)), [1.],
                                        np.zeros(model.nv), .06, labels=("test_warning",))
    monkeypatch.setattr(core_module, "joint_boundary_damper", warning)
    real_assemble = core_module.assemble_problem
    assembled = []
    def assemble(*args, **kwargs):
        result = real_assemble(*args, **kwargs)
        assembled.append(result)
        return result
    monkeypatch.setattr(core_module, "assemble_problem", assemble)
    real_solve = core_module.qpsolvers.solve_problem
    problems = []
    def solve(problem, **kwargs):
        problems.append(problem)
        if len(problems) == 1 or both_fail:
            return SimpleNamespace(found=False, x=None)
        return real_solve(problem, **kwargs)
    monkeypatch.setattr(core_module.qpsolvers, "solve_problem", solve)
    saved = core._last_safe_positions.copy()
    result = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    assert len(problems) == 2
    hard = assembled[0][1]
    assert len(problems[0].h) > len(hard.h)
    np.testing.assert_array_equal(problems[1].G, hard.G)
    np.testing.assert_array_equal(problems[1].h, hard.h)
    for field in ("A", "b", "lb", "ub"):
        np.testing.assert_array_equal(getattr(problems[1], field), getattr(hard, field))
    if both_fail:
        assert not result.executable
        assert core.control_state == "FAULT"
        np.testing.assert_array_equal(core._last_safe_positions, saved)
        assert core.last_qp_failure is not None
    else:
        assert result.executable, result.diagnostics.fault
        assert result.diagnostics.braking_retries == 1
        assert core.control_state == "TRACKING"  # Recovery is transient, not a latched stop.
        assert result.diagnostics.recovery_reason == "boundary_constraints_infeasible"


def test_packet_rejection_does_not_advance_reference_history(monkeypatch):
    core, q, targets = make_core()
    saved = [value.copy() for value in (core._last_safe_positions, core._last_velocity,
                                       core._last_acceleration, core._last_jerk)]
    def invalid_packet(*args, **kwargs):
        raise ValueError("test packet rejection")
    monkeypatch.setattr(core, "_mit_command", invalid_packet)
    targets["left"].translation[0] += .02
    command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    assert not command.executable
    for before, after in zip(saved, (core._last_safe_positions, core._last_velocity,
                                    core._last_acceleration, core._last_jerk), strict=True):
        np.testing.assert_array_equal(before, after)


def test_joint_warning_activation_is_continuous_at_the_speed_cap():
    from hexfellow_deploy.robotics.control.command_limits import joint_boundary_damper
    model = SimpleNamespace(nv=1, lowerPositionLimit=np.array([-3.]),
                            upperPositionLimit=np.array([3.]), velocityLimit=np.array([1.2]))
    limit = joint_boundary_damper(model, [0], np.array([1.9]), np.array([1.9]),
                                 np.array([.3]), np.array([.05]), .06)
    # h=1.1 is outside a stopping-only activation band but already below vmax/gain.
    assert "joint_upper:0" in limit.labels
    matrix, bound = limit.compute_qp_inequalities(None, .02)
    assert (matrix @ np.array([.3 * .02]) <= bound).all()


def test_collision_guard_limits_closing_without_forcing_an_escape():
    limit = DirectionalVelocityLimit([[1.]], [.001], [0.], .06,
                                     labels=("pair",), guard_distance=.002)
    assert limit.bound[0] == 0.
    assert not limit.reaction_clearance_exhausted
    matrix, bound = limit.compute_qp_inequalities(None, .02)
    assert (matrix @ np.array([.001]) <= bound).all()


def test_native_distance_workspace_matches_direct_fcl_queries():
    core, _, _ = make_core()
    system = core.self_collision_system
    travel = np.ones(core.model.nv)
    native = system.update(core.configuration, .02, joint_safety_travel=travel)
    native_distances = [item for barrier in native for item in barrier.distances]
    assert system._native_distance_indices
    system._native_distance_indices = {}
    direct = system.update(core.configuration, .02, joint_safety_travel=travel)
    direct_distances = [item for barrier in direct for item in barrier.distances]
    assert len(native_distances) == len(direct_distances)
    for left, right in zip(native_distances, direct_distances, strict=True):
        assert (left.pair.first, left.pair.second) == (right.pair.first, right.pair.second)
        np.testing.assert_allclose(left.distance_m, right.distance_m, atol=1e-9, rtol=0.)
        np.testing.assert_allclose(left.first_point, right.first_point, atol=1e-8, rtol=0.)
        np.testing.assert_allclose(left.second_point, right.second_point, atol=1e-8, rtol=0.)


@pytest.mark.parametrize("offset,separated", [(.18, True), (.12, False)])
def test_mesh_certificate_checks_actual_vertex_separation(monkeypatch, offset, separated):
    import mujoco

    from hexfellow_deploy.robotics.control.self_collision import MuJoCoSelfCollisionOracle

    vertices = " ".join(str(v) for x in (-.1, .1) for y in (-.1, .1)
                        for z in (-.1, .1) for v in (x, y, z))
    xml = f"""<mujoco><asset><mesh name="cube" vertex="{vertices}"/></asset>
    <worldbody><body name="a"><geom type="mesh" mesh="cube"/></body>
    <body name="b" pos="{offset} {offset} 0" quat=".923879532511 0 0 .382683432365">
    <geom type="mesh" mesh="cube"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    document = {
        "links": [{"name": name, "mujoco_bodies": [name]} for name in ("a", "b")],
        "pairs": [{"first": "a", "second": "b", "minimum_distance_m": .01}],
    }
    oracle = MuJoCoSelfCollisionOracle(model, document)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: -.03)
    result = oracle.sample(data)[0]
    if separated:
        expected = offset * np.sqrt(2) - .1 * np.sqrt(2) - .1
        assert result.distance_m == pytest.approx(expected, abs=1e-7)
        assert not result.exact
    else:
        assert result.distance_m == -.03
        assert oracle._mesh_separation_certificate(data, 0, 1) is None

@pytest.mark.parametrize("delay", [.001, .01, .02, .04, .08])
def test_closed_form_inverse_matches_the_existing_reference_envelope(delay):
    from hexfellow_deploy.robotics.control.command_limits import (
        delayed_stoppable_speed,
        stoppable_speed,
        stopping_distance,
    )

    rng = np.random.default_rng(0)
    distance = rng.uniform(0., 3., 128)
    vmax = rng.uniform(.1, 9.4, 128)
    amax = rng.uniform(1., 80., 128)
    jerk = rng.uniform(50., 2000., 128)
    acceleration = rng.uniform(-1., 1., 128) * amax
    actual = delayed_stoppable_speed(distance, vmax, amax, jerk, delay)
    old = stoppable_speed(distance, acceleration, vmax, amax, jerk, delay)
    assert np.all(np.abs(actual - old) <= vmax / 2**24 + 1e-12)
    travel = stopping_distance(actual, acceleration, amax, jerk, delay)
    assert np.all(travel[actual > 0] <= distance[actual > 0] + 1e-12)


def test_core_remains_portable_without_mujoco_imports():
    import subprocess
    import sys

    source = '''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] == "mujoco":
        raise AssertionError("MuJoCo entered the portable controller")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore, TCP_FRAMES
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile
import numpy as np
core = PinkIKCore(config=controller_config_for_profile("x7-mit-highspeed-v10"))
q = core.home_q.copy()
core.configuration.update(q)
targets = {side: core.configuration.get_transform_frame_to_world(frame).copy()
           for side, frame in TCP_FRAMES.items()}
core.reset(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
command = core.step(q, targets, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
assert command.executable, command.diagnostics.fault
assert command.mit is not None
'''
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
