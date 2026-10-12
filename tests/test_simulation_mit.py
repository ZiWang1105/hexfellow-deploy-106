"""Check the actual MuJoCo actuator against the independent MIT torque equation."""
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from hexfellow_deploy.workspace import activate

WORKSPACE = activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.simulation import SimulationPlant
from hexfellow_deploy.control.umi_ik import load_umi
load_umi()
from policy_sim.control.mit_command import MitCommand, mit_torque


def test_saved_plant_xml_preserves_mit_torque_and_lift_actuator_configuration(tmp_path):
    import mujoco
    bounds = np.array([52., 52., 28., 28., 6.8, 6.8, 6.8]*2)
    kp, kd = np.array([250., 250., 200., 200., 150., 150., 150.]*2), np.ones(14)*3.
    profile = SimpleNamespace(mit_servo=SimpleNamespace(kp=kp, kd=kd))
    plant = SimulationPlant(WORKSPACE/'configs/scenes/apple_on_table.yaml', tmp_path,
                            effort_limits=bounds, profile=profile)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_path/'plant.xml'))
        compiled = mujoco.MjModel.from_binary_path(str(tmp_path/'plant.mjb'))
        for field in ('actuator_ctrllimited', 'actuator_forcelimited', 'actuator_forcerange',
                      'actuator_gainprm', 'actuator_biasprm', 'body_gravcomp', 'jnt_actfrcrange'):
            np.testing.assert_allclose(getattr(model, field), getattr(compiled, field), atol=1e-10, rtol=0.)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('initial').id)
        qids = [int(model.joint(n).qposadr[0]) for n in plant.arm_names]
        vids = [int(model.joint(n).dofadr[0]) for n in plant.arm_names]
        q = data.qpos[qids].copy()
        v = np.linspace(-.3, .3, 14)
        data.qvel[vids] = v
        for delta in (.001, -2., 2.):
            command = MitCommand(plant.arm_names, q+delta, v+.02, np.linspace(-.5, .5, 14), kp, kd, 1.)
            data.ctrl[plant.arm_ids] = command.motor_input()
            mujoco.mj_forward(model, data)
            expected, _ = mit_torque(command, q, v, bounds)
            np.testing.assert_allclose(data.actuator_force[plant.arm_ids], expected, atol=1e-10, rtol=0.)
            np.testing.assert_allclose(data.qfrc_actuator[vids], expected, atol=1e-10, rtol=0.)
        assert np.all(model.body_gravcomp == 0.)
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert model.jnt_actfrcrange[model.joint(plant.joints[0]).id, 1] == 1800.
        assert model.actuator_biasprm[plant.lift_servo.actuator, 2] == -plant.lift_servo.velocity_gain
        assert not model.actuator_ctrllimited[plant.lift_servo.actuator]
    finally:
        plant.close()


def test_mujoco_motor_matches_joint_pd_feedforward_and_combined_effort_saturation(tmp_path):
    bounds = np.array([52., 52., 28., 28., 6.8, 6.8, 6.8]*2)
    plant = SimulationPlant(WORKSPACE/'configs/scenes/apple_on_table.yaml', tmp_path,
                            effort_limits=bounds, profile='test_joint_torque_semantics')
    try:
        model, data = plant.model, plant.data
        qids = [int(model.joint(name).qposadr[0]) for name in plant.arm_names]
        vids = [int(model.joint(name).dofadr[0]) for name in plant.arm_names]
        q = data.qpos[qids].copy()
        velocity = np.linspace(-.3, .3, 14)
        data.qvel[vids] = velocity
        for sequence, displacement in enumerate((.001, 2., -2.), 1):
            command = MitCommand(plant.arm_names, q+displacement, velocity+.05,
                                 np.linspace(-.5, .5, 14), np.ones(14)*100., np.ones(14)*3., 1.)
            reference = dict(joint_positions={name:float(data.qpos[model.joint(name).qposadr[0]])
                                             for name in plant.joints},
                             base_velocity_body=[0., 0., 0.], gripper_open=dict(left=None, right=None))
            plant.submit(sequence, reference, command)
            deadline = time.monotonic()+2.
            while plant.acknowledgement(sequence) is None and time.monotonic() < deadline:
                plant.check()
                time.sleep(.001)
            assert plant.acknowledgement(sequence) is not None
            # acquire() was never called: no integration runs concurrently.
            plant.mj.mj_forward(model, data)
            expected, unsaturated = mit_torque(command, q, velocity, bounds)
            np.testing.assert_allclose(data.actuator_force[plant.arm_ids], expected, atol=1e-10, rtol=0.)
            np.testing.assert_allclose(data.qfrc_actuator[vids], expected, atol=1e-10, rtol=0.)
            assert np.any(np.abs(unsaturated) > bounds) == (abs(displacement) > 1.)
        assert np.all(model.body_gravcomp == 0.)
        assert not np.any(model.actuator_ctrllimited[plant.arm_ids])
    finally:
        plant.close()


def test_lift_inner_servo_holds_unknown_load_and_respects_bounded_velocity_demand():
    import mujoco
    from hexfellow_deploy.simulation import LiftPositionServo
    model = mujoco.MjModel.from_xml_string('''
        <mujoco><option timestep="0.002" integrator="implicitfast"/>
        <worldbody><body><joint name="lift" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="50"/></body></worldbody>
        <actuator><position name="position_lift" joint="lift" kp="8000" kv="500"
        forcelimited="true" forcerange="-1800 1800"/></actuator></mujoco>''')
    for payload_force in (0., -10., -100.):
        data = mujoco.MjData(model)
        data.qpos[0] = .3
        mujoco.mj_forward(model, data)
        servo = LiftPositionServo(model, data, mujoco, 'lift', .025)
        samples = []
        for step in range(12500):
            if step % 10 == 0:
                # Match v1 lift commands: measured q plus a solved 20 ms step.
                # HOLD requests measured q again, so the motor must reject load.
                servo.target = float(data.qpos[0]) + (.0003 if 1000 <= step < 2500 else 0.)
            data.qfrc_applied[0] = payload_force if step >= 100 else 0.
            servo.step(data, .002)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            samples.append((float(data.qpos[0]), float(data.qvel[0]), float(data.actuator_force[0])))
        samples = np.asarray(samples)
        assert np.max(np.abs(samples[:, 1])) <= .025000001
        assert .02 < samples[2500, 0]-samples[1000, 0] < .06
        assert abs(samples[-1, 0]-samples[2500, 0]) < .001
        assert abs(samples[-1, 1]) < 1e-7
        assert np.max(np.abs(samples[:, 2])) <= 1800.


def test_full_robot_lift_supports_payload_disturbance_during_arm_motion(tmp_path):
    bounds = np.array([52., 52., 28., 28., 6.8, 6.8, 6.8]*2)
    plant = SimulationPlant(WORKSPACE/'configs/scenes/apple_on_table.yaml', tmp_path,
                            effort_limits=bounds, profile='test_coupled_lift_servo')
    # Deterministic physics stepping below uses the actual configured plant.
    plant.stop_event.set()
    plant.thread.join()
    try:
        model, data, servo = plant.model, plant.data, plant.lift_servo
        qids = [int(model.joint(name).qposadr[0]) for name in plant.arm_names]
        vids = [int(model.joint(name).dofadr[0]) for name in plant.arm_names]
        initial_q = data.qpos[qids].copy()
        model.actuator_biasprm[plant.arm_ids, 1] = -200.
        model.actuator_biasprm[plant.arm_ids, 2] = -8.
        plant.mj.mj_forward(model, data)
        assert data.qfrc_bias[servo.dof] > 100.
        assert model.jnt_actfrcrange[model.joint(plant.joints[0]).id, 1] == 1800.
        samples = []
        for step in range(12500):
            t = step*.002
            q = initial_q.copy()
            v = np.zeros(14)
            if 1. <= t < 4.:
                q[3] += .15*np.sin(2*np.pi*(t-1.))
                v[3] = .3*np.pi*np.cos(2*np.pi*(t-1.))
            data.ctrl[plant.arm_ids] = 200.*q+8.*v+data.qfrc_bias[vids]
            if step % 10 == 0:
                servo.target = float(data.qpos[servo.qpos])+(.0003 if 1000 <= step < 2500 else 0.)
            # A generic unmodeled downward load on the wrist reaches the lift
            # through robot dynamics. No object truth reaches the servo.
            data.xfrc_applied[model.body('robot2_base').id, 2] = -10. if t >= 1. else 0.
            servo.step(data, .002)
            plant.mj.mj_step(model, data)
            plant.mj.mj_forward(model, data)
            samples.append((float(data.qpos[servo.qpos]), float(data.qvel[servo.dof]),
                            float(data.qfrc_actuator[servo.dof])))
        samples = np.asarray(samples)
        assert np.max(np.abs(samples[:, 1])) < .026
        assert .02 < samples[2500, 0]-samples[1000, 0] < .06
        assert abs(samples[-1, 0]-samples[2500, 0]) < .001
        assert abs(samples[-1, 1]) < 1e-6
        assert np.max(np.abs(samples[:, 2])) <= 1800.
    finally:
        plant.close()
