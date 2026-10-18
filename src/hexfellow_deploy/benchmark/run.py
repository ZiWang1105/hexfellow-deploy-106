"""Reproducible fixed-base IK benchmark using the deployment MIT plant.

Run with ``python -m hexfellow_deploy.benchmark.run --help``. This module never
opens hardware. Requested, conditioned, command and encoder-FK poses share
the same simulation timestamp. Wall-clock solve timing is recorded separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import subprocess
import time

import qpsolvers  # Required shared-library initialization order.
import numpy as np
import pinocchio as pin
import yaml

from hexfellow_deploy.robot.joints import JOINTS, SIDES, BASE
from hexfellow_deploy.robot.upstream import PROJECT_ROOT
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile, load_motion_profile
from .profiles import experiment_profile
from .trajectories import Trajectory, ScrewTrajectory, JointPathTrajectory, envelope_report


# Reachable lower endpoint, independently solved from the owned X7 geometry.
# These are initial simulation conditions, never a hardware startup command.
RIGHT_START = [-1.2474970114201773, .5043878145917352, -.5124911622895918,
               1.3629251005083702, 1.932356295405531, .6020770602933879, -.19650667000133545]


def initial_state(posture='nominal'):
    right = np.array(RIGHT_START)
    if posture in ('extended', 'extended_offset'):
        right = np.array([-.8, 1., -1., 1.2, 0., 0., 0.])
    if posture in ('heldout', 'extended_offset'):
        right += np.array([.10, -.08, .10, .05, -.10, -.05, .05])
    left = right*np.array([-1., 1., -1., 1., -1., 1., -1.])
    return np.r_[.3, left, right]


def poses(model, data, q):
    pin.framesForwardKinematics(model, data, q)
    return {s: data.oMf[model.getFrameId(s+'_tcp')].copy() for s in SIDES}


def matrices(values):
    return np.array([values[s].homogeneous for s in SIDES])


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False)+'\n')


def scene_for_run(output, q, dt):
    source = PROJECT_ROOT/'configs/scenes/apple_on_table.yaml'
    scene = yaml.safe_load(source.read_text())
    for key in ('assembly', 'source_checkout'):
        scene['robot'][key] = str((source.parent/scene['robot'][key]).resolve())
    scene['camera']['calibration'] = str((source.parent/scene['camera']['calibration']).resolve())
    scene['robot']['initial_joints'] = dict(zip(JOINTS, map(float, q)))
    scene['entities'] = []
    # Use an integer number of physics steps at each control period.
    scene['physics']['timestep_s'] = .002 if np.isclose(dt/.002, round(dt/.002)) else .001
    path = Path(output)/'scene.yaml'
    path.write_text(yaml.safe_dump(scene, sort_keys=False))
    return path


class ContactAudit:
    """Offline evaluator, including all MuJoCo collision-enabled contact pairs."""
    def __init__(self, inner, model):
        self.inner, self.model = inner, model
        self.pairs = Counter()
        self.steps = 0
        self.minimum_distance_m = 0.
        self.arm_ids = [model.actuator('position_'+n).id for n in JOINTS[1:]]
        self.maximum_torque_nm = np.zeros(14)
        self.saturated_steps = 0

    def step(self, data):
        self.inner.step(data)
        effort = np.abs(data.actuator_force[self.arm_ids])
        self.maximum_torque_nm = np.maximum(self.maximum_torque_nm, effort)
        self.saturated_steps += bool(np.any(effort >= self.model.actuator_forcerange[self.arm_ids, 1]-1e-7))
        contacts = set()
        for c in data.contact:
            if c.dist >= 0 or c.geom1 < 0 or c.geom2 < 0:
                continue
            pair = tuple(sorted(self.model.body(int(self.model.geom_bodyid[g])).name or 'world'
                                for g in (c.geom1, c.geom2)))
            contacts.add(pair)
            self.minimum_distance_m = min(self.minimum_distance_m, float(c.dist))
        self.steps += bool(contacts)
        self.pairs.update(contacts)

    def save(self, directory):
        self.inner.save(directory)

    def report(self):
        return dict(contact_steps=self.steps, minimum_distance_m=self.minimum_distance_m,
                    maximum_torque_nm=self.maximum_torque_nm, saturated_steps=self.saturated_steps,
                    pairs=[dict(bodies=k, steps=v) for k, v in self.pairs.items()],
                    scope='Every physics step; only collision-enabled MuJoCo geometry, not complete mesh clearance')


def statistics(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(rms=float(np.sqrt(np.mean(values**2))), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                max=float(np.max(values)))


def longest_interval(flags, dt):
    longest = current = 0
    for flag in flags:
        current = current+1 if flag else 0
        longest = max(longest, current)
    return longest*dt


def summarize(trace, trajectory, dt):
    t = trace['time_s']
    active = trajectory.motion_mask(t)
    arms = [SIDES.index(s) for s in trajectory.moving_arms] if hasattr(trajectory, 'moving_arms') else ([0, 1] if trajectory.dual else [1])
    output = {}
    for field in ('measured', 'conditioned', 'command'):
        reference, actual = trace['requested'][:, arms], trace[field][:, arms]
        xyz = np.linalg.norm(reference[:, :, :3, 3]-actual[:, :, :3, 3], axis=2)*1000
        angle = np.array([[np.linalg.norm(pin.log3(a[:3, :3].T@b[:3, :3]))*180/np.pi
                           for a, b in zip(r, x)] for r, x in zip(reference, actual)])
        output[field] = dict(position_mm=statistics(xyz[active]), orientation_deg=statistics(angle[active]),
                            final_position_mm=xyz[-1].tolist(), final_orientation_deg=angle[-1].tolist())
    speeds = np.linalg.norm(trace['tcp_velocity'][:, arms, :3], axis=2)
    requested_speed = np.linalg.norm(trace['target_twist'][:, arms, :3], axis=2)
    # Local twist derivatives exclude the moving-frame term. Add it for the
    # physical Cartesian acceleration, including centripetal acceleration.
    target_a = trace['target_acceleration'][:, :, :3]+np.cross(
        trace['target_twist'][:, :, 3:], trace['target_twist'][:, :, :3])
    acceleration = np.linalg.norm(target_a[:, arms], axis=2)
    per_arm = {SIDES[arm]: dict(peak_m_s=float(np.max(speeds[:, i])),
        time_at_or_above_1p2_m_s=float(np.sum(speeds[:, i] >= 1.2)*dt),
        longest_continuous_interval_at_or_above_1p2_m_s=longest_interval(speeds[:, i] >= 1.2, dt))
        for i, arm in enumerate(arms)}
    warmed = t >= trajectory.settle_s
    output.update(achieved_speed_m_s=statistics(speeds[active]), requested_speed_m_s=statistics(requested_speed[active]),
        requested_acceleration_m_s2=statistics(acceleration[active]),
        time_at_or_above_1p2_m_s=float(np.sum(np.any(speeds >= 1.2, axis=1))*dt),
        longest_continuous_interval_at_or_above_1p2_m_s=longest_interval(np.any(speeds >= 1.2, axis=1), dt),
        moving_arm_speed=per_arm,
        solve_ms=statistics(trace['solve_ms']), deadline_exceeded_ticks=int(np.sum(trace['solve_ms'] > dt*1000)),
        first_solve_ms=float(trace['solve_ms'][0]), solve_ms_after_initial_hold=statistics(trace['solve_ms'][warmed]),
        deadline_exceeded_ticks_after_initial_hold=int(np.sum(trace['solve_ms'][warmed] > dt*1000)),
        reference_joint_speed_max_rad_s=np.max(np.abs(trace['reference_velocity'][:, 1:]), axis=0),
        measured_joint_speed_max_rad_s=np.max(np.abs(trace['velocity'][:, 1:]), axis=0))
    if len(t) > 3:
        acceleration = np.diff(trace['reference_velocity'][:, 1:], axis=0)/dt
        jerk = np.diff(acceleration, axis=0)/dt
        output.update(reference_acceleration_max_rad_s2=np.max(np.abs(acceleration), axis=0),
                      reference_jerk_max_rad_s3=np.max(np.abs(jerk), axis=0),
                      reference_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))))
    hold = t > trajectory.motion_end_s+.5
    if np.any(hold):
        output['hold_tcp_speed_m_s'] = statistics(speeds[hold])
    if 'plant_fk_error_m' in trace:
        output['encoder_fk_vs_mujoco_m'] = statistics(trace['plant_fk_error_m'])
    return output


def run(args):
    from hexfellow_deploy.scheduling import python_scheduling
    # Match the existing deployment control worker, including collection before
    # control starts. An explicit ablation can re-enable cyclic collection.
    with python_scheduling(dict(gil_switch_s=.0002, cyclic_gc=args.cyclic_gc)):
        return _run(args)


def _run(args):
    from hexfellow_deploy.scheduling import runtime_state
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists() or (output/'trace.npz').exists():
        raise ValueError('Use a new output directory; benchmark evidence must not be overwritten')
    tuning = json.loads(args.tuning)
    profile_path = (Path(args.profile).resolve() if args.profile else experiment_profile(output,
        version=args.version, hz=args.hz, fast=args.fast, tuning=tuning, dynamics=args.dynamics))
    profile = load_motion_profile(profile_path)
    config = controller_config_for_profile(profile, lift_position_margin=0.)
    from hexfellow_deploy.control.umi_ik import UMIPinkIK
    from hexfellow_deploy.robot.kinematics import Kinematics
    from hexfellow_deploy.robot.assembly import load_assembly
    adapter_config = output/'controller.yaml'
    adapter_config.write_text(yaml.safe_dump(dict(motion_profile=str(profile_path),
        measured_pose_tracking=args.measured_pose_tracking, arm_velocity_limit_rad_s='profile', lift_velocity_limit_m_s=.025)))
    kinematics = Kinematics(PROJECT_ROOT/'robot_resources/urdf/maver_x4_gp80_zed_40deg.urdf',
                           load_assembly(PROJECT_ROOT/'configs/robot/maver_gp80.yaml'))
    adapter = UMIPinkIK(kinematics, adapter_config)
    core = adapter.core
    if [core.model.names[i] for i in range(1, core.model.njoints)] != list(JOINTS):
        raise ValueError('Benchmark requires the documented 15-coordinate model order')
    dt = config.control_dt
    spec = json.loads(Path(args.trajectory_spec).read_text()) if args.trajectory_spec else None
    q, velocity = initial_state(args.posture if spec is None else spec.get('posture', args.posture)), np.zeros(15)
    data = core.model.createData()
    start = poses(core.model, data, q)
    if spec is not None:
        from .diverse import DiverseTrajectory
        trajectory = DiverseTrajectory(core.model, q, spec, args.speed)
    elif args.trajectory == 'swing':
        screws = {s: pin.computeFrameJacobian(core.model, data, q,
            core.model.getFrameId(s+'_tcp'), pin.LOCAL)[:, 1 if s == 'left' else 8]
            * (-1. if s == 'left' else 1.) for s in SIDES}
        trajectory = ScrewTrajectory(start, screws, args.speed, args.dual,
                                     .8 if args.distance is None else args.distance)
    elif args.trajectory == 'joint_curve':
        trajectory = JointPathTrajectory(core.model, q, args.speed, args.dual,
                                         1. if args.distance is None else args.distance)
    else:
        trajectory = Trajectory(start, args.trajectory, args.speed, args.dual, args.distance)
    scene_path = scene_for_run(output, q, dt)
    plant = audit = nonlinear = None
    failure = None
    trace, diagnostics = [], []
    statuses, reasons = Counter(), Counter()
    reference_q, reference_v = q.copy(), velocity.copy()
    conditioned = {s: x.copy() for s, x in start.items()}
    if args.solver == 'nonlinear':
        from .nonlinear import NonlinearPoseIK
        nonlinear = NonlinearPoseIK(core.model, q)
    def feedback_packet(q, velocity, timestamp):
        return dict(timestamp_s=timestamp, joint_positions=dict(zip(JOINTS, map(float, q))),
                    joint_velocities=dict(zip(JOINTS, map(float, velocity))),
                    gripper_state=dict(left='closed', right='closed'))
    policy_start = {s: start[s]*adapter.policy_T_umi[s].inverse() for s in SIDES}
    adapter.track({s: (p.translation, p.rotation) for s, p in policy_start.items()})
    adapter.initialize_reference(feedback_packet(q, velocity, 0.))
    metadata = dict(arguments=vars(args), profile=str(profile_path), profile_sha256=profile.sha256,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'diff', '--name-only'], cwd=PROJECT_ROOT, text=True).strip()),
        clock='deterministic simulation time; compute timing is host wall time',
        tcp='Owned X7 core fingertip frames left_tcp/right_tcp in robot base; pad-center offset recorded separately',
        base='fixed command', lift='hard locked at initial measured height', dt_s=dt,
        trajectory_spec=spec, moving_arms=getattr(trajectory, 'moving_arms', list(SIDES) if args.dual else ['right']),
        duration_s=trajectory.duration_s, initial_q=q, motion_start_s=trajectory.settle_s,
        motion_end_s=trajectory.motion_end_s, solver_parameters=None if nonlinear is None else nonlinear.parameters,
        python_runtime=runtime_state(),
        controller=adapter.report, source_sha256={str(p.relative_to(PROJECT_ROOT)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in
            sorted((PROJECT_ROOT/'src/hexfellow_deploy').rglob('*.py'))},
        request_envelope=envelope_report(trajectory, profile), hardware_qualified=False)
    write_json(output/'metadata.json', metadata)
    try:
        if args.plant == 'mit':
            from hexfellow_deploy.simulation import SimulationPlant
            plant = SimulationPlant(scene_path, output/'plant', profile=profile,
                effort_limits=profile.mit_servo.torque_limit_nm, realtime=False)
            if args.plant_mass_scale != 1.:
                # Evaluator-only mismatch; IK retains its nominal model.
                plant.model.body_mass[:] *= args.plant_mass_scale
                plant.model.body_inertia[:] *= args.plant_mass_scale
                plant.mj.mj_setConst(plant.model, plant.mj.MjData(plant.model))
                plant.mj.mj_forward(plant.model, plant.data)
                plant.mj.mj_saveModel(plant.model, str(output/'plant/plant.mjb'))
            plant.physics_audit = audit = ContactAudit(plant.physics_audit, plant.model)
            plant.acquire()
        for step in range(int(np.ceil(trajectory.duration_s/dt))+1):
            now = step*dt
            if plant is not None:
                measured = plant.read()
                q = np.array([measured['joint_positions'][n] for n in JOINTS])
                velocity = np.array([measured['joint_velocities'][n] for n in JOINTS])
                if not np.isclose(measured['timestamp_s'], now, atol=1e-8, rtol=0):
                    raise RuntimeError('Plant clock and benchmark schedule diverged')
            target, twist, acceleration = trajectory.sample(now)
            input_target, input_twist, input_acceleration = trajectory.sample(now+args.preview_s)
            actual = poses(core.model, data, q)
            commanded = poses(core.model, data, reference_q)
            tcp_velocity = np.array([pin.computeFrameJacobian(core.model, data, q,
                core.model.getFrameId(s+'_tcp'), pin.LOCAL)@velocity for s in SIDES])
            row = dict(time_s=now, q=q.copy(), velocity=velocity.copy(), reference_q=reference_q.copy(),
                reference_velocity=reference_v.copy(), requested=matrices(target), measured=matrices(actual),
                command=matrices(commanded), conditioned=matrices(conditioned), tcp_velocity=tcp_velocity,
                target_twist=np.array([twist[s] for s in SIDES]),
                target_acceleration=np.array([acceleration[s] for s in SIDES]))
            if plant is not None:
                row.update(plant_qpos=plant.data.qpos.copy(), plant_qvel=plant.data.qvel.copy(),
                           torque=plant.data.actuator_force[plant.arm_ids].copy())
                base = plant.data.body(BASE)
                base_T_world = pin.SE3(base.xmat.reshape(3, 3), base.xpos).inverse()
                site_poses = {s: base_T_world*pin.SE3(plant.data.site(s+'_tcp').xmat.reshape(3, 3),
                    plant.data.site(s+'_tcp').xpos)*adapter.policy_T_umi[s] for s in SIDES}
                row['plant_fk_error_m'] = np.array([np.linalg.norm(actual[s].translation-site_poses[s].translation)
                                                    for s in SIDES])
            started = time.perf_counter()
            options = {}
            if args.derivatives:
                options.update(target_twists=input_twist, target_accelerations=input_acceleration)
            if nonlinear is not None:
                endpoint = nonlinear.solve(input_target, q)
                joint_velocity = nonlinear.velocity(endpoint, input_twist)
                joint_velocity += (args.joint_correction_gain-1.)*(endpoint-reference_q)
                options.update(posture_target=endpoint, posture_velocity=joint_velocity)
            if nonlinear is None:
                policy_targets = {s: input_target[s]*adapter.policy_T_umi[s].inverse() for s in SIDES}
                kwargs = {}
                if args.derivatives:
                    kwargs = dict(body_twists={s: adapter.policy_T_umi[s].action@input_twist[s] for s in SIDES},
                        body_accelerations={s: adapter.policy_T_umi[s].action@input_acceleration[s] for s in SIDES},
                        sample_timestamp_s=now)
                adapter.track({s: (p.translation, p.rotation) for s, p in policy_targets.items()}, **kwargs)
                try:
                    adapter.step(feedback_packet(q, velocity, now))
                except RuntimeError:
                    # Preserve the failing state in the benchmark artifact.
                    row['solve_ms'] = (time.perf_counter()-started)*1000
                    trace.append(row)
                    raise
                result = adapter.last_result
            else:
                result = core.step(q, input_target, measured_velocity=velocity, state_timestamp=now,
                    control_timestamp=now, feedback_received={s: now for s in ('left', 'right', 'theta', 'base')},
                    feedback_policy='strict', **options)
            row['solve_ms'] = (time.perf_counter()-started)*1000
            trace.append(row)
            diagnostic = asdict(result.diagnostics)
            diagnostics.append(dict(time_s=now, **diagnostic))
            statuses[result.diagnostics.status] += 1
            reasons.update(result.diagnostics.boundary_limiting_reasons)
            if not result.executable or result.diagnostics.fault:
                raise RuntimeError(result.diagnostics.fault or result.diagnostics.status)
            reference_q, reference_v = result.positions.copy(), result.velocity.copy()
            conditioned = (input_target if nonlinear is not None else
                           {s: p.copy() for s, p in core._limited_targets.items()})
            if plant is not None:
                command = dict(joint_positions=dict(zip(JOINTS, map(float, reference_q))),
                               base_velocity_body=[0., 0., 0.], gripper_open=dict(left=None, right=None))
                plant.submit(step, command, result.mit)
                plant.advance(dt)
                if plant.acknowledgement(step) is None:
                    raise RuntimeError('MIT reference was not applied')
            else:
                q, velocity = reference_q.copy(), reference_v.copy()
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        if plant is not None:
            plant.close()
    arrays = {name: np.asarray([row[name] for row in trace]) for name in trace[0]} if trace else {}
    if arrays:
        np.savez_compressed(output/'trace.npz', **arrays)
    with (output/'diagnostics.jsonl').open('w') as stream:
        for row in diagnostics:
            stream.write(json.dumps(row, default=json_default)+'\n')
    summary = dict(completed=failure is None, failure=failure, samples=len(trace), status_counts=dict(statuses),
                   boundary_counts=dict(reasons), metrics=summarize(arrays, trajectory, dt) if arrays else None,
                   contacts=None if audit is None else audit.report(), hardware_qualified=False)
    if arrays:
        m = summary['metrics']
        ref_margin = np.minimum(arrays['reference_q'][:, 1:]-core.model.lowerPositionLimit[1:],
                                core.model.upperPositionLimit[1:]-arrays['reference_q'][:, 1:])
        measured_margin = np.minimum(arrays['q'][:, 1:]-core.model.lowerPositionLimit[1:],
                                     core.model.upperPositionLimit[1:]-arrays['q'][:, 1:])
        summary['constraint_checks'] = dict(reference_position_margin_rad=float(np.min(ref_margin)),
            measured_position_margin_rad=float(np.min(measured_margin)),
            reference_velocity_ratio=float(np.max(np.abs(arrays['reference_velocity'][:, 1:])/
                                                    core.model.velocityLimit[1:])),
            measured_velocity_ratio=float(np.max(np.abs(arrays['velocity'][:, 1:])/core.model.velocityLimit[1:])),
            reference_acceleration_ratio=float(np.max(m.get('reference_acceleration_max_rad_s2', [0.])/
                                                        np.asarray(profile.arm_acceleration_rad_s2))),
            reference_jerk_ratio=float(np.max(m.get('reference_jerk_max_rad_s3', [0.])/
                                                np.asarray(profile.arm_jerk_rad_s3))),
            planner_failures=max((d['target_planner_failures'] for d in diagnostics), default=0))
        tracking = m['measured']
        c = summary['constraint_checks']
        summary['tracking_acceptance'] = dict(
            completed=summary['completed'], position_rms_5mm=bool(tracking['position_mm'] and tracking['position_mm']['rms'] <= 5.),
            position_p95_10mm=bool(tracking['position_mm'] and tracking['position_mm']['p95'] <= 10.),
            orientation_p95_2deg=bool(tracking['orientation_deg'] and tracking['orientation_deg']['p95'] <= 2.),
            final_position_1mm=bool(max(tracking['final_position_mm']) <= 1.),
            limits=bool(c['reference_position_margin_rad'] >= -1e-6 and c['measured_position_margin_rad'] >= -1e-6
                and c['reference_velocity_ratio'] <= 1.0001
                and c['reference_acceleration_ratio'] <= 1.0001 and c['reference_jerk_ratio'] <= 1.0001
                and c['measured_velocity_ratio'] <= 1.0001), no_planner_failures=c['planner_failures'] == 0,
            no_modeled_contacts=bool(audit is not None and audit.steps == 0),
            no_torque_saturation=bool(audit is not None and audit.saturated_steps == 0))
        summary['tracking_accepted'] = all(summary['tracking_acceptance'].values())
        summary['fast_tracking_accepted'] = summary['tracking_accepted'] and all(
            speed['longest_continuous_interval_at_or_above_1p2_m_s'] >= .1
            for speed in m['moving_arm_speed'].values())
    write_json(output/'summary.json', summary)
    m = summary['metrics']
    print(json.dumps(dict(output=str(output), completed=summary['completed'], failure=failure,
        rms_mm=None if m is None or m['measured']['position_mm'] is None else m['measured']['position_mm']['rms'],
        peak_m_s=None if m is None or m['achieved_speed_m_s'] is None else m['achieved_speed_m_s']['max'],
        solve_p95_ms=None if m is None else m['solve_ms']['p95'])))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--plant', choices=('kinematic', 'mit'), default='mit')
    p.add_argument('--plant-mass-scale', type=float, default=1., help='Evaluator-only mass/inertia multiplier; nominal IK model unchanged')
    p.add_argument('--solver', choices=('pink', 'nonlinear'), default='pink')
    p.add_argument('--version', choices=('v1', 'v2', 'v3'), default='v1')
    p.add_argument('--hz', type=float, default=50)
    p.add_argument('--profile', help='Existing immutable experimental profile; overrides profile-generation flags')
    p.add_argument('--fast', action='store_true', help='Candidate Cartesian envelope 1.5 m/s, 8 m/s², 120 m/s³')
    p.add_argument('--derivatives', action='store_true', help='Supply analytic target velocity/acceleration')
    p.add_argument('--measured-pose-tracking', action='store_true', help='Use measured TCP feedback for the objective')
    p.add_argument('--dynamics', action='store_true', help='Nominal inertia feedforward ablation, MIT gains unchanged')
    p.add_argument('--tuning', default='{}', help='JSON controller_tuning overrides, saved in derived profile')
    p.add_argument('--trajectory', default='vertical', choices=('vertical', 'lateral', 'forward', 'circle', 'figure8', 'rotate', 'swing', 'joint_curve', 'rounded_rectangle'))
    p.add_argument('--trajectory-spec', help='Frozen diverse path JSON; overrides trajectory, posture and dual flags')
    p.add_argument('--posture', default='nominal', choices=('nominal', 'heldout', 'extended', 'extended_offset'))
    p.add_argument('--preview-s', type=float, default=0., help='Explicit known-trajectory preview, never an evaluation time shift')
    p.add_argument('--joint-correction-gain', type=float, default=1., help='Nonlinear endpoint error feedback in s^-1')
    p.add_argument('--speed', type=float, default=.4)
    p.add_argument('--distance', type=float)
    p.add_argument('--dual', action='store_true')
    p.add_argument('--cyclic-gc', action='store_true', help='Ablation: enable cyclic GC instead of matching deployment scheduling')
    return p


def main():
    args = parser().parse_args()
    if not np.isfinite(args.plant_mass_scale) or not .5 <= args.plant_mass_scale <= 1.5:
        raise ValueError('Plant mass scale must be in [0.5, 1.5]')
    if args.hz not in (50, 100):
        raise ValueError('The current motion-profile schema supports 50 and 100 Hz')
    if not np.isfinite(args.preview_s) or not 0 <= args.preview_s <= .02:
        raise ValueError('Preview must be finite and between zero and 20 ms')
    if not np.isfinite(args.joint_correction_gain) or not 0 < args.joint_correction_gain <= 50:
        raise ValueError('Joint correction gain must be in (0, 50] s^-1')
    summary = run(args)
    raise SystemExit(0 if summary['completed'] else 1)


if __name__ == '__main__':
    main()
