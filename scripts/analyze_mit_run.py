"""Offline TCP tracking, timing and pickup evidence for a completed MIT run.

Reads controller telemetry first. Privileged trajectory/evaluator records are only
read after both owners have written final records. This tool never sends commands.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from hexfellow_deploy.workspace import activate

SIDES = ('left', 'right')


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def completed_run(directory):
    if not all((directory/name).is_file() for name in ('control_final.json', 'backend_final.json')):
        raise ValueError('Only completed runs may be analyzed; both owner final records are required')
    return read_json(directory/'control_final.json'), read_json(directory/'backend_final.json')


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist()))


def video_annotation(directory):
    """Describe the entire closed experiment, independently of per-frame grasp."""
    import textwrap
    final, backend = completed_run(directory)
    run = read_json(directory/'run.json') if (directory/'run.json').is_file() else {}
    termination = run.get('termination') or (read_json(directory/'termination.json')
        if (directory/'termination.json').is_file() else {})
    error = run.get('fault') or final.get('fault') or backend.get('fault') or backend.get('shutdown_error')
    if error:
        error = str(error)
        while error.startswith(('RuntimeError: ', 'IK failed: ')):
            error = error.split(': ', 1)[1]
        heading = 'RUN FAILED: '+error
    elif run.get('evaluation', {}).get('accepted'):
        heading = 'RUN PASSED: simulated pickup validation'
    elif run.get('evaluation'):
        heading = 'RUN FAILED: pickup validation criteria not met'
    else:
        heading = 'RUN OUTCOME: '+str(run.get('status', termination.get('reason', final.get('phase', 'unknown'))))
    reason = termination.get('reason', 'unrecorded termination')
    budget = 'Full budget completed' if reason == 'completed_budget' else 'Incomplete budget: '+reason
    hold_complete = (reason == 'completed_budget' and termination.get('settle_s', 0.) >= 10.
        and termination.get('stationary_since_s') is not None and final.get('final_hold')
        and final.get('diagnostic', {}).get('ik_status') == 'HOLD'
        and final.get('diagnostic', {}).get('base_stopped', False))
    hold = 'Final HOLD verified' if hold_complete else 'No verified final HOLD'
    lines = textwrap.wrap(heading, width=106, break_long_words=True)[:2]
    return lines+[budget+' | '+hold, 'Offline recorded-state replay | Native hardware not qualified']


def pose_error(actual, target):
    position = float(np.linalg.norm(np.asarray(actual['position_m'])-target['position_m']))
    ar = Rotation.from_quat(actual['quaternion_xyzw'])
    tr = Rotation.from_quat(target['quaternion_xyzw'])
    angle = float((tr*ar.inv()).magnitude())
    return position, angle


def mit_summary(records, effort_limits):
    available = [row for row in records if row.get('mit')]
    if not available:
        return dict(recorded_frames=0, complete_recording=False)
    names = available[0]['mit']['joint_names']
    values = {field: [] for field in ('position', 'velocity', 'torque_ff', 'kp', 'kd')}
    torques, mismatch = [], []
    for row in available:
        frame = row['mit']
        if frame['joint_names'] != names:
            raise ValueError('MIT joint order changed within the recording')
        command = {field: np.asarray(frame[field], dtype=float) for field in values}
        if any(value.shape != (len(names),) or not np.all(np.isfinite(value)) for value in command.values()):
            raise ValueError('Malformed recorded MIT vector')
        for field, value in command.items():
            values[field].append(value)
        q = np.array([row['measured']['joint_positions'][name] for name in names])
        v = np.array([row['measured']['joint_velocities'][name] for name in names])
        torques.append(command['kp']*(command['position']-q)+command['kd']*(command['velocity']-v)
                       +command['torque_ff'])
        mismatch.append(np.abs(command['position']-[row['reference']['joint_positions'][name] for name in names]))
    result = dict(recorded_frames=len(available), complete_recording=len(available) == len(records),
                  joint_names=names, maximum_reference_position_mismatch_rad=float(np.max(mismatch)),
                  predicted_torque_scope='Unsaturated MIT equation at source encoder feedback, before actuator application')
    for field, matrix in values.items():
        matrix = np.asarray(matrix)
        result[field] = dict(minimum=np.min(matrix, axis=0).tolist(), maximum=np.max(matrix, axis=0).tolist())
    torques = np.abs(torques)
    result['maximum_absolute_predicted_torque_nm'] = np.max(torques, axis=0).tolist()
    if effort_limits is not None:
        result['predicted_effort_limit_exceedances'] = int(np.count_nonzero(torques > np.asarray(effort_limits)+1e-7))
    return result


def target_for(row, actual, candidate):
    target = row.get('active_tcp_targets')
    if target:
        target = {side: dict(position_m=value['position_m'], quaternion_xyzw=(
                    Rotation.from_matrix(value['rotation']).as_quat().tolist()
                    if 'rotation' in value else value['quaternion_xyzw']))
                  for side, value in target.items()}
        return target, 'recorded_active_target'
    errors = row['diagnostic'].get('arm_errors', {})
    if not candidate or set(errors) != set(SIDES):
        return None, None
    # Older recordings saved proposals rather than active targets. Infer a match
    # only if both independently computed errors equal the controller's errors.
    for side in SIDES:
        position, angle = pose_error(actual[side], candidate[side])
        if abs(position-errors[side]['position_m']) > 1e-6 or abs(angle-errors[side]['orientation_rad']) > 1e-6:
            return None, None
    return candidate, 'proposal_matched_to_recorded_errors'


def pickup_summary(directory, final_control):
    path = directory/'evaluation.jsonl'
    if not path.exists():
        return None
    maximum = {}
    last = None
    samples = 0
    for row in rows(path):
        samples += 1
        last = row
        for obj in row['objects']:
            item = maximum.setdefault(obj['name'], dict(height_gain_m=0., retained_s=0., opposing_jaw_frames=0))
            item['height_gain_m'] = max(item['height_gain_m'], obj['height_gain_m'])
            item['retained_s'] = max(item['retained_s'], obj['retained_s'])
            item['opposing_jaw_frames'] += int(obj['opposing_jaw_contacts'])
    if last is None:
        return dict(passed=False, reason='No recorded evaluator samples')
    recorded_final, recorded_samples = last, samples
    dense_path = directory/'physics_evaluation.json'
    dense = read_json(dense_path) if dense_path.is_file() else None
    if dense is not None:
        last, samples = dense, dense['samples']
    contact_count = last['robot_static_contact_steps'] if dense is not None else last['robot_static_contact_frames']
    coverage_complete = dense['complete'] if dense is not None else True
    termination = read_json(directory/'termination.json') if (directory/'termination.json').is_file() else {}
    run = read_json(directory/'run.json') if (directory/'run.json').is_file() else {}
    backend = read_json(directory/'backend_final.json')
    evaluated_body = run.get('evaluation', {}).get('evaluated_body')
    diagnostic = final_control.get('diagnostic', {})
    hold_verified = bool(final_control.get('final_hold', False) and diagnostic.get('ik_status') == 'HOLD'
                         and diagnostic.get('base_stopped', False))
    # A transient retained grasp or early operator stop must not qualify the
    # fixed-budget experiment. The supervisor records completed_budget only
    # after the entire requested uninterrupted final HOLD interval.
    budget_completed = termination.get('reason') == 'completed_budget' and run.get('status') == 'completed_budget'
    hold_interval_completed = (budget_completed and termination.get('settle_s', 0.) >= 10.
                               and termination.get('stationary_since_s') is not None)
    retained_final = [obj['name'] for obj in last['objects'] if obj['height_gain_m'] >= .1
                      and obj['retained_s'] >= 10 and obj['opposing_jaw_contacts']
                      and obj['grasping_arms'] and not obj['support_contact']]
    return dict(passed=evaluated_body in retained_final and contact_count == 0 and coverage_complete
                and hold_verified and hold_interval_completed and not final_control.get('fault')
                and not backend.get('fault') and not backend.get('shutdown_error'),
                final_hold_verified=hold_verified, final_retained_objects=retained_final,
                evaluated_body=evaluated_body, completed_budget=budget_completed,
                final_hold_interval_completed=hold_interval_completed,
                maximum_by_object=maximum, maximum_by_object_sampling='recorded_frames_15_hz',
                final=last, samples=samples, recorded_frame_final=recorded_final,
                recorded_frame_samples=recorded_samples, coverage_complete=coverage_complete,
                scope=('Every physics step; not a continuous-time or hardware guarantee' if dense is not None else
                       'Recorded 15 Hz contact samples; not a continuous-contact guarantee or hardware qualification'))


def analyze(directory, output):
    final, backend = completed_run(directory)
    invocation = read_json(directory/'invocation.json')
    workspace = activate(invocation['workspace'])
    from hexfellow_deploy.robot.kinematics import Kinematics
    from hexfellow_deploy.robot.assembly import load_assembly
    backend_metadata = backend.get('metadata') or read_json(directory/'backend_metadata.json')
    metadata = backend_metadata['robot']
    kin = Kinematics(Path(metadata['urdf']), load_assembly(metadata['assembly']))
    records = list(rows(directory/'control.jsonl'))
    if not records:
        raise ValueError('No controller samples')
    fixture = None
    if invocation.get('action_file'):
        given = Path(invocation['action_file'])
        for path in (given, workspace/given, Path(__file__).resolve().parents[1]/given):
            if path.is_file():
                fixture = read_json(path)
                break
    proposals = []
    if (directory/'decisions.jsonl').exists():
        proposals = [r for r in rows(directory/'decisions.jsonl') if r.get('phase') == 'submitted' and r.get('action')]
    proposal_index, candidate = 0, fixture
    length = len(records)
    arrays = dict(time_s=np.array([r['started_s']-records[0]['started_s'] for r in records]))
    for side in SIDES:
        for label in ('actual', 'reference', 'desired'):
            arrays[f'{side}_{label}_position_m'] = np.full((length, 3), np.nan)
        for label in ('actual_to_reference', 'actual_to_desired', 'reference_to_desired'):
            arrays[f'{side}_{label}_position_mm'] = np.full(length, np.nan)
            arrays[f'{side}_{label}_orientation_deg'] = np.full(length, np.nan)
    target_sources = {}
    last_pose = {}
    for index, row in enumerate(records):
        while proposal_index < len(proposals) and proposals[proposal_index]['published_s'] <= row['started_s']:
            candidate = proposals[proposal_index]['action']
            proposal_index += 1
        poses = {}
        for label, source in (('actual', row['measured']), ('reference', row['reference'])):
            kin.update(source['joint_positions'])
            poses[label] = kin.packet()['tcp_poses']
            for side in SIDES:
                arrays[f'{side}_{label}_position_m'][index] = poses[label][side]['position_m']
        target, source = target_for(row, poses['actual'], candidate)
        if target:
            poses['desired'] = target
            target_sources[source] = target_sources.get(source, 0)+1
        for side in SIDES:
            if target:
                arrays[f'{side}_desired_position_m'][index] = target[side]['position_m']
            for label, (first, second) in {
                'actual_to_reference': ('actual', 'reference'),
                'actual_to_desired': ('actual', 'desired'),
                'reference_to_desired': ('reference', 'desired')}.items():
                if first in poses and second in poses:
                    position, angle = pose_error(poses[first][side], poses[second][side])
                    arrays[f'{side}_{label}_position_mm'][index] = position*1000
                    arrays[f'{side}_{label}_orientation_deg'][index] = np.rad2deg(angle)
        last_pose = poses
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'tcp_samples.npz', **arrays)
    result = dict(run=str(directory), phase=final.get('phase'), fault=final.get('fault'),
                  evidence_kind=final.get('evidence_kind'), native_qualified=False,
                  ticks=final.get('ticks'), generated=final.get('generated'), accepted=final.get('accepted'),
                  pending=final.get('pending'), wall_span_s=float(arrays['time_s'][-1]),
                  overruns=sum(r['timing']['budget_overrun'] for r in records),
                  history_advanced_without_acceptance=sum(r['accepted_before_solve'] != r['sequence']-1 for r in records),
                  target_sources=target_sources, final_tcp=last_pose, tcp_error={}, timing={})
    for side in SIDES:
        result['tcp_error'][side] = {}
        for label in ('actual_to_reference', 'actual_to_desired', 'reference_to_desired'):
            result['tcp_error'][side][label] = {}
            for measure in ('position_mm', 'orientation_deg'):
                values = arrays[f'{side}_{label}_{measure}']
                finite = np.flatnonzero(np.isfinite(values))
                result['tcp_error'][side][label][measure] = dict(statistics=statistics(values),
                    last_available=None if not len(finite) else float(values[finite[-1]]),
                    last_available_time_s=None if not len(finite) else float(arrays['time_s'][finite[-1]]))
    for name, values in {
        'compute_and_submit_ms': [r['timing']['elapsed_ms'] for r in records],
        'interval_ms': [r['interval_ms'] for r in records if r.get('interval_ms') is not None],
        'acceptance_ms': [r['previous_acceptance_ms'] for r in records if r.get('previous_acceptance_ms') is not None],
        'feedback_age_at_solve_ms': [(r['solved_s']-r['measured']['timestamp_s'])*1000 for r in records]
    }.items():
        result['timing'][name] = statistics(values)
    base = np.array([r['reference']['base_velocity_body'] for r in records])
    result['base_commands'] = dict(translation_samples=int(np.count_nonzero(np.linalg.norm(base[:, :2], axis=1)>1e-8)),
                                  rotation_samples=int(np.count_nonzero(np.abs(base[:, 2])>1e-8)),
                                  max_absolute_xyz=np.max(np.abs(base), axis=0).tolist(),
                                  scope='Commanded body velocity only; no inferred measured travel')
    result['grippers'] = {side: dict(commanded_values=sorted({r['reference']['gripper_open'][side] for r in records
                                    if r['reference']['gripper_open'][side] is not None}),
                                    measured_states=sorted({r['measured']['gripper_state'][side] for r in records}),
                                    final_state=records[-1]['measured']['gripper_state'][side]) for side in SIDES}
    if (directory/'actuation.json').exists():
        result['actuation'] = read_json(directory/'actuation.json')
    result['mit'] = mit_summary(records, backend_metadata.get('actuation_limits', {}).get('effort'))
    result['pickup'] = pickup_summary(directory, final)
    (output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(arrays, output/'tcp_tracking.png', directory.name)
    return result


def plot(arrays, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True)
    t = arrays['time_s']
    colors = dict(actual='#16639c', reference='#e28620', desired='#282828')
    for column, side in enumerate(SIDES):
        for coordinate, name in enumerate('XYZ'):
            axis = axes[coordinate, column]
            for label in ('actual', 'reference', 'desired'):
                axis.plot(t, arrays[f'{side}_{label}_position_m'][:, coordinate], label=label,
                          color=colors[label], linewidth=1.25, linestyle='--' if label == 'desired' else '-')
            axis.set_ylabel(f'{name} in base frame (m)')
            axis.grid(alpha=.2)
        axes[0, column].set_title(side.capitalize()+' TCP')
        axes[0, column].legend(loc='best', ncol=3, fontsize=8)
        for row, measure, unit in ((3, 'position_mm', 'Position error (mm)'),
                                   (4, 'orientation_deg', 'Orientation error (deg)')):
            for label, color in (('actual_to_desired', colors['actual']), ('reference_to_desired', colors['reference']),
                                 ('actual_to_reference', '#6d9840')):
                axes[row, column].plot(t, arrays[f'{side}_{label}_{measure}'], label=label.replace('_', ' '), color=color, linewidth=1.)
            axes[row, column].set_ylabel(unit)
            axes[row, column].grid(alpha=.2)
        axes[3, column].legend(fontsize=7)
        axes[4, column].set_xlabel('Elapsed controller wall time (s)')
    figure.suptitle(title+' — measured encoders, MIT reference, and active TCP target\n'
                   'Blank target segments are HOLD/navigation or an unrecorded target; poses are robot-relative.', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .955))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def replay(directory, output, *, video=False):
    """Replay recorded states without integrating or running the controller."""
    completed_run(directory)
    import hashlib
    import mujoco
    from hexfellow_deploy.sim.scene import BuiltScene
    from hexfellow_deploy.sim.scene_config import load_scene
    from hexfellow_deploy.sim.recording import SceneRecorder
    config = yaml.safe_load((directory/'config.yaml').read_text())
    workspace = Path(read_json(directory/'invocation.json')['workspace'])
    scene_config = load_scene(workspace/config['interface']['backend']['scene'])
    camera = scene_config.data['camera']
    calibration_path = scene_config.resolve(camera['calibration'])
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != camera['calibration_sha256']:
        raise ValueError('Camera calibration changed; cannot faithfully replay this recording')
    calibration = read_json(calibration_path)
    k = np.array(calibration['K'], dtype=float)
    width, height = camera['image_resolution_wh']
    old_width, old_height = camera['calibration_resolution_wh']
    sx, sy = width/old_width, height/old_height
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2]+.5)*sx-.5
    k[1, 2] = (k[1, 2]+.5)*sy-.5
    compiled = directory/'plant.mjb'
    model = (mujoco.MjModel.from_binary_path(str(compiled)) if compiled.is_file()
             else mujoco.MjModel.from_xml_path(str(directory/'plant.xml')))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('initial').id)
    mujoco.mj_forward(model, data)
    scene = BuiltScene(directory/'plant.xml', scene_config, model, data, k, (width, height),
                       (width+160, height+160), bool(calibration['calibration_accepted']))
    output.mkdir(parents=True, exist_ok=False)
    annotation = video_annotation(directory)
    recorder = SceneRecorder(scene, output, video=video, sensors=video, operator_annotation=annotation)
    previous = -1.
    try:
        for sample in rows(directory/'trajectory.jsonl'):
            if sample['time_s'] <= previous:
                raise ValueError('Recorded trajectory time is not strictly increasing')
            previous = sample['time_s']
            data.time = sample['time_s']
            data.qpos[:] = sample['qpos']
            data.qvel[:] = sample['qvel']
            data.ctrl[:] = sample['ctrl']
            mujoco.mj_forward(model, data)
            recorder.step()
    finally:
        recorder.close()
    (output/'replay.json').write_text(json.dumps(dict(source=str(directory),
        scope='Offline recorded-state rendering and sampled contact re-evaluation; no controller or physics stepping',
        model_source='plant.mjb' if compiled.is_file() else 'plant.xml',
        operator_annotation=annotation,
        complete_trajectory=True, last_time_s=previous, video=video), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    directory = args.run.resolve()
    output = (args.output or directory/'analysis').resolve()
    result = analyze(directory, output)
    if args.replay or args.video:
        replay(directory, output/'replay', video=args.video)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
