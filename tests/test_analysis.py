"""Offline evidence rejects incomplete runs and incomplete pickup retention."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('mit_run_analyzer',
    Path(__file__).resolve().parents[1]/'scripts/analyze_mit_run.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


@pytest.mark.parametrize('missing', ['control_final.json', 'backend_final.json'])
def test_analyzer_and_replay_refuse_live_run_before_reading_privileged_records(tmp_path, missing):
    for name in ('control_final.json', 'backend_final.json'):
        if name != missing:
            (tmp_path/name).write_text('{}')
    # Any attempt to parse these files before owner completion would fail with
    # JSONDecodeError rather than the intended completion guard.
    (tmp_path/'evaluation.jsonl').write_text('unreadable live evaluator record')
    (tmp_path/'trajectory.jsonl').write_text('unreadable live trajectory record')
    for operation in (analyzer.analyze, analyzer.replay):
        with pytest.raises(ValueError, match='both owner final records are required'):
            operation(tmp_path, tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('failure', [None, 'early_stop', 'short_hold', 'no_stationary_interval',
    'braking', 'moving_base', 'control_fault', 'backend_fault', 'shutdown_error', 'wrong_body',
    'height', 'retention', 'jaw_contact', 'support', 'robot_contact',
    'dense_contact', 'dense_coverage', 'dense_contact_loss'])
def test_pickup_requires_completed_fixed_budget_and_final_supported_evidence(tmp_path, failure):
    control = dict(final_hold=True, diagnostic=dict(ik_status='HOLD', base_stopped=True), fault=None)
    backend = dict(fault=None, shutdown_error=None)
    termination = dict(reason='completed_budget', settle_s=14., stationary_since_s=100.)
    run = dict(status='completed_budget', evaluation=dict(evaluated_body='target_body'))
    obj = dict(name='target_body', height_gain_m=.2, retained_s=14.,
               opposing_jaw_contacts=True, grasping_arms=['right'], support_contact=False)
    row = dict(time_s=120., objects=[obj], robot_static_contact_frames=0)
    if failure == 'early_stop': termination['reason'] = 'operator_stop'
    if failure == 'short_hold': termination['settle_s'] = 2.
    if failure == 'no_stationary_interval': termination['stationary_since_s'] = None
    if failure == 'braking': control['diagnostic']['ik_status'] = 'BRAKING'
    if failure == 'moving_base': control['diagnostic']['base_stopped'] = False
    if failure == 'control_fault': control['fault'] = 'controller failure after grasp'
    if failure == 'backend_fault': backend['fault'] = 'publication failed'
    if failure == 'shutdown_error': backend['shutdown_error'] = 'stop unconfirmed'
    if failure == 'wrong_body': run['evaluation']['evaluated_body'] = 'other_body'
    if failure == 'height': obj['height_gain_m'] = .09
    if failure == 'retention': obj['retained_s'] = 9.
    if failure == 'jaw_contact': obj['opposing_jaw_contacts'] = False
    if failure == 'support': obj['support_contact'] = True
    if failure == 'robot_contact': row['robot_static_contact_frames'] = 1
    for name, value in [('control_final.json', control), ('backend_final.json', backend),
                        ('termination.json', termination), ('run.json', run)]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'evaluation.jsonl').write_text(json.dumps(row)+'\n')
    if failure and failure.startswith('dense_'):
        dense = dict(time_s=120., objects=[dict(obj)], robot_static_contact_steps=0,
                     samples=60000, complete=True)
        if failure == 'dense_contact': dense['robot_static_contact_steps'] = 1
        if failure == 'dense_coverage': dense['complete'] = False
        if failure == 'dense_contact_loss': dense['objects'][0]['retained_s'] = .002
        (tmp_path/'physics_evaluation.json').write_text(json.dumps(dense))
    result = analyzer.pickup_summary(tmp_path, control)
    assert result['passed'] == (failure is None)


def test_old_target_inference_requires_position_and_orientation_agreement():
    pose = dict(position_m=[.4, .2, .8], quaternion_xyzw=[0., 0., 0., 1.])
    actual = {side: pose for side in ('left', 'right')}
    proposed = {side: dict(position_m=[.5, .2, .8], quaternion_xyzw=[0., 0., 0., 1.])
                for side in ('left', 'right')}
    row = dict(diagnostic=dict(arm_errors={side: dict(position_m=.1, orientation_rad=0.)
                                          for side in ('left', 'right')}))
    assert analyzer.target_for(row, actual, proposed)[0] is proposed
    proposed['right']['quaternion_xyzw'] = [0., 0., 1., 0.]
    assert analyzer.target_for(row, actual, proposed) == (None, None)


def test_video_labels_retained_grasp_as_failed_when_entire_run_faulted(tmp_path):
    values = {'control_final.json': dict(phase='fault', fault='RuntimeError: IK failed: NoSolutionFound'),
              'backend_final.json': {}, 'run.json': dict(status='error',
                  evaluation=dict(retained_pickup=True, accepted=False),
                  termination=dict(reason='control_fault'))}
    for name, record in values.items():
        (tmp_path/name).write_text(json.dumps(record))
    lines = analyzer.video_annotation(tmp_path)
    assert lines[0] == 'RUN FAILED: NoSolutionFound'
    assert any('Incomplete budget' in line and 'No verified final HOLD' in line for line in lines)
    assert any('Native hardware not qualified' in line for line in lines)


def test_recorded_mit_analysis_uses_full_frame_and_source_encoders():
    row = dict(mit=dict(joint_names=['a', 'b'], position=[.2, -.3], velocity=[.1, .2],
                        torque_ff=[-.1, .2], kp=[5., 7.], kd=[2., 3.]),
               measured=dict(joint_positions=dict(a=.1, b=-.2), joint_velocities=dict(a=0., b=.1)),
               reference=dict(joint_positions=dict(a=.2, b=-.3)))
    result = analyzer.mit_summary([row], [.5, 2.])
    assert result['complete_recording']
    assert result['maximum_reference_position_mismatch_rad'] == 0.
    assert result['maximum_absolute_predicted_torque_nm'] == pytest.approx([.6, .2])
    assert result['predicted_effort_limit_exceedances'] == 1
    assert not analyzer.mit_summary([row, {}], [.5, 2.])['complete_recording']
