"""Public command rejection must happen before any native SDK import."""
from pathlib import Path
import subprocess
import sys
import json
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def guarded_run(script, *arguments):
    code = '''
import importlib.abc
import runpy
import sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'hex_device', 'pyzed', 'pyrealsense2'}:
            raise RuntimeError('Native SDK import attempted before command admission')
sys.meta_path.insert(0, BlockNative())
script = sys.argv[1]
sys.argv = sys.argv[1:]
if script == 'hexfellow_deploy.cli':
    runpy.run_module(script, run_name='__main__')
else:
    runpy.run_path(script, run_name='__main__')
'''
    target = script if script == 'hexfellow_deploy.cli' else str(ROOT/'scripts'/script)
    return subprocess.run([sys.executable, '-c', code, target, *arguments],
                          cwd=ROOT, text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize('script', ['run_agent.py', 'run_real_agent.py'])
def test_public_help_uses_shared_runtime_without_native_import(script):
    result = guarded_run(script, '--help')
    assert result.returncode == 0, result.stderr
    assert 'Shared DA/PinkIK/MIT deployment' in result.stdout
    assert '--runtime-config' in result.stdout


@pytest.mark.parametrize('flags', [[], ['--execute'], ['--confirm-onsite-ready']])
def test_real_launcher_requires_explicit_admission_before_import(tmp_path, flags):
    output = tmp_path/'never_created'
    result = guarded_run('run_real_agent.py', '--hold-only', '--output', str(output), *flags)
    assert result.returncode == 2
    assert 'requires --execute --confirm-onsite-ready' in result.stderr
    assert 'Native SDK import' not in result.stderr
    assert not output.exists()


def test_simulation_rejects_execution_flags_before_import(tmp_path):
    result = guarded_run('run_agent.py', '--execute', '--hold-only', '--output', str(tmp_path/'unused'))
    assert result.returncode == 2
    assert 'Simulation does not accept --execute' in result.stderr


def test_empty_instruction_is_rejected_before_runtime(tmp_path):
    instruction = tmp_path/'task.txt'
    instruction.write_text('  \n')
    result = guarded_run('run_agent.py', '--instruction', str(instruction), '--output', str(tmp_path/'unused'))
    assert result.returncode == 2
    assert 'Task instruction must not be empty' in result.stderr


def test_yaml_override_cannot_supply_hardware_authorization(tmp_path):
    output = tmp_path/'rejected'
    result = guarded_run('hexfellow_deploy.cli', '--workspace', str(ROOT), '--output', str(output),
                         '--set', 'interface.backend.backend=hardware',
                         '--set', 'interface.backend.execute=true')
    assert result.returncode != 0
    assert 'Hardware requires --execute --confirm-onsite-ready' in result.stderr
    assert 'Native SDK import' not in result.stderr
    assert not (output/'topics.json').exists()


@pytest.mark.parametrize('backend', ['simulation','hardware'])
def test_build_uses_deployment_relative_base_config_for_policy(monkeypatch, tmp_path, backend):
    import hydra.utils
    from hexfellow_deploy import cli, manager
    import deploy_anything.utils.config_utils as config_utils
    deployment = tmp_path/'deployment.yaml'
    deployment.write_text('base_motion: controls/base.yaml\n')
    monkeypatch.setenv('HEXFELLOW_WORKSPACE', str(ROOT))
    monkeypatch.setenv('HEXFELLOW_OUTPUT', str(tmp_path/'unused'))
    monkeypatch.setenv('HEXFELLOW_TOPIC_PREFIX', 'unused_')
    monkeypatch.setattr(config_utils, 'build_topic_registry', lambda spec: {})
    monkeypatch.setattr(config_utils, 'register_topic_resolver', lambda topics: None)
    monkeypatch.setattr(hydra.utils, 'instantiate', lambda spec: SimpleNamespace(
        name=spec.name, base_config=spec.get('base_config')))
    monkeypatch.setattr(manager, 'InterfaceManager', lambda **kwargs: SimpleNamespace(**kwargs))
    config_path = ROOT/'third_party/hexfellow-deploy/configs/simulation.yaml'
    config, _, selected = cli.build(config_path, [f'deployment_config={deployment}',
        f'interface.backend.backend={backend}',f'interface.backend.execute={str(backend=="hardware").lower()}'],
        prompt='Diagnostic', execute_authorized=backend=='hardware')
    assert selected.interface['agent'].base_config == str(tmp_path/'controls/base.yaml')
    assert ('wrist_images' in config.topics) == (backend=='hardware')
    assert ('wrist_camera' in config.interface) == (backend=='hardware')
    if backend=='hardware':
        assert config.topics.wrist_images._target_ == 'hexfellow_deploy.topics.RGBTopic'
        assert config.interface.wrist_camera.serial == '243222074562'
        assert config.interface.wrist_camera.calibration.endswith('realsense_d435i_right_candidate_20261006.json')
        assert 'feedback' in config.interface.wrist_camera
        assert 'wrist_images' in config.interface.agent and 'wrist_images' in config.interface.dashboard


@pytest.mark.parametrize('missing', ['control', 'backend', 'both'])
def test_final_report_survives_abrupt_component_exit(monkeypatch, tmp_path, missing):
    from omegaconf import OmegaConf
    from hexfellow_deploy import cli, provenance
    output = tmp_path/'crashed'
    root_fault = 'hexfellow_backend: UnexpectedExit: process exited with code -9'
    def run_all():
        (output/'component_health.json').write_text(json.dumps(dict(fault=root_fault)))
        if missing not in ('control', 'both'):
            (output/'control_final.json').write_text(json.dumps(dict(phase='fault', fault='Actual feedback is stale')))
        if missing not in ('backend', 'both'):
            (output/'backend_final.json').write_text(json.dumps(dict(phase='closed', backend='simulation')))
    def build(*args, **kwargs):
        return OmegaConf.create(dict(interface=dict(backend=dict(backend='simulation')))), {}, SimpleNamespace(run_all=run_all)
    monkeypatch.setattr(cli, 'build', build)
    monkeypatch.setattr(cli.mp, 'Process', lambda **kwargs: SimpleNamespace(start=lambda:None,
        join=lambda *args:None, is_alive=lambda:False))
    monkeypatch.setattr(provenance, 'collect', lambda workspace: {})
    monkeypatch.setattr(sys, 'argv', ['shared', '--workspace', str(ROOT), '--output', str(output), '--no-camera'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    report = json.loads((output/'run.json').read_text())
    assert report['status'] == 'error' and report['fault'] == root_fault
    assert report['shutdown_reports_present'] is False
    assert len(report['report_errors']) == (2 if missing == 'both' else 1)
    assert all('shutdown is unconfirmed' in item for item in report['report_errors'])


def test_corrupt_final_report_and_partial_decision_preserve_original_failure(tmp_path):
    from hexfellow_deploy.cli import read_final_reports
    (tmp_path/'component_health.json').write_text(json.dumps(dict(fault='camera: capture stopped')))
    (tmp_path/'control_final.json').write_text('{')
    (tmp_path/'backend_final.json').write_text(json.dumps(dict(backend='hardware', phase='closed')))
    (tmp_path/'decisions.jsonl').write_text('{"policy_calls": 3}\n{"policy_calls":')
    final, backend, health, termination, decision, errors = read_final_reports(tmp_path, 'hardware')
    assert not final and health['fault'] == 'camera: capture stopped'
    assert backend['backend'] == 'hardware'
    assert decision['policy_calls'] == 3
    assert len(errors) == 2


def test_catalog_launch_forwards_explicit_budget_and_never_legacy(monkeypatch, tmp_path):
    from hexfellow_deploy import cli, entrypoints
    captured = []
    monkeypatch.setattr(cli, 'main', lambda: captured.extend(sys.argv[1:]))
    entrypoints.main('simulation', ['--task', 'sort_cubes', '--seed', '7',
        '--respect-policy-limits', '--output', str(tmp_path/'new')])
    assert captured[captured.index('--task')+1] == 'sort_cubes'
    assert captured[captured.index('--seed')+1] == '7'
    assert '--respect-policy-limits' in captured
    assert '--legacy-runtime' not in captured
    assert '--prompt' not in captured
    assert not (tmp_path/'new').exists()


def test_legacy_controller_cannot_silently_replace_shared_profile(tmp_path):
    result = guarded_run('run_agent.py', '--task', 'sort_cubes',
        '--controller', 'configs/controller/umi_pink_mobile.yaml', '--output', str(tmp_path/'unused'))
    assert result.returncode == 2
    assert 'must match' in result.stderr
    assert 'shared deployment YAML' in result.stderr
    assert not (tmp_path/'unused').exists()


def test_catalog_instruction_is_only_task_input_to_policy_and_scoring_is_offline(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from hexfellow_deploy import cli, provenance
    from mobile_manipulator.eval.benchmark import load_catalog
    from mobile_manipulator.eval import scoring
    output = tmp_path/'episode'
    task = load_catalog()[0]['sort_cubes']
    captured = {}
    def run_all():
        captured['closed'] = True
        (output/'control_final.json').write_text(json.dumps(dict(ticks=2,
            final_hold=True, diagnostic=dict(ik_status='HOLD',base_stopped=True))))
        (output/'backend_final.json').write_text(json.dumps(dict(backend='simulation')))
        (output/'termination.json').write_text(json.dumps(dict(reason='completed_budget')))
    def build(config_path, overrides, **kwargs):
        captured.update(kwargs)
        scene = [v.split('=',1)[1] for v in overrides if v.startswith('interface.backend.scene=')][-1]
        captured['scene'] = Path(scene)
        return OmegaConf.create(dict(interface=dict(backend=dict(backend='simulation')),
            policy=dict(agent_config=str(ROOT/'configs/agent/rgbd_full.yaml')))), {}, SimpleNamespace(run_all=run_all)
    def evaluate(directory, selected, **kwargs):
        assert captured['closed']
        assert json.loads((directory/'run.json').read_text())['status'] == 'completed_budget'
        assert selected.name == 'sort_cubes'
        assert kwargs == dict(model_filename='plant.xml')
        return dict(accepted=False, task=selected.name)
    monkeypatch.setattr(cli, 'build', build)
    monkeypatch.setattr(cli.mp, 'Process', lambda **kwargs: SimpleNamespace(start=lambda:None,
        join=lambda *args:None, is_alive=lambda:False))
    monkeypatch.setattr(provenance, 'collect', lambda workspace: {})
    monkeypatch.setattr(scoring, 'evaluate_run', evaluate)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', ['shared', '--workspace', str(ROOT), '--output', str(output),
        '--task', 'sort_cubes', '--seed', '7', '--respect-policy-limits'])
    cli.main()
    assert captured['prompt'] == task.instruction
    assert captured['respect_policy_limits'] is True
    assert captured['scene'] == output/'scene.yaml'
    assert captured['scene'].is_file()
    assert 'evaluation' not in captured
    assert 'task' not in captured
    invocation = json.loads((output/'invocation.json').read_text())
    assert invocation['duration_source'] == 'agent.max_episode_s_wall_clock'
    assert invocation['seed'] == 7
    assert json.loads((output/'run.json').read_text())['evaluation']['accepted'] is False
