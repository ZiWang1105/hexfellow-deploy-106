"""Benchmark launchers must use the shared stack and explicit finite budgets."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def test_benchmark_keeps_episode_directory_fresh_and_forwards_shared_arguments(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('shared_benchmark_launcher', ROOT/'scripts/run_benchmark.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index('--output')+1])
        config = Path(command[command.index('--agent')+1])
        assert not output.exists()
        assert config.is_file()
        assert config.parent.name == 'inputs'
        output.mkdir()
        (output/'run.json').write_text(json.dumps(dict(status='completed_budget',
            evaluation=dict(accepted=False), policy_calls=1, cost_usd=.1,
            termination=dict(budget_reason='max_calls'))))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', execute)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', ['run_benchmark.py', '--tasks', 'sort_cubes',
        '--seeds', '0', '--max-total-cost-usd', '1', '--output', str(tmp_path/'benchmark')])
    launcher.main()
    assert len(commands) == 1
    assert '--respect-policy-limits' in commands[0]
    assert '--deployment-config' in commands[0]
    assert '--controller' not in commands[0]
    assert '--base-motion' not in commands[0]
    assert '--legacy-runtime' not in commands[0]
    report = json.loads((tmp_path/'benchmark/benchmark.json').read_text())
    assert report['summary']['runtime_errors'] == 0
    assert report['episodes'][0]['termination']['budget_reason'] == 'max_calls'
