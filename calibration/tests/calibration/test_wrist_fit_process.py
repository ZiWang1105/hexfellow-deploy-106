"""Fit subprocess completion/cancellation cannot command or strand a robot."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = load(ROOT/'tools/camera/calibrate_robot.py', 'calibration_process_driver')
FIXTURES = load(ROOT/'tests/calibration/test_wrist_handeye.py', 'wrist_process_fixtures')


def test_real_offline_fit_is_saved_before_return(tmp_path):
    dataset = tmp_path/'dataset.json'
    dataset.write_text(json.dumps(FIXTURES.image_dataset()))
    output = tmp_path/'wrist-fit.json'
    result = DRIVER.run_wrist_fit(dataset, output, cancelled=lambda: False, python=sys.executable)
    assert output.exists() and output.with_suffix('.log').exists()
    saved = json.loads(output.read_text())
    assert result['T_wrist_from_camera'] == saved['T_wrist_from_camera']
    assert result['converged'] is True
    assert result['validation_reprojection']['rms_px'] < .001
    assert result['fit_execution']['isolated_process'] is True
    assert result['fit_execution']['hardware_commands_sent'] == 0


class RunningProcess:
    """A child that stays running until its parent explicitly terminates it."""
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.wait_timeouts = []

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.returncode is not None:
            return self.returncode
        raise subprocess.TimeoutExpired('offline solver', timeout)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def test_pause_cancels_child_and_does_not_return_a_solution(tmp_path, monkeypatch):
    child = RunningProcess()
    calls = 0
    launched = {}

    def launch(command, **kwargs):
        launched.update(command=command, **kwargs)
        return child

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    monkeypatch.setattr(DRIVER.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='cancelled'):
        DRIVER.run_wrist_fit(tmp_path/'dataset.json', tmp_path/'fit.json', cancelled=cancelled)
    assert child.terminated
    assert child.wait_timeouts[0] <= .2
    assert child.wait_timeouts[-1] == 2.
    assert all(launched['env'][name] == '1' for name in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    assert not (tmp_path/'fit.json').exists()


def test_timeout_terminates_child_and_preexisting_fit_is_preserved(tmp_path, monkeypatch):
    child = RunningProcess()

    def wait(timeout):
        if child.returncode is not None:
            return child.returncode
        time.sleep(.02)
        raise subprocess.TimeoutExpired('offline solver', timeout)

    child.wait = wait
    monkeypatch.setattr(DRIVER.subprocess, 'Popen', lambda *args, **kwargs: child)
    with pytest.raises(TimeoutError, match='exceeded'):
        DRIVER.run_wrist_fit(tmp_path/'dataset.json', tmp_path/'fit.json',
                             cancelled=lambda: False, timeout_s=.01)
    assert child.terminated
    previous = tmp_path/'previous.json'
    previous.write_text('important old fit')
    with pytest.raises(ValueError, match='already exists'):
        DRIVER.run_wrist_fit(tmp_path/'dataset.json', previous, cancelled=lambda: False)
    assert previous.read_text() == 'important old fit'


def test_nonconverged_cli_result_is_saved_and_reported(tmp_path, monkeypatch):
    dataset, output = tmp_path/'dataset.json', tmp_path/'fit.json'
    dataset.write_text('{}')
    provisional = dict(converged=False, T_wrist_from_camera=[], T_base_from_board=[], validation={},
                       source_dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    child = RunningProcess()
    child.returncode = 1

    def launch(command, **kwargs):
        output.write_text(json.dumps(provisional))
        return child

    monkeypatch.setattr(DRIVER.subprocess, 'Popen', launch)
    result = DRIVER.run_wrist_fit(dataset, output, cancelled=lambda: False)
    assert result['converged'] is False
    assert result['fit_execution']['exit_code'] == 1
    assert json.loads(output.read_text()) == provisional
