"""Lifecycle diagnostics use real DA processes and SHM, without any devices."""
import json
import multiprocessing as mp
import os
import threading
import time
import uuid
from urllib.request import urlopen

import numpy as np
import pytest
from deploy_anything.components.interface.base_interface import DABaseInterface

from hexfellow_deploy.dashboard import DashboardComponent
from hexfellow_deploy.manager import InterfaceManager
from hexfellow_deploy.topics import JsonTopic


class Probe(DABaseInterface):
    def __init__(self, name, behavior):
        super().__init__(name, [], time_budget=.001)
        self.behavior = behavior
        self.ticks = mp.Value('i', 0)

    def initialize(self):
        self._initialized.value = True
        return True

    def clean_up(self):
        self._cleaned_up.value = True
        return True

    def run(self):
        self._ready.value = True
        if self.behavior == 'exception':
            raise ValueError('Camera renderer failed')
        if self.behavior == 'crash':
            os._exit(7)
        if self.behavior == 'return':
            return
        if self.behavior == 'stop':
            self.signal_stop()
            return
        while not self.stopping:
            with self.ticks.get_lock():
                self.ticks.value += 1
            # Deliberately exceeds the declared budget. Lifecycle reporting does
            # not turn an ordinary slow cycle into a component failure.
            time.sleep(.02)


def start_manager(monkeypatch, tmp_path, behavior):
    import deploy_anything.manager as da_manager
    monkeypatch.setattr(da_manager, 'set_rt', lambda **kwargs: None)
    monkeypatch.setattr(da_manager, 'set_affinity', lambda affinity: None)
    monkeypatch.setenv('HEXFELLOW_OUTPUT', str(tmp_path))
    health = JsonTopic('health_'+uuid.uuid4().hex)
    camera, control = Probe('camera', behavior), Probe('control', 'clock')
    manager = InterfaceManager(topics={'health': health}, interface={'camera': camera, 'control': control},
        preprocess={}, input={}, inference={}, output={}, loop_time=.02, shutdown_timeout_s=.5)
    for process in manager._processes:
        process.start()
    return manager, health, camera, control


@pytest.mark.parametrize('behavior, detail', [
    ('exception', 'ValueError: Camera renderer failed'),
    ('crash', 'UnexpectedExit: process exited with code 7'),
    ('return', 'UnexpectedExit: component returned without a stop request'),
])
def test_component_failure_visible_without_blocking_control(monkeypatch, tmp_path, behavior, detail):
    manager, health, camera, control = start_manager(monkeypatch, tmp_path, behavior)
    try:
        until = time.monotonic()+3.
        while time.monotonic() < until:
            snapshot = manager._poll_health()
            if detail in (snapshot['fault'] or ''):
                break
            time.sleep(.01)
        assert snapshot['fault'] == f'camera: {detail}'
        assert health.get(timeout=.01)['fault'] == snapshot['fault']
        assert json.loads((tmp_path/'component_health.json').read_text())['fault'] == snapshot['fault']
        before = control.ticks.value
        time.sleep(.1)
        assert control.ticks.value > before
        assert not control.stopping
        if behavior == 'exception':
            assert 'raise ValueError' in snapshot['components']['camera']['traceback']
    finally:
        manager._shutdown()
    assert all(process.exitcode is not None for process in manager._processes)


@pytest.mark.parametrize('behavior', ['clock', 'stop'])
def test_budget_overrun_and_requested_stop_are_not_process_faults(monkeypatch, tmp_path, behavior):
    manager, health, camera, control = start_manager(monkeypatch, tmp_path, behavior)
    try:
        until = time.monotonic()+.15
        while time.monotonic() < until:
            assert manager._poll_health()['fault'] is None
            time.sleep(.01)
        assert control.ticks.value > 1
    finally:
        manager._shutdown()
    assert json.loads((tmp_path/'component_health.json').read_text())['fault'] is None


def test_dashboard_serves_fault_and_registered_depth(monkeypatch):
    import hexfellow_deploy.dashboard as module
    servers = []
    original = module.ThreadingHTTPServer
    def server(*args, **kwargs):
        result = original(*args, **kwargs)
        servers.append(result)
        return result
    monkeypatch.setattr(module, 'ThreadingHTTPServer', server)
    class Topic:
        name = 'test'
        def __init__(self, value):
            self.value = value
        def get(self, **kwargs):
            return self.value
    health = dict(fault='camera: ValueError: camera failed', components={'camera': {'phase': 'failed'}})
    image = dict(rgb=np.zeros((24,32,3),dtype=np.uint8), depth_m=np.ones((24,32),dtype=np.float32))
    dashboard = DashboardComponent('dashboard', Topic({'phase':'running'}), Topic({'phase':'thinking'}),
                                   Topic(image), health=Topic(health), port=0)
    dashboard.initialize()
    thread = threading.Thread(target=dashboard.run, daemon=True)
    thread.start()
    try:
        until = time.monotonic()+2.
        while not dashboard.ready and time.monotonic() < until:
            time.sleep(.005)
        assert dashboard.ready
        base = f'http://127.0.0.1:{servers[0].server_port}'
        with urlopen(base+'/status', timeout=1.) as response:
            assert json.load(response)['health'] == health
        with urlopen(base+'/', timeout=1.) as response:
            page = response.read()
            assert b'Component health' in page and b's.health?.fault' in page and b'/depth.jpg' in page
        with urlopen(base+'/depth.jpg', timeout=1.) as response:
            assert response.headers['Content-Type'] == 'image/jpeg'
            assert response.read().startswith(b'\xff\xd8')
    finally:
        dashboard.signal_stop()
        thread.join(2.)
        dashboard.clean_up()
    assert not thread.is_alive()
