"""Camera-only subprocesses can reuse existing vendor Python/SDK environments."""
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys


def resolve_camera_config(config, directory):
    config = dict(config)
    for name in ('python', 'sdk_root'):
        if config.get(name):
            path = Path(config[name]).expanduser()
            if not path.is_absolute():
                path = Path(directory) / path
            # Preserve .venv/bin/python: resolving its symlink loses the venv.
            config[name] = os.path.abspath(path)
    return config


def camera_environment(config):
    env = os.environ.copy()
    source = str(Path(__file__).absolute().parents[2])
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (source, env.get('PYTHONPATH'))))
    if config.get('sdk_root'):
        sdk = str(Path(config['sdk_root']).expanduser())
        env['ZED_DIR'] = sdk
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (
            f'{sdk}/lib', f'{sdk}/dependencies/ai_dependencies/lib', env.get('LD_LIBRARY_PATH'))))
    return env


def camera_python(config):
    return str(Path(config['python']).expanduser()) if config.get('python') else sys.executable


def check_camera_sdk(config):
    """Import bindings in the actual capture interpreter; never open a device."""
    module = {'zed': 'pyzed.sl', 'realsense': 'pyrealsense2'}[config['type']]
    try:
        result = subprocess.run([camera_python(config), '-c',
            'import importlib; import numpy; import cv2; importlib.import_module(__import__("sys").argv[1])',
            module], env=camera_environment(config), capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'{module} import timed out in {camera_python(config)}') from error
    if result.returncode:
        raise RuntimeError(f'{module} cannot load in {camera_python(config)}: {result.stderr.strip()}')


class ExternalCameraSource:
    """A local owned pipe carries only camera frames, timestamps and intrinsics."""
    def __init__(self, config, *, synthetic=False):
        self.config, self.synthetic = dict(config), synthetic
        self.process, self.connection = None, None

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise RuntimeError('Camera worker timed out')
        try:
            message = self.connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError('Camera worker disconnected') from error
        if message['kind'] == 'error':
            raise RuntimeError(message['error'])
        return message

    def open(self):
        parent, child = mp.Pipe()
        self.connection = parent
        command = [camera_python(self.config), '-u', '-m',
                   'hexfellow_deploy.hardware.camera_worker', '--fd', str(child.fileno())]
        if self.synthetic:
            command.append('--synthetic')
        try:
            self.process = subprocess.Popen(command, env=camera_environment(self.config),
                                            pass_fds=(child.fileno(),))
            child.close()
            parent.send(self.config)
            if self._receive(15)['kind'] != 'ready':
                raise RuntimeError('Camera worker failed startup')
        except BaseException:
            child.close()
            self.close()
            raise

    def read(self):
        self.connection.send('read')
        message = self._receive(2)
        if message['kind'] != 'frame':
            raise RuntimeError('Unexpected camera worker response')
        return message['frame']

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.connection.send('close')
                    self._receive(1)
                    self.process.wait(timeout=1)
                except (OSError, RuntimeError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=1)
        finally:
            if self.connection is not None:
                self.connection.close()
