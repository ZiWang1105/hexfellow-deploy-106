"""Robot imports and launch admission must not depend on an agent application."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_robot_stack_imports_without_application_and_owns_sources(tmp_path):
    code = '''
import importlib.abc
import inspect
import pathlib
import sys
class BlockApplication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mobile_manipulator' or fullname.startswith('mobile_manipulator.'):
            raise AssertionError('Robot runtime imported its consuming application')
sys.meta_path.insert(0, BlockApplication())
from hexfellow_deploy.robot.upstream import PROJECT_ROOT, UMI_ROOT, load_umi
from hexfellow_deploy.actuation import deployment
from hexfellow_deploy.workspace import build_shared_executor
root = pathlib.Path(sys.argv[1])
assert PROJECT_ROOT == root and UMI_ROOT == root/'third_party/UMI_Hardware'
_, _, _, metadata = deployment(root, pathlib.Path('configs/hardware/desktop.yaml'))
executor = build_shared_executor(metadata, .02)
assert pathlib.Path(inspect.getfile(type(executor))).is_relative_to(root/'src/hexfellow_deploy')
assert pathlib.Path(inspect.getfile(type(executor.ik.core))).is_relative_to(UMI_ROOT)
assert not any(n == 'mobile_manipulator' or n.startswith('mobile_manipulator.') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(ROOT)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr


def test_viewer_requires_display_before_starting_components(tmp_path):
    output = tmp_path/'not_started'
    env = {k:v for k,v in os.environ.items() if k != 'DISPLAY'}
    result = subprocess.run([sys.executable, '-I', '-m', 'hexfellow_deploy.cli',
                             '--interactive-ik', '--no-camera', '--output', str(output)],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'desktop DISPLAY' in result.stderr
    assert not output.exists()
