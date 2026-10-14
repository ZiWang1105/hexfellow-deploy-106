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
        if fullname.split('.')[0] in {'mobile_manipulator', 'policy_sim', 'deploy_anything', '_policy_sim_collision_kernel'}:
            raise AssertionError('Removed dependency imported: '+fullname)
sys.meta_path.insert(0, BlockApplication())
from hexfellow_deploy.robot.upstream import PROJECT_ROOT, UMI_ROOT, load_umi
from hexfellow_deploy.actuation import deployment
from hexfellow_deploy.workspace import build_shared_executor
root = pathlib.Path(sys.argv[1])
assert PROJECT_ROOT == root and UMI_ROOT == root/'robot_resources'
_, _, _, metadata = deployment(root, pathlib.Path('configs/hardware/desktop.yaml'))
executor = build_shared_executor(metadata, .02)
assert pathlib.Path(inspect.getfile(type(executor))).is_relative_to(root/'src/hexfellow_deploy')
assert pathlib.Path(inspect.getfile(type(executor.ik.core))).is_relative_to(root/'src/hexfellow_deploy/robotics')
assert not ({n.split('.')[0] for n in sys.modules} & {'mobile_manipulator', 'policy_sim', 'deploy_anything', '_policy_sim_collision_kernel'})
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


def test_no_external_robot_package_or_git_submodule_contract():
    import ast
    assert not (ROOT/'.gitmodules').exists()
    assert not (ROOT/'third_party/UMI_Hardware').exists()
    assert not (ROOT/'third_party/deploy-anything').exists()
    forbidden = {'policy_sim', 'deploy_anything', 'mobile_manipulator', '_policy_sim_collision_kernel'}
    for path in (ROOT/'src/hexfellow_deploy').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or '']
            else:
                continue
            assert not any(module.split('.')[0] in forbidden for module in modules), path
    setup = (ROOT/'scripts/setup_sim.sh').read_text()
    assert not any(name in setup for name in ('policy-sim', 'deploy-anything', 'git submodule'))


def test_calibration_resources_resolve_inside_owned_repository():
    for relative in ('calibration/assets', 'calibration/urdf', 'calibration/config/control'):
        path = ROOT/relative
        assert path.is_dir(), relative
        assert path.resolve().is_relative_to(ROOT/'robot_resources'), relative


def test_extracted_robot_resources_preserve_recorded_source_bytes():
    import hashlib
    import json
    provenance = json.loads((ROOT/'docs/OWNED_RUNTIME_PROVENANCE.json').read_text())
    resources = [row for row in provenance['files'] if row['destination'].startswith('robot_resources/')]
    assert resources
    for row in resources:
        assert hashlib.sha256((ROOT/row['destination']).read_bytes()).hexdigest() == row['source_sha256'], row['destination']
