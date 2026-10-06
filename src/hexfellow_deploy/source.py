"""Resolve the deployment stack's pinned transitive controller source.

Applications supply robot assets, not Python search paths or alternate solvers.
Loading an unrelated policy_sim package or compiled kernel fails closed.
"""
from functools import lru_cache
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys


UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / 'third_party/policy-sim'
PINNED_SOURCE_REVISION = '9398b61972bf983c7ab5609591b7eb4c95c6357d'


@lru_cache(maxsize=1)
def verify_source():
    source = UPSTREAM_ROOT / 'src'
    if not (source / 'policy_sim/control/ik_core.py').is_file():
        raise RuntimeError('Initialize hexfellow-deploy/third_party/policy-sim recursively')
    revision = subprocess.check_output(
        ['git', '-C', str(UPSTREAM_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PINNED_SOURCE_REVISION:
        raise RuntimeError('Controller source revision differs from the pinned deployment revision')
    return source


def upstream(module):
    source = verify_source()
    if 'policy_sim' not in sys.modules:
        sys.path.insert(0, str(source))
    package = importlib.import_module('policy_sim')
    if Path(package.__file__).resolve().parent != (source / 'policy_sim').resolve():
        raise RuntimeError('policy_sim was imported from outside hexfellow-deploy')
    try:
        return importlib.import_module(f'policy_sim.{module}')
    except ImportError as error:
        raise RuntimeError('Install the pinned hexfellow-deploy controller dependencies and native extension') from error


def verify_native_extension():
    native = importlib.import_module('_policy_sim_collision_kernel')
    installed = Path(native.__file__).resolve()
    if installed.parent == verify_source().resolve():
        return
    provenance = Path(sys.prefix) / 'policy_native_provenance.json'
    if not provenance.is_file():
        raise RuntimeError('External native extension requires verified build provenance')
    record = json.loads(provenance.read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    required = ('setup.py', 'pyproject.toml', 'src/policy_sim/control/collision_kernel.cpp')
    if (record.get('policy_source_pin') != PINNED_SOURCE_REVISION
            or record.get('native_abi') != 3 or native.KERNEL_ABI_VERSION != 3
            or record.get('trajectory_version') != native.TRAJECTORY_KERNEL_VERSION
            or Path(record.get('policy_source', '')).resolve() != UPSTREAM_ROOT.resolve()
            or Path(record.get('installed_extension', '')).resolve() != installed
            or not installed.is_relative_to(Path(sys.prefix).resolve())
            or record.get('installed_extension_sha256') != sha(installed)
            or any(record.get('source_hashes', {}).get(name) != sha(UPSTREAM_ROOT/name) for name in required)):
        raise RuntimeError('Native extension provenance does not match the pinned source and environment')
