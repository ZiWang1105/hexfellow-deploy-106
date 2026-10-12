"""Resolve the Git submodule, never an unrelated sibling policy_sim checkout."""
import importlib
import hashlib
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UMI_ROOT = PROJECT_ROOT / "third_party/UMI_Hardware"


def load_umi():
    if not (UMI_ROOT / "src/policy_sim/control/ik_core.py").is_file():
        raise RuntimeError("Initialize UMI: git submodule update --init --recursive")
    source = str(UMI_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    package = importlib.import_module("policy_sim")
    if Path(package.__file__).resolve() != UMI_ROOT / "src/policy_sim/__init__.py":
        raise RuntimeError("policy_sim was imported from outside third_party/UMI_Hardware")
    return UMI_ROOT


def umi_revision():
    return subprocess.check_output(
        ["git", "-C", str(UMI_ROOT), "rev-parse", "HEAD"], text=True).strip()


def gitlink_revision():
    entry = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "--stage", "third_party/UMI_Hardware"],
        text=True).split()
    if len(entry) != 4 or entry[0] != "160000":
        raise RuntimeError("third_party/UMI_Hardware must be a staged Git submodule")
    return entry[1]


def umi_provenance():
    """Record editable source/configuration as well as the compiled extension."""
    load_umi()
    import _policy_sim_collision_kernel as native

    paths = sorted((UMI_ROOT / "src/policy_sim").rglob("*.py"))
    paths += sorted(p for p in (UMI_ROOT / "config/control").rglob("*")
                    if p.suffix in {".yaml", ".json"})
    paths += [UMI_ROOT / "urdf/maver_x4_gp80.urdf", UMI_ROOT / "setup.py",
              UMI_ROOT / "src/policy_sim/control/collision_kernel.cpp"]
    return {
        "commit": umi_revision(), "gitlink": gitlink_revision(),
        "worktree_status": subprocess.check_output(
            ["git", "-C", str(UMI_ROOT), "status", "--porcelain"], text=True),
        "source_sha256": {str(p.relative_to(UMI_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        "native_kernel_abi": native.KERNEL_ABI_VERSION,
        "native_kernel_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
    }
