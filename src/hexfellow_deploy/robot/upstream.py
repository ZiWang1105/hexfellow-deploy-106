"""Compatibility names for the deployment-owned robot resources and provenance.

No external solver package, Git submodule, or sys.path modification is used.
Existing application adapters retain these names while using the owned runtime.
"""
import hashlib
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = PROJECT_ROOT / "robot_resources"
UMI_ROOT = RESOURCE_ROOT  # Historical application import; resources are owned here.

def load_umi():
    from hexfellow_deploy.robotics import paths
    if paths.PROJECT_ROOT != RESOURCE_ROOT or not (RESOURCE_ROOT / "urdf").is_dir():
        raise RuntimeError("The hexfellow-deploy robot resources are missing")
    return RESOURCE_ROOT

def umi_revision():
    result = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "source-archive"

def gitlink_revision():
    """Legacy scene field: now identifies this repository, not a submodule."""
    return umi_revision()

def umi_provenance():
    """Hash owned code, static resources, and the compiled native kernel."""
    import _hexfellow_collision_kernel as native
    paths = sorted((PROJECT_ROOT / "src/hexfellow_deploy").rglob("*.py"))
    paths += sorted((RESOURCE_ROOT / "config").rglob("*.yaml"))
    paths += sorted((RESOURCE_ROOT / "config").rglob("*.json"))
    paths += [RESOURCE_ROOT / "urdf/maver_x4_gp80.urdf", PROJECT_ROOT / "setup.py",
              PROJECT_ROOT / "src/hexfellow_deploy/robotics/control/collision_kernel.cpp"]
    return {"commit": umi_revision(), "source_owner": "hexfellow-deploy",
            "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "native_kernel_abi": native.KERNEL_ABI_VERSION,
            "native_kernel_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()}
