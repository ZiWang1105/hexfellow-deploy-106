"""Owned robot resource locations; no external checkout or import injection."""
from pathlib import Path
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPOSITORY_ROOT / "robot_resources"
ASSET_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"
SCENE_DIR = PROJECT_ROOT / "scenes"
URDF_DIR = PROJECT_ROOT / "urdf"
RENDER_DIR = REPOSITORY_ROOT / "renders"
REFERENCE_DIR = REPOSITORY_ROOT / "artifacts/reference"
RUN_DIR = REPOSITORY_ROOT / "runs"
LOCAL_CONFIG_DIR = REPOSITORY_ROOT / "configs/local"

def config_reference(owner: str | Path, reference: str | Path) -> Path:
    """Resolve a configuration reference relative to its owning file."""
    path = Path(reference).expanduser()
    return path.resolve() if path.is_absolute() else (Path(owner).resolve().parent / path).resolve()
