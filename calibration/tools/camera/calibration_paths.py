"""Locations for the owned calibration tools and shared deployment dependencies."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[2]
UPSTREAM = Path(os.environ.get('HEXFELLOW_CALIBRATION_UMI', WORKSPACE/'third_party/UMI_Hardware')).resolve()
PROJECTS = WORKSPACE.parent

def desktop_robot_paths():
    """Reuse the operator's configuration; never select an example for live control."""
    import yaml
    path = Path(os.environ.get('HEXFELLOW_CALIBRATION_DESKTOP', WORKSPACE/'configs/local/desktop.yaml'))
    if not path.is_file():
        raise FileNotFoundError(f'Provide --robot-config and --research-release, or create {path}')
    robot = yaml.safe_load(path.read_text())['robot']
    return Path(robot['mapping']).expanduser(), Path(robot['research_release']).expanduser()

def sdk_directory():
    return Path(os.environ.get('ZED_DIR', PROJECTS/'ZED_camera/sdk')).expanduser()

def camera_python(kind):
    name = 'HEXFELLOW_REALSENSE_PYTHON' if kind == 'realsense' else 'HEXFELLOW_ZED_PYTHON'
    default = PROJECTS/('RealSense_camera' if kind == 'realsense' else 'ZED_camera')/'.venv/bin/python'
    return Path(os.environ.get(name, default)).expanduser()
