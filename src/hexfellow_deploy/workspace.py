"""Reuse robot algorithms/assets, never the previous control runtime."""
from pathlib import Path
import os
import sys


def activate(path):
    root = Path(path).expanduser().resolve()
    if not (root / "src/mobile_manipulator/control/umi_ik.py").is_file():
        raise ValueError("workspace must be the agentic-mobile-manipulator checkout")
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    return root


def build_executor(scene_config, controller, period_s):
    from mobile_manipulator.control.executor import PoseExecutor
    from mobile_manipulator.robot.assembly import load_assembly
    from mobile_manipulator.robot.kinematics import Kinematics
    robot = scene_config.data["robot"]
    kin = Kinematics(scene_config.resolve(robot["source_checkout"]) / robot["urdf"],
                     load_assembly(scene_config.resolve(robot["assembly"])))
    return PoseExecutor(kin, dt=period_s, controller_config=controller)
