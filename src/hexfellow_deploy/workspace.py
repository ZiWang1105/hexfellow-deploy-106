"""Resolve configuration roots; robot implementation belongs to this package."""
from pathlib import Path
import os
import sys


def activate(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Configuration workspace does not exist: {root}")
    os.chdir(root)
    return root


def build_executor(scene_config, controller, period_s):
    from hexfellow_deploy.control.executor import PoseExecutor
    from hexfellow_deploy.robot.assembly import load_assembly
    from hexfellow_deploy.robot.kinematics import Kinematics
    robot = scene_config.data["robot"]
    kin = Kinematics(scene_config.resolve(robot["source_checkout"]) / robot["urdf"],
                     load_assembly(scene_config.resolve(robot["assembly"])))
    return PoseExecutor(kin, dt=period_s, controller_config=controller)


def build_shared_executor(metadata, period_s):
    from hexfellow_deploy.skills.executor import SkillExecutor as PoseExecutor
    from hexfellow_deploy.control.base_motion import BaseMotionConfig
    from hexfellow_deploy.robot.assembly import load_assembly
    from hexfellow_deploy.robot.kinematics import Kinematics
    robot = metadata['robot']
    kin = Kinematics(Path(robot['urdf']), load_assembly(Path(robot['assembly'])))
    return PoseExecutor(kin, dt=period_s, controller_config=metadata['controller'],
                        base_motion=BaseMotionConfig.load(metadata['base_motion']),
                        runtime_effort_limits_nm=metadata['effort_limits_nm'],
                        runtime_velocity_bounds=metadata['velocity_bounds'])
