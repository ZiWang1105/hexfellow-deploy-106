"""Robot model, calibration, and whole-body control."""

from .allocator import WholeBodyAllocationDiagnostics, WholeBodyAllocatorConfig
from .ik_core import IKCommand, IKDiagnostics, PinkControllerConfig, PinkIKCore
from .motion_profile import (
    DEFAULT_MOTION_PROFILE_ID,
    MotionProfile,
    available_motion_profiles,
    controller_config_for_profile,
    load_motion_profile,
)
from .posture import NullPose
from .robot import BaseMode, PlanarBaseConfig

__all__ = [
    "BaseMode",
    "DEFAULT_MOTION_PROFILE_ID",
    "IKCommand",
    "IKDiagnostics",
    "MaverX4PinkController",
    "MotionProfile",
    "NullPose",
    "PinkControllerConfig",
    "PinkIKCore",
    "PlanarBaseConfig",
    "WholeBodyAllocationDiagnostics",
    "WholeBodyAllocatorConfig",
    "available_motion_profiles",
    "controller_config_for_profile",
    "load_motion_profile",
]


def __getattr__(name: str):
    if name == "MaverX4PinkController":
        from .ik import MaverX4PinkController

        return MaverX4PinkController
    raise AttributeError(name)
