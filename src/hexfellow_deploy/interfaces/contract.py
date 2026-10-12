"""The policy-facing contract, independent of simulator and hardware."""
from dataclasses import asdict, dataclass, field
from typing import Protocol
import math

import numpy as np
from ..robot.joints import JOINTS, SIDES
from .depth import validate_depth
from ..control.gripper import validate_gripper

MEASURED_FIELDS = {"timestamp_s", "joint_positions", "joint_velocities", "gripper_state"}


def validate_rgb_camera(value):
    """RGB context optionally carries measured, calibrated wrist geometry."""
    if not isinstance(value, dict) or set(value) not in ({'rgb', 'received_s'}, {'rgb', 'received_s', 'geometry'}):
        raise ValueError('RGB-only camera violates the sensor allowlist')
    rgb = value['rgb']
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or rgb.size == 0):
        raise ValueError('Invalid RGB-only camera image')
    if not isinstance(value['received_s'], (int, float)) or not math.isfinite(value['received_s']):
        raise ValueError('Invalid RGB-only camera receipt timestamp')
    if 'geometry' in value:
        from .wrist_geometry import validate_wrist_geometry
        validate_wrist_geometry(value['geometry'], value['received_s'])


def validate_sensor_packet(packet, *, image=False):
    if set(packet) != MEASURED_FIELDS | ({"rgb", "depth_m"} if image else set()):
        raise ValueError("Observation violates the RGB-D/proprioception allowlist")
    if any(set(packet[field]) != set(JOINTS) for field in ("joint_positions", "joint_velocities")):
        raise ValueError("Joint feedback must contain only the 15 physical lift/arm encoder channels")
    states = packet["gripper_state"]
    if not isinstance(states, dict) or set(states) != set(SIDES) or any(s not in ('open', 'closed', 'between') for s in states.values()):
        raise ValueError("Measured gripper state must contain left/right open, closed or between")
    values = [packet["timestamp_s"],
              *packet["joint_positions"].values(), *packet["joint_velocities"].values()]
    if not np.all(np.isfinite(values)):
        raise ValueError("Invalid proprioception")
    if image:
        rgb = packet["rgb"]
        if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
                or rgb.dtype != np.uint8 or rgb.size == 0):
            raise ValueError("Invalid RGB frame")
        validate_depth(packet["depth_m"], rgb.shape)


@dataclass(frozen=True)
class ArmTarget:
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    gripper_open: int

    def __post_init__(self):
        object.__setattr__(self, 'gripper_open', validate_gripper(self.gripper_open))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"position_m", "quaternion_xyzw", "gripper_open"}:
            raise ValueError("Each arm target contains only TCP pose and gripper opening")
        p, q, g = value["position_m"], value["quaternion_xyzw"], value["gripper_open"]
        if (not isinstance(p, (list, tuple)) or not isinstance(q, (list, tuple))
                or len(p) != 3 or len(q) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                or not math.isfinite(x) for x in [*p, *q, g])):
            raise ValueError("Action values must be finite numbers with the specified dimensions")
        validate_gripper(g)
        if abs(np.linalg.norm(q) - 1) > .02:
            raise ValueError("Invalid gripper opening or non-unit quaternion")
        if np.linalg.norm(p[:2]) > 4 or not .3 <= p[2] <= 1.8:
            raise ValueError("Target outside configured Cartesian command envelope")
        return cls(tuple(p), tuple(np.asarray(q)/np.linalg.norm(q)), int(g))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NavigationTarget:
    """Visual object estimate and requested relative base turn; no localization."""
    object_xy_m: tuple[float, float] | None
    yaw_rad: float = 0.

    def __post_init__(self):
        p = self.object_xy_m
        if p is not None and (not isinstance(p, (list, tuple)) or len(p) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in p)
                or math.hypot(*p) > 4):
            raise ValueError("Navigation estimate must be two finite meters within 4 m horizontal radius")
        object.__setattr__(self, 'object_xy_m', None if p is None else tuple(p))
        if (isinstance(self.yaw_rad, bool) or not isinstance(self.yaw_rad, (int, float))
                or not math.isfinite(self.yaw_rad) or abs(self.yaw_rad) > math.pi):
            raise ValueError("Navigation yaw_rad must be finite radians in [-pi, pi]")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) not in ({'object_xy_m'}, {'object_xy_m', 'yaw_rad'}):
            raise ValueError("Navigation contains only object_xy_m and yaw_rad")
        return cls(value['object_xy_m'], value.get('yaw_rad', 0.))


@dataclass(frozen=True)
class Action:
    left: ArmTarget
    right: ArmTarget
    navigation: NavigationTarget | None = None
    skill: dict | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) not in (set(SIDES), set(SIDES) | {'navigation'}, set(SIDES) | {'navigation', 'skill'}):
            raise ValueError("An action must contain separate left and right TCP/gripper targets")
        navigation = value.get('navigation')
        skill = value.get('skill')
        if skill is not None:
            from ..skills.hand_v15 import validate_skill
            skill = validate_skill(skill)
            if navigation is not None:
                raise ValueError('Skill actions cannot contain navigation')
        return cls(**{side: ArmTarget.from_dict(value[side]) for side in SIDES},
                   navigation=None if navigation is None else NavigationTarget.from_dict(navigation), skill=skill)

    def to_dict(self):
        result = asdict(self)
        if self.skill is None:
            result.pop('skill')
        return result


@dataclass(frozen=True)
class Observation:
    observation_id: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    proprioception: dict
    # Optional named registered RGB-D cameras; identical schema on either backend.
    cameras: dict = field(default_factory=dict)
    # Optional RGB views with original receipt time and measured calibration geometry.
    rgb_cameras: dict = field(default_factory=dict)


class RobotInterface(Protocol):
    def observe(self) -> Observation: ...
    def submit_target(self, action: Action, observation_id: int) -> None: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...


class SensorActuatorTransport(Protocol):
    """Both adapters implement these measured-state/actuator operations.

    No task, target object, reward, world scene, or ground truth is exposed.
    """
    def read(self, *, image: bool) -> dict: ...
    def advance(self, command: dict, dt: float) -> dict: ...
    def close(self) -> None: ...
