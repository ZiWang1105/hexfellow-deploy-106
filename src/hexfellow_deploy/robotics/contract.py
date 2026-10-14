"""Versioned policy-frame observation and anchor-relative action contract.

The GenRobot source calls its pose topic vio/eef_pose and reports the
das_gripper_v4/base_link frame. That identifier is a policy frame: it is not
assumed to be either the finger TCP or a gripper pose.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

CONTRACT_ID = "umi_bimanual_v3"
ACTION_HORIZON = 16
REPLAN_STRIDE_STEPS = 8
ACTION_DT_NS = 100_000_000
HISTORY_DT_NS = 100_000_000
HISTORY_DT_MIN_RATIO = 0.5
HISTORY_DT_MAX_RATIO = 1.8
HISTORY_LENGTH = 2
STATE_DIM = 32
ACTION_DIM = 20
PER_ARM_DIM = 10
ARM_ORDER = ("left", "right")
IMAGE_SHAPE = (224, 224, 3)
IMAGE_HISTORY_SHAPE = (HISTORY_LENGTH, *IMAGE_SHAPE)
POLICY_FRAME_NAME = "das_gripper_v4/base_link"
GRIPPER_MAX_WIDTH_M = 0.105

LEFT_IMAGE_KEY = "left_wrist_rgb"
RIGHT_IMAGE_KEY = "right_wrist_rgb"
IMAGE_KEYS = (LEFT_IMAGE_KEY, RIGHT_IMAGE_KEY)
POLICY_FRAME_POSE_KEYS = {
    "left": "left_policy_frame_pose_abs_wxyz",
    "right": "right_policy_frame_pose_abs_wxyz",
}
GRIPPER_KEYS = {
    "left": "left_gripper_width_m",
    "right": "right_gripper_width_m",
}
TIMESTAMP_KEY = "timestamp_ns"
HISTORY_VALID_KEY = "history_valid"
PROMPT_KEY = "prompt"
EPISODE_ID_KEY = "episode_id"
REQUEST_ID_KEY = "request_id"
ANCHOR_TIMESTAMP_KEY = "anchor_timestamp_ns"

OBSERVATION_KEYS = (
    EPISODE_ID_KEY,
    REQUEST_ID_KEY,
    ANCHOR_TIMESTAMP_KEY,
    *IMAGE_KEYS,
    *POLICY_FRAME_POSE_KEYS.values(),
    *GRIPPER_KEYS.values(),
    TIMESTAMP_KEY,
    HISTORY_VALID_KEY,
    PROMPT_KEY,
)

_SCHEMA = {
    "contract_id": CONTRACT_ID,
    "arm_order": ARM_ORDER,
    "timing": {
        "history_length": HISTORY_LENGTH,
        "history_dt_ns": HISTORY_DT_NS,
        "action_dt_ns": ACTION_DT_NS,
        "action_horizon": ACTION_HORIZON,
        "replan_stride_steps": REPLAN_STRIDE_STEPS,
    },
    "observation": {
        EPISODE_ID_KEY: {"dtype": "string", "constraint": "non-empty"},
        REQUEST_ID_KEY: {"dtype": "non-negative integer"},
        ANCHOR_TIMESTAMP_KEY: {"dtype": "integer", "equals": "timestamp_ns[-1]"},
        LEFT_IMAGE_KEY: {"dtype": "uint8", "shape": list(IMAGE_HISTORY_SHAPE)},
        RIGHT_IMAGE_KEY: {"dtype": "uint8", "shape": list(IMAGE_HISTORY_SHAPE)},
        POLICY_FRAME_POSE_KEYS["left"]: {
            "dtype": "float32",
            "shape": [HISTORY_LENGTH, 7],
            "layout": "xyz+wxyz",
            "policy_frame": POLICY_FRAME_NAME,
            "reference_frame": "episode_vio_world",
        },
        POLICY_FRAME_POSE_KEYS["right"]: {
            "dtype": "float32",
            "shape": [HISTORY_LENGTH, 7],
            "layout": "xyz+wxyz",
            "policy_frame": POLICY_FRAME_NAME,
            "reference_frame": "episode_vio_world",
        },
        GRIPPER_KEYS["left"]: {"dtype": "float32", "shape": [HISTORY_LENGTH], "unit": "m"},
        GRIPPER_KEYS["right"]: {"dtype": "float32", "shape": [HISTORY_LENGTH], "unit": "m"},
        TIMESTAMP_KEY: {"dtype": "uint64", "shape": [HISTORY_LENGTH]},
        HISTORY_VALID_KEY: {"dtype": "bool", "shape": [HISTORY_LENGTH]},
        PROMPT_KEY: {"dtype": "string"},
    },
    "derived_state": {
        "dtype": "float32",
        "shape": [STATE_DIM],
        "history_time_feature": "invalid=-1; valid=actual_dt/nominal_dt-1; never data-normalized",
    },
    "action": {
        "dtype": "float32",
        "shape": [ACTION_HORIZON, ACTION_DIM],
        "representation": "anchor_relative_policy_frame_trajectory",
        "per_arm": ["anchor_relative_xyz", "rotation_6d_columns", "absolute_gripper_width_m"],
        "policy_frame": POLICY_FRAME_NAME,
    },
    "response": {
        "identity": [EPISODE_ID_KEY, REQUEST_ID_KEY, ANCHOR_TIMESTAMP_KEY],
        "action_dt_ns": ACTION_DT_NS,
        "target_timestamp_ns": {
            "dtype": "uint64",
            "shape": [ACTION_HORIZON],
            "formula": "anchor_timestamp_ns + k*action_dt_ns, k=1..H",
        },
    },
}


def schema_hash() -> str:
    """Return the stable SHA-256 fingerprint used in the server handshake."""
    encoded = json.dumps(_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ContractError(ValueError):
    """Raised before malformed policy data can reach a controller."""


def _rigid_transform(value: object, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ContractError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ContractError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ContractError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ContractError(f"{name} rotation must be right-handed and orthonormal")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = _rigid_transform(transform, name="transform")
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Encode the first two columns of a rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ContractError(f"expected (..., 3, 3) rotations, got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ContractError("rotation matrix contains non-finite values")
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ContractError(f"expected (..., 6) rotations, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ContractError("rotation_6d contains non-finite values")
    first = value[..., :3]
    second = value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ContractError("rotation_6d first column is degenerate")
    x_axis = first / first_norm
    second = second - np.sum(x_axis * second, axis=-1, keepdims=True) * x_axis
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ContractError("rotation_6d second column is degenerate")
    y_axis = second / second_norm
    z_axis = np.cross(x_axis, y_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ContractError("policy-frame quaternion must be finite wxyz[4]")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8 or not np.isclose(norm, 1.0, atol=1e-4):
        raise ContractError(f"policy-frame quaternion must be unit length, got norm={norm:g}")
    w, x, y, z = quaternion / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    rotation = _rigid_transform(transform, name="rotation")[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            quaternion = np.array(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            quaternion = np.array(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            quaternion = np.array(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def pose_wxyz_to_transform(pose: object) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ContractError(f"policy-frame pose must be finite xyz+wxyz[7], got {pose.shape}")
    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = quaternion_wxyz_to_matrix(pose[3:])
    return transform


def transform_to_pose_wxyz(transform: object) -> np.ndarray:
    transform = _rigid_transform(transform, name="policy-frame transform")
    return np.concatenate((transform[:3, 3], matrix_to_quaternion_wxyz(transform[:3, :3])))


def validate_observation(observation: Mapping[str, object]) -> None:
    missing = [key for key in OBSERVATION_KEYS if key not in observation]
    extra = sorted(set(observation) - set(OBSERVATION_KEYS))
    if missing or extra:
        raise ContractError(f"observation keys mismatch: missing={missing}, extra={extra}")
    if not isinstance(observation[EPISODE_ID_KEY], str) or not observation[EPISODE_ID_KEY]:
        raise ContractError("episode_id must be a non-empty string")
    request_id = observation[REQUEST_ID_KEY]
    if not isinstance(request_id, (int, np.integer)) or isinstance(request_id, (bool, np.bool_)) or request_id < 0:
        raise ContractError("request_id must be a non-negative integer")
    for key in IMAGE_KEYS:
        value = np.asarray(observation[key])
        if value.shape != IMAGE_HISTORY_SHAPE or value.dtype != np.uint8:
            raise ContractError(f"{key} must be uint8{IMAGE_HISTORY_SHAPE}, got {value.dtype}{value.shape}")
    for key in POLICY_FRAME_POSE_KEYS.values():
        value = np.asarray(observation[key])
        if value.shape != (HISTORY_LENGTH, 7) or value.dtype != np.float32:
            raise ContractError(f"{key} must be float32[{HISTORY_LENGTH},7], got {value.dtype}{value.shape}")
        for pose in value:
            pose_wxyz_to_transform(pose)
    for key in GRIPPER_KEYS.values():
        value = np.asarray(observation[key])
        if value.shape != (HISTORY_LENGTH,) or value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ContractError(f"{key} must be finite float32[{HISTORY_LENGTH}], got {value.dtype}{value.shape}")
    timestamps = np.asarray(observation[TIMESTAMP_KEY])
    if timestamps.shape != (HISTORY_LENGTH,) or timestamps.dtype != np.uint64:
        raise ContractError(f"timestamp_ns must be uint64[{HISTORY_LENGTH}], got {timestamps.dtype}{timestamps.shape}")
    valid = np.asarray(observation[HISTORY_VALID_KEY])
    if valid.shape != (HISTORY_LENGTH,) or valid.dtype != np.bool_ or not bool(valid[-1]):
        raise ContractError("history_valid must be bool[2] with a valid current sample")
    anchor_timestamp = observation[ANCHOR_TIMESTAMP_KEY]
    if not isinstance(anchor_timestamp, (int, np.integer)) or int(anchor_timestamp) != int(timestamps[-1]):
        raise ContractError("anchor_timestamp_ns must equal the current observation timestamp")
    if not isinstance(observation[PROMPT_KEY], str):
        raise ContractError("prompt must be a string")


def history_pair_valid(observation: Mapping[str, object]) -> bool:
    validate_observation(observation)
    valid = np.asarray(observation[HISTORY_VALID_KEY], dtype=bool)
    timestamps = np.asarray(observation[TIMESTAMP_KEY], dtype=np.uint64).astype(np.int64)
    delta = int(timestamps[1] - timestamps[0])
    ratio = delta / HISTORY_DT_NS
    return bool(valid.all() and HISTORY_DT_MIN_RATIO <= ratio <= HISTORY_DT_MAX_RATIO)


def policy_frame_transforms_from_observation(
    observation: Mapping[str, object], *, index: int = -1
) -> dict[str, np.ndarray]:
    validate_observation(observation)
    return {
        side: pose_wxyz_to_transform(np.asarray(observation[key])[index])
        for side, key in POLICY_FRAME_POSE_KEYS.items()
    }


def build_policy_state(observation: Mapping[str, object]) -> np.ndarray:
    """Build the invariant 32D state without data-normalizing history dt."""
    validate_observation(observation)
    pair_valid = history_pair_valid(observation)
    current = policy_frame_transforms_from_observation(observation, index=1)
    previous = policy_frame_transforms_from_observation(observation, index=0) if pair_valid else current
    blocks: list[np.ndarray] = []
    for side in ARM_ORDER:
        history = invert_transform(current[side]) @ previous[side]
        blocks.append(np.concatenate((history[:3, 3], matrix_to_rot6d(history[:3, :3]))))
    inter_policy_frame = invert_transform(current["left"]) @ current["right"]
    blocks.append(
        np.concatenate((inter_policy_frame[:3, 3], matrix_to_rot6d(inter_policy_frame[:3, :3])))
    )
    grippers = {
        side: np.asarray(observation[key], dtype=np.float64).copy()
        for side, key in GRIPPER_KEYS.items()
    }
    if not pair_valid:
        for value in grippers.values():
            value[0] = value[1]
    blocks.append(
        np.array((grippers["left"][0], grippers["left"][1], grippers["right"][0], grippers["right"][1]))
    )
    if pair_valid:
        timestamps = np.asarray(observation[TIMESTAMP_KEY], dtype=np.uint64).astype(np.int64)
        history_time_feature = (int(timestamps[1] - timestamps[0]) / HISTORY_DT_NS) - 1.0
    else:
        history_time_feature = -1.0
    state = np.concatenate((*blocks, (history_time_feature,))).astype(np.float32)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ContractError(f"derived policy state must be finite float32[{STATE_DIM}]")
    return state


def validate_action_chunk(raw: object, *, require_full_horizon: bool = True) -> np.ndarray:
    """Validate de-normalized anchor-relative actions without implicit casting."""
    chunk = np.asarray(raw)
    minimum_rows = ACTION_HORIZON if require_full_horizon else 1
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] < minimum_rows:
        raise ContractError(f"expected at least ({minimum_rows}, {ACTION_DIM}) actions, got {chunk.shape}")
    if require_full_horizon and chunk.shape[0] != ACTION_HORIZON:
        raise ContractError(f"expected exactly {ACTION_HORIZON} rows, got {chunk.shape[0]}")
    if chunk.dtype != np.float32:
        raise ContractError(f"policy actions must be float32, got {chunk.dtype}")
    if not np.all(np.isfinite(chunk)):
        raise ContractError("policy action chunk contains non-finite values")
    for arm_index, side in enumerate(ARM_ORDER):
        offset = arm_index * PER_ARM_DIM
        try:
            rot6d_to_matrix(chunk[:, offset + 3 : offset + 9])
        except ContractError as error:
            raise ContractError(f"{side} policy rotations are invalid: {error}") from error
    return chunk


@dataclasses.dataclass(frozen=True)
class ArmTarget:
    transform: np.ndarray
    width_m: float
    requested_width_m: float
    gripper_clamped: bool


def decode_anchor_relative_chunk(
    chunk: np.ndarray,
    anchor_world: Mapping[str, np.ndarray],
    *,
    rows: int | None = None,
    max_gripper_width_m: float = GRIPPER_MAX_WIDTH_M,
) -> tuple[dict[str, ArmTarget], ...]:
    """Decode every trajectory row from the same captured policy-frame anchor."""
    chunk = validate_action_chunk(chunk, require_full_horizon=False)
    row_count = len(chunk) if rows is None else rows
    if row_count < 1 or row_count > min(len(chunk), ACTION_HORIZON):
        raise ContractError(f"invalid decoded row count {row_count}")
    if not np.isfinite(max_gripper_width_m) or max_gripper_width_m <= 0:
        raise ContractError("maximum gripper width must be finite and positive")
    missing = [side for side in ARM_ORDER if side not in anchor_world]
    if missing:
        raise ContractError(f"missing policy-frame anchors for {missing}")
    anchors = {
        side: _rigid_transform(anchor_world[side], name=f"{side} policy-frame anchor").copy()
        for side in ARM_ORDER
    }
    result: list[dict[str, ArmTarget]] = []
    for row_index in range(row_count):
        targets: dict[str, ArmTarget] = {}
        for arm_index, side in enumerate(ARM_ORDER):
            offset = arm_index * PER_ARM_DIM
            relative = np.eye(4)
            relative[:3, :3] = rot6d_to_matrix(chunk[row_index, offset + 3 : offset + 9])
            relative[:3, 3] = chunk[row_index, offset : offset + 3]
            requested = float(chunk[row_index, offset + 9])
            width = float(np.clip(requested, 0.0, max_gripper_width_m))
            targets[side] = ArmTarget(
                transform=anchors[side] @ relative,
                width_m=width,
                requested_width_m=requested,
                gripper_clamped=not np.isclose(requested, width),
            )
        result.append(targets)
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class PolicyResponse:
    episode_id: str
    request_id: int
    anchor_timestamp_ns: int
    action_dt_ns: int
    target_timestamp_ns: np.ndarray
    actions: np.ndarray


def validate_policy_response(raw: object, request: Mapping[str, object]) -> PolicyResponse:
    if not isinstance(raw, Mapping):
        raise ContractError("v3 policy response must be a mapping")
    required = {
        EPISODE_ID_KEY,
        REQUEST_ID_KEY,
        ANCHOR_TIMESTAMP_KEY,
        "action_dt_ns",
        "target_timestamp_ns",
        "actions",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ContractError(f"policy response is missing {missing}")
    if not isinstance(raw[EPISODE_ID_KEY], str):
        raise ContractError("policy response episode_id must be a string")
    for key in (REQUEST_ID_KEY, ANCHOR_TIMESTAMP_KEY, "action_dt_ns"):
        if not isinstance(raw[key], (int, np.integer)) or isinstance(raw[key], (bool, np.bool_)):
            raise ContractError(f"policy response {key} must be an integer")
    for key in (EPISODE_ID_KEY, REQUEST_ID_KEY, ANCHOR_TIMESTAMP_KEY):
        if raw[key] != request[key]:
            raise ContractError(f"policy response {key} does not match request")
    action_dt_ns = int(raw["action_dt_ns"])
    if action_dt_ns != ACTION_DT_NS:
        raise ContractError(f"policy response action_dt_ns must be {ACTION_DT_NS}, got {action_dt_ns}")
    actions = validate_action_chunk(raw["actions"])
    timestamps = np.asarray(raw["target_timestamp_ns"])
    if timestamps.shape != (ACTION_HORIZON,) or timestamps.dtype != np.uint64:
        raise ContractError("target_timestamp_ns must be uint64[16]")
    expected = int(request[ANCHOR_TIMESTAMP_KEY]) + ACTION_DT_NS * np.arange(
        1, ACTION_HORIZON + 1, dtype=np.uint64
    )
    if not np.array_equal(timestamps, expected):
        raise ContractError("target timestamps are not anchor_timestamp_ns + k*action_dt_ns")
    return PolicyResponse(
        episode_id=str(raw[EPISODE_ID_KEY]),
        request_id=int(raw[REQUEST_ID_KEY]),
        anchor_timestamp_ns=int(raw[ANCHOR_TIMESTAMP_KEY]),
        action_dt_ns=action_dt_ns,
        target_timestamp_ns=timestamps.copy(),
        actions=actions.copy(),
    )


def identity_action_chunk(
    widths_m: Sequence[float] = (GRIPPER_MAX_WIDTH_M, GRIPPER_MAX_WIDTH_M),
) -> np.ndarray:
    """Return a fixed-anchor hold trajectory."""
    if len(widths_m) != 2:
        raise ContractError("identity chunk needs one gripper width per arm")
    chunk = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    identity = matrix_to_rot6d(np.eye(3)).astype(np.float32)
    for arm_index, width in enumerate(widths_m):
        offset = arm_index * PER_ARM_DIM
        chunk[:, offset + 3 : offset + 9] = identity
        chunk[:, offset + 9] = width
    return chunk
