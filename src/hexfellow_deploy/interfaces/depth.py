"""Shared registered depth encoding; no segmentation or scene interpretation."""
import cv2
import numpy as np

MAX_DEPTH_M = 65.535
VISUAL_RANGE_M = (.2, 4.)


def validate_depth(depth_m, rgb_shape):
    if (not isinstance(depth_m, np.ndarray) or depth_m.dtype != np.float32 or depth_m.ndim != 2
            or depth_m.shape != tuple(rgb_shape[:2]) or depth_m.size == 0
            or not np.isfinite(depth_m).all() or np.any(depth_m < 0)
            or np.any(depth_m > MAX_DEPTH_M)):
        raise ValueError("Invalid registered depth: require float32 meters, RGB shape, zero for invalid")


def depth_millimeters(depth_m):
    validate_depth(depth_m, depth_m.shape)
    return np.rint(depth_m.astype(np.float64) * 1000).astype(np.uint16)


def encode_depth_png(depth_m):
    ok, encoded = cv2.imencode(".png", depth_millimeters(depth_m))
    if not ok:
        raise ValueError("Cannot encode depth PNG")
    return encoded.tobytes()


def decode_depth_png(encoded):
    value = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if value is None or value.dtype != np.uint16 or value.ndim != 2:
        raise ValueError("Depth PNG must be single-channel uint16 millimeters")
    return value.astype(np.float32) / 1000


def visualize_depth(depth_m):
    """Fixed-scale false color, retaining invalid pixels and all image locations."""
    validate_depth(depth_m, depth_m.shape)
    low, high = VISUAL_RANGE_M
    scaled = np.rint(np.clip((depth_m - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[depth_m == 0] = 0
    return rgb


def model_depth_metadata(depth_m):
    """Uniform sensor samples allow metric reads without object/pixel selection."""
    millimeters = depth_millimeters(depth_m)
    height, width = millimeters.shape
    u = np.floor((np.arange(min(32, width)) + .5) * width / min(32, width)).astype(int)
    v = np.floor((np.arange(min(18, height)) + .5) * height / min(18, height)).astype(int)
    return {"frame": "registered_rgb_optical_z", "unit": "millimeters", "invalid_value": 0,
            "image_size_wh": [width, height], "u_pixels": u.tolist(), "v_pixels": v.tolist(),
            "samples_mm": millimeters[np.ix_(v, u)].tolist(),
            "visualization": {"colormap": "OpenCV TURBO", "range_m": list(VISUAL_RANGE_M),
                              "invalid_rgb": [0, 0, 0]}}
