"""Rectify native wrist RGB using the active RealSense optical model.

The SDK constructs a destination-pinhole to source-native pixel map once at
startup. Inverse Brown-Conrady coefficients are never passed to OpenCV as a
forward distortion model. Per-frame work is one RGB remap; no depth is used.
"""
import numpy as np


class WristRectifier:
    def __init__(self, rs, profile, width, height):
        intr = profile.as_video_stream_profile().get_intrinsics()
        self.k = [[float(intr.fx), 0., float(intr.ppx)],
                  [0., float(intr.fy), float(intr.ppy)], [0., 0., 1.]]
        coeffs = np.asarray(intr.coeffs, dtype=float)
        if (coeffs.shape != (5,) or not np.isfinite(coeffs).all()
                or not np.isfinite(self.k).all() or intr.fx <= 0 or intr.fy <= 0):
            raise ValueError('Invalid active wrist RGB intrinsics')
        self.native = dict(K=self.k, distortion=coeffs.tolist(),
                           distortion_model=str(getattr(intr, 'model', 'none')),
                           width=width, height=height)
        self.map_x = self.map_y = None
        self.processing = 'native_pinhole'
        if not np.any(coeffs):
            return
        if intr.model not in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
            raise ValueError('Unsupported wrist RGB distortion model')
        # Use the same SDK geometry as the physical calibration corner mapping.
        pixels = np.array([
            rs.rs2_project_point_to_pixel(intr, [(u-intr.ppx)/intr.fx,
                                                (v-intr.ppy)/intr.fy, 1.])
            for v in range(height) for u in range(width)
        ], dtype=np.float32).reshape(height, width, 2)
        if not np.isfinite(pixels).all():
            raise ValueError('Invalid SDK wrist rectification map')
        self.map_x, self.map_y = pixels[:, :, 0].copy(), pixels[:, :, 1].copy()
        self.processing = 'sdk_rectified_pinhole'

    def apply(self, rgb):
        if self.map_x is None:
            return rgb
        import cv2
        if rgb.shape[:2] != self.map_x.shape:
            raise ValueError('Wrist RGB shape differs from active profile')
        return cv2.remap(rgb, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
