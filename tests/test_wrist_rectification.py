"""Use installed SDK math without opening any physical device."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from hexfellow_deploy.wrist_rectification import WristRectifier


def test_d405_rectified_pixels_match_calibration_sdk_rays():
    rs = pytest.importorskip('pyrealsense2')
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root/'calibration/config/camera/realsense_d405_rgb_intrinsics_20261008.json').read_text())
    k = np.array(metadata['K'])
    intr = rs.intrinsics()
    intr.width, intr.height = metadata['width'], metadata['height']
    intr.fx, intr.fy, intr.ppx, intr.ppy = k[0,0], k[1,1], k[0,2], k[1,2]
    intr.model = rs.distortion.inverse_brown_conrady
    intr.coeffs = metadata['native_distortion']
    profile = SimpleNamespace(as_video_stream_profile=lambda:SimpleNamespace(get_intrinsics=lambda:intr))
    rectifier = WristRectifier(rs, profile, intr.width, intr.height)
    for u,v in ((0,0),(50,300),(1279,719),(640,360),(1200,50)):
        native = [float(rectifier.map_x[v,u]),float(rectifier.map_y[v,u])]
        ray = rs.rs2_deproject_pixel_to_point(intr,native,1.)
        uv = k @ np.asarray(ray)
        np.testing.assert_allclose(uv[:2]/uv[2],[u,v],atol=.001)
    assert rectifier.processing=='sdk_rectified_pinhole'
    assert rectifier.k==metadata['K']
    # Identity optical orientation and RGB channels are preserved by the remap.
    rgb=np.full((intr.height,intr.width,3),[20,90,170],np.uint8)
    np.testing.assert_array_equal(rectifier.apply(rgb)[360,640],[20,90,170])


def test_zero_distortion_preserves_exact_rgb():
    intr=SimpleNamespace(fx=10.,fy=11.,ppx=3.,ppy=2.,coeffs=[0.]*5)
    profile=SimpleNamespace(as_video_stream_profile=lambda:SimpleNamespace(get_intrinsics=lambda:intr))
    rectifier=WristRectifier(None,profile,8,6)
    rgb=np.arange(144,dtype=np.uint8).reshape(6,8,3)
    assert rectifier.apply(rgb) is rgb
    assert rectifier.processing=='native_pinhole'
