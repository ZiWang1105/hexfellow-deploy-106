"""PnP sees SDK-mapped points; the image and raw pixel labels stay unchanged."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from policy_sim.calibration.charuco import detect, make_board
from wrist_camera_worker import SDKCornerRectifier

CONFIG={'dictionary':'DICT_4X4_50','squares_x':5,'squares_y':5,
        'square_length_m':.0174,'marker_length_m':.0116,'marker_ids':list(range(12))}

@pytest.mark.parametrize('model_name', ['none','brown_conrady','inverse_brown_conrady'])
def test_installed_sdk_projection_deprojection_without_opening_devices(model_name):
    rs=pytest.importorskip('pyrealsense2')
    intrinsics=rs.intrinsics()
    intrinsics.width=1280; intrinsics.height=720
    intrinsics.fx=900.; intrinsics.fy=880.; intrinsics.ppx=630.; intrinsics.ppy=355.
    intrinsics.model=getattr(rs.distortion,model_name)
    intrinsics.coeffs=[-.05,.01,0.,0.,.001] if model_name!='none' else [0.]*5
    rays=np.array([[.1,-.12,1.],[-.23,.16,1.],[0.,0.,1.]])
    native=np.array([rs.rs2_project_point_to_pixel(intrinsics,p.tolist()) for p in rays])
    rectifier=SDKCornerRectifier(rs,intrinsics)
    expected=(rays@rectifier.k.T)[:,:2]
    np.testing.assert_allclose(rectifier(native),expected,atol=1e-3)
    if model_name!='none':
        assert np.max(abs(native-expected))>.1

def test_sdk_rectifier_uses_sdk_rays_not_opencv_inverse_coefficients():
    calls=[]
    distortion=SimpleNamespace(none=0,brown_conrady=1,inverse_brown_conrady=2)
    intrinsics=SimpleNamespace(fx=900.,fy=880.,ppx=630.,ppy=355.,model=2)
    def deproject(intr,pixel,depth):
        calls.append((intr,pixel,depth))
        return [(pixel[0]-630.)/900.+.003,(pixel[1]-355.)/880.-.002,1.]
    rectifier=SDKCornerRectifier(SimpleNamespace(distortion=distortion,rs2_deproject_pixel_to_point=deproject),intrinsics)
    points=np.array([[550.,280.],[770.,470.]])
    np.testing.assert_allclose(rectifier(points),points+[2.7,-1.76])
    assert len(calls)==2 and all(c[0] is intrinsics and c[2]==1 for c in calls)

def test_detector_preserves_raw_corners_but_fits_pinhole_pixels():
    image=make_board(CONFIG).generateImage((600,600),marginSize=50)
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    k=np.array([[900.,0,300.],[0,900.,300.],[0,0,1.]])
    native,_=detect(image,CONFIG,k,np.zeros(5))
    shift=np.array([5.,-3.])
    mapped,overlay=detect(image,CONFIG,k,np.zeros(5),pixel_to_pinhole=lambda p:p+shift)
    assert native['accepted'] and mapped['accepted']
    # Local-homography interpolation and intrinsics-guided interpolation can
    # differ by a few hundredths of a pixel even on this ideal flat board.
    np.testing.assert_allclose(mapped['raw_image_points'],native['image_points'],atol=.03)
    np.testing.assert_allclose(mapped['image_points'],np.asarray(mapped['raw_image_points'])+shift,atol=1e-4)
    assert mapped['pnp_distortion']==[0.]*5 and overlay.shape==image.shape
    assert not np.allclose(mapped['T_camera_from_board'],native['T_camera_from_board'])
