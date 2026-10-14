"""Run under both the simulation and ZED Python/OpenCV environments."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from hexfellow_deploy.robotics.calibration.charuco import make_board,detect


class CharucoCompatibility(unittest.TestCase):
    def setUp(self):
        self.config=dict(squares_x=5,squares_y=5,square_length_m=.0174,
                         marker_length_m=.0116,dictionary='DICT_4X4_50',marker_ids=list(range(12)))
        self.board=make_board(self.config)
        gray=cv2.copyMakeBorder(self.board.generateImage((600,600)),100,100,100,100,cv2.BORDER_CONSTANT,value=255)
        self.image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        self.k=np.array([[1000.,0,400],[0,1000,400],[0,0,1.]])
        params=cv2.aruco.CharucoParameters();params.cameraMatrix=self.k;params.distCoeffs=np.zeros(5)
        self.raw=cv2.aruco.CharucoDetector(self.board,params).detectBoard(self.image)

    def check_shape(self,flat):
        corners,ids,markers,marker_ids=self.raw
        self.assertIsNotNone(ids)
        c=corners.reshape(-1,2) if flat else corners.reshape(-1,1,2)
        i=ids.reshape(-1) if flat else ids.reshape(-1,1)
        with patch('cv2.aruco.CharucoDetector') as factory:
            factory.return_value.detectBoard.return_value=(c,i,markers,marker_ids)
            result,_=detect(self.image,self.config,self.k,np.zeros(5))
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['corners'],16)
        self.assertLess(result['reprojection_rms_px'],.1)
        self.assertGreater(np.asarray(result['T_camera_from_board'])[2,3],0)
        return result

    def test_opencv4_and_5_shapes_produce_same_pose_and_rms(self):
        old,new=self.check_shape(False),self.check_shape(True)
        np.testing.assert_allclose(old['T_camera_from_board'],new['T_camera_from_board'],atol=1e-10)
        self.assertAlmostEqual(old['reprojection_rms_px'],new['reprojection_rms_px'])

    def test_real_runtime_detector_returns_valid_pose(self):
        result,_=detect(self.image,self.config,self.k,np.zeros(5))
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['corners'],16)
        self.assertLess(result['reprojection_rms_px'],.1)

    def test_overlay_failure_does_not_invalidate_pose(self):
        with patch('cv2.aruco.drawDetectedCornersCharuco',side_effect=cv2.error('overlay test')):
            result,_=detect(self.image,self.config,self.k,np.zeros(5))
        self.assertTrue(result['accepted'],result)
        self.assertTrue(result['overlay_warnings'])

    def test_bad_frame_does_not_prevent_next_frame(self):
        with patch('cv2.aruco.CharucoDetector') as factory:
            factory.return_value.detectBoard.side_effect=cv2.error('bad frame')
            result,image=detect(self.image,self.config,self.k,np.zeros(5))
        self.assertFalse(result['accepted'])
        self.assertIn('processing failed',result['reason'])
        self.assertEqual(image.shape,self.image.shape)
        self.assertTrue(detect(self.image,self.config,self.k,np.zeros(5))[0]['accepted'])

    def test_mismatched_correspondences_rejected(self):
        corners,ids,markers,marker_ids=self.raw
        with patch('cv2.aruco.CharucoDetector') as factory:
            factory.return_value.detectBoard.return_value=(corners[:-1],ids,markers,marker_ids)
            result,_=detect(self.image,self.config,self.k,np.zeros(5))
        self.assertFalse(result['accepted'])
        self.assertIn('correspondence',result['reason'])

if __name__=='__main__':unittest.main()
