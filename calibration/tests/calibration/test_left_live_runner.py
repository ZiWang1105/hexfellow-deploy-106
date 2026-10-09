"""Left wrist route, profile identity and measured capture admission."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from calibrate_robot import validate_wrist_camera_profile
from wrist_camera_worker import simulation_intrinsics
from calibration_runtime import MeasuredPoseArrival, capture_position_check
from calibration_outcomes import InvalidCalibrationPose


def test_d405_simulation_and_live_profile_are_camera_specific():
    plan=json.loads((ROOT/'config/camera/realsense_d405_left_fixed32_plan.json').read_text())
    profile=json.loads((ROOT/'config/camera/realsense_d405_rgb_intrinsics_20261008.json').read_text())
    validate_wrist_camera_profile(plan,profile)
    np.testing.assert_array_equal(simulation_intrinsics(ROOT/'config/camera/realsense_d405_left_fixed32_plan.json',plan['serial']),profile['K'])
    with pytest.raises(ValueError,match='requires'):simulation_intrinsics(None,plan['serial'])
    with pytest.raises(ValueError,match='serial'):simulation_intrinsics(ROOT/'config/camera/realsense_d405_rgb_intrinsics_20261008.json','243222074562')
    for change,message in [({'serial':'243222074562'},'serial'),({'width':640},'resolution'),({'frame':'wrong'},'frame'),({'K':np.eye(3).tolist()},'intrinsics')]:
        with pytest.raises(ValueError,match=message):validate_wrist_camera_profile(plan,dict(profile,**change))


def test_left_settling_accepts_measured_offset_and_keeps_right_guard():
    target=np.zeros(18);q=target.copy();q[9]=np.deg2rad(4.9)
    gate=MeasuredPoseArrival(side='left')
    for i in range(51):
        t=i*.02
        state=SimpleNamespace(q=q.copy(),v=np.zeros(18),received=dict.fromkeys(('left','right','base','theta'),t))
        accepted=gate.update(state,target,target,target,t,.02)
    assert accepted
    assert gate.details['measured_q'][9]==q[9]
    assert capture_position_check(q,target,side='left')['maximum_active_offset_deg']==pytest.approx(4.9)
    q[9]=np.deg2rad(5.1)
    with pytest.raises(InvalidCalibrationPose):capture_position_check(q,target,side='left')
    q[:]=0;q[11]=np.deg2rad(2.)
    with pytest.raises(ValueError):capture_position_check(q,target,side='left')
