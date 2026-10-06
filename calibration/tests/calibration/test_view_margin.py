import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/camera'))
from calibration_path import usable_nominal_view


def test_failed_pose_margin_is_rejected_even_if_corners_are_detectable():
    transform = np.eye(4)
    transform[2, 3] = .5
    view = dict(margin_px=85.1069854432144, facing_cosine=.9,
                distance_m=.5, T_camera_from_board=transform.tolist())
    assert not usable_nominal_view(view)
    view['margin_px'] = 180
    assert usable_nominal_view(view)
    view['facing_cosine'] = -.9
    assert not usable_nominal_view(view)
    view['facing_cosine'] = .9
    view['margin_px'] = float('nan')
    assert not usable_nominal_view(view)
