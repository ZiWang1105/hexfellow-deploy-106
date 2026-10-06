"""Regression: degenerate narrow-phase zeros must not defeat independent bounds."""
from pathlib import Path
import sys
from types import SimpleNamespace
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/camera'))
from validate_calibration_poses import Study


def test_separated_rotated_boxes_have_positive_certified_clearance(monkeypatch):
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="box" size=".05 .05 .05" pos="0 0 0"/>
      <geom type="box" size=".05 .05 .05" pos=".3 0 0" euler="0 0 45"/>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *a:0.)
    distance=Study.distance(SimpleNamespace(m=model),data,0,1,1.)
    np.testing.assert_allclose(distance,.3-.05-.05*np.sqrt(2),atol=1e-7)


def test_overlapping_boxes_are_not_certified_safe():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="box" size=".05 .05 .05" pos="0 0 0"/>
      <geom type="box" size=".05 .05 .05" pos=".08 0 0"/>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert Study.distance(SimpleNamespace(m=model),data,0,1,1.)<=0
