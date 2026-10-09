"""A separating certificate must preserve real overlaps and small clearances."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/camera'))
from convex_clearance import ConvexSeparation
from wrist_calibration_geometry import LocalWristChecker, WristStudy, load_plan


@pytest.mark.parametrize('x, expected', [(0., 0.), (.11, 0.), (.205, .005), (.24, .04)])
def test_certificate_does_not_clear_overlapping_or_near_boxes(x, expected):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom type="box" size=".1 .1 .1"/>
      <geom type="box" size=".1 .1 .1" pos="{x} 0 0"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert ConvexSeparation(model).certificate(data, 0, 1) == pytest.approx(expected, abs=2e-7)


def test_measured_hanging_arms_default_path_is_not_false_mesh_collision(tmp_path):
    import json
    fixture = ROOT / 'artifacts/reference/left-default-clearance-20261008/measured-start.json'
    q = np.asarray(json.loads(fixture.read_text())['q'])
    plan = load_plan(ROOT / 'config/camera/realsense_d405_left_fixed32_plan.json')
    study = WristStudy(plan, tmp_path)
    checker = object.__new__(LocalWristChecker)
    checker.route, checker.study = plan, study
    sample = q + (np.asarray(plan['initial_q']) - q) * .076
    study.update(sample)
    torso = next(g for g in range(study.m.ngeom)
                 if study.m.body(study.m.geom_bodyid[g]).name == 'maver_x4_asm_URDF_base_link')
    jaw = study.m.geom('calibration_left_jaw_positive').id
    assert (torso, jaw) in study.pairs  # Pair remains protected.
    assert study.separation.certificate(study.d, torso, jaw) == pytest.approx(.030070716, abs=2e-7)
    assert study.distance(study.d, torso, jaw) >= .015
    result = checker.plan(q, -2, plan['bounds'], plan['initial_q'], reference_origin=q)
    assert result['minimum_sampled_clearance_m'] >= .015
    assert result['path_origins_checked'] == ['measured', 'commanded_reference']
