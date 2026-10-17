import numpy as np
import pytest

from hexfellow_deploy.benchmark.general_suite import screened_speed


@pytest.mark.parametrize('order,key', [(1, 'linear_velocity'), (2, 'angular_acceleration'), (3, 'linear_jerk')])
def test_time_scaling_scales_different_derivative_orders(order, key):
    # A 2x speed reduction scales velocity/acceleration/jerk by 2/4/8.
    result = screened_speed({'ratios': {key: 2.**order}}, 1.6)
    assert result == pytest.approx(.76)
    assert (result/1.6)**order * 2.**order <= 1.


def test_public_acceptance_does_not_hide_one_bad_arm(tmp_path):
    import json
    from hexfellow_deploy.benchmark.public_tcp import evaluate
    identity = np.eye(4).tolist()
    metadata = dict(arguments={'dual': True}, moving_arms=['left', 'right'],
        controller={'policy_T_umi_tcp': {'left': identity, 'right': identity}},
        motion_start_s=0., motion_end_s=.9, dt_s=.1)
    checks = dict(completed=True, position_rms_5mm=True, position_p95_10mm=True,
                  orientation_p95_2deg=True, final_position_1mm=True)
    summary = dict(tracking_acceptance=checks, metrics={'measured': {'orientation_deg': {'p95': 0.}}})
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    requested = np.tile(np.eye(4), (10, 2, 1, 1))
    measured = requested.copy()
    measured[:-1, 0, 0, 3] = .006
    np.savez(tmp_path/'trace.npz', requested=requested, measured=measured,
             tcp_velocity=np.zeros((10, 2, 6)), time_s=np.arange(10)*.1)
    result = evaluate(tmp_path)
    assert result['position_mm']['rms'] < 5.
    assert result['per_arm_position_mm']['left']['rms'] == pytest.approx(6.)
    assert not result['tracking_accepted']
