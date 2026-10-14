"""Hard lift locks retain their reference despite changing measured encoders."""
import numpy as np
import pytest

from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile
from hexfellow_deploy.robotics.control.robot import LIFT_JOINT
from hexfellow_deploy.robotics.control.target_replay import tcp_poses


@pytest.mark.parametrize('base_mode', ['fixed', 'planar'])
def test_locked_lift_preserves_initial_reference_through_tracking_and_hold(base_mode):
    config = controller_config_for_profile('x7-gp80-highspeed-fixed-base-v1')
    assert config.lift_reference_max_error_m is None
    core = PinkIKCore(config=config, base_mode=base_mode, locked_joint_names=(LIFT_JOINT,))
    q = core.home_q.copy()
    index = core.model.joints[core.model.getJointId(LIFT_JOINT)].idx_q
    q[index] = .3
    goals = tcp_poses(core, q)
    core.reset(q, goals, measured_velocity=np.zeros(core.model.nv), state_timestamp=0.)
    for tick in range(1, 41):
        if tick == 20:
            core.request_stop()
        measured = q.copy()
        measured[index] += tick * .00001
        result = core.step(measured, goals, measured_velocity=np.zeros(core.model.nv),
                           state_timestamp=tick*.02, control_timestamp=tick*.02)
        assert result.executable, result.diagnostics.fault
        assert result.positions[index] == pytest.approx(.3, abs=1e-10)
        assert result.velocity[index] == pytest.approx(0., abs=1e-10)
        assert core._last_measured_q[index] == measured[index]
