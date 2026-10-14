import numpy as np
import pinocchio as pin

from hexfellow_deploy.robotics.control.ik import JerkLimit


def test_jerk_limit_builds_affine_displacement_envelope():
    model = pin.Model()
    model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "joint")
    limit = JerkLimit(model, np.array([2.0]))
    limit.reset(np.array([0.2]), np.array([0.5]))

    matrix, vector = limit.compute_qp_inequalities(None, 0.1)

    # v_next = 0.2 + 0.5*0.1 +/- 2.0*0.1**2, and Pink's
    # decision variable is displacement v_next*dt.
    assert np.array_equal(matrix, np.array([[1.0], [-1.0]]))
    assert np.allclose(vector, (0.027, -0.023))
