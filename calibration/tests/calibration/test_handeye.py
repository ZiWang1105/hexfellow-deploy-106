import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from policy_sim.calibration.handeye import transform, inverse, solve, residual, jacobian, retract


def fixture(noise=0):
    rng=np.random.default_rng(271)
    x=transform(Rotation.from_euler('xyz',[.2,-.15,.6]).as_matrix(),[.1,-.04,.18])
    y=transform(Rotation.from_euler('xyz',[.4,.2,-.7]).as_matrix(),[.01,.04,.13])
    g=np.array([transform(Rotation.from_rotvec(rng.normal(0,.4,3)).as_matrix(),rng.uniform(-.25,.25,3)) for _ in range(30)])
    c=np.array([inverse(x)@t@y for t in g])
    for i in range(len(c)):c[i]=retract(c[i],rng.normal(0,noise,6))
    return g,c,x,y


def test_recovers_both_transforms_and_direction():
    g,c,x,y=fixture()
    result=solve(g,c)
    assert result['converged']
    np.testing.assert_allclose(result['T_mount_from_camera'],x,atol=1e-7)
    np.testing.assert_allclose(result['T_hand_from_board'],y,atol=1e-7)


def test_noisy_refinement_and_heldout():
    g,c,x,y=fixture(.0004)
    result=solve(g[:24],c[:24])
    xn,yn=np.array(result['T_mount_from_camera']),np.array(result['T_hand_from_board'])
    assert result['final_loss'] <= result['initial_loss']
    assert np.linalg.norm(xn[:3,3]-x[:3,3]) < .002
    assert np.linalg.norm(yn[:3,3]-y[:3,3]) < .002
    assert np.max(np.abs(residual(g[24:],c[24:],xn,yn,1,1))) < .003


def test_gradient_matches_directional_difference():
    g,c,x,y=fixture(.001); rng=np.random.default_rng(33)
    direction=rng.normal(size=12);eps=2e-7
    j=jacobian(g,c,x,y)
    diff=(residual(g,c,retract(x,eps*direction[:6]),retract(y,eps*direction[6:]))-
          residual(g,c,retract(x,-eps*direction[:6]),retract(y,-eps*direction[6:])))/(2*eps)
    np.testing.assert_allclose(j@direction,diff,rtol=1e-5,atol=1e-4)


def test_rejects_unobservable_and_bad_matrices():
    g,c,x,y=fixture()
    g[:,:3,:3]=np.eye(3)
    with pytest.raises(ValueError,match='independent axes'):solve(g,c)
    g,c,_,_=fixture();c[0,0,0]=8
    with pytest.raises(ValueError,match='rigid transform'):solve(g,c)
