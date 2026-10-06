from copy import deepcopy

import numpy as np
import pytest

from hexfellow_deploy.actuation import base_reference_next


LIMITS = dict(dt_s=.02, base_linear=.4, base_angular=.25,
              base_acceleration=1.2, base_angular_acceleration=2.4)


def test_profile_bound_first_step_and_no_mutation():
    requested, previous = [.05, 0., .1], [0., 0., 0.]
    limits = deepcopy(LIMITS)
    assert base_reference_next(requested, previous, limits) == pytest.approx([.024, 0., .048])
    assert requested == [.05, 0., .1] and previous == [0., 0., 0.] and limits == LIMITS


def test_diagonal_translation_uses_vector_norm_and_reversal_brakes_without_overshoot():
    previous = np.array([.12, .16, .2])
    requested = np.array([-.12, -.16, -.2])
    for _ in range(30):
        current = np.array(base_reference_next(requested, previous, LIMITS))
        change = current-previous
        assert np.linalg.norm(change[:2]) <= .024+1e-12
        assert abs(change[2]) <= .048+1e-12
        assert np.linalg.norm(current[:2]) <= LIMITS['base_linear']+1e-12
        assert abs(current[2]) <= LIMITS['base_angular']+1e-12
        previous = current
    np.testing.assert_allclose(previous, requested)


def test_zero_request_reaches_exact_zero_within_existing_park_delay_for_full_native_request():
    previous = [.2, 0., .25]
    count = 0
    while any(previous):
        previous = base_reference_next([0., 0., 0.], previous, LIMITS)
        count += 1
        assert count <= 13
    assert count*LIMITS['dt_s'] <= .25


@pytest.mark.parametrize('requested,previous', [([.5, 0., 0.], [0.]*3),
    ([.3, .3, 0.], [0.]*3), ([0., 0., .3], [0.]*3),
    ([0.]*3, [.5, 0., 0.]), ([float('nan'), 0., 0.], [0.]*3),
    ([True, 0., 0.], [0.]*3), ([0., 0.], [0.]*3)])
def test_invalid_reference_fails_instead_of_hiding_request(requested, previous):
    with pytest.raises(ValueError):
        base_reference_next(requested, previous, LIMITS)


@pytest.mark.parametrize('key,value', [('dt_s', 0.), ('base_acceleration', -1.),
    ('base_angular_acceleration', float('nan')), ('base_linear', True)])
def test_invalid_trusted_limits_fail(key, value):
    with pytest.raises(ValueError):
        base_reference_next([0.]*3, [0.]*3, dict(LIMITS, **{key: value}))
