"""Discrete lift boundary viability with measured-origin position references."""
from types import SimpleNamespace

import numpy as np
import pytest

from hexfellow_deploy.robotics.control.command_limits import BoundaryStateUnrecoverable, FiniteBaseBoundaryGovernor
from hexfellow_deploy.robotics.control.ik_core import PinkIKCore
from hexfellow_deploy.robotics.control.motion_profile import controller_config_for_profile
from hexfellow_deploy.robotics.control.robot import LIFT_JOINT


def governor():
    model=SimpleNamespace(nv=1,lowerPositionLimit=np.array([.002]),
        upperPositionLimit=np.array([.398]),velocityLimit=np.array([.025]))
    return FiniteBaseBoundaryGovernor(model,[0],np.array([.5]),np.array([40.]))


def bounds(gov,q,v,a,dt):
    """The same scalar configuration, acceleration, jerk and velocity rows."""
    lower=np.maximum.reduce([-gov.vmax,v+np.maximum(-gov.amax,a-gov.jmax*dt)*dt,
        -.9*(q-gov.model.lowerPositionLimit)/dt,
        -np.sqrt(2*gov.amax*(q-gov.model.lowerPositionLimit))])
    upper=np.minimum.reduce([gov.vmax,v+np.minimum(gov.amax,a+gov.jmax*dt)*dt,
        .9*(gov.model.upperPositionLimit-q)/dt,
        np.sqrt(2*gov.amax*(gov.model.upperPositionLimit-q))])
    return lower,upper


@pytest.mark.parametrize('direction',[-1,1])
def test_continuous_braking_distance_alone_enters_discrete_infeasible_state(direction):
    gov=governor()
    q=np.array([.012 if direction<0 else .388])
    v,a=np.zeros(1),np.zeros(1)
    for _ in range(100):
        lower,upper=bounds(gov,q,v,a,.02)
        if lower[0]>upper[0]+1e-8:
            return
        next_v=np.clip(direction*gov.vmax,lower,upper)
        q+=next_v*.02
        v,a=next_v,(next_v-v)/.02
    pytest.fail('The previous rows should reproduce the discrete infeasibility')


@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('response',[.7,1.])
def test_measured_origin_lift_stops_with_all_original_rows_feasible(direction,response):
    gov=governor()
    q=np.array([.012 if direction<0 else .388])
    reference=q.copy()
    v,a,measured_v=np.zeros(1),np.zeros(1),np.zeros(1)
    for _ in range(300):
        lower,upper=bounds(gov,q,v,a,.02)
        for limit in gov.limits(q,reference,measured_v,v,a,.04,np.zeros(1),.02):
            for row,cap in zip(limit.jacobian,limit.bound,strict=True):
                if row[0]>0:
                    lower[0]=max(lower[0],-cap)
                else:
                    upper[0]=min(upper[0],cap)
        assert lower[0]<=upper[0]+1e-8
        next_v=np.clip(direction*gov.vmax,lower,upper)
        next_a=(next_v-v)/.02
        assert abs(next_v[0])<=.025+1e-10
        assert abs(next_a[0])<=.5+1e-8
        assert abs((next_a-a)[0]/.02)<=40.+1e-6
        # The SDK receives measured q + one solved step, not an accumulated lift q.
        reference=q+next_v*.02
        next_q=q+response*(reference-q)
        measured_v=(next_q-q)/.02
        q,v,a=next_q,next_v,next_a
        assert gov.model.lowerPositionLimit[0]<=q[0]<=gov.model.upperPositionLimit[0]
    assert abs(v[0])<1e-6
    assert .008<direction*(q[0]-(.012 if direction<0 else .388))<.01


def test_already_infeasible_recorded_state_is_rejected_without_resetting_history():
    gov=governor()
    q=np.array([.3977130408830765])
    v=np.array([.025])
    a=np.array([0.])
    with pytest.raises(BoundaryStateUnrecoverable):
        gov.limits(q,q,np.array([.01814278775856]),v,a,.042218485,np.zeros(1),.02)
    assert q[0]==.3977130408830765
    assert v[0]==.025 and a[0]==0.


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('measured_speed', [0., -1e-5])
def test_recorded_endpoint_inward_command_does_not_reserve_outward_travel(direction, measured_speed):
    # 20261008T234924.788901Z: upper clearance was one encoder pulse,
    # measured lift was stationary and the previous command pointed inward.
    gov = governor()
    clearance = 7.490283230326966e-7
    endpoint = gov.model.upperPositionLimit[0] if direction > 0 else gov.model.lowerPositionLimit[0]
    q = np.array([endpoint - direction * clearance])
    previous = np.array([-direction * 1.978781328327739e-5])
    reference = q + previous * .02
    measured = np.array([direction * measured_speed])
    reaction = .04529894500505179
    (limit,) = gov.limits(q, reference, measured, previous, np.zeros(1),
                         reaction, np.zeros(1), .02)
    label = f'joint_{"upper" if direction > 0 else "lower"}:0'
    row = limit.labels.index(label)
    assert limit.clearance[row] == pytest.approx(clearance, abs=1e-16)
    # No outward command is allowed at this endpoint, even during recovery.
    assert limit.bound[row] == 0.


@pytest.mark.parametrize('direction', [-1, 1])
def test_stationary_reference_still_reserves_actual_outward_motion(direction):
    gov = governor()
    q = np.array([.2])
    measured = np.array([direction * .005])
    (limit,) = gov.limits(q, q, measured, np.zeros(1), np.zeros(1),
                         .04, np.zeros(1), .02)
    row = limit.labels.index(f'joint_{"upper" if direction > 0 else "lower"}:0')
    clearance = ((gov.model.upperPositionLimit[0] - q[0]) if direction > 0
                 else (q[0] - gov.model.lowerPositionLimit[0]))
    assert limit.clearance[row] == pytest.approx(clearance - .005 * .04)
    endpoint = gov.model.upperPositionLimit[0] if direction > 0 else gov.model.lowerPositionLimit[0]
    q = np.array([endpoint - direction * 7.490283230326966e-7])
    previous = np.array([-direction * 1.978781328327739e-5])
    with pytest.raises(BoundaryStateUnrecoverable):
        gov.limits(q, q, measured, previous, np.zeros(1), .04, np.zeros(1), .02)


@pytest.mark.parametrize('direction',[-1,1])
def test_fixed_base_core_applies_lift_governor_without_changing_reference_semantics(direction):
    config=controller_config_for_profile('x7-gp80-highspeed-fixed-base-v1')
    core=PinkIKCore(config=config,base_mode='fixed',runtime_velocity_bounds={LIFT_JOINT:.025},
        runtime_position_bounds={LIFT_JOINT:(0.,.4)})
    assert core._persistent_start==1
    assert core._lift_reference_governor is None
    assert core._measured_lift_boundary_governor is not None
    q=core.home_q.copy()
    q[0]=.012 if direction<0 else .388
    core.configuration.update(q)
    targets={side:core.configuration.get_transform_frame_to_world(task.frame).copy()
             for side,task in core.frame_tasks.items()}
    v=np.zeros(15)
    core.reset(q,targets,measured_velocity=v,state_timestamp=0.)
    goal=q.copy()
    goal[0]=core.model.lowerPositionLimit[0] if direction<0 else core.model.upperPositionLimit[0]
    previous_v,previous_a=v.copy(),v.copy()
    for step in range(180):
        command=core.step(q,targets,measured_velocity=v,state_timestamp=step*.02,
            posture_target=goal,posture_velocity=np.zeros(15))
        assert command.executable,command.diagnostics.fault
        assert command.positions[0]==pytest.approx(q[0]+command.velocity[0]*.02,abs=1e-10)
        acceleration=(command.velocity-previous_v)/.02
        assert abs(acceleration[0])<=.5+1e-6
        assert abs((acceleration-previous_a)[0]/.02)<=40.+1e-4
        previous_v,previous_a=command.velocity.copy(),acceleration
        next_q=command.positions.copy()
        v=(next_q-q)/.02
        q=next_q
    assert abs(command.velocity[0])<1e-6


@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('excursion',[6.137472715e-8,.0008])
@pytest.mark.parametrize('inward_speed',[0.,.005])
def test_physical_margin_excursion_recovers_inward_without_changing_raw_feedback(side,excursion,inward_speed):
    core=PinkIKCore(config=controller_config_for_profile('x7-gp80-highspeed-fixed-base-v1'),
        base_mode='fixed',runtime_velocity_bounds={LIFT_JOINT:.025},
        runtime_position_bounds={LIFT_JOINT:(0.,.4)})
    bound=core.model.lowerPositionLimit[0] if side<0 else core.model.upperPositionLimit[0]
    q=core.home_q.copy()
    q[0]=bound+side*excursion
    original=q.copy()
    core.configuration.update(q)
    targets={s:core.configuration.get_transform_frame_to_world(t.frame).copy()
             for s,t in core.frame_tasks.items()}
    v=np.zeros(15)
    v[0]=-side*inward_speed
    core.reset(q,targets,measured_velocity=v,state_timestamp=0.)
    previous_v=core._last_velocity.copy()
    goal=q.copy()
    goal[0]=bound-side*.01
    command=core.step(q,targets,measured_velocity=v,state_timestamp=0.,
        posture_target=goal,posture_velocity=np.zeros(15))
    assert command.executable,command.diagnostics.fault
    np.testing.assert_array_equal(q,original)
    np.testing.assert_array_equal(core._last_measured_q,original)
    assert side*command.velocity[0]<=0.
    assert side*(command.positions[0]-bound)<0.
    assert abs(command.velocity[0]-previous_v[0])<=.5*.02+1e-8
