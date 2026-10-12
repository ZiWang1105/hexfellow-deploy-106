"""Endpoint admission and brake supervision without connected hardware."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
import time

from hexfellow_deploy.workspace import activate
ROOT=activate(Path(__file__).resolve().parents[1])
from hexfellow_deploy.actuation import deployment
from hexfellow_deploy.native import StreamEndpoint, NativeDriver, ContinuousGate
from hexfellow_deploy.hardware.umi_endpoint import MobileEndpoint


def endpoint(height=.422):
    e=object.__new__(StreamEndpoint)
    e.mapping=deployment(ROOT,Path('configs/hardware/desktop.yaml'))[1]
    e.fixed_lift_reference=height
    e.fixed_lift_confirmation={'acceptance':'observed_stationary_after_brake_receipt'}
    e.fixed_lift_brake_evidence=lambda:dict(brake_received=True,no_pending_motion=True,acknowledged_at=10.)
    s=SimpleNamespace(q=np.r_[np.zeros(3),height,np.zeros(14)],v=np.zeros(18),
        received=dict(theta=time.monotonic()),health=dict(theta=dict(range=[0.,.422],state='LsBrake')))
    return e,s


@pytest.mark.parametrize('height',[.3,.420,.421,.4219988120410797,.422])
def test_fixed_brake_admits_actual_travel(height):
    e,s=endpoint(height)
    e.check_fixed_lift(s)
    e.validate_lift_command(SimpleNamespace(positions=s.q,velocity=s.v))


@pytest.mark.parametrize('sdk_counts', [0, 1, 2, 3])
def test_recorded_encoder_endpoint_rounding_preserves_raw_height(sdk_counts):
    # Reconstructed from the October 9 20:39 run's paired SDK/model positions.
    # One count is 0.749028 micrometers; the nominal 422-mm endpoint rounds
    # to 563397 counts, or 422.0003100977257 mm.
    pulses = 1335063.
    high = 563397 / pulses
    height = high - sdk_counts / pulses
    e, s = endpoint(height)
    s.health['theta'].update(range=[0., high], pulse_per_meter=pulses)
    e.check_fixed_lift(s)
    assert s.q[3] == height  # Never clip physical feedback used by FK.
    if sdk_counts == 0:
        assert height > .422  # Reproduces the old exact model-bound rejection.


@pytest.mark.parametrize('sdk_fraction', [-.1, 563397.1])
def test_nominal_rounding_does_not_expand_sdk_range(sdk_fraction):
    pulses = 1335063.
    high = 563397 / pulses
    e, s = endpoint(high - sdk_fraction / pulses)
    s.health['theta'].update(range=[0., high], pulse_per_meter=pulses)
    with pytest.raises(RuntimeError, match='physical device/model range'):
        e.validate_lift_start(s)
    assert e.fixed_lift_range_failure['nominal_rounding_tolerance_m'] == .5 / pulses


@pytest.mark.parametrize('pulses', [None, 1335063., 1.])
def test_unknown_resolution_or_large_reported_endpoint_does_not_expand_model(pulses):
    high = .422002 if pulses is not None else 563397 / 1335063.
    e, s = endpoint(high)
    s.health['theta'].update(range=[0., high], pulse_per_meter=pulses)
    with pytest.raises(RuntimeError, match='physical device/model range'):
        e.validate_lift_start(s)


@pytest.mark.parametrize('height',[-.001,.422001,.435,float('nan')])
def test_fixed_brake_still_rejects_outside_physical_travel(height):
    e,s=endpoint(height)
    with pytest.raises(RuntimeError,match='physical|finite'):
        e.validate_lift_start(s)


@pytest.mark.parametrize('failure',['brake','position','velocity','device_range','stale','unconfirmed'])
def test_fixed_brake_loss_or_motion_fails(failure):
    e,s=endpoint(.421)
    if failure=='brake': e.fixed_lift_brake_evidence=lambda:dict(brake_received=False)
    if failure=='position': s.q[3]-=.003
    if failure=='velocity': s.v[3]=.003
    if failure=='device_range': s.health['theta']['range']=[0.,.4]
    if failure=='stale': s.received['theta']-=.1
    if failure=='unconfirmed': e.fixed_lift_confirmation=None
    with pytest.raises(RuntimeError): e.check_fixed_lift(s)


def test_endpoint_rejects_lift_motion_request():
    e,s=endpoint()
    v=s.v.copy(); v[3]=.001
    with pytest.raises(RuntimeError,match='requested motion'):
        e.validate_lift_command(SimpleNamespace(positions=s.q,velocity=v))


@pytest.mark.parametrize('missing_receipt', [False, True])
def test_acquisition_requires_new_brake_feedback_and_sends_no_position(monkeypatch, missing_receipt):
    e,s=endpoint()
    clock=[10.]
    monkeypatch.setattr('hexfellow_deploy.native.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('hexfellow_deploy.native.time.sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    monkeypatch.setattr(MobileEndpoint,'acquire',lambda *args:s)
    calls=[]
    e.command_type=SimpleNamespace(BRAKE='BRAKE')
    e.devices=dict(theta=SimpleNamespace(motor_command=lambda *args:calls.append(args)))
    e.timeout=2.
    e.direct_gates={n:ContinuousGate(n,clock=lambda:clock[0],defer_activation=True)
                    for n in ('left','right','theta','base')}
    for gate in e.direct_gates.values():
        gate.transition('initializing'); gate.transition('active')
    clock[0] += 1.2  # Setup lasts longer than the previous active startup lease.
    if missing_receipt:
        e.fixed_lift_brake_evidence=lambda:dict(brake_received=False,no_pending_motion=False)
    e.check_health=lambda *args:None
    e.check_transport=lambda:None
    def read():
        result=deepcopy(s)
        result.received['theta']=clock[0]
        result.health['theta']['state']='LsAlgorithmControl'
        return result
    e.state=read
    if missing_receipt:
        with pytest.raises(RuntimeError,match='brake was not confirmed'):
            e.acquire('whole-body',None)
        assert all(e.direct_gates[n].phase=='ready' for n in ('left','right','base'))
        return
    result=e.acquire('whole-body',None)
    assert result.health['theta']['state']=='LsAlgorithmControl'
    assert clock[0]>=11.7 and calls==[('BRAKE',True)]
    assert e.fixed_lift_reference==.422
    assert e.fixed_lift_confirmation['firmware_brake_confirmed'] is False
    assert e.direct_gates['theta'].phase=='holding'
    assert all(e.direct_gates[n].phase=='active' for n in ('left','right','base'))


@pytest.mark.parametrize('operation', ['receipt', 'check'])
def test_receipt_and_poll_cannot_ignore_brake_loss(operation):
    e,s=endpoint(); s.health['theta']['state']='LsEmergencyStop'
    e.direct_gates={}
    e.check_transport=lambda:None
    e.check_health=lambda *a,**kw:None
    driver=object.__new__(NativeDriver); driver.endpoint=e; driver.state=s
    driver.acquired=True
    with pytest.raises(RuntimeError,match='brake lost'):
        if operation=='receipt': driver.acknowledgement(1)
        else: driver.check()
