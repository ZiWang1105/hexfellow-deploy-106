import json
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

from hexfellow_deploy.policy import AgentComponent
from hexfellow_deploy.workspace import activate


@pytest.mark.parametrize('include_wrist', [False, True])
def test_policy_boundary_and_decision_archive(monkeypatch, tmp_path, include_wrist):
    """Exercise the component contract with a stub API; no model/hardware claim."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    activate(root)
    from mobile_manipulator.agent import policy
    from mobile_manipulator.interfaces.contract import Action, MEASURED_FIELDS
    from mobile_manipulator.sim.scene_config import load_scene
    from hexfellow_deploy.camera import simulation_kinematics
    joints = load_scene(root/'configs/scenes/apple_on_table.yaml').data['robot']['initial_joints']
    measured = dict(timestamp_s=time.monotonic(), joint_positions=dict(joints),
                    joint_velocities={n:0. for n in joints}, gripper_state={'left':'closed','right':'closed'})
    frame = dict(measured=measured, rgb=np.zeros((360,640,3), dtype=np.uint8),
                 depth_m=np.ones((360,640), dtype=np.float32), intrinsics=np.eye(3).tolist())
    kin = simulation_kinematics(load_scene(root/'configs/scenes/apple_on_table.yaml'))
    kin.update(measured['joint_positions'])
    frame['robot_fk'] = kin.packet()
    action = Action.from_dict(json.loads((Path(__file__).parent/'fixtures/paired_pose.json').read_text()))
    observations = []
    from mobile_manipulator.control.base_motion import BaseMotionConfig
    base_config = tmp_path/'selected_base.yaml'
    base_config.write_text(yaml.safe_dump(dict(BaseMotionConfig.load().to_dict(), approach_distance_m=.61)))
    class Client:
        def __init__(self, prompt, system, intrinsics, config, directory):
            self.calls, self.cost = 0, 0.
            assert config['model'] == 'openai/gpt-6-astra'
            assert config['reasoning_effort'] == 'medium'
            assert config['max_cost_usd'] is None
            assert config['max_calls'] == float('inf')
            assert 'Object approach distance 0.61 m' in system
        def act(self, observation):
            observations.append(observation)
            self.calls += 1
            return action, {}
    monkeypatch.setattr(policy, 'OpenRouterPolicy', Client)
    class Topic:
        name = 'stub'
        def __init__(self, value=None):
            self.value = value
        def get(self, **kwargs):
            return self.value
        def put(self, value, **kwargs):
            self.value = value.copy()
            return True
    status = Topic(dict(phase='running', diagnostic={'mode':'hold'}))
    class Requests(Topic):
        def put(self, value, **kwargs):
            if value['operation'] == 'hold':
                agent.signal_stop()
            else:
                assert set(value['observed']) == MEASURED_FIELDS
                status.value = dict(phase='running', request_id=value['id'], diagnostic={'arrived': True})
            return super().put(value, **kwargs)
    decision = Topic()
    agent = AgentComponent('test_agent', Requests(), status, decision, Topic(frame), str(root), str(tmp_path),
                           'Diagnostic paired target', base_config=str(base_config),
                           wrist_images=Topic(dict(rgb=frame['rgb'],received_s=measured['timestamp_s'])) if include_wrist else None)
    agent.initialize()
    agent.run()
    assert len(observations) == 1
    assert set(observations[0].proprioception) == policy.PROPRIOCEPTION_FIELDS
    assert observations[0].timestamp_s == measured['timestamp_s']
    assert observations[0].cameras == {}
    assert set(observations[0].rgb_cameras) == ({'wrist'} if include_wrist else set())
    if include_wrist:
        assert set(observations[0].rgb_cameras['wrist']) == {'rgb','received_s'}
    events = [json.loads(line) for line in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert [row['phase'] for row in events] == ['waiting_for_control','thinking','submitted','executing','stopped']
    assert events[0]['policy_calls'] == 0 and events[0]['prompt'] == 'Diagnostic paired target'
    assert Action.from_dict(events[-1]['action']) == action


@pytest.mark.parametrize('reason,calls,cost,limits', [
    ('agent.max_calls', 1, 0., dict(max_calls=1,max_cost_usd=None)),
    ('agent.max_cost_usd', 0, .1, dict(max_calls=10,max_cost_usd=.1)),
])
def test_explicit_exhausted_budget_stops_without_request_retry(monkeypatch, tmp_path, reason, calls, cost, limits):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    from mobile_manipulator.agent import policy
    config_path = tmp_path/'agent.yaml'
    config_path.write_text(yaml.safe_dump(dict(limits,max_episode_s=20,action_horizon_s=1)))
    class Client:
        def __init__(self, prompt, system, intrinsics, config, directory):
            self.calls, self.cost = calls, cost
            assert config['max_calls'] == limits['max_calls']
            assert config['max_cost_usd'] == limits['max_cost_usd']
        def act(self, observation):
            raise AssertionError('An exhausted budget must not call or retry the API')
    class Topic:
        name = 'stub'
        def __init__(self, value=None):
            self.value = value
        def get(self, **kwargs):
            return self.value
        def put(self, value, **kwargs):
            self.value = value.copy()
            return True
    monkeypatch.setattr(policy, 'OpenRouterPolicy', Client)
    decision = Topic()
    frame = dict(measured=dict(timestamp_s=time.monotonic()),robot_fk=dict(test='unused'),intrinsics=[])
    agent = AgentComponent('test_budget', Topic(), Topic(dict(phase='running')), decision, Topic(frame),
        str(root), str(tmp_path), 'Diagnostic', agent_config=str(config_path), respect_limits=True)
    agent.initialize()
    agent.run()
    assert decision.value['budget_exhausted'] is True
    assert decision.value['stop_reason'] == reason
    assert decision.value['policy_calls'] == calls
    assert agent.stopping
    events = [json.loads(line) for line in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert [row['phase'] for row in events] == ['waiting_for_control','budget_exhausted','stopped']


def test_startup_and_failure_are_visible_before_any_api_call(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    from mobile_manipulator.agent import policy
    def forbidden(*args, **kwargs):
        raise AssertionError('Startup must not construct the model client')
    monkeypatch.setattr(policy, 'OpenRouterPolicy', forbidden)
    class Topic:
        name = 'stub'
        def __init__(self, value=None): self.value = value
        def get(self, **kwargs): return self.value
        def put(self, value, **kwargs): self.value = value.copy(); return True
    status = Topic(dict(phase='running', diagnostic=dict(mode='startup_carry', startup_stage=1, startup_stages=2)))
    class Decisions(Topic):
        def put(self, value, **kwargs):
            if value['phase'] == 'waiting_for_startup':
                status.value = dict(phase='fault', fault='test publication failure')
            return super().put(value, **kwargs)
    decision = Decisions()
    agent = AgentComponent('startup_test', Topic(), status, decision, Topic(), str(root), str(tmp_path), 'Pick up the orange.')
    agent.initialize()
    agent.run()
    events = [json.loads(line) for line in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert [row['phase'] for row in events] == ['waiting_for_control', 'waiting_for_startup', 'stopped']
    assert 'stage 1/2' in events[1]['message']
    assert events[-1]['policy_calls'] == 0
    assert 'before the first model call' in events[-1]['message']
    assert events[-1]['error'] == 'test publication failure'
