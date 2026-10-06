import json
from pathlib import Path
import time

import numpy as np

from hexfellow_deploy.policy import AgentComponent
from hexfellow_deploy.workspace import activate


def test_policy_boundary_and_decision_archive(monkeypatch, tmp_path):
    """Exercise the component contract with a stub API; no model/hardware claim."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    activate(root)
    from mobile_manipulator.agent import policy
    from mobile_manipulator.interfaces.contract import Action, MEASURED_FIELDS
    from mobile_manipulator.sim.scene_config import load_scene
    joints = load_scene(root/'configs/scenes/apple_on_table.yaml').data['robot']['initial_joints']
    measured = dict(timestamp_s=time.monotonic(), joint_positions=dict(joints),
                    joint_velocities={n:0. for n in joints}, gripper_state={'left':'closed','right':'closed'})
    frame = dict(measured=measured, rgb=np.zeros((360,640,3), dtype=np.uint8),
                 depth_m=np.ones((360,640), dtype=np.float32), intrinsics=np.eye(3).tolist())
    action = Action.from_dict(json.loads((Path(__file__).parent/'fixtures/paired_pose.json').read_text()))
    observations = []
    class Client:
        def __init__(self, prompt, system, intrinsics, config, directory):
            self.calls, self.cost = 0, 0.
            assert config['model'] == 'openai/gpt-6-astra'
            assert config['reasoning_effort'] == 'medium'
            assert config['max_cost_usd'] is None
            assert config['max_calls'] == float('inf')
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
                           'Diagnostic paired target')
    agent.initialize()
    agent.run()
    assert len(observations) == 1
    assert set(observations[0].proprioception) == policy.PROPRIOCEPTION_FIELDS
    assert observations[0].timestamp_s == measured['timestamp_s']
    assert observations[0].cameras == {}
    events = [json.loads(line) for line in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert [row['phase'] for row in events] == ['thinking','submitted','stopped']
    assert Action.from_dict(events[-1]['action']) == action
