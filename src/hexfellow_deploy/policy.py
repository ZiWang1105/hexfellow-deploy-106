"""Optional asynchronous policy component, outside the control clock."""
from concurrent.futures import Future
import threading
import json
from pathlib import Path
import time
import yaml
from deploy_anything.components.interface.base_interface import DABaseInterface
from .workspace import activate


def request_async(callback, observation):
    """A single cancellable wait; shutdown need not wait for an HTTP response."""
    future = Future()
    def run():
        try:
            future.set_result(callback(observation))
        except BaseException as error:
            future.set_exception(error)
    threading.Thread(target=run, name='policy-api', daemon=True).start()
    return future


class AgentComponent(DABaseInterface):
    def __init__(self, name, request, status, decision, images, workspace, output,
                 prompt, agent_config='configs/agent/rgbd_full.yaml', time_budget=.1,
                 model='openai/gpt-6-astra', reasoning_effort='medium'):
        super().__init__(name, [request, status, decision, images], time_budget)
        self.request, self.status, self.decision, self.images = request, status, decision, images
        self.workspace, self.output, self.prompt, self.agent_config = workspace, output, prompt, agent_config
        self.model, self.reasoning_effort = model, reasoning_effort

    def initialize(self):
        if not self.prompt.strip():
            raise ValueError('Agent instruction must not be empty')
        self._initialized.value = True
        return True

    def clean_up(self):
        return True

    def run(self):
        activate(self.workspace)
        from mobile_manipulator.agent.policy import OpenRouterPolicy
        from mobile_manipulator.interfaces.contract import Observation
        from mobile_manipulator.control.base_motion import BaseMotionConfig
        from mobile_manipulator.robot.kinematics import Kinematics
        from mobile_manipulator.robot.assembly import load_assembly
        # Robot-only configuration; this component has no plant-state topic.
        kin = Kinematics(Path('third_party/UMI_Hardware/urdf/maver_x4_gp80_zed_40deg.urdf'),
                         load_assembly('configs/robot/maver_gp80.yaml'))
        config = yaml.safe_load(Path(self.agent_config).read_text())
        config.update(model=self.model, reasoning_effort=self.reasoning_effort,
                      max_calls=float('inf'), max_cost_usd=None, max_episode_s=float('inf'))
        system = Path('prompts/system.txt').read_text() + BaseMotionConfig.load().prompt_context()
        client = None
        future = None
        identifier = 0
        decision_state = {}
        def publish(**changes):
            decision_state.update(changes, published_s=time.monotonic())
            self.decision.put(decision_state, timeout=.001)
            with (Path(self.output)/'decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(decision_state, allow_nan=False)+'\n')
        self._ready.value = True
        # API waiting cannot block either control or the diagnostic component.
        try:
            while not self.stopping:
                state = self.status.get()
                if not state or state.get('phase') != 'running' or state.get('diagnostic', {}).get('mode') == 'startup_carry':
                    time.sleep(.1)
                    continue
                frame = self.images.get()
                if frame is None or time.monotonic()-frame['measured']['timestamp_s'] > .5:
                    time.sleep(.05)
                    continue
                identifier += 1
                measured = frame['measured']
                kin.update(measured['joint_positions'])
                observation = Observation(identifier, measured['timestamp_s'], frame['rgb'], frame['depth_m'],
                                          {**measured, **kin.packet()})
                if client is None:
                    client = OpenRouterPolicy(self.prompt, system, frame['intrinsics'], config, Path(self.output)/'policy')
                publish(phase='thinking', observation_id=identifier, model=config['model'], error=None)
                future = request_async(client.act, observation)
                while not future.done() and not self.stopping:
                    time.sleep(.05)
                if self.stopping:
                    break
                try:
                    action, record = future.result()
                except Exception as error:
                    publish(phase='api_error', error=str(error), policy_calls=client.calls)
                    time.sleep(.5)
                    continue
                proposal = dict(id=identifier, operation='target', action=action.to_dict(), observed=measured)
                while not self.stopping and not self.request.put(proposal):
                    time.sleep(.001)
                publish(phase='submitted', action=action.to_dict(), observation_id=identifier,
                        action_observation_id=identifier, policy_calls=client.calls, cost_usd=client.cost)
                until = time.monotonic()+float(config['action_horizon_s'])
                while not self.stopping:
                    state = self.status.get()
                    if state and state.get('phase') in ('fault', 'closed'):
                        break
                    if time.monotonic() >= until:
                        break
                    if state and state.get('request_id') == identifier:
                        if state.get('request_error'):
                            client.reject_stale_action(record)
                            publish(phase='rejected', error=state['request_error'])
                            break
                        diagnostic = state.get('diagnostic', {})
                        if state.get('phase') == 'fault' or diagnostic.get('needs_observation') or diagnostic.get('arrived'):
                            break
                    time.sleep(.05)
                identifier += 1
                while not self.stopping and not self.request.put(dict(id=identifier, operation='hold')):
                    time.sleep(.001)
                while not self.stopping:
                    state = self.status.get()
                    if state and state.get('phase') == 'fault':
                        break
                    if state and state.get('request_id') == identifier and state.get('diagnostic', {}).get('ik_status') == 'HOLD':
                        break
                    time.sleep(.05)
        finally:
            publish(phase='stopped', policy_calls=0 if client is None else client.calls,
                    cost_usd=0 if client is None else client.cost,
                    request_interrupted=future is not None and not future.done())
