"""Optional asynchronous policy component, outside the control clock."""
from concurrent.futures import Future
from collections import deque
import threading
import json
import math
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


def wrist_context(topic, head_timestamp, now, max_age=.5, max_skew=.15, name="wrist"):
    """Do not send stale context or pretend the two cameras are synchronized."""
    from mobile_manipulator.interfaces.contract import validate_rgb_camera
    frame = topic.get()
    if frame is None:
        return None
    validate_rgb_camera(frame)
    if not all(math.isfinite(x) for x in (head_timestamp,now)):
        raise ValueError('Invalid observation clock')
    age = now-frame['received_s']
    if not 0 <= age <= max_age or abs(frame['received_s']-head_timestamp) > max_skew:
        return None
    return {name:frame}


class AgentComponent(DABaseInterface):
    def __init__(self, name, request, status, decision, images, workspace, output,
                 prompt, agent_config='configs/agent/rgbd_full.yaml', time_budget=.1,
                 model='openai/gpt-6-astra', reasoning_effort='medium', respect_limits=False, base_config=None,
                 wrist_images=None, wrist_topics=None):
        self.wrist_topics = dict(wrist_topics or {})
        if wrist_images is not None:
            self.wrist_topics['wrist'] = wrist_images
        super().__init__(name, [request, status, decision, images]
                         +list(self.wrist_topics.values()), time_budget)
        self.wrist_images = wrist_images
        self.request, self.status, self.decision, self.images = request, status, decision, images
        self.workspace, self.output, self.prompt, self.agent_config = workspace, output, prompt, agent_config
        self.model, self.reasoning_effort = model, reasoning_effort
        self.respect_limits = bool(respect_limits)
        self.base_config = base_config

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
        config = yaml.safe_load(Path(self.agent_config).read_text())
        config.update(model=self.model, reasoning_effort=self.reasoning_effort)
        if not self.respect_limits:
            config.update(max_calls=float('inf'), max_cost_usd=None, max_episode_s=float('inf'))
        base = BaseMotionConfig.load() if self.base_config is None else BaseMotionConfig.load(self.base_config)
        system = Path('prompts/system.txt').read_text() + base.prompt_context()
        client = None
        future = None
        identifier = 0
        decision_state = {}
        events = deque(maxlen=12)
        last_state = None
        budget_exhausted = False
        api_blocked = False
        failures = 0
        def stopped():
            nonlocal last_state
            state = self.status.get()
            if state is not None:
                last_state = state
            return self.stopping or bool(state and (state.get('final_hold') or state.get('phase') in ('fault', 'closed')))
        def publish(**changes):
            decision_state.update(changes, published_s=time.monotonic())
            # Keep the report once in the snapshot; duplicating long summaries
            # in every event can exhaust the fixed-size diagnostic topic.
            events.append(dict({k: v for k, v in changes.items() if k not in
                                ('decision_summary', 'grounding', 'execution_feedback')},
                               observed_s=decision_state['published_s']))
            decision_state['events'] = list(events)
            self.decision.put(decision_state, timeout=.001)
            with (Path(self.output)/'decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(decision_state, allow_nan=False)+'\n')
        self._ready.value = True
        publish(phase='waiting_for_control', message='Waiting for controller startup; no model call yet.',
                prompt=self.prompt, model=self.model, reasoning_effort=self.reasoning_effort,
                policy_calls=0, cost_usd=0.)
        last_wait = None
        last_heartbeat = time.monotonic()
        def waiting(phase, message):
            nonlocal last_wait, last_heartbeat
            now = time.monotonic()
            if (phase, message) != last_wait or now-last_heartbeat >= 1.:
                publish(phase=phase, message=message)
                last_wait, last_heartbeat = (phase, message), now
        # API waiting cannot block either control or the diagnostic component.
        try:
            while not stopped():
                if api_blocked:
                    # Preserve controller HOLD, camera and diagnostics without
                    # repeatedly issuing a rejected provider request.
                    time.sleep(.1)
                    continue
                state = self.status.get()
                if state and state.get('final_hold'):
                    break
                if not state or state.get('phase') != 'running' or state.get('diagnostic', {}).get('mode') == 'startup_carry':
                    diagnostic = (state or {}).get('diagnostic', {})
                    if diagnostic.get('mode') == 'startup_carry':
                        waiting('waiting_for_startup',
                                f"Preparing carrying pose: stage {diagnostic.get('startup_stage', '?')}/"
                                f"{diagnostic.get('startup_stages', '?')}. No model call yet.")
                    else:
                        waiting('waiting_for_control', 'Waiting for controller readiness.')
                    time.sleep(.1)
                    continue
                frame = self.images.get()
                if frame is None or time.monotonic()-frame['measured']['timestamp_s'] > .5:
                    waiting('waiting_for_camera', 'Waiting for a fresh matched RGB-D observation.')
                    time.sleep(.05)
                    continue
                rgb_cameras = {}
                missing = []
                now = time.monotonic()
                for name, topic in self.wrist_topics.items():
                    context = wrist_context(topic, frame['measured']['timestamp_s'], now, name=name)
                    if context is None:
                        missing.append(name)
                    else:
                        rgb_cameras.update(context)
                if missing:
                    waiting('waiting_for_wrist_camera',
                            'Waiting for fresh RGB near the head observation time: ' + ', '.join(missing))
                    time.sleep(.05)
                    continue
                identifier += 1
                measured = frame['measured']
                if not frame.get('robot_fk'):
                    raise ValueError('Registered camera frame is missing encoder-derived robot FK')
                if client is None:
                    client = OpenRouterPolicy(self.prompt, system, frame['intrinsics'], config, Path(self.output)/'policy')
                from mobile_manipulator.agent.budget import api_cost_limit
                stop_reason = ('agent.max_calls' if client.calls >= config['max_calls'] else
                               'agent.max_cost_usd' if client.cost >= api_cost_limit(config) else None)
                if stop_reason is not None:
                    budget_exhausted = True
                    publish(phase='budget_exhausted', budget_exhausted=True, stop_reason=stop_reason,
                            policy_calls=client.calls, cost_usd=client.cost)
                    break
                observation_id = client.calls+1
                observation = Observation(observation_id, measured['timestamp_s'], frame['rgb'], frame['depth_m'],
                                          {**measured, **frame['robot_fk']}, rgb_cameras=rgb_cameras)
                publish(phase='thinking', message='Model request in progress.', thinking_started_s=time.monotonic(),
                        observation_id=observation_id, model=config['model'], error=None)
                future = request_async(client.act, observation)
                while not future.done() and not stopped():
                    time.sleep(.05)
                if stopped():
                    break
                try:
                    action, record = future.result()
                except Exception as error:
                    failures += 1
                    retryable = getattr(error, 'retryable', False)
                    api_blocked = not retryable or failures >= config.get('max_consecutive_policy_failures', 3)
                    publish(phase='api_blocked' if api_blocked else 'api_error',
                            message='Model requests paused; controller remains in HOLD. Resolve the API error before restarting.'
                            if api_blocked else 'Model request failed; retrying with a fresh observation.',
                            error=str(error), retryable=retryable, policy_calls=client.calls, cost_usd=client.cost)
                    time.sleep(.5)
                    continue
                failures = 0
                if action is None:
                    publish(phase='grounding', message='Point inspected; no motion.' if
                            record.get('grounding', {}).get('status') == 'inspected' else
                            'Point rejected; holding and reobserving.',
                            grounding=record.get('grounding'), decision_summary=record.get('decision_summary'),
                            action=None, action_observation_id=observation_id,
                            policy_calls=client.calls, cost_usd=client.cost)
                    continue
                proposal = dict(id=identifier, operation='target', action=action.to_dict(), observed=measured)
                while not stopped() and not self.request.put(proposal):
                    time.sleep(.001)
                if stopped():
                    break
                publish(phase='submitted', message='Decision submitted to controller.', action=action.to_dict(), observation_id=observation_id,
                        decision_summary=record.get('decision_summary'),
                        grounding=record.get('grounding'),
                        action_observation_id=observation_id, request_id=identifier,
                        policy_calls=client.calls, cost_usd=client.cost)
                accepted_at = None
                terminal_diagnostic = {}
                until = time.monotonic()+float(config['action_horizon_s'])
                while not stopped():
                    state = self.status.get()
                    if state and state.get('phase') in ('fault', 'closed'):
                        break
                    if time.monotonic() >= until:
                        break
                    if state and state.get('request_id') == identifier:
                        if state.get('request_error'):
                            client.reject_stale_action(record)
                            publish(phase='rejected', message='Controller rejected this decision; reobserving.', error=state['request_error'])
                            break
                        if accepted_at is None:
                            accepted_at = time.monotonic()
                            publish(phase='executing', message='Controller accepted the decision.')
                        diagnostic = state.get('diagnostic', {})
                        terminal_diagnostic = diagnostic
                        if state.get('phase') == 'fault' or diagnostic.get('needs_observation') or (diagnostic.get('arrived') and time.monotonic()-accepted_at >= 1.):
                            break
                    time.sleep(.05)
                if stopped():
                    break
                if accepted_at is not None and hasattr(client, 'record_execution'):
                    client.record_execution(record, terminal_diagnostic)
                    publish(phase='action_complete', message='Execution interval ended; reobserving.',
                            execution_feedback=record.get('execution_feedback'))
                identifier += 1
                while not stopped() and not self.request.put(dict(id=identifier, operation='hold')):
                    time.sleep(.001)
                while not stopped():
                    state = self.status.get()
                    if state and state.get('phase') == 'fault':
                        break
                    if state and state.get('request_id') == identifier and state.get('diagnostic', {}).get('ik_status') == 'HOLD' and state.get('diagnostic', {}).get('base_stopped', True):
                        break
                    time.sleep(.05)
        finally:
            if stopped() or budget_exhausted:
                self.signal_stop()
            cause = (last_state or {}).get('fault')
            publish(phase='stopped', message=('Stopped before the first model call. ' if client is None else 'Agent stopped. ')
                    + (cause or 'Control ended or stop requested.'), error=cause or decision_state.get('error'),
                    policy_calls=0 if client is None else client.calls,
                    cost_usd=0 if client is None else client.cost,
                    request_interrupted=future is not None and not future.done())
