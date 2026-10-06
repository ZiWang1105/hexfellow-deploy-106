"""Instantiate the actual DA topic registry and manager from YAML."""
import argparse
import json
import os
from pathlib import Path
import signal
import multiprocessing as mp
import time
import uuid


def build(config_path, overrides=(), *, prompt=None, camera=True):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from .manager import InterfaceManager
    from deploy_anything.utils.config_utils import build_topic_registry, register_topic_resolver
    config = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(list(overrides)))
    if config.interface.control.backend != 'simulation':
        raise ValueError('Native MIT adapter is not implemented; no devices or shared memory opened')
    if not camera:
        del config.interface.camera
    if prompt:
        if not camera:
            raise ValueError('Policy requires registered RGB-D camera')
        config.interface.agent = dict(_target_='hexfellow_deploy.policy.AgentComponent',
            name='hexfellow_agent', request='${topic:request}', status='${topic:status}',
            decision='${topic:decision}', images='${topic:images}', workspace='${workspace}',
            output='${output}', prompt=prompt, agent_config='${policy.agent_config}',
            model='${policy.model}', reasoning_effort='${policy.reasoning_effort}')
    topics = build_topic_registry(config.topics)
    register_topic_resolver(topics)
    groups = {key: {name: instantiate(spec) for name, spec in config[key].items()}
              for key in ('interface', 'preprocess', 'input', 'inference', 'output_components')}
    manager = InterfaceManager(topics=topics, interface=groups['interface'],
        preprocess=groups['preprocess'], input=groups['input'], inference=groups['inference'],
        output=groups['output_components'], loop_time=config.timing.control_period_s,
        affinities=OmegaConf.to_container(config.affinities), shutdown_timeout_s=8.)
    return config, topics, manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2]/'configs/simulation.yaml'))
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration-s', type=float, default=None, help='Optional diagnostic duration; unset means unlimited')
    parser.add_argument('--action', help='JSON paired TCP action for a controller diagnostic, not an agent run')
    parser.add_argument('--prompt', help='Run the asynchronous Astra policy with this instruction')
    parser.add_argument('--no-camera', action='store_true', help='Controller-only simulation diagnostic')
    parser.add_argument('--set', action='append', default=[], help='Hydra/OmegaConf dotlist override')
    args = parser.parse_args()
    if args.action and args.prompt:
        parser.error('Choose an agent prompt or a controller fixture, not both')
    if args.duration_s is not None and (not 0 < args.duration_s < float('inf')):
        parser.error('duration must be finite and positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['HEXFELLOW_WORKSPACE'] = str(Path(args.workspace).expanduser().resolve())
    os.environ['HEXFELLOW_OUTPUT'] = str(output)
    os.environ['HEXFELLOW_TOPIC_PREFIX'] = 'hf_' + uuid.uuid4().hex[:12] + '_'
    from .provenance import collect
    (output/'provenance.json').write_text(json.dumps(collect(os.environ['HEXFELLOW_WORKSPACE']), indent=2)+'\n')
    from omegaconf import OmegaConf
    config, topics, manager = build(Path(args.config).resolve(), args.set,
                                    prompt=args.prompt, camera=not args.no_camera)
    # Keep topic interpolations symbolic; resolving them would serialize comm objects.
    OmegaConf.save(config, output/'config.yaml', resolve=False)
    (output/'invocation.json').write_text(json.dumps(dict(
        workspace=os.environ['HEXFELLOW_WORKSPACE'], config=str(Path(args.config).resolve()),
        overrides=args.set, prompt=args.prompt, action_file=args.action,
        duration_s=args.duration_s, camera=not args.no_camera), indent=2)+'\n')
    (output/'topics.json').write_text(json.dumps({n:t.name for n,t in topics.items()}, indent=2))
    action = None if args.action is None else json.loads(Path(args.action).read_text())
    finished = mp.Event()
    parent_pid = os.getpid()
    def supervise():
        started = None
        submitted = False
        init_started = time.monotonic()
        while not finished.wait(.05):
            status = topics['status'].get(timeout=.001)
            if status and status.get('phase') in ('running', 'waiting_for_acceptance'):
                if started is None:
                    started = time.monotonic()
                if action is not None and not submitted:
                    measured = topics['measured'].get(timeout=.001)
                    if measured:
                        submitted = topics['request'].put(dict(id=1, operation='target',
                            action=action, observed=measured), timeout=.001)
                        topics['decision'].put(dict(kind='controller_fixture', action=action,
                                                    policy_calls=0, request_id=1))
            ended = started is not None and args.duration_s is not None and time.monotonic()-started >= args.duration_s
            if ended or (args.duration_s is not None and status and status.get('phase') == 'fault'):
                os.kill(parent_pid, signal.SIGINT)
                return
            if started is None and time.monotonic()-init_started > 60:
                os.kill(parent_pid, signal.SIGINT)
                return
    # DA forks its component processes. Do not start parent threads or hold a
    # shared-memory semaphore across that fork boundary.
    watcher = mp.Process(target=supervise, name='run-supervisor', daemon=True)
    watcher.start()
    try:
        manager.run_all()
    finally:
        finished.set()
        watcher.join(2)
        if watcher.is_alive():
            watcher.terminate()
            watcher.join(2)
    final_path = output/'control_final.json'
    if not final_path.exists():
        raise RuntimeError('Control component did not produce a final report')
    final = json.loads(final_path.read_text())
    print(json.dumps({k: final.get(k) for k in ('phase','ticks','generated','accepted','fault','evidence_kind')}))
    if final.get('fault'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
