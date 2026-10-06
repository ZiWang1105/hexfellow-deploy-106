"""Instantiate the actual DA topic registry and manager from YAML."""
import argparse
import json
import os
from pathlib import Path
import signal
import multiprocessing as mp
import time
import uuid


def build(config_path, overrides=(), *, prompt=None, camera=True, execute_authorized=False, respect_policy_limits=False):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from .manager import InterfaceManager
    from deploy_anything.utils.config_utils import build_topic_registry, register_topic_resolver
    config = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(list(overrides)))
    if config.interface.backend.backend == 'hardware' and not (config.interface.backend.execute and execute_authorized):
        raise ValueError('Hardware requires --execute --confirm-onsite-ready')
    if config.interface.backend.backend == 'hardware':
        config.interface.camera = dict(_target_='hexfellow_deploy.camera.NativeCamera',
            name='hexfellow_camera', feedback='${topic:feedback}', images='${topic:images}',
            workspace='${workspace}', deployment_config='${deployment_config}', output='${output}', fps=15)
    if not camera:
        del config.interface.camera
    wrist = config.get('wrist_camera', {})
    use_wrist = camera and config.interface.backend.backend == 'hardware' and wrist.get('enabled', False)
    if use_wrist:
        config.topics.wrist_images = dict(_target_='hexfellow_deploy.topics.RGBTopic', prefix='${topic_prefix}',
                                         width=wrist.width, height=wrist.height)
        config.interface.wrist_camera = dict(_target_='hexfellow_deploy.wrist_camera.NativeWristCamera',
            name='hexfellow_wrist_camera', images='${topic:wrist_images}', output='${output}',
            serial=wrist.get('serial'), fps=wrist.fps,
            feedback='${topic:feedback}', workspace='${workspace}',
            deployment_config='${deployment_config}', calibration=wrist.get('calibration'))
        config.interface.dashboard.wrist_images = '${topic:wrist_images}'
    if prompt:
        if not camera:
            raise ValueError('Policy requires registered RGB-D camera')
        deployment_path = Path(config.deployment_config)
        if not deployment_path.is_absolute():
            deployment_path = Path(config.workspace)/deployment_path
        deployment_config = OmegaConf.load(deployment_path)
        base_config = str((deployment_path.parent/deployment_config.base_motion).resolve())
        config.interface.agent = dict(_target_='hexfellow_deploy.policy.AgentComponent',
            name='hexfellow_agent', request='${topic:request}', status='${topic:status}',
            decision='${topic:decision}', images='${topic:images}', workspace='${workspace}',
            output='${output}', prompt=prompt, agent_config='${policy.agent_config}',
            model='${policy.model}', reasoning_effort='${policy.reasoning_effort}',
            respect_limits=respect_policy_limits, base_config=base_config)
        if use_wrist:
            config.interface.agent.wrist_images = '${topic:wrist_images}'
    topics = build_topic_registry(config.topics)
    register_topic_resolver(topics)
    groups = {key: {name: instantiate(spec) for name, spec in config[key].items()}
              for key in ('interface', 'preprocess', 'input', 'inference', 'output_components')}
    manager = InterfaceManager(topics=topics, interface=groups['interface'],
        preprocess=groups['preprocess'], input=groups['input'], inference=groups['inference'],
        output=groups['output_components'], loop_time=config.timing.control_period_s,
        affinities=OmegaConf.to_container(config.affinities), shutdown_timeout_s=8.,
        python_scheduling=OmegaConf.to_container(OmegaConf.create(config.get('python_scheduling', {}))))
    return config, topics, manager


def read_report(path, errors, *, required=False):
    """Keep final diagnostics usable when a child died before writing its report."""
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except FileNotFoundError:
        if required:
            errors.append(f'Missing final report: {path.name}; component shutdown is unconfirmed')
    except (OSError, ValueError) as error:
        errors.append(f'Unreadable report {path.name}: {type(error).__name__}: {error}')
    return {}


def read_final_reports(output, selected_backend):
    errors = []
    # Lifecycle error may be the only surviving evidence after an abrupt exit.
    health = read_report(output/'component_health.json', errors)
    control = read_report(output/'control_final.json', errors, required=True)
    backend = read_report(output/'backend_final.json', errors, required=True)
    backend.setdefault('backend', selected_backend)
    termination = read_report(output/'termination.json', errors) or {'reason':'operator_stop'}
    decision = {}
    try:
        decision_path = output/'decisions.jsonl'
        if decision_path.exists():
            for line in reversed(decision_path.read_text().splitlines()):
                try:
                    decision = json.loads(line)
                    if not isinstance(decision, dict):
                        raise ValueError('expected a JSON object')
                    break
                except ValueError as error:
                    errors.append(f'Unreadable decision record: {type(error).__name__}: {error}')
                    decision = {}
    except OSError as error:
        errors.append(f'Unreadable decisions.jsonl: {type(error).__name__}: {error}')
    return control, backend, health, termination, decision, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2]/'configs/simulation.yaml'))
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration-s', type=float, default=None, help='Optional diagnostic duration; unset means unlimited')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-onsite-ready', action='store_true')
    parser.add_argument('--settle-s', type=float, default=0., help='Optional final HOLD interval after diagnostic duration')
    parser.add_argument('--evaluate-body', help='Offline scoring body name; never supplied to policy/control')
    parser.add_argument('--action', help='JSON paired TCP action for a controller diagnostic, not an agent run')
    parser.add_argument('--prompt', help='Run the asynchronous Astra policy with this instruction')
    parser.add_argument('--task', help='Simulation catalog task; scene and scoring remain outside policy/control')
    parser.add_argument('--benchmark', default='configs/benchmarks/mobile_manipulation.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--respect-policy-limits', action='store_true',
                        help='Opt in to agent YAML request, cost and wall-clock episode budgets')
    parser.add_argument('--no-camera', action='store_true', help='Controller-only simulation diagnostic')
    parser.add_argument('--set', action='append', default=[], help='Hydra/OmegaConf dotlist override')
    args = parser.parse_args()
    if args.action and args.prompt:
        parser.error('Choose an agent prompt or a controller fixture, not both')
    if args.task and (args.action or args.prompt or args.execute):
        parser.error('--task selects a simulation instruction; cannot combine with action, prompt or hardware execution')
    if args.seed < 0 or (args.seed and not args.task):
        parser.error('A nonnegative --seed requires --task')
    if not 0 <= args.settle_s < float('inf') or (args.settle_s and args.duration_s is None and not args.respect_policy_limits):
        parser.error('settle-s must be finite, nonnegative and used with duration-s or respect-policy-limits')
    if args.duration_s is not None and (not 0 < args.duration_s < float('inf')):
        parser.error('duration must be finite and positive')
    if args.execute != args.confirm_onsite_ready:
        parser.error('Hardware execution requires both --execute and --confirm-onsite-ready')
    if args.execute:
        args.set += ['interface.backend.backend=hardware', 'interface.backend.execute=true']
    task = None
    if args.task:
        from .workspace import activate
        activate(args.workspace)
        from mobile_manipulator.eval.benchmark import load_catalog
        tasks, _ = load_catalog(args.benchmark)
        if args.task not in tasks:
            parser.error(f'Unknown task: {args.task}')
        task = tasks[args.task]
        args.prompt = task.instruction
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if task:
        from mobile_manipulator.eval.benchmark import materialize_scene
        scene = materialize_scene(task, args.seed, output/'scene.yaml')
        args.set += [f'interface.backend.scene={scene}']
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['HEXFELLOW_WORKSPACE'] = str(Path(args.workspace).expanduser().resolve())
    os.environ['HEXFELLOW_OUTPUT'] = str(output)
    os.environ['HEXFELLOW_TOPIC_PREFIX'] = 'hf_' + uuid.uuid4().hex[:12] + '_'
    from .provenance import collect
    (output/'provenance.json').write_text(json.dumps(collect(os.environ['HEXFELLOW_WORKSPACE']), indent=2)+'\n')
    from omegaconf import OmegaConf
    config, topics, manager = build(Path(args.config).resolve(), args.set,
                                    prompt=args.prompt, camera=not args.no_camera, execute_authorized=args.execute,
                                    respect_policy_limits=args.respect_policy_limits)
    if task and config.interface.backend.backend != 'simulation':
        raise ValueError('Catalog tasks require the simulation backend')
    duration_source = 'duration_s' if args.duration_s is not None else None
    if args.respect_policy_limits and args.duration_s is None:
        import yaml
        agent_path = Path(config.policy.agent_config)
        if not agent_path.is_absolute():
            agent_path = Path(args.workspace)/agent_path
        policy_config = yaml.safe_load(agent_path.read_text())
        duration = policy_config.get('max_episode_s')
        if duration is not None and float(duration) < float('inf'):
            if not 0 < float(duration):
                parser.error('Agent max_episode_s must be positive')
            args.duration_s = float(duration)
            duration_source = 'agent.max_episode_s_wall_clock'
    # Keep topic interpolations symbolic; resolving them would serialize comm objects.
    OmegaConf.save(config, output/'config.yaml', resolve=False)
    (output/'invocation.json').write_text(json.dumps(dict(
        workspace=os.environ['HEXFELLOW_WORKSPACE'], config=str(Path(args.config).resolve()),
        overrides=args.set, prompt=args.prompt, action_file=args.action,
        duration_s=args.duration_s, duration_source=duration_source, settle_s=args.settle_s,
        task=args.task, catalog=args.benchmark if task else None, seed=args.seed if task else None,
        respect_policy_limits=args.respect_policy_limits, camera=not args.no_camera), indent=2)+'\n')
    (output/'topics.json').write_text(json.dumps({n:t.name for n,t in topics.items()}, indent=2))
    action = None if args.action is None else json.loads(Path(args.action).read_text())
    finished = mp.Event()
    parent_pid = os.getpid()
    def supervise():
        from .experiment import FinalHold
        started = None
        submitted = False
        final_hold = FinalHold(args.settle_s)
        health_hold = FinalHold(0.)
        health_fault = None
        budget_reason = None
        init_started = time.monotonic()
        while not finished.wait(.05):
            status = topics['status'].get(timeout=.001)
            health = topics['health'].get(timeout=.001)
            if health and health.get('fault'):
                health_fault = health_fault or health['fault']
            if health_fault:
                agent = manager._interface.get('agent')
                if agent is not None:
                    agent.signal_stop()
                health_result = health_hold.update(time.monotonic(), status)
                if health_result == 'request':
                    topics['request'].put(dict(id=-1, operation='final_hold'), timeout=.01)
                if (args.duration_s is not None or args.respect_policy_limits) and health_result in ('completed_budget', 'final_hold_failed'):
                    (output/'termination.json').write_text(json.dumps(dict(reason='component_fault', fault=health_fault)))
                    os.kill(parent_pid, signal.SIGINT)
                    return
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
            decision = topics['decision'].get(timeout=.001) if args.respect_policy_limits else None
            if budget_reason is None and decision and decision.get('budget_exhausted'):
                budget_reason = decision.get('stop_reason', 'policy_budget')
            if budget_reason is None and started is not None and args.duration_s is not None and time.monotonic()-started >= args.duration_s:
                budget_reason = duration_source
            ended = budget_reason is not None
            reason = 'completed_budget' if ended else None
            if ended and (args.settle_s > 0 or args.prompt):
                result = final_hold.update(time.monotonic(), status)
                if result == 'request':
                    topics['request'].put(dict(id=-1,operation='final_hold'),timeout=.01)
                ended = result in ('completed_budget', 'final_hold_failed')
                reason = result if ended else None
            if ended or ((args.duration_s is not None or args.respect_policy_limits) and status and status.get('phase') == 'fault'):
                (output/'termination.json').write_text(json.dumps(dict(reason=reason if ended else 'control_fault',
                    budget_reason=budget_reason,
                    elapsed_wall_s=None if started is None else time.monotonic()-started,settle_s=args.settle_s,
                    final_hold_requested_s=final_hold.requested_at, stationary_since_s=final_hold.stationary_since)))
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
    final, backend, health, termination, decision, report_errors = read_final_reports(
        output, config.interface.backend.backend)
    print(json.dumps({k: final.get(k) for k in ('phase','ticks','generated','accepted','fault','evidence_kind')}))
    fault = (health.get('fault') or final.get('fault') or backend.get('fault') or backend.get('shutdown_error')
             or (report_errors[0] if report_errors else None))
    if termination['reason'] == 'final_hold_failed':
        fault = fault or 'Final HOLD did not remain stationary for the requested interval'
    report = dict(status='error' if fault else termination['reason'],fault=fault,backend=backend['backend'],
                  policy_calls=decision.get('policy_calls',0),cost_usd=decision.get('cost_usd',0),
                  request_interrupted=decision.get('request_interrupted',False),
                  ticks=final.get('ticks',0),termination=termination,native_qualified=False,
                  report_errors=report_errors,
                  shutdown_reports_present=bool(final and backend.get('phase') is not None))
    if task:
        report.update(task=task.name, seed=args.seed, catalog=str(Path(args.benchmark).resolve()))
    # Seal the episode before importing or reading the privileged offline referee.
    (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    # Only closed simulator runs may read scoring records; never feed them back.
    if task and backend['backend'] == 'simulation':
        from mobile_manipulator.eval.scoring import evaluate_run
        try:
            evaluation = evaluate_run(output, task, model_filename='plant.xml')
        except Exception as error:
            fault = fault or f'Offline evaluation failed: {type(error).__name__}: {error}'
            report.update(status='error', fault=fault, evaluation_error=str(error))
            (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
            raise SystemExit(1)
        final_hold = bool(final.get('final_hold') and final.get('diagnostic',{}).get('ik_status') == 'HOLD'
                          and final.get('diagnostic',{}).get('base_stopped', False))
        evaluation.update(final_hold_verified=final_hold, accepted=bool(evaluation['accepted'] and final_hold))
        (output/'benchmark_evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        report['evaluation'] = evaluation
    elif backend['backend'] == 'simulation' and (output/'physics_evaluation.json').exists():
        from .physics_audit import pickup_evaluation
        audit = json.loads((output/'physics_evaluation.json').read_text())
        final_hold = bool(final.get('final_hold') and final.get('diagnostic',{}).get('ik_status') == 'HOLD'
                          and final.get('diagnostic',{}).get('base_stopped', False))
        report['evaluation'] = pickup_evaluation(audit, args.evaluate_body, final_hold=final_hold,
            completed_budget=bool(report['status'] == 'completed_budget' and args.prompt))
    elif backend['backend'] == 'simulation' and (output/'final_evaluation.json').exists():
        evaluation = json.loads((output/'final_evaluation.json').read_text())
        final_hold = final.get('final_hold') and final.get('diagnostic',{}).get('ik_status') == 'HOLD'
        retained = any(item['name'] == args.evaluate_body and item['retained_s'] >= 10 for item in evaluation['objects']) if evaluation else False
        contacts = evaluation['robot_static_contact_frames'] if evaluation else None
        report['evaluation'] = dict(retained_pickup=retained,robot_static_contact_frames=contacts,
            sampling='recorded_frames_15_hz',
            final_hold_verified=bool(final_hold),evaluated_body=args.evaluate_body,accepted=bool(retained and contacts == 0 and final_hold
                and report['status']=='completed_budget' and args.prompt))
    (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if fault:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
