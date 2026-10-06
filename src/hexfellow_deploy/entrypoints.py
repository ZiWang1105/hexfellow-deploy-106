"""Compatible task launchers selecting leaves of the same DA runtime."""
import argparse
from pathlib import Path
import sys


def main(backend, argv=None):
    parser = argparse.ArgumentParser(description='Shared DA/PinkIK/MIT deployment')
    if backend not in ('simulation', 'hardware'):
        parser.error('Backend must be simulation or hardware')
    parser.add_argument('--config', '--deployment-config', default='configs/local/desktop.yaml',
                        help='Shared hardware calibration and actuator profile YAML')
    parser.add_argument('--runtime-config', help='DA runtime YAML, default configs/simulation.yaml')
    parser.add_argument('--evaluate-body', default='entity_apple_body', help='Offline scoring only')
    parser.add_argument('--scene', help='Physical scene YAML; selected from catalog when --task is used')
    parser.add_argument('--task', help='Simulation task ID; only its instruction reaches the policy')
    parser.add_argument('--benchmark', default='configs/benchmarks/mobile_manipulation.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--controller', help='Compatibility assertion: must match the shared deployment controller')
    parser.add_argument('--base-motion', help='Compatibility assertion: must match the shared base configuration')
    parser.add_argument('--respect-policy-limits', action='store_true',
                        help='Opt in to the agent YAML call, cost and episode limits for a bounded benchmark')
    parser.add_argument('--agent', default='configs/agent/rgbd_full.yaml')
    parser.add_argument('--prompt')
    parser.add_argument('--instruction')
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration-s', type=float, default=None)
    parser.add_argument('--settle-s', type=float, default=0.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-onsite-ready', action='store_true')
    parser.add_argument('--research', action='store_true')
    parser.add_argument('--hold-only', action='store_true')
    parser.add_argument('--live-view', '--live-viewer', action='store_true')
    parser.add_argument('--viewer-port', type=int, default=8765)
    parser.add_argument('--diagnose-timing', action='store_true', help='Timing records are always enabled')
    parser.add_argument('--no-video', action='store_true', help='Raw trajectory always recorded; video is an offline export')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args(argv)
    if backend == 'hardware' and not (args.execute and args.confirm_onsite_ready):
        parser.error('Real actuation requires --execute --confirm-onsite-ready; passive legacy tools use --legacy-runtime')
    if backend == 'simulation' and args.execute:
        parser.error('Simulation does not accept --execute')
    if args.seed < 0 or (args.seed and not args.task):
        parser.error('A nonnegative --seed requires --task')
    if args.task and (backend != 'simulation' or args.scene or args.prompt or args.instruction or args.hold_only):
        parser.error('--task selects a simulation scene and instruction; do not combine it with scene/prompt/instruction/hold-only')
    if args.prompt and args.instruction:
        parser.error('Choose --prompt or --instruction')
    prompt = args.prompt or (Path(args.instruction).read_text() if args.instruction else None)
    if prompt is not None and not prompt.strip():
        parser.error('Task instruction must not be empty')
    if not args.hold_only and not prompt and not args.task:
        parser.error('Provide --prompt or --instruction; use --hold-only for an actuator diagnostic')
    if args.hold_only and prompt:
        parser.error('--hold-only cannot execute a policy prompt')
    root = Path(__file__).resolve().parents[4]
    if args.controller or args.base_motion:
        import yaml
        deployment_path = Path(args.config).resolve()
        deployment = yaml.safe_load(deployment_path.read_text())
        for key, requested in (('controller', args.controller), ('base_motion', args.base_motion)):
            if requested is None:
                continue
            configured = deployment.get(key)
            expected = (deployment_path.parent / configured).resolve() if configured else None
            if expected is None or Path(requested).resolve() != expected:
                parser.error(f'--{key.replace("_", "-")} must match {expected}; select it in the shared deployment YAML to keep simulator and hardware identical')
    command = ['--workspace',str(root),'--output',args.output,'--settle-s',str(args.settle_s)]
    if backend == 'simulation':
        command += ['--evaluate-body',args.evaluate_body]
    if args.runtime_config:
        command += ['--config',args.runtime_config]
    if args.duration_s is not None:
        command += ['--duration-s',str(args.duration_s)]
    if prompt:
        command += ['--prompt',prompt]
    if args.task:
        command += ['--task',args.task,'--benchmark',str(Path(args.benchmark).resolve()),'--seed',str(args.seed)]
    if args.respect_policy_limits:
        command += ['--respect-policy-limits']
    if backend == 'hardware':
        command += ['--execute','--confirm-onsite-ready']
    overrides = [f'deployment_config={str(Path(args.config).resolve())}',
                 f'policy.agent_config={str(Path(args.agent).resolve())}',
                 f'interface.backend.scene={str(Path(args.scene or "configs/scenes/apple_on_table.yaml").resolve())}',
                 f'interface.dashboard.port={args.viewer_port}',
                 f'interface.control.startup_carry={str(not args.hold_only).lower()}']
    for value in overrides+args.set:
        command += ['--set',value]
    from .cli import main as deploy
    previous = sys.argv
    try:
        sys.argv = [previous[0],*command]
        deploy()
    finally:
        sys.argv = previous
