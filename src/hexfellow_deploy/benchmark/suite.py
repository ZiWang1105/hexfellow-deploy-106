"""Serial benchmark matrix. Failures remain in the manifest and comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from hexfellow_deploy.robot.upstream import PROJECT_ROOT


def cases():
    profile = str(PROJECT_ROOT/'robot_resources/config/control/motion_profiles/x7-gp80-fast-trajectory-sim-v1.yaml')
    candidate = ['--profile', profile, '--derivatives']
    common = ['--posture', 'extended', '--trajectory', 'swing', '--speed', '1.3']
    # Single-factor ablations before the candidate comparison. All use the
    # same X7 geometry and unchanged MIT gains and per-joint physical limits.
    variants = {
        'baseline_v1': [], 'baseline_v2': ['--version', 'v2'], 'baseline_v3': ['--version', 'v3'],
        'envelope_only': ['--fast'], 'derivatives_50hz': ['--fast', '--derivatives'],
        'derivatives_100hz': ['--fast', '--derivatives', '--hz', '100'],
        'inertia_100hz': ['--fast', '--derivatives', '--hz', '100', '--dynamics'],
        'candidate': candidate,
        'nonlinear_gain1': ['--profile', profile, '--solver', 'nonlinear'],
        'nonlinear_gain10': ['--profile', profile, '--solver', 'nonlinear', '--joint-correction-gain', '10'],
    }
    for name, flags in variants.items():
        yield name, common+flags, 'comparison'
    for speed in (.2, .4, .8, 1., 1.2, 1.4, 1.5):
        yield f'speed_{speed:g}', ['--posture', 'extended', '--trajectory', 'swing', '--speed', str(speed)]+candidate, 'speed_sweep'
    for posture in ('nominal', 'heldout', 'extended_offset'):
        for dual in (False, True):
            yield f'heldout_swing_{posture}_{"dual" if dual else "right"}', [
                '--posture', posture, '--trajectory', 'swing', '--speed', '1.3',
                *(['--dual'] if dual else [])]+candidate, 'heldout'
    yield 'heldout_dual_extended', common+['--dual']+candidate, 'heldout'
    for distance in (1., 1.5):
        for dual in (False, True):
            yield f'joint_curve_{distance:g}_{"dual" if dual else "right"}', [
                '--posture', 'extended', '--trajectory', 'joint_curve', '--speed', '1.3',
                '--distance', str(distance), *(['--dual'] if dual else [])]+candidate, 'additional_paths'
    for speed in (.4, .8):
        yield f'baseline_speed_{speed:g}', ['--posture', 'extended', '--trajectory', 'swing',
                                           '--speed', str(speed)], 'baseline_within_envelope'
    for kind in ('vertical', 'lateral', 'forward', 'circle', 'figure8', 'rounded_rectangle', 'rotate'):
        for speed in (.4, 1.2):
            yield f'heldout_{kind}_{speed:g}', ['--posture', 'extended', '--trajectory', kind,
                '--speed', str(speed)]+candidate, 'heldout_stress'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    manifest = []
    for name, flags, group in cases():
        command = [sys.executable, '-m', 'hexfellow_deploy.benchmark.run', '--output', str(root/name), *flags]
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
        (root/f'{name}.log').write_text(result.stdout+result.stderr)
        summary = root/name/'summary.json'
        value = json.loads(summary.read_text()) if summary.exists() else dict(completed=False,
            failure=result.stderr[-3000:], tracking_accepted=False, fast_tracking_accepted=False)
        manifest.append(dict(name=name, group=group, command=command, exit_code=result.returncode,
                             summary=str(summary.relative_to(root)), result=value))
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        metric = value.get('metrics') or {}
        position = metric.get('measured', {}).get('position_mm') or {}
        print(f'{name}: completed={value["completed"]} rms_mm={position.get("rms")} '
              f'accepted={value.get("tracking_accepted", False)}', flush=True)


if __name__ == '__main__':
    main()
