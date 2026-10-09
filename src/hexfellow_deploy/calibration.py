"""Camera calibration launchers, independent of the production DA observation path."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]/'calibration'
TOOLS = ROOT/'tools/camera'
ACTIONS = ('run', 'camera', 'fit', 'refine', 'validate', 'overlay', 'review', 'plan', 'preview', 'verify')
OFFLINE_TOOLS = {
    'head': {
        'refine': 'fit_landmark_extrinsics.py',
        'overlay': 'validate_camera_overlay.py',
        'review': 'verify_centered_calibration.py',
        'verify': 'verify_handeye_ground_truth.py',
    },
    'right': {
        'plan': 'wrist_calibration_geometry.py',
        'preview': 'render_wrist_open_gripper.py',
        'verify': 'verify_wrist_ground_truth.py',
    },
    'left': {'plan': 'plan_left_wrist_calibration.py', 'preview': 'preview_left_wrist_calibration.py', 'verify': 'verify_wrist_ground_truth.py'},
}

def cameras():
    return json.loads((ROOT/'config/camera/cameras.json').read_text())['cameras']

def supported_actions(camera):
    actions = ['camera', 'fit', *OFFLINE_TOOLS[camera]]
    actions.insert(0, 'run')
    if camera == 'head':
        actions.append('validate')
    return actions

def build_command(camera, action, arguments):
    selected = cameras()[camera]
    python = sys.executable
    environment = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[variable] = '1'
    environment.setdefault('MUJOCO_GL', 'egl')
    if action == 'run':
        command = [python, str(TOOLS/'calibrate_robot.py')]
        if camera in ('left', 'right'):
            command += ['--wrist-plan', str(ROOT/selected['pose_plan'])]
        if camera == 'left':
            command += ['--port', '8774']
    elif action == 'fit':
        if arguments == ['--help']:
            data = None
        elif not arguments or arguments[0].startswith('-'):
            raise ValueError('fit requires a dataset JSON followed by --output FILE')
        else:
            data = json.loads(Path(arguments[0]).read_text())
        if camera != 'head' and data is not None:
            parent = data.get('parent_frame', data.get('wrist_frame'))
            serial = str(data.get('camera', {}).get('serial', data.get('camera_serial', '')))
            if parent != selected['parent_frame'] or serial != selected['serial']:
                raise ValueError(f"Dataset must identify {selected['parent_frame']} and serial {selected['serial']}; refusing to relabel another camera's fit")
        command = [python, str(TOOLS/('solve_handeye.py' if camera == 'head' else 'solve_wrist_calibration.py'))]
    elif action == 'camera':
        # A camera-only worker: never imports a robot endpoint or sends motion.
        sys.path.insert(0, str(TOOLS))
        from calibration_paths import camera_python, sdk_directory
        python = str(camera_python('zed' if camera == 'head' else 'realsense'))
        command = [python, str(TOOLS/selected['worker']), '--mode', 'live']
        if camera != 'head':
            command += ['--serial', selected['serial']]
        else:
            sdk = sdk_directory()
            environment['ZED_DIR'] = str(sdk)
            environment['LD_LIBRARY_PATH'] = f'{sdk}/lib:{sdk}/dependencies/ai_dependencies/lib:'+environment.get('LD_LIBRARY_PATH', '')
    elif action == 'validate' and camera == 'head':
        # This moves the left arm to validate the HEAD camera, not the D405.
        command = [python, str(TOOLS/'calibrate_robot.py'),
                   '--validation-plan', str(ROOT/'config/camera/left_hand_validation.json'),
                   '--port', '8773']
    elif action in OFFLINE_TOOLS[camera]:
        command = [python, str(TOOLS/OFFLINE_TOOLS[camera][action])]
        if camera == 'head' and action == 'review':
            command += ['--snapshot', str(ROOT/'config/camera/both_j6_sim_start.json'),
                        '--attachment', str(ROOT/'config/camera/board_right_centered.json'),
                        '--candidates', str(ROOT/'config/camera/right_centered_candidates.json')]
    else:
        raise ValueError(f'{camera} {action} is not available; supported actions: {", ".join(supported_actions(camera))}')
    return [*command, *arguments], environment

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Examples: right run --mode sim; head refine --points POINTS --refine-all --output FIT; head validate --mode sim; left camera --directory /tmp/left-board; left fit DATASET --output FIT')
    parser.add_argument('camera', choices=['list', 'head', 'right', 'left'])
    parser.add_argument('action', nargs='?', choices=ACTIONS)
    parser.add_argument('--dry-run', action='store_true', help='Print argv only; open no devices')
    tokens = list(sys.argv[1:] if argv is None else argv)
    if len(tokens) >= 2 and tokens[0] in cameras() and tokens[1] in ACTIONS:
        # Forward downstream --help and options without consuming them here.
        args = parser.parse_args(tokens[:2]+(['--dry-run'] if '--dry-run' in tokens[2:] else []))
        remaining = [token for token in tokens[2:] if token != '--dry-run']
    else:
        args, remaining = parser.parse_known_args(tokens)
    if args.camera == 'list':
        if remaining or args.action:
            parser.error('list takes no further arguments')
        print(json.dumps({name: dict(value, supported_actions=supported_actions(name))
                          for name, value in cameras().items()}, indent=2))
        return 0
    if args.action is None:
        parser.error('Choose a calibration action; use list to see each camera\'s supported actions')
    try:
        command, environment = build_command(args.camera, args.action, remaining)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command, env=environment)

if __name__ == '__main__':
    raise SystemExit(main())
