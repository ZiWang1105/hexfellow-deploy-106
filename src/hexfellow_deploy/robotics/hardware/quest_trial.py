"""Explicit Quest-only research trial, independent of the replay demonstration ledger."""

import argparse
import fcntl
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from ..paths import PROJECT_ROOT
from . import session
from .demo import ACCEPTED, failure_detail, machine_pass
from .research_profiles import SUPPORTED_PROFILES
from .research_profiles import PROFILE as PROFILE
from .verification import device_binding, digest, software_binding
from .x7 import HardwareSafetyError

COMMANDS = frozenset({'default', 'init', 'quest', 'stop', 'status', 'quit', 'help'})


def audit_passed(report):
    gateway = report.get('gateway', {})
    return (report.get('passed') is True
            and report.get('mode') == 'audit'
            and report.get('motion_readiness_passed') is True
            and report.get('gateway_observed_feedback_clean') is True
            and report.get('motion_commands_sent') == 0
            and report.get('cycles') == 1500
            and report.get('failed_cycles') == 0
            and report.get('skipped_slots') == 0
            and report.get('trace_complete') is True
            and not report.get('trace_dropped_rows')
            and gateway.get('kind') == 'closed'
            and not gateway.get('fault') and not gateway.get('stop_failures')
            and bool(report.get('initial_health')))


def run(args):
    from .verify import save_json, timing_probe

    if args.action not in {'status', 'run'} or args.step or args.feedback_only:
        raise ValueError('quest-trial supports status/run only; no replay or commissioning steps')
    if not 1 <= args.port <= 65535:
        raise ValueError('Quest port must be in [1, 65535]')
    parser = argparse.ArgumentParser()
    session.add_arguments(parser)
    values = parser.parse_args([
        '--mode', 'live', '--robot-config', str(args.robot_config),
        '--bind-ip', args.bind_ip, '--port', str(args.port),
        '--quest-pose-rate', args.quest_pose_rate,
        '--quest-link-policy', args.quest_link_policy,
        '--quest-resume', 'require-release', '--confirm-onsite-ready',
    ])
    values.research_release = getattr(args, "research_release", None)
    values.quest_stream_config = getattr(args, 'quest_stream_config', None)
    values.demo_authorized = True
    values.quest_trial = True
    mapping, profile, _, _ = session.load_configuration(values)
    if (profile.profile_id not in SUPPORTED_PROFILES or mapping.feedback_policy != 'strict'
            or mapping.latency_policy != 'hold-resume' or profile.motion_envelope != 'none'):
        raise HardwareSafetyError(
            'quest-trial requires the GP80 research runtime with strict feedback and hold-resume')
    from .research_profiles import describe, require_offline_validation
    print(describe(profile), flush=True)
    if args.action == 'run':
        require_offline_validation(profile, mapping=mapping)
    values.allow_uncommissioned_profile_sha256 = profile.sha256
    mapping.authorize(profile, 'whole-body', profile.sha256)
    binding = software_binding(mapping)
    from ..quest.link_policy import link_policy
    input_policy = link_policy(args.quest_link_policy, getattr(args, 'quest_stream_config', None)).report()
    root = args.run_dir.resolve()
    ledger_path = root / 'quest-trial.json'

    def read_ledger():
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
            suite='quest-trial', binding=binding, attempts=[], hardware_qualified=False,
            replay_skipped_by_operator=True, experimental_checks_accepted=list(ACCEPTED),
            quest_resume='require-release', historical_demo_results_inherited=False,
            quest_link_policy=input_policy, quest_pose_rate=args.quest_pose_rate)
        if ledger.get('suite') != 'quest-trial' or ledger.get('binding') != binding:
            raise HardwareSafetyError('Quest trial software/configuration binding changed; use a new run-dir to preserve history')
        if ledger.get('quest_link_policy') != input_policy or ledger.get('quest_pose_rate') != args.quest_pose_rate:
            raise HardwareSafetyError('Quest input policy/rate changed; use a new run-dir to preserve comparison records')
        return ledger

    ledger = read_ledger()
    if args.action == 'status':
        print(f"Independent Quest trial; {profile.profile_id}; strict; release and squeeze again after delay recovery.")
        print(f"Quest input: {args.quest_link_policy}; pose rate: {args.quest_pose_rate}; robot feedback remains strict.")
        print(f"Recorded {len(ledger['attempts'])} attempts; replay is unqualified; full hardware qualification is not asserted.")
        if ledger['attempts']:
            print(json.dumps(ledger['attempts'][-1], ensure_ascii=False, indent=2))
        return ledger

    # Use the same exclusive owner as the other onsite entry points.
    (PROJECT_ROOT / 'runs').mkdir(exist_ok=True)
    with (PROJECT_ROOT / 'runs/.hexfellow-onsite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise HardwareSafetyError('Another verification/control process is running') from error
        ledger = read_ledger()
        answer = input('Confirm empty grippers, mechanical zero, clearance/cables, charger disconnected, and accessible E-stop; enter y for the read-only check:')
        if answer.strip().lower() != 'y':
            return
        from ..quest.bridge import resolve_bind_ip

        values.bind_ip = resolve_bind_ip(args.bind_ip)
        root.mkdir(parents=True, exist_ok=True)
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        audit_dir = root / f'audit-{tag}'
        print('30-second read-only feedback, IK, and local SDK check; no robot control acquisition.', flush=True)
        audit = timing_probe(mapping, profile, audit_dir, 30, sdk_sink=True)
        audit_path = audit_dir / 'summary.json'
        entry = dict(at=tag, audit=str(audit_path.relative_to(root)), audit_sha256=digest(audit_path),
                     audit_passed=audit_passed(audit), hardware_qualified=False,
                     quest_link_policy=input_policy, quest_pose_rate=args.quest_pose_rate)
        ledger['attempts'].append(entry)
        save_json(ledger_path, ledger)
        if not entry['audit_passed']:
            reason = audit.get('error', 'readiness/feedback/trace incomplete')
            raise HardwareSafetyError(f"Read-only check failed：{reason}；{audit_path}")
        if (software_binding(mapping) != binding or link_policy(args.quest_link_policy,
                getattr(args, 'quest_stream_config', None)).report() != input_policy):
            raise HardwareSafetyError('software changed during preflight; use a new run-dir')
        values.demo_device_binding = device_binding(SimpleNamespace(health=audit['initial_health']))
        values.trial_binding = binding
        values.save_trace = root / f'quest-{tag}'
        print('Read-only check passed. Enter default, init, quest in order, waiting for each move; stop ends following, quit releases control.', flush=True)
        result = session.run(values)
        report_path = values.save_trace / 'summary.json'
        okay = (machine_pass('quest', result) and result.get('trace_complete') is True
                and not result.get('quest_status_error'))
        # Require positive per-device stop evidence, not merely an empty error list.
        observations = result.get('gateway', {}).get('stop_observations', {})
        okay = okay and all(observations.get(n, {}).get('continuous_stop_confirmed') is True
                            for n in ('left', 'right', 'theta', 'base'))
        entry.update(report=str(report_path.relative_to(root)), sha256=digest(report_path),
                     machine_passed=okay, operator_normal=False)
        save_json(ledger_path, ledger)
        print('Report:', report_path, flush=True)
        if not okay:
            raise HardwareSafetyError(f'Trial failed:{failure_detail(result)}')
        answer = input('Program closed; confirm normal motion and stopping, and a stationary robot; enter y:')
        entry['operator_normal'] = answer.strip().lower() == 'y'
        save_json(ledger_path, ledger)
        if not entry['operator_normal']:
            raise HardwareSafetyError('Operator confirmation of motion and stopping is missing; report retained')
        print('Quest trial passed; replay remains unqualified; formal operating qualification is unchanged.')
        return result
