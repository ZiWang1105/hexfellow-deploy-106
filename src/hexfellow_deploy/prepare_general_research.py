"""Prepare an explicit local research copy for the frozen general IK candidate.

This writes configuration and source identities only. It never opens devices,
changes calibration/gains/physical limits, or marks hardware as qualified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from .robot.upstream import PROJECT_ROOT
from .robotics.control.motion_profile import load_motion_profile
from .robotics.hardware.research_profiles import GENERAL_IK_PROFILE, HIGHSPEED_PROFILE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source_deployment, output):
    source = Path(source_deployment).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    config = copy.deepcopy(raw)
    resolve = lambda name: (source.parent/Path(name)).resolve()
    original_mapping = resolve(config['robot']['mapping'])
    original_release = resolve(config['robot']['research_release'])
    mapping = json.loads(original_mapping.read_text())
    release = json.loads(original_release.read_text())
    baseline = load_motion_profile(HIGHSPEED_PROFILE)
    candidate = load_motion_profile(GENERAL_IK_PROFILE)
    if (release.get('authorization') != 'operator_requested_research_trial'
            or release.get('hardware_qualified') is not False
            or release.get('runtime_sha256') != sha(original_mapping)
            or mapping.get('commissioning', {}).get('sequence') != 'hardware-session-v1'
            or mapping['commissioning'].get('motion_profile') != baseline.profile_id
            or mapping['commissioning'].get('profile_sha256') != baseline.sha256
            or release.get('profile_sha256') != baseline.sha256):
        raise ValueError('Source must be an existing unqualified research configuration for fixed-base-v1')
    profile_dir = PROJECT_ROOT/'robot_resources/config/control/motion_profiles'
    freeze_path = PROJECT_ROOT/'docs/validation/general_ik_20261016/frozen_selection.json'
    freeze = json.loads(freeze_path.read_text())
    if sha(profile_dir/(candidate.profile_id+'.yaml')) != freeze['sha256']:
        raise ValueError('Candidate differs from the frozen benchmark profile')
    old_doc = yaml.safe_load((profile_dir/(baseline.profile_id+'.yaml')).read_text())
    new_doc = yaml.safe_load((profile_dir/(candidate.profile_id+'.yaml')).read_text())
    # Only benchmarked control changes may differ. In particular this rejects
    # modified MIT gains/ceilings, joint limits, base/lift limits and robot geometry.
    allowed = {'profile_id', 'provenance', 'control_dt_s', 'controller_tuning',
               'tracking_objective', 'target_limits', 'mit_dynamic_feedforward'}
    differences = {k for k in old_doc.keys() | new_doc.keys() if old_doc.get(k) != new_doc.get(k)}
    if differences-allowed or candidate.control_dt_s != .01 or candidate.hardware_commissioned:
        raise ValueError('Candidate changes physical settings or is not the uncommissioned 100 Hz candidate')
    output.mkdir(parents=True, exist_ok=False)
    # Keep the original evidence alongside the new identities for auditability.
    (output/'source_deployment.yaml').write_bytes(source.read_bytes())
    (output/'source_mapping.json').write_bytes(original_mapping.read_bytes())
    (output/'source_release.json').write_bytes(original_release.read_bytes())
    for key in ('controller','base_motion','cameras'):
        config[key] = str(resolve(config[key]))
    for key in ('urdf','assembly','verification_report'):
        if config['robot'].get(key):
            config['robot'][key] = str(resolve(config['robot'][key]))
    config['camera_calibration'] = {k:str(resolve(v)) if v else None
                                   for k,v in config['camera_calibration'].items()}
    # Preserve the measured assembly and mapping. Use only the new controller.
    controller_path = output/'controller.yaml'
    controller_path.write_bytes((PROJECT_ROOT/'configs/controller/x7_general_trajectory_sim.yaml').read_bytes())
    config['controller'] = str(controller_path)
    mapping['commissioning']['motion_profile'] = candidate.profile_id
    mapping['commissioning']['profile_sha256'] = candidate.sha256
    from .robotics.paths import config_reference
    for name in ('default_pose',):
        if name in mapping:
            mapping[name]['path'] = str(config_reference(original_mapping, mapping[name]['path']))
    mapping_path = output/'mapping.json'
    mapping_path.write_text(json.dumps(mapping,indent=2)+'\n')
    config['robot']['mapping'] = str(mapping_path)
    config['robot']['research_release'] = str(output/'research_release.json')
    config_path = output/'desktop.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    release['profile_id'], release['profile_sha256'] = candidate.profile_id, candidate.sha256
    release['runtime_sha256'] = sha(mapping_path)
    old_sources = release['source_sha256']
    paths = set(old_sources) | {str(p.relative_to(PROJECT_ROOT)) for p in
                               (PROJECT_ROOT/'src/hexfellow_deploy').rglob('*.py')}
    paths.add(str((profile_dir/(candidate.profile_id+'.yaml')).relative_to(PROJECT_ROOT)))
    paths.add(str(freeze_path.relative_to(PROJECT_ROOT)))
    release['source_sha256'] = {name:sha(PROJECT_ROOT/name) for name in sorted(paths)}
    release['general_ik_research'] = dict(source_deployment=str(source), source_release_sha256=sha(original_release),
        source_mapping_sha256=sha(original_mapping), hardware_tested=False, qualification_changed=False,
        selected_profile=candidate.profile_id, control_period_s=.01,
        changed_profile_sections=sorted(differences),
        note='Operator-requested hardware evaluation of the frozen benchmark profile; not hardware qualification.')
    (output/'research_release.json').write_text(json.dumps(release,indent=2)+'\n')
    report = dict(deployment=str(config_path), profile=candidate.profile_id, profile_sha256=candidate.sha256,
                  control_period_s=.01, commissioned=config.get('commissioned'), hardware_qualified=False,
                  review_unchanged=mapping.get('review')==json.loads(original_mapping.read_text()).get('review'),
                  devices_opened=False)
    (output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-deployment', required=True)
    parser.add_argument('--output', required=True, help='New local directory; never overwrite existing bindings')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_deployment,args.output),indent=2))


if __name__ == '__main__':
    main()
