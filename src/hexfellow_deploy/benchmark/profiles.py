"""Create hash-bound experiment profiles without changing commissioned defaults."""
from pathlib import Path

import yaml

from hexfellow_deploy.robot.upstream import PROJECT_ROOT


def experiment_profile(output, *, version='v1', hz=50, fast=False, tuning=None, dynamics=False):
    source = PROJECT_ROOT/'robot_resources/config/control/motion_profiles'/f'x7-gp80-highspeed-fixed-base-{version}.yaml'
    document = yaml.safe_load(source.read_text())
    document['profile_id'] = 'x7-offline-benchmark'
    document['control_dt_s'] = 1./hz
    if fast:
        document['target_limits'].update(linear_velocity_m_s=1.5,
            linear_acceleration_m_s2=8., linear_jerk_m_s3=120.)
    document['controller_tuning'].update(tuning or {})
    if dynamics:
        document['mit_dynamic_feedforward'] = dict(coulomb_nm=[0.]*7, viscous_nm_s=[0.]*7,
            armature_kg_m2=[0.]*7, inertia_scale=1., coulomb_velocity_epsilon_rad_s=.02,
            headroom_fraction=.5, source='Offline nominal-model inertia ablation; not hardware identified')
    document['qualification'] = dict(hardware='uncommissioned', simulation='candidate')
    document['provenance']['benchmark'] = 'Derived offline experiment; MIT gains and joint/effort limits retained'
    path = Path(output)/'profile.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path
