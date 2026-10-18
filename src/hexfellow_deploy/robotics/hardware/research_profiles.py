"""Current GP80 research profile registry; never opens a device connection."""

from ..paths import CONFIG_DIR
from .x7 import HardwareSafetyError

PROFILE = 'x7-gp80-highspeed-fixed-base-v1'
CANDIDATE_PROFILE = 'x7-gp80-highspeed-fixed-base-v2'
HIGHSPEED_PROFILE = 'x7-gp80-highspeed-fixed-base-v1'
HIGHSPEED_V2_PROFILE = 'x7-gp80-highspeed-fixed-base-v2'
HIGHSPEED_V3_PROFILE = 'x7-gp80-highspeed-fixed-base-v3'
# Exact frozen benchmark candidate: research-only, never a commissioned default.
GENERAL_IK_PROFILE = 'x7-gp80-general-trajectory-sim-v2'
HIGHSPEED_PROFILES = frozenset({HIGHSPEED_PROFILE, HIGHSPEED_V2_PROFILE, HIGHSPEED_V3_PROFILE, GENERAL_IK_PROFILE})
SUPPORTED_PROFILES = {PROFILE, CANDIDATE_PROFILE, *HIGHSPEED_PROFILES}
RUNTIME = CONFIG_DIR / 'hardware/hexfellow_gp80_highspeed_j357.json'


def require_offline_validation(profile, *, mapping=None):
    """Require explicit research binding; never infer live permission from a score."""
    if profile.profile_id not in HIGHSPEED_PROFILES:
        raise HardwareSafetyError("unsupported research profile; restore historical experiments explicitly")
    from .highspeed_trial import require_research_release
    require_research_release(profile, getattr(mapping, "research_release", None),
                             getattr(mapping, "source_path", None))


def describe(profile, *, control_group='coupled'):
    """Display the selected experiment, including its absent collision check."""
    return (f"{profile.robot_model.model_id} | reference ≤{profile.target_linear_velocity_m_s*100:g} cm/s, "
            f"rotation ≤{profile.target_angular_velocity_rad_s*180/3.141592653589793:g} deg/s | "
            f"IK {14 if control_group == 'arms' else (18 if profile.base_control_enabled else 15)} variables | "
            f"control {control_group} | base {'active' if profile.base_control_enabled else 'disabled'} | "
            f"lift ≤{profile.lift_velocity_m_s:g} m/s, D/I ≤{profile.preparation_lift_velocity_m_s:g} m/s | "
            f"COLLISION {'ON' if profile.self_collision_enabled else 'OFF'} | hardware unqualified")
