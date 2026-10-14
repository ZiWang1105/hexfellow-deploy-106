import copy

import pytest
import yaml

from hexfellow_deploy.robotics.control.motion_profile import (
    DEFAULT_MOTION_PROFILE_ID,
    controller_config_for_profile,
    load_motion_profile,
)
from hexfellow_deploy.robotics.control.self_collision import load_collision_envelope


@pytest.mark.parametrize("invalid", [-0.1, 1.01, float("nan")])
def test_invalid_feedforward_gain_rejected(invalid):
    from hexfellow_deploy.robotics.control.ik import PinkControllerConfig

    with pytest.raises(ValueError):
        PinkControllerConfig(target_feedforward_gain=invalid)










def test_safe_default_requires_requalification_after_manufacturer_migration():
    profile = load_motion_profile()

    assert profile.profile_id == DEFAULT_MOTION_PROFILE_ID == "x7-safe-v1"
    assert profile.simulation_status == "candidate"




def test_profile_hash_is_canonical_and_velocity_cannot_exceed_vendor_caps(tmp_path):
    source = load_motion_profile("x7-mit-highspeed-v10")
    assert load_motion_profile(source.path).sha256 == source.sha256

    invalid = copy.deepcopy(dict(source.raw))
    invalid["profile_id"] = "invalid"
    invalid["arm_limits"]["velocity_rad_s"][0] = min(source.urdf_velocity_rad_s[0], source.sdk_velocity_rad_s[0]) + .01
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hard limit"):
        load_motion_profile(path)


def test_collision_envelope_rejects_a_stale_profile_or_geometry_hash():
    profile = load_motion_profile()
    with pytest.raises(ValueError, match="stale for the selected motion profile"):
        load_collision_envelope(
            profile.collision_envelope_config,
            profile_id=profile.profile_id,
            profile_sha256="0" * 64,
            geometry_payload_sha256=profile.collision_geometry_sha256,
        )
    with pytest.raises(ValueError, match="stale for the reviewed proxy geometry"):
        load_collision_envelope(
            profile.collision_envelope_config,
            profile_id=profile.profile_id,
            profile_sha256=profile.sha256,
            geometry_payload_sha256="0" * 64,
        )
