"""Relocated code must bind to this repository without promoting hardware evidence."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from hexfellow_deploy.robotics.hardware import highspeed_trial
from hexfellow_deploy.robotics.hardware.x7 import HardwareSafetyError


def test_research_binding_checks_owned_source_root_and_original_evidence(tmp_path, monkeypatch):
    code = tmp_path/'src/hexfellow_deploy/controller.py'
    code.parent.mkdir(parents=True)
    code.write_text('owned source\n')
    mapping = tmp_path/'mapping.json'
    mapping.write_text('{}')
    profile = SimpleNamespace(profile_id=highspeed_trial.HIGHSPEED_PROFILE,
                              schema_version=23, base_control_enabled=False,
                              lift_velocity_m_s=.24, sha256='profile-hash')
    record = dict(profile_id=profile.profile_id, profile_sha256=profile.sha256,
                  authorization='operator_requested_research_trial', hardware_qualified=False,
                  runtime_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
                  source_sha256={str(code.relative_to(tmp_path)): hashlib.sha256(code.read_bytes()).hexdigest()})
    release = tmp_path/'release.json'
    release.write_text(json.dumps(record))
    monkeypatch.setattr(highspeed_trial, 'REPOSITORY_ROOT', tmp_path)
    highspeed_trial.require_research_release(profile, release, mapping)
    code.write_text('changed source\n')
    with pytest.raises(HardwareSafetyError, match='binding mismatch'):
        highspeed_trial.require_research_release(profile, release, mapping)
    assert json.loads(release.read_text()) == record
