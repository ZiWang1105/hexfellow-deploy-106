from dataclasses import replace
from types import SimpleNamespace

import pytest

from hexfellow_deploy.lifecycle import RestartProof, restart_execution
from hexfellow_deploy.reference import ReferenceOwner


def proof():
    return RestartProof(True, True, 'old', 'new', 'new', 'new', 'epoch', 'epoch',
                        10, 11, 2., 1.99, .5, False, None, None, None)


def execution():
    return SimpleNamespace(controller=SimpleNamespace(core=object()), guard=None, sequence=0, fault=None)


def test_explicit_restart_preserves_unacknowledged_history_and_returns_unacquired_core():
    old, owner, audits = execution(), ReferenceOwner(), []
    old.guard, old.sequence = object(), 4
    owner.generate(lambda: object())
    fresh, new_owner, audit = restart_execution(old, owner, execution, proof=proof(), record_retirement=audits.append)
    assert fresh.controller.core is not old.controller.core and fresh.guard is None
    assert old.sequence == 4 and owner.pending == 1 and owner.accepted == 0 and owner.fault
    assert new_owner.generated == new_owner.accepted == 0 and new_owner.pending is None
    assert audit['pending_disposition'] == 'retired_without_acknowledgement' and audits == [audit]


@pytest.mark.parametrize('field,value', [('operator_requested', False), ('sender_joined', False),
    ('new_session', 'old'), ('claimed_session', 'other'), ('feedback_session', 'old'),
    ('feedback_epoch', 'other'), ('feedback_sequence', 10), ('feedback_received_s', 1.),
    ('plant_interrupted', True), ('active_reference', 1), ('application_receipt', {}),
    ('safety_fault', 'overspeed')])
def test_unsafe_restart_does_not_create_or_retire_any_core(field, value):
    owner = ReferenceOwner()
    with pytest.raises(ValueError):
        restart_execution(execution(), owner, lambda: pytest.fail('Unsafe factory call'),
                          proof=replace(proof(), **{field: value}))
    assert owner.fault is None


def test_factory_failure_retains_retirement_audit_and_cannot_reuse_previous_core():
    old, owner, audit = execution(), ReferenceOwner(), []
    def fail():
        raise RuntimeError('Initialization failed')
    with pytest.raises(RuntimeError, match='Initialization failed'):
        restart_execution(old, owner, fail, proof=proof(), record_retirement=audit.append)
    assert owner.fault and len(audit) == 1
    with pytest.raises(ValueError, match='distinct'):
        restart_execution(old, ReferenceOwner(), lambda: old, proof=proof())
