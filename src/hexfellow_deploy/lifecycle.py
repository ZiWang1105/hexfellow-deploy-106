"""Explicit new-session acquisition; a retired controller is never resumed."""
from dataclasses import dataclass
import math

from .reference import ReferenceOwner


@dataclass(frozen=True)
class RestartProof:
    operator_requested: bool
    sender_joined: bool
    previous_session: str
    new_session: str
    claimed_session: str
    feedback_session: str
    epoch: object
    feedback_epoch: object
    claim_sequence: int
    feedback_sequence: int
    now_s: float
    feedback_received_s: float
    maximum_feedback_age_s: float
    plant_interrupted: bool
    active_reference: object
    application_receipt: object
    safety_fault: object

    def validate(self):
        if self.operator_requested is not True or self.sender_joined is not True:
            raise ValueError('Explicit operator request and joined previous sender are required')
        if (not isinstance(self.previous_session, str) or not self.previous_session
                or not isinstance(self.new_session, str) or not self.new_session
                or self.previous_session == self.new_session
                or self.claimed_session != self.new_session or self.feedback_session != self.new_session):
            raise ValueError('A distinct plant-confirmed owner is required')
        if self.epoch is None or self.feedback_epoch != self.epoch:
            raise ValueError('Restart requires unchanged plant epoch')
        if (type(self.claim_sequence) is not int or type(self.feedback_sequence) is not int
                or self.feedback_sequence <= self.claim_sequence):
            raise ValueError('Restart requires feedback after the acknowledged claim')
        clocks = (self.now_s, self.feedback_received_s, self.maximum_feedback_age_s)
        if (any(isinstance(x, bool) or not math.isfinite(x) for x in clocks)
                or self.maximum_feedback_age_s <= 0
                or not 0 <= self.now_s-self.feedback_received_s <= self.maximum_feedback_age_s):
            raise ValueError('Restart requires fresh physical feedback')
        if (self.plant_interrupted is not False or self.active_reference is not None
                or self.application_receipt is not None or self.safety_fault is not None):
            raise ValueError('Plant claim must clear old motion; safety faults cannot be cleared')


def restart_execution(previous, previous_owner, factory, *, proof, record_retirement=None):
    """Retire pending evidence and construct a new, still-unacquired executor.

    No acknowledgement or reset is issued to the old executor. The adapter must
    still complete its original measured stationary window and first actual
    application receipt before reporting Resume complete. Factory failure leaves
    the old owner retired; this function never retries or arms a transport.
    """
    proof.validate()
    audit = dict(event='explicit_controller_lifecycle_retired',
                 previous_session=proof.previous_session, new_session=proof.new_session,
                 epoch=proof.epoch, generated=previous_owner.generated,
                 accepted=previous_owner.accepted, pending=previous_owner.pending,
                 owner_fault=previous_owner.fault,
                 execution_fault=str(previous.fault) if previous.fault else None,
                 execution_sequence=previous.sequence,
                 pending_disposition='retired_without_acknowledgement',
                 claim_sequence=proof.claim_sequence, feedback_sequence=proof.feedback_sequence)
    previous_owner.fail('Retired by explicit operator restart into a newly claimed session')
    if record_retirement is not None:
        record_retirement(audit)
    fresh = factory()
    if (fresh is previous or fresh.controller is previous.controller
            or fresh.controller.core is previous.controller.core
            or fresh.guard is not None or fresh.sequence != 0 or fresh.fault is not None):
        raise ValueError('Restart requires a distinct, unacquired controller and executor')
    return fresh, ReferenceOwner(), audit
