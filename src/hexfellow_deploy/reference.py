"""Only one PinkIK result may be awaiting actuator acceptance.

PinkIK commits internally during step(). It does not expose rollback. Never
solve past a pending/rejected command or silently reset the solver history.
An adapter acknowledgement must state its scope; SDK acceptance is not motion.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Receipt:
    sequence: int
    accepted_at: float
    scope: str


class ReferenceOwner:
    def __init__(self):
        self.generated = 0
        self.accepted = 0
        self.pending = None
        self.fault = None
        self.last_receipt = None

    def generate(self, solve):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.pending is not None:
            return None
        try:
            result = solve()
        except Exception as error:
            self.fail(f"IK failed: {type(error).__name__}: {error}")
            raise
        self.generated += 1
        self.pending = self.generated
        return self.generated, result

    def acknowledge(self, receipt):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.pending is None or receipt.sequence != self.pending:
            self.fail("Actuator acknowledgement does not match pending reference")
            raise RuntimeError(self.fault)
        if not math.isfinite(receipt.accepted_at) or receipt.accepted_at <= 0:
            self.fail("Invalid actuator acknowledgement timestamp")
            raise RuntimeError(self.fault)
        if self.last_receipt and receipt.accepted_at < self.last_receipt.accepted_at:
            self.fail("Actuator acknowledgement timestamp regressed")
            raise RuntimeError(self.fault)
        if receipt.scope not in {"simulator_applied", "sdk_accepted", "udp_sent"}:
            self.fail("Unknown actuator acknowledgement scope")
            raise RuntimeError(self.fault)
        self.accepted = receipt.sequence
        self.last_receipt = receipt
        self.pending = None

    def fail(self, detail):
        if self.fault is None:
            self.fault = str(detail)
