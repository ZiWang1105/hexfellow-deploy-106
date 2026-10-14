"""Absolute slot and feedback deadlines shared by all whole-body controllers."""
import dataclasses
import math
import time

from .x7 import ControlDelay, FeedbackSkewSafetyError, FeedbackUnavailable, HardwareSafetyError

MAX_WAIT = .008
COHERENCE_WAIT = .004
# Admission reserve covers measured ~3ms IK plus validation/dispatch. This is
# scheduling headroom, not a replacement for the hard checks at actual send.
COMPLETION_RESERVE = .006


def feedback_deadline(state):
    from ..control.feedback_policy import received_deadline
    return received_deadline(state.received, state.feedback_policy)


def dispatch_deadline(state, created, slot_deadline):
    return min(slot_deadline, created + .020, feedback_deadline(state))


def check_compute_state(state, now, slot_deadline):
    state.validate(now)
    if not math.isfinite(slot_deadline):
        raise HardwareSafetyError("invalid control slot deadline")
    if now >= slot_deadline:
        raise ControlDelay("control slot expired before IK")
    if min(slot_deadline, feedback_deadline(state)) - now < COMPLETION_RESERVE - 1e-12:
        raise ControlDelay("insufficient remaining slot/feedback budget before IK")


def select_control_state(read, slot_deadline, *, clock=time.monotonic, sleep=time.sleep,
                         preferred_age_s=None):
    started = clock()
    if not math.isfinite(slot_deadline):
        raise HardwareSafetyError("invalid control slot deadline")
    deadline = min(started + MAX_WAIT, slot_deadline - COMPLETION_RESERVE)
    if preferred_age_s is not None and (not math.isfinite(preferred_age_s) or preferred_age_s <= 0):
        raise ValueError('Preferred feedback age must be positive and finite')
    preference_deadline = min(started + .0045, deadline)
    skew_deadline = None
    ipc_retries = 0
    while True:
        read_deadline = min(deadline, skew_deadline if skew_deadline is not None else deadline)
        try:
            state = read(fresh=False, deadline=read_deadline)
        except FeedbackUnavailable:
            ipc_retries += 1
            now = clock()
            if now >= read_deadline:
                raise ControlDelay("feedback IPC unavailable within control budget")
            sleep(min(.0002, read_deadline - now))
            continue
        now = clock()
        coherent = True
        try:
            state.validate(now)
        except FeedbackSkewSafetyError:
            coherent = False
            if skew_deadline is None:
                skew_deadline = min(now + COHERENCE_WAIT, deadline, feedback_deadline(state))
            if now >= skew_deadline:
                raise FeedbackSkewSafetyError("robot feedback skew exceeds 20 ms after bounded coherence wait")
        except ControlDelay:
            # A bounded-policy stale snapshot may be replaced within THIS slot.
            # It never reaches IK or extends command/feedback deadlines.
            if (state.latency_policy != "hold-resume" and state.feedback_policy != "theta-bounded") or now >= deadline:
                raise
            sleep(min(.0002, deadline - now))
            continue
        if now >= slot_deadline:
            raise ControlDelay("control slot expired while waiting for feedback")
        remaining = min(slot_deadline, feedback_deadline(state)) - now
        if skew_deadline is not None and now >= skew_deadline:
            raise FeedbackSkewSafetyError("feedback coherence recovery missed its deadline")
        if coherent and remaining >= COMPLETION_RESERVE - 1e-12:
            age = now - state.timestamp
            if preferred_age_s is not None and age > preferred_age_s and now < preference_deadline:
                # A healthy frame may still be older than necessary. Prefer a
                # newer actual observation, bounded by the original slot and
                # completion reserve. Never renew timestamps or reset history.
                sleep(min(.0002, preference_deadline - now))
                continue
            return dataclasses.replace(state, timing={**(state.timing or {}),
                "feedback_wait_started": started, "feedback_wait_ms": (now - started) * 1000,
                "feedback_selected_at": now, "slot_deadline": slot_deadline,
                "feedback_deadline": feedback_deadline(state), "remaining_budget_ms": remaining * 1000,
                "ipc_retries": ipc_retries, "coherence_waited": skew_deadline is not None,
                "preferred_age_s": preferred_age_s,
                "freshness_preference_met": preferred_age_s is None or age <= preferred_age_s})
        if now >= deadline:
            raise ControlDelay("no feedback with sufficient remaining control budget")
        sleep(min(.0002, (min(deadline, skew_deadline) if not coherent else deadline) - now))
