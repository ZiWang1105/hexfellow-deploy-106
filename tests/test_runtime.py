import math
import multiprocessing as mp
import time
import uuid

import pytest

from hexfellow_deploy.reference import Receipt, ReferenceOwner
from hexfellow_deploy.timing import CycleBudget
from hexfellow_deploy.topics import JsonTopic


def test_overrun_warns_and_remaining_sleep_is_nonnegative():
    now = [1.]
    budget = CycleBudget(.02, .006, clock=lambda: now[0])
    now[0] = 1.003
    assert budget.finish(1.)['sleep_s'] == pytest.approx(.017)
    now[0] = 1.030
    result = budget.finish(1.)
    assert result['budget_overrun'] and result['sleep_s'] == 0
    now[0] = 2.001
    assert not budget.finish(2.)['budget_overrun']


def test_pending_reference_cannot_advance_solver_history():
    owner = ReferenceOwner()
    history = []
    def solve():
        history.append(len(history)+1)
        return history[-1]
    assert owner.generate(solve) == (1, 1)
    for _ in range(100):
        assert owner.generate(solve) is None
    assert history == [1]
    owner.acknowledge(Receipt(1, 1., 'sdk_accepted'))
    assert owner.generate(solve) == (2, 2)
    owner.fail('Submission failed')
    with pytest.raises(RuntimeError, match='Submission failed'):
        owner.generate(solve)
    assert history == [1, 2] and owner.accepted == 1


@pytest.mark.parametrize('receipt', [Receipt(2, 1., 'sdk_accepted'),
    Receipt(1, math.nan, 'sdk_accepted'), Receipt(1, 1., 'motor_reached')])
def test_invalid_acceptance_latches_fault(receipt):
    owner = ReferenceOwner()
    owner.generate(lambda: None)
    with pytest.raises(RuntimeError):
        owner.acknowledge(receipt)
    assert owner.fault and owner.accepted == 0


def test_da_shared_memory_is_latest_only_and_coherent_across_processes():
    topic = JsonTopic('test_'+uuid.uuid4().hex, capacity=4096)
    topic.initialize()
    def send():
        for index in range(100):
            while not topic.put(dict(index=index, values=[index]*50), timeout=.01):
                pass
    process = mp.Process(target=send)
    try:
        process.start()
        deadline = time.monotonic()+3
        row = None
        while time.monotonic() < deadline:
            row = topic.get(timeout=.01)
            if row:
                assert row['values'] == [row['index']]*50
                if row['index'] == 99:
                    break
        assert row['index'] == 99
        process.join(2)
        assert process.exitcode == 0
        with pytest.raises(ValueError, match='capacity'):
            topic.put({'value': 'x'*5000})
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        topic.clean_up()
