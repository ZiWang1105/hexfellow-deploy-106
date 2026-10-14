from typing import List, Optional
from multiprocessing.synchronize import Lock as LockType
from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm
from hexfellow_deploy.runtime.components.base_component_with_comm import DABaseComponentWithComm


class DABaseInput(DABaseComponentWithComm):
    """Base class for all inference inputs.

    All inference inputs must inherit from this class.
    """

    def __init__(self, name: str, comms: List[DASHMNumPyComm], time_budget: float):
        super().__init__(name, comms, time_budget)
        self._start_gathering_lock: Optional[LockType] = (
            None  # signals when to start processing
        )
        self._end_gathering_lock: Optional[LockType] = (
            None  # signals when to end processing
        )

    def set_start_gathering_lock(self, lock: LockType) -> None:
        """Set the lock to start processing."""
        self._start_gathering_lock = lock

    def set_end_gathering_lock(self, lock: LockType) -> None:
        """Set the lock to end processing."""
        self._end_gathering_lock = lock

    @property
    def start_gathering_lock(self) -> Optional[LockType]:
        """The lock to start gathering."""
        return self._start_gathering_lock

    @property
    def end_gathering_lock(self) -> Optional[LockType]:
        """The lock to end gathering."""
        return self._end_gathering_lock
