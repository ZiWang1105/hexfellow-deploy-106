from typing import List, Optional
from multiprocessing.synchronize import Lock as LockType

from hexfellow_deploy.runtime.components.base_component_with_comm import DABaseComponentWithComm
from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm


class DABaseOutput(DABaseComponentWithComm):
    """Base class for all inference outputs.

    All inference outputs must inherit from this class.
    """

    def __init__(self, name: str, comms: List[DASHMNumPyComm], time_budget: float):
        super().__init__(name, comms, time_budget)
        self._start_output_lock: Optional[LockType] = None  # signals when to start output
        self._end_output_lock: Optional[LockType] = None  # signals when to end output

    def set_start_output_lock(self, lock: LockType) -> None:
        """Set the lock to start output."""
        self._start_output_lock = lock

    def set_end_output_lock(self, lock: LockType) -> None:
        """Set the lock to end output."""
        self._end_output_lock = lock

    @property
    def start_output_lock(self) -> Optional[LockType]:
        """The lock to start output."""
        return self._start_output_lock

    @property
    def end_output_lock(self) -> Optional[LockType]:
        """The lock to end output."""
        return self._end_output_lock
