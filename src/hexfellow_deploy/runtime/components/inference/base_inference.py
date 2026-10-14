from typing import List, Optional
from multiprocessing.synchronize import Lock as LockType

from hexfellow_deploy.runtime.components.base_component_with_comm import DABaseComponentWithComm
from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm


class DABaseInference(DABaseComponentWithComm):
    """Base class for all inference components.

    All inference components must inherit from this class.
    """

    def __init__(self, name: str, comms: List[DASHMNumPyComm], time_budget: float):
        super().__init__(name, comms, time_budget)
        self._start_inference_lock: Optional[LockType] = (
            None  # signals when to start inference
        )
        self._end_inference_lock: Optional[LockType] = None  # signals when to end inference

    def set_start_inference_lock(self, lock: LockType) -> None:
        """Set the lock to start inference."""
        self._start_inference_lock = lock

    def set_end_inference_lock(self, lock: LockType) -> None:
        """Set the lock to end inference."""
        self._end_inference_lock = lock

    @property
    def start_inference_lock(self) -> Optional[LockType]:
        """The lock to start inference."""
        return self._start_inference_lock

    @property
    def end_inference_lock(self) -> Optional[LockType]:
        """The lock to end inference."""
        return self._end_inference_lock
