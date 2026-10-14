from typing import List, Optional
from multiprocessing.synchronize import Lock as LockType

import numpy as np
from omegaconf import ListConfig

from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm
from hexfellow_deploy.runtime.components.base_component_with_comm import DABaseComponentWithComm


class DABasePreprocess(DABaseComponentWithComm):
    """Base class for all preprocess components.

    All preprocess components must inherit from this class.
    """

    def __init__(
        self,
        name: str,
        comms: List[DASHMNumPyComm],
        time_budget: float,
        scales: float | List[float],
        offsets: float | List[float],
    ):
        super().__init__(name, comms, time_budget)
        self._start_processing_lock: Optional[LockType] = (
            None  # signals when to start processing
        )
        self._end_processing_lock: Optional[LockType] = (
            None  # signals when to end processing
        )
        self._scales = (
            np.array(scales) if isinstance(scales, (list, ListConfig)) else scales
        )
        self._offsets = (
            np.array(offsets) if isinstance(offsets, (list, ListConfig)) else offsets
        )

    def set_start_processing_lock(self, lock: LockType) -> None:
        """Set the lock to start processing."""
        self._start_processing_lock = lock

    def set_end_processing_lock(self, lock: LockType) -> None:
        """Set the lock to end processing."""
        self._end_processing_lock = lock

    @property
    def start_processing_lock(self) -> Optional[LockType]:
        """The lock to start processing."""
        return self._start_processing_lock

    @property
    def end_processing_lock(self) -> Optional[LockType]:
        """The lock to end processing."""
        return self._end_processing_lock
