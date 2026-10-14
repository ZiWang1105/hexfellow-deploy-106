from typing import List
import logging

from hexfellow_deploy.runtime.components.base_component import DABaseComponent
from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm


class DABaseComponentWithComm(DABaseComponent):
    """Base class for all components that have a communication channel.
    Currently, only DASHMNumPyComm is supported.
    """

    def __init__(self, name: str, comms: List[DASHMNumPyComm], time_budget: float):
        logging.info(f"Initializing component {name}")
        super().__init__(name)
        self._comms = comms
        self._time_budget = time_budget

    @property
    def comms(self) -> List[DASHMNumPyComm]:
        """Get the list of comms."""
        return self._comms

    @property
    def time_budget(self) -> float:
        """Get the time budget."""
        return self._time_budget
