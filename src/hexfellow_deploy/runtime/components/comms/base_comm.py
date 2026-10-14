from typing import Any, Optional
from abc import abstractmethod

from hexfellow_deploy.runtime.components.base_component import DABaseComponent


class DABaseComm(DABaseComponent):
    """Base class for all communication components.

    All communication components must inherit from this class. Concrete comms own
    their own synchronization (e.g. a per-topic named semaphore); there is no
    externally-assigned lock.
    """

    @abstractmethod
    def send(self, data: Any, timeout: Optional[float] = None) -> bool:
        """Send data to the other end.

        Returns:
            bool: True if the data is sent successfully, False otherwise.
        """

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Any:
        """Receive data from the other end.

        Returns:
            Any: The received data.
        """
