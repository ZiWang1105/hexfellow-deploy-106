from typing_extensions import override

import numpy as np
import time

from hexfellow_deploy.runtime.components.base_component_with_comm import DABaseComponentWithComm
from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm


class DABaseInterface(DABaseComponentWithComm):
    """Base class for all robot interfaces.

    All robot interfaces must inherit from this class.
    """


class TestInterface(DABaseInterface):
    """Test interface. Timing end-to-end delays."""

    def __init__(
        self, name: str, state_pub: DASHMNumPyComm, control_sub: DASHMNumPyComm
    ):
        super().__init__(name, [control_sub, state_pub])

        self._state_data = np.zeros_like(state_pub.receive())
        self._control_data = np.zeros_like(control_sub.receive())
        self._state_pub = state_pub
        self._control_sub = control_sub

    @override
    def initialize(self) -> bool:
        return True

    @override
    def run(self) -> None:
        """Run the test interface."""
        # generate random state data at 100 Hz
        last_received_data = None
        while not self.stopping:
            last_time = time.monotonic()
            self._state_data[:] = np.random.randn(self._state_data.shape[0])
            self._state_data[0, 0] = last_time
            self._state_pub.send(self._state_data)

            control_data = self._control_sub.receive()[0, 0]
            if last_received_data is not None:
                if control_data != last_received_data:
                    print(f"Latency: {(time.monotonic() - control_data) * 1000.0} ms")
            last_received_data = control_data

            time.sleep(1.0 / 100.0 - (time.monotonic() - last_time))
