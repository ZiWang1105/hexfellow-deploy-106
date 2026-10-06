"""DA lifecycle with cooperative shutdown for independently clocked interfaces."""
import time
from deploy_anything.manager import DAManager


class InterfaceManager(DAManager):
    def _run(self):
        # DA's parent _stopping is a plain bool, not shared after fork. For an
        # interface-only deployment use the components' shared stopping flags.
        # There is no inference/output barrier to schedule in this mode.
        if any((self._preprocess, self._input, self._inference, self._output)):
            raise ValueError("This manager owns independently clocked DA interfaces only")
        while not all(component.stopping for component in self._all_components):
            time.sleep(.01)
