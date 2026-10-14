from abc import ABC, abstractmethod
from multiprocessing import Value

from hexfellow_deploy.runtime.utils.naming import clean_name


class DABaseComponent(ABC):
    """Base class for all components.

    All components must inherit from this class.
    """

    def __init__(self, name: str):
        self._raw_name = name
        self._name = clean_name(name)
        self._initialized = Value("b", False)
        self._ready = Value("b", False)
        self._stopping = Value("b", False)
        self._cleaned_up = Value("b", False)
        self._global_step = Value("i", 0)

    def __str__(self):
        return self._name

    @property
    def name(self) -> str:
        """The name of the component."""
        return self._name

    @property
    def raw_name(self) -> str:
        """The raw name of the component."""
        return self._raw_name

    @property
    def initialized(self) -> bool:
        """Whether the component is initialized."""
        return self._initialized.value

    @property
    def cleaned_up(self) -> bool:
        """Whether the component is cleaned up."""
        return self._cleaned_up.value

    @property
    def ready(self) -> bool:
        """Whether the component is ready."""
        return self._ready.value

    @property
    def stopping(self) -> bool:
        """Whether the component is stopping."""
        return self._stopping.value

    def signal_stop(self) -> None:
        """Signal the component to stop."""
        self._stopping.value = True

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the component.

        Returns:
            bool: True if the component is initialized successfully, False otherwise.
        """

    @abstractmethod
    def run(self) -> None:
        """Run the component in a separate process.

        This method should be implemented by the subclass.
        """

    @abstractmethod
    def clean_up(self) -> bool:
        """Clean up the component.

        Returns:
            bool: True if the component is cleaned up successfully, False otherwise.
        """

    @property
    def global_step(self) -> int:
        """The global step of the component."""
        return self._global_step.value

    @global_step.setter
    def global_step(self, value: int) -> None:
        self._global_step.value = value

    def tick(self) -> None:
        """Tick the component."""
        with self._global_step.get_lock():
            self._global_step.value += 1
