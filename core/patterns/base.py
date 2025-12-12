"""
Base class for orchestration pattern handlers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.orchestrator import Orchestrator


class PatternHandler(ABC):
    """Abstract base class for orchestration pattern handlers."""

    # pylint: disable=too-few-public-methods

    def __init__(self, orchestrator: "Orchestrator") -> None:
        """Initialize the handler.

        Args:
            orchestrator: The orchestrator instance
        """
        self.orchestrator = orchestrator

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the pattern.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the execution
        """
