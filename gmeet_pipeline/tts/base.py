"""Abstract base class for TTS backends."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseTTS(ABC):
    """Interface that every TTS backend must implement."""

    @abstractmethod
    async def generate(self, text: str, bot_id: str) -> Optional[str]:
        """Generate TTS audio for *text* on behalf of *bot_id*.

        Returns a filename or identifier on success, or ``None`` on failure.
        """
        ...
