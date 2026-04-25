from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseTransport(ABC):
    @abstractmethod
    async def join(self, meeting_url: str, bot_name: str = "Hank Bob", **kwargs) -> dict:
        """Join a meeting. Returns dict with bot_id and status."""
        ...

    @abstractmethod
    async def leave(self, bot_id: str) -> dict:
        """Leave a meeting. Returns dict with status."""
        ...

    @abstractmethod
    async def get_status(self, bot_id: str) -> Optional[str]:
        """Get current bot status."""
        ...
