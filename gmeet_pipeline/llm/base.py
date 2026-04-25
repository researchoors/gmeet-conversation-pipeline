"""
gmeet_pipeline.llm.base — Abstract base class for LLM backends.

All LLM implementations must implement ``generate()``, which receives
the conversation history, a new user message, and optional bot state,
returning either a response string or ``None`` when the agent should
stay silent.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Abstract base for LLM response generators."""

    @abstractmethod
    async def generate(
        self,
        conversation: list,
        message: str,
        bot_state: Optional[dict] = None,
    ) -> Optional[str]:
        """Generate a response to *message* given *conversation* history.

        Parameters
        ----------
        conversation : list
            Recent chat messages as dicts (``{"role": ..., "content": ...}``).
        message : str
            The latest user utterance to respond to.
        bot_state : dict or None
            Mutable per-bot state bag (e.g. for tracking expanded memory
            entries across turns).  Implementations may read *and* write.

        Returns
        -------
        str or None
            The assistant's reply, or ``None`` if the agent should stay
            silent.
        """
        ...
