"""
gmeet_pipeline.llm — LLM abstraction layer.

Public API:
    BaseLLM              — abstract base class
    SimpleOpenRouterLLM  — single-model OpenRouter implementation
    VoiceGatewayLLM      — memory-aware multi-model OpenRouter implementation
"""

from .base import BaseLLM
from .openrouter import SimpleOpenRouterLLM, VoiceGatewayLLM

__all__ = [
    "BaseLLM",
    "SimpleOpenRouterLLM",
    "VoiceGatewayLLM",
]
