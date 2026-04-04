"""LLM integration module.

This module provides the LLM abstraction layer for the longtext-pipeline.
All LLM provider implementations should inherit from the base classes defined
in this module.

Exports:
    LLMClient: Abstract base class defining the LLM interface
    OpenAICompatibleClient: OpenAI-compatible HTTP client implementation
"""

from .base import LLMClient
from .openai_compatible import OpenAICompatibleClient

__all__ = ["LLMClient", "OpenAICompatibleClient"]
