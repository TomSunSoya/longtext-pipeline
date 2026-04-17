"""LLM integration module.

This module provides the LLM abstraction layer for the longtext-pipeline.
All LLM provider implementations should inherit from the base classes defined
in this module.

Exports:
    LLMClient: Abstract base class defining the LLM interface
    OpenAICompatibleClient: OpenAI-compatible HTTP client implementation
    ProviderRegistry: Registry for managing LLM provider configurations
    ProviderInfo: Dataclass for provider metadata
    get_default_registry: Get the global default provider registry
    reset_default_registry: Reset the global default registry (for testing)
"""

from .base import LLMClient
from .openai_compatible import OpenAICompatibleClient
from .registry import (
    ProviderInfo,
    ProviderRegistry,
    get_default_registry,
    reset_default_registry,
)
from .dispatcher import (
    ParallelMode,
    ProviderResponse,
    ParallelResult,
)
from .results import (
    ProviderMetrics,
    QualityEstimate,
    QualityMetrics,
    RankingStrategy,
    ResultRanker,
    rank_responses,
)

__all__ = [
    "LLMClient",
    "OpenAICompatibleClient",
    "ProviderInfo",
    "ProviderRegistry",
    "get_default_registry",
    "reset_default_registry",
    "ParallelMode",
    "ProviderResponse",
    "ParallelResult",
    "ProviderMetrics",
    "QualityEstimate",
    "QualityMetrics",
    "RankingStrategy",
    "ResultRanker",
    "rank_responses",
]
