"""LLM client factory.

This module provides a factory function for creating LLM client instances
based on configuration. The factory supports environment variable fallbacks
and provides a clean interface for future provider expansion.
"""

import os
from typing import Optional

from .base import LLMClient
from .openai_compatible import OpenAICompatibleClient


def get_llm_client(
    config: dict,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> LLMClient:
    """Create an LLM client instance based on configuration.
    
    This factory function creates the appropriate LLM client based on the
    provided configuration. For MVP, only OpenAI-compatible clients are
    supported, but the interface is designed for easy provider expansion.
    
    Configuration precedence (highest to lowest):
    1. Explicit function arguments
    2. Config dictionary values
    3. Environment variables
    4. Provider defaults
    
    Args:
        config: Configuration dictionary containing LLM settings
        model: Model name override (default: from config or env)
        api_key: API key override (default: from config or env)
        base_url: Base URL override (default: from config or env)
        timeout: Timeout override in seconds (default: from config or 30.0)
        
    Returns:
        Configured LLMClient instance (OpenAICompatibleClient for MVP)
        
    Raises:
        ValueError: If an unsupported provider is specified
        TypeError: If config is not a dictionary
        
    Example:
        >>> config = {
        ...     "model": "gpt-4o-mini",
        ...     "timeout": 60.0
        ... }
        >>> client = get_llm_client(config)
        >>> response = client.complete("Hello!")
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")
    
    # Extract values from config with environment variable fallbacks
    # Use OpenAICompatibleClient defaults when no value is provided
    resolved_model = (
        model
        or config.get("model")
        or config.get("name")
        or os.getenv("LONGTEXT_MODEL_NAME")
        or OpenAICompatibleClient.DEFAULT_MODEL
    )
    
    resolved_api_key = (
        api_key
        or config.get("api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    
    resolved_base_url = (
        base_url
        or config.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or OpenAICompatibleClient.DEFAULT_BASE_URL
    )
    
    resolved_timeout = (
        timeout
        or config.get("timeout")
        or OpenAICompatibleClient.DEFAULT_TIMEOUT
    )
    
    # Determine provider (default to "openai" for MVP)
    provider = config.get("provider", "openai").lower()
    
    # Create appropriate client based on provider
    # MVP only supports OpenAI-compatible providers
    if provider in ("openai", "openrouter", "ollama", "vllm"):
        return OpenAICompatibleClient(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=resolved_timeout,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"MVP supports only OpenAI-compatible providers (openai, openrouter, ollama, vllm)."
        )
