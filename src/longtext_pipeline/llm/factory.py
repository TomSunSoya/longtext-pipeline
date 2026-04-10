"""LLM client factory.

This module provides a factory function for creating LLM client instances
based on configuration. The factory supports environment variable fallbacks
and provides a clean interface for future provider expansion.
"""

import os
from typing import Optional

from ..config import get_agent_model_config
from .base import LLMClient
from .openai_compatible import OpenAICompatibleClient


def get_llm_client(
    config: dict,
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    temperature: Optional[float] = None,
) -> LLMClient:
    """Create an LLM client instance based on configuration.
    
    This factory function creates the appropriate LLM client based on the
    provided configuration. For MVP, only OpenAI-compatible clients are
    supported, but the interface is designed for easy provider expansion.
    
    Configuration precedence (highest to lowest):
    1. Explicit function arguments
    2. Agent-specific config (config.agents.{agent_type}.model) if agent_type provided
    3. Config dictionary values
    4. Environment variables
    5. Provider defaults
    
    Args:
        config: Configuration dictionary containing LLM settings
        agent_type: Optional agent type for agent-specific model lookup
                   (e.g., 'summarizer', 'stage_synthesizer', 'analyst', 'auditor').
                   When provided, uses config.agents.{agent_type}.model if available,
                   falling back to config.model. When None, uses config.model directly.
        model: Model name override (default: from config or env)
        api_key: API key override (default: from config or env)
        base_url: Base URL override (default: from config or env)
        timeout: Timeout override in seconds (default: from config or 30.0)
        temperature: Temperature override (default: from config or 0.7)

    Returns:
        Configured LLMClient instance (OpenAICompatibleClient for MVP)
        
    Raises:
        ValueError: If an unsupported provider is specified
        TypeError: If config is not a dictionary
        ConfigError: If agent_type is provided but unknown
        
    Example:
        >>> config = {
        ...     "model": "gpt-4o-mini",
        ...     "agents": {
        ...         "summarizer": {"model": {"name": "claude-sonnet"}}
        ...     }
        ... }
        >>> client = get_llm_client(config)  # Uses gpt-4o-mini
        >>> summarizer_client = get_llm_client(config, agent_type='summarizer')  # Uses claude-sonnet
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")
    
    # Get model config based on agent_type (if provided)
    if agent_type is not None:
        # Use agent-specific model config with fallback to top-level model config
        model_config = get_agent_model_config(config, agent_type)
    else:
        # Use top-level model config directly
        model_config = config.get("model", config)
    
    # Extract values from model_config with environment variable fallbacks
    # Use OpenAICompatibleClient defaults when no value is provided
    resolved_model = (
        model
        or model_config.get("model")
        or model_config.get("name")
        or os.getenv("LONGTEXT_MODEL_NAME")
        or OpenAICompatibleClient.DEFAULT_MODEL
    )
    
    resolved_api_key = (
        api_key
        or model_config.get("api_key")
        or os.getenv("OPENAI_API_KEY")
    )
    
    resolved_base_url = (
        base_url
        or model_config.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or OpenAICompatibleClient.DEFAULT_BASE_URL
    )
    
    resolved_timeout = (
        timeout
        or model_config.get("timeout")
        or OpenAICompatibleClient.DEFAULT_TIMEOUT
    )

    resolved_temperature = (
        temperature
        if temperature is not None
        else model_config.get("temperature", 0.7)
    )
    
    # Determine provider (default to "openai" for MVP)
    provider = model_config.get("provider", "openai").lower()
    
    # Create appropriate client based on provider
    # MVP only supports OpenAI-compatible providers
    if provider in ("openai", "openrouter", "ollama", "vllm"):
        return OpenAICompatibleClient(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=resolved_timeout,
            temperature=resolved_temperature,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"MVP supports only OpenAI-compatible providers (openai, openrouter, ollama, vllm)."
        )
