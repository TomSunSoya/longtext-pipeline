"""Provider Registry for LLM client management.

This module provides a registry system for managing multiple LLM providers,
enabling registration, retrieval, and client creation based on configuration.

The registry supports:
- Provider registration with metadata
- Provider lookup by name or type
- Client instance creation from provider configuration
"""

from dataclasses import dataclass
from typing import Any, Callable

from .base import LLMClient


@dataclass
class ProviderInfo:
    """Metadata about a registered LLM provider.

    Attributes:
        name: Unique identifier for the provider (e.g., "openai", "openrouter")
        display_name: Human-readable name for the provider
        description: Description of the provider's capabilities
        client_class: The LLMClient subclass for this provider
        default_model: Default model name to use with this provider
        supported_features: List of features supported by this provider
        config_schema: Optional schema validate provider config
    """

    name: str
    client_class: type[LLMClient]
    display_name: str = ""
    description: str = ""
    default_model: str = ""
    supported_features: list[str] | None = None
    config_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Set default display_name if not provided."""
        if not self.display_name:
            self.display_name = self.name.capitalize()


class ProviderRegistry:
    """Registry for managing LLM provider configurations and client creation.

    The registry provides a central place to:
    - Register new LLM providers
    - Look up provider metadata
    - Create client instances from configuration

    Providers are stored by name and can be retrieved by name or type.
    Client creation is delegated to the provider's client_class.

    Example:
        >>> registry = ProviderRegistry()
        >>> registry.register_provider(
        ...     ProviderInfo(
        ...         name="openai",
        ...         client_class=OpenAICompatibleClient,
        ...         display_name="OpenAI Compatible"
        ...     )
        ... )
        >>> client = registry.create_client("openai", {"model": "gpt-4o"})
    """

    def __init__(self) -> None:
        """Initialize the provider registry with no registered providers."""
        self._providers: dict[str, ProviderInfo] = {}
        self._client_factory: dict[str, Callable[[dict[str, Any]], LLMClient]] = {}

    def register_provider(self, provider_info: ProviderInfo) -> None:
        """Register a new provider with the registry.

        Args:
            provider_info: ProviderInfo containing provider metadata and client class

        Raises:
            ValueError: If provider with same name is already registered
        """
        if provider_info.name in self._providers:
            raise ValueError(f"Provider '{provider_info.name}' is already registered")

        self._providers[provider_info.name] = provider_info

        # Create factory function for client creation
        def factory(config: dict[str, Any]) -> LLMClient:
            return provider_info.client_class(**config)

        self._client_factory[provider_info.name] = factory

    def unregister_provider(self, name: str) -> bool:
        """Remove a provider from the registry.

        Args:
            name: Name of the provider to remove

        Returns:
            True if provider was found and removed, False otherwise
        """
        if name in self._providers:
            del self._providers[name]
            del self._client_factory[name]
            return True
        return False

    def get_provider(self, name: str) -> ProviderInfo:
        """Get provider metadata by name.

        Args:
            name: Name of the provider to look up

        Returns:
            ProviderInfo for the requested provider

        Raises:
            KeyError: If provider is not registered
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' is not registered")
        return self._providers[name]

    def has_provider(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._providers

    def list_providers(self) -> list[str]:
        """Get list of all registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def get_client_factory(self, name: str) -> Callable[[dict[str, Any]], LLMClient]:
        """Get the client factory function for a provider.

        Args:
            name: Name of the provider

        Returns:
            Factory function that creates LLMClient instances

        Raises:
            KeyError: If provider is not registered
        """
        if name not in self._client_factory:
            raise KeyError(f"Provider '{name}' is not registered")
        return self._client_factory[name]

    def create_client(self, name: str, config: dict[str, Any]) -> LLMClient:
        """Create an LLM client instance for a provider.

        Args:
            name: Name of the provider to use
            config: Configuration dictionary for the client

        Returns:
            Configured LLMClient instance

        Raises:
            KeyError: If provider is not registered
        """
        factory = self.get_client_factory(name)
        return factory(config)

    def create_from_config(
        self, provider_name: str, config: dict[str, Any]
    ) -> LLMClient:
        """Create a client using configuration with provider lookup.

        This is a convenience method that combines provider lookup
        and client creation.

        Args:
            provider_name: Name of the provider
            config: Full configuration dictionary

        Returns:
            Configured LLMClient instance
        """
        provider = self.get_provider(provider_name)
        provider_config: dict[str, Any] = config.get(provider_name, {})

        # Merge provider-specific config with base config
        base_config: dict[str, Any] = {
            "model": config.get("model", provider.default_model),
            "api_key": config.get("api_key"),
            "base_url": config.get("base_url"),
            "timeout": config.get("timeout"),
            "temperature": config.get("temperature", 0.7),
        }
        base_config.update(provider_config)

        return self.create_client(provider_name, base_config)


# Default registry instance
_default_registry: ProviderRegistry | None = None


def get_default_registry() -> ProviderRegistry:
    """Get the global default provider registry.

    Returns:
        The singleton ProviderRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ProviderRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the global default registry to None.

    This is useful for testing to ensure clean state between tests.
    """
    global _default_registry
    _default_registry = None
