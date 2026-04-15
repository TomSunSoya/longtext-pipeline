"""Tests for the Provider Registry module."""

import pytest

from src.longtext_pipeline.llm.registry import (
    ProviderRegistry,
    ProviderInfo,
    get_default_registry,
    reset_default_registry,
)
from src.longtext_pipeline.llm.base import LLMClient
from src.longtext_pipeline.llm.openai_compatible import OpenAICompatibleClient


class TestProviderInfo:
    """Test cases for ProviderInfo dataclass."""

    def test_provider_info_minimal(self):
        """Test ProviderInfo creation with minimal arguments."""
        provider_info = ProviderInfo(
            name="test",
            client_class=OpenAICompatibleClient,
        )

        assert provider_info.name == "test"
        assert provider_info.client_class == OpenAICompatibleClient
        # Default display_name should be capitalized name
        assert provider_info.display_name == "Test"
        assert provider_info.description == ""
        assert provider_info.default_model == ""
        assert provider_info.supported_features is None
        assert provider_info.config_schema is None

    def test_provider_info_with_all_fields(self):
        """Test ProviderInfo creation with all fields."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
            display_name="OpenAI",
            description="OpenAI compatible provider",
            default_model="gpt-4o-mini",
            supported_features=["json_mode", "streaming"],
            config_schema={"type": "object"},
        )

        assert provider_info.name == "openai"
        assert provider_info.display_name == "OpenAI"
        assert provider_info.description == "OpenAI compatible provider"
        assert provider_info.default_model == "gpt-4o-mini"
        assert provider_info.supported_features == ["json_mode", "streaming"]
        assert provider_info.config_schema == {"type": "object"}

    def test_provider_info_custom_display_name(self):
        """Test ProviderInfo respects custom display_name."""
        provider_info = ProviderInfo(
            name="my_provider",
            client_class=OpenAICompatibleClient,
            display_name="My Custom Provider",
        )

        assert provider_info.display_name == "My Custom Provider"


class TestProviderRegistry:
    """Test cases for ProviderRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return ProviderRegistry()

    def test_registry_initially_empty(self, registry):
        """Test that registry starts with no providers."""
        assert len(registry.list_providers()) == 0

    def test_register_provider(self, registry):
        """Test registering a provider."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )

        registry.register_provider(provider_info)

        assert "openai" in registry.list_providers()
        assert registry.has_provider("openai")

    def test_register_provider_raises_on_duplicate(self, registry):
        """Test that registering duplicate provider raises ValueError."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )

        registry.register_provider(provider_info)

        with pytest.raises(ValueError) as exc_info:
            registry.register_provider(provider_info)

        assert "already registered" in str(exc_info.value)

    def test_unregister_provider(self, registry):
        """Test unregistering a provider."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )

        registry.register_provider(provider_info)
        result = registry.unregister_provider("openai")

        assert result is True
        assert "openai" not in registry.list_providers()
        assert not registry.has_provider("openai")

    def test_unregister_nonexistent_provider(self, registry):
        """Test unregistering non-existent provider returns False."""
        result = registry.unregister_provider("nonexistent")

        assert result is False

    def test_get_provider(self, registry):
        """Test getting provider info."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )
        registry.register_provider(provider_info)

        retrieved = registry.get_provider("openai")

        assert retrieved.name == "openai"
        assert retrieved.client_class == OpenAICompatibleClient

    def test_get_provider_raises_on_missing(self, registry):
        """Test that getting non-existent provider raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_provider("nonexistent")

        assert "not registered" in str(exc_info.value)

    def test_has_provider(self, registry):
        """Test has_provider method."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )
        registry.register_provider(provider_info)

        assert registry.has_provider("openai") is True
        assert registry.has_provider("nonexistent") is False

    def test_list_providers(self, registry):
        """Test listing all registered providers."""
        registry.register_provider(
            ProviderInfo(name="provider1", client_class=OpenAICompatibleClient)
        )
        registry.register_provider(
            ProviderInfo(name="provider2", client_class=OpenAICompatibleClient)
        )

        providers = registry.list_providers()

        assert set(providers) == {"provider1", "provider2"}

    def test_get_client_factory(self, registry):
        """Test getting client factory function."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )
        registry.register_provider(provider_info)

        factory = registry.get_client_factory("openai")

        assert callable(factory)

    def test_get_client_factory_raises_on_missing(self, registry):
        """Test that getting non-existent factory raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_client_factory("nonexistent")

        assert "not registered" in str(exc_info.value)

    def test_create_client(self, registry):
        """Test creating a client from registry."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
        )
        registry.register_provider(provider_info)

        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "timeout": 120.0,
            "temperature": 0.7,
        }

        client = registry.create_client("openai", config)

        assert isinstance(client, LLMClient)
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "gpt-4o-mini"

    def test_create_client_raises_on_missing_provider(self, registry):
        """Test that creating client for non-existent provider raises KeyError."""
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
        }

        with pytest.raises(KeyError) as exc_info:
            registry.create_client("nonexistent", config)

        assert "not registered" in str(exc_info.value)

    def test_create_from_config(self, registry):
        """Test creating client with full config lookup."""
        provider_info = ProviderInfo(
            name="openai",
            client_class=OpenAICompatibleClient,
            default_model="gpt-4o-mini",
        )
        registry.register_provider(provider_info)

        config = {
            "model": "custom-model",
            "api_key": "test-key",
            "openai": {
                "temperature": 0.2,
            },
        }

        client = registry.create_from_config("openai", config)

        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "custom-model"
        # Provider-specific config should override
        assert client.temperature == 0.2


class TestDefaultRegistry:
    """Test cases for default registry functions."""

    def test_get_default_registry_creates_instance(self):
        """Test that get_default_registry creates instance if None."""
        reset_default_registry()  # Ensure clean state

        registry = get_default_registry()

        assert isinstance(registry, ProviderRegistry)

    def test_get_default_registry_returns_same_instance(self):
        """Test that get_default_registry returns singleton."""
        reset_default_registry()  # Ensure clean state

        registry1 = get_default_registry()
        registry2 = get_default_registry()

        assert registry1 is registry2

    def test_reset_default_registry(self):
        """Test that reset_default_registry clears the singleton."""
        reset_default_registry()  # Ensure clean state

        registry1 = get_default_registry()
        assert registry1 is not None

        reset_default_registry()
        registry2 = get_default_registry()

        # Should be a new instance
        assert registry2 is not registry1
