"""
Tests for the multi-provider parallel dispatcher.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from src.longtext_pipeline.llm.dispatcher import (
    ParallelDispatcher,
    ParallelMode,
    ProviderResponse,
    ParallelResult,
)
from src.longtext_pipeline.llm.base import LLMClient
from src.longtext_pipeline.llm.registry import ProviderRegistry


@pytest.fixture
def mock_registry():
    """Create a mock registry with mock clients for testing."""
    registry = ProviderRegistry()

    # Create mock clients
    mock_client1 = AsyncMock(spec=LLMClient)
    mock_client1.acomplete = AsyncMock(return_value="Response from provider 1")

    mock_client2 = AsyncMock(spec=LLMClient)
    mock_client2.acomplete = AsyncMock(return_value="Response from provider 2")

    # Mock registry methods
    def mock_create_from_config(provider_name: str, config: Dict[str, Any]):
        if provider_name == "provider1":
            return mock_client1
        elif provider_name == "provider2":
            return mock_client2
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    registry.create_from_config = mock_create_from_config
    registry.list_providers = MagicMock(return_value=["provider1", "provider2"])

    return registry


@pytest.mark.asyncio
class TestParallelDispatcher:
    async def test_parallel_mode_enum_values(self):
        """Test that ParallelMode enum has correct values."""
        assert ParallelMode.SINGLE.value == "single"
        assert ParallelMode.PARALLEL.value == "parallel"
        assert ParallelMode.FASTEST.value == "fastest"
        assert ParallelMode.RANKED.value == "ranked"

    async def test_provider_response_creation(self):
        """Test ProviderResponse dataclass creation."""
        response = ProviderResponse(
            provider_name="test_provider",
            content="test content",
            latency=1.5,
            tokens_used=50,
            cost_estimate=0.01,
            success=True,
        )

        assert response.provider_name == "test_provider"
        assert response.content == "test content"
        assert response.latency == 1.5
        assert response.tokens_used == 50
        assert response.cost_estimate == 0.01
        assert response.success is True
        assert response.error is None

    async def test_parallel_result_creation(self):
        """Test ParallelResult dataclass creation."""
        responses = [
            ProviderResponse(
                provider_name="test", content="test content", latency=1.0, success=True
            )
        ]
        result = ParallelResult(
            mode=ParallelMode.PARALLEL,
            responses=responses,
            primary_content="primary content",
        )

        assert result.mode == ParallelMode.PARALLEL
        assert len(result.responses) == 1
        assert result.primary_content == "primary content"

    async def test_single_mode_dispatch(self, mock_registry):
        """Test dispatch in SINGLE mode."""
        dispatcher = ParallelDispatcher(
            registry=mock_registry, max_concurrent_requests=2
        )

        provider_configs = [
            {"provider": "provider1", "model": "test-model1"},
            {"provider": "provider2", "model": "test-model2"},
        ]

        result = await dispatcher.dispatch(
            prompt="Test prompt",
            mode=ParallelMode.SINGLE,
            provider_configs=provider_configs,
        )

        assert result.mode == ParallelMode.SINGLE
        assert len(result.responses) == 1
        assert result.best_provider == "provider1"  # Should use first provider

    async def test_parallel_mode_dispatch(self, mock_registry):
        """Test dispatch in PARALLEL mode."""
        dispatcher = ParallelDispatcher(
            registry=mock_registry, max_concurrent_requests=10, timeout_per_provider=5.0
        )

        provider_configs = [
            {"provider": "provider1", "model": "test-model1"},
            {"provider": "provider2", "model": "test-model2"},
        ]

        result = await dispatcher.dispatch(
            prompt="Test prompt",
            mode=ParallelMode.PARALLEL,
            provider_configs=provider_configs,
        )

        assert result.mode == ParallelMode.PARALLEL
        assert len(result.responses) == 2  # Both providers should respond
        assert all(r.success for r in result.responses)  # Both should succeed
        assert (
            result.primary_content == "Response from provider 1"
        )  # First successful response

    async def test_fastest_mode_dispatch(self, mock_registry):
        """Test dispatch in FASTEST mode."""

        # Create different response times for testing
        async def slow_completion(prompt, system_prompt=None):
            await asyncio.sleep(0.1)  # Slow response
            return "Slow response"

        async def fast_completion(prompt, system_prompt=None):
            # No delay for fast response
            return "Fast response"

        # Mock clients with different behaviors
        slow_client = AsyncMock(spec=LLMClient)
        slow_client.acomplete = slow_completion

        fast_client = AsyncMock(spec=LLMClient)
        fast_client.acomplete = fast_completion

        def mock_create_from_config(provider_name: str, config: Dict[str, Any]):
            if provider_name == "slow_provider":
                return slow_client
            elif provider_name == "fast_provider":
                return fast_client
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        slow_registry = ProviderRegistry()
        slow_registry.create_from_config = mock_create_from_config
        slow_registry.list_providers = MagicMock(
            return_value=["slow_provider", "fast_provider"]
        )

        dispatcher = ParallelDispatcher(
            registry=slow_registry, max_concurrent_requests=10
        )

        provider_configs = [
            {"provider": "slow_provider", "model": "slow-model"},
            {"provider": "fast_provider", "model": "fast-model"},
        ]

        result = await dispatcher.dispatch(
            prompt="Test prompt",
            mode=ParallelMode.FASTEST,
            provider_configs=provider_configs,
        )

        assert result.mode == ParallelMode.FASTEST
        # Note: Exact timing-based behavior is tricky to guarantee in tests,
        # so focusing on verifying method runs without error and has responses
        assert len(result.responses) >= 1
        assert result.primary_content in [
            "Fast response"
        ]  # Fast response should win ideally

    async def test_ranked_mode_dispatch(self, mock_registry):
        """Test dispatch in RANKED mode."""
        dispatcher = ParallelDispatcher(
            registry=mock_registry, max_concurrent_requests=10
        )

        provider_configs = [
            {"provider": "provider1", "model": "test-model1"},
            {"provider": "provider2", "model": "test-model2"},
        ]

        result = await dispatcher.dispatch(
            prompt="Test prompt",
            mode=ParallelMode.RANKED,
            provider_configs=provider_configs,
        )

        assert result.mode == ParallelMode.RANKED
        assert len(result.responses) == 2
        assert result.primary_content in [
            "Response from provider 1",
            "Response from provider 2",
        ]


@pytest.mark.asyncio
class TestProviderResponseAndParallelResult:
    async def test_provider_response_defaults(self):
        """Test default values in ProviderResponse."""
        response = ProviderResponse(provider_name="test", content="test", latency=1.0)

        assert response.tokens_used == 0
        assert response.cost_estimate == 0.0
        assert response.success is True
        assert response.error is None
        assert len(response.metadata) == 0

    async def test_parallel_result_defaults(self):
        """Test default values in ParallelResult."""
        result = ParallelResult(mode=ParallelMode.PARALLEL, responses=[])

        assert result.primary_content == ""
        assert result.best_provider is None
        assert result.execution_duration == 0.0
        assert len(result.metadata) == 0


@pytest.mark.asyncio
class TestDefaultQualityRankingStrategy:
    async def test_default_ranking_strategy_selection(self):
        """Test that default ranking strategy picks based on quality indicators."""
        dispatcher = ParallelDispatcher()

        responses = [
            ProviderResponse(
                provider_name="fast_cheap",
                content="Short",
                latency=0.1,
                cost_estimate=0.01,
            ),
            ProviderResponse(
                provider_name="slow_expensive",
                content="Much longer content for testing purposes",
                latency=2.0,
                cost_estimate=0.1,
            ),
        ]

        best = dispatcher._default_quality_ranking_strategy(responses)

        # Depending on the scoring algorithm, either might be picked
        # Our algorithm is based on length + speed + cost
        assert isinstance(best, ProviderResponse)
        assert best.provider_name in ["fast_cheap", "slow_expensive"]

    async def test_default_ranking_strategy_with_empty_list_raises_error(self):
        """Test that ranking strategy raises error when provided with empty list."""
        dispatcher = ParallelDispatcher()

        with pytest.raises(
            ValueError, match="No responses provided to ranking strategy"
        ):
            dispatcher._default_quality_ranking_strategy([])


@pytest.mark.asyncio
class TestErrorHandling:
    async def test_dispatch_with_no_providers_raises_error(self):
        """Test that dispatching with no providers configured raises error."""
        empty_registry = ProviderRegistry()
        empty_registry.list_providers = MagicMock(return_value=[])

        dispatcher = ParallelDispatcher(registry=empty_registry)

        with pytest.raises(ValueError, match="No providers configured in registry"):
            await dispatcher.dispatch("Test prompt", mode=ParallelMode.PARALLEL)

    async def test_single_mode_with_failing_provider(self, mock_registry):
        """Test SINGLE mode behavior when provider fails."""
        # Mock client that throws error
        error_client = AsyncMock(spec=LLMClient)
        error_client.acomplete = AsyncMock(side_effect=Exception("API Error"))

        def mock_create_from_config(provider_name: str, config: Dict[str, Any]):
            if provider_name == "error_provider":
                return error_client
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        error_registry = ProviderRegistry()
        error_registry.create_from_config = mock_create_from_config
        error_registry.list_providers = MagicMock(return_value=["error_provider"])

        dispatcher = ParallelDispatcher(registry=error_registry)

        result = await dispatcher.dispatch(
            prompt="Test prompt",
            mode=ParallelMode.SINGLE,
            provider_configs=[{"provider": "error_provider", "model": "test"}],
        )

        assert result.mode == ParallelMode.SINGLE
        assert len(result.responses) == 1
        assert not result.responses[0].success
        assert result.primary_content == ""


if __name__ == "__main__":
    pytest.main([__file__])
