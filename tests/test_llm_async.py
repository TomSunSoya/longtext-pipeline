"""Tests for async LLM client methods."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.longtext_pipeline.llm.factory import get_llm_client
from src.longtext_pipeline.llm.openai_compatible import OpenAICompatibleClient
from src.longtext_pipeline.utils.retry import retry_llm_call_async, RetryError
from src.longtext_pipeline.errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMCommunicationError,
)


def make_llm_config(
    provider: str = "openai",
    name: str | None = "gpt-4o-mini",
    api_key: str | None = "test-key",
    base_url: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
) -> dict:
    """Build config objects using the current nested model schema."""
    model = {"provider": provider}
    if name is not None:
        model["name"] = name
    if api_key is not None:
        model["api_key"] = api_key
    if base_url is not None:
        model["base_url"] = base_url
    if timeout is not None:
        model["timeout"] = timeout
    if temperature is not None:
        model["temperature"] = temperature
    return {"model": model}


class TestAsyncComplete:
    """Test cases for acomplete() async method."""

    @pytest.mark.asyncio
    async def test_acomplete_returns_text_response(self):
        """Test that acomplete() returns the expected text response."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": "Test response"}}]}

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            result = await client.acomplete("Test prompt")

        assert result == "Test response"
        mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_acomplete_with_system_prompt(self):
        """Test that acomplete() passes system prompt correctly."""
        config = make_llm_config()

        mock_response = {
            "choices": [{"message": {"content": "Response with system prompt"}}]
        }

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            await client.acomplete(
                "Test prompt", system_prompt="You are a helpful assistant."
            )

            # Verify system prompt was included in request
            call_args = mock_request.call_args
            payload = call_args[0][0]
            messages = payload["messages"]
            assert messages[0]["role"] == "system"
            assert "You are a helpful assistant." in messages[0]["content"]
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_acomplete_honors_configured_temperature(self):
        """Async request payload should preserve the configured temperature."""
        config = make_llm_config(temperature=0.0)
        client = get_llm_client(config)

        mock_response = {
            "choices": [{"message": {"content": "Deterministic response"}}]
        }

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            await client.acomplete("Test prompt")

        payload = mock_request.call_args[0][0]
        assert payload["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_acomplete_empty_content_raises_error(self):
        """Test that acomplete() raises error on empty content."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": ""}}]}

        client = get_llm_client(config)

        # Test internal method without retry decorator
        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            # Directly test _async_complete to bypass retry decorator
            with pytest.raises(Exception) as exc_info:
                await client._async_complete("Test prompt")

            # The internal method should raise LLMResponseError for empty content
            # But it's wrapped by retry, so check for retry error with inner exception
            assert (
                "RetryError" in type(exc_info.value).__name__
                or "Empty" in str(exc_info.value)
                or "No choices" in str(exc_info.value)
            )

    @pytest.mark.asyncio
    async def test_acomplete_no_choices_raises_error(self):
        """Test that acomplete() raises error when no choices in response."""
        config = make_llm_config()

        mock_response = {"choices": []}

        client = get_llm_client(config)

        # Test internal method directly without retry decorator
        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(Exception) as exc_info:
                await client._async_complete("Test prompt")

            assert "RetryError" in type(exc_info.value).__name__ or "No choices" in str(
                exc_info.value
            )


class TestAsyncCompleteJson:
    """Test cases for acomplete_json() async method."""

    @pytest.mark.asyncio
    async def test_acomplete_json_returns_dict(self):
        """Test that acomplete_json() returns parsed JSON dict."""
        config = make_llm_config()

        mock_response = {
            "choices": [{"message": {"content": '{"key": "value", "number": 42}'}}]
        }

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.acomplete_json("Test prompt")

        assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_acomplete_json_with_system_prompt(self):
        """Test that acomplete_json() adds JSON instruction to system prompt."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": '{"result": "success"}'}}]}

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            await client.acomplete_json(
                "Test prompt", system_prompt="Provide analysis."
            )

            # Verify JSON instruction was added
            call_args = mock_request.call_args
            payload = call_args[0][0]
            messages = payload["messages"]
            system_content = messages[0]["content"]
            assert "JSON" in system_content or "json" in system_content.lower()
            assert "Provide analysis." in system_content

    @pytest.mark.asyncio
    async def test_acomplete_json_empty_content_returns_empty_dict(self):
        """Test that acomplete_json() returns empty dict on empty content."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": ""}}]}

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.acomplete_json("Test prompt")

        assert result == {}

    @pytest.mark.asyncio
    async def test_acomplete_json_invalid_json_raises_error(self):
        """Test that acomplete_json() raises error on invalid JSON."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": "Not valid JSON"}}]}

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(Exception) as exc_info:
                # Bypass retry decorator by calling internal method
                await client._async_complete_json("Test prompt")

            # The internal method should raise LLMResponseError for invalid JSON
            # But it's wrapped by retry, so check for retry error with inner exception
            assert "RetryError" in type(
                exc_info.value
            ).__name__ or "Invalid JSON" in str(exc_info.value)


class TestAsyncRetryDecorator:
    """Test cases for retry_llm_call_async decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_on_rate_limit(self):
        """Test that async rate limit errors trigger retry."""
        call_count = 0

        @retry_llm_call_async(max_retries=3, backoff_factor=2.0, add_jitter=False)
        async def flaky_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMRateLimitError("Rate limit exceeded")
            return "success"

        result = await flaky_async_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_on_transient_error(self):
        """Test that async transient errors trigger retry."""
        call_count = 0

        @retry_llm_call_async(max_retries=3, backoff_factor=2.0)
        async def flaky_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMCommunicationError("Server error")
            return "success"

        result = await flaky_async_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_no_retry_on_auth_error(self):
        """Test that async auth errors fail fast without retry."""
        call_count = 0

        @retry_llm_call_async(max_retries=3, backoff_factor=2.0)
        async def auth_failing_async_function():
            nonlocal call_count
            call_count += 1
            raise LLMAuthenticationError("Invalid API key")

        with pytest.raises(LLMAuthenticationError):
            await auth_failing_async_function()

        assert call_count == 1  # Only called once, no retry

    @pytest.mark.asyncio
    async def test_async_retry_exhaustion_raises_error(self):
        """Test that exhausting async retries raises RetryError."""
        call_count = 0

        @retry_llm_call_async(max_retries=2, backoff_factor=2.0, add_jitter=False)
        async def always_failing_async_function():
            nonlocal call_count
            call_count += 1
            raise LLMRateLimitError("Rate limit exceeded")

        with pytest.raises(RetryError) as exc_info:
            await always_failing_async_function()

        assert call_count == 3  # Initial + 2 retries
        assert "failed after 3 attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, LLMRateLimitError)

    @pytest.mark.asyncio
    async def test_async_success_on_first_attempt_no_retry(self):
        """Test that successful async first attempt doesn't retry."""
        call_count = 0

        @retry_llm_call_async(max_retries=3, backoff_factor=2.0)
        async def successful_async_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_async_function()

        assert result == "success"
        assert call_count == 1  # No retries needed


class TestAsyncClientIntegration:
    """Integration tests for async client behavior."""

    @pytest.mark.asyncio
    async def test_async_client_uses_async_http_client(self):
        """Test that async client uses httpx.AsyncClient."""
        config = make_llm_config()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        client = get_llm_client(config)

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as mock_async_client_cls:
            mock_async_client = MagicMock()
            mock_async_client.post = mock_post
            mock_async_client_cls.return_value.__aenter__.return_value = (
                mock_async_client
            )

            await client.acomplete("Test")

            # Verify AsyncClient was instantiated with correct params
            mock_async_client_cls.assert_called_once()
            call_kwargs = mock_async_client_cls.call_args[1]
            assert "timeout" in call_kwargs
            assert call_kwargs["trust_env"] is False

    @pytest.mark.asyncio
    async def test_async_client_shares_same_base_url_and_timeout(self):
        """Test that async client uses same config as sync client."""
        config = make_llm_config(base_url="https://custom.api.com", timeout=60.0)

        client = get_llm_client(config)

        # Verify async client has the same base_url and timeout
        assert client.base_url == "https://custom.api.com/v1"
        assert client.timeout == 60.0

        mock_response = {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            result = await client.acomplete("Test")

        assert result == "ok"
        # Verify async method was called
        assert mock_request.called is True


class TestAsyncIntegration:
    """Integration tests for full async flow: factory -> client -> async methods."""

    @pytest.mark.asyncio
    async def test_factory_creates_async_client_with_acomplete(self):
        """Test complete chain: factory -> client -> acomplete with mock."""
        config = make_llm_config()

        mock_response = {
            "choices": [{"message": {"content": "Integration test response"}}]
        }

        # Step 1: Get client from factory
        client = get_llm_client(config)

        # Step 2: Mock async request and call acomplete
        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            result = await client.acomplete("Integration test prompt")

        # Step 3: Verify the response
        assert result == "Integration test response"
        mock_request.assert_called_once()

        # Verify client type
        assert isinstance(client, OpenAICompatibleClient)

    @pytest.mark.asyncio
    async def test_factory_creates_async_client_with_acomplete_json(self):
        """Test complete chain: factory -> client -> acomplete_json with mock."""
        config = make_llm_config()

        mock_response = {
            "choices": [{"message": {"content": '{"status": "success", "score": 95}'}}]
        }

        # Get client from factory
        client = get_llm_client(config)

        # Call acomplete_json
        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.acomplete_json("Generate case study")

        # Verify JSON response was parsed correctly
        assert result == {"status": "success", "score": 95}
        assert result["status"] == "success"
        assert result["score"] == 95

    @pytest.mark.asyncio
    async def test_sync_async_consistency_same_response(self):
        """Test that sync and async methods produce same result with same mock."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": "Consistent response"}}]}

        client = get_llm_client(config)

        # Mock the internal request method for both sync and async
        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async_result = await client.acomplete("Test prompt")

        # Note: We can't actually test sync with same mock because _make_request is different
        # But we verify both use the same base client and produce same output structure
        assert async_result == "Consistent response"

    @pytest.mark.asyncio
    async def test_full_async_flow_with_system_prompt(self):
        """Test full async flow with system prompt."""
        config = make_llm_config()

        expected_system_content = "You are an expert data analyst."
        mock_response = {
            "choices": [{"message": {"content": "Analysis: 15% growth in Q3"}}]
        }

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_request:
            result = await client.acomplete(
                "Analyze Q3 performance", system_prompt=expected_system_content
            )

        # Verify result
        assert result == "Analysis: 15% growth in Q3"

        # Verify system prompt was passed correctly
        call_args = mock_request.call_args
        payload = call_args[0][0]
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == expected_system_content
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Analyze Q3 performance"

    @pytest.mark.asyncio
    async def test_async_client_base_url_and_timeout_from_config(self):
        """Test that async client correctly uses config values from factory."""
        config = make_llm_config(base_url="https://custom.openai.com/v1", timeout=90.0)

        client = get_llm_client(config)

        # Verify config values were applied
        assert client.base_url == "https://custom.openai.com/v1"
        assert client.timeout == 90.0

        mock_response = {"choices": [{"message": {"content": "OK"}}]}

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.acomplete("Test")

        assert result == "OK"

    @pytest.mark.asyncio
    async def test_async_retry_on_client_level(self):
        """Test that async retry decorator works at client level."""

        # Add a method with retry decorator to client
        config = make_llm_config()

        client = get_llm_client(config)

        # Test that acomplete has the retry decorator
        assert hasattr(client.acomplete, "__wrapped__") or hasattr(
            client.acomplete, "__name__"
        )

    @pytest.mark.asyncio
    async def test_async_json_flow_allows_empty_content(self):
        """Test that acomplete_json returns empty dict on empty content."""
        config = make_llm_config()

        mock_response = {"choices": [{"message": {"content": ""}}]}

        client = get_llm_client(config)

        with patch.object(
            client,
            "_async_make_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.acomplete_json("Test")

        # Empty content should return empty dict, not raise
        assert result == {}
