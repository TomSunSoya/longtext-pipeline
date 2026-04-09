"""Tests for the LLM factory module."""

import os
import pytest
import time
from unittest.mock import patch, MagicMock
from unittest.mock import call

from src.longtext_pipeline.llm.factory import get_llm_client
from src.longtext_pipeline.llm.base import LLMClient
from src.longtext_pipeline.llm.openai_compatible import OpenAICompatibleClient
from src.longtext_pipeline.utils.retry import retry_llm_call, RetryError
from src.longtext_pipeline.errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMCommunicationError,
)


def make_llm_config(
    provider: str = "openai",
    name: str | None = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
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
    return {"model": model}


class TestGetLLMClient:
    """Test cases for get_llm_client factory function."""
    
    def test_factory_returns_openai_compatible_client(self):
        """Test that factory returns OpenAICompatibleClient instance."""
        config = make_llm_config(api_key="test-key")
        
        client = get_llm_client(config)
        
        assert isinstance(client, OpenAICompatibleClient)
        assert isinstance(client, LLMClient)
    
    def test_factory_passes_config_values_correctly(self):
        """Test that config values are passed to client correctly."""
        config = make_llm_config(
            api_key="test-api-key",
            base_url="https://custom.api.com",
            timeout=60.0,
        )
        
        with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
            try:
                get_llm_client(config)
            except TypeError:
                # Expected because we're mocking __init__
                pass
            
            mock_init.assert_called_once_with(
                model="gpt-4o-mini",
                api_key="test-api-key",
                base_url="https://custom.api.com",
                timeout=60.0,
            )

    def test_factory_accepts_name_key_for_model(self):
        """Test that model configs using `name` are supported."""
        config = make_llm_config(name="deepseek-chat", api_key="test-api-key")

        with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
            try:
                get_llm_client(config)
            except TypeError:
                pass

            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["model"] == "deepseek-chat"
    
    def test_factory_uses_default_timeout_when_not_specified(self):
        """Test that default timeout is used when not specified."""
        config = make_llm_config(api_key="test-key")
        
        with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
            try:
                get_llm_client(config)
            except TypeError:
                pass
            
            # Should use the long-text friendly default timeout of 120.0
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["timeout"] == 120.0
    
    def test_env_var_fallback_for_api_key(self):
        """Test that environment variable is used when api_key not in config."""
        config = make_llm_config(api_key=None)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}):
            with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
                try:
                    get_llm_client(config)
                except TypeError:
                    pass
                
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs["api_key"] == "env-api-key"
    
    def test_env_var_fallback_for_base_url(self):
        """Test that environment variable is used when base_url not in config."""
        config = make_llm_config(api_key="test-key")
        
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.api.com"}):
            with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
                try:
                    get_llm_client(config)
                except TypeError:
                    pass
                
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs["base_url"] == "https://env.api.com"
    
    def test_env_var_fallback_for_model_name(self):
        """Test that LONGTEXT_MODEL_NAME env var is used when model not in config."""
        config = make_llm_config(name=None, api_key="test-key")
        
        with patch.dict(os.environ, {"LONGTEXT_MODEL_NAME": "env-model"}):
            with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
                try:
                    get_llm_client(config)
                except TypeError:
                    pass
                
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs["model"] == "env-model"
    
    def test_explicit_args_override_config(self):
        """Test that explicit function arguments override config values."""
        config = make_llm_config(
            name="config-model",
            api_key="config-key",
            base_url="https://config.api.com",
            timeout=30.0,
        )
        
        with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
            try:
                get_llm_client(
                    config,
                    model="override-model",
                    api_key="override-key",
                    base_url="https://override.api.com",
                    timeout=120.0,
                )
            except TypeError:
                pass
            
            mock_init.assert_called_once_with(
                model="override-model",
                api_key="override-key",
                base_url="https://override.api.com",
                timeout=120.0,
            )
    
    def test_explicit_args_override_env_vars(self):
        """Test that explicit arguments override environment variables."""
        config = make_llm_config(name=None, api_key="config-key")
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "env-key",
            "OPENAI_BASE_URL": "https://env.api.com",
            "LONGTEXT_MODEL_NAME": "env-model",
        }):
            with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
                try:
                    get_llm_client(
                        config,
                        model="override-model",
                        api_key="override-key",
                    )
                except TypeError:
                    pass
                
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs["api_key"] == "override-key"
                assert call_kwargs["model"] == "override-model"
    
    def test_config_values_override_env_vars(self):
        """Test that config values override environment variables."""
        config = make_llm_config(name="config-model", api_key="config-key")
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "env-key",
            "LONGTEXT_MODEL_NAME": "env-model",
        }):
            with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
                try:
                    get_llm_client(config)
                except TypeError:
                    pass
                
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs["api_key"] == "config-key"
                assert call_kwargs["model"] == "config-model"
    
    def test_invalid_config_type_raises_type_error(self):
        """Test that non-dict config raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            get_llm_client("not-a-dict")
        
        assert "config must be a dictionary" in str(exc_info.value)
    
    def test_none_config_raises_type_error(self):
        """Test that None config raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            get_llm_client(None)
        
        assert "config must be a dictionary" in str(exc_info.value)
    
    def test_unsupported_provider_raises_value_error(self):
        """Test that unsupported provider raises ValueError."""
        config = make_llm_config(provider="anthropic", name=None, api_key="test-key")
        
        with pytest.raises(ValueError) as exc_info:
            get_llm_client(config)
        
        assert "Unsupported LLM provider: 'anthropic'" in str(exc_info.value)
    
    def test_openai_provider_creates_client(self):
        """Test that 'openai' provider creates OpenAICompatibleClient."""
        config = make_llm_config(name=None, api_key="test-key")
        
        client = get_llm_client(config)
        
        assert isinstance(client, OpenAICompatibleClient)
    
    def test_openrouter_provider_creates_client(self):
        """Test that 'openrouter' provider creates OpenAICompatibleClient."""
        config = make_llm_config(provider="openrouter", name=None, api_key="test-key")
        
        client = get_llm_client(config)
        
        assert isinstance(client, OpenAICompatibleClient)
    
    def test_ollama_provider_creates_client(self):
        """Test that 'ollama' provider creates OpenAICompatibleClient."""
        config = make_llm_config(provider="ollama", name=None, api_key="test-key")

        client = get_llm_client(config)

        assert isinstance(client, OpenAICompatibleClient)


class TestOpenAICompatibleClient:
    """Targeted regression coverage for the raw OpenAI-compatible client."""

    def test_make_request_disables_env_proxy_resolution(self):
        """httpx should ignore ambient proxy env vars unless configured explicitly."""
        client = OpenAICompatibleClient(
            model="gpt-4o-mini",
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        mock_httpx_client = MagicMock()
        mock_httpx_client.post.return_value = mock_response

        with patch("src.longtext_pipeline.llm.openai_compatible.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__.return_value = mock_httpx_client

            result = client._make_request({"messages": []})

        assert result == {"choices": [{"message": {"content": "ok"}}]}
        mock_client_cls.assert_called_once_with(timeout=client.timeout, trust_env=False)
    
    def test_vllm_provider_creates_client(self):
        """Test that 'vllm' provider creates OpenAICompatibleClient."""
        config = make_llm_config(provider="vllm", name=None, api_key="test-key")
        
        client = get_llm_client(config)
        
        assert isinstance(client, OpenAICompatibleClient)
    
    def test_provider_case_insensitive(self):
        """Test that provider name is case-insensitive."""
        config = make_llm_config(provider="OPENAI", name=None, api_key="test-key")
        
        client = get_llm_client(config)
        
        assert isinstance(client, OpenAICompatibleClient)
    
    def test_default_provider_is_openai(self):
        """Test that missing provider defaults to 'openai'."""
        config = {"model": {"api_key": "test-key"}}
        
        with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
            try:
                get_llm_client(config)
            except TypeError:
                pass
            
            # Should default to gpt-4o-mini when no model specified
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o-mini"
    
    def test_empty_config_uses_defaults_and_env(self):
        """Test that empty config uses defaults and environment variables."""
        config = {}
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            with patch.object(OpenAICompatibleClient, '__init__', return_value=None) as mock_init:
                try:
                    get_llm_client(config)
                except TypeError:
                    pass
                
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs["api_key"] == "env-key"
                assert call_kwargs["model"] == "gpt-4o-mini"
                assert call_kwargs["timeout"] == 120.0
    
    def test_client_can_be_used_for_completion(self):
        """Test that created client has required methods (integration test)."""
        config = make_llm_config(api_key="test-key-for-validation")
        
        client = get_llm_client(config)
        
        # Verify client has the required abstract methods
        assert hasattr(client, "complete")
        assert hasattr(client, "complete_json")
        assert callable(client.complete)
        assert callable(client.complete_json)


class TestRetryLLMCall:
    """Test cases for retry_llm_call decorator."""
    
    def test_retry_on_rate_limit_with_backoff(self):
        """Test that rate limit errors (429) trigger exponential backoff retry."""
        call_count = 0
        
        @retry_llm_call(max_retries=3, backoff_factor=2.0, add_jitter=False)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMRateLimitError("Rate limit exceeded")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third attempt
    
    def test_retry_on_transient_error(self):
        """Test that transient errors (500) trigger retry."""
        call_count = 0
        
        @retry_llm_call(max_retries=3, backoff_factor=2.0)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMCommunicationError("Server error")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 2
    
    def test_no_retry_on_auth_error(self):
        """Test that auth errors (401) fail fast without retry."""
        call_count = 0
        
        @retry_llm_call(max_retries=3, backoff_factor=2.0)
        def auth_failing_function():
            nonlocal call_count
            call_count += 1
            raise LLMAuthenticationError("Invalid API key")
        
        with pytest.raises(LLMAuthenticationError):
            auth_failing_function()
        
        assert call_count == 1  # Only called once, no retry
    
    def test_retry_exhaustion_raises_retry_error(self):
        """Test that exhausting all retries raises RetryError."""
        call_count = 0
        
        @retry_llm_call(max_retries=2, backoff_factor=2.0, add_jitter=False)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise LLMRateLimitError("Rate limit exceeded")
        
        with pytest.raises(RetryError) as exc_info:
            always_failing_function()
        
        assert call_count == 3  # Initial + 2 retries
        assert "failed after 3 attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, LLMRateLimitError)
    
    def test_retry_with_jitter(self):
        """Test that jitter adds randomness to delay."""
        call_count = 0
        delays = []
        
        @retry_llm_call(max_retries=2, backoff_factor=2.0, add_jitter=True, initial_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMRateLimitError("Rate limit exceeded")
            return "success"
        
        start = time.time()
        result = flaky_function()
        elapsed = time.time() - start
        
        assert result == "success"
        assert call_count == 3
        # With jitter, delays should vary (at least some delay occurred)
        assert elapsed > 0.01  # At least some delay occurred
    
    def test_success_on_first_attempt_no_retry(self):
        """Test that successful first attempt doesn't retry."""
        call_count = 0
        
        @retry_llm_call(max_retries=3, backoff_factor=2.0)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count == 1  # No retries needed
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        call_count = 0
        
        @retry_llm_call(max_retries=5, backoff_factor=2.0, max_delay=0.05, add_jitter=False)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise LLMRateLimitError("Rate limit exceeded")
            return "success"
        
        start = time.time()
        result = flaky_function()
        elapsed = time.time() - start
        
        assert result == "success"
        # With max_delay=0.05 and 4 retries, should complete in reasonable time
        # (without cap, exponential backoff would be 1+2+4+8=15 seconds)
        assert elapsed < 1.0  # Should complete quickly due to max_delay cap
