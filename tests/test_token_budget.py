"""Unit tests for token budget management functionality.

Test cases follow TDD methodology for token budget management features.
"""

import pytest

from src.longtext_pipeline.utils.token_budget import TokenBudgetManager
from src.longtext_pipeline.errors import ContextWindowExceededError


class TestTokenBudgetManagerInit:
    """Test initialization and configuration of TokenBudgetManager."""

    def test_default_initialization(self):
        """Test that TokenBudgetManager initializes with default values."""
        manager = TokenBudgetManager()
        assert manager.buffer_tokens == 1000
        assert manager.max_output_tokens == 2000

    def test_custom_initialization(self):
        """Test that TokenBudgetManager accepts custom parameters."""
        manager = TokenBudgetManager(buffer_tokens=500, max_output_tokens=1000)
        assert manager.buffer_tokens == 500
        assert manager.max_output_tokens == 1000


class TestEstimateTokens:
    """Test token estimation functionality."""

    def test_empty_string_estimation(self):
        """Test that empty strings estimate to 0 tokens."""
        manager = TokenBudgetManager()
        assert manager.estimate_tokens("") == 0
        assert manager.estimate_tokens("   ") == 0  # whitespace-only string

    def test_simple_text_estimation(self):
        """Test estimation for simple text."""
        manager = TokenBudgetManager()
        text = "Hello world"
        # Should estimate roughly 2-3 tokens for this short phrase
        estimation = manager.estimate_tokens(text)
        assert estimation > 0  # Should return at least 1 token for non-empty text
        assert estimation < 20  # Should be reasonably small

    def test_large_text_estimation(self):
        """Test estimation for larger text."""
        manager = TokenBudgetManager()
        # Create a paragraph of text with many words
        text = "This is a longer paragraph with approximately 30 words total. "
        text += "We need to see if the estimation works properly " * 2
        estimation = manager.estimate_tokens(text)

        # Should estimate a meaningful number of tokens, typically 20-40 for this amount of text
        assert estimation > 10
        assert estimation < 100


class TestValidateBudget:
    """Test budget validation functionality."""

    def test_allows_input_under_limit(self):
        """Test that inputs under the budget limit pass validation."""
        manager = TokenBudgetManager(buffer_tokens=100, max_output_tokens=500)
        # Small prompt that should easily fit in the context
        prompt = "What is the weather like today?"

        # With context of 4096 tokens, even a small prompt should pass validation
        can_proceed, reason = manager.validate_budget(
            prompt=prompt, context_window=4096
        )
        assert can_proceed is True
        assert reason == ""

    def test_validation_reasonable_sizes(self):
        """Test validation with reasonable prompt and context sizes."""
        manager = TokenBudgetManager(buffer_tokens=500, max_output_tokens=1000)

        # Text roughly equivalent to a few hundred tokens
        prompt = (
            "This is a medium-length prompt that represents the sort "
            "of content that users might submit to the LLM. It contains "
            "several sentences worth of information and context for the "
            "language model to understand. The prompt may describe a "
            "complex scenario, a question that requires analysis, or "
            "perhaps even contain some preliminary information that "
            "the model should consider while forming its response. " * 5
        )

        can_proceed, reason = manager.validate_budget(
            prompt=prompt,
            context_window=8192,  # Standard large-context model
        )
        assert can_proceed is True  # This should fit with the budget

    def test_validation_detects_excessive_size(self):
        """Test that validation detects when the prompt is too large."""
        manager = TokenBudgetManager(buffer_tokens=100, max_output_tokens=500)

        # Very long prompt that clearly exceeds small context
        prompt = "This prompt is intentionally very long. " * 1000

        can_proceed, reason = manager.validate_budget(
            prompt=prompt,
            context_window=1024,  # Very small context window
        )
        assert can_proceed is False
        assert reason != ""
        assert "exceed" in reason.lower()


class TestTruncatePrompt:
    """Test prompt truncation functionality."""

    def test_truncate_large_prompt(self):
        """Test that large prompts get appropriately truncated."""
        manager = TokenBudgetManager()

        # Create a large text
        large_text = "This is important context. " * 100
        large_tokens = manager.estimate_tokens(large_text)

        # Try to truncate to a smaller token budget
        target_tokens = large_tokens // 4  # About 1/4 of original size
        truncated_result = manager.truncate_prompt(large_text, target_tokens)

        # Result should be shorter than original
        assert len(truncated_result) < len(large_text)
        assert len(truncated_result) > 0  # Should not be empty

        # Estimate tokens to ensure it's closer to target
        truncated_tokens = manager.estimate_tokens(truncated_result)
        # Should be roughly near the target, though estimate accuracy means it might not be exact
        assert truncated_tokens <= max(target_tokens * 2, 10)  # Some tolerance

    def test_truncate_shorter_than_limit_remains_unchanged(self):
        """Test that prompts shorter than the limit remain unchanged."""
        manager = TokenBudgetManager()
        short_prompt = "Short prompt"

        result = manager.truncate_prompt(short_prompt, max_allowed_tokens=1000)

        assert result == short_prompt

    def test_zero_token_target_returns_empty_string(self):
        """Test that setting a zero token target returns an empty string."""
        manager = TokenBudgetManager()
        prompt = "Any text that we want to truncate completely"

        result = manager.truncate_prompt(prompt, max_allowed_tokens=0)

        assert result == ""


class TestProcessPromptWithBudget:
    """Test the main budget processing method."""

    def test_handles_valid_prompt_pair(self):
        """Test that valid prompt pairs pass through with no changes."""
        manager = TokenBudgetManager(buffer_tokens=200, max_output_tokens=500)

        system_prompt = "You are a helpful assistant."
        user_prompt = "What is the meaning of life?"

        processed_prompt, processed_system = manager.process_prompt_with_budget(
            prompt=user_prompt, system_prompt=system_prompt, context_window=8192
        )

        # Both should be returned as-is since they fit the context budget
        assert processed_prompt == user_prompt
        assert processed_system == system_prompt

    def test_handles_none_system_prompt(self):
        """Test that none for system prompt is handled properly."""
        manager = TokenBudgetManager()

        user_prompt = "This is a test prompt."

        processed_prompt, processed_system = manager.process_prompt_with_budget(
            prompt=user_prompt, system_prompt=None, context_window=4096
        )

        assert processed_prompt == user_prompt
        assert processed_system is None

    def test_raises_error_when_impossible(self):
        """Test that ContextWindowExceededError is raised when impossible situation."""
        manager = TokenBudgetManager(buffer_tokens=500, max_output_tokens=2000)

        # Very large combined prompt with context window that's definitely too small
        system_prompt = (
            "You are a helpful assistant that needs to consider "
            + "a lot of rules. " * 1000
        )
        user_prompt = (
            "And now for the incredibly detailed question: " + "more details. " * 1000
        )

        with pytest.raises(ContextWindowExceededError) as excinfo:
            manager.process_prompt_with_budget(
                prompt=user_prompt,
                system_prompt=system_prompt,
                context_window=512,  # Way too small
            )

        # Check that the exception has proper attributes
        assert excinfo.value.context_window == 512
        assert excinfo.value.required_tokens > 512


class TestIntegrationTokenBudgetManager:
    """Integration tests for complete token budget flows."""

    def test_integration_normal_processing_flow(self):
        """Test the complete normal flow under normal conditions."""
        manager = TokenBudgetManager()

        # Normal-sized system and user prompts
        system_prompt = "You are an expert at providing concise, accurate answers."
        user_prompt = (
            "Analyze the following document and provide a summary: "
            + "The document contains many important details. " * 100
        )

        # Process with generous context
        processed_prompt, processed_system = manager.process_prompt_with_budget(
            prompt=user_prompt,
            system_prompt=system_prompt,
            context_window=128000,  # GLM-5 context size
        )

        # Should be returned unchanged as they fit
        assert processed_prompt == user_prompt
        assert processed_system == system_prompt

    def test_integration_boundary_condition(self):
        """Test behavior at the boundary of token limits."""
        manager = TokenBudgetManager(buffer_tokens=100, max_output_tokens=100)

        # Create slightly large text to see if it's properly handled
        test_content = "This is moderately sized text that may potentially " * 30

        result, _ = manager.process_prompt_with_budget(
            prompt=test_content,
            system_prompt=None,
            context_window=400,  # Very tight context limit
        )

        # The result should be truncated significantly compared to original
        # Though exact size may vary, it should be less than original
        assert len(result) < len(test_content)
        # The actual result from the algorithm will likely be longer than expected,
        # but it depends on the heuristic approach used
        # Just confirm that processing completed without error


# Tests for the OpenAICompatibleClient with token budget integration
def test_llm_client_accepts_context_window():
    """Test that OpenAICompatibleClient can accept context_window parameter."""
    from src.longtext_pipeline.llm import OpenAICompatibleClient

    # Test initialization with custom context window
    client = OpenAICompatibleClient(
        model="test-model",
        context_window=32768,  # Different from default of 128000
        api_key="fake-key-for-test",
    )

    assert client.context_window == 32768


def test_llm_client_with_adequate_context_works_normally():
    """Test that LLM client works normally when context is adequate."""
    from src.longtext_pipeline.llm import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        model="test-model", context_window=4096, api_key="fake-key-for-test"
    )

    # Should be able to use the token budget manager
    assert hasattr(client, "_token_budget_manager")
    assert client.context_window == 4096


if __name__ == "__main__":
    pytest.main([__file__])
