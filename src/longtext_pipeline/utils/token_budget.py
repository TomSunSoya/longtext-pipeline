"""Token Budget Manager for preventing exceedance of model context windows.

This utility manages token budgets before LLM calls, ensuring efficient usage
while preventing context window overruns.
"""

import warnings
from typing import Optional, Tuple
from .token_estimator import estimate_tokens
from ..errors import ContextWindowExceededError


class TokenBudgetManager:
    """Manages token budgets to prevent context window overruns."""

    def __init__(self, buffer_tokens: int = 1000, max_output_tokens: int = 2000):
        """
        Initialize the TokenBudgetManager.

        Args:
            buffer_tokens: Number of tokens to reserve for additional prompt overhead
            max_output_tokens: Expected maximum output from the model
        """
        self.buffer_tokens = buffer_tokens
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        return estimate_tokens(text)

    def validate_budget(
        self, prompt: str, context_window: int, max_output_tokens: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Validate that a prompt fits within the token budget.

        Args:
            prompt: Text of the prompt to validate
            context_window: Total available tokens in the model's context
            max_output_tokens: Expected output tokens (uses default if None)

        Returns:
            Tuple of (can_proceed, reason_for_failure_or_empty_string)
        """
        if max_output_tokens is None:
            max_output_tokens = self.max_output_tokens

        prompt_tokens = self.estimate_tokens(prompt)

        # Calculate required tokens: prompt + output + buffer
        required_tokens = prompt_tokens + max_output_tokens + self.buffer_tokens

        if required_tokens > context_window:
            return (
                False,
                f"Prompt uses {prompt_tokens} tokens, leaving only "
                f"{context_window - prompt_tokens} tokens for model output. "
                f"Expected {max_output_tokens} + {self.buffer_tokens} buffer = "
                f"{max_output_tokens + self.buffer_tokens} tokens reserved for model, "
                f"but total required ({required_tokens}) exceeds context window ({context_window}).",
            )

        return True, ""

    def truncate_prompt(
        self, prompt: str, max_allowed_tokens: int, truncate_to_percentage: float = 0.9
    ) -> str:
        """
        Truncate a prompt to fit within a specific token count.

        Args:
            prompt: Original prompt to truncate
            max_allowed_tokens: Maximum tokens allowed for the prompt
            truncate_to_percentage: Percentage of available tokens to use (0.0 to 1.0)

        Returns:
            Truncated prompt that fits within the token budget
        """
        # Calculate target tokens to use
        target_tokens = int(max_allowed_tokens * truncate_to_percentage)

        if target_tokens <= 0:
            return ""

        # Simple character-based truncation (heuristic-based)
        # We know that ~4 chars ≈ 1 token on average
        estimated_char_ratio = 4.0
        max_chars = int(target_tokens * estimated_char_ratio)

        if len(prompt) <= max_chars:
            return prompt

        # Perform intelligent truncation
        # First try to split at sentence boundaries (try periods, newlines)
        truncated = prompt[:max_chars]

        # Look for the last period/newline within the allowed range
        # to avoid cutting mid-sentence
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        best_split_point = max(last_period, last_newline)

        # Ensure the split point is reasonable (close enough to our desired length)
        if best_split_point > max_chars * 0.7:  # If we can cut within the last 30%
            truncated = truncated[: best_split_point + 1]  # Include the period/newline

        return truncated

    def process_prompt_with_budget(
        self,
        prompt: str,
        system_prompt: str,
        context_window: int,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Process a prompt pair with respect to token budget.

        Args:
            prompt: Main prompt content
            system_prompt: System prompt content (can be None)
            context_window: Total context window of the model
            max_output_tokens: Expected output tokens (uses default if None)

        Returns:
            Tuple of (processed_prompt, processed_system_prompt)

        Raises:
            ContextWindowExceededError: If neither prompt nor system_prompt can fit
        """
        if max_output_tokens is None:
            max_output_tokens = self.max_output_tokens

        # Combine prompts to validate against combined length
        combined_prompt = (system_prompt or "") + "\n" + prompt
        prompt_tokens = self.estimate_tokens(combined_prompt)

        # Reserve tokens for potential overhead
        total_reserved_for_output = max_output_tokens + self.buffer_tokens
        available_prompt_tokens = context_window - total_reserved_for_output

        if available_prompt_tokens <= 0:
            raise ContextWindowExceededError(
                context_window=context_window, required_tokens=prompt_tokens
            )

        # If the combined content fits, return as-is
        if prompt_tokens <= available_prompt_tokens:
            return prompt, system_prompt

        # Otherwise, warn about truncation possibility
        warnings.warn(
            f"Combined prompt tokens ({prompt_tokens}) exceed available context "
            f"for output ({available_prompt_tokens} tokens available for input). "
            f"This may lead to model truncation.",
            UserWarning,
        )

        # If system prompt is too long, we might consider truncating it
        if system_prompt:
            system_tokens = self.estimate_tokens(system_prompt)
            estimated_remaining_for_user = available_prompt_tokens - system_tokens

            # If even with shortened system prompt there's not much room
            if estimated_remaining_for_user < 500:  # minimum room for user prompt
                # Truncate system prompt to make more room
                max_system_tokens = available_prompt_tokens // 2
                new_system_prompt = self.truncate_prompt(
                    system_prompt, max_system_tokens
                )

                # Recalculate available space for user prompt
                new_system_tokens = self.estimate_tokens(new_system_prompt)
                remaining_tokens = available_prompt_tokens - new_system_tokens

                # If we still don't have enough room, truncate the user prompt too
                if remaining_tokens > 500:  # only bother if there's significant room
                    new_prompt = self.truncate_prompt(prompt, remaining_tokens)
                else:
                    # The situation is hopeless, raise an exception
                    raise ContextWindowExceededError(
                        context_window=context_window, required_tokens=prompt_tokens
                    )

                return new_prompt, new_system_prompt

            # If system prompt is okay but combined is too large, focus on main prompt
            remaining_tokens = available_prompt_tokens - system_tokens
            new_prompt = self.truncate_prompt(prompt, remaining_tokens)
            return new_prompt, system_prompt
        else:
            # No system prompt, so just truncate the main prompt
            new_prompt = self.truncate_prompt(prompt, available_prompt_tokens)
            return new_prompt, system_prompt
