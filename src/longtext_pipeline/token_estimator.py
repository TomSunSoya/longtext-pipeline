"""Token estimation utilities for the longtext pipeline.

This module provides simple token counting approximations for MVP.
The estimation is based on the rule of thumb that ~3 characters ≈ 1 token
for English text.
"""


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple approximation:	total tokens ≈ length / 3
    This is reasonably accurate for English text with typical
    word lengths and doesn't require heavy dependencies.

    Args:
        text: The input text to estimate token count for

    Returns:
        Approximate token count
    """
    if not text:
        return 0

    # Simple approximation: ~3 characters per token
    return max(1, len(text) // 3)


def estimate_words_to_tokens(words: int) -> int:
    """Estimate tokens from word count.

    Args:
        words: Number of words

    Returns:
        Approximate token count
    """
    # Average word length ~5 chars + space = ~2 tokens per word
    return max(1, words * 2 // 3)
