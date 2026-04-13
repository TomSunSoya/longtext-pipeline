"""Progress utilities for LLM streaming functionality."""

import sys
from typing import Callable


def _format_progress_text(tokens_so_far: int, elapsed: float) -> str:
    """Build the progress line shown during streaming."""
    progress_text = f"\rProcessing: {tokens_so_far:,} tokens... "
    if elapsed > 0.1:
        tokens_per_sec = tokens_so_far / elapsed
        progress_text += f"({tokens_per_sec:.1f} tokens/sec)"
    return progress_text


def create_token_progress_callback() -> Callable[[str, int, float], None]:
    """Create a callback function that displays token progression during streaming."""
    return default_progress_callback


def print_final_streaming_stats(elapsed: float, tokens_count: int) -> None:
    """Print final statistics for streamed content.

    Args:
        elapsed: Total time elapsed for streaming
        tokens_count: Total number of tokens received
    """
    if elapsed > 0:
        avg_tokens_per_sec = tokens_count / elapsed if elapsed > 0 else 0
        print(
            f"\nCompleted in {elapsed:.2f}s at ~{avg_tokens_per_sec:.1f} tokens/sec ({tokens_count:,} tokens total)"
        )
    else:
        print(f"\nCompleted ({tokens_count:,} tokens)")


def default_progress_callback(token: str, tokens_so_far: int, elapsed: float) -> None:
    """Default callback that prints token count while streaming."""
    sys.stdout.write(_format_progress_text(tokens_so_far, elapsed))
    sys.stdout.flush()
