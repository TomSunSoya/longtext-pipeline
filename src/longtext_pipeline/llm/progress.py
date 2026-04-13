"""Progress utilities for LLM streaming functionality."""

import time
import sys
from typing import Callable


def create_token_progress_callback() -> Callable[[str, int, float], None]:
    """Create a callback function that displays token progression during streaming.

    Returns:
        A callback function that shows current token count and estimated tokens per second
    """
    start_time = time.time()

    def callback(token: str, tokens_so_far: int, elapsed: float) -> None:
        """Progress callback that shows token count in human-readable format.

        Args:
            token: Current token/chunk from the LLM
            tokens_so_far: Total number of tokens received so far
            elapsed: Elapsed time since streaming started
        """
        # Format the count with commas like "1,234 tokens..."
        formatted_count = f"{tokens_so_far:,}".replace(",", ",")
        progress_text = f"\rProcessing: {formatted_count} tokens... "

        # Add estimated speed if we have reasonable timing
        if elapsed > 0.1:
            tokens_per_sec = tokens_so_far / elapsed
            progress_text += f"({tokens_per_sec:.1f} tokens/sec)"

        sys.stdout.write(progress_text)
        sys.stdout.flush()

    # Store the start time on the function
    callback.start_time = start_time

    def progress_callback(token: str, tokens_so_far: int, elapsed: float) -> None:
        # Format the count with commas like "1,234 tokens..."
        formatted_count = f"{tokens_so_far:,}"
        progress_text = f"\rProcessing: {formatted_count} tokens... "

        # Add estimated speed if we have reasonable timing
        if elapsed > 0.1:
            tokens_per_sec = tokens_so_far / elapsed
            progress_text += f"({tokens_per_sec:.1f} tokens/sec)"

        sys.stdout.write(progress_text)
        sys.stdout.flush()

    return progress_callback


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
    # Format the count with commas like "1,234 tokens..."
    formatted_count = f"{tokens_so_far:,}"
    progress_text = f"\rProcessing: {formatted_count} tokens... "

    # Add estimated speed if we have reasonable timing
    if elapsed > 0.1:
        tokens_per_sec = tokens_so_far / elapsed
        progress_text += f"({tokens_per_sec:.1f} tokens/sec)"

    sys.stdout.write(progress_text)
    sys.stdout.flush()
