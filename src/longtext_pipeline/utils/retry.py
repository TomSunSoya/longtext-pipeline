"""
Retry utilities for handling transient failures.
Provides generic retry wrapper with exponential backoff.
"""

import asyncio
import random
import time
from functools import wraps
from typing import Callable, TypeVar, Optional

from ..errors import (
    PipelineError,
    LLMRateLimitError,
    LLMCommunicationError,
    LLMAuthenticationError,
)


def _get_metrics():
    """Lazy import of metrics to avoid registration issues during testing."""
    from .metrics import (
        retry_attempts_total,
        retry_delay_seconds,
        rate_limit_hits_total,
    )

    return retry_attempts_total, retry_delay_seconds, rate_limit_hits_total


class RetryError(PipelineError):
    """Raised when all retry attempts have failed."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


T = TypeVar("T")
CoroutineT = TypeVar("CoroutineT")


def retry_llm_call(
    func: Optional[Callable[..., T]] = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    add_jitter: bool = True,
) -> Callable[..., T]:
    """
    LLM-specific retry wrapper with exponential backoff and jitter.

    Handles different error types appropriately:
    - Rate limit (429): Exponential backoff with jitter
    - Transient errors (500): Linear retry with short delay
    - Auth errors (401): Fail fast, NO retry

    Usage:
        @retry_llm_call(max_retries=3, backoff_factor=2)
        def call_llm(prompt: str) -> str:
            return client.complete(prompt)

    Args:
        func: Function to wrap (typically LLM complete/complete_json methods)
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        add_jitter: Whether to add random jitter to prevent thundering herd (default: True)

    Returns:
        Wrapped function with retry logic

    Raises:
        LLMAuthenticationError: Raised immediately without retry on 401
        RetryError: If all retry attempts fail for retryable errors
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            delay = initial_delay
            retry_attempts, retry_delay, rate_limit_hits = _get_metrics()

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return f(*args, **kwargs)

                except LLMAuthenticationError:
                    # Auth errors (401) - fail fast, no retry
                    raise

                except (LLMRateLimitError, LLMCommunicationError) as e:
                    last_exception = e

                    # If we've exhausted retries, raise
                    if attempt >= max_retries:
                        raise RetryError(
                            f"LLM call failed after {max_retries + 1} attempts",
                            last_exception=last_exception,
                        )

                    # Record rate limit hit
                    if isinstance(e, LLMRateLimitError):
                        rate_limit_hits.labels(stage="unknown").inc()

                    # Record retry attempt
                    error_type = (
                        "rate_limit" if isinstance(e, LLMRateLimitError) else "other"
                    )
                    retry_attempts.labels(stage="unknown", error_type=error_type).inc()

                    # Calculate delay based on error type
                    if isinstance(e, LLMRateLimitError):
                        # Rate limit (429): exponential backoff
                        calculated_delay = delay * (backoff_factor**attempt)
                    else:
                        # Transient server errors (500): shorter linear retry
                        calculated_delay = delay * (attempt + 1)

                    # Apply jitter to prevent thundering herd
                    if add_jitter:
                        jitter = random.uniform(0, calculated_delay * 0.5)
                        calculated_delay += jitter

                    # Cap delay at max_delay
                    sleep_time = min(calculated_delay, max_delay)

                    # Record retry delay
                    retry_delay.labels(stage="unknown").observe(sleep_time)

                    time.sleep(sleep_time)

                except Exception as e:
                    # Unknown errors - retry with exponential backoff
                    last_exception = e

                    if attempt >= max_retries:
                        raise RetryError(
                            f"LLM call failed after {max_retries + 1} attempts",
                            last_exception=last_exception,
                        )

                    # Record retry attempt
                    retry_attempts.labels(stage="unknown", error_type="other").inc()

                    calculated_delay = delay * (backoff_factor**attempt)

                    if add_jitter:
                        jitter = random.uniform(0, calculated_delay * 0.5)
                        calculated_delay += jitter

                    sleep_time = min(calculated_delay, max_delay)

                    # Record retry delay
                    retry_delay.labels(stage="unknown").observe(sleep_time)

                    time.sleep(sleep_time)

            # Should never reach here
            if last_exception:
                raise RetryError(
                    f"LLM call failed after {max_retries + 1} attempts",
                    last_exception=last_exception,
                )
            raise RetryError(f"LLM call failed after {max_retries + 1} attempts")

        return wrapper

    # Support both @retry_llm_call and @retry_llm_call(...) usage
    if func is not None:
        result = decorator(func)
        return result  # type: ignore[return-value]
    return decorator  # type: ignore[return-value]


def _make_async_retry_decorator():
    """Factory to create async retry decorator (to avoid async def calling issue)."""

    def retry_llm_call_async(
        func: Optional[Callable[..., T]] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        add_jitter: bool = True,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        LLM-specific async retry wrapper with exponential backoff and jitter.

        Handles different error types appropriately:
        - Rate limit (429): Exponential backoff with jitter
        - Transient errors (500): Linear retry with short delay
        - Auth errors (401): Fail fast, NO retry

        Usage:
            @retry_llm_call_async(max_retries=3, backoff_factor=2)
            async def call_llm(prompt: str) -> str:
                return await client.complete_async(prompt)

        Args:
            func: Function to wrap (typically LLM complete/complete_json methods)
            max_retries: Maximum number of retry attempts (default: 3)
            backoff_factor: Multiplier for delay between retries (default: 2.0)
            initial_delay: Initial delay in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 60.0)
            add_jitter: Whether to add random jitter to prevent thundering herd (default: True)

        Returns:
            Wrapped function with retry logic

        Raises:
            LLMAuthenticationError: Raised immediately without retry on 401
            RetryError: If all retry attempts fail for retryable errors
        """

        def decorator(f: Callable[..., T]) -> Callable[..., T]:
            @wraps(f)
            async def wrapper(*args, **kwargs) -> T:
                last_exception: Optional[Exception] = None
                delay = initial_delay
                retry_attempts, retry_delay, rate_limit_hits = _get_metrics()

                for attempt in range(max_retries + 1):  # +1 for initial attempt
                    try:
                        result: T = await f(*args, **kwargs)  # type: ignore[misc]
                        return result

                    except LLMAuthenticationError:
                        # Auth errors (401) - fail fast, no retry
                        raise

                    except (LLMRateLimitError, LLMCommunicationError) as e:
                        last_exception = e

                        # If we've exhausted retries, raise
                        if attempt >= max_retries:
                            raise RetryError(
                                f"LLM call failed after {max_retries + 1} attempts",
                                last_exception=last_exception,
                            )

                        # Record rate limit hit
                        if isinstance(e, LLMRateLimitError):
                            rate_limit_hits.labels(stage="unknown").inc()

                        # Record retry attempt
                        error_type = (
                            "rate_limit"
                            if isinstance(e, LLMRateLimitError)
                            else "other"
                        )
                        retry_attempts.labels(
                            stage="unknown", error_type=error_type
                        ).inc()

                        # Calculate delay based on error type
                        if isinstance(e, LLMRateLimitError):
                            # Rate limit (429): exponential backoff
                            calculated_delay = delay * (backoff_factor**attempt)
                        else:
                            # Transient server errors (500): shorter linear retry
                            calculated_delay = delay * (attempt + 1)

                        # Apply jitter to prevent thundering herd
                        if add_jitter:
                            jitter = random.uniform(0, calculated_delay * 0.5)
                            calculated_delay += jitter

                        # Cap delay at max_delay
                        sleep_time = min(calculated_delay, max_delay)

                        # Record retry delay
                        retry_delay.labels(stage="unknown").observe(sleep_time)

                        await asyncio.sleep(sleep_time)

                    except Exception as e:
                        # Unknown errors - retry with exponential backoff
                        last_exception = e

                        if attempt >= max_retries:
                            raise RetryError(
                                f"LLM call failed after {max_retries + 1} attempts",
                                last_exception=last_exception,
                            )

                        # Record retry attempt
                        retry_attempts.labels(stage="unknown", error_type="other").inc()

                        calculated_delay = delay * (backoff_factor**attempt)

                        if add_jitter:
                            jitter = random.uniform(0, calculated_delay * 0.5)
                            calculated_delay += jitter

                        sleep_time = min(calculated_delay, max_delay)

                        # Record retry delay
                        retry_delay.labels(stage="unknown").observe(sleep_time)

                        await asyncio.sleep(sleep_time)

                # Should never reach here
                if last_exception:
                    raise RetryError(
                        f"LLM call failed after {max_retries + 1} attempts",
                        last_exception=last_exception,
                    )
                raise RetryError(f"LLM call failed after {max_retries + 1} attempts")

            return wrapper  # type: ignore[misc, return-value]

        # Support both @retry_llm_call_async and @retry_llm_call_async(...) usage
        result_async: Callable[[Callable[..., T]], Callable[..., T]] = decorator
        if func is not None:
            result_async = decorator(func)  # type: ignore[assignment]
        return result_async  # type: ignore[return-value]

    return retry_llm_call_async


retry_llm_call_async = _make_async_retry_decorator()
