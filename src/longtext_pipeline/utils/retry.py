"""
Retry utilities for handling transient failures.
Provides generic retry wrapper with exponential backoff.
"""

import random
import time
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Tuple, Union

from ..errors import PipelineError, LLMRateLimitError, LLMCommunicationError, LLMAuthenticationError


class RetryError(PipelineError):
    """Raised when all retry attempts have failed."""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


T = TypeVar('T')


def retry_with_backoff(
    func: Optional[Callable[..., T]] = None,
    max_retries: int = 3,
    backoff_factor: float = 2,
    retry_exceptions: Tuple[type, ...] = (Exception,),
    max_delay: float = 60.0,
) -> Callable[..., T]:
    """
    Generic retry wrapper with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_retries=3, backoff_factor=2)
        def my_function():
            pass
            
        # Or directly:
        result = retry_with_backoff(my_function, max_retries=5)(args)
    
    Args:
        func: Function to wrap (optional when used as decorator)
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries (exponential)
        retry_exceptions: Tuple of exception types to retry on
        max_delay: Maximum delay between retries in seconds
        
    Returns:
        Wrapped function with retry logic
    
    Raises:
        RetryError: If all retry attempts fail
    """
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            delay = 1.0  # Initial delay in seconds
            
            for attempt in range(max_retries + 1):  # +1 because first attempt is try 0
                try:
                    return f(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    
                    # If we've exhausted retries, raise
                    if attempt >= max_retries:
                        raise RetryError(
                            f"Failed after {max_retries + 1} attempts",
                            last_exception=last_exception
                        )
                    
                    # Apply exponential backoff with jitter
                    time.sleep(min(delay, max_delay))
                    delay *= backoff_factor
            
            # Should never reach here, but just in case
            if last_exception:
                raise RetryError(
                    f"Failed after {max_retries + 1} attempts",
                    last_exception=last_exception
                )
            raise RetryError(f"Failed after {max_retries + 1} attempts")
        
        return wrapper
    
    # Support both @decorator and decorator(func) usage
    if func is not None:
        return decorator(func)
    return decorator


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
                            last_exception=last_exception
                        )
                    
                    # Calculate delay based on error type
                    if isinstance(e, LLMRateLimitError):
                        # Rate limit (429): exponential backoff
                        calculated_delay = delay * (backoff_factor ** attempt)
                    else:
                        # Transient server errors (500): shorter linear retry
                        calculated_delay = delay * (attempt + 1)
                    
                    # Apply jitter to prevent thundering herd
                    if add_jitter:
                        jitter = random.uniform(0, calculated_delay * 0.5)
                        calculated_delay += jitter
                    
                    # Cap delay at max_delay
                    sleep_time = min(calculated_delay, max_delay)
                    
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    # Unknown errors - retry with exponential backoff
                    last_exception = e
                    
                    if attempt >= max_retries:
                        raise RetryError(
                            f"LLM call failed after {max_retries + 1} attempts",
                            last_exception=last_exception
                        )
                    
                    calculated_delay = delay * (backoff_factor ** attempt)
                    
                    if add_jitter:
                        jitter = random.uniform(0, calculated_delay * 0.5)
                        calculated_delay += jitter
                    
                    sleep_time = min(calculated_delay, max_delay)
                    time.sleep(sleep_time)
            
            # Should never reach here
            if last_exception:
                raise RetryError(
                    f"LLM call failed after {max_retries + 1} attempts",
                    last_exception=last_exception
                )
            raise RetryError(f"LLM call failed after {max_retries + 1} attempts")
        
        return wrapper
    
    # Support both @retry_llm_call and @retry_llm_call(...) usage
    if func is not None:
        return decorator(func)
    return decorator
