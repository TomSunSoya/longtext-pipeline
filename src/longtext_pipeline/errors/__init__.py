"""Custom exceptions for the longtext pipeline.

This module defines an exception hierarchy for proper error categorization
and handling throughout the pipeline. The hierarchy supports the
Continue-with-Partial strategy for robustness.
"""

from typing import Any, List

from .continuation import ErrorAggregator, PartialResult


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    pass


class ConfigError(PipelineError):
    """Raised when there is an error in configuration validation or loading."""

    pass


class InputError(PipelineError):
    """Raised when there is an error with the input file or data."""

    pass


class LLMError(PipelineError):
    """Raised when there is an error communicating with or using an LLM."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when the LLM request exceeds the timeout threshold."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limiting is encountered."""

    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails."""

    pass


class LLMContentFilterError(LLMError):
    """Raised when content violates provider policies."""

    pass


class LLMCommunicationError(LLMError):
    """Raised for network/communication failures."""

    pass


class LLMResponseError(LLMError):
    """Raised when response cannot be parsed or is invalid."""

    pass


class ManifestError(PipelineError):
    """Raised when there is an error reading or writing the manifest."""

    pass


class PipelineLockError(PipelineError):
    """Raised when another process already holds the pipeline run lock."""

    pass


class StageFailedError(PipelineError):
    """Raised when a stage fails but pipeline should continue.

    This exception is used to track partial success scenarios where
    the pipeline can continue processing other parts despite individual
    stage failures.

    Attributes:
        stage_name: Name of the stage that failed
        errors: List of individual errors that occurred
        partial_result: Any partial result that can be used
    """

    def __init__(
        self, stage_name: str, errors: List[Exception], partial_result: Any = None
    ):
        self.stage_name = stage_name
        self.errors = errors
        self.partial_result = partial_result
        error_msgs = "; ".join(str(e) for e in errors)
        super().__init__(f"Stage '{stage_name}' failed: {error_msgs}")


__all__ = [
    # Base errors
    "PipelineError",
    "ConfigError",
    "InputError",
    "LLMError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMContentFilterError",
    "LLMCommunicationError",
    "LLMResponseError",
    "ManifestError",
    "PipelineLockError",
    "StageFailedError",
    # Continuation errors
    "PartialResult",
    "ErrorAggregator",
]
