"""
Continuation error types and utilities for the Continue-with-Partial error strategy.

This module defines additional error-related classes and utilities
specifically for improved continued execution strategies, complementing
the base error types in the parent errors.py module.
"""

import threading
from typing import Any, List, Union
from dataclasses import dataclass


@dataclass
class PartialResult:
    """Wrapper for results that include partial success information.

    When a function encounters errors but still produces usable output,
    it can wrap the result with error information using this class.

    Attributes:
        success: Whether the operation was considered successful
        data: The actual result data (can be partial)
        errors: List of errors that occurred during processing
        warnings: List of warnings about the partial state
        metadata: Additional metadata about the processing state
    """

    success: bool
    data: Any
    errors: List[str]
    warnings: List[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

    def merge_with_other(self, other: "PartialResult") -> "PartialResult":
        """Merge this partial result with another, combining data and errors."""
        return PartialResult(
            success=self.success and other.success,
            data=self.data
            if hasattr(other, "data")
            and other.data is not None
            and len(other.data) > len(self.data)
            else self.data,
            errors=list(set(self.errors + other.errors)),
            warnings=list(set((self.warnings or []) + (other.warnings or []))),
            metadata={**self.metadata, **other.metadata},
        )


class ErrorAggregator:
    """Aggregate errors across multiple pipeline stages for reporting.

    This class collects error information across all stages to provide
    comprehensive error reports at the end of pipeline execution.

    Attributes:
        errors: Dictionary mapping stage names to their respective errors
        warnings: Dictionary mapping stage names to their respective warnings
        stats: Statistics about error patterns and frequencies
    """

    def __init__(self):
        self.errors = {}
        self.warnings = {}
        self.stats = {
            "total_errors": 0,
            "stages_with_errors": set(),
            "recovery_success": [],
            "recovery_failed": [],
        }
        self._lock = threading.Lock()

    def add_errors(self, stage_name: str, errors: List[Union[Exception, str]]) -> None:
        """Add errors for a specific stage."""
        # Convert exceptions to strings
        new_errors = [
            str(error) if isinstance(error, Exception) else error for error in errors
        ]

        with self._lock:
            if stage_name not in self.errors:
                self.errors[stage_name] = []

            self.errors[stage_name].extend(new_errors)
            self.stats["total_errors"] += len(new_errors)
            self.stats["stages_with_errors"].add(stage_name)

    def add_warning(self, stage_name: str, warning: str) -> None:
        """Add a warning for a specific stage."""
        with self._lock:
            if stage_name not in self.warnings:
                self.warnings[stage_name] = []

            self.warnings[stage_name].append(warning)

    def get_stage_summary(self, stage_name: str) -> dict:
        """Get error/warning summary for a specific stage."""
        return {
            "stage": stage_name,
            "errors": self.errors.get(stage_name, []),
            "warnings": self.warnings.get(stage_name, []),
            "has_errors": stage_name in self.errors,
            "has_warnings": stage_name in self.warnings,
        }

    def get_full_summary(self) -> dict:
        """Get a comprehensive error summary across all stages."""
        stage_summaries = {}
        for stage_name in set(list(self.errors.keys()) + list(self.warnings.keys())):
            stage_summaries[stage_name] = self.get_stage_summary(stage_name)

        return {
            "stages": stage_summaries,
            "overall_stats": {
                "total_errors": self.stats["total_errors"],
                "stages_with_errors": list(self.stats["stages_with_errors"]),
                "total_warning_count": sum(
                    len(warn_list) for warn_list in self.warnings.values()
                ),
            },
            "has_errors": len(self.stats["stages_with_errors"]) > 0,
        }

    def clear(self) -> None:
        """Clear all aggregated errors."""
        with self._lock:
            self.errors.clear()
            self.warnings.clear()
            self.stats = {
                "total_errors": 0,
                "stages_with_errors": set(),
                "recovery_success": [],
                "recovery_failed": [],
            }
