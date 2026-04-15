"""Batch processing module for longtext-pipeline.

This module provides batch processing capabilities for processing
multiple files with sequential or parallel execution strategies.
"""

from .orchestrator import BatchOrchestrator, FileResult, BatchResult

__all__ = ["BatchOrchestrator", "FileResult", "BatchResult"]
