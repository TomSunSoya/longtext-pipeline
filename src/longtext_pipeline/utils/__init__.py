"""
Utility modules for longtext-pipeline.

This package provides general-purpose utility functions:
- I/O operations (file reading/writing with atomic writes)
- Hashing (SHA-256 content hashing)
- Retry logic (exponential backoff)
- Text cleaning (whitespace normalization)
- Token estimation (approximate token counting)
"""

from .io import read_file, write_file, ensure_dir, FileOperationError
from .hashing import hash_content, hash_file
from .retry import retry_with_backoff, RetryError
from .text_clean import clean_text, extract_sections
from .token_estimator import estimate_tokens

__all__ = [
    # io
    'read_file',
    'write_file',
    'ensure_dir',
    'FileOperationError',
    # hashing
    'hash_content',
    'hash_file',
    # retry
    'retry_with_backoff',
    'RetryError',
    # text_clean
    'clean_text',
    'extract_sections',
    # token_estimator
    'estimate_tokens',
]
