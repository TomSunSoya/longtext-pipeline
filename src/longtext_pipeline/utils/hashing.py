"""
Hashing utilities for SHA-256 content hashing.
Provides content identity verification and cache key generation.
"""

import hashlib
from pathlib import Path
from typing import Union

from .io import read_file


def hash_content(content: str) -> str:
    """
    Calculate SHA-256 hash of string content.
    
    Args:
        content: String content to hash
        
    Returns:
        Hexadecimal hash string (64 characters)
    """
    if not isinstance(content, str):
        content = str(content)
    
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def hash_file(path: Union[str, Path]) -> str:
    """
    Calculate SHA-256 hash of file content.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        Hexadecimal hash string (64 characters)
    """
    content = read_file(path)
    return hash_content(content)
