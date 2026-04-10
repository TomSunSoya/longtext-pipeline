"""
I/O utilities for file reading and writing operations.
Provides safe file operations with UTF-8 encoding and atomic writes.
"""

import os
import tempfile
from pathlib import Path
from typing import Union

from ..errors import PipelineError


class FileOperationError(PipelineError):
    """Raised when file operation fails."""

    pass


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create (or file path to ensure parent exists)
    """
    dir_path = Path(path)
    if not dir_path.suffix:  # It's a directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
    else:  # It's a file path, ensure parent directory exists
        parent = dir_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)


def read_file(path: Union[str, Path]) -> str:
    """
    Read file content with UTF-8 encoding.

    Args:
        path: Path to the file to read

    Returns:
        File content as string

    Raises:
        FileOperationError: If file cannot be read
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise FileOperationError(f"File not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise FileOperationError(f"Encoding error reading {path}: {e}")
    except IOError as e:
        raise FileOperationError(f"Failed to read {path}: {e}")


def write_file(path: Union[str, Path], content: str) -> None:
    """
    Write content to file with atomic write protection.
    Writes to temporary file first, then renames to target path.

    Args:
        path: Target file path
        content: Content to write

    Raises:
        FileOperationError: If write operation fails
    """
    try:
        file_path = Path(path)

        # Ensure parent directory exists
        ensure_dir(file_path)

        # Write to temporary file first (atomic write protection)
        dir_path = file_path.parent

        fd, temp_path = tempfile.mkstemp(
            dir=str(dir_path), prefix=".tmp_", suffix=".tmp"
        )

        try:
            # Write content to temp file
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_path, str(file_path))

        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    except IOError as e:
        raise FileOperationError(f"Failed to write {path}: {e}")
    except OSError as e:
        raise FileOperationError(f"OS error writing {path}: {e}")
