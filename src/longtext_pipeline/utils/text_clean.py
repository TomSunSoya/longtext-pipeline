"""
Text cleaning utilities for preprocessing input text.
Provides normalization and section extraction helpers.
"""

from typing import List


def clean_text(
    content: str, strip_extra_whitespace: bool = True, normalize_encoding: bool = True
) -> str:
    """
    Enhanced text cleaning with multiple normalization options.

    - Handles multiple whitespace normalizations (spaces, tabs, newlines)
    - Collapses multiple consecutive blank lines to single blank line
    - Strips leading/trailing blank lines from content
    - Normalizes encoding (assumes UTF-8 per spec)
    - Normalizes line endings

    Args:
        content: Raw text content to clean
        strip_extra_whitespace: Whether to remove leading/trailing whitespace (default True)
        normalize_encoding: Whether to ensure UTF-8 encoding normalization (default True)

    Returns:
        Cleaned text with normalized whitespace and encoding
    """
    if not content:
        return ""

    # Normalize encoding to UTF-8 - per SPEC.md requirement for UTF-8 exclusivity
    if normalize_encoding:
        try:
            # Ensure we're working with properly decoded string
            if isinstance(content, bytes):
                content = content.decode("utf-8")
        except UnicodeDecodeError:
            # Attempt to handle common encoding issues by ignoring errors
            content = content.encode("utf-8", errors="ignore").decode("utf-8")

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    if strip_extra_whitespace:
        # Split into lines and strip each line (removes trailing spaces/tabs)
        lines = [line.rstrip() for line in content.split("\n")]

        # Collapse multiple consecutive blank lines
        cleaned_lines: List[str] = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()

            if is_blank:
                if not prev_blank:
                    cleaned_lines.append("")  # Keep single blank line
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False

        # Remove leading and trailing blank lines
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        # Strip leading whitespace from first non-blank line
        if cleaned_lines:
            # Find first non-blank line
            first_idx = 0
            while (
                first_idx < len(cleaned_lines) and not cleaned_lines[first_idx].strip()
            ):
                first_idx += 1

            if first_idx < len(cleaned_lines):
                # Strip leading whitespace from first content line
                cleaned_lines[first_idx] = cleaned_lines[first_idx].lstrip()

        return "\n".join(cleaned_lines)
    else:
        return content
