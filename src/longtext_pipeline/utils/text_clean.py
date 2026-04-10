"""
Text cleaning utilities for preprocessing input text.
Provides normalization and section extraction helpers.
"""

import re
from typing import List, Optional


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


def extract_sections(
    content: str,
    delimiter: str = r"^(#{1,6})\s+(.+)$",
    include_headers: bool = True,
    preserve_structure: bool = True,
) -> List[str]:
    """
    Enhanced section extraction with markdown headers fallback to length-based.

    Args:
        content: Text content to split
        delimiter: Regex pattern for section delimiter (default: markdown headers)
        include_headers: Whether to include the header line in each section
        preserve_structure: Whether to attempt preservation of text structure with fallback

    Returns:
        List of section content strings
    """
    if not content.strip():
        return []

    lines = content.split("\n")
    sections: List[str] = []
    current_section: List[str] = []
    current_header: Optional[str] = None

    header_pattern = re.compile(delimiter, re.IGNORECASE)

    for line in lines:
        match = header_pattern.match(line.strip())

        if match:
            # Save previous section if exists
            if current_section:
                section_content = "\n".join(current_section)
                if section_content.strip():
                    sections.append(section_content)

            # Start new section
            current_section = []
            if include_headers:
                current_header = match.group(0)
                current_section.append(current_header)
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_content = "\n".join(current_section)
        if section_content.strip():
            sections.append(section_content)

    # If no sections from markdown headers and preserve_structure is True, try fallback
    # Fallback strategy: if no headers found or only headers, split by length-based sections
    if preserve_structure and len(sections) <= 1:
        # Try to split content by length or paragraphs as fallback
        fallback_sections = _length_based_extraction(content, max_length=500)
        if len(fallback_sections) > len(sections):
            return fallback_sections

    # Remove empty sections
    sections = [s for s in sections if s.strip()]

    return sections


def _length_based_extraction(content: str, max_length: int = 500) -> List[str]:
    """
    Helper for extracting sections based on content length with paragraph awareness.
    Used as fallback when markdown header extraction fails.

    Args:
        content: Text content to split
        max_length: Maximum length of each section (approximate)

    Returns:
        List of section content strings
    """
    if not content.strip():
        return []

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    sections = []
    current_section = ""

    for paragraph in paragraphs:
        if len(current_section) + len(paragraph) <= max_length:
            # Add paragraph to current section
            if current_section:
                current_section += "\n\n" + paragraph
            else:
                current_section = paragraph
        else:
            # Current section is full, finalize it and start a new one
            if current_section.strip():
                sections.append(current_section)
            current_section = paragraph

    # Add the last section
    if current_section.strip():
        sections.append(current_section)

    return sections


def preserve_structure(content: str, encoding_handling: bool = True) -> str:
    """
    Wrapper function to preserve text structure while normalizing for processing.

    Args:
        content: Text content to preserve structure for
        encoding_handling: Whether to normalize encoding (UTF-8 per spec)

    Returns:
        Text content with structure preserved but normalized encoding
    """
    # Clean text but preserve essential structure
    cleaned = clean_text(
        content, strip_extra_whitespace=False, normalize_encoding=encoding_handling
    )
    return cleaned
