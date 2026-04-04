"""Text chunking logic for the longtext pipeline.

This module provides the TextSplitter class for splitting text into
manageable segments according to token limits with overlap support
for context preservation.
"""

from typing import List

from .errors import InputError
from .models import Part
from .utils.token_estimator import estimate_tokens
from .utils.text_clean import clean_text


class TextSplitter:
    """Splits text into parts by token count with overlap support.
    
    This splitter uses a simple token estimation approximation (words/3)
    for MVP. It preserves context between chunks using overlap and handles
    edge cases like empty input, tiny input, and very long sections.
    
    Attributes:
        chunk_size: Maximum token count per chunk
        overlap: Token count to overlap between consecutive chunks
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """Initialize the splitter with configuration.
        
        Args:
            chunk_size: Maximum token count per chunk (default: 1000)
            overlap: Token count to overlap between chunks (default: 100)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, content: str, chunk_size: int | None = None, overlap: int | None = None, preprocess: bool = True) -> List[Part]:
        """Split text into parts by token count.
        
        Args:
            content: The input text to split
            chunk_size: Override for max tokens per chunk (optional)
            overlap: Override for overlap tokens (optional)
            preprocess: Whether to preprocess content with clean_text (default True)
            
        Returns:
            List of Part objects containing the split text
            NOTE: Parts are indexed consecutively starting at 0
        Raises:
            InputError: If content is empty or whitespace only
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        
        # Preprocess content first if requested
        if preprocess:
            content = clean_text(content)
        
        # Edge case: empty input
        if not content or not content.strip():
            raise InputError("Cannot split empty or whitespace-only content")
        
        # Edge case: tiny input - single chunk
        tokens = estimate_tokens(content)
        if tokens <= chunk_size:
            return [Part(index=0, content=content, token_count=tokens)]
        
        # Split content into overlapping chunks
        parts = []
        max_overlap = min(overlap, chunk_size // 4)  # Max 25% of chunk size
        
        # Split by words to preserve semantic coherence
        words = content.split()
        current_start = 0
        part_index = 0
        
        while current_start < len(words):
            # Build chunk by accumulating words until near limit
            chunk_words = []
            token_count = 0
            end = current_start
            
            while end < len(words) and token_count < chunk_size:
                chunk_words.append(words[end])
                # Estimate tokens: 3 characters ≈ 1 token
                token_count += len(words[end]) // 3 + 1
                end += 1
            
            # Create part with this chunk
            part_content = ' '.join(chunk_words)
            part_tokens = estimate_tokens(part_content)
            parts.append(Part(
                index=part_index,
                content=part_content,
                token_count=part_tokens
            ))
            
            # Move to next chunk with overlap
            if end >= len(words):
                break  # No more content
            
            # Calculate overlap words (10% of chunk or configured overlap)
            overlap_words = max(1, len(chunk_words) // 10)
            current_start = max(0, end - overlap_words)
            part_index += 1
        
        return parts
