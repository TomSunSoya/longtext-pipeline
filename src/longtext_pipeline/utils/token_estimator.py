"""
Token estimation utilities for cost management.
Provides approximate token counting based on word counts (MVP approach).
"""

import re
from typing import List
from ..models import Part


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using word-based approximation.
    
    For MVP: Uses simple heuristic where ~4-5 characters = 1 token,
    or ~0.75 words = 1 token. This is a rough approximation suitable
    for initial cost estimates and chunk sizing decisions.
    
    Args:
        text: Text to estimate token count for
        
    Returns:
        Estimated token count (integer)
        
    Note:
        This is an approximation. Actual token counts vary by:
        - Tokenizer used (GPT-3.5 vs GPT-4 vs others)
        - Text content (code, prose, mixed)
        - Punctuation and special characters
    """
    if not text or not text.strip():
        return 0
    
    text = text.strip()
    
    # Method 1: Character-based estimation
    # GPT tokenizers typically produce ~4 characters per token on average
    char_estimate = len(text) / 4
    
    # Method 2: Word-based estimation  
    # Average English word is ~4.7 characters, plus spaces
    # Approximation: 0.75 words per token
    words = len(re.findall(r'\b\w+\b', text))
    word_estimate = words / 0.75
    
    # Method 3: Simpler heuristic - ~300 words per 1000 tokens
    # This aligns with OpenAI's general estimates
    simple_estimate = (len(text.split()) / 300) * 1000
    
    # Return average of word-based estimates (more stable than char estimate)
    # Use word estimate as primary, clamp to reasonable bounds
    estimate = int(word_estimate)
    
    # Basic sanity check - ensure estimate is within reasonable bounds
    # Minimum 1 token for any non-empty text
    # Maximum: don't exceed raw character count (each char = token in worst case)
    return max(1, min(estimate, len(text)))


def estimate_tokens_for_part(part: Part) -> int:
    """
    Estimate token count for a single Part object.
    
    Args:
        part: Part object containing content to estimate tokens for
        
    Returns:
        Estimated token count (integer)
        
    Note:
        This is an approximation and not exact token count.
        For budget awareness purposes only, not strict limits.
    """
    return estimate_tokens(part.content)


def estimate_total_tokens(parts: List[Part]) -> int:
    """
    Estimate total token count for a list of Part objects.
    
    Args:
        parts: List of Part objects to calculate total tokens for
        
    Returns:
        Estimated total token count (integer)
        
    Note:
        This is an approximation and not exact token count.
        Useful for budget awareness when processing multiple parts.
    """
    return sum(estimate_tokens(part.content) for part in parts)
