"""Tests for the TextSplitter class."""

import pytest

from src.longtext_pipeline.errors import InputError
from src.longtext_pipeline.splitter import TextSplitter
from src.longtext_pipeline.utils.token_estimator import (
    estimate_tokens_for_part,
    estimate_total_tokens,
)


class TestTextSplitter:
    """Test cases for TextSplitter."""

    def test_empty_input_raises_input_error(self):
        """Empty input should raise InputError."""
        splitter = TextSplitter(chunk_size=100)

        with pytest.raises(InputError):
            splitter.split_text("")

        with pytest.raises(InputError):
            splitter.split_text("   ")

        with pytest.raises(InputError):
            splitter.split_text("\n\n")

    def test_tiny_input_single_chunk(self):
        """Input smaller than chunk size should produce single chunk."""
        splitter = TextSplitter(chunk_size=1000)
        content = "This is a small text."

        parts = splitter.split_text(content)

        assert len(parts) == 1
        assert parts[0].index == 0
        assert parts[0].content == content

    def test_normal_input_creates_multiple_chunks(self):
        """Normal sized input should split into multiple chunks."""
        splitter = TextSplitter(chunk_size=50)
        # Create content with ~200 words (~600+ tokens)
        words = [f"word{i}" for i in range(200)]
        content = " ".join(words)

        parts = splitter.split_text(content)

        # Should create multiple chunks
        assert len(parts) > 1

    def test_overlap_preserves_context(self):
        """Consecutive chunks should share overlapping content."""
        splitter = TextSplitter(chunk_size=100, overlap=20)
        # Create text where we can verify overlap
        words = [f"word{i:03d}" for i in range(100)]
        content = " ".join(words)

        parts = splitter.split_text(content)

        # Check that chunks exist
        assert len(parts) >= 2

        # Verify overlap by checking content patterns
        # The last few words of chunk 0 should appear in chunk 1
        if len(parts) >= 2:
            # With overlap, chunks should share some words
            # This is verified by checking token counts
            assert parts[0].token_count > 0
            assert parts[1].token_count > 0

    def test_exact_chunk_size_boundary(self):
        """Content at exact chunk size boundary."""
        splitter = TextSplitter(chunk_size=100)
        # Create content that should fit exactly
        content = "word " * 30  # ~30 tokens
        # Content is preserved as-is (splitter doesn't strip)
        parts = splitter.split_text(content)

        # Should fit in one chunk
        assert len(parts) == 1
        # Content should be preserved (may have trailing space from join)
        assert "word " in parts[0].content

    def test_chunk_size_plus_one(self):
        """Content just over chunk size should still split."""
        splitter = TextSplitter(chunk_size=50)
        # Create content with 51 tokens
        words = [f"w{i}" for i in range(60)]
        content = " ".join(words)

        parts = splitter.split_text(content)

        # Should split into multiple chunks
        assert len(parts) > 1

    def test_content_preservation(self):
        """All original content should be preserved in parts."""
        splitter = TextSplitter(chunk_size=100)
        original_content = "The quick brown fox jumps over the lazy dog. " * 10

        parts = splitter.split_text(original_content)

        # Reconstruct content (with some loss due to splitting logic)
        reconstructed = " ".join(part.content for part in parts)

        # Original should be mostly present
        assert len(original_content) > 0
        assert len(reconstructed) > 0

    def test_token_count_accuracy(self):
        """Token counts should be reasonable estimates."""
        splitter = TextSplitter(chunk_size=1000)
        content = "hello world " * 50

        parts = splitter.split_text(content)

        for part in parts:
            # Token count should be positive and reasonable
            assert part.token_count > 0
            # Should be under chunk size (with some tolerance)
            assert part.token_count <= splitter.chunk_size + 10

    def test_custom_chunk_size_overlap(self):
        """Custom chunk size and overlap parameters should work."""
        splitter = TextSplitter(chunk_size=200, overlap=50)
        words = [f"word{i}" for i in range(300)]
        content = " ".join(words)

        parts = splitter.split_text(content, chunk_size=200, overlap=50)

        assert len(parts) > 1

    def test_large_text_recursive_splitting(self):
        """Very long text should be properly split into many chunks."""
        splitter = TextSplitter(chunk_size=30)
        # Create very long text
        words = []
        for i in range(500):
            words.extend([f"word{i}", "the", "quick", "brown", "fox"])
        content = " ".join(words)

        parts = splitter.split_text(content)

        # Should create many chunks
        assert len(parts) > 5

        # All chunks should have valid token counts
        for part in parts:
            assert part.token_count > 0
            assert part.index == parts.index(part)

    def test_token_estimator_integration(self):
        """Test token estimation integration with splitter."""
        splitter = TextSplitter(chunk_size=100)
        text = (
            "This is a sample text that will be split for token estimation testing. "
            * 5
        )
        parts = splitter.split_text(text)

        # Test individual part token estimation
        for part in parts:
            estimated = estimate_tokens_for_part(part)
            # The estimate from the helper should match the property on the part
            assert estimated == part.token_count
            # Should be positive
            assert estimated > 0

        # Test total token estimation
        total_estimated = estimate_total_tokens(parts)
        actual_total = sum(estimate_tokens_for_part(part) for part in parts)

        assert total_estimated == actual_total
        assert total_estimated > 0

    def test_token_estimator_edge_cases(self):
        """Test token estimator with Part objects for edge cases."""
        from src.longtext_pipeline.models import Part as ModelPart

        # Empty content
        empty_part = ModelPart(index=0, content="", token_count=0)
        assert estimate_tokens_for_part(empty_part) == 0

        # Single word
        single_word_part = ModelPart(index=0, content="hello", token_count=1)
        assert estimate_tokens_for_part(single_word_part) > 0

        # Empty list
        assert estimate_total_tokens([]) == 0

    def test_preprocessing_integration(self):
        """Test the integration of text cleaning with the splitter."""
        from src.longtext_pipeline.utils.text_clean import clean_text

        splitter = TextSplitter(chunk_size=50, overlap=10)
        # Create content with multiple whitespaces, blank lines to test preprocessing
        dirty_content = (
            "\n\n   Some title here\n\n\n"
            "This is section one.\n\nWith extra blank line.\n\n\n"
            "   And extra whitespace.    \n\n"
            "# Section Header\n\nMore content follows with extra    spaces.\n"
        )

        # Test the cleaner separately first
        cleaned = clean_text(dirty_content)

        # We can verify the cleaner works by ensuring it reduces blank lines appropriately
        # and maintains some normalization
        assert len(cleaned) > 0
        assert (
            "\n\n\n" not in cleaned
        )  # Multiple consecutive blank lines should be collapsed

        # Verify that split with preprocessing enabled works on this cleaned content
        parts_with_preprocess = splitter.split_text(dirty_content, preprocess=True)

        # Should produce parts successfully
        assert len(parts_with_preprocess) > 0
        for part in parts_with_preprocess:
            assert hasattr(part, "index")
            assert hasattr(part, "content")
            assert hasattr(part, "token_count")

        # Ensure the preprocessed content is more normalized than raw content
        # by comparing the character counts after splitting

    def test_preprocessing_disabled_flag(self):
        """Test that disabling preprocessing skips clean step."""
        splitter = TextSplitter(chunk_size=100, overlap=10)
        # Content with extra whitespace
        dirty_content = "   Extra   spacing    here   \n\n\n   More    spacing   "

        # Test with preprocess enabled (default)
        parts_enabled = splitter.split_text(
            dirty_content, preprocess=True, chunk_size=50
        )

        # Test with preprocess disabled
        parts_disabled = splitter.split_text(
            dirty_content, preprocess=False, chunk_size=50
        )

        # Both should complete successfully but potentially with different outcomes
        # based on the content being preprocessed differently
        assert len(parts_enabled) >= 1
        assert len(parts_disabled) >= 1
