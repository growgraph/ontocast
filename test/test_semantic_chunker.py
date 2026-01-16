"""Test suite for SemanticChunker.

This test suite ensures that:
1. Chunks, when joined, reproduce the original text (length and content)
2. If max_size and min_size are provided, all chunks are >= min_size and <= max_size
"""

import json
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.util import SENTENCE_SPLIT_REGEX, SemanticChunker


class TestSemanticChunker:
    """Core tests for SemanticChunker focusing on text reconstruction and size constraints."""

    def test_chunks_reproduce_original_text_when_joined(
        self, embeddings: Embeddings, sample_text: str
    ):
        """Test that chunks, when joined, reproduce the original text."""
        chunk_config = ChunkConfig(
            min_size=1,  # Very small min_size to allow any chunk size
            max_size=100000,  # Very large max_size to allow any chunk size
        )
        chunker = SemanticChunker(
            embeddings=embeddings,
            chunk_config=chunk_config,
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
        )

        chunks = chunker.split_text(sample_text)
        joined_text = "".join(chunks)

        # Verify length is approximately the same
        length_diff = abs(len(joined_text) - len(sample_text))
        assert length_diff <= len(chunks), (
            f"Joined text length difference ({length_diff}) is too large. "
            f"Original: {len(sample_text)}, Joined: {len(joined_text)}"
        )

        # Verify content is preserved (normalize whitespace for comparison)
        original_normalized = re.sub(r"\s+", " ", sample_text.strip())
        joined_normalized = re.sub(r"\s+", " ", joined_text.strip())

        # Check word coverage
        original_words = set(re.findall(r"\b\w+\b", original_normalized.lower()))
        joined_words = set(re.findall(r"\b\w+\b", joined_normalized.lower()))
        missing_words = original_words - joined_words
        coverage = (
            1 - (len(missing_words) / len(original_words)) if original_words else 1
        )

        assert coverage >= 0.95, (
            f"Word coverage too low: {coverage:.1%}. "
            f"Missing {len(missing_words)} words: {list(missing_words)[:10]}"
        )

    def test_chunks_respect_min_and_max_size(
        self, embeddings: Embeddings, long_text: str
    ):
        """Test that chunks respect both min_size and max_size constraints."""
        min_size = 200
        max_size = 1000
        chunk_config = ChunkConfig(
            min_size=min_size,
            max_size=max_size,
        )
        chunker = SemanticChunker(
            embeddings=embeddings,
            chunk_config=chunk_config,
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
        )

        chunks = chunker.split_text(long_text)

        assert len(chunks) > 0, "Should produce at least one chunk"
        for i, chunk in enumerate(chunks):
            # All chunks must respect max_size
            assert len(chunk) <= max_size, (
                f"Chunk {i} has length {len(chunk)} which exceeds max_size {max_size}"
            )
            # All but last chunk should meet min_size
            if i < len(chunks) - 1:
                assert len(chunk) >= min_size, (
                    f"Chunk {i} has length {len(chunk)} which is less than min_size {min_size}"
                )

        # Verify joined text exactly reproduces original
        joined_text = "".join(chunks)
        assert joined_text == long_text, (
            f"Joined text does not exactly match original text. "
            f"Length difference: {abs(len(joined_text) - len(long_text))} characters. "
            f"Original length: {len(long_text)}, Joined length: {len(joined_text)}. "
            f"First difference at position: {next((i for i, (a, b) in enumerate(zip(long_text, joined_text)) if a != b), min(len(long_text), len(joined_text)))}"
        )

    def test_chunker_test_json_with_strict_size_constraints(
        self, embeddings: Embeddings
    ):
        """Test with chunker.test.json using strict size constraints (min_size=2000, max_size=4000).

        This test reproduces a bug where:
        1. Chunks smaller than min_size are produced
        2. Chunks are almost exactly max_size (suggesting brute force cutting)
        """
        # Load test data
        json_file = Path(__file__).parent / "data" / "chunker.test.json"
        if not json_file.exists():
            pytest.skip(f"Test data file not found: {json_file}")

        data = json.load(open(json_file))
        text = data.get("text", "")
        if not text:
            pytest.skip("No text found in test data")

        min_size = 2000
        max_size = 4000
        chunk_config = ChunkConfig(
            min_size=min_size,
            max_size=max_size,
        )
        chunker = SemanticChunker(
            embeddings=embeddings,
            chunk_config=chunk_config,
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
        )

        chunks = chunker.split_text(text)
        chunk_sizes = [len(c) for c in chunks]

        # Verify all chunks respect max_size
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= max_size, (
                f"Chunk {i} has length {len(chunk)} which exceeds max_size {max_size}. "
                f"Chunk sizes: {chunk_sizes}"
            )

        # Verify chunks meet min_size (except possibly the last one)
        # All but the last chunk should meet min_size
        # The last chunk may be smaller if remaining text is less than min_size
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                assert len(chunks[i]) >= min_size, (
                    f"Chunk {i} (not last) has length {len(chunks[i])} which is less than "
                    f"min_size {min_size}. Chunk sizes: {chunk_sizes}"
                )

        # Even the last chunk should be reasonably sized (at least 50% of min_size)
        # unless the total remaining text is very small
        if len(chunks) > 0:
            last_chunk_size = len(chunks[-1])
            if last_chunk_size < min_size * 0.5 and len(chunks) > 1:
                # Check if this is really the last chunk or if there's a problem
                total_remaining = sum(len(c) for c in chunks if len(c) < min_size)
                if total_remaining >= min_size:
                    pytest.fail(
                        f"Last chunk has length {last_chunk_size} which is too small. "
                        f"Total size of small chunks: {total_remaining} >= {min_size}, "
                        f"so they should have been merged. Chunk sizes: {chunk_sizes}"
                    )

        # Check for brute force cutting - chunks should not all be clustered near max_size
        chunks_near_max = sum(1 for size in chunk_sizes if size >= max_size * 0.98)
        ratio_near_max = chunks_near_max / len(chunks) if chunks else 0

        # If more than 60% of chunks are near max_size, it suggests brute force cutting
        assert ratio_near_max < 0.6, (
            f"Too many chunks ({chunks_near_max}/{len(chunks)} = {ratio_near_max:.1%}) "
            f"are near max_size ({max_size * 0.98:.0f}), suggesting brute force cutting. "
            f"Chunk sizes: {chunk_sizes}"
        )

        # Verify joined text exactly reproduces original
        joined_text = "".join(chunks)
        assert joined_text == text, (
            f"Joined text does not exactly match original text. "
            f"Length difference: {abs(len(joined_text) - len(text))} characters. "
            f"Original length: {len(text)}, Joined length: {len(joined_text)}. "
            f"First difference at position: {next((i for i, (a, b) in enumerate(zip(text, joined_text)) if a != b), min(len(text), len(joined_text)))}"
        )
