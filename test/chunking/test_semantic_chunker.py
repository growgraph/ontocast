"""Test suite for SemanticChunker.

This test suite ensures that:
1. Chunks, when joined, reproduce the original text (length and content)
2. If max_size and min_size are provided, all chunks are >= min_size and <= max_size
3. The same text and config always produce the same chunks, in any process
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.proposition import SENTENCE_SPLIT_REGEX
from ontocast.tool.chunk.util import SemanticChunker

pytestmark = pytest.mark.unit


@pytest.mark.slow
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


class HashEmbeddings(Embeddings):
    """Deterministic stand-in encoder: each text maps to a fixed random vector.

    Keeps the determinism tests about the reduction and clustering, not the
    model, and loads no weights.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=64).tolist()


def _topic_text(n_sentences: int) -> str:
    topics = ["Alpha", "Beta", "Gamma", "Delta"]
    return "".join(
        f"{topics[i % len(topics)]} item {i} states a fact about the matter here. "
        for i in range(n_sentences)
    )


def _split_lengths(n_sentences: int) -> list[int]:
    chunker = SemanticChunker(
        embeddings=HashEmbeddings(),
        chunk_config=ChunkConfig(min_size=800, max_size=4000),
        sentence_split_regex=SENTENCE_SPLIT_REGEX,
    )
    return [len(chunk) for chunk in chunker.split_text(_topic_text(n_sentences))]


class TestSemanticChunkerDeterminism:
    """The same text and config must always produce the same unit boundaries."""

    # 700 sentences is large enough for UMAP's spectral init to vary between
    # seeded runs, which is what the PCA init guards against.
    @pytest.mark.parametrize("n_sentences", [60, 700])
    def test_split_is_deterministic_across_calls(self, n_sentences: int):
        first = _split_lengths(n_sentences)
        assert len(first) > 1
        for _ in range(2):
            assert _split_lengths(n_sentences) == first

    @pytest.mark.slow
    def test_split_is_deterministic_across_processes(self):
        # Load this file by path, so the child needs no cwd or package root.
        code = (
            "import importlib.util as u;"
            f"s = u.spec_from_file_location('chunker_probe', {str(Path(__file__).resolve())!r});"
            "m = u.module_from_spec(s); s.loader.exec_module(m);"
            "print(m._split_lengths(250))"
        )
        outputs = [
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(2)
        ]
        assert outputs[0]
        assert outputs[0] == outputs[1]

    def test_few_sentences_over_max_size_are_packed_without_clustering(self):
        text = "".join(f"Sentence{i} " + "x" * 600 + ". " for i in range(3))
        embeddings = HashEmbeddings()
        chunker = SemanticChunker(
            embeddings=embeddings,
            chunk_config=ChunkConfig(min_size=10, max_size=1000),
            sentence_split_regex=SENTENCE_SPLIT_REGEX,
        )

        chunks = chunker.split_text(text)

        assert "".join(chunks) == text
        assert len(chunks) == 3
        assert all(len(chunk) <= 1000 for chunk in chunks)
