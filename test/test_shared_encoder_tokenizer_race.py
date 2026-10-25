"""Token counting and encoding share one fast tokenizer; both must hold one lock.

A HuggingFace fast tokenizer keeps its truncation setting as mutable state in
Rust. ``encode`` tokenizes with truncation, token counting without, so each call
flips that state -- and flipping it while another thread is mid-call raises
``RuntimeError: Already borrowed`` (transformers 4.x fast tokenizers). Loads a
real checkpoint, hence ``slow``.
"""

from __future__ import annotations

import threading

import pytest

from ontocast.config import EmbeddingConfig
from ontocast.tool import sentence_transformer
from ontocast.tool.vector_store.embedding import HuggingFaceEmbeddingTool

pytestmark = pytest.mark.slow

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COUNTS = 2000
_ENCODERS = 3


@pytest.fixture
def clean_encoder_cache():
    sentence_transformer._ENCODER_CACHE.clear()
    yield
    sentence_transformer._ENCODER_CACHE.clear()


def test_token_counting_during_encode_does_not_race(clean_encoder_cache) -> None:
    tool = HuggingFaceEmbeddingTool(config=EmbeddingConfig(model_name=_MODEL))
    # Longer than max_seq_length, so counting really does differ from encoding.
    long_texts = ["perovskite film aging under humidity " * 40] * 64
    errors: list[BaseException] = []
    done = threading.Event()

    def encode() -> None:
        while not done.is_set():
            try:
                tool.embed_query(["short query"] * 4)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                errors.append(exc)
                done.set()

    def count() -> None:
        try:
            for _ in range(_COUNTS):
                if done.is_set():
                    return
                # None is a failure too: a local tokenizer always yields counts.
                if tool.token_lengths(long_texts) is None:
                    raise AssertionError("token counting failed under contention")
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)
        finally:
            done.set()

    threads = [threading.Thread(target=encode) for _ in range(_ENCODERS)]
    threads.append(threading.Thread(target=count))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent tokenizer use failed: {errors[0]!r}"
    assert tool.count_over_limit(long_texts) == len(long_texts)
