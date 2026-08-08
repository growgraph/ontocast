"""One guarded encoder per checkpoint, shared across every local-embedding consumer.

Three subsystems load local sentence-transformers — semantic chunking, retrieval
embeddings and entity clustering. Sharing the weights is the memory win; sharing
the *lock* is what makes the sharing safe. Getting only the first half is worse
than not sharing at all: two subsystems holding one model object with a lock in
just one of them is a lock that protects nothing.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from ontocast.config import ChunkConfig, EmbeddingConfig
from ontocast.tool import sentence_transformer
from ontocast.tool.agg.clustering import EntityClusterer
from ontocast.tool.chunk import chunker as chunker_module
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.sentence_transformer import (
    SharedSentenceTransformerEmbeddings,
    get_shared_encoder,
)
from ontocast.tool.vector_store.embedding import HuggingFaceEmbeddingTool


@pytest.fixture
def clean_encoder_cache():
    """Isolate the process-wide encoder cache for one test.

    Not autouse: legitimate cross-test sharing elsewhere should be unaffected.
    """
    sentence_transformer._ENCODER_CACHE.clear()
    yield
    sentence_transformer._ENCODER_CACHE.clear()


class _FakeModel:
    """Stand-in for SentenceTransformer that records what it was asked to encode."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.seen: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        self.seen.append(list(texts))
        return np.ones((len(texts), 4), dtype=float)


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_cls: type = _FakeModel,
) -> list[str]:
    """Patch the lazy import so no real weights load; return the created-name log."""
    created: list[str] = []

    def _build(model_name: str, **kwargs: Any) -> Any:
        # A factory rather than a subclass: the code under test only ever calls
        # SentenceTransformer, never subclasses or isinstance-checks it.
        created.append(model_name)
        return model_cls(model_name, **kwargs)

    def _fake_require(name: str, *, feature: str = "") -> SimpleNamespace | ModuleType:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=_build)
        return importlib.import_module(name)

    monkeypatch.setattr(sentence_transformer, "require", _fake_require)
    return created


def test_all_three_consumers_share_one_model(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    created = _install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(chunker_module, "_embedding_model_available", lambda: True)

    retrieval = HuggingFaceEmbeddingTool(config=EmbeddingConfig(model_name="shared"))
    clusterer = EntityClusterer(embedding_model="shared")
    chunk_tool = ChunkerTool(chunk_config=ChunkConfig(embedding_model="shared"))

    from_retrieval = retrieval._get_embedder()
    from_clustering = clusterer.embedder
    chunk_embeddings = chunk_tool.embeddings()

    assert chunk_embeddings is not None
    assert from_retrieval is from_clustering
    assert chunk_embeddings._encoder is from_retrieval
    assert created == ["shared"], f"expected one load, got {created}"


def test_distinct_names_get_distinct_encoders_and_locks(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    created = _install_fake_sentence_transformers(monkeypatch)

    first = get_shared_encoder("model-a")
    second = get_shared_encoder("model-b")

    assert first is not second
    assert first.model is not second.model
    # The point of per-handle locks: a single process-wide lock would make two
    # unrelated checkpoints queue behind one another for nothing.
    assert first._lock is not second._lock
    assert created == ["model-a", "model-b"]


def test_device_is_part_of_the_cache_key(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    created = _install_fake_sentence_transformers(monkeypatch)

    auto = get_shared_encoder("m")
    pinned = get_shared_encoder("m", device="cpu")

    assert auto is not pinned
    assert auto.model.kwargs == {}
    assert pinned.model.kwargs == {"device": "cpu"}
    assert created == ["m", "m"]


def test_encodes_serialize_per_model(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """Two threads must not be inside the same model's encode() at once."""
    barrier = threading.Barrier(2, timeout=0.3)

    class _Rendezvous(_FakeModel):
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            barrier.wait()
            return super().encode(texts, **kwargs)

    _install_fake_sentence_transformers(monkeypatch, model_cls=_Rendezvous)
    encoder = get_shared_encoder("m")
    errors: list[BaseException] = []

    def run() -> None:
        try:
            encoder.encode(["x"])
        except BaseException as exc:  # noqa: BLE001 - recording for assertion
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The barrier can only break if the second thread never got in — which is
    # exactly the mutual exclusion under test. It cannot pass by accident.
    assert errors, "both threads entered encode() concurrently; the lock is not held"
    assert all(isinstance(exc, threading.BrokenBarrierError) for exc in errors)


def test_encodes_do_not_serialize_across_models(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """Different checkpoints must run concurrently — the global-lock regression."""
    barrier = threading.Barrier(2, timeout=5.0)

    class _Rendezvous(_FakeModel):
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            barrier.wait()
            return super().encode(texts, **kwargs)

    _install_fake_sentence_transformers(monkeypatch, model_cls=_Rendezvous)
    first = get_shared_encoder("model-a")
    second = get_shared_encoder("model-b")
    errors: list[BaseException] = []

    def run(encoder) -> None:
        try:
            encoder.encode(["x"])
        except BaseException as exc:  # noqa: BLE001 - recording for assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(first,)),
        threading.Thread(target=run, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"unrelated checkpoints serialized against each other: {errors}"


def test_serialize_false_leaves_encodes_unguarded(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """The GPU opt-out, where concurrency is parallelism rather than contention."""
    barrier = threading.Barrier(2, timeout=5.0)

    class _Rendezvous(_FakeModel):
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            barrier.wait()
            return super().encode(texts, **kwargs)

    _install_fake_sentence_transformers(monkeypatch, model_cls=_Rendezvous)
    encoder = get_shared_encoder("m", serialize=False)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            encoder.encode(["x"])
        except BaseException as exc:  # noqa: BLE001 - recording for assertion
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors


def test_adapter_satisfies_semantic_chunker_contract(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """SemanticChunker calls exactly one method; drive that line directly.

    Constructed via __new__ so this stays offline — importing the real class
    would pull umap and hdbscan.
    """
    from ontocast.tool.chunk.util import SemanticChunker

    _install_fake_sentence_transformers(monkeypatch)
    adapter = SharedSentenceTransformerEmbeddings(get_shared_encoder("m"))

    assert isinstance(adapter, Embeddings)

    splitter = SemanticChunker.__new__(SemanticChunker)
    splitter.embeddings = adapter
    vectors = SemanticChunker._get_embeddings(splitter, ["a", "b", "c"])

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 4)
    assert vectors.dtype.kind == "f"

    query = adapter.embed_query("x")
    assert isinstance(query, list)
    assert all(isinstance(value, float) for value in query)


def test_adapter_collapses_newlines_like_langchain(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """Guards the bit-identity claim against langchain_huggingface._embed.

    Semantic chunking embeds multi-sentence windows that contain newlines, so
    dropping this collapse would move chunk boundaries in what is meant to be a
    pure refactor.
    """
    _install_fake_sentence_transformers(monkeypatch)
    encoder = get_shared_encoder("m")
    adapter = SharedSentenceTransformerEmbeddings(encoder)

    adapter.embed_documents(["a\nb", "plain"])

    assert encoder.model.seen == [["a b", "plain"]]


def test_chunker_probe_requires_clustering_dependencies(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """A model alone is not semantic chunking — util.py needs hdbscan and umap."""
    _install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(chunker_module, "_embedding_model_available", lambda: True)
    monkeypatch.setattr(chunker_module, "_semantic_chunking_available", lambda: False)

    tool = ChunkerTool(chunk_config=ChunkConfig(embedding_model="m"))

    assert tool.chunking_mode == "naive"
    # Schema detection only needs the model, so it must keep working.
    assert tool.embed_texts(["heading"]) == [[1.0, 1.0, 1.0, 1.0]]


def test_chunker_embed_texts_returns_none_without_a_model(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    """schema_detect.TextEmbedder expects None, not an exception."""
    monkeypatch.setattr(chunker_module, "_embedding_model_available", lambda: False)
    monkeypatch.setattr(chunker_module, "_semantic_chunking_available", lambda: False)

    tool = ChunkerTool(chunk_config=ChunkConfig(embedding_model="m"))

    assert tool.embed_texts(["heading"]) is None
    assert tool.embed_texts([]) == []


def test_failed_model_load_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache
) -> None:
    attempts: list[int] = []

    def _boom(name: str, *, feature: str = ""):
        attempts.append(1)
        raise RuntimeError("no such checkpoint")

    monkeypatch.setattr(sentence_transformer, "require", _boom)
    monkeypatch.setattr(chunker_module, "_embedding_model_available", lambda: True)

    tool = ChunkerTool(chunk_config=ChunkConfig(embedding_model="missing"))

    assert tool.embeddings() is None
    assert tool.embeddings() is None
    assert len(attempts) == 1, "a failed load must be recorded, not retried per call"


def test_chunk_cache_key_tracks_the_configured_model(
    monkeypatch: pytest.MonkeyPatch, clean_encoder_cache, tmp_path
) -> None:
    from ontocast.tool.cache import Cacher

    monkeypatch.setattr(chunker_module, "_embedding_model_available", lambda: False)
    monkeypatch.setattr(chunker_module, "_semantic_chunking_available", lambda: False)

    cache = Cacher(cache_dir=tmp_path)
    text = "One sentence. Another sentence."

    first = ChunkerTool(
        chunk_config=ChunkConfig(embedding_model="model-a"), cache=cache
    )
    second = ChunkerTool(
        chunk_config=ChunkConfig(embedding_model="model-b"), cache=cache
    )

    first(text)
    # A different model must miss rather than silently reuse chunks produced by
    # a different embedding geometry.
    assert second.cache.get(text, config=_cache_config(second)) is None
    assert first.cache.get(text, config=_cache_config(first)) is not None


def _cache_config(tool: ChunkerTool) -> dict[str, Any]:
    return {
        "model": tool.config.embedding_model,
        "chunking_mode": tool.chunking_mode,
        "max_size": tool.config.max_size,
        "min_size": tool.config.min_size,
    }
