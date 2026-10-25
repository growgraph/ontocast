"""One process-wide cache of *guarded* local sentence-transformer encoders.

Three independent subsystems load local sentence-transformers: semantic chunking
(:mod:`ontocast.tool.chunk.chunker`), retrieval embeddings
(:mod:`ontocast.tool.vector_store.embedding`) and entity clustering
(:mod:`ontocast.tool.agg.clustering`). Point two of them at the same checkpoint
and a per-subsystem cache means the same weights resident twice; point all three
at it and it is three times.

Caching alone is not enough, though, and getting it half-right is worse than not
sharing at all: once two subsystems hold the *same* model object, a lock that
lives in one of them protects nothing. So the cache hands out a
:class:`SharedEncoder` that owns the model **and** the lock that serialises it.
Every consumer encodes through that one guarded path, and — unlike a single
process-wide lock — two different checkpoints never serialise against each other.

Sharing weights is not the same as sharing semantics: retrieval applies
``EmbeddingConfig`` document/query prefixes and clustering and chunking do not.
The three consumers are interchangeable in what they *load*, not in what they
*mean*.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from ontocast.util.optional import require

logger = logging.getLogger(__name__)

_ENCODER_CACHE: dict[tuple[str, str | None], "SharedEncoder"] = {}
_CACHE_LOCK = threading.Lock()


class SharedEncoder:
    """A process-shared ``SentenceTransformer`` and the only way to use it.

    The model is private: every use of it *or its tokenizer* goes through this
    handle, under one lock. A HuggingFace fast tokenizer keeps its truncation
    setting as mutable state, and ``encode`` (which truncates) and
    :meth:`token_lengths` (which must not) each set it. Two threads doing so at
    once raise ``RuntimeError: Already borrowed`` from the tokenizer. The lock
    also bounds peak memory: every concurrent encode allocates its own
    activation batch, and the unit fan-out is ``PARALLEL_WORKERS`` wide.
    """

    def __init__(
        self, model_name: str, model: Any, *, device: str | None = None
    ) -> None:
        """Wrap a loaded model.

        Args:
            model_name: Checkpoint id this was loaded from.
            model: The loaded ``SentenceTransformer``.
            device: Device it was requested on; ``None`` means auto-select.
        """
        self.model_name = model_name
        self.device = device
        self._model = model
        self._lock = threading.Lock()

    @property
    def max_seq_length(self) -> int | None:
        """Tokens the model sees before truncating; ``None`` if it states none."""
        with self._lock:
            limit = getattr(self._model, "max_seq_length", None)
        return int(limit) if limit else None

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        """Encode ``texts``, serialised against other users of this model.

        Args:
            texts: Strings to encode.
            **kwargs: Passed through to ``SentenceTransformer.encode`` verbatim.
                Deliberately not normalised — the callers pass different
                arguments, and quietly reconciling them would change behaviour.

        Returns:
            Whatever ``SentenceTransformer.encode`` returns for those arguments.
        """
        with self._lock:
            return self._model.encode(texts, **kwargs)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Word pieces per text, special tokens included, without truncation.

        Args:
            texts: Strings to measure, exactly as they would be encoded.

        Returns:
            list[int]: One length per text.
        """
        with self._lock:
            encoded = self._model.tokenizer(texts, add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]


def get_shared_encoder(
    model_name: str,
    *,
    device: str | None = None,
    feature: str = "Local sentence-transformer models",
) -> SharedEncoder:
    """Return the process-wide guarded encoder for ``(model_name, device)``.

    Args:
        model_name: HuggingFace model id or local path.
        device: Device to load on; ``None`` lets sentence-transformers choose.
            Part of the cache key, so two consumers asking for the same
            checkpoint on different devices get different handles rather than
            one silently pinned to whichever loaded first.
        feature: What needs the model, used in the missing-dependency message.

    Returns:
        SharedEncoder: The shared handle.
    """
    key = (model_name, device)
    cached = _ENCODER_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        # Re-check: another thread may have loaded it while we waited, and
        # loading twice would defeat the point of the cache.
        cached = _ENCODER_CACHE.get(key)
        if cached is not None:
            return cached
        sentence_transformers = require("sentence_transformers", feature=feature)
        logger.info(
            "Loading sentence-transformer model: %s (device=%s)",
            model_name,
            device or "auto",
        )
        kwargs: dict[str, Any] = {} if device is None else {"device": device}
        model = sentence_transformers.SentenceTransformer(model_name, **kwargs)
        encoder = SharedEncoder(model_name, model, device=device)
        _ENCODER_CACHE[key] = encoder
        return encoder


class SharedSentenceTransformerEmbeddings(Embeddings):
    """LangChain ``Embeddings`` view over a :class:`SharedEncoder`.

    ``langchain_huggingface.HuggingFaceEmbeddings`` always constructs its own
    ``SentenceTransformer`` (its model config forbids extra fields, so a
    prebuilt model cannot be injected), which is why semantic chunking used to
    hold a second copy of a checkpoint the process already had resident. This
    adapter is the whole of the interface that class provided to us.
    """

    def __init__(self, encoder: SharedEncoder, *, normalize: bool = False) -> None:
        """Wrap a shared encoder.

        Args:
            encoder: The guarded encoder to delegate to.
            normalize: Whether to L2-normalise the returned vectors.
        """
        self._encoder = encoder
        self._normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` as documents.

        Args:
            texts: Strings to embed.

        Returns:
            list[list[float]]: One vector per input.
        """
        # Reproduces HuggingFaceEmbeddings._embed exactly. The newline collapse
        # is not cosmetic: semantic chunking embeds multi-sentence windows that
        # routinely contain newlines, so dropping this would feed the model
        # different strings and move chunk boundaries -- silently, in what is
        # otherwise a pure refactor.
        texts = [text.replace("\n", " ") for text in texts]
        vectors = self._encoder.encode(
            texts, show_progress_bar=False, normalize_embeddings=self._normalize
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: String to embed.

        Returns:
            list[float]: The embedding vector.
        """
        return self.embed_documents([text])[0]
