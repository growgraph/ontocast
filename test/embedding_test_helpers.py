"""Helpers for keeping tests off the real sentence-transformer models.

``EntityClusterer.embedder`` is a lazy property that constructs a real
``SentenceTransformer`` on first access (``ontocast/tool/agg/clustering.py``).
Patching ``clusterer.embedder.encode`` therefore *loads the model first* and only
then replaces the method -- the download and the ~470 MB resident model are paid
in full by a test that intended to use a fake. Assign the cached slot instead.

Only ``.encode`` is ever called on the embedder (``embed_representations``), so a
namespace carrying that one attribute is a complete stand-in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = ["install_fake_encoder", "distinct_encoder"]


def install_fake_encoder(
    clusterer: Any,
    encode: Callable[..., Sequence[np.ndarray] | np.ndarray],
) -> None:
    """Give ``clusterer`` a fake embedder without materializing the real one.

    Args:
        clusterer: An ``EntityClusterer`` (or anything exposing ``_embedder``).
        encode: Stand-in for ``SentenceTransformer.encode``; receives the batch of
            texts and arbitrary keyword arguments.
    """
    clusterer._embedder = SimpleNamespace(encode=encode)


def distinct_encoder() -> Callable[..., np.ndarray]:
    """An ``encode`` stand-in giving every text its own orthogonal basis vector.

    Cosine similarity between any two entities is then 0, so the real
    ``cluster_by_similarity`` code path still runs but nothing merges on embedding
    proximity. Use this where a test asserts something structural (namespaces, URI
    style, what the selector was handed) and merging is incidental.

    A test that asserts entities *do* or *must not* merge should stub
    ``clusterer.cluster_entities`` outright instead, so the grouping under test is
    stated in the test rather than inherited from a model.
    """

    def _encode(texts: Sequence[str], **_kwargs: Any) -> np.ndarray:
        return np.eye(len(texts), dtype=float)

    return _encode
