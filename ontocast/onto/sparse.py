"""Backend-neutral sparse vector representation.

BM25-style sparse embeddings are produced by
:class:`~ontocast.tool.vector_store.embedding.FastembedBm25SparseTool` and
consumed by whichever vector backend is configured. Modelling them locally --
rather than reusing ``qdrant_client.http.models.SparseVector`` -- keeps the
Qdrant SDK off the import path of every module that merely *passes* a sparse
vector around, and lets the LanceDB and in-memory backends work without it.

Conversion to a backend's own type happens at that backend's boundary; see
:mod:`ontocast.tool.vector_store.qdrant`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SparseVector(BaseModel):
    """A sparse vector as parallel index and value arrays.

    Attributes:
        indices: Dimension indices carrying a non-zero weight.
        values: The weight at each corresponding index.
    """

    model_config = ConfigDict(frozen=True)

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_parallel(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError(
                "SparseVector indices and values must have equal length "
                f"(got {len(self.indices)} and {len(self.values)})"
            )
        return self

    def __len__(self) -> int:
        return len(self.indices)

    def dot(self, other: "SparseVector") -> float:
        """Return the dot product with another sparse vector."""
        if not self.indices or not other.indices:
            return 0.0
        rhs = dict(zip(other.indices, other.values))
        return sum(v * rhs.get(i, 0.0) for i, v in zip(self.indices, self.values))
