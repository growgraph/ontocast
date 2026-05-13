"""Vector retrieval prerequisites for ``selected_vector_search_ontology`` context mode."""

from __future__ import annotations

from ontocast.onto.enum import OntologyContextMode
from ontocast.toolbox import ToolBox


class OntologyContextConfigError(ValueError):
    """Raised when vector-search ontology context mode is requested but Qdrant is missing."""


class VectorStoreUnavailableError(OntologyContextConfigError):
    """Raised when vector retrieval is requested but vector infra is unavailable."""

    error_code: str = "VECTOR_STORE_UNAVAILABLE"


def vector_retrieval_available(tools: ToolBox) -> bool:
    """True when Qdrant vector store and patch retriever are both configured."""
    return (
        tools.vector_store is not None
        and tools.patch_retriever is not None
        and tools.is_vector_store_ready()
    )


def require_vector_retrieval(tools: ToolBox) -> None:
    """Raise a single canonical error if vector ensemble cannot run."""
    if vector_retrieval_available(tools):
        return
    last_error = tools.vector_store_last_error
    details = ""
    if last_error is not None:
        details = f" Last vector-store init error: {last_error}"
    raise VectorStoreUnavailableError(
        "ontology_context_mode='selected_vector_search_ontology' requires a configured Qdrant "
        "vector store (set tool qdrant.uri, matching embedding dimension) so "
        "vector_store and patch_retriever are available and initialized."
        f"{details}"
    )


def validate_ontology_context_mode(
    ontology_context_mode: OntologyContextMode,
    tools: ToolBox,
) -> None:
    """Raise if the requested ontology context mode cannot be satisfied."""
    if ontology_context_mode == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY:
        require_vector_retrieval(tools)
