"""Vector retrieval prerequisites for ``selected_vector_search_ontology`` context mode."""

from __future__ import annotations

from ontocast.onto.enum import OntologyContextMode
from ontocast.toolbox import ToolBox


class OntologyContextConfigError(ValueError):
    """Raised when vector-search ontology context mode is requested without a vector store."""


class VectorStoreUnavailableError(OntologyContextConfigError):
    """Raised when vector retrieval is requested but vector infra is unavailable."""

    error_code: str = "VECTOR_STORE_UNAVAILABLE"


def vector_retrieval_available(tools: ToolBox) -> bool:
    """True when a vector store (Qdrant or LanceDB) and the patch retriever are ready."""
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
        "ontology_context_mode='selected_vector_search_ontology' requires a vector "
        "store: set QDRANT_URI or LANCEDB_ENABLED=true, with an embedding "
        "dimension matching the index, so the vector store and patch retriever "
        "are available and initialized."
        f"{details}"
    )


def validate_ontology_context_mode(
    ontology_context_mode: OntologyContextMode,
    tools: ToolBox,
) -> None:
    """Raise if the requested ontology context mode cannot be satisfied."""
    if ontology_context_mode == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY:
        require_vector_retrieval(tools)


class EmptyOntologyContextError(OntologyContextConfigError):
    """Raised when a unit's ontology context resolves to zero triples.

    An empty context is not a milder version of a good one. The renderer is
    instructed to extract "based on provided domain ontology"; handed nothing,
    it falls back on whatever standard vocabulary the prompt names, and the
    SHACL gate then finds no node its shapes target -- so the run reports a
    vacuous pass over an empty focus set. Continuing produces a finished,
    plausible, ungrounded graph, which is worse than stopping.
    """

    error_code: str = "EMPTY_ONTOLOGY_CONTEXT"
