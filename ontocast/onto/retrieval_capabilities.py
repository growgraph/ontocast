"""Vector retrieval prerequisites for per-unit ``vector_retrieval`` context mode."""

from ontocast.toolbox import ToolBox


class OntologyContextConfigError(ValueError):
    """Raised when ``ontology_context_mode=vector_retrieval`` but the toolbox lacks Qdrant."""


def vector_retrieval_available(tools: ToolBox) -> bool:
    """True when Qdrant vector store and patch retriever are both configured."""
    return (
        getattr(tools, "vector_store", None) is not None
        and getattr(tools, "patch_retriever", None) is not None
    )


def require_vector_retrieval(tools: ToolBox) -> None:
    """Raise a single canonical error if vector ensemble cannot run."""
    if vector_retrieval_available(tools):
        return
    raise OntologyContextConfigError(
        "ontology_context_mode='vector_retrieval' requires a configured Qdrant "
        "vector store (set tool qdrant.uri, matching embedding dimension) so "
        "vector_store and patch_retriever are available. See ToolBox initialization."
    )
