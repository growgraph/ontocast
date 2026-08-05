"""Helpers for importing optional dependencies with actionable error messages.

OntoCast ships a light core: the extraction pipeline, the in-memory triple and
vector stores, and the ontology tooling all work on a bare ``pip install
ontocast``. Everything heavier -- LLM provider SDKs, document conversion,
external vector backends, the HTTP server -- lives behind an extra.

The cost of that split is that a missing package surfaces at call time rather
than at install time, so the error has to say exactly which extra to install.
:func:`require` centralises that message.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Maps an importable module name to the extra that provides it.
_EXTRA_FOR_MODULE: dict[str, str] = {
    "langchain_openai": "openai",
    "langchain_anthropic": "anthropic",
    "langchain_google_genai": "google",
    "langchain_ollama": "ollama",
    "docling": "doc-processing",
    "docling_core": "documents",
    "qdrant_client": "qdrant",
    "fastembed": "sparse",
    "lancedb": "lancedb",
    "networkx": "graph",
    "sentence_transformers": "doc-processing",
    "pyshacl": "shacl",
    "fastapi": "server",
    "uvicorn": "server",
    "click": "server",
}


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is needed but not installed."""


def extra_for(module: str) -> str | None:
    """Return the extra that provides ``module``, if OntoCast declares one."""
    return _EXTRA_FOR_MODULE.get(module.split(".")[0])


def install_hint(module: str, *, feature: str | None = None) -> str:
    """Build the ``pip install`` guidance for a missing optional module.

    Args:
        module: The importable module name that failed to resolve.
        feature: Optional human-readable name of the capability that needs it.

    Returns:
        A one-line message naming the extra to install.
    """
    extra = extra_for(module)
    what = feature or module
    if extra is None:
        return f"{what} requires the {module!r} package, which is not installed."
    return (
        f"{what} requires the {module!r} package. "
        f'Install it with: pip install "ontocast[{extra}]"'
    )


def require(module: str, *, feature: str | None = None) -> ModuleType:
    """Import an optional module or raise with an install hint.

    Args:
        module: Fully qualified module name, e.g. ``"langchain_openai"``.
        feature: Optional human-readable name of the capability that needs it,
            used to make the error message concrete.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(install_hint(module, feature=feature)) from exc


def is_available(module: str) -> bool:
    """Return whether an optional module can be imported.

    Used for capability gating -- deciding whether to expose a tool at all --
    rather than for error reporting. Prefer :func:`require` when the caller
    genuinely needs the module.
    """
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
