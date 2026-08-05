"""Guard the light-core import contract.

OntoCast is meant to be embeddable: ``pip install ontocast`` followed by
``from ontocast import ToolBox, ontocast_tools`` must not drag a gRPC stack, an
ONNX runtime, a document-conversion pipeline, or four competing LLM provider
SDKs into the caller's process. Every one of those lives behind an extra.

That contract is invisible to the rest of the suite, which runs under
``uv sync --all-extras`` where every optional package *is* importable. A module
that regains a module-scope ``import qdrant_client`` would pass every other test
and only break for users. These tests fail instead.

They assert on ``sys.modules`` in a *subprocess*, not on import attempts:
``langchain_core`` and ``httpx`` both probe for optional packages inside
``try/except ImportError``, so counting attempted imports over-reports.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

# Packages that must not be resolved by importing the documented entry points.
# Each names the extra that owns it, so a failure says what regressed.
FORBIDDEN: dict[str, str] = {
    "qdrant_client": "qdrant",
    "grpc": "qdrant (via qdrant-client)",
    "fastembed": "sparse",
    "onnxruntime": "sparse (via fastembed)",
    "lancedb": "lancedb",
    "docling_core": "documents",
    "docling": "doc-processing",
    "pandas": "documents (via docling-core)",
    "pyarrow": "documents (via docling-core)",
    "sentence_transformers": "doc-processing",
    "fastapi": "server",
    "starlette": "server",
    "uvicorn": "server",
    "langchain_openai": "openai",
    "langchain_anthropic": "anthropic",
    "langchain_google_genai": "google",
    "langchain_ollama": "ollama",
    "networkx": "graph",
}

_PROBE = """
import json
import sys

{import_statement}

forbidden = {forbidden!r}
loaded = sorted(name for name in forbidden if name in sys.modules)
print(json.dumps(loaded))
"""


def _modules_loaded_by(import_statement: str) -> list[str]:
    """Return which forbidden packages a fresh interpreter resolves."""
    source = _PROBE.format(
        import_statement=import_statement,
        forbidden=sorted(FORBIDDEN),
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )
    return __import__("json").loads(result.stdout.strip().splitlines()[-1])


def _assert_clean(import_statement: str) -> None:
    loaded = _modules_loaded_by(import_statement)
    if loaded:
        detail = ", ".join(f"{name} (extra: {FORBIDDEN[name]})" for name in loaded)
        pytest.fail(
            f"`{import_statement}` pulled optional packages into sys.modules: {detail}.\n"
            "Move the offending import inside the function that needs it, or resolve "
            "it through ontocast.util.optional.require()."
        )


def test_bare_import_pulls_nothing_optional() -> None:
    """`import ontocast` resolves no third-party backend at all."""
    _assert_clean("import ontocast")


def test_config_import_is_light() -> None:
    """Reading configuration must not pull a vector backend (PLANNING #68)."""
    _assert_clean("from ontocast import Config")


def test_toolbox_import_is_light() -> None:
    """The dependency container must be importable on a base install."""
    _assert_clean("from ontocast import ToolBox")


def test_agent_state_import_is_light() -> None:
    """AgentState must build without docling-core; its field is coerced lazily."""
    _assert_clean("from ontocast import AgentState")


def test_graph_import_is_light() -> None:
    """The pipeline graph builders must not require an optional backend."""
    _assert_clean("from ontocast import build_agent_graph, create_agent_graph")


def test_documented_entry_points_are_light() -> None:
    """The full public surface an embedder imports, in one interpreter."""
    _assert_clean(
        "from ontocast import ("
        "AgentState, Config, ToolBox, build_agent_graph, create_agent_graph, "
        "make_ontocast_node, ontocast_tool_names, ontocast_tools, run_unit_pipeline)"
    )
