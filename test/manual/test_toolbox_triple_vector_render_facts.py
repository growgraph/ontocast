"""Manual integration: ingest ontology to Fuseki + Qdrant, retrieve patches, render facts.

Run with live logs for retrieval diagnostics::

    ONTOCAST_RUN_MANUAL_TESTS=1 ... uv run pytest \\
        test/manual/test_toolbox_triple_vector_render_facts.py -v \\
        --log-cli-level=INFO
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import httpx
import pytest
from rdflib import URIRef

from ontocast.agent.render_facts import render_facts
from ontocast.config import Config, LLMProvider
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.toolbox import ToolBox

logger = logging.getLogger(__name__)

RUN_MANUAL_TESTS = os.getenv("ONTOCAST_RUN_MANUAL_TESTS", "0") == "1"

pytestmark = [
    pytest.mark.skipif(
        not RUN_MANUAL_TESTS,
        reason="Set ONTOCAST_RUN_MANUAL_TESTS=1 to run manual integration tests.",
    ),
]

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def finance_ontology_ttl_text() -> str:
    """Raw Turtle for the finance integration ontology fixture."""
    path = _FIXTURES_DIR / "finance_integration_ontology.ttl"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def finance_source_document_text() -> str:
    """Longer concrete finance narrative; micro-chunks are sentences from this file."""
    path = _FIXTURES_DIR / "finance_source_document.txt"
    return path.read_text(encoding="utf-8").strip()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        pytest.fail(f"Missing required environment variable: {name}")
    return value


def _split_into_sentences(text: str) -> list[str]:
    """Split prose into sentence-sized micro-chunks (naive English punctuation split)."""
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def _qdrant_reachable(uri: str, api_key: str | None) -> bool:
    candidates = [api_key] if api_key else [None, "abc123-qwe"]
    for candidate in candidates:
        headers = {"api-key": candidate} if candidate else None
        try:
            response = httpx.get(
                f"{uri.rstrip('/')}/collections",
                headers=headers,
                timeout=3.0,
            )
            if response.status_code == 200:
                return True
        except Exception:
            continue
    return False


def _fuseki_service_ok(base_uri: str, auth: str | None) -> bool:
    try:
        httpx_auth: httpx.Auth | None = None
        if auth and "/" in auth:
            user, _, password = auth.partition("/")
            httpx_auth = httpx.BasicAuth(user, password)

        admin_url = f"{base_uri.rstrip('/')}/$/datasets"

        response = httpx.get(
            admin_url,
            auth=httpx_auth,
            headers={"Accept": "application/json"},
            timeout=5.0,
        )

        return response.status_code == 200

    except httpx.HTTPError:
        return False


def _prepare_config() -> Config:
    _ = _require_env("LLM_PROVIDER")
    _ = _require_env("LLM_MODEL_NAME")
    provider = LLMProvider(_require_env("LLM_PROVIDER").lower())
    if provider == LLMProvider.OPENAI:
        _ = _require_env("LLM_API_KEY")
    elif provider == LLMProvider.OLLAMA:
        _ = _require_env("LLM_BASE_URL")

    _ = _require_env("ONTOCAST_WORKING_DIRECTORY")
    _ = _require_env("FUSEKI_URI")
    _ = _require_env("FUSEKI_AUTH")
    _ = _require_env("QDRANT_URI")

    cfg = Config()
    cfg.validate_llm_config()
    wd = Path(cfg.tool_config.path_config.working_directory or "").expanduser()
    od = Path(cfg.tool_config.path_config.ontology_directory or "").expanduser()
    if not wd or not od:
        pytest.fail(
            "ONTOCAST_WORKING_DIRECTORY and ONTOCAST_ONTOLOGY_DIRECTORY must be set."
        )
    wd.mkdir(parents=True, exist_ok=True)
    od.mkdir(parents=True, exist_ok=True)
    cfg.tool_config.path_config.working_directory = wd
    cfg.tool_config.path_config.ontology_directory = od
    return cfg


@pytest.fixture(scope="module")
def integration_tools() -> ToolBox:
    cfg = _prepare_config()
    fuseki_uri = cfg.tool_config.fuseki.uri
    fuseki_auth = cfg.tool_config.fuseki.auth
    qdrant_uri = cfg.tool_config.qdrant.uri
    qdrant_key = cfg.tool_config.qdrant.api_key

    if not fuseki_uri or not fuseki_auth:
        pytest.skip("FUSEKI_URI and FUSEKI_AUTH are required for this manual test.")
    if not _fuseki_service_ok(fuseki_uri, fuseki_auth):
        pytest.skip(f"Fuseki service not reachable at {fuseki_uri}")
    if not qdrant_uri or not _qdrant_reachable(qdrant_uri, qdrant_key):
        pytest.skip(f"Qdrant not reachable at {qdrant_uri}")

    return ToolBox(cfg)


@pytest.mark.anyio
async def test_ingest_retrieve_micro_chunks_render_facts(
    integration_tools: ToolBox,
    finance_ontology_ttl_text: str,
    finance_source_document_text: str,
) -> None:
    """Ingest fixture TTL to disk + Fuseki + Qdrant; sentence splits drive patch logging + render_facts."""
    tools = integration_tools
    ttl = finance_ontology_ttl_text.encode("utf-8")

    if tools.vector_store is None:
        pytest.fail("ToolBox has no vector store (configure QDRANT_URI).")
    if tools.patch_retriever is None:
        pytest.fail("ToolBox has no OntologyPatchRetriever (Qdrant not configured).")

    await tools.vector_store.initialize()

    try:
        ingested = await tools.ingest_ontology_ttl(ttl)
    except Exception as exc:  # pragma: no cover - integration diagnostic
        pytest.fail(f"ingest_ontology_ttl failed: {exc}")

    assert ingested.iri
    assert ingested.hash

    remote_list = await tools.require_triple_store_manager().afetch_ontologies()
    iris = {o.iri for o in remote_list}
    assert ingested.iri in iris, "Ingested ontology IRI not visible via triple store"

    document = finance_source_document_text
    micro_chunks = _split_into_sentences(document)
    logger.info(
        "Document has %d sentence micro-chunks (chars=%d)",
        len(micro_chunks),
        len(document),
    )

    top_k = 8
    subgraph_depth = 1
    max_triples = 500
    batches = await tools.patch_retriever.aretrieve_many(
        queries=micro_chunks,
        top_k=top_k,
        expand_sparql=True,
        subgraph_depth=subgraph_depth,
        max_triples=max_triples,
    )
    assert len(batches) == len(micro_chunks)

    stitched = RDFGraph()
    all_atom_ontology_iris: set[str] = set()
    non_empty_induced = 0

    for idx, sentence in enumerate(micro_chunks):
        patch_graph, atoms = batches[idx]
        all_atom_ontology_iris.update(a.ontology_iri for a in atoms if a.ontology_iri)
        if len(patch_graph) > 0:
            non_empty_induced += 1
        stitched += patch_graph

        logger.info(
            "--- micro_chunk[%d/%d] chars=%d text=%r",
            idx + 1,
            len(micro_chunks),
            len(sentence),
            sentence,
        )
        if not atoms:
            logger.warning(
                "    vector search returned no ontology atoms for this sentence"
            )
            continue

        for rank, atom in enumerate(atoms):
            preview = atom.core_representation.replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:217] + "..."
            logger.info(
                "    vector_match[%d] score=%s entity_iri=%s role=%s\n        core=%r",
                rank,
                atom.score,
                atom.iri,
                atom.entity_role,
                preview,
            )
            nh = atom.neighborhood_representation.replace("\n", " ")
            if len(nh) > 180:
                nh = nh[:177] + "..."
            logger.info("        neighborhood=%r", nh)

        logger.info(
            "    induced_subgraph: triples=%d ontology_iris_from_atoms=%s",
            len(patch_graph),
            sorted({a.ontology_iri for a in atoms if a.ontology_iri}),
        )

    logger.info(
        "Stitch summary: sentences=%d non_empty_induced=%d stitched_triples=%d "
        "distinct_atom_ontology_iris=%d",
        len(micro_chunks),
        non_empty_induced,
        len(stitched),
        len(all_atom_ontology_iris),
    )

    assert non_empty_induced > 0, (
        "Expected at least one sentence to expand to a non-empty induced subgraph "
        "(vector hit + Fuseki-backed subgraph)."
    )
    assert ingested.iri in all_atom_ontology_iris, (
        "Expected vector hits to include atoms from the ingested ontology IRI"
    )

    snapshot = Ontology(
        ontology_id=None,
        title="Stitched patch context (manual finance test)",
        description="Composite induced subgraph from vector-retrieved ontology atoms per sentence.",
        graph=stitched,
        iri=ingested.iri,
        current_domain=tools.config.tool_config.domain.current_domain,
    )

    unit = ContentUnit(
        text=document,
        index=0,
        doc_iri=URIRef("https://ontocast.manual.test/doc/finance/fixture"),
    )
    state = UnitFactsState(
        content_unit=unit,
        ontology_snapshot=snapshot,
        facts_user_instruction=(
            "Use ONLY classes and properties from the domain ontology when typing facts. "
            "Link mentions to the closest ontology IRIs. Prefer concrete facility/covenant terms."
        ),
    )

    result = await render_facts(state, tools.get_atomic_tools())
    assert result.failure_stage is None
    assert result.status == Status.SUCCESS
    assert result.budget_tracker.calls_count > 0

    out_ttl = result.content_unit.graph.serialize(format="turtle")
    assert "ontocast.manual.test" in out_ttl
    assert len(result.content_unit.graph) > 0

    try:
        await tools.delete_ontology_by_iri(ingested.iri)
    except Exception:
        pass
