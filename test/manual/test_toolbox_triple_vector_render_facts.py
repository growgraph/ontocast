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
from ontocast.api.tenancy_resolution import stores_use_tenancy_partitions
from ontocast.config import Config, LLMProvider
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import Status
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.tenancy import DEFAULT_PROJECT, DEFAULT_TENANT
from ontocast.onto.unit_states import UnitFactsState
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from ontocast.tool.vector_store.core import (
    OntologySearchHit,
    OntologySearchHitsByChannel,
)
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

FINANCE_ONTOLOGY_IRI = "https://ontocast.manual.test/ontology/fin-test"
BIOMED_ONTOLOGY_IRI = "https://ontocast.manual.test/ontology/biomed-test"

# Fixture-only classes whose labels match unique phrases in the source documents
# (deterministic vector hits independent of loose domain similarity).
FINANCE_ANCHOR_ENTITY_IRI = f"{FINANCE_ONTOLOGY_IRI}#OntoTestFinanceAnchor"
BIOMED_ANCHOR_ENTITY_IRI = f"{BIOMED_ONTOLOGY_IRI}#OntoTestBiomedAnchor"
FINANCE_ANCHOR_PHRASE = "ontotest finance retrieval anchor violet crate"
BIOMED_ANCHOR_PHRASE = "ontotest biomed retrieval anchor cobalt lantern"


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


@pytest.fixture
def biomed_ontology_ttl_text() -> str:
    """Raw Turtle for the biomed integration ontology fixture."""
    path = _FIXTURES_DIR / "biomed_integration_ontology.ttl"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def biomed_source_document_text() -> str:
    """Clinical-trial narrative; micro-chunks are sentences from this file."""
    path = _FIXTURES_DIR / "biomed_source_document.txt"
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


def _hit_counts_by_ontology_iri(
    hits_by_query: list[OntologySearchHitsByChannel],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hits in hits_by_query:
        for hit in [*hits.core_hits, *hits.neighborhood_hits, *hits.bm25_hits]:
            iri = hit.atom.ontology_iri
            if not iri:
                continue
            counts[iri] = counts.get(iri, 0) + 1
    return counts


def _majority_iri(counts: dict[str, int], candidates: frozenset[str]) -> str | None:
    """Return the IRI among ``candidates`` with the highest count, or None if empty."""
    sub = {k: counts[k] for k in candidates if k in counts}
    if not sub:
        return None
    return max(sub.items(), key=lambda item: item[1])[0]


def _focal_iris_from_hits(hits: list[OntologySearchHit]) -> list[str]:
    return [h.atom.iri for h in hits if h.atom.iri]


def _assert_expected_entity_in_hits_for_anchor_sentence(
    chunks: list[str],
    hits_by_query: list[OntologySearchHitsByChannel],
    anchor_phrase: str,
    expected_entity_iri: str,
    *,
    domain_label: str,
) -> None:
    """Require at least one top-k hit whose focal IRI matches the anchor class for that sentence."""
    for chunk, hits in zip(chunks, hits_by_query, strict=True):
        if anchor_phrase not in chunk:
            continue
        focal = _focal_iris_from_hits(
            [*hits.core_hits, *hits.neighborhood_hits, *hits.bm25_hits]
        )
        assert expected_entity_iri in focal, (
            f"{domain_label}: expected vector hit for focal entity {expected_entity_iri!r} "
            f"when querying the sentence containing {anchor_phrase!r}. "
            f"Top-k focal IRIs: {focal}"
        )
        return
    pytest.fail(
        f"{domain_label}: no sentence contained anchor phrase {anchor_phrase!r}"
    )


def _graph_mentions_entity_iri(graph: RDFGraph, entity_iri: str) -> bool:
    node = URIRef(entity_iri)
    return any(s == node or o == node for s, _, o in graph)


async def _prepare_clean_integration_stores(tools: ToolBox) -> None:
    """Empty Fuseki ontology/facts datasets and the Qdrant ontology collection."""
    triple = tools.require_triple_store_manager()
    if isinstance(triple, FusekiTripleStoreManager):
        await triple.clean()
    vs = tools.vector_store
    if vs is not None:
        col = vs.config.ontology_collection
        if col and vs.client.collection_exists(collection_name=col):
            vs.client.delete_collection(collection_name=col)
        await vs.initialize()


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


async def _bootstrap_like_server_startup(tools: ToolBox) -> None:
    """Mirror CLI startup: apply default tenancy, then initialize backends."""
    if stores_use_tenancy_partitions(tools):
        await tools.update_tenancy(DEFAULT_TENANT, DEFAULT_PROJECT)
    await tools.initialize()


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
    biomed_ontology_ttl_text: str,
    biomed_source_document_text: str,
) -> None:
    """Cross-domain manual test: clean stores, two ontologies, retrieval skew + render_facts."""
    tools = integration_tools

    # Match `ontocast/cli/server.py` initialization ordering.
    await _bootstrap_like_server_startup(tools)

    if tools.vector_store is None:
        pytest.fail("ToolBox has no vector store (configure QDRANT_URI).")
    if tools.patch_retriever is None:
        pytest.fail("ToolBox has no OntologyPatchRetriever (Qdrant not configured).")

    await _prepare_clean_integration_stores(tools)

    try:
        fin_onto = await tools.ingest_ontology_ttl(
            finance_ontology_ttl_text.encode("utf-8")
        )
        bio_onto = await tools.ingest_ontology_ttl(
            biomed_ontology_ttl_text.encode("utf-8")
        )
    except Exception as exc:  # pragma: no cover - integration diagnostic
        pytest.fail(f"ingest_ontology_ttl failed: {exc}")

    assert fin_onto.iri == FINANCE_ONTOLOGY_IRI
    assert bio_onto.iri == BIOMED_ONTOLOGY_IRI

    remote_list = await tools.require_triple_store_manager().afetch_ontologies()
    remote_iris = {o.iri for o in remote_list}
    assert FINANCE_ONTOLOGY_IRI in remote_iris
    assert BIOMED_ONTOLOGY_IRI in remote_iris

    q_top_k = tools.config.tool_config.qdrant.top_k
    iris = frozenset({FINANCE_ONTOLOGY_IRI, BIOMED_ONTOLOGY_IRI})

    fin_chunks = _split_into_sentences(finance_source_document_text)
    bio_chunks = _split_into_sentences(biomed_source_document_text)
    logger.info(
        "Finance doc: %d sentences; biomed doc: %d sentences; patch top_k=%d (QdrantConfig)",
        len(fin_chunks),
        len(bio_chunks),
        q_top_k,
    )

    fin_hits = await tools.vector_store.asearch_patch_hits_many(
        queries=fin_chunks,
        top_k=q_top_k,
    )
    fin_counts = _hit_counts_by_ontology_iri(fin_hits)
    logger.info("Finance micro-chunks — hit counts by ontology_iri: %s", fin_counts)
    dominant_fin = _majority_iri(fin_counts, iris)
    assert dominant_fin == FINANCE_ONTOLOGY_IRI, (
        "Expected most vector hits for finance sentences to come from the finance "
        f"ontology (counts={fin_counts})"
    )
    assert fin_counts.get(FINANCE_ONTOLOGY_IRI, 0) > fin_counts.get(
        BIOMED_ONTOLOGY_IRI, 0
    )
    _assert_expected_entity_in_hits_for_anchor_sentence(
        fin_chunks,
        fin_hits,
        FINANCE_ANCHOR_PHRASE,
        FINANCE_ANCHOR_ENTITY_IRI,
        domain_label="Finance",
    )

    bio_hits = await tools.vector_store.asearch_patch_hits_many(
        queries=bio_chunks,
        top_k=q_top_k,
    )
    bio_counts = _hit_counts_by_ontology_iri(bio_hits)
    logger.info("Biomed micro-chunks — hit counts by ontology_iri: %s", bio_counts)
    dominant_bio = _majority_iri(bio_counts, iris)
    assert dominant_bio == BIOMED_ONTOLOGY_IRI, (
        "Expected most vector hits for biomed sentences to come from the biomed "
        f"ontology (counts={bio_counts})"
    )
    assert bio_counts.get(BIOMED_ONTOLOGY_IRI, 0) > bio_counts.get(
        FINANCE_ONTOLOGY_IRI, 0
    )
    _assert_expected_entity_in_hits_for_anchor_sentence(
        bio_chunks,
        bio_hits,
        BIOMED_ANCHOR_PHRASE,
        BIOMED_ANCHOR_ENTITY_IRI,
        domain_label="Biomed",
    )

    document = finance_source_document_text
    micro_chunks = fin_chunks
    subgraph_depth = 1
    max_triples = 500
    # Omit top_k → uses QdrantConfig.top_k (same as explicit q_top_k above).
    stitched, source_iris = await tools.patch_retriever.aretrieve_ensemble(
        queries=micro_chunks,
        expand_sparql=True,
        subgraph_depth=subgraph_depth,
        max_total_triples=max_triples,
    )

    for idx, sentence in enumerate(micro_chunks):
        logger.info(
            "--- micro_chunk[%d/%d] chars=%d text=%r",
            idx + 1,
            len(micro_chunks),
            len(sentence),
            sentence,
        )

    logger.info(
        "Ensemble summary: sentences=%d stitched_triples=%d source_ontology_iris=%s",
        len(micro_chunks),
        len(stitched),
        source_iris,
    )

    assert len(stitched) > 0, (
        "Expected a non-empty induced subgraph (vector hits + Fuseki-backed subgraph)."
    )
    assert FINANCE_ONTOLOGY_IRI in source_iris
    assert _graph_mentions_entity_iri(stitched, FINANCE_ANCHOR_ENTITY_IRI), (
        "Ensemble retrieval should expand the finance anchor entity from Fuseki "
        f"(missing triples mentioning {FINANCE_ANCHOR_ENTITY_IRI!r})."
    )

    snapshot = Ontology(
        ontology_id=None,
        title="Stitched patch context (manual finance test)",
        description="Composite induced subgraph from vector-retrieved ontology atoms (ensemble over sentences).",
        graph=stitched,
        iri=fin_onto.iri,
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

    for iri in (FINANCE_ONTOLOGY_IRI, BIOMED_ONTOLOGY_IRI):
        try:
            await tools.delete_ontology_by_iri(iri)
        except Exception:
            pass
