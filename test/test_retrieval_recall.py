"""Recall measurement for vector-mode ontology patch retrieval.

Unlike the rest of the vector suite, this module uses **real** embeddings and a **real**
Qdrant collection. Hash-based fake vectors make recall numbers meaningless, and recall is
precisely what was never measured: the plumbing tests assert ordering, counts, and
parameter pass-through, but nothing asserts that a relevant catalog term survives to the
prompt snapshot.

Two numbers are reported per run:

* **seed recall** — the expected term reached ``atoms_final`` (survived vector search,
  cross-window merge, per-ontology round-robin, and the window-scaled cap).
* **snapshot recall** — the expected term is *defined* in the returned graph (also
  survived induced-subgraph expansion: BFS quotas, budget caps, component pruning).

The gap between them attributes losses to the graph stage rather than the vector stage.
Both are printed as a funnel alongside the metrics the pipeline already emits, so a
regression can be localised without bisecting.

Scale and corpus are environment-controlled so the same harness serves CI (small, fast)
and tuning sweeps / embedding bake-offs (large):

* ``ONTOCAST_RECALL_ROOT``        — Text2KGBench corpus root
* ``ONTOCAST_RECALL_ONTOLOGIES``  — ontologies loaded into the catalog (default 6)
* ``ONTOCAST_RECALL_CASES``       — cases per ontology (default 15)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Generator
from typing import Any

import pytest

from ontocast.config import (
    Config,
    EmbeddingConfig,
    FusekiConfig,
    LLMConfig,
    PathConfig,
    QdrantConfig,
    ToolConfig,
)
from ontocast.onto.ontology import Ontology
from ontocast.toolbox import ToolBox
from test.qdrant_util import qdrant_reachable
from test.retrieval_gt import (
    RecallCase,
    StageCounts,
    graph_defines,
    load_anchor_cases,
    load_text2kgbench,
    text2kgbench_root,
)

pytestmark = pytest.mark.integration


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@pytest.fixture
def recall_qdrant_config() -> Generator[QdrantConfig, Any, None]:
    """Dedicated Qdrant collections for one recall run.

    Deliberately *not* the shared ``qdrant_session_test_context``: that fixture is used by
    smoke tests that create 8-dimensional collections, and the embedding contract would
    reject the real 384-dimensional model against those.

    Function-scoped so each tier indexes only its own ontologies; sharing a collection let
    one tier's catalog leak into the other's ``seeds_by_ontology`` and made the measured
    ontology count depend on test order.
    """
    from qdrant_client import QdrantClient

    base = QdrantConfig()
    if base.uri is None:
        pytest.skip("QDRANT_URI not configured")
    if not qdrant_reachable(uri=base.uri, api_key=base.api_key):
        pytest.skip(f"Qdrant not reachable at {base.uri}")

    run_id = uuid.uuid4().hex[:8]
    config = base.model_copy(
        update={
            "ontology_collection": f"ontocast_recall_{run_id}_ontologies",
            "facts_collection": f"ontocast_recall_{run_id}_facts",
        }
    )

    yield config

    client = QdrantClient(
        url=config.uri,
        api_key=config.api_key,
        grpc_port=config.grpc_port,
        prefer_grpc=config.use_grpc,
    )
    for name in (config.ontology_collection, config.facts_collection):
        if name and client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)


def _build_toolbox(qdrant_config: QdrantConfig, tmp_path_factory: Any) -> ToolBox:
    """ToolBox wired to real embeddings, real Qdrant, in-memory triple store."""
    workspace = tmp_path_factory.mktemp("recall_workspace")
    ontology_dir = workspace / "ontologies"
    ontology_dir.mkdir()
    tool_config = ToolConfig(
        llm_config=LLMConfig(),
        path_config=PathConfig(
            working_directory=workspace,
            ontology_directory=ontology_dir,
        ),
        # EmbeddingConfig() resolves the configured production model, not a fake.
        embedding=EmbeddingConfig(),
        qdrant=qdrant_config,
        # Isolate from host Fuseki so the in-memory triple store is used.
        fuseki=FusekiConfig(uri=None, auth=None),
    )
    return ToolBox(Config(tool_config=tool_config))


async def _index(tools: ToolBox, ontologies: list[Ontology]) -> None:
    """Bring the vector store up and index every ontology into the catalog."""
    assert tools.vector_store is not None
    await tools.vector_store.initialize()
    tools.vector_store_ready = True
    tools.vector_store_last_error = None
    for ontology in ontologies:
        ttl = ontology.graph.serialize(format="turtle").encode("utf-8")
        await tools.ingest_ontology_ttl(ttl)


async def _score(tools: ToolBox, cases: list[RecallCase]) -> StageCounts:
    """Run every case through the production retrieval path and fold the outcomes."""
    retriever = tools.patch_retriever
    assert retriever is not None
    store_config = tools.config.tool_config.vector_store
    counts = StageCounts()

    for case in cases:
        graph, _sources = await retriever.aretrieve_ensemble(
            queries=[case.text],
            top_k=store_config.top_k,
            expand_sparql=True,
            subgraph_depth=store_config.induced_subgraph_depth,
            max_total_triples=store_config.induced_subgraph_max_total_triples,
            estimated_triples_per_query=(
                store_config.induced_subgraph_estimated_triples_per_query
            ),
        )
        metrics = dict(retriever.last_retrieval_metrics)
        seed_iris = {str(iri) for iri in metrics.get("seed_iris", [])}
        seed_hit = bool(seed_iris & set(case.expected_iris))
        snapshot_hit = graph_defines(graph, case.expected_iris)
        subjects = {
            str(subject)
            for subject in graph.subjects()
            if str(subject).startswith("http")
        }
        on_topic = sum(1 for s in subjects if s.startswith(case.ontology_iri))
        counts.observe(
            seed_hit=seed_hit,
            snapshot_hit=snapshot_hit,
            metrics=metrics,
            on_topic_subjects=on_topic,
            total_subjects=len(subjects),
        )

    return counts


def _run(tools: ToolBox, ontologies: list[Ontology], cases: list[RecallCase]):
    async def _main() -> StageCounts:
        await _index(tools, ontologies)
        return await _score(tools, cases)

    return asyncio.run(_main())


def test_anchor_recall(
    recall_qdrant_config: QdrantConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """In-repo anchor fixtures: near-verbatim label matches across two ontologies.

    These labels appear almost literally in the source text, so this is close to a
    best case. Failure here indicates a plumbing or lexical-lane defect rather than a
    weak embedding model.
    """
    cases, ontologies = load_anchor_cases()
    if not cases:
        pytest.skip("anchor fixtures unavailable")

    tools = _build_toolbox(recall_qdrant_config, tmp_path_factory)
    counts = _run(tools, ontologies, cases)

    report = counts.render("anchor recall")
    print(f"\n{report}")

    assert counts.cases > 0
    assert counts.seed_recall > 0.0, (
        "no anchor term reached the seed set; retrieval is not functioning at all\n"
        + report
    )


@pytest.mark.slow
def test_text2kgbench_recall(
    recall_qdrant_config: QdrantConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Text2KGBench: real prose against a multi-ontology catalog.

    Ground truth is derived, not hand-labelled: each row's relation labels resolve to
    ontology IRIs, so every case is an unambiguous retrieval target. Loading several
    ontologies at once also stresses per-ontology seed allocation and multi-component
    snapshot assembly, which two fixtures cannot reproduce.
    """
    root = text2kgbench_root()
    if root is None:
        pytest.skip(
            "Text2KGBench corpus not found; set ONTOCAST_RECALL_ROOT to its root"
        )

    cases, ontologies = load_text2kgbench(
        root,
        max_ontologies=_env_int("ONTOCAST_RECALL_ONTOLOGIES", 6),
        max_cases_per_ontology=_env_int("ONTOCAST_RECALL_CASES", 15),
    )
    if not cases:
        pytest.skip("Text2KGBench corpus present but yielded no resolvable cases")

    tools = _build_toolbox(recall_qdrant_config, tmp_path_factory)
    counts = _run(tools, ontologies, cases)

    report = counts.render(
        f"text2kgbench recall ({len(ontologies)} ontologies, {len(cases)} cases)"
    )
    print(f"\n{report}")

    assert counts.cases > 0
    assert counts.seed_recall > 0.0, (
        "no expected term reached the seed set across the whole corpus\n" + report
    )
