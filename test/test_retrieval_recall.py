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

* ``ONTOCAST_RECALL_CORPUS``      — prebuilt corpus directory (``cases.jsonl`` +
  ``ontologies/``), the domain-neutral tier; build one with
  ``ontocast-validation/run/build_recall_corpus.py``
* ``ONTOCAST_RECALL_ROOT``        — Text2KGBench corpus root
* ``ONTOCAST_RECALL_ONTOLOGIES``  — ontologies loaded into the catalog (default 6);
  Text2KGBench tier only
* ``ONTOCAST_RECALL_CASES``       — cases per ontology (default 15); Text2KGBench tier
  only

Ablation controls, for asking whether indexing a large external vocabulary (QUDT's 2,575
units, say) helps or dilutes. Both are opt-in and inert when unset:

* ``ONTOCAST_RECALL_EXTRA_ONTOLOGIES`` — ``os.pathsep``-separated .ttl files/directories
  appended to the corpus catalog. Keeps the *index* axis a one-variable flip while the
  corpus on disk stays byte-identical across arms.
* ``ONTOCAST_RECALL_COLLECTION_SUFFIX`` — pin the Qdrant collection name and skip
  teardown, so it can be reused. With ``ONTOCAST_RECALL_SKIP_INDEX=1`` a later arm
  scores against it without re-embedding. Everything on the *retrieval* axis
  (``ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA``, merge mode, caps) is applied at merge
  time, so a whole sweep needs one index.

Case text is split into proposition windows exactly as production does, so a passage
spanning several sentences issues several queries rather than one.
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
from ontocast.tool.chunk.proposition import split_proposition_windows
from ontocast.toolbox import ToolBox
from test.qdrant_util import qdrant_reachable
from test.retrieval_gt import (
    RecallCase,
    StageCounts,
    corpus_root,
    load_anchor_cases,
    load_corpus,
    load_text2kgbench,
    owner_index,
    owner_of,
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

    # A pinned suffix names a reusable collection instead of a throwaway one, so a
    # config sweep (per-ontology quota, merge mode, caps -- all applied at merge time)
    # scores against an index built once. Indexing a large external vocabulary costs
    # minutes; re-paying that per arm is the difference between a tight loop and an
    # afternoon. Unset means the original per-run uuid, deleted on teardown.
    pinned = os.getenv("ONTOCAST_RECALL_COLLECTION_SUFFIX", "").strip()
    run_id = pinned or uuid.uuid4().hex[:8]
    config = base.model_copy(
        update={
            "ontology_collection": f"ontocast_recall_{run_id}_ontologies",
            "facts_collection": f"ontocast_recall_{run_id}_facts",
        }
    )

    yield config

    if pinned:
        return

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
    """Bring the vector store up and index every ontology into the catalog.

    ``ONTOCAST_RECALL_SKIP_INDEX=1`` reuses whatever a pinned collection already holds.
    Only meaningful with ``ONTOCAST_RECALL_COLLECTION_SUFFIX``; the catalog itself is
    still registered in-process, so scoring and attribution are unaffected.
    """
    assert tools.vector_store is not None
    await tools.vector_store.initialize()
    tools.vector_store_ready = True
    tools.vector_store_last_error = None
    skip = os.getenv("ONTOCAST_RECALL_SKIP_INDEX", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    for ontology in ontologies:
        if skip:
            # Register in the catalog without re-embedding: the vectors are already in
            # the pinned collection from the arm that built it. The triple store is
            # in-memory per run, so the graph stage still needs the ontology written —
            # without it every induced subgraph is silently empty.
            if tools.triple_store_manager is not None:
                await tools.triple_store_manager.aserialize(ontology)
            tools.ontology_manager.add_ontology(ontology, skip_vector_index=True)
            continue
        ttl = ontology.graph.serialize(format="turtle").encode("utf-8")
        await tools.ingest_ontology_ttl(ttl)


async def _score(
    tools: ToolBox, ontologies: list[Ontology], cases: list[RecallCase]
) -> StageCounts:
    """Run every case through the production retrieval path and fold the outcomes."""
    retriever = tools.patch_retriever
    assert retriever is not None
    store_config = tools.config.tool_config.vector_store
    counts = StageCounts()
    owners = owner_index(ontologies)

    for case in cases:
        # Window exactly as production does, so multi-sentence cases exercise the
        # cross-window merge and the window budget. A single sentence yields one
        # window, leaving the sentence-level tiers unchanged.
        queries = split_proposition_windows(
            case.text,
            max_sentences=store_config.proposition_window_sentences,
            max_windows=store_config.proposition_max_windows,
        )
        graph, _sources = await retriever.aretrieve_ensemble(
            queries=queries,
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
        # Subject position, not mere mention: pruning drops triples by subject but leaves
        # object-position references, so a named term may carry no usable definition.
        subjects = {str(subject) for subject in graph.subjects()}
        http_subjects = {s for s in subjects if s.startswith("http")}
        on_topic = sum(1 for s in http_subjects if s.startswith(case.ontology_iri))
        counts.observe(
            expected=case.expected_iris,
            seed_iris=seed_iris,
            snapshot_subjects=subjects,
            metrics=metrics,
            on_topic_subjects=on_topic,
            total_subjects=len(http_subjects),
            expected_owner={
                iri: owner
                for iri in case.expected_iris
                if (owner := owner_of(iri, owners))
            },
        )

    return counts


def _run(tools: ToolBox, ontologies: list[Ontology], cases: list[RecallCase]):
    async def _main() -> StageCounts:
        await _index(tools, ontologies)
        return await _score(tools, ontologies, cases)

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
def test_corpus_recall(
    recall_qdrant_config: QdrantConfig,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A prebuilt domain corpus: multi-sentence passages against a linked catalog.

    The other two tiers score one sentence at a time against mutually disjoint
    ontologies, which makes the cross-window merge a no-op and hides both the window
    budget and cross-ontology expansion. A corpus built by
    ``ontocast-validation/run/build_recall_corpus.py`` supplies passages long enough to
    produce several proposition windows and a catalog whose ontologies reference each
    other.
    """
    root = corpus_root()
    if root is None:
        pytest.skip(
            "no recall corpus; set ONTOCAST_RECALL_CORPUS to a directory holding "
            "cases.jsonl and ontologies/"
        )

    cases, ontologies = load_corpus(root)
    tools = _build_toolbox(recall_qdrant_config, tmp_path_factory)
    counts = _run(tools, ontologies, cases)

    report = counts.render(
        f"corpus recall: {root.name} ({len(ontologies)} ontologies, {len(cases)} cases)"
    )
    print(f"\n{report}")

    assert counts.cases > 0
    assert counts.seed_recall > 0.0, (
        "no expected term reached the seed set across the whole corpus\n" + report
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
