"""Toolbox wiring and scoring loop shared by the recall tests and the sweep.

The pytest tiers in :mod:`test.test_retrieval_recall` answer "is retrieval working";
:mod:`test.retrieval_sweep` answers "which parameters work best". Both need the same
three things -- a toolbox on real embeddings, an indexed catalog, and one scoring pass
over a corpus -- so they live here rather than being duplicated, where they would drift
and quietly stop measuring the same pipeline.

**No LLM call is made anywhere in this module.** Retrieval is embeddings and graph work
only. The toolbox still constructs an LLM client eagerly, so the provider is pinned to
one that needs no API key.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from ontocast.config import (
    Config,
    EmbeddingConfig,
    FusekiConfig,
    LanceDBConfig,
    LLMConfig,
    PathConfig,
    QdrantConfig,
    ToolConfig,
)
from ontocast.config.settings import LLMProvider
from ontocast.onto.ontology import Ontology
from ontocast.tool.chunk.proposition import split_proposition_windows
from ontocast.toolbox import ToolBox
from test.retrieval_gt import RecallCase, StageCounts, owner_index, owner_of

COLLECTION_SUFFIX_ENV = "ONTOCAST_RECALL_COLLECTION_SUFFIX"
SKIP_INDEX_ENV = "ONTOCAST_RECALL_SKIP_INDEX"


def pinned_suffix() -> str:
    """Collection/table suffix pinning a reusable index, or ``""``."""
    return os.getenv(COLLECTION_SUFFIX_ENV, "").strip()


def skip_index() -> bool:
    """Whether to reuse a pinned index instead of re-embedding the catalog."""
    return os.getenv(SKIP_INDEX_ENV, "").strip().lower() in {"1", "true", "yes"}


def lancedb_store_config(data_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    """``ToolConfig`` overrides naming an embedded LanceDB index.

    Args:
        data_dir: Directory handed to ``lancedb.connect``.
        run_id: Table-name suffix; a pinned suffix, else a fresh one.

    Returns:
        dict[str, Any]: ``lancedb`` and ``qdrant`` keyword overrides.
    """
    suffix = run_id or pinned_suffix() or uuid.uuid4().hex[:8]
    config = LanceDBConfig(enabled=True).model_copy(
        update={
            "data_dir": data_dir,
            "ontology_table": f"ontocast_recall_{suffix}_ontologies",
            "facts_table": f"ontocast_recall_{suffix}_facts",
        }
    )
    return {"lancedb": config, "qdrant": QdrantConfig(uri=None)}


def build_toolbox(store_config: dict[str, Any], ontology_dir: Path) -> ToolBox:
    """ToolBox on real embeddings, a real vector store, in-memory triple store."""
    tool_config = ToolConfig(
        # No LLM call is ever made, but the runtime builds a client eagerly and the
        # default provider refuses to construct without an API key. Ollama needs
        # none, so this stays runnable with no credentials in the environment.
        llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
        path_config=PathConfig(ontology_directory=ontology_dir),
        # EmbeddingConfig() resolves the configured production model, not a fake.
        embedding=EmbeddingConfig(),
        # Isolate from host Fuseki so the in-memory triple store is used.
        fuseki=FusekiConfig(uri=None, auth=None),
        **store_config,
    )
    return ToolBox(Config(tool_config=tool_config))


async def index_catalog(tools: ToolBox, ontologies: list[Ontology]) -> None:
    """Bring the vector store up and index every ontology into the catalog.

    ``ONTOCAST_RECALL_SKIP_INDEX=1`` reuses whatever a pinned index already holds.
    Only meaningful with a pinned suffix; the catalog itself is still registered
    in-process, so scoring and attribution are unaffected.
    """
    assert tools.vector_store is not None
    await tools.vector_store.initialize()
    tools.vector_store_ready = True
    tools.vector_store_last_error = None
    reuse = skip_index()
    for ontology in ontologies:
        if reuse:
            # Register in the catalog without re-embedding: the vectors are already
            # in the pinned index from the arm that built it. The triple store is
            # in-memory per run, so the graph stage still needs the ontology written
            # -- without it every induced subgraph is silently empty.
            if tools.triple_store_manager is not None:
                await tools.triple_store_manager.aserialize(ontology)
            tools.ontology_manager.add_ontology(ontology, skip_vector_index=True)
            continue
        ttl = ontology.graph.serialize(format="turtle").encode("utf-8")
        await tools.ingest_ontology_ttl(ttl)


async def score_cases(
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
            stride=store_config.proposition_window_stride,
            max_chars=store_config.proposition_window_max_chars,
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
            trigger_text=case.text,
        )
        metrics = dict(retriever.last_retrieval_metrics)
        seed_iris = {str(iri) for iri in metrics.get("seed_iris", [])}
        # Subject position, not mere mention: pruning drops triples by subject but
        # leaves object-position references, so a named term may carry no usable
        # definition.
        subjects = {str(subject) for subject in graph.subjects()}
        http_subjects = {s for s in subjects if s.startswith("http")}
        on_topic = sum(
            1
            for s in http_subjects
            if owner_of(s, owners, ontologies) == case.ontology_iri
        )
        # Unowned expected IRIs are attributed explicitly rather than silently
        # dropped, so a vocabulary whose header namespace excludes its own terms
        # reads as "(unattributed)" instead of "contributed nothing".
        counts.observe(
            expected=case.expected_iris,
            seed_iris=seed_iris,
            snapshot_subjects=subjects,
            metrics=metrics,
            on_topic_subjects=on_topic,
            total_subjects=len(http_subjects),
            expected_owner={
                iri: owner_of(iri, owners, ontologies) or "(unattributed)"
                for iri in case.expected_iris
            },
            case_id=case.case_id,
        )

    return counts
