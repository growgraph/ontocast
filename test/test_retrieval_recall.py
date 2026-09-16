"""Recall measurement for vector-mode ontology patch retrieval.

Unlike the rest of the vector suite, this module uses **real** embeddings and a **real**
vector store. Hash-based fake vectors make recall numbers meaningless, and recall is
precisely what was never measured: the plumbing tests assert ordering, counts, and
parameter pass-through, but nothing asserts that a relevant catalog term survives to the
prompt snapshot.

Two numbers are reported per run:

* **seed recall** -- the expected term reached ``atoms_final`` (survived vector search,
  cross-window merge, per-ontology round-robin, and the window-scaled cap).
* **snapshot recall** -- the expected term is *defined* in the returned graph (also
  survived induced-subgraph expansion: BFS quotas, budget caps, component pruning).

The gap between them attributes losses to the graph stage rather than the vector stage.
Both are printed as a funnel alongside the metrics the pipeline already emits, so a
regression can be localised without bisecting.

**No LLM call is made**, so the whole harness runs with no provider credentials and no
generation spend. Wiring and scoring live in :mod:`test.retrieval_runner`, shared with
:mod:`test.retrieval_sweep`.

Scale and corpus are environment-controlled so the same harness serves CI (small, fast)
and tuning sweeps / embedding bake-offs (large):

* ``ONTOCAST_RECALL_CORPUS``      -- prebuilt corpus directory (``cases.jsonl`` +
  ``ontologies/``), the domain-neutral tier
* ``ONTOCAST_RECALL_ROOT``        -- Text2KGBench corpus root
* ``ONTOCAST_RECALL_ONTOLOGIES``  -- ontologies loaded into the catalog (default 6);
  Text2KGBench tier only
* ``ONTOCAST_RECALL_CASES``       -- cases per ontology (default 15); Text2KGBench tier
  only
* ``ONTOCAST_RECALL_JSON``        -- write the funnel to this path as JSON

Ablation controls, for asking whether indexing a large external vocabulary helps or
dilutes. Both are opt-in and inert when unset:

* ``ONTOCAST_RECALL_EXTRA_ONTOLOGIES`` -- ``os.pathsep``-separated .ttl files/directories
  appended to the corpus catalog. Keeps the *index* axis a one-variable flip while the
  corpus on disk stays byte-identical across arms.
* ``ONTOCAST_RECALL_COLLECTION_SUFFIX`` -- pin the collection/table names and skip
  teardown, so the index can be reused. With ``ONTOCAST_RECALL_SKIP_INDEX=1`` a later
  arm scores against it without re-embedding. Everything on the *retrieval* axis
  (seed quotas, merge mode, caps, closure budgets) is applied at merge time, so a whole
  sweep needs one index.

Case text is split into proposition windows exactly as production does, so a passage
spanning several sentences issues several queries rather than one.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from ontocast.config import LanceDBConfig, QdrantConfig
from ontocast.onto.ontology import Ontology
from ontocast.toolbox import ToolBox
from test.qdrant_util import qdrant_reachable
from test.retrieval_gt import (
    RecallCase,
    StageCounts,
    corpus_root,
    load_anchor_cases,
    load_corpus,
    load_text2kgbench,
    text2kgbench_root,
)
from test.retrieval_runner import (
    build_toolbox,
    index_catalog,
    lancedb_store_config,
    pinned_suffix,
    score_cases,
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
def recall_store_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[dict[str, Any], Any, None]:
    """Vector-store settings for one recall run, on whichever backend is configured.

    LanceDB is preferred when enabled: it is embedded, so a parameter sweep needs no
    running service, and the index it builds is a directory that can be pinned and
    reused across arms. Qdrant is used when ``QDRANT_URI`` names a reachable instance.

    Deliberately *not* the shared ``qdrant_session_test_context``: that fixture is used
    by smoke tests that create 8-dimensional collections, and the embedding contract
    would reject the real 384-dimensional model against those.

    Function-scoped so each tier indexes only its own ontologies; sharing a collection
    let one tier's catalog leak into the other's ``seeds_by_ontology`` and made the
    measured ontology count depend on test order.

    Yields:
        dict[str, Any]: ``ToolConfig`` keyword overrides naming this run's store.
    """
    pinned = pinned_suffix()
    run_id = pinned or uuid.uuid4().hex[:8]

    if LanceDBConfig().enabled:
        # A pinned run keeps its directory so a later arm can score against it with
        # ONTOCAST_RECALL_SKIP_INDEX=1; an unpinned one gets a throwaway.
        data_dir = (
            Path(LanceDBConfig().data_dir).expanduser()
            if pinned
            else tmp_path_factory.mktemp(f"recall_lancedb_{run_id}")
        )
        yield lancedb_store_config(data_dir, run_id)
        if not pinned:
            shutil.rmtree(data_dir, ignore_errors=True)
        return

    base = QdrantConfig()
    if base.uri is None:
        pytest.skip("no vector backend: set LANCEDB_ENABLED=true or QDRANT_URI")
    if not qdrant_reachable(uri=base.uri, api_key=base.api_key):
        pytest.skip(f"Qdrant not reachable at {base.uri}")

    config = base.model_copy(
        update={
            "ontology_collection": f"ontocast_recall_{run_id}_ontologies",
            "facts_collection": f"ontocast_recall_{run_id}_facts",
        }
    )

    yield {"qdrant": config, "lancedb": LanceDBConfig(enabled=False)}

    if pinned:
        return

    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=config.uri,
        api_key=config.api_key,
        grpc_port=config.grpc_port,
        prefer_grpc=config.use_grpc,
    )
    for name in (config.ontology_collection, config.facts_collection):
        if name and client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)


def _run(tools: ToolBox, ontologies: list[Ontology], cases: list[RecallCase]):
    async def _main() -> StageCounts:
        await index_catalog(tools, ontologies)
        return await score_cases(tools, ontologies, cases)

    return asyncio.run(_main())


def _emit(counts: StageCounts, title: str) -> str:
    """Print the funnel, write it as JSON when asked, and return the report.

    ``ONTOCAST_RECALL_JSON`` names a file the run's counters are written to. A
    printed table that has to be re-parsed is how a measurement stops being
    comparable across arms, so the numbers can also leave the process as data.
    """
    report = counts.render(title)
    print(f"\n{report}")
    destination = os.getenv("ONTOCAST_RECALL_JSON", "").strip()
    if destination:
        payload = counts.as_dict()
        payload["title"] = title
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return report


@pytest.mark.slow
def test_anchor_recall(
    recall_store_config: dict[str, Any],
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

    tools = build_toolbox(
        recall_store_config, tmp_path_factory.mktemp("recall_ontologies")
    )
    counts = _run(tools, ontologies, cases)
    report = _emit(counts, "anchor recall")

    assert counts.cases > 0
    assert counts.seed_recall > 0.0, (
        "no anchor term reached the seed set; retrieval is not functioning at all\n"
        + report
    )


@pytest.mark.slow
def test_corpus_recall(
    recall_store_config: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A prebuilt domain corpus: multi-sentence passages against a linked catalog.

    The other two tiers score one sentence at a time against mutually disjoint
    ontologies, which makes the cross-window merge a no-op and hides both the window
    budget and cross-ontology expansion. A prebuilt corpus supplies passages long
    enough to produce several proposition windows and a catalog whose ontologies
    reference each other.
    """
    root = corpus_root()
    if root is None:
        pytest.skip(
            "no recall corpus; set ONTOCAST_RECALL_CORPUS to a directory holding "
            "cases.jsonl and ontologies/"
        )

    cases, ontologies = load_corpus(root)
    tools = build_toolbox(
        recall_store_config, tmp_path_factory.mktemp("recall_ontologies")
    )
    counts = _run(tools, ontologies, cases)
    report = _emit(
        counts,
        f"corpus recall: {root.name} "
        f"({len(ontologies)} ontologies, {len(cases)} cases)",
    )

    assert counts.cases > 0
    assert counts.seed_recall > 0.0, (
        "no expected term reached the seed set across the whole corpus\n" + report
    )


@pytest.mark.slow
def test_text2kgbench_recall(
    recall_store_config: dict[str, Any],
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

    tools = build_toolbox(
        recall_store_config, tmp_path_factory.mktemp("recall_ontologies")
    )
    counts = _run(tools, ontologies, cases)
    report = _emit(
        counts,
        f"text2kgbench recall ({len(ontologies)} ontologies, {len(cases)} cases)",
    )

    assert counts.cases > 0
    assert counts.seed_recall > 0.0, (
        "no expected term reached the seed set across the whole corpus\n" + report
    )
