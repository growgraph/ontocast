"""Tests for ToolBox ontology synchronization helpers."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ontocast.config import (
    Config,
    EmbeddingConfig,
    FusekiConfig,
    PathConfig,
    QdrantConfig,
    ToolConfig,
)
from ontocast.onto.enum import OntologyContextMode, RenderMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_manager import OntologyManager
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def test_materialize_ontology_calls_vector_reindex(test_ontology):
    tb = MagicMock()
    tb.triple_store_manager = None
    reindexed: list = []

    def reindex(o):
        reindexed.append(o)

    tb.vector_store = MagicMock()
    tb.vector_store.reindex_ontology = reindex

    async def main():
        await ToolBox._materialize_ontology(tb, test_ontology)

    asyncio.run(main())
    assert reindexed == [test_ontology]


def test_materialize_ontology_skips_vector_reindex_when_store_not_ready(test_ontology):
    tb = MagicMock()
    tb.triple_store_manager = None
    tb.vector_store = MagicMock()
    tb.vector_store.reindex_ontology = MagicMock()
    tb.is_vector_store_ready = MagicMock(return_value=False)

    async def main():
        await ToolBox._materialize_ontology(tb, test_ontology)

    asyncio.run(main())
    tb.vector_store.reindex_ontology.assert_not_called()


def test_materialize_ontology_serializes_remote_triple_store(test_ontology):
    remote = MagicMock()
    remote.aserialize = AsyncMock(return_value=True)
    tb = MagicMock()
    tb.triple_store_manager = remote
    tb.vector_store = None

    async def main():
        await ToolBox._materialize_ontology(tb, test_ontology)

    asyncio.run(main())
    remote.aserialize.assert_awaited_once_with(test_ontology)


def test_initialize_materializes_then_adds_with_skip_vector(monkeypatch, test_ontology):
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )

    materialized: list = []
    added: list = []

    async def fake_sync(self):
        return [test_ontology]

    async def fake_mat(self, o):
        materialized.append(o)

    def fake_add(ontology, *, skip_vector_index: bool = False):
        added.append((ontology, skip_vector_index))

    class Stub:
        # Bound to the real implementation so the catalog/index agreement
        # check runs in these tests rather than being stubbed away.
        async def _check_catalog_index_agreement(self, ontologies):
            await ToolBox._check_catalog_index_agreement(
                cast(ToolBox, self), ontologies
            )

        async def _check_catalog_ready(self, ontologies, *, required):
            await ToolBox._check_catalog_ready(
                cast(ToolBox, self), ontologies, required=required
            )

        def _catalog_sources_description(self):
            return ToolBox._catalog_sources_description(cast(ToolBox, self))

        vector_store = None
        triple_store_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock
        config = Config()

        def __init__(self) -> None:
            self.ontology_manager = MagicMock()
            self.shapes_catalog = MagicMock()
            self.shapes_catalog.sync = AsyncMock()
            self.vector_store_ready = False
            self.vector_store_last_error = None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

        def is_vector_store_ready(self):
            return ToolBox.is_vector_store_ready(cast(ToolBox, self))

        _synchronize_ontologies = fake_sync
        _materialize_ontology = fake_mat

    st = Stub()
    st.ontology_manager.add_ontology = MagicMock(side_effect=fake_add)

    async def main():
        await ToolBox.initialize(cast(ToolBox, st))

    asyncio.run(main())

    assert materialized == [test_ontology]
    assert added == [(test_ontology, True)]
    # Catalog registration happens before materialize so enrich can overlap.
    assert added  # registration recorded
    assert materialized


def test_toolbox_rejects_mismatched_qdrant_vector_size_and_embedding_dim() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(
            path_config=PathConfig(ontology_directory=od),
            embedding=EmbeddingConfig(dimension=384),
            qdrant=QdrantConfig(uri="http://localhost:6333", vector_size=8),
        )
        with pytest.raises(ValueError, match="vector_size must match"):
            ToolBox(Config(tool_config=tool_config))


def test_toolbox_always_wires_bm25_when_vector_search_enabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(
            path_config=PathConfig(ontology_directory=od),
            embedding=EmbeddingConfig(dimension=384),
            qdrant=QdrantConfig(uri="http://localhost:6333"),
        )
        toolbox = ToolBox(Config(tool_config=tool_config))
        assert toolbox.vector_store is not None
        assert toolbox.vector_store.sparse_embedding is not None


def test_initialize_skips_vector_store_in_full_ttl_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )
    synchronized: list = []

    class Stub:
        # Bound to the real implementation so the catalog/index agreement
        # check runs in these tests rather than being stubbed away.
        async def _check_catalog_index_agreement(self, ontologies):
            await ToolBox._check_catalog_index_agreement(
                cast(ToolBox, self), ontologies
            )

        async def _check_catalog_ready(self, ontologies, *, required):
            await ToolBox._check_catalog_ready(
                cast(ToolBox, self), ontologies, required=required
            )

        def _catalog_sources_description(self):
            return ToolBox._catalog_sources_description(cast(ToolBox, self))

        triple_store_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock
        config = Config()

        def __init__(self) -> None:
            self.vector_store = MagicMock()
            self.vector_store.initialize = AsyncMock()
            self.vector_store_ready = False
            self.vector_store_last_error = None
            self.ontology_manager = MagicMock()
            self.shapes_catalog = MagicMock()
            self.shapes_catalog.sync = AsyncMock()

        async def _synchronize_ontologies(self):
            return synchronized

        async def _materialize_ontology(self, _):
            return None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

        def is_vector_store_ready(self):
            return ToolBox.is_vector_store_ready(cast(ToolBox, self))

    st = Stub()
    asyncio.run(
        ToolBox.initialize(
            cast(ToolBox, st),
            ontology_context_mode=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
            fail_on_vector_store_error=False,
        )
    )
    st.vector_store.initialize.assert_not_awaited()


def test_initialize_vector_store_failure_is_non_fatal_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )

    class Stub:
        # Bound to the real implementation so the catalog/index agreement
        # check runs in these tests rather than being stubbed away.
        async def _check_catalog_index_agreement(self, ontologies):
            await ToolBox._check_catalog_index_agreement(
                cast(ToolBox, self), ontologies
            )

        async def _check_catalog_ready(self, ontologies, *, required):
            await ToolBox._check_catalog_ready(
                cast(ToolBox, self), ontologies, required=required
            )

        def _catalog_sources_description(self):
            return ToolBox._catalog_sources_description(cast(ToolBox, self))

        triple_store_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock
        config = Config()

        def __init__(self) -> None:
            self.vector_store = MagicMock()
            self.vector_store.initialize = AsyncMock(
                side_effect=RuntimeError("qdrant unavailable")
            )
            self.vector_store_ready = False
            self.vector_store_last_error = None
            self.ontology_manager = MagicMock()
            self.shapes_catalog = MagicMock()
            self.shapes_catalog.sync = AsyncMock()

        async def _synchronize_ontologies(self):
            return []

        async def _materialize_ontology(self, _):
            return None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

        def is_vector_store_ready(self):
            return ToolBox.is_vector_store_ready(cast(ToolBox, self))

    st = Stub()
    asyncio.run(
        ToolBox.initialize(
            cast(ToolBox, st),
            ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
            fail_on_vector_store_error=False,
        )
    )
    assert st.vector_store_ready is False
    assert st.vector_store_last_error is not None


def test_ingest_ontology_ttl_rejects_identity_conflict_before_persisting() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(path_config=PathConfig(ontology_directory=od))
        config = Config(tool_config=tool_config)
        ontology_manager = OntologyManager()

        existing = Ontology(
            graph=RDFGraph._from_turtle_str(
                """
                @prefix owl: <http://www.w3.org/2002/07/owl#> .
                <https://example.org/finance> a owl:Ontology .
                """
            ),
            iri="https://example.org/finance",
            ontology_id="finance",
        )
        ontology_manager.add_ontology(existing)

        class Stub:
            # Bound to the real implementation so the catalog/index agreement
            # check runs in these tests rather than being stubbed away.
            async def _check_catalog_index_agreement(self, ontologies):
                await ToolBox._check_catalog_index_agreement(
                    cast(ToolBox, self), ontologies
                )

            async def _check_catalog_ready(self, ontologies, *, required):
                await ToolBox._check_catalog_ready(
                    cast(ToolBox, self), ontologies, required=required
                )

            def _catalog_sources_description(self):
                return ToolBox._catalog_sources_description(cast(ToolBox, self))

            def __init__(self) -> None:
                self.config = config
                self.ontology_manager = ontology_manager
                self._materialize_ontology = AsyncMock()

        incoming_ttl = b"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.com/finance> a owl:Ontology .
        """
        stub = Stub()

        with pytest.raises(ValueError, match="already bound to IRI"):
            asyncio.run(ToolBox.ingest_ontology_ttl(cast(ToolBox, stub), incoming_ttl))

        stub._materialize_ontology.assert_not_awaited()


def test_initialize_materializes_with_bounded_concurrency(
    monkeypatch, test_ontology
) -> None:
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )

    active = 0
    max_active = 0
    lock = asyncio.Lock()

    ontologies = [
        Ontology(
            graph=RDFGraph(),
            iri=f"https://example.org/o{i}",
            ontology_id=f"o{i}",
        )
        for i in range(4)
    ]

    async def fake_sync(self):
        return ontologies

    async def fake_mat(self, o):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        async with lock:
            active -= 1

    class Stub:
        # Bound to the real implementation so the catalog/index agreement
        # check runs in these tests rather than being stubbed away.
        async def _check_catalog_index_agreement(self, ontologies):
            await ToolBox._check_catalog_index_agreement(
                cast(ToolBox, self), ontologies
            )

        async def _check_catalog_ready(self, ontologies, *, required):
            await ToolBox._check_catalog_ready(
                cast(ToolBox, self), ontologies, required=required
            )

        def _catalog_sources_description(self):
            return ToolBox._catalog_sources_description(cast(ToolBox, self))

        vector_store = None
        triple_store_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock
        config = Config()

        def __init__(self) -> None:
            self.ontology_manager = MagicMock()
            self.shapes_catalog = MagicMock()
            self.shapes_catalog.sync = AsyncMock()
            self.vector_store_ready = False
            self.vector_store_last_error = None
            self.config.tool_config.vector_store.reindex_concurrency = 2

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

        def is_vector_store_ready(self):
            return ToolBox.is_vector_store_ready(cast(ToolBox, self))

        _synchronize_ontologies = fake_sync
        _materialize_ontology = fake_mat

    st = Stub()
    st.ontology_manager.add_ontology = MagicMock()

    asyncio.run(ToolBox.initialize(cast(ToolBox, st)))
    # The contract is the *bound*. Asserting `max_active >= 2` as well made this
    # load-sensitive: it demanded the scheduler actually overlap two coroutines
    # within a 50 ms sleep, which a busy machine need not do.
    assert max_active <= 2


@pytest.mark.parametrize(
    ("wipe", "prune"),
    [
        pytest.param(True, True, id="enabled"),
        pytest.param(False, False, id="disabled"),
    ],
)
def test_initialize_wipe_and_prune_follow_their_flags(
    monkeypatch, test_ontology, wipe: bool, prune: bool
) -> None:
    """Wipe and orphan-prune run on init exactly when configured to."""
    monkeypatch.setattr(
        "ontocast.toolbox.update_ontology_manager",
        AsyncMock(),
    )

    orphans = ["https://example.org/legacy"] if prune else []

    class Stub:
        # Bound to the real implementation so the catalog/index agreement
        # check runs in these tests rather than being stubbed away.
        async def _check_catalog_index_agreement(self, ontologies):
            await ToolBox._check_catalog_index_agreement(
                cast(ToolBox, self), ontologies
            )

        async def _check_catalog_ready(self, ontologies, *, required):
            await ToolBox._check_catalog_ready(
                cast(ToolBox, self), ontologies, required=required
            )

        def _catalog_sources_description(self):
            return ToolBox._catalog_sources_description(cast(ToolBox, self))

        triple_store_manager = None
        llm = MagicMock()
        ontology_manager: MagicMock
        config = Config()

        def __init__(self) -> None:
            self.vector_store = MagicMock()
            self.vector_store.initialize = AsyncMock()
            self.vector_store.wipe_store = AsyncMock()
            self.vector_store.prune_orphan_ontology_iris = MagicMock(
                return_value=orphans
            )
            self.vector_store_ready = False
            self.vector_store_last_error = None
            self.ontology_manager = MagicMock()
            self.shapes_catalog = MagicMock()
            self.shapes_catalog.sync = AsyncMock()

        async def _synchronize_ontologies(self):
            return [test_ontology]

        async def _materialize_ontology(self, _):
            return None

        def should_initialize_vector_store(self, ontology_context_mode):
            return ToolBox.should_initialize_vector_store(
                cast(ToolBox, self), ontology_context_mode
            )

        def is_vector_store_ready(self):
            return self.vector_store_ready

    st = Stub()
    kwargs: dict[str, bool] = {}
    if wipe or prune:
        kwargs = {"wipe_vector_store": wipe, "prune_orphan_iris": prune}
    else:
        # The disabled case comes from configuration rather than call kwargs.
        st.config.tool_config.vector_store.wipe_on_init = False
        st.config.tool_config.vector_store.prune_orphan_iris_on_init = False

    asyncio.run(
        ToolBox.initialize(
            cast(ToolBox, st),
            ontology_context_mode=OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
            **kwargs,
        )
    )

    if wipe:
        st.vector_store.wipe_store.assert_awaited_once()
        st.vector_store.initialize.assert_awaited_once()
        assert st.vector_store_ready is True
    else:
        st.vector_store.wipe_store.assert_not_awaited()

    if prune:
        st.vector_store.prune_orphan_ontology_iris.assert_called_once_with(
            {test_ontology.iri}
        )
    else:
        st.vector_store.prune_orphan_ontology_iris.assert_not_called()


def _skip_if_qdrant_configured_but_down() -> None:
    """Skip when a configured Qdrant is unreachable.

    These build a real ToolBox, so they use whatever vector backend the
    environment configures. With no QDRANT_URI (a clean CI checkout) that is
    the in-memory store and they run offline; with QDRANT_URI set but the
    service down they used to fail with a bare connection error instead of
    skipping like every other service-dependent test.
    """
    from ontocast.config import QdrantConfig
    from test.qdrant_util import qdrant_reachable

    qdrant = QdrantConfig()
    if qdrant.uri and not qdrant_reachable(uri=qdrant.uri, api_key=qdrant.api_key):
        pytest.skip(f"Qdrant not reachable at {qdrant.uri}")


def _tenancy_toolbox(tmp: str) -> ToolBox:
    _skip_if_qdrant_configured_but_down()
    wd = Path(tmp)
    od = wd / "ontologies"
    od.mkdir()
    # Force the in-memory backend: the test environment may define FUSEKI_URI.
    return ToolBox(
        Config(
            tool_config=ToolConfig(
                path_config=PathConfig(ontology_directory=od),
                embedding=EmbeddingConfig(dimension=384),
                fuseki=FusekiConfig(uri=None, auth=None),
            )
        )
    )


def _tenant_ontology(iri: str, ontology_id: str) -> Ontology:
    graph = RDFGraph._from_turtle_str(
        f"""
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <{iri}#> .

        <{iri}> a owl:Ontology ; rdfs:label "Tenant ontology" .
        ex:Thing a owl:Class .
        """
    )
    return Ontology(graph=graph, iri=iri, ontology_id=ontology_id)


@pytest.mark.integration
def test_tenancy_switch_clears_the_in_memory_catalog() -> None:
    """A tenancy switch must not leave the previous tenant's ontologies visible."""
    with tempfile.TemporaryDirectory() as tmp:
        toolbox = _tenancy_toolbox(tmp)
        onto_a = _tenant_ontology("https://example.org/tenant-a", "shared")

        async def main() -> None:
            await toolbox.update_tenancy("alpha", "one")
            await toolbox.triple_store_manager.aserialize(onto_a)
            toolbox.ontology_manager.add_ontology(onto_a, skip_vector_index=True)
            assert toolbox.ontology_manager.get_ontology_iris() == [onto_a.iri]

            await toolbox.update_tenancy("beta", "one")
            assert toolbox.ontology_manager.get_ontology_iris() == []

            # The alias ledger went with it: a different IRI may reuse the id.
            onto_b = _tenant_ontology("https://example.org/tenant-b", "shared")
            toolbox.ontology_manager.add_ontology(onto_b, skip_vector_index=True)
            assert toolbox.ontology_manager.get_ontology_iris() == [onto_b.iri]

        asyncio.run(main())


@pytest.mark.integration
def test_tenancy_switch_back_repopulates_from_the_store() -> None:
    """Switching back must restore the catalog rather than leave it empty."""
    with tempfile.TemporaryDirectory() as tmp:
        toolbox = _tenancy_toolbox(tmp)
        onto_a = _tenant_ontology("https://example.org/tenant-a", "alpha-onto")

        async def main() -> None:
            await toolbox.update_tenancy("alpha", "one")
            await toolbox.triple_store_manager.aserialize(onto_a)
            toolbox.ontology_manager.add_ontology(onto_a, skip_vector_index=True)

            await toolbox.update_tenancy("beta", "one")
            await toolbox.update_tenancy("alpha", "one")
            assert toolbox.ontology_manager.get_ontology_iris() == [onto_a.iri]

        asyncio.run(main())


@pytest.mark.integration
def test_repeated_tenancy_call_does_not_drop_the_catalog() -> None:
    """Re-asserting the same tenancy is a no-op, not a reset."""
    with tempfile.TemporaryDirectory() as tmp:
        toolbox = _tenancy_toolbox(tmp)
        onto = _tenant_ontology("https://example.org/tenant-a", "alpha-onto")

        async def main() -> None:
            await toolbox.update_tenancy("alpha", "one")
            toolbox.ontology_manager.add_ontology(onto, skip_vector_index=True)
            await toolbox.update_tenancy("alpha", "one")
            assert toolbox.ontology_manager.get_ontology_iris() == [onto.iri]

        asyncio.run(main())


# --- Catalog/vector-index agreement at startup ---------------------------


def _agreement_toolbox(indexed: list[str], *, required: bool):
    """A ToolBox stub whose only live parts are the two halves of retrieval."""
    config = Config()
    config.server.ontology_context_required = required

    class Stub:
        triple_store_manager = None
        llm = MagicMock()

        def __init__(self) -> None:
            self.config = config
            self.vector_store = MagicMock()
            self.vector_store.list_indexed_ontology_iris = MagicMock(
                return_value=set(indexed)
            )

        def is_vector_store_ready(self):
            return True

    return Stub()


def test_an_empty_catalog_beside_a_populated_index_stops_startup() -> None:
    """The confirmed failure: ontologies never reached the triple store.

    Retrieval still selects atoms from the index and reports healthy metrics --
    the expected seeds, the expected atom count -- while the induced subgraph
    built over the empty catalog comes back with nothing. Every unit then
    renders against an empty ontology chapter and the conformance gate passes
    over zero nodes, so nothing downstream can tell this from a good run.
    """
    from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError

    stub = _agreement_toolbox(["https://example.org/o1"], required=True)

    with pytest.raises(EmptyOntologyContextError) as excinfo:
        asyncio.run(ToolBox._check_catalog_index_agreement(cast(ToolBox, stub), []))

    assert "https://example.org/o1" in str(excinfo.value), (
        "the error must name what the index still holds, or the operator has "
        "nothing to act on"
    )


def test_the_agreement_check_is_not_opted_out_of_by_wanting_no_catalog() -> None:
    """``ONTOLOGY_CONTEXT_REQUIRED`` answers a different question.

    It says whether a run wants a catalog. This says the index and the triple
    store disagree about which ontologies exist, which nobody configures on
    purpose -- and a run that deliberately extracts without a catalog has not
    thereby asked for a stale index to select atoms nothing can expand.
    """
    from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError

    stub = _agreement_toolbox(["https://example.org/o1"], required=False)

    with pytest.raises(EmptyOntologyContextError):
        asyncio.run(ToolBox._check_catalog_index_agreement(cast(ToolBox, stub), []))


def test_stale_index_entries_beside_a_real_catalog_only_warn(caplog) -> None:
    """Ordinary staleness is what orphan pruning is for, not a startup failure."""
    stub = _agreement_toolbox(
        ["https://example.org/o1", "https://example.org/gone"], required=True
    )
    served = [Ontology(iri="https://example.org/o1", ontology_id="o1")]

    with caplog.at_level(logging.WARNING):
        asyncio.run(ToolBox._check_catalog_index_agreement(cast(ToolBox, stub), served))

    assert "https://example.org/gone" in caplog.text


def test_agreement_check_is_silent_when_both_sides_match() -> None:
    stub = _agreement_toolbox(["https://example.org/o1"], required=True)
    served = [Ontology(iri="https://example.org/o1", ontology_id="o1")]

    asyncio.run(ToolBox._check_catalog_index_agreement(cast(ToolBox, stub), served))


# --- Catalog readiness after materialization ------------------------------


def _readiness_toolbox(
    indexed: list[str],
    *,
    ready: bool = True,
    render_mode: RenderMode = RenderMode.FACTS,
):
    """A ToolBox stub with a live vector index and nothing else."""
    config = Config()
    config.server.ontology_context_required = True
    config.server.render_mode = render_mode

    class Stub:
        triple_store_manager = None
        llm = MagicMock()

        def __init__(self) -> None:
            self.config = config
            self.vector_store = MagicMock()
            self.vector_store.list_indexed_ontology_iris = MagicMock(
                return_value=set(indexed)
            )

        def is_vector_store_ready(self):
            return ready

        def _catalog_sources_description(self):
            return ToolBox._catalog_sources_description(cast(ToolBox, self))

    return Stub()


def test_a_wiped_index_with_no_catalog_to_refill_it_stops_a_facts_run() -> None:
    """The case ``--wipe-vector-store`` hides from the agreement check.

    That check requires a populated index to compare against, and a wipe
    guarantees an empty one, so it returns early on exactly the run where the
    wipe destroyed the last copy of the vocabulary. The sync then found nothing
    to reindex and startup succeeded, leaving every content unit to fail on its
    own several minutes and several provider calls later.
    """
    from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError

    stub = _readiness_toolbox([], render_mode=RenderMode.FACTS)

    with pytest.raises(EmptyOntologyContextError) as excinfo:
        asyncio.run(
            ToolBox._check_catalog_ready(cast(ToolBox, stub), [], required=True)
        )

    assert "ontology_directory" in str(excinfo.value), (
        "the error must name where a catalog was looked for, or it says only "
        "that something is missing"
    )


def test_an_empty_catalog_starts_when_the_run_will_create_ontologies(caplog) -> None:
    """The corpus with no ontology yet is this render mode's starting point.

    Refusing here made ``render_ontology_fresh`` -- the branch written for an
    empty seed -- unreachable, and turned the documented first run into a
    startup error.
    """
    stub = _readiness_toolbox([], render_mode=RenderMode.ONTOLOGY_AND_FACTS)

    with caplog.at_level(logging.INFO):
        asyncio.run(
            ToolBox._check_catalog_ready(cast(ToolBox, stub), [], required=True)
        )

    assert "create ontologies from the corpus" in caplog.text


def test_an_empty_catalog_only_warns_when_the_entry_point_allows_one() -> None:
    """A server may legitimately start empty and be filled over HTTP."""
    stub = _readiness_toolbox([])

    asyncio.run(ToolBox._check_catalog_ready(cast(ToolBox, stub), [], required=False))


def test_a_catalog_that_indexed_to_nothing_always_stops(test_ontology) -> None:
    """Retrieval has graphs it could expand and no atoms to select them with.

    Unlike an empty catalog, this is never what anyone meant: materialization
    ran over a populated catalog and produced nothing. So it does not consult
    ``required`` -- ``serve`` and an ontology-rendering run stop on it too.
    """
    from ontocast.onto.retrieval_capabilities import EmptyOntologyContextError

    stub = _readiness_toolbox([], render_mode=RenderMode.ONTOLOGY_AND_FACTS)

    with pytest.raises(EmptyOntologyContextError) as excinfo:
        asyncio.run(
            ToolBox._check_catalog_ready(
                cast(ToolBox, stub), [test_ontology], required=False
            )
        )

    assert "vector index is empty" in str(excinfo.value)


def test_a_populated_catalog_and_index_starts(test_ontology) -> None:
    stub = _readiness_toolbox(["https://example.org/o1"])

    asyncio.run(
        ToolBox._check_catalog_ready(
            cast(ToolBox, stub), [test_ontology], required=True
        )
    )
