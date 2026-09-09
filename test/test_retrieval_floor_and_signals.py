"""Tests for the per-source atom floor, query unit signals, and module closure."""

from types import SimpleNamespace

import pytest
from rdflib import URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.vector_store.core import GraphAtom, OntologySearchHit
from ontocast.tool.vector_store.patch_retriever import (
    _select_hits_round_robin_by_ontology,
)
from ontocast.tool.vector_store.query_signals import (
    CatalogSurfaceIndex,
    number_adjacent_tokens,
)

pytestmark = pytest.mark.unit


def _hit(iri: str, ontology_iri: str, score: float) -> OntologySearchHit:
    atom = GraphAtom(
        atom_id=f"atom-{iri}",
        ontology_iri=ontology_iri,
        ontology_id=ontology_iri.rsplit("/", 1)[-1],
        ontology_hash="h",
        ontology_version="1",
        iri=iri,
        entity_role="resource",
        core_representation=iri,
        minimal_representation=iri,
        neighborhood_representation="",
        score=score,
    )
    return OntologySearchHit(atom=atom, score=score)


BIG = "https://x.org/big"
SMALL = "https://x.org/small"


def _ranked_hits() -> list[OntologySearchHit]:
    # 6 dominant-ontology hits outscore both small-module hits.
    hits = [_hit(f"{BIG}#e{i}", BIG, 1.0 - i * 0.01) for i in range(6)]
    hits.append(_hit(f"{SMALL}#lowerBound", SMALL, 0.10))
    hits.append(_hit(f"{SMALL}#range", SMALL, 0.05))
    return hits


def test_floor_reserves_slots_for_starved_module() -> None:
    # Without a floor, the cap admits only dominant-ontology hits.
    selected = _select_hits_round_robin_by_ontology(
        _ranked_hits(), per_ontology_seed_quota=0, max_atoms=4
    )
    assert {hit.atom.ontology_iri for hit in selected} == {BIG}

    # With the floor, the small module is guaranteed its share; leftover
    # slots still fill in global score order.
    selected = _select_hits_round_robin_by_ontology(
        _ranked_hits(),
        per_ontology_seed_quota=0,
        max_atoms=4,
        per_ontology_atom_floor=2,
    )
    small = [hit for hit in selected if hit.atom.ontology_iri == SMALL]
    big = [hit for hit in selected if hit.atom.ontology_iri == BIG]
    assert len(small) == 2
    assert len(big) == 2
    assert big[0].score == 1.0


def test_floor_never_exceeds_candidate_count() -> None:
    selected = _select_hits_round_robin_by_ontology(
        _ranked_hits(),
        per_ontology_seed_quota=0,
        max_atoms=8,
        per_ontology_atom_floor=5,
    )
    # Small module only has 2 candidates; the rest go to the dominant one.
    assert sum(1 for hit in selected if hit.atom.ontology_iri == SMALL) == 2
    assert len(selected) == 8


def test_floor_zero_keeps_global_order() -> None:
    ranked = _ranked_hits()
    selected = _select_hits_round_robin_by_ontology(
        ranked, per_ontology_seed_quota=0, max_atoms=3, per_ontology_atom_floor=0
    )
    assert selected == ranked[:3]


def test_quota_backfill_preserved_with_floor() -> None:
    selected = _select_hits_round_robin_by_ontology(
        _ranked_hits(),
        per_ontology_seed_quota=1,
        max_atoms=5,
        per_ontology_atom_floor=0,
    )
    # Quota takes one per ontology, backfill fills the rest globally.
    assert len(selected) == 5


# --- query unit signals ------------------------------------------------------


def test_number_adjacent_tokens_extraction() -> None:
    text = (
        "aged for 4-15 days at 10 °C, measured at 77 K and 200 kV, "
        "threshold 0.5 %, shift of 96 meV, over 5 of the samples"
    )
    tokens = number_adjacent_tokens(text)
    assert {"days", "K", "kV", "%", "meV"} <= tokens
    assert "of" not in tokens


def test_catalog_surface_index_matches_case_and_plural() -> None:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix unit: <http://qudt.org/vocab/unit/> .
        @prefix qudt: <http://qudt.org/schema/qudt/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://x.org/units> a owl:Ontology .
        unit:DAY a qudt:Unit ; rdfs:label "Day"@en ; qudt:symbol "d" .
        unit:KiloV a qudt:Unit ; rdfs:label "Kilovolt"@en ; qudt:symbol "kV" .
        unit:PERCENT a qudt:Unit ; rdfs:label "Percent"@en ; qudt:symbol "%" .
        """,
        format="turtle",
    )
    ontology = Ontology(graph=graph, iri="https://x.org/units")
    # Symbol predicates are configuration (the retriever passes
    # VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES), not module constants.
    index = CatalogSurfaceIndex(
        symbol_predicates=[URIRef("http://qudt.org/schema/qudt/symbol")]
    )

    matched = index.match({"days", "kV", "%"}, [ontology])
    assert matched == {
        "http://qudt.org/vocab/unit/DAY": "https://x.org/units",
        "http://qudt.org/vocab/unit/KiloV": "https://x.org/units",
        "http://qudt.org/vocab/unit/PERCENT": "https://x.org/units",
    }
    # Unmatched tokens yield nothing; multi-word surfaces are not indexed.
    assert index.match({"nonexistent"}, [ontology]) == {}


def test_catalog_surface_index_without_symbol_predicates_matches_names_only() -> None:
    """No configured symbol predicates -> no vocabulary-specific surfaces."""
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix unit: <http://qudt.org/vocab/unit/> .
        @prefix qudt: <http://qudt.org/schema/qudt/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://x.org/units> a owl:Ontology .
        unit:KiloV a qudt:Unit ; rdfs:label "Kilovolt"@en ; qudt:symbol "kV" .
        """,
        format="turtle",
    )
    ontology = Ontology(graph=graph, iri="https://x.org/units")
    index = CatalogSurfaceIndex()

    assert index.match({"kV"}, [ontology]) == {}
    assert index.match({"kilovolt"}, [ontology]) == {
        "http://qudt.org/vocab/unit/KiloV": "https://x.org/units"
    }


def test_catalog_surface_index_caches_per_hash() -> None:
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <https://x.org/o#T> rdfs:label "meV" .
        """,
        format="turtle",
    )
    ontology = Ontology(graph=graph, iri="https://x.org/o")
    index = CatalogSurfaceIndex()
    first = index.match({"meV"}, [ontology])
    graph.add(
        (
            URIRef("https://x.org/o#T2"),
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            URIRef("https://x.org/other"),
        )
    )
    # Same (iri, hash) key -> cached surface map is reused.
    assert index.match({"meV"}, [ontology]) == first


def _closure_retriever(manager, closure_max: int, max_total: int | None = None):
    from ontocast.config import PatchRetrievalConfig
    from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever

    # vector_store is untouched by the closure path; bypass validation.
    retriever = OntologyPatchRetriever.model_construct(
        vector_store=None,
        sparql_tool=None,
        ontology_manager=manager,
        patch=PatchRetrievalConfig(
            small_module_closure_max_triples=closure_max,
            small_module_closure_max_total_triples=max_total,
        ),
    )
    return retriever


@pytest.mark.anyio
async def test_small_module_closure_merges_whole_module() -> None:
    from types import SimpleNamespace

    module_graph = RDFGraph()
    module_graph.parse(
        data="""
        @prefix qqval: <https://x.org/qqval#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <https://x.org/qqval> a owl:Ontology ; rdfs:label "qqval" .
        qqval:QuantityRange a owl:Class ; rdfs:label "Quantity range" .
        qqval:hasLowerBound a owl:ObjectProperty ; rdfs:label "has lower bound" .
        """,
        format="turtle",
    )
    ontology = Ontology(graph=module_graph, iri="https://x.org/qqval")
    manager = SimpleNamespace(
        get_freshest_terminal_ontology_by_iri=lambda iri: (
            ontology if iri == "https://x.org/qqval" else None
        )
    )
    retriever = _closure_retriever(manager, closure_max=10)
    snapshot = RDFGraph()
    await retriever._apply_small_module_closure(snapshot, ["https://x.org/qqval"])

    assert (
        URIRef("https://x.org/qqval#hasLowerBound"),
        URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        None,
    ) in snapshot
    # Header triples are stripped.
    assert (URIRef("https://x.org/qqval"), None, None) not in snapshot
    assert retriever.last_retrieval_metrics["module_closure_iris"] == [
        "https://x.org/qqval"
    ]

    # Over-threshold module is not closed.
    retriever_off = _closure_retriever(manager, closure_max=2)
    snapshot_off = RDFGraph()
    await retriever_off._apply_small_module_closure(
        snapshot_off, ["https://x.org/qqval"]
    )
    assert len(snapshot_off) == 0


def test_closure_floor_score_stays_below_weakest_seed() -> None:
    from ontocast.tool.vector_store.patch_retriever import _closure_floor_score

    # Positive floor: half the minimum, strictly below it.
    assert _closure_floor_score({"a": 0.8, "b": 0.4}) == 0.2
    # Zero floor: `0.0 * 0.5` would TIE with the weakest seed — must be below.
    assert _closure_floor_score({"a": 0.8, "b": 0.0}) < 0.0
    # Negative floor: `-0.4 * 0.5 == -0.2` would RANK ABOVE the weakest seed.
    assert _closure_floor_score({"a": 0.8, "b": -0.4}) < -0.4
    # No seeds at all: still finite and non-positive.
    assert _closure_floor_score({}) <= 0.0


def _qqval_manager():
    """A small module with one class and one property, resolvable by IRI."""
    from types import SimpleNamespace

    module_graph = RDFGraph()
    module_graph.parse(
        data="""
        @prefix qqval: <https://x.org/qqval#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <https://x.org/qqval> a owl:Ontology ; rdfs:label "qqval" .
        qqval:QuantityRange a owl:Class ; rdfs:label "Quantity range" .
        qqval:hasLowerBound a owl:ObjectProperty ; rdfs:label "has lower bound" .
        """,
        format="turtle",
    )
    ontology = Ontology(graph=module_graph, iri="https://x.org/qqval")
    return SimpleNamespace(
        get_freshest_terminal_ontology_by_iri=lambda iri: (
            ontology if iri == "https://x.org/qqval" else None
        )
    )


def _module(iri: str, terms: int) -> "Ontology":
    """A module of a given size, resolvable by IRI."""
    lines = [
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        f'<{iri}> a owl:Ontology ; rdfs:label "module" .',
    ]
    for index in range(terms):
        lines.append(f'<{iri}#T{index}> a owl:Class ; rdfs:label "term {index}" .')
    graph = RDFGraph()
    graph.parse(data="\n".join(lines), format="turtle")
    return Ontology(graph=graph, iri=iri)


def _multi_module_manager(*modules):
    from types import SimpleNamespace

    by_iri = {module.iri: module for module in modules}
    return SimpleNamespace(
        get_freshest_terminal_ontology_by_iri=lambda iri: by_iri.get(iri)
    )


BIG = "https://x.org/big"
SMALL = "https://x.org/small"


@pytest.mark.anyio
async def test_unlimited_budget_closes_every_candidate() -> None:
    """The default. Nothing is excluded for being unpopular or small."""
    retriever = _closure_retriever(
        _multi_module_manager(_module(BIG, 40), _module(SMALL, 4)), closure_max=200
    )
    snapshot = RDFGraph()

    await retriever._apply_small_module_closure(snapshot, [BIG, SMALL], {BIG: 0.9})

    assert set(retriever.last_retrieval_metrics["module_closure_iris"]) == {BIG, SMALL}
    assert "module_closure_declined_iris" not in retriever.last_retrieval_metrics


@pytest.mark.anyio
async def test_a_small_sharply_relevant_module_outranks_a_large_vague_one() -> None:
    """The case that rules out ranking by seed count.

    A module can win at most as many seeds as it has terms, so counting them
    ranks modules by size -- and would drop a tiny vocabulary that is precisely
    the document's subject in favour of a large peripheral one. Scores do not
    have that bias.
    """
    retriever = _closure_retriever(
        _multi_module_manager(_module(BIG, 40), _module(SMALL, 4)),
        closure_max=200,
        max_total=20,  # room for one of them
    )
    snapshot = RDFGraph()

    await retriever._apply_small_module_closure(
        snapshot, [BIG, SMALL], {BIG: 0.20, SMALL: 0.95}
    )

    assert retriever.last_retrieval_metrics["module_closure_iris"] == [SMALL]
    assert retriever.last_retrieval_metrics["module_closure_declined_iris"] == [BIG]
    assert (URIRef(f"{SMALL}#T0"), None, None) in snapshot


@pytest.mark.anyio
async def test_a_module_too_large_for_the_remainder_is_skipped_not_terminal() -> None:
    """The usual need is a combination of modules, not a single winner.

    Stopping at the first module that does not fit would spend the tail of the
    budget on nothing and drop vocabularies that would have fitted.
    """
    retriever = _closure_retriever(
        _multi_module_manager(_module(BIG, 40), _module(SMALL, 4)),
        closure_max=200,
        max_total=30,
    )
    snapshot = RDFGraph()

    # BIG ranks first on relevance but does not fit; SMALL must still get in.
    await retriever._apply_small_module_closure(
        snapshot, [BIG, SMALL], {BIG: 0.95, SMALL: 0.20}
    )

    assert retriever.last_retrieval_metrics["module_closure_iris"] == [SMALL]
    assert retriever.last_retrieval_metrics["module_closure_declined_iris"] == [BIG]


@pytest.mark.anyio
async def test_closure_order_is_stable_for_equal_relevance() -> None:
    """The snapshot is a prompt; an unstable prompt is uncacheable."""
    manager = _multi_module_manager(_module(BIG, 4), _module(SMALL, 4))
    runs = []
    for _ in range(3):
        # Room for exactly one of the two, and nothing to separate them on
        # relevance -- so only the tie-break decides, and it must not drift.
        retriever = _closure_retriever(manager, closure_max=200, max_total=15)
        await retriever._apply_small_module_closure(RDFGraph(), [BIG, SMALL], {})
        runs.append(retriever.last_retrieval_metrics["module_closure_iris"])

    assert len(set(map(tuple, runs))) == 1
    assert len(runs[0]) == 1


@pytest.mark.anyio
async def test_budget_spend_is_reported() -> None:
    retriever = _closure_retriever(
        _multi_module_manager(_module(SMALL, 4)), closure_max=200, max_total=100
    )
    snapshot = RDFGraph()

    await retriever._apply_small_module_closure(snapshot, [SMALL], {SMALL: 0.5})

    assert retriever.last_retrieval_metrics["module_closure_triples"] > 0


# ------------------------------------------------------- the signals flag


def _signal_retriever(manager, *, enabled: bool):
    """A retriever wired for the query-signals lane only.

    `model_construct` bypasses validation because neither the vector store nor
    the SPARQL tool is on this path; only the store config's flag is.
    """
    from ontocast.config import PatchRetrievalConfig, VectorStoreConfig
    from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever

    store = SimpleNamespace(
        store_config=VectorStoreConfig(query_unit_signals_enabled=enabled)
    )
    return OntologyPatchRetriever.model_construct(
        vector_store=store,
        sparql_tool=None,
        ontology_manager=manager,
        patch=PatchRetrievalConfig(),
    )


def _units_manager():
    graph = RDFGraph()
    graph.parse(
        data="""
        @prefix unit: <http://qudt.org/vocab/unit/> .
        @prefix qudt: <http://qudt.org/schema/qudt/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://x.org/units> a owl:Ontology .
        unit:DAY a qudt:Unit ; rdfs:label "Day"@en ; qudt:symbol "days" .
        """,
        format="turtle",
    )
    ontology = Ontology(graph=graph, iri="https://x.org/units")
    return SimpleNamespace(ontologies=[ontology])


def test_query_unit_signals_flag_gates_the_lane() -> None:
    """The flag itself, not just the matcher underneath it.

    The lane ships disabled, so nothing in a default deployment executes it;
    without this the flag could be flipped -- or silently stop working -- with
    no test noticing.
    """
    manager = _units_manager()
    text = "aged for 4-15 days under illumination"

    assert (
        _signal_retriever(manager, enabled=False)._match_query_unit_signals(text) == {}
    )
    assert _signal_retriever(manager, enabled=True)._match_query_unit_signals(text) == {
        "http://qudt.org/vocab/unit/DAY": "https://x.org/units"
    }


def test_query_unit_signals_need_a_number_to_key_on() -> None:
    """The lane keys on the token *after* a number; prose alone matches nothing."""
    retriever = _signal_retriever(_units_manager(), enabled=True)
    assert retriever._match_query_unit_signals("aged for several days") == {}


def test_query_unit_signals_are_inert_without_a_catalog() -> None:
    retriever = _signal_retriever(None, enabled=True)
    assert retriever._match_query_unit_signals("4-15 days") == {}
