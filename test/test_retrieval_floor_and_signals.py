"""Tests for the per-source atom floor, query unit signals, and module closure."""

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


def _closure_retriever(manager, closure_max: int):
    from ontocast.config import PatchRetrievalConfig
    from ontocast.tool.vector_store.patch_retriever import OntologyPatchRetriever

    # vector_store is untouched by the closure path; bypass validation.
    retriever = OntologyPatchRetriever.model_construct(
        vector_store=None,
        sparql_tool=None,
        ontology_manager=manager,
        patch=PatchRetrievalConfig(small_module_closure_max_triples=closure_max),
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
