"""The shared ontology snapshot must serialise its graph once, not per unit.

Serialising the ontology is the most expensive step in building a facts prompt.
Once the fan-out shares one snapshot across every unit, repeating it per unit
(and per render attempt) is pure waste -- and it is waste that lands on the
event loop, so it stalls the other units rather than just costing CPU.
"""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, Literal, Namespace

from ontocast.onto.enum import LLMGraphFormat, OntologyAssemblyMode
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.prompt.graph_format import get_graph_format_profile

EX = Namespace("https://example.org/onto#")


def _graph(n: int = 20) -> RDFGraph:
    graph = RDFGraph()
    graph.bind("ex", EX)
    for index in range(n):
        subject = EX[f"C{index}"]
        graph.add((subject, RDF.type, OWL.Class))
        graph.add((subject, RDFS.label, Literal(f"Class {index}")))
    return graph


def _snapshot(graph: RDFGraph) -> OntologySnapshot:
    return OntologySnapshot(
        graph=graph,
        source_iris=["https://example.org/onto"],
        assembly_mode=OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED,
    )


def _counting_profile(monkeypatch) -> tuple[object, list[int]]:
    """Graph-format profile whose serialisation is counted."""
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    calls: list[int] = []
    original = type(profile).serialize_graph_for_prompt

    def counted(self, graph):
        calls.append(1)
        return original(self, graph)

    monkeypatch.setattr(type(profile), "serialize_graph_for_prompt", counted)
    return profile, calls


def test_repeated_calls_serialise_once(monkeypatch) -> None:
    profile, calls = _counting_profile(monkeypatch)
    snapshot = _snapshot(_graph())

    first = snapshot.prompt_chapter(profile)
    for _ in range(9):
        assert snapshot.prompt_chapter(profile) == first

    assert len(calls) == 1, f"expected one serialisation, got {len(calls)}"


def test_reassigning_the_graph_invalidates_the_memo(monkeypatch) -> None:
    profile, calls = _counting_profile(monkeypatch)
    snapshot = _snapshot(_graph())

    first = snapshot.prompt_chapter(profile)
    snapshot.graph = _graph(5)
    second = snapshot.prompt_chapter(profile)

    assert len(calls) == 2
    assert first != second, "a different graph must produce a different chapter"


def test_explicit_invalidation_forces_a_rebuild(monkeypatch) -> None:
    profile, calls = _counting_profile(monkeypatch)
    graph = _graph()
    snapshot = _snapshot(graph)

    snapshot.prompt_chapter(profile)
    # In-place mutation is outside the read-only contract, so the caller is
    # responsible for saying so; this is the escape hatch that makes it safe.
    graph.add((EX.Extra, RDF.type, OWL.Class))
    snapshot.invalidate_prompt_cache()
    rebuilt = snapshot.prompt_chapter(profile)

    assert len(calls) == 2
    assert "Extra" in rebuilt


def test_different_wire_formats_do_not_collide(monkeypatch) -> None:
    profile, calls = _counting_profile(monkeypatch)
    snapshot = _snapshot(_graph())

    turtle = snapshot.prompt_chapter(profile)
    jsonld_profile = get_graph_format_profile(LLMGraphFormat.JSONLD)
    jsonld = snapshot.prompt_chapter(jsonld_profile)

    assert turtle != jsonld
    assert len(calls) == 2


def test_budget_is_part_of_the_memo_key(monkeypatch) -> None:
    """A shared snapshot must not serve one unit's budget to the next.

    The snapshot is shared by reference across the whole fan-out. With the
    budget outside the key, whichever caller arrived first would fix the chapter
    for everyone, and changing ONTOLOGY_CONTEXT_MAX_TRIPLES would appear to do
    nothing after the first call.
    """
    profile, calls = _counting_profile(monkeypatch)
    # The shared `_graph` helper is all labels and types, which condensing must
    # never drop -- so it would shrink by nothing. Add glosses to give the
    # condenser something it is allowed to remove.
    graph = _graph(40)
    for index in range(40):
        graph.add((EX[f"C{index}"], RDFS.comment, Literal(f"About class {index}.")))
    snapshot = _snapshot(graph)

    uncapped = snapshot.prompt_chapter(profile)
    capped = snapshot.prompt_chapter(profile, max_triples=90)

    assert len(calls) == 2, "a different budget must miss the memo"
    assert capped != uncapped
    assert len(capped) < len(uncapped)

    # ...and the same budget still hits it.
    assert snapshot.prompt_chapter(profile, max_triples=90) == capped
    assert len(calls) == 2


def test_memo_does_not_grow_without_bound() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    snapshot = _snapshot(_graph())

    for size in range(1, 6):
        snapshot.graph = _graph(size)
        snapshot.prompt_chapter(profile)

    assert len(snapshot._prompt_cache) == 1
