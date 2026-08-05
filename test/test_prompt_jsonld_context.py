"""The prompt @context lists only what the payload uses.

Every binding on a graph reaches the namespace manager, rdflib's built-ins
included. Emitting all of them put ~15 vocabularies (brick, csvw, dcat, odrl,
qb, void, wgs, …) in front of the model that nothing in the payload
references — noise at best, and at worst a menu the renderer reads as
available.
"""

from __future__ import annotations

import json

from ontocast.onto.rdfgraph import RDFGraph

_TURTLE = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <https://example.org/o#> .

ex:Sample a owl:Class ;
    rdfs:label "sample"@en .
"""


def _payload() -> dict:
    graph = RDFGraph()
    graph.parse(data=_TURTLE, format="turtle")
    # Bindings a real catalog picks up but this payload never mentions.
    graph.bind("brick", "https://brickschema.org/schema/Brick#")
    graph.bind("csvw", "http://www.w3.org/ns/csvw#")
    graph.bind("odrl", "http://www.w3.org/ns/odrl/2/")
    return json.loads(graph.serialize_compact_jsonld_for_prompt())


def test_unused_bindings_are_dropped() -> None:
    context = _payload()["@context"]

    for unused in ("brick", "csvw", "odrl"):
        assert unused not in context


def test_referenced_prefixes_are_kept() -> None:
    context = _payload()["@context"]

    assert context["ex"] == "https://example.org/o#"
    for referenced in ("owl", "rdfs"):
        assert referenced in context


def test_every_emitted_prefix_resolves() -> None:
    """A compact IRI in the payload must be expandable from the context."""
    payload = _payload()
    context = payload["@context"]

    for node in payload["@graph"]:
        for key, value in node.items():
            for token in (key, value if isinstance(value, str) else ""):
                head, sep, _ = token.partition(":")
                if sep and not token.startswith("@") and not token.startswith("http"):
                    assert head in context, f"{token} has no context entry"
