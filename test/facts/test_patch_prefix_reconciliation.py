"""A model-declared prefix never overrides the binding it was shown.

A fix payload carries its own prefix declarations, and a model writing one is
re-transcribing namespaces it read in the prompt. When the transcription
disagrees with the graph under repair, every term under that prefix expands to
an IRI no catalog declares: the payload fails the term check and is rolled back
with whatever it carried -- so a measurement can be produced by the model and
still be absent from the output, while the subject that referenced it survives
as a stub.
"""

from rdflib import Literal, URIRef

from ontocast.onto.model import TripleFix
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.critic_patch import compile_critic_fixes

CD = "https://growgraph.dev/facts/"
QQVAL = "https://growgraph.dev/ontologies/qqval#"
OBS = "https://growgraph.dev/ontologies/observation#"
QUDT = "http://qudt.org/schema/qudt/"

_TTL = f"""
@prefix cd: <{CD}> .
@prefix qqval: <{QQVAL}> .
@prefix obs: <{OBS}> .
@prefix qudt: <{QUDT}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

cd:red_shift_obs a obs:QuantitativeObservation ;
    rdfs:label "PL red shift observation" .
"""


def _inserted(update) -> set[tuple]:
    """Every triple the compiled patch would insert, as ``(s, p, o)``."""
    return {
        (str(s), str(p), o)
        for op in update.triple_operations
        if op.type == "insert"
        for s, p, o in op.graph
    }


def _graph() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_TTL, format="turtle")
    return graph


def _add(correct: str) -> TripleFix:
    return TripleFix(
        text_fragment="a red shift of ~15 meV",
        action="ADD",
        severity="important",
        incorrect_value="",
        correct_value=correct,
        triple_ids=[],
        explanation="add the measurement",
    )


_JSONLD_WRONG_QQVAL = (
    '{"@context": {"qqval": "%s", "qudt": "%s", "cd": "%s"}, '
    '"@id": "cd:red_shift_value", "@type": "qqval:QualifiedQuantityValue", '
    '"qudt:numericValue": {"@value": "15", "@type": "http://www.w3.org/2001/XMLSchema#decimal"}}'
) % (OBS, QUDT, CD)


def test_jsonld_payload_rebinding_a_known_prefix_is_corrected() -> None:
    graph = _graph()
    compiled = compile_critic_fixes([_add(_JSONLD_WRONG_QQVAL)], graph)

    assert compiled.applied, "the fix must compile, not be discarded"
    assert compiled.update is not None
    inserted = _inserted(compiled.update)
    assert (
        f"{CD}red_shift_value",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        URIRef(f"{QQVAL}QualifiedQuantityValue"),
    ) in inserted, "the unit graph's qqval binding wins over the payload's"
    assert not any(
        str(obj) == f"{OBS}QualifiedQuantityValue" for _, _, obj in inserted
    ), "the hallucinated namespace must not reach the graph"


def test_jsonld_payload_keeps_an_undeclared_new_namespace() -> None:
    """Overriding is for prefixes the graph binds; a new one is left alone."""
    graph = _graph()
    payload = (
        '{"@context": {"ext": "https://example.org/ext#", "cd": "%s"}, '
        '"@id": "cd:red_shift_value", "@type": "ext:Custom"}' % CD
    )
    compiled = compile_critic_fixes([_add(payload)], graph)

    assert compiled.update is not None
    objects = {str(obj) for _, _, obj in _inserted(compiled.update)}
    assert "https://example.org/ext#Custom" in objects


def test_jsonld_term_definitions_are_not_treated_as_prefixes() -> None:
    """A context entry mapping a term to a CURIE is not a namespace binding."""
    graph = _graph()
    payload = (
        '{"@context": {"cd": "%s", "qqval": "%s", '
        '"rdfs": "http://www.w3.org/2000/01/rdf-schema#", "label": "rdfs:label"}, '
        '"@id": "cd:red_shift_value", "@type": "qqval:QualifiedQuantityValue", '
        '"label": "15 meV"}' % (CD, QQVAL)
    )
    compiled = compile_critic_fixes([_add(payload)], graph)

    assert compiled.update is not None
    inserted = _inserted(compiled.update)
    assert (
        f"{CD}red_shift_value",
        "http://www.w3.org/2000/01/rdf-schema#label",
        Literal("15 meV"),
    ) in inserted


def test_turtle_payload_redeclaring_a_known_prefix_is_corrected() -> None:
    graph = _graph()
    payload = (
        f"@prefix qqval: <{OBS}> .\n"
        "cd:red_shift_value a qqval:QualifiedQuantityValue .\n"
    )
    compiled = compile_critic_fixes([_add(payload)], graph)

    assert compiled.update is not None
    objects = {str(obj) for _, _, obj in _inserted(compiled.update)}
    assert f"{QQVAL}QualifiedQuantityValue" in objects
    assert f"{OBS}QualifiedQuantityValue" not in objects


def test_known_prefix_map_overrides_a_conflicting_jsonld_context() -> None:
    """The render path applies the same rule through ``set_known_prefixes``."""
    RDFGraph.set_known_prefixes({"qqval": QQVAL, "cd": CD})
    try:
        graph = RDFGraph._from_str(_JSONLD_WRONG_QQVAL)
    finally:
        RDFGraph.set_known_prefixes(None)

    assert (
        URIRef(f"{CD}red_shift_value"),
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        URIRef(f"{QQVAL}QualifiedQuantityValue"),
    ) in graph
