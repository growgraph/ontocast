"""Tests for GraphFormatProfile and format-bound canonical LLM parsers."""

import json

import pytest

from ontocast.onto.enum import LLMGraphFormat
from ontocast.onto.llm_graph_payload import (
    _coerce_jsonld_graph_payload,
    _coerce_turtle_graph_payload,
    coerce_llm_graph_wire,
    llm_graph_format_ctx,
)
from ontocast.onto.model import (
    FactsCritiqueReport,
    FactsRenderReport,
    GraphUpdateRenderReport,
    OntologyRenderReport,
    Suggestions,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import GraphUpdate
from ontocast.prompt.graph_format import get_graph_format_profile
from ontocast.prompt.llm_json_schema import schema_for_model


@pytest.mark.parametrize("fmt", list(LLMGraphFormat))
def test_fresh_output_instruction_single_format(fmt: LLMGraphFormat) -> None:
    profile = get_graph_format_profile(fmt)
    text = profile.render_fresh_output_instruction(target="facts")
    assert " OR " not in text
    if fmt == LLMGraphFormat.TURTLE:
        assert "Turtle" in text
        assert "@prefix" in text or "turtle" in text.lower()
    else:
        assert "JSON-LD" in text
        assert "Never use Turtle syntax" in text


def test_turtle_update_instruction_names_triple_op_graph() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    text = profile.render_update_output_instruction()
    assert "TripleOp.graph" in text
    assert "Turtle string" in text


def test_jsonld_operational_guidelines_forbid_caret_caret() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.JSONLD)
    text = profile.facts_operational_guidelines(
        facts_namespace="https://growgraph.dev/facts/",
        domain_ontologies_clause="the domain ontology",
    )
    assert "^^ syntax" in text


def test_turtle_schema_semantic_graph_is_string() -> None:
    schema = schema_for_model(FactsRenderReport, LLMGraphFormat.TURTLE)
    sg = schema["properties"]["semantic_graph"]
    assert sg.get("type") == "string"


def test_jsonld_schema_semantic_graph_is_object() -> None:
    schema = schema_for_model(FactsRenderReport, LLMGraphFormat.JSONLD)
    sg = schema["properties"]["semantic_graph"]
    assert sg.get("type") == "object"


def test_turtle_schema_triple_op_graph_is_string() -> None:
    schema = schema_for_model(GraphUpdateRenderReport, LLMGraphFormat.TURTLE)
    triple_op = schema["$defs"]["TripleOp"]
    assert triple_op["properties"]["graph"].get("type") == "string"


def test_strict_turtle_rejects_dict() -> None:
    with pytest.raises(ValueError, match="turtle"):
        _coerce_turtle_graph_payload({"@context": {}, "@graph": []})


def test_strict_jsonld_rejects_turtle_string() -> None:
    with pytest.raises(ValueError, match="jsonld"):
        _coerce_jsonld_graph_payload(
            "@prefix ex: <http://example.org/> . ex:a ex:b ex:c ."
        )


def test_coerce_llm_graph_wire_uses_context() -> None:
    token = llm_graph_format_ctx.set(LLMGraphFormat.JSONLD)
    try:
        graph = coerce_llm_graph_wire(
            {
                "@context": {"ex": "http://example.org/"},
                "@graph": [{"@id": "ex:item", "@type": "ex:Thing"}],
            }
        )
        assert len(graph) >= 1
    finally:
        llm_graph_format_ctx.reset(token)


def test_parse_report_jsonld_facts_is_canonical() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.JSONLD)
    payload = {
        "semantic_graph": {
            "@context": {
                "ex": "http://example.org/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            },
            "@graph": [
                {
                    "@id": "ex:item",
                    "@type": "ex:Thing",
                }
            ],
        },
        "ontology_relevance_score": 90,
        "triples_generation_score": 91,
        "external_evidence_request": {
            "initiate_search": False,
            "rationale": "",
            "query_hints": [],
        },
    }
    report = profile.parse_report(FactsRenderReport, json.dumps(payload))
    assert isinstance(report, FactsRenderReport)
    assert len(report.semantic_graph) >= 1


def test_parse_report_turtle_graph_update_has_generate_sparql() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    payload = {
        "graph_update": {
            "triple_operations": [
                {
                    "type": "insert",
                    "graph": (
                        "@prefix ex: <http://example.org/> .\nex:item a ex:Thing ."
                    ),
                }
            ],
            "sparql_operations": [],
        },
        "external_evidence_request": {
            "initiate_search": False,
            "rationale": "",
            "query_hints": [],
        },
    }
    report = profile.parse_report(GraphUpdateRenderReport, json.dumps(payload))
    assert isinstance(report, FactsRenderReport) is False
    assert isinstance(report.graph_update, GraphUpdate)
    queries = report.graph_update.generate_sparql_queries()
    assert len(queries) == 1
    assert "INSERT DATA" in queries[0]


def test_suggestions_from_parsed_facts_critique() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    payload = {
        "success": False,
        "score": 50,
        "actionable_triple_fixes": [],
        "systemic_critique_summary": "needs work",
        "external_evidence_request": {
            "initiate_search": False,
            "rationale": "",
            "query_hints": [],
        },
    }
    critique = profile.parse_report(FactsCritiqueReport, json.dumps(payload))
    suggestions = Suggestions.from_critique_report(critique)
    assert suggestions.systemic_critique_summary == "needs work"


def test_legacy_nested_facts_report_flattens_on_parse() -> None:
    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    payload = {
        "facts_report": {
            "semantic_graph": "@prefix ex: <http://example.org/> .\nex:item a ex:Thing .",
            "ontology_relevance_score": 80,
            "triples_generation_score": 85,
        },
        "external_evidence_request": {
            "initiate_search": False,
            "rationale": "",
            "query_hints": [],
        },
    }
    report = profile.parse_report(FactsRenderReport, json.dumps(payload))
    assert len(report.semantic_graph) >= 1


def test_serialize_compact_jsonld_for_prompt() -> None:
    from rdflib import URIRef

    g = RDFGraph()
    ex = "http://example.org/"
    g.bind("ex", ex)
    g.add((URIRef(f"{ex}item"), URIRef(f"{ex}type"), URIRef(f"{ex}Thing")))
    text = g.serialize_compact_jsonld_for_prompt()
    data = json.loads(text)
    assert "@context" in data
    assert "@graph" in data
    assert any(n.get("@id") == "ex:item" for n in data["@graph"])


def test_parsed_graph_update_applies_via_unit_facts_state() -> None:
    from rdflib import URIRef

    from ontocast.onto.content_unit import ContentUnit
    from ontocast.onto.ontology import Ontology
    from ontocast.onto.unit_states import UnitFactsState

    profile = get_graph_format_profile(LLMGraphFormat.TURTLE)
    payload = {
        "graph_update": {
            "triple_operations": [
                {
                    "type": "insert",
                    "graph": (
                        "@prefix ex: <http://example.org/> .\nex:bob a ex:Person ."
                    ),
                }
            ],
            "sparql_operations": [],
        },
        "external_evidence_request": {
            "initiate_search": False,
            "rationale": "",
            "query_hints": [],
        },
    }
    report = profile.parse_report(GraphUpdateRenderReport, json.dumps(payload))
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="Bob is a person.",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
        ),
        ontology_snapshot=Ontology(iri="https://example.com/onto"),
    )
    state.facts_updates.append(report.graph_update)
    state.update_facts()
    assert len(state.content_unit.graph) >= 1
    assert state.facts_updates == []


def test_format_instructions_no_dual_or_on_graph_fields() -> None:
    for fmt in LLMGraphFormat:
        profile = get_graph_format_profile(fmt)
        instructions = profile.format_instructions(OntologyRenderReport)
        assert "Turtle string OR" not in instructions
        assert "OR a compact JSON-LD" not in instructions
