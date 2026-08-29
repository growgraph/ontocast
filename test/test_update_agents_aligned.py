"""The two graph-update agents must run the same post-parse hygiene.

Facts updates and ontology updates consume the same
:class:`GraphUpdateRenderReport`, but everything after the parse had drifted
apart: the facts path quarantined invalid typed literals and cleared consumed
findings, and the ontology path did neither -- a malformed literal went from
the model straight into the working graph and on into a compiled SPARQL
UPDATE. These tests hold the two paths together.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import URIRef

from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.model import GraphUpdateRenderReport, Suggestions
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState, UnitOntologyState
from ontocast.tool.atomic import AtomicToolBox
from test.snapshot_helpers import snapshot_from_ontology

pytestmark = pytest.mark.anyio

render_facts_module = importlib.import_module("ontocast.agent.render_facts")
render_ontology_module = importlib.import_module("ontocast.agent.render_ontology")

ONTO_TTL = """
@prefix onto: <https://example.com/onto#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
onto:CompanyOntology a owl:Ontology .
"""

#: A range no XSD parser accepts as a decimal. The model emits these; before
#: this alignment only the facts path caught them.
BAD_LITERAL_TTL = """
@prefix ex: <https://example.com/ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:item ex:value "10-15"^^xsd:decimal .
"""

GOOD_TTL = """
@prefix ex: <https://example.com/ns#> .
ex:item a ex:Thing .
"""

STALE_TTL = """
@prefix ex: <https://example.com/ns#> .
ex:item a ex:Obsolete .
"""


def _graph(ttl: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return graph


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _tools() -> AtomicToolBox:
    async def get_llm_tool(_budget_tracker):
        return object()

    return cast(
        AtomicToolBox,
        SimpleNamespace(
            get_llm_tool=get_llm_tool,
            web_grounding_enabled_for_node=lambda _node: False,
            object_property_literal_check=True,
            property_alias_min_ratio=0.85,
            code_predicates=(),
            citation_vocabulary={},
            quantity_fallback_vocabulary=None,
            acceptance_policy=None,
        ),
    )


def _ontology() -> Ontology:
    graph = RDFGraph()
    graph.parse(data=ONTO_TTL, format="turtle")
    return Ontology(graph=graph, iri="https://example.com/onto")


def _facts_state() -> UnitFactsState:
    unit = ContentUnit(
        text="Alice works for ACME.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
    )
    unit.graph.parse(
        data="@prefix ex: <https://example.com/ns#> . ex:alice ex:worksFor ex:acme .",
        format="turtle",
    )
    return UnitFactsState(
        content_unit=unit,
        ontology_snapshot=snapshot_from_ontology(_ontology()),
    )


def _ontology_state() -> UnitOntologyState:
    ontology = _ontology()
    state = UnitOntologyState(
        content_unit=ContentUnit(
            text="Alice works for ACME.",
            index=0,
            doc_iri=URIRef("https://example.com/doc/d1"),
        ),
        ontology_snapshot=OntologySnapshot(
            graph=ontology.graph, source_iris=[ontology.iri]
        ),
    )
    state.working_graph = ontology.graph.copy()
    state.writable_iris = [ontology.iri]
    return state


def _stub_render(monkeypatch, module, ttl: str) -> None:
    async def fake_call_llm_with_retry(**kwargs):
        return GraphUpdateRenderReport(insert_graph=_graph(ttl))

    monkeypatch.setattr(module, "call_llm_with_retry", fake_call_llm_with_retry)


async def test_both_update_agents_quarantine_an_invalid_typed_literal(
    monkeypatch,
) -> None:
    _stub_render(monkeypatch, render_facts_module, BAD_LITERAL_TTL)
    facts = await render_facts_module.render_facts_update(_facts_state(), _tools())

    _stub_render(monkeypatch, render_ontology_module, BAD_LITERAL_TTL)
    ontology = await render_ontology_module.render_ontology_update(
        _ontology_state(), _tools()
    )

    assert len(facts.quarantined_literal_triples) == 1
    assert len(ontology.quarantined_literal_triples) == 1
    assert (
        facts.quarantined_literal_triples[0].datatype
        == ontology.quarantined_literal_triples[0].datatype
    )


async def test_both_update_agents_consume_findings_and_suggestions(
    monkeypatch,
) -> None:
    facts_state = _facts_state()
    facts_state.suggestions = Suggestions(systemic_critique_summary="fix it")
    _stub_render(monkeypatch, render_facts_module, GOOD_TTL)
    facts = await render_facts_module.render_facts_update(facts_state, _tools())

    ontology_state = _ontology_state()
    ontology_state.suggestions = Suggestions(systemic_critique_summary="fix it")
    _stub_render(monkeypatch, render_ontology_module, GOOD_TTL)
    ontology = await render_ontology_module.render_ontology_update(
        ontology_state, _tools()
    )

    for result in (facts, ontology):
        assert result.deterministic_findings == []
        assert result.suggestions.systemic_critique_summary == ""


def test_insert_hook_never_sees_the_delete_side() -> None:
    """A delete must match the stored triple verbatim or it silently no-ops.

    Previously an ``if op.type == "insert"`` branch repeated in the facts agent;
    the flat wire makes it structural, and this holds it that way.
    """
    from ontocast.agent.update_common import finalize_update_report

    seen: list[RDFGraph] = []

    def hook(graph: RDFGraph):
        seen.append(graph)
        return graph, []

    report = GraphUpdateRenderReport(
        insert_graph=_graph(GOOD_TTL),
        delete_graph=_graph(STALE_TTL),
    )
    update, _ = finalize_update_report(report, insert_hook=hook)

    assert len(seen) == 1
    assert seen[0] is report.insert_graph
    # Deletes run first: removing then re-adding the same triple must not
    # cancel the addition.
    assert [op.type for op in update.triple_operations] == ["delete", "insert"]


def test_empty_sides_contribute_no_operation() -> None:
    report = GraphUpdateRenderReport(insert_graph=_graph(GOOD_TTL))
    assert [op.type for op in report.to_graph_update().triple_operations] == ["insert"]
    assert GraphUpdateRenderReport().to_graph_update().triple_operations == []
