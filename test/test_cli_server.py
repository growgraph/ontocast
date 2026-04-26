import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDF, URIRef

from ontocast.api.schemas import ProcessResultData
from ontocast.cli.server import (
    _persist_unit_pipeline_outputs,
    _select_unit_facts_ontology_graph,
    parse_ontology_context_mode_param,
    validate_ontology_context_mode,
)
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import OntologyContextConfigError
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox


def test_parse_ontology_context_mode_param_accepts_request_override() -> None:
    result = parse_ontology_context_mode_param(
        "vector_retrieval",
        OntologyContextMode.FULL_TTL,
    )
    assert result == OntologyContextMode.VECTOR_RETRIEVAL


def _tools(vector_store: object | None, patch_retriever: object | None) -> ToolBox:
    return cast(
        ToolBox,
        SimpleNamespace(
            vector_store=vector_store,
            patch_retriever=patch_retriever,
        ),
    )


def test_validate_ontology_context_mode_rejects_vector_without_qdrant() -> None:
    with pytest.raises(OntologyContextConfigError, match="vector_retrieval"):
        validate_ontology_context_mode(
            OntologyContextMode.VECTOR_RETRIEVAL,
            _tools(None, None),
        )


def test_validate_ontology_context_mode_allows_full_ttl_without_vector_store() -> None:
    validate_ontology_context_mode(
        OntologyContextMode.FULL_TTL,
        _tools(None, None),
    )


def test_validate_ontology_context_mode_allows_vector_when_both_set() -> None:
    validate_ontology_context_mode(
        OntologyContextMode.VECTOR_RETRIEVAL,
        _tools(object(), object()),
    )


def test_process_result_data_uses_artifacts_and_deprecates_singular_ontology() -> None:
    payload = ProcessResultData(
        facts="",
        ontology=None,
        ontology_artifacts=[{"iri": "https://example.org/o", "ttl": ""}],
    )
    assert payload.ontology is None
    assert len(payload.ontology_artifacts) == 1


def _graph_with_one_triple(suffix: str) -> RDFGraph:
    graph = RDFGraph()
    subject = URIRef(f"https://example.org/{suffix}")
    graph.add((subject, RDF.type, URIRef("https://example.org/T")))
    return graph


def test_select_unit_facts_ontology_graph_prefers_facts_snapshot() -> None:
    facts_graph = _graph_with_one_triple("facts")
    onto_graph = _graph_with_one_triple("onto")
    facts_result = SimpleNamespace(
        ontology_snapshot=Ontology(
            graph=facts_graph, iri="https://example.org/facts-onto"
        ),
    )
    onto_result = SimpleNamespace(
        current_ontology=Ontology(graph=onto_graph, iri="https://example.org/onto"),
    )

    selected = _select_unit_facts_ontology_graph(onto_result, facts_result)

    assert selected is facts_graph


def test_select_unit_facts_ontology_graph_falls_back_to_onto_result() -> None:
    onto_graph = _graph_with_one_triple("onto")
    onto_result = SimpleNamespace(
        current_ontology=Ontology(graph=onto_graph, iri="https://example.org/onto"),
    )

    selected = _select_unit_facts_ontology_graph(onto_result, None)

    assert len(selected) > 0
    assert set(selected) == set(onto_result.current_ontology.graph)


def test_persist_unit_pipeline_outputs_uses_facts_snapshot_for_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts_graph = _graph_with_one_triple("facts")
    facts_result = SimpleNamespace(
        ontology_snapshot=Ontology(
            graph=facts_graph, iri="https://example.org/facts-onto"
        ),
        content_unit=ContentUnit(
            text="unit",
            index=0,
            doc_iri=URIRef("https://example.org/doc"),
        ),
    )
    onto_result = SimpleNamespace(
        current_ontology=Ontology(graph=RDFGraph(), iri="https://example.org/onto"),
    )
    state = AgentState(input_text="x")
    captured: dict[str, RDFGraph] = {}

    class _Aggregator:
        def postprocess_facts_units(
            self,
            units: list[ContentUnit],
            ontology_graph: RDFGraph,
        ) -> RDFGraph:
            captured["ontology_graph"] = ontology_graph
            graph = RDFGraph()
            graph += units[0].graph
            return graph

    tools = cast(ToolBox, SimpleNamespace(aggregator=_Aggregator()))
    monkeypatch.setattr("ontocast.cli.server.serialize_agent_state", lambda *_: None)

    asyncio.run(
        _persist_unit_pipeline_outputs(
            state=state,
            onto_result=onto_result,
            facts_result=facts_result,
            tools=tools,
        )
    )

    assert captured["ontology_graph"] is facts_graph
