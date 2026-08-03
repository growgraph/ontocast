import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDF, URIRef
from starlette.testclient import TestClient

from ontocast.api import app as app_module
from ontocast.api.app import create_app
from ontocast.api.parse import (
    parse_max_visits_param,
    parse_ontology_context_mode_param,
    resolve_ontology_context_mode,
)
from ontocast.api.process_helpers import (
    calculate_recursion_limit,
    persist_unit_pipeline_outputs,
    select_unit_facts_ontology_graph,
)
from ontocast.api.process_request import (
    ParsedProcessRequest,
    build_agent_state_from_parsed,
)
from ontocast.api.responses import ontology_context_config_error_response
from ontocast.api.schemas import ProcessResultData
from ontocast.config import ServerConfig
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.docling_helpers import plain_text_to_docling_doc
from ontocast.onto.enum import OntologyContextMode
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.retrieval_capabilities import (
    OntologyContextConfigError,
    VectorStoreUnavailableError,
    validate_ontology_context_mode,
)
from ontocast.onto.state import AgentState
from ontocast.toolbox import ToolBox


def test_parse_ontology_context_mode_param_accepts_request_override() -> None:
    result = parse_ontology_context_mode_param(
        "selected_vector_search_ontology",
        OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
    )
    assert result == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY


def test_resolve_ontology_context_mode_forces_fixed_mode_when_id_provided() -> None:
    result = resolve_ontology_context_mode(
        OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        "catalog-finance-v3",
    )
    assert result == OntologyContextMode.FIXED_SINGLE_ONTOLOGY


def test_resolve_ontology_context_mode_keeps_requested_mode_when_id_missing() -> None:
    result = resolve_ontology_context_mode(
        OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
        "   ",
    )
    assert result == OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY


def test_parse_max_visits_param_accepts_positive_integer_override() -> None:
    assert parse_max_visits_param("3", default=1) == 3


def test_parse_max_visits_param_uses_default_when_missing() -> None:
    assert parse_max_visits_param(None, default=2) == 2


def test_parse_max_visits_param_rejects_zero_or_negative_values() -> None:
    with pytest.raises(ValueError, match="max_visits must be an integer >= 1"):
        parse_max_visits_param("0", default=1)
    with pytest.raises(ValueError, match="max_visits must be an integer >= 1"):
        parse_max_visits_param("-2", default=1)


def test_parse_max_visits_param_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="max_visits must be an integer >= 1"):
        parse_max_visits_param("abc", default=1)


def test_calculate_recursion_limit_uses_per_request_max_visits() -> None:
    server_config = ServerConfig(
        max_visits_per_node=1,
        base_recursion_limit=10,
        estimated_chunks=10,
    )
    default_limit = calculate_recursion_limit(5, server_config)
    override_limit = calculate_recursion_limit(5, server_config, max_visits_per_node=4)
    assert default_limit == 50
    assert override_limit == 200


def test_build_agent_state_from_parsed_sets_max_visits() -> None:
    parsed = ParsedProcessRequest(
        files_dict={"input.json": b'{"text": "hello"}'},
        max_visits=6,
        strip_provenance=False,
        ontology_user_instruction="",
        ontology_selection_user_instruction="",
        facts_user_instruction="",
        ontology_context_fixed_ontology_id="onto-1",
        render_mode=None,
        llm_graph_format=None,
        ontology_context_mode_value=OntologyContextMode.FIXED_SINGLE_ONTOLOGY,
        target_sections=None,
        summarize_sections=None,
        summary_max_sentences=5,
        document_type_hint=None,
        section_schema_id=None,
        document_metadata={},
    )
    state = build_agent_state_from_parsed(
        parsed,
        server_config=ServerConfig(max_visits_per_node=2),
        resolved_tenant="t",
        resolved_project="p",
        max_chunks=1,
    )
    assert state.max_visits == 6


def test_build_agent_state_from_parsed_sets_document_metadata() -> None:
    parsed = ParsedProcessRequest(
        files_dict={"input.json": b'{"text": "hello"}'},
        max_visits=1,
        strip_provenance=False,
        ontology_user_instruction="",
        ontology_selection_user_instruction="",
        facts_user_instruction="",
        ontology_context_fixed_ontology_id="",
        render_mode=None,
        llm_graph_format=None,
        ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        target_sections=None,
        summarize_sections=None,
        summary_max_sentences=5,
        document_type_hint=None,
        section_schema_id=None,
        document_metadata={
            "doi": "10.1234/example",
            "identifiers": [{"scheme": "erp:doc", "value": "INV-1"}],
        },
    )
    state = build_agent_state_from_parsed(
        parsed,
        server_config=ServerConfig(max_visits_per_node=1),
        resolved_tenant="t",
        resolved_project="p",
        max_chunks=1,
    )
    assert state.document_metadata["doi"] == "10.1234/example"
    assert state.document_metadata["identifiers"][0]["value"] == "INV-1"


def _tools(vector_store: object | None, patch_retriever: object | None) -> ToolBox:
    is_ready = vector_store is not None and patch_retriever is not None
    return cast(
        ToolBox,
        SimpleNamespace(
            vector_store=vector_store,
            patch_retriever=patch_retriever,
            vector_store_last_error=None,
            is_vector_store_ready=lambda: is_ready,
        ),
    )


def test_validate_ontology_context_mode_rejects_vector_without_qdrant() -> None:
    with pytest.raises(
        OntologyContextConfigError,
        match="selected_vector_search_ontology",
    ):
        validate_ontology_context_mode(
            OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
            _tools(None, None),
        )


def test_validate_ontology_context_mode_allows_selected_single_without_vector_store() -> (
    None
):
    validate_ontology_context_mode(
        OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        _tools(None, None),
    )


def test_validate_ontology_context_mode_allows_vector_when_both_set() -> None:
    validate_ontology_context_mode(
        OntologyContextMode.SELECTED_VECTOR_SEARCH_ONTOLOGY,
        _tools(object(), object()),
    )


def test_ontology_context_error_response_maps_vector_unavailable_to_409() -> None:
    response = ontology_context_config_error_response(
        VectorStoreUnavailableError("vector store unavailable")
    )
    assert response.status_code == 409
    assert b"VECTOR_STORE_UNAVAILABLE" in response.body


def test_ontology_context_error_response_keeps_generic_config_error_as_400() -> None:
    response = ontology_context_config_error_response(
        OntologyContextConfigError("generic context error")
    )
    assert response.status_code == 400


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
    from ontocast.onto.enum import OntologyAssemblyMode
    from ontocast.onto.ontology_snapshot import OntologySnapshot

    facts_graph = _graph_with_one_triple("facts")
    onto_graph = _graph_with_one_triple("onto")
    facts_result = SimpleNamespace(
        ontology_snapshot=OntologySnapshot.from_graph(
            facts_graph,
            source_iris=["https://example.org/facts-onto"],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
            strip_headers=False,
        ),
    )
    onto_result = SimpleNamespace(
        fresh_ontology=Ontology(graph=onto_graph, iri="https://example.org/onto"),
        working_graph=RDFGraph(),
        ontology_snapshot=OntologySnapshot.empty(),
    )

    selected = select_unit_facts_ontology_graph(onto_result, facts_result)

    assert selected is facts_result.ontology_snapshot.graph


def test_select_unit_facts_ontology_graph_falls_back_to_onto_result() -> None:
    from ontocast.onto.ontology_snapshot import OntologySnapshot

    onto_graph = _graph_with_one_triple("onto")
    onto_result = SimpleNamespace(
        fresh_ontology=Ontology(graph=onto_graph, iri="https://example.org/onto"),
        working_graph=RDFGraph(),
        ontology_snapshot=OntologySnapshot.empty(),
    )

    selected = select_unit_facts_ontology_graph(onto_result, None)

    assert len(selected) > 0
    assert set(selected) == set(onto_result.fresh_ontology.graph)


def test_persist_unit_pipeline_outputs_uses_facts_snapshot_for_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ontocast.onto.enum import OntologyAssemblyMode
    from ontocast.onto.ontology_snapshot import OntologySnapshot

    facts_graph = _graph_with_one_triple("facts")
    facts_result = SimpleNamespace(
        ontology_snapshot=OntologySnapshot.from_graph(
            facts_graph,
            source_iris=["https://example.org/facts-onto"],
            assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
            strip_headers=False,
        ),
        content_unit=ContentUnit(
            text="unit",
            index=0,
            doc_iri=URIRef("https://example.org/doc"),
        ),
    )
    onto_result = SimpleNamespace(
        fresh_ontology=None,
        working_graph=RDFGraph(),
        ontology_snapshot=OntologySnapshot.empty(),
    )
    state = AgentState(docling_doc=plain_text_to_docling_doc("x", "doc"))
    captured: dict[str, object] = {}

    class _Aggregator:
        def postprocess_facts_units(
            self,
            units: list[ContentUnit],
            ontology_graph: RDFGraph,
            **kwargs,
        ) -> RDFGraph:
            captured["ontology_graph"] = ontology_graph
            captured["kwargs"] = kwargs
            graph = RDFGraph()
            graph += units[0].graph
            return graph

    tools = cast(ToolBox, SimpleNamespace(aggregator=_Aggregator()))
    monkeypatch.setattr(
        "ontocast.api.process_helpers.serialize_agent_state", lambda *_: None
    )

    asyncio.run(
        persist_unit_pipeline_outputs(
            state=state,
            onto_result=onto_result,
            facts_result=facts_result,
            tools=tools,
        )
    )

    ontology_graph = captured["ontology_graph"]
    assert isinstance(ontology_graph, RDFGraph)
    assert set(ontology_graph) == set(facts_graph)


def _match_test_app(monkeypatch: pytest.MonkeyPatch):
    class _FakeAligner:
        def __init__(self, embedding_model: str, similarity_threshold: float) -> None:
            pass

        def align_graphs(self, graphs, *, regime):
            class _Result:
                def model_dump(self, mode: str = "python") -> dict:
                    return {
                        "regime": str(regime),
                        "similarity_threshold": 0.8,
                        "entity_count": 2,
                        "cluster_count": 1,
                        "clusters": [
                            {
                                "members": [
                                    {
                                        "graph_id": "predicted",
                                        "entity": "https://predicted.example/a",
                                        "similarity": 1.0,
                                    },
                                    {
                                        "graph_id": "gt",
                                        "entity": "https://gt.example/a",
                                        "similarity": 1.0,
                                    },
                                ]
                            }
                        ],
                    }

            return _Result()

    class _FakeEvaluator:
        def evaluate(self, **_kwargs):
            from ontocast.tool.agg.match_models import MatchMetrics

            return MatchMetrics(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                true_positives=1,
                false_positives=0,
                false_negatives=0,
                predicted_count=1,
                ground_truth_count=1,
                entity_precision=1.0,
                entity_recall=1.0,
                entity_f1=1.0,
                entity_true_positives=1,
                entity_false_positives=0,
                entity_false_negatives=0,
                domain_entity_matches=1,
                fact_precision=1.0,
                fact_recall=1.0,
                fact_f1=1.0,
                fact_true_positives=1,
                fact_false_positives=0,
                fact_false_negatives=0,
                fact_predicted_count=1,
                fact_ground_truth_count=1,
            )

    monkeypatch.setattr(app_module, "TripleSetEvaluator", _FakeEvaluator)
    monkeypatch.setattr(
        app_module,
        "derive_pair_matches",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        app_module, "create_agent_graph", lambda _tools: SimpleNamespace()
    )
    tools = cast(
        ToolBox,
        SimpleNamespace(
            get_entity_aligner=lambda embedding_model, similarity_threshold: (
                _FakeAligner(embedding_model, similarity_threshold)
            ),
        ),
    )
    return create_app(
        tools=tools,
        server_config=ServerConfig(),
        active_tenant="tenant-a",
        active_project="project-a",
    )


def test_align_entities_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _match_test_app(monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/match/entities",
        json={
            "graphs": [
                {
                    "id": "predicted",
                    "graph": (
                        "@prefix ex: <https://predicted.example/> . "
                        "ex:a <https://pred.example/relatedTo> ex:b ."
                    ),
                },
                {
                    "id": "gt",
                    "graph": (
                        "@prefix ex: <https://gt.example/> . "
                        "ex:a <https://pred.example/relatedTo> ex:b ."
                    ),
                },
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["cluster_count"] == 1


def test_evaluate_match_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _match_test_app(monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/match/evaluate",
        json={
            "predicted_graph": (
                "@prefix ex: <https://predicted.example/> . "
                "ex:a <https://pred.example/relatedTo> ex:b ."
            ),
            "gt_graph": (
                "@prefix ex: <https://gt.example/> . "
                "ex:a <https://pred.example/relatedTo> ex:b ."
            ),
            "entity_matches": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["f1"] == 1.0


def test_derive_matches_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module, "create_agent_graph", lambda _tools: SimpleNamespace()
    )
    tools = cast(ToolBox, SimpleNamespace())
    app = create_app(
        tools=tools,
        server_config=ServerConfig(),
        active_tenant="tenant-a",
        active_project="project-a",
    )
    client = TestClient(app)
    response = client.post(
        "/match/derive-matches",
        json={
            "clusters": [
                {
                    "members": [
                        {
                            "graph_id": "predicted",
                            "entity": "http://predicted.example/a",
                            "similarity": 1.0,
                        },
                        {
                            "graph_id": "gt",
                            "entity": "http://gt.example/a",
                            "similarity": 1.0,
                        },
                    ]
                }
            ],
            "predicted_graph_id": "predicted",
            "gt_graph_id": "gt",
        },
    )
    assert response.status_code == 200
    matches = response.json()["data"]["entity_matches"]
    assert len(matches) == 1
    assert matches[0]["predicted_entity"] == "http://predicted.example/a"


def test_parse_document_metadata_param_accepts_dict_and_json() -> None:
    from ontocast.api.parse import parse_document_metadata_param

    assert parse_document_metadata_param(None) == {}
    assert parse_document_metadata_param("") == {}
    assert parse_document_metadata_param({"doi": "10.1/x"}) == {"doi": "10.1/x"}
    assert parse_document_metadata_param('{"title": "Report"}') == {"title": "Report"}


def test_parse_document_metadata_param_rejects_non_object() -> None:
    from ontocast.api.parse import parse_document_metadata_param

    with pytest.raises(ValueError, match="document_metadata must be a JSON object"):
        parse_document_metadata_param("[1, 2]")


def test_expand_input_to_states_filename_fallback(tmp_path) -> None:
    from ontocast.api.process_helpers import expand_input_to_states
    from ontocast.config import Config

    doc = tmp_path / "annual-report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    config = Config()
    states = expand_input_to_states(
        doc,
        config=config,
        head_chunks=1,
        ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        tenant="t",
        project="p",
        document_metadata=None,
    )
    assert len(states) == 1
    assert states[0].document_metadata == {"title": "annual-report.pdf"}


def test_expand_input_to_states_keeps_explicit_metadata(tmp_path) -> None:
    from ontocast.api.process_helpers import expand_input_to_states
    from ontocast.config import Config

    doc = tmp_path / "annual-report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    config = Config()
    states = expand_input_to_states(
        doc,
        config=config,
        head_chunks=1,
        ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
        tenant="t",
        project="p",
        document_metadata={"doi": "10.1234/x", "title": "Custom"},
    )
    assert states[0].document_metadata == {"doi": "10.1234/x", "title": "Custom"}


def test_facts_ttl_output_path_and_dump(tmp_path) -> None:
    from rdflib import DCTERMS, Literal

    from ontocast.api.process_helpers import (
        dump_facts_ttl,
        dump_ontology_ttls,
        facts_ttl_output_path,
        ontology_ttl_output_path,
        resolve_batch_output_dirs,
    )
    from ontocast.onto.constants import PROV
    from ontocast.onto.docling_helpers import plain_text_to_docling_doc

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    out_dir = tmp_path / "out"
    facts_dir = tmp_path / "facts"
    onto_dir = tmp_path / "ontologies"

    assert facts_ttl_output_path(src) == tmp_path / "paper.facts.ttl"
    assert facts_ttl_output_path(src, line_number=3) == tmp_path / "paper.L3.facts.ttl"
    assert facts_ttl_output_path(src, output_dir=out_dir) == out_dir / "paper.facts.ttl"
    assert (
        facts_ttl_output_path(src, line_number=3, output_dir=out_dir)
        == out_dir / "paper.L3.facts.ttl"
    )
    assert ontology_ttl_output_path(src) == tmp_path / "paper.ontology.ttl"
    assert (
        ontology_ttl_output_path(src, ontology_id="matsci", output_dir=onto_dir)
        == onto_dir / "paper.matsci.ontology.ttl"
    )
    assert resolve_batch_output_dirs(out_dir, None, None) == (out_dir, out_dir)
    assert resolve_batch_output_dirs(out_dir, facts_dir, onto_dir) == (
        facts_dir,
        onto_dir,
    )
    assert resolve_batch_output_dirs(None, facts_dir, None) == (facts_dir, None)

    state = AgentState(docling_doc=plain_text_to_docling_doc("hello", "doc"))
    state.aggregated_facts = RDFGraph()
    state.aggregated_facts.add((state.doc_iri, DCTERMS.title, Literal("paper.pdf")))
    state.aggregated_facts.add((state.doc_iri, RDF.type, PROV.Entity))
    out = dump_facts_ttl(state, src)
    assert out is not None
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "paper.pdf" in text

    out2 = dump_facts_ttl(state, src, output_dir=out_dir)
    assert out2 == out_dir / "paper.facts.ttl"

    onto_graph = RDFGraph()
    onto_graph.add(
        (
            URIRef("https://example.com/onto#Thing"),
            RDF.type,
            URIRef("http://www.w3.org/2002/07/owl#Class"),
        )
    )
    ontology = Ontology(graph=onto_graph, iri="https://example.com/onto")
    state.reduced_ontology_artifacts = [ontology]
    written = dump_ontology_ttls(state, src, output_dir=onto_dir)
    assert written == [onto_dir / "paper.ontology.ttl"]
    assert written[0].exists()

    second = Ontology(
        graph=onto_graph,
        iri="https://example.com/other",
        ontology_id="other",
    )
    state.reduced_ontology_artifacts = [ontology, second]
    written_multi = dump_ontology_ttls(state, src, output_dir=onto_dir)
    assert {p.name for p in written_multi} == {
        "paper.onto.ontology.ttl",
        "paper.other.ontology.ttl",
    }


def test_cli_serve_process_help() -> None:
    from click.testing import CliRunner

    from ontocast.cli.server import cli

    runner = CliRunner()
    root = runner.invoke(cli, ["--help"])
    assert root.exit_code == 0
    assert "serve" in root.output
    assert "process" in root.output

    serve_help = runner.invoke(cli, ["serve", "--help"])
    assert serve_help.exit_code == 0
    assert "--wipe-vector-store" in serve_help.output
    assert "--input-path" not in serve_help.output

    process_help = runner.invoke(cli, ["process", "--help"])
    assert process_help.exit_code == 0
    assert "--input-path" in process_help.output
    assert "--output-dir" in process_help.output
    assert "--facts-output-dir" in process_help.output
    assert "--ontology-output-dir" in process_help.output
    assert "dcterms:title" in process_help.output


def test_cli_requires_subcommand() -> None:
    from click.testing import CliRunner

    from ontocast.cli.server import cli

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
