import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from langgraph.graph.state import CompiledStateGraph
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
from ontocast.config import Config, ServerConfig
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
from ontocast.tool.agg.aggregate import AggregationResult
from ontocast.tool.converter import ConverterTool
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


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
        exclude_sections=None,
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
        exclude_sections=None,
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
        ) -> AggregationResult:
            captured["ontology_graph"] = ontology_graph
            captured["kwargs"] = kwargs
            graph = RDFGraph()
            graph += units[0].graph
            return AggregationResult(graph=graph)

    # persist_unit_pipeline_outputs now runs the post-aggregation invariant gate,
    # which reads the facts-validation config the same way the graph node does.
    tools = cast(
        ToolBox,
        SimpleNamespace(
            aggregator=_Aggregator(),
            shapes_catalog=SimpleNamespace(graph=lambda: None),
            config=Config(),
        ),
    )
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
        app_module, "create_agent_graph", lambda _tools, **_kwargs: SimpleNamespace()
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
        app_module, "create_agent_graph", lambda _tools, **_kwargs: SimpleNamespace()
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


def _batch_state(title: str = "paper.pdf") -> AgentState:
    from rdflib import DCTERMS, Literal

    from ontocast.onto.constants import PROV
    from ontocast.onto.docling_helpers import plain_text_to_docling_doc

    state = AgentState(docling_doc=plain_text_to_docling_doc("hello", "doc"))
    state.aggregated_facts = RDFGraph()
    state.aggregated_facts.add((state.doc_iri, DCTERMS.title, Literal(title)))
    state.aggregated_facts.add((state.doc_iri, RDF.type, PROV.Entity))
    return state


def _sample_ontology(iri: str = "https://example.com/onto", **kwargs) -> Ontology:
    graph = RDFGraph()
    graph.add(
        (
            URIRef("https://example.com/onto#Thing"),
            RDF.type,
            URIRef("http://www.w3.org/2002/07/owl#Class"),
        )
    )
    return Ontology(graph=graph, iri=iri, **kwargs)


def test_facts_ttl_output_path_naming(tmp_path) -> None:
    from ontocast.api.process_helpers import facts_ttl_output_path

    src = tmp_path / "paper.pdf"
    out_dir = tmp_path / "out"

    assert facts_ttl_output_path(src) == tmp_path / "paper.facts.ttl"
    assert facts_ttl_output_path(src, line_number=3) == tmp_path / "paper.L3.facts.ttl"
    assert facts_ttl_output_path(src, output_dir=out_dir) == out_dir / "paper.facts.ttl"
    assert (
        facts_ttl_output_path(src, line_number=3, output_dir=out_dir)
        == out_dir / "paper.L3.facts.ttl"
    )


def test_ontology_ttl_output_path_naming(tmp_path) -> None:
    from ontocast.api.process_helpers import ontology_ttl_output_path

    src = tmp_path / "paper.pdf"
    onto_dir = tmp_path / "ontologies"

    assert ontology_ttl_output_path(src) == tmp_path / "paper.ontology.ttl"
    assert (
        ontology_ttl_output_path(src, ontology_id="matsci", output_dir=onto_dir)
        == onto_dir / "paper.matsci.ontology.ttl"
    )


def test_resolve_batch_output_dirs_precedence(tmp_path) -> None:
    from ontocast.api.process_helpers import resolve_batch_output_dirs

    out_dir = tmp_path / "out"
    facts_dir = tmp_path / "facts"
    onto_dir = tmp_path / "ontologies"

    assert resolve_batch_output_dirs(out_dir, None, None) == (out_dir, out_dir)
    assert resolve_batch_output_dirs(out_dir, facts_dir, onto_dir) == (
        facts_dir,
        onto_dir,
    )
    assert resolve_batch_output_dirs(None, facts_dir, None) == (facts_dir, None)


def test_dump_facts_ttl_writes_the_graph(tmp_path) -> None:
    from ontocast.api.process_helpers import dump_facts_ttl

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    out_dir = tmp_path / "out"
    state = _batch_state()

    out = dump_facts_ttl(state, src)
    assert out is not None
    assert out.exists()
    assert "paper.pdf" in out.read_text(encoding="utf-8")

    assert dump_facts_ttl(state, src, output_dir=out_dir) == out_dir / "paper.facts.ttl"


def test_dump_facts_ttl_can_keep_provenance(tmp_path) -> None:
    """The batch dump must be able to emit a traceable graph.

    Stripping stays the default; without the option a batch output carries no
    chunk references at all, so nothing in it can be traced back to a source
    span or re-verified against the document.
    """
    from rdflib import Literal

    from ontocast.api.process_helpers import dump_facts_ttl
    from ontocast.onto.constants import PROV

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    state = _batch_state()
    state.aggregated_facts.add((state.doc_iri, PROV.wasDerivedFrom, Literal("chunk-3")))

    stripped = dump_facts_ttl(state, src, output_dir=tmp_path / "stripped")
    assert stripped is not None
    assert "wasDerivedFrom" not in stripped.read_text(encoding="utf-8")

    kept = dump_facts_ttl(
        state, src, output_dir=tmp_path / "kept", strip_provenance=False
    )
    assert kept is not None
    assert "wasDerivedFrom" in kept.read_text(encoding="utf-8")


def test_dump_run_manifest_records_cost_and_configuration(tmp_path) -> None:
    import json

    from ontocast.api.process_helpers import dump_run_manifest
    from ontocast.config import Config, LLMConfig, LLMProvider, ToolConfig
    from ontocast.onto.token_usage import TokenUsage

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    out_dir = tmp_path / "out"
    state = _batch_state()
    state.reduced_ontology_artifacts = [_sample_ontology()]
    state.budget_tracker.add_usage(
        100, 50, usage=TokenUsage(input_tokens=900, output_tokens=300)
    )
    state.budget_tracker.add_cache_hit(
        10, 5, usage=TokenUsage(input_tokens=40, output_tokens=20)
    )
    state.budget_tracker.add_duration("Render Facts", 1.5)

    config = Config(
        tool_config=ToolConfig(
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA, model_name="kimi-k3", think=True
            )
        )
    )

    out = dump_run_manifest(state, src, config=config, output_dir=out_dir)
    assert out is not None
    assert out == out_dir / "paper.run.json"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "paper.pdf"
    assert payload["llm"] == {
        "provider": "ollama",
        "model_name": "kimi-k3",
        "temperature": 0.0,
        "think": True,
    }
    # Billed and replayed stay distinct all the way to disk -- the whole point
    # of persisting this is comparing runs, and a replay is not a cost.
    assert payload["budget"]["input_tokens"] == 900
    assert payload["budget"]["cached_input_tokens"] == 40
    assert payload["budget"]["node_durations"] == {"Render Facts": 1.5}
    assert payload["facts_triples"] == 2
    assert payload["ontology_triples"] == len(state.reduced_ontology_artifacts[0].graph)


def test_dump_run_manifest_populates_completion_from_facts_loop_telemetry(
    tmp_path,
) -> None:
    """``RunManifest.completion`` reads the same telemetry ``critic`` does.

    All-zero when the completion pass never ran (the library default), and
    reflecting the attempt log once it has -- ``summarize_completion``'s own
    contract, wired into the manifest that ships beside the facts dump.
    """
    import json

    from ontocast.api.process_helpers import dump_run_manifest
    from ontocast.config import Config
    from ontocast.onto.model import LoopAttempt

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")

    state = _batch_state()
    out_off = dump_run_manifest(
        state, src, config=Config(), output_dir=tmp_path / "off"
    )
    assert out_off is not None
    payload_off = json.loads(out_off.read_text(encoding="utf-8"))
    assert payload_off["completion"] == {
        "calls": 0,
        "units": 0,
        "subjects_inserted": 0,
        "subjects_rolled_back": 0,
        "triples_inserted": 0,
        "measurements_recovered": 0,
    }

    state.facts_loop_telemetry[0] = [
        LoopAttempt(
            kind="completion",
            n_fixes_applied=1,
            n_fixes_rolled_back=1,
            n_triples_inserted=3,
            n_measurements_recovered=2,
        )
    ]
    out_on = dump_run_manifest(state, src, config=Config(), output_dir=tmp_path / "on")
    assert out_on is not None
    payload_on = json.loads(out_on.read_text(encoding="utf-8"))
    assert payload_on["completion"] == {
        "calls": 1,
        "units": 1,
        "subjects_inserted": 1,
        "subjects_rolled_back": 1,
        "triples_inserted": 3,
        "measurements_recovered": 2,
    }


def test_dump_run_manifest_uses_the_line_number_for_jsonl_inputs(tmp_path) -> None:
    from ontocast.api.process_helpers import dump_run_manifest
    from ontocast.config import Config

    src = tmp_path / "corpus.jsonl"
    src.write_bytes(b"x")
    out = dump_run_manifest(_batch_state(), src, config=Config(), line_number=3)
    assert out == tmp_path / "corpus.L3.run.json"


def test_dump_validation_report_carries_unit_repairs_and_failures(tmp_path) -> None:
    """A predicate the machine substituted is invisible in the TTL; name it.

    ``gate_repairs`` covered the document-level gate only. The per-unit
    passes (alias rewrites, literal coercions) were logged and dropped, so an
    audit of what the machine changed in a render had to be mined from logs
    -- and a unit that failed looked like a unit that found nothing.
    """
    import json

    from ontocast.api.process_helpers import dump_validation_report
    from ontocast.onto.model import (
        FactsUnitFindingKind,
        GraphRepairRecord,
        UnitFailure,
    )

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    state = _batch_state()
    state.facts_repairs_applied = {
        3: [
            GraphRepairRecord(
                kind=FactsUnitFindingKind.PROPERTY_ALIAS,
                source="ex:hasASiteComponent",
                target="ex:hasBSiteComponent",
                triple_count=2,
            )
        ]
    }
    state.unit_failures = [UnitFailure(unit_index=5, phase="facts", stage="render")]

    out = dump_validation_report(state, src, output_dir=tmp_path / "out")
    assert out is not None
    assert out == tmp_path / "out" / "paper.facts.validation.json"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["unit_repairs"] == {
        "3": [
            {
                "kind": "property_alias",
                "source": "ex:hasASiteComponent",
                "target": "ex:hasBSiteComponent",
                "triple_count": 2,
            }
        ]
    }
    assert payload["unit_failures"] == [
        {"unit_index": 5, "phase": "facts", "stage": "render", "reason": None}
    ]
    # Additive: readers of the previous shape find every key they knew.
    assert set(payload) >= {"source", "conformance", "gate_repairs", "findings"}


def test_dump_ontology_ttls_names_files_per_ontology(tmp_path) -> None:
    from ontocast.api.process_helpers import dump_ontology_ttls

    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    onto_dir = tmp_path / "ontologies"
    state = _batch_state()
    ontology = _sample_ontology()

    state.reduced_ontology_artifacts = [ontology]
    written = dump_ontology_ttls(state, src, output_dir=onto_dir)
    assert written == [onto_dir / "paper.ontology.ttl"]
    assert written[0].exists()

    second = _sample_ontology(iri="https://example.com/other", ontology_id="other")
    state.reduced_ontology_artifacts = [ontology, second]
    written_multi = dump_ontology_ttls(state, src, output_dir=onto_dir)
    assert {path.name for path in written_multi} == {
        "paper.onto.ontology.ttl",
        "paper.other.ontology.ttl",
    }


def test_cli_requires_subcommand() -> None:
    from click.testing import CliRunner

    from ontocast.cli.server import cli

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0


def test_inspect_sections_reads_json_and_text_documents(tmp_path) -> None:
    """`ontocast sections` must read the inputs the pipeline is driven with.

    JSON and plain-text documents are routed *around* the Docling converter by
    the Convert node -- Docling rejects them outright -- so a CLI that called
    the converter for everything could not inspect a converted `*.json` at all.
    """
    import json as _json

    from ontocast.cli.inspect_sections import _load_document
    from ontocast.tool.chunk.sections import document_text_for_section_tagging

    converter = SimpleNamespace(supported_extensions={".pdf", ".docx"})

    json_path = tmp_path / "doc.json"
    json_path.write_text(
        _json.dumps({"text": "# Risk Factors\n\nOur business is subject to risk.\n"}),
        encoding="utf-8",
    )
    doc = _load_document(json_path, cast(ConverterTool, converter))
    assert "Risk Factors" in document_text_for_section_tagging(doc)

    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("Plain body text.\n", encoding="utf-8")
    assert "Plain body text." in document_text_for_section_tagging(
        _load_document(txt_path, cast(ConverterTool, converter))
    )


def test_inspect_sections_rejects_a_json_payload_with_no_text(tmp_path) -> None:
    import json as _json

    import click

    from ontocast.cli.inspect_sections import _load_document

    path = tmp_path / "record.json"
    # The shape of a clinical-registry API record, not
    # a document -- it must fail loudly rather than inspect as an empty document.
    path.write_text(_json.dumps({"protocolSection": {"id": 1}}), encoding="utf-8")
    with pytest.raises(click.ClickException):
        _load_document(
            path, cast(ConverterTool, SimpleNamespace(supported_extensions=set()))
        )


# --- /process_unit runs the validation gate ----------------------------------


def test_process_unit_route_runs_the_validation_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route serves the gate-repaired graph and reports its conformance.

    Previously only the CLI ``--use-unit-pipeline`` path ran the
    post-aggregation gate; ``/process_unit`` shipped unvalidated facts and no
    ``facts_conformance``/``facts_gate_repairs`` metadata.
    """
    from rdflib import Literal
    from rdflib.namespace import XSD

    from ontocast.config import FactsValidationConfig
    from ontocast.onto.constants import DEFAULT_IRI
    from ontocast.onto.content_unit import OutputType
    from ontocast.onto.enum import Status

    q = "https://x.org/schema#"
    stored_shapes = RDFGraph._from_turtle_str(
        f"""
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix q: <{q}> .

        q:ValueShape a sh:NodeShape ;
            sh:targetClass q:QuantityValue ;
            sh:property q:p_numeric .

        # Named, not blank: the retype repair reads sh:datatype off the
        # reported sh:sourceShape, which is only kept when it is an IRI.
        q:p_numeric sh:path q:numericValue ;
            sh:datatype xsd:decimal ;
            sh:minCount 1 .
        """
    )

    node = URIRef(f"{DEFAULT_IRI}/v1")
    facts_graph = RDFGraph()
    facts_graph.add((node, RDF.type, URIRef(q + "QuantityValue")))
    facts_graph.add((node, URIRef(q + "numericValue"), Literal("230")))

    facts_result = SimpleNamespace(
        status=Status.SUCCESS,
        content_unit=ContentUnit(
            text="unit",
            index=0,
            doc_iri=URIRef("https://x.org/doc/1"),
            graph=facts_graph,
            type=OutputType.FACTS,
        ),
        ontology_snapshot=SimpleNamespace(graph=RDFGraph()),
    )

    async def fake_run_unit_pipeline(_state, _tools):
        return None, facts_result

    class _Aggregator:
        def postprocess_facts_units(self, units, ontology_graph, **kwargs):
            graph = RDFGraph()
            graph += units[0].graph
            return AggregationResult(graph=graph)

    monkeypatch.setattr(app_module, "run_unit_pipeline", fake_run_unit_pipeline)
    monkeypatch.setattr(
        app_module, "create_agent_graph", lambda _tools, **_kwargs: SimpleNamespace()
    )
    tools = cast(
        ToolBox,
        SimpleNamespace(
            aggregator=_Aggregator(),
            shapes_catalog=SimpleNamespace(graph=lambda: stored_shapes),
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=FactsValidationConfig.model_construct()
                )
            ),
        ),
    )
    app = create_app(
        tools=tools,
        server_config=ServerConfig(),
        active_tenant="tenant-a",
        active_project="project-a",
    )

    response = TestClient(app).post(
        # Pin the context mode: the ambient .env may select vector search,
        # which this ToolBox stub deliberately lacks.
        "/process_unit?ontology_context_mode=selected_single_ontology",
        json={"text": "230 something"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    metadata = payload["metadata"]
    assert metadata["facts_conformance"]["shacl_evaluated"] is True
    assert metadata["facts_conformance"]["conforms"] is True
    assert [record["kind"] for record in metadata["facts_gate_repairs"]] == [
        "shacl_retype"
    ]
    # The served Turtle is the repaired graph, not the raw aggregation.
    assert (
        str(XSD.decimal) in payload["data"]["facts"]
        or "xsd:decimal" in (payload["data"]["facts"])
    )


def _batch_toolbox() -> ToolBox:
    """A ToolBox light enough for a batch-loop test: no external services."""
    from ontocast.config import LLMConfig, LLMProvider, PathConfig, ToolConfig
    from ontocast.config.settings import OllamaModel

    return ToolBox(
        Config(
            tool_config=ToolConfig(
                path_config=PathConfig(),
                llm_config=LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model_name=OllamaModel.LLAMA3_1,
                    base_url="http://localhost:11434",
                ),
            ),
        )
    )


class _RejectingWorkflow:
    """A workflow whose every call is refused by the provider."""

    def astream(self, *args, **kwds):
        from ontocast.tool.llm import LLMConfigurationError

        async def _stream():
            raise LLMConfigurationError("openai/gpt-x rejected the request")
            yield  # pragma: no cover - makes this an async generator

        return _stream()


def test_a_rejected_request_aborts_the_batch_and_writes_nothing(tmp_path) -> None:
    """The failure this exists for: every call refused, run still "succeeded".

    Observed: the batch swallowed the rejection per unit, serialized an empty
    graph, dumped a run manifest and a validation report next to no facts, and
    exited 0 -- artifacts a downstream aggregator cannot tell from real ones.
    Every remaining file would have paid for its own conversion to reach the
    same rejection.
    """
    from ontocast.api.process_helpers import process_files_input
    from ontocast.tool.llm import LLMConfigurationError

    tools = _batch_toolbox()
    first = tmp_path / "a.txt"
    first.write_text("some text", encoding="utf-8")
    second = tmp_path / "b.txt"
    second.write_text("more text", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(LLMConfigurationError):
        asyncio.run(
            process_files_input(
                [first, second],
                config=tools.config,
                head_chunks=None,
                use_unit_pipeline=False,
                tools=tools,
                workflow=cast(CompiledStateGraph, _RejectingWorkflow()),
                ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
                tenant=None,
                project=None,
                output_dir=out_dir,
            )
        )

    assert list(out_dir.iterdir()) == []


class _AllUnitsFailedWorkflow:
    """A workflow that ran to the end with every unit dead."""

    def astream(self, *args, **kwds):
        from ontocast.onto.enum import Status

        async def _stream():
            yield {"status": Status.FAILED, "aggregated_facts": RDFGraph()}

        return _stream()


def test_a_document_whose_every_unit_failed_is_recorded_as_failed(tmp_path) -> None:
    """The generic backstop, for causes the rejection classifier does not catch.

    The map nodes already computed FAILED for a document that produced nothing
    and merge_facts preserved it -- the batch path just never read it, so the
    run exited 0. cli/server.py turns a non-empty failed_files into a non-zero
    exit, so nothing else is needed to make it scriptable.
    """
    from ontocast.api.process_helpers import process_files_input

    tools = _batch_toolbox()
    src = tmp_path / "a.txt"
    src.write_text("some text", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    failed = asyncio.run(
        process_files_input(
            [src],
            config=tools.config,
            head_chunks=None,
            use_unit_pipeline=False,
            tools=tools,
            workflow=cast(CompiledStateGraph, _AllUnitsFailedWorkflow()),
            ontology_context_mode_value=OntologyContextMode.SELECTED_SINGLE_ONTOLOGY,
            tenant=None,
            project=None,
            output_dir=out_dir,
        )
    )

    assert failed == [src]
