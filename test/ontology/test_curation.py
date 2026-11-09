"""Catalog curation: lineage attribution, usage flags, decisions, apply."""

import asyncio
import importlib
from datetime import datetime, timedelta, timezone

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.config import Config, LLMConfig, PathConfig, ToolConfig
from ontocast.onto.model import OntologyConsolidationReport
from ontocast.onto.ontology import Ontology
from ontocast.onto.ontology_header import stamp_version
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import ToolBox
from test.ontology_helpers import DOMAIN, stamped_ontology

curation = importlib.import_module("ontocast.tool.curation")

pytestmark = pytest.mark.unit

NS = f"{DOMAIN}/cat#"
DOC1 = f"{DOMAIN}/doc/d1/"
DOC2 = f"{DOMAIN}/doc/d2/"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def u(local: str) -> URIRef:
    return URIRef(f"{NS}{local}")


def _classes(graph: RDFGraph, parent: str | None, *pairs: tuple[str, str]) -> None:
    for local, label in pairs:
        graph.add((u(local), RDF.type, OWL.Class))
        graph.add((u(local), RDFS.label, Literal(label)))
        if parent:
            graph.add((u(local), RDFS.subClassOf, u(parent)))


def _child(base: Ontology, graph: RDFGraph, source: str, minutes: int) -> Ontology:
    version = stamp_version(
        base.derive_updated_version(graph),
        source_doc=source,
        now=T0 + timedelta(minutes=minutes),
    )
    assert version is not None
    return version


def _tools() -> ToolBox:
    """A catalog grown by two documents over one authored root, with facts.

    - root (authored): Sample
    - d1 adds the films (a value set under Sample) and Widget
    - d2 adds Device
    - facts: d1 and d2 both type something Widget; only d2 types a Device.
    """
    tools = ToolBox(
        Config(tool_config=ToolConfig(path_config=PathConfig(), llm_config=LLMConfig()))
    )
    root_graph = RDFGraph()
    _classes(root_graph, None, ("Sample", "Sample"))
    root = stamped_ontology(root_graph, ontology_id="cat", now=T0)

    v1_graph = root.graph.copy()
    _classes(
        v1_graph,
        "Sample",
        ("CesiumFilm", "Cesium film"),
        ("LeadFilm", "Lead film"),
        ("TinFilm", "Tin film"),
    )
    _classes(v1_graph, None, ("Widget", "Widget"))
    v1 = _child(root, v1_graph, DOC1, 1)

    v2_graph = v1.graph.copy()
    _classes(v2_graph, None, ("Device", "Device"))
    v2 = _child(v1, v2_graph, DOC2, 2)

    async def load() -> None:
        store = tools.triple_store_manager
        assert store is not None
        for version in (root, v1, v2):
            await store.aserialize(version)
            await tools.ontology_manager.aadd_ontology(version)
        for doc, typed in ((DOC1, ["Widget"]), (DOC2, ["Widget", "Device"])):
            facts = RDFGraph()
            for index, local in enumerate(typed):
                facts.add((URIRef(f"{doc}e{index}"), RDF.type, u(local)))
            await store.aserialize(facts, graph_uri=doc)

    asyncio.run(load())
    return tools


def _fake_model(monkeypatch, actions: list[dict]) -> list[str]:
    prompts: list[str] = []

    async def answer(**kwargs) -> OntologyConsolidationReport:
        prompts.append(str(kwargs["prompt_kwargs"]["terms"]))
        return OntologyConsolidationReport.model_validate({"actions": actions})

    monkeypatch.setattr(curation, "call_llm_with_retry", answer)
    return prompts


def test_terms_are_attributed_to_the_version_that_introduced_them() -> None:
    tools = _tools()
    head = tools.ontology_manager.get_head_by_iri(f"{DOMAIN}/cat")
    assert head is not None

    minted_by, authored = asyncio.run(curation.attribute_terms(tools, head))

    assert set(authored) == {str(u("Sample"))}
    assert minted_by == {
        str(u("CesiumFilm")): DOC1,
        str(u("LeadFilm")): DOC1,
        str(u("TinFilm")): DOC1,
        str(u("Widget")): DOC1,
        str(u("Device")): DOC2,
    }


def test_attribution_walks_priors_a_store_loaded_manager_does_not_hold() -> None:
    """A manager loaded from a triple store holds only the head; the priors are
    read from their stored graphs."""
    tools = _tools()
    head = tools.ontology_manager.get_head_by_iri(f"{DOMAIN}/cat")
    assert head is not None
    tools.ontology_manager.reset_catalog()
    asyncio.run(tools.ontology_manager.aadd_ontology(head))
    assert len(tools.ontology_manager.get_ontology_versions_by_iri(head.iri)) == 1

    minted_by, authored = asyncio.run(curation.attribute_terms(tools, head))

    assert set(authored) == {str(u("Sample"))}
    assert minted_by[str(u("Widget"))] == DOC1
    assert minted_by[str(u("Device"))] == DOC2


def test_a_dry_run_flags_and_decides_but_writes_nothing(monkeypatch) -> None:
    tools = _tools()
    head_before = tools.ontology_manager.get_head_by_iri(f"{DOMAIN}/cat")
    assert head_before is not None
    prompts = _fake_model(
        monkeypatch,
        [
            {"term": "cat:TinFilm", "action": "drop"},
            {"term": "cat:Widget", "action": "drop"},
            {"term": "cat:Sample", "action": "drop"},
        ],
    )

    (report,) = asyncio.run(curation.curate_catalog(tools, ontology="cat"))

    flags = {term.iri.rsplit("#", 1)[1]: set(term.flags) for term in report.terms}
    assert flags["TinFilm"] == {"value_set", "unused"}
    assert flags["Device"] == {"single_document_use"}
    assert flags["Widget"] == set()
    assert report.authored_terms == 1
    # Only flagged terms are shown in full; Widget is a one-line target.
    (prompt,) = prompts
    assert "cat:TinFilm`" in prompt and "cat:Widget`" not in prompt
    assert report.actions[str(u("TinFilm"))] == "drop"
    assert sorted(reason for _, reason in report.ignored) == [
        "not a listed minted term",
        "not up for review",
    ]
    assert report.triples_deleted > 0 and report.written_version_iri is None
    head_after = tools.ontology_manager.get_head_by_iri(f"{DOMAIN}/cat")
    assert head_after is not None
    assert head_after.version_iri == head_before.version_iri


def test_apply_writes_one_new_head_that_curation_reads_back(monkeypatch) -> None:
    tools = _tools()
    _fake_model(
        monkeypatch,
        [
            {"term": "cat:TinFilm", "action": "drop"},
            {
                "term": "cat:LeadFilm",
                "action": "merge_into",
                "target": "cat:CesiumFilm",
            },
        ],
    )

    (report,) = asyncio.run(curation.curate_catalog(tools, ontology="cat", apply=True))

    head = tools.ontology_manager.get_head_by_iri(f"{DOMAIN}/cat")
    assert head is not None
    assert report.written_version_iri == head.version_iri
    assert head.version == report.predicted_version
    assert (u("TinFilm"), None, None) not in head.graph
    assert (u("LeadFilm"), None, None) not in head.graph
    assert (u("CesiumFilm"), RDF.type, OWL.Class) in head.graph
    # The curation version is not a document: attribution is unchanged.
    minted_by, authored = asyncio.run(curation.attribute_terms(tools, head))
    assert set(authored) == {str(u("Sample"))}
    assert minted_by[str(u("CesiumFilm"))] == DOC1


def test_nothing_flagged_means_no_model_call(monkeypatch) -> None:
    tools = _tools()
    _fake_model(monkeypatch, [])

    (report,) = asyncio.run(
        curation.curate_catalog(tools, ontology="cat", use_facts=False)
    )

    # Without usage counts only shape flags remain: the film value set.
    flagged = {term.iri for term in report.terms if term.flags}
    assert flagged == {str(u(n)) for n in ("CesiumFilm", "LeadFilm", "TinFilm")}
    assert report.calls == 1

    monkeypatch.setattr(curation, "structural_flags", lambda *_a, **_k: {})
    (report,) = asyncio.run(
        curation.curate_catalog(tools, ontology="cat", use_facts=False)
    )
    assert report.calls == 0
    assert report.actions == {}
