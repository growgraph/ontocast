"""Every prompt shows each namespace under one prefix, and parses what it shows.

The fixture catalog reproduces each way a prompt used to show one namespace
under several prefixes: an author prefix that differs from the ontology id, a
catalog file whose terms live in a well-known namespace, a file borrowing
another's namespace under its own prefix, and shapes whose messages use yet
another alias.
"""

from __future__ import annotations

import importlib
import re
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import RDF, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode
from ontocast.onto.model import FactsCritiqueReport, FactsRenderReport
from ontocast.onto.ontology_access import build_llm_prefix_map
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.prefixes import PrefixTable, build_prefix_table
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.facts_guidelines import DEFAULT_QUANTITY_FALLBACK_VOCABULARY
from ontocast.prompt.shapes_contract import (
    derive_shape_requirements,
    format_conformance_chapter,
)
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.ontology_validation.catalog_terms import (
    catalog_prefixes,
    catalog_terms,
    format_term_sheet,
)
from test.ontology.test_prefix_table import LIFE, MAT, OBS, UNIT, _catalog

pytestmark = pytest.mark.unit

SHAPES_TTL = f"""
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix lifec: <{LIFE}> .
@prefix obs:   <{OBS}> .
@prefix ex:    <https://ex.org/shapes#> .
ex:AgingShape a sh:NodeShape ;
    sh:targetClass lifec:AgingProcess ;
    sh:property [
        sh:path lifec:hasDuration ;
        sh:message "An aging process states lifec:hasDuration, not an obs:Process condition." ;
    ] .
"""

#: A CURIE in running text: what :mod:`ontocast.onto.prefixes` rewrites.
_CURIE = re.compile(r"(?<![\w:/#<.-])([A-Za-z][\w.-]*):([A-Za-z_][\w.-]*)")
#: A prefix declared beside its namespace: "`p:` (<ns>)" or "`p:` <ns>".
_DECLARED = re.compile(r"`([A-Za-z][\w.-]*):`\s*\(?<([^>]+)>")


def prefixes_by_namespace(
    prompt: str, parse_map: dict[str, str]
) -> dict[str, set[str]]:
    """Namespace -> every prefix the prompt writes it with."""
    found: dict[str, set[str]] = {}
    for prefix, _ in _CURIE.findall(prompt):
        namespace = parse_map.get(prefix)
        if namespace is not None:
            found.setdefault(namespace, set()).add(prefix)
    for prefix, namespace in _DECLARED.findall(prompt):
        found.setdefault(namespace, set()).add(prefix)
    return found


def _shapes() -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=SHAPES_TTL, format="turtle")
    return graph


def _context(table: PrefixTable) -> OntologySnapshot:
    """The merged document context, as the resolver hands it to a prompt."""
    merged = RDFGraph()
    catalog = _catalog()
    for ontology in catalog:
        merged += ontology.graph
    merged.sanitize_prefixes_namespaces()
    snapshot = OntologySnapshot.from_graph(
        merged,
        source_iris=[o.iri for o in catalog],
        assembly_mode=OntologyAssemblyMode.DOCUMENT_MERGED_REDUCED,
        title="Merged",
        description="Merged catalog",
        strip_headers=True,
    )
    table.normalize(snapshot.graph)
    return snapshot


def _tools(table: PrefixTable) -> AtomicToolBox:
    async def get_llm_tool(_budget_tracker):
        return object()

    return cast(
        AtomicToolBox,
        SimpleNamespace(
            prefix_table=lambda: table,
            get_llm_tool=get_llm_tool,
            web_grounding_enabled_for_node=lambda _node: False,
            object_property_literal_check=True,
            property_alias_min_ratio=0.85,
            catalog_terms=lambda: set(),
            code_predicates=(),
            citation_vocabulary={},
            quantity_fallback_vocabulary=dict(DEFAULT_QUANTITY_FALLBACK_VOCABULARY),
            facts_vocabulary="strict",
            additional_standard_namespaces=(),
            acceptance_policy=None,
        ),
    )


@pytest.mark.anyio
async def test_facts_render_and_critic_prompts_show_one_prefix_per_namespace(
    monkeypatch,
) -> None:
    render_agent = importlib.import_module("ontocast.agent.render_facts")
    critic_agent = importlib.import_module("ontocast.agent.criticise_facts")
    prompts: dict[str, str] = {}

    def capturing(name: str, report):
        async def fake(**kwargs):
            prompts[name] = kwargs["prompt"].format(**kwargs["prompt_kwargs"])
            return report

        return fake

    rendered = RDFGraph()
    rendered.parse(
        data=(
            f"@prefix cd: <{DEFAULT_IRI}/> . @prefix life: <{LIFE}> .\n"
            "cd:aging a life:AgingProcess ."
        ),
        format="turtle",
    )
    monkeypatch.setattr(
        render_agent,
        "call_llm_with_retry",
        capturing(
            "render",
            FactsRenderReport(
                semantic_graph=rendered,
                ontology_relevance_score=90,
                triples_generation_score=90,
            ),
        ),
    )
    monkeypatch.setattr(
        critic_agent,
        "call_llm_with_retry",
        capturing(
            "critic",
            FactsCritiqueReport(
                success=True,
                score=95,
                actionable_triple_fixes=[],
                systemic_critique_summary="",
            ),
        ),
    )
    shapes = _shapes()
    table = build_prefix_table(_catalog(), [shapes])
    snapshot = _context(table)
    state = UnitFactsState(
        content_unit=ContentUnit(
            text="The films aged for 30 days at 25 °C.",
            index=0,
            doc_iri=URIRef("https://ex.org/doc/d1"),
        ),
        ontology_snapshot=snapshot,
        conformance_chapter=format_conformance_chapter(
            derive_shape_requirements(shapes, table)
        ),
    )
    tools = _tools(table)

    await render_agent.render_facts_fresh(state, tools=tools)
    await critic_agent.criticise_facts(state, tools=tools)

    parse_map = build_llm_prefix_map(snapshot, (), table)
    for name, prompt in prompts.items():
        found = prefixes_by_namespace(prompt, parse_map)
        shared = {ns: p for ns, p in found.items() if len(p) > 1}
        assert not shared, f"{name}: one namespace, several prefixes: {shared}"
        assert found[LIFE] == {"life"}
        assert found[OBS] == {"obs"}
        assert found[UNIT] == {"unit"}
        assert found[MAT] == {"mat"}
        assert "life:hasDuration, not an obs:Process" in prompt


def test_the_ontology_lane_term_sheet_names_terms_as_the_facts_lane_does() -> None:
    catalog = _catalog()
    table = build_prefix_table(catalog, [_shapes()])
    prefixes = catalog_prefixes(catalog, None, table)

    sheet = format_term_sheet(catalog_terms(catalog), prefixes, 200)

    found = prefixes_by_namespace(sheet, prefixes)
    assert found[LIFE] == {"life"}
    assert found[MAT] == {"mat"}
    assert not {ns: p for ns, p in found.items() if len(p) > 1}


@pytest.mark.parametrize(
    "payload",
    [
        "cd:aging a lc:AgingProcess ; lifecycle:hasDuration cd:span .",
        "cd:aging a lifec:AgingProcess .",
    ],
)
def test_an_alias_the_model_copies_parses_to_its_namespace(payload: str) -> None:
    table = build_prefix_table(_catalog(), [_shapes()])
    previous = RDFGraph.get_known_prefixes()
    RDFGraph.set_known_prefixes(build_llm_prefix_map(_context(table), (), table))
    try:
        graph = RDFGraph._from_turtle_str(f"@prefix cd: <{DEFAULT_IRI}/> .\n{payload}")
    finally:
        RDFGraph.set_known_prefixes(previous)

    aging = URIRef(f"{DEFAULT_IRI}/aging")
    assert (aging, RDF.type, URIRef(f"{LIFE}AgingProcess")) in graph


def test_an_alias_parses_in_json_ld_too() -> None:
    import json

    table = build_prefix_table(_catalog(), [_shapes()])
    previous = RDFGraph.get_known_prefixes()
    RDFGraph.set_known_prefixes(build_llm_prefix_map(_context(table), (), table))
    try:
        graph = RDFGraph._from_jsonld_str(
            json.dumps(
                {
                    "@context": {"cd": f"{DEFAULT_IRI}/"},
                    "@id": "cd:aging",
                    "@type": "lc:AgingProcess",
                    "lifecycle:hasDuration": {"@id": "cd:span"},
                }
            )
        )
    finally:
        RDFGraph.set_known_prefixes(previous)

    assert (None, RDF.type, URIRef(f"{LIFE}AgingProcess")) in graph
    assert (None, URIRef(f"{LIFE}hasDuration"), None) in graph
