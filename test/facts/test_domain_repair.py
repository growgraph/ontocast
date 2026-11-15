"""A statement whose subject contradicts its property's declared domain.

After the critic loop, one call may swap the property of each such statement
for one whose domain the subject fits; only a pure property swap is applied.
Whatever is still in violation is dropped with a record.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import OWL, PROV, RDF, RDFS

from ontocast.agent.repair_facts_domain import violations_index
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import OntologyAssemblyMode, Status
from ontocast.onto.model import FactsGateRepairKind, FactsUnitFinding, TripleFix
from ontocast.onto.ontology_snapshot import OntologySnapshot
from ontocast.onto.prefixes import PrefixTable
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.unit_states import UnitFactsState
from ontocast.prompt.repair_facts_domain import build_violations_chapter
from ontocast.stategraph import atomic as atomic_module
from ontocast.stategraph.atomic import (
    FACTS_PHASE,
    _drop_domain_violations,
    _run_domain_repair,
)
from ontocast.tool.atomic import AtomicToolBox
from ontocast.tool.facts_validation import CriticPatchPolicy, ValidationPolicy
from ontocast.tool.facts_validation.terms import CatalogRoles, domain_fitting_properties
from ontocast.tool.facts_validation.unit_findings import domain_violation_findings

pytestmark = pytest.mark.unit

_CD = "http://example.org/"
_DATASET = URIRef(f"{_CD}reanalysis")
_SOURCE = URIRef(f"{_CD}station_record")
_RUN = URIRef(f"{_CD}model_run")


def _ontology() -> RDFGraph:
    onto = RDFGraph()
    onto.bind("prov", str(PROV))
    for cls in (PROV.Activity, PROV.Entity):
        onto.add((cls, RDF.type, OWL.Class))
    onto.add((PROV.used, RDFS.domain, PROV.Activity))
    onto.add((PROV.wasAssociatedWith, RDFS.domain, PROV.Activity))
    onto.add((PROV.wasDerivedFrom, RDFS.domain, PROV.Entity))
    onto.add((PROV.wasDerivedFrom, RDFS.range, PROV.Entity))
    onto.add((PROV.wasGeneratedBy, RDFS.domain, PROV.Entity))
    onto.add((PROV.wasGeneratedBy, RDFS.range, PROV.Activity))
    return onto


def _state() -> UnitFactsState:
    facts = RDFGraph()
    facts.bind("prov", str(PROV))
    facts.bind("cd", _CD)
    facts.add((_DATASET, RDF.type, PROV.Entity))
    facts.add((_DATASET, RDFS.label, Literal("reanalysis")))
    facts.add((_SOURCE, RDF.type, PROV.Entity))
    facts.add((_SOURCE, RDFS.label, Literal("station record")))
    facts.add((_RUN, RDF.type, PROV.Activity))
    facts.add((_DATASET, PROV.used, _SOURCE))
    facts.add((_DATASET, PROV.wasGeneratedBy, _RUN))
    unit = ContentUnit(
        text="The reanalysis is derived from the station record.",
        index=0,
        doc_iri=URIRef("https://example.com/doc/d1"),
        graph=facts,
    )
    snapshot = OntologySnapshot.from_graph(
        _ontology(),
        source_iris=["https://example.com/onto/prov"],
        assembly_mode=OntologyAssemblyMode.FIXED_SINGLE_ONTOLOGY,
    )
    state = UnitFactsState(content_unit=unit, ontology_snapshot=snapshot)
    state.status = Status.SUCCESS
    return state


def _atomic(*, completion_passes: int = 1) -> AtomicToolBox:
    return cast(
        AtomicToolBox,
        SimpleNamespace(
            prefix_table=lambda: PrefixTable({}, {}),
            facts_completion_passes=completion_passes,
            facts_patch_policy=CriticPatchPolicy(),
            validation_policy=ValidationPolicy(),
            acceptance_policy=None,
            quantity_fallback_vocabulary=None,
            additional_standard_namespaces=(),
            catalog_terms=lambda: set(),
            catalog_roles=lambda: CatalogRoles(),
            subject_contract=lambda: None,
        ),
    )


def _violations(state, _atomic) -> list[FactsUnitFinding]:
    return domain_violation_findings(
        state.content_unit.graph, state.ontology_snapshot.graph
    )


_PHASE = replace(FACTS_PHASE, collect_findings=_violations)


def _replace(correct: str, triple_ids: list[int]) -> TripleFix:
    return TripleFix(
        text_fragment="derived from the station record",
        action="REPLACE",
        severity="important",
        correct_value=correct,
        explanation="the property that fits an entity",
        triple_ids=triple_ids,
    )


def _statement(subject: URIRef, predicate: URIRef, obj: URIRef) -> str:
    return f'{{"@id": "{subject}", "{predicate}": {{"@id": "{obj}"}}}}'


def _fake_repair(*fixes: TripleFix):
    calls: list[int] = []

    async def fake(state, _atomic, findings):
        calls.append(1)
        from ontocast.agent.repair_facts_domain import violating_statements

        graph = state.content_unit.graph
        return list(fixes), violations_index(
            graph, violating_statements(graph, findings)
        )

    return fake, calls


def test_the_violation_is_found_and_a_fitting_property_is_a_candidate() -> None:
    state = _state()
    graph, onto = state.content_unit.graph, state.ontology_snapshot.graph

    assert [
        (f.subject, f.predicate) for f in domain_violation_findings(graph, onto)
    ] == [(str(_DATASET), str(PROV.used))]
    candidates = [
        predicate
        for predicate, _ in domain_fitting_properties(
            _DATASET, _SOURCE, graph, onto, replacing=PROV.used
        )
    ]
    # wasGeneratedBy fits the subject but its range (an activity) not the object.
    assert candidates == [PROV.wasDerivedFrom]


def test_the_chapter_lists_each_statement_by_id_with_its_options() -> None:
    graph = _state().content_unit.graph
    statement = (_DATASET, PROV.used, _SOURCE)
    index = violations_index(graph, [statement])

    chapter = build_violations_chapter(
        graph,
        index,
        {statement: {PROV.Activity}},
        {statement: [(PROV.wasDerivedFrom, {PROV.Entity})]},
    )

    assert "[1] cd:reanalysis prov:used cd:station_record" in chapter
    assert "prov:used applies to: prov:Activity" in chapter
    assert "prov:wasDerivedFrom  (applies to prov:Entity)" in chapter
    assert list(index.by_id) == [1]


@pytest.mark.anyio
async def test_a_property_swap_resolves_the_violation(monkeypatch) -> None:
    state = _state()
    fake, _ = _fake_repair(
        _replace(_statement(_DATASET, PROV.wasDerivedFrom, _SOURCE), [1])
    )
    monkeypatch.setattr(atomic_module, "repair_domain_violations", fake)

    await _run_domain_repair(state, _atomic(), _PHASE, render_attempt=1)

    graph = state.content_unit.graph
    assert (_DATASET, PROV.wasDerivedFrom, _SOURCE) in graph
    assert (_DATASET, PROV.used, _SOURCE) not in graph
    attempt = state.attempt_log[-1]
    assert attempt.kind == "domain_repair"
    assert attempt.n_fixes_applied == 1


@pytest.mark.parametrize(
    "correct",
    [
        # Changes the object, not only the property.
        _statement(_DATASET, PROV.wasDerivedFrom, _RUN),
        # Another property the subject does not fit either.
        _statement(_DATASET, PROV.wasAssociatedWith, _SOURCE),
        # Swaps the property and adds a statement besides.
        _statement(_DATASET, PROV.wasDerivedFrom, _SOURCE)[:-1]
        + f', "{RDFS.comment}": "extra"}}',
    ],
)
@pytest.mark.anyio
async def test_a_fix_doing_more_than_a_fitting_swap_is_refused(
    monkeypatch, correct: str
) -> None:
    state = _state()
    fake, _ = _fake_repair(_replace(correct, [1]))
    monkeypatch.setattr(atomic_module, "repair_domain_violations", fake)

    await _run_domain_repair(state, _atomic(), _PHASE, render_attempt=1)

    assert (_DATASET, PROV.used, _SOURCE) in state.content_unit.graph
    assert state.attempt_log[-1].n_fixes_applied == 0


@pytest.mark.anyio
async def test_no_violation_means_no_call(monkeypatch) -> None:
    state = _state()
    state.content_unit.graph.remove((_DATASET, PROV.used, _SOURCE))
    fake, calls = _fake_repair()
    monkeypatch.setattr(atomic_module, "repair_domain_violations", fake)

    await _run_domain_repair(state, _atomic(), _PHASE, render_attempt=1)

    assert calls == []
    assert state.attempt_log == []


def test_a_statement_still_in_violation_is_dropped_with_a_record() -> None:
    state = _state()

    _drop_domain_violations(state, _atomic(), _PHASE)

    graph = state.content_unit.graph
    assert (_DATASET, PROV.used, _SOURCE) not in graph
    # The subject's other statements stay.
    assert (_DATASET, PROV.wasGeneratedBy, _RUN) in graph
    [record] = state.applied_repairs
    assert record.kind == FactsGateRepairKind.DOMAIN_VIOLATION_DROPPED
    assert (record.source, record.target, record.triple_count) == (
        str(_DATASET),
        str(PROV.used),
        1,
    )
    assert state.deterministic_findings == []


def test_the_prompt_template_and_its_variables_agree() -> None:
    import string

    from ontocast.agent.repair_facts_domain import _build_prompt
    from ontocast.prompt.repair_facts_domain import template_prompt

    named = {
        field for _, field, _, _ in string.Formatter().parse(template_prompt) if field
    }
    assert named == set(_build_prompt().input_variables)
