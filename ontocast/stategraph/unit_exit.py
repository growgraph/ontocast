"""What leaves the unit loop: one exit action per finding kind.

The unit loop keeps serving a content unit the critic did not accept, but no
statement a deterministic check rejected leaves it. :data:`EXIT_ACTIONS` says,
for every kind a unit finding can have, what happens to the statements an
unresolved finding of that kind names when the unit leaves the loop, after
the gap pass. A finding about an absence -- a missing role, a missing number,
a missing type -- or about no single statement (results that disagree in
dimension name no one wrong number) cannot be repaired by removing one, so it
is reported: it stays on the unit and is counted in the document report.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from rdflib import RDF, BNode, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsUnitFinding,
    FactsUnitFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import (
    RDFGraph,
    RejectedLiteralTriple,
    drop_reifiers_mentioning,
)
from ontocast.stategraph.facts_gate import (
    _remove_instances,
    _remove_orphaned_blank_nodes,
    drop_domain_violations,
)
from ontocast.tool.facts_validation.grounding import node_key, node_of
from ontocast.tool.facts_validation.literal_repair import write_back_quarantine
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    value_components,
)
from ontocast.tool.facts_validation.term_verdict import TermVerdictKind


class ExitAction(StrEnum):
    """What leaving the loop does to the statements a finding names."""

    #: Every statement of the (subject, property) pair goes.
    DROP_STATEMENTS = "drop_statements"
    #: Every statement using a term a vocabulary check rejects goes, when the
    #: finding is of error severity (mandatory); its suggested terms are
    #: reported on the record, never applied.
    DROP_TERM = "drop_term"
    #: The statement typing a node with a string goes.
    DROP_TRIPLE = "drop_triple"
    #: The number goes, the label stays; a value left with no number goes,
    #: its carrier stays.
    DROP_NUMBER = "drop_number"
    #: The value's link to the individual goes.
    DROP_INDIVIDUAL = "drop_individual"
    #: The literal set aside goes back on its subject as text.
    WRITE_BACK = "write_back"
    #: Nothing is removed; the finding is carried to the document report.
    REPORT = "report"


#: The exit action of every kind a unit finding can have.
EXIT_ACTIONS: dict[FactsUnitFindingKind, ExitAction] = {
    FactsUnitFindingKind.QUARANTINED_LITERAL: ExitAction.WRITE_BACK,
    FactsUnitFindingKind.UNKNOWN_TERM: ExitAction.DROP_TERM,
    FactsUnitFindingKind.LITERAL_TYPE_OBJECT: ExitAction.DROP_TRIPLE,
    FactsUnitFindingKind.DOMAIN_VIOLATION: ExitAction.DROP_STATEMENTS,
    FactsUnitFindingKind.UNGROUNDED_NUMBER: ExitAction.DROP_NUMBER,
    FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL: ExitAction.DROP_INDIVIDUAL,
    FactsUnitFindingKind.NUMERIC_COVERAGE: ExitAction.REPORT,
    FactsUnitFindingKind.LABEL_ONLY_NUMBER: ExitAction.REPORT,
    FactsUnitFindingKind.SCALAR_AS_BOUNDS: ExitAction.REPORT,
    FactsUnitFindingKind.UNTYPED_INSTANCE: ExitAction.REPORT,
    FactsUnitFindingKind.MISSING_ROLE: ExitAction.REPORT,
    FactsUnitFindingKind.DIMENSION_CONFLICT: ExitAction.REPORT,
    FactsUnitFindingKind.DOMAIN_ADHERENCE: ExitAction.REPORT,
}

#: Kinds about an absence (a role, a number, a type) or about no single
#: statement: no removal resolves them, so they are reported.
ABSENCE_KINDS: frozenset[FactsUnitFindingKind] = frozenset(
    {
        FactsUnitFindingKind.NUMERIC_COVERAGE,
        FactsUnitFindingKind.LABEL_ONLY_NUMBER,
        FactsUnitFindingKind.SCALAR_AS_BOUNDS,
        FactsUnitFindingKind.UNTYPED_INSTANCE,
        FactsUnitFindingKind.MISSING_ROLE,
        FactsUnitFindingKind.DIMENSION_CONFLICT,
        FactsUnitFindingKind.DOMAIN_ADHERENCE,
    }
)

#: Kinds that name a repair already applied, or a critic's proposal: never a
#: deterministic finding a unit can leave with.
NOT_FINDINGS: frozenset[FactsUnitFindingKind] = frozenset(
    {
        FactsUnitFindingKind.PROPERTY_ALIAS,
        FactsUnitFindingKind.CLOSED_RANGE_LITERAL,
        FactsUnitFindingKind.COMPACT_IRI_LITERAL,
        FactsUnitFindingKind.ENTAILED_TYPE,
        FactsUnitFindingKind.TRIPLE_REVERSED,
        FactsUnitFindingKind.CLASS_ALIAS,
        FactsUnitFindingKind.PAIRED_OBJECT_PROPERTY,
        FactsUnitFindingKind.CRITIC_FIX,
    }
)


class UnitExit(BaseModel):
    """What leaving the loop changed.

    Attributes:
        records: One repair record per removal or write-back.
        quarantined: The unit's quarantined literals, those written back
            marked so.
    """

    model_config = ConfigDict(frozen=True)

    records: list[GraphRepairRecord]
    quarantined: list[RejectedLiteralTriple]


def _remove(graph: RDFGraph, triples: Sequence[tuple[Node, Node, Node]]) -> int:
    """Remove ``triples``, the blank nodes only they reached, and their reifiers."""
    orphan_candidates: set[BNode] = set()
    removed = 0
    for triple in triples:
        if triple not in graph:
            continue
        graph.remove(triple)
        removed += 1
        if isinstance(triple[2], BNode):
            orphan_candidates.add(triple[2])
        if graph.has_reifications:
            reifiers = graph.reifications.reifiers_of(triple)
            graph.reifications.remove_reifiers(reifiers)
            for reifier in reifiers:
                for arc in list(graph.triples((reifier, None, None))):
                    graph.remove(arc)
    orphaned, orphan_triples = _remove_orphaned_blank_nodes(graph, orphan_candidates)
    drop_reifiers_mentioning(graph, orphaned)
    return removed + orphan_triples


def _uses(
    graph: RDFGraph, finding: FactsUnitFinding, contract: StatementContract
) -> list[tuple[Node, Node, Node]]:
    """The statements using the term an ``unknown_term`` finding names, in its
    position: as a predicate, as an ``rdf:type`` object, or as another object.

    The term's own description (its triples as a subject) is not a use.
    """
    term = URIRef(finding.predicate)
    if finding.facet == TermVerdictKind.CLOSED_RANGE_MISSING.value:
        # Unknown only where its property takes a known individual.
        return [
            (subject, URIRef(component.predicate), term)
            for subject in {s for s, _, o in graph.triples((None, None, term))}
            for component in value_components(graph, subject, contract).individuals
            if component.value == term and component.closed_range
        ]
    if finding.position == "predicate":
        found = set(graph.triples((None, term, None)))
    elif finding.position == "type":
        found = set(graph.triples((None, RDF.type, term)))
    elif finding.position == "object":
        found = {
            triple
            for triple in graph.triples((None, None, term))
            if triple[1] != RDF.type
        }
    else:
        found = {
            *graph.triples((None, term, None)),
            *graph.triples((None, None, term)),
        }
    return sorted(found, key=lambda triple: tuple(str(t) for t in triple))


def _drop_value(graph: RDFGraph, value: Node) -> tuple[int, list[str]]:
    """Remove ``value`` and the links to it; return the triples and its carriers."""
    carriers = sorted(node_key(carrier) for carrier in graph.subjects(None, value))
    if isinstance(value, URIRef):
        return _remove_instances(graph, [value]), carriers
    links = list(graph.triples((None, None, value)))
    return _remove(graph, links), carriers


class _Exit(BaseModel):
    """What one exit works on: the graph, the contract, the records so far."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: RDFGraph
    contract: StatementContract
    quarantined: list[RejectedLiteralTriple]
    records: list[GraphRepairRecord] = []
    marked: list[RejectedLiteralTriple] = []


def _drop_statements(state: _Exit, findings: Sequence[FactsUnitFinding]) -> None:
    state.records.extend(drop_domain_violations(state.graph, findings))


def _drop_term(state: _Exit, findings: Sequence[FactsUnitFinding]) -> None:
    for finding in findings:
        removed = _remove(state.graph, _uses(state.graph, finding, state.contract))
        if removed:
            state.records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.UNKNOWN_TERM_DROPPED,
                    source=finding.predicate,
                    target=finding.facet,
                    triple_count=removed,
                    suggestions=list(finding.suggestions),
                )
            )


def _drop_triple(state: _Exit, findings: Sequence[FactsUnitFinding]) -> None:
    graph = state.graph
    for finding in findings:
        # One finding per string: every node typed with it loses that type.
        triples = [
            (subject, RDF.type, obj)
            for subject, obj in graph.subject_objects(RDF.type)
            if isinstance(obj, Literal) and str(obj).strip() == finding.value
        ]
        removed = _remove(graph, triples)
        if removed:
            state.records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.UNKNOWN_TERM_DROPPED,
                    source=finding.value,
                    target=finding.subject,
                    triple_count=removed,
                )
            )


def _drop_individual(state: _Exit, findings: Sequence[FactsUnitFinding]) -> None:
    for finding in findings:
        triple = (
            node_of(finding.subject),
            URIRef(finding.predicate),
            URIRef(finding.value),
        )
        if _remove(state.graph, [triple]):
            state.records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.CONTRADICTED_INDIVIDUAL_DROPPED,
                    source=finding.subject,
                    target=finding.value,
                )
            )


def _drop_number(state: _Exit, findings: Sequence[FactsUnitFinding]) -> None:
    graph = state.graph
    stripped: set[Node] = set()
    for finding in findings:
        value = node_of(finding.subject)
        triples = [
            (value, URIRef(finding.predicate), obj)
            for obj in graph.objects(value, URIRef(finding.predicate))
            if isinstance(obj, Literal) and str(obj) == finding.value
        ]
        if _remove(graph, triples):
            stripped.add(value)
            state.records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.UNGROUNDED_NUMBER_DROPPED,
                    source=finding.subject,
                    target=f"<{finding.predicate}> {finding.value}",
                )
            )
    for value in sorted(stripped, key=str):
        if value_components(graph, value, state.contract).numbers:
            continue
        removed, carriers = _drop_value(graph, value)
        state.records.append(
            GraphRepairRecord(
                kind=FactsGateRepairKind.VALUE_WITHOUT_NUMBER_DROPPED,
                source=node_key(value),
                target=", ".join(carriers),
                triple_count=removed,
            )
        )


def _write_back(state: _Exit, findings: Sequence[FactsUnitFinding]) -> None:
    # The unit's quarantine is the record, the findings only mirror it.
    written, state.marked = write_back_quarantine(state.graph, state.quarantined)
    state.records.extend(written)


ExitHandler = Callable[[_Exit, Sequence[FactsUnitFinding]], None]

#: Each exit action with its handler, in the order the exit runs them: a
#: statement removed by an earlier action is not judged again, and literals
#: set aside go back last, so a subject removed meanwhile gets none.
EXIT_HANDLERS: dict[ExitAction, ExitHandler] = {
    ExitAction.DROP_STATEMENTS: _drop_statements,
    ExitAction.DROP_TERM: _drop_term,
    ExitAction.DROP_TRIPLE: _drop_triple,
    ExitAction.DROP_INDIVIDUAL: _drop_individual,
    ExitAction.DROP_NUMBER: _drop_number,
    ExitAction.WRITE_BACK: _write_back,
}


def exit_unit(
    graph: RDFGraph,
    findings: Sequence[FactsUnitFinding],
    quarantined: Sequence[RejectedLiteralTriple],
    contract: StatementContract,
) -> UnitExit:
    """Apply :data:`EXIT_ACTIONS` to the unit's unresolved findings.

    Each mandatory finding goes to the handler of its kind's action
    (:data:`EXIT_HANDLERS`), in the handlers' order: domain violations; every
    use of a term an ``unknown_term`` finding names, in the position it names
    -- a term a closed vocabulary does not have, a placeholder term, a class
    or property minted in a facts namespace, a term outside a strict
    vocabulary -- with the finding's suggested terms on its record, never
    applied; literal type objects; contradicted individuals; ungrounded
    numbers, a value whose numbers all went removed with the links to it;
    then literals still set aside go back on their subjects. Only mandatory
    findings act: a warning stays in the graph and on the unit, and a kind
    whose action is to report is carried to the document report.

    Args:
        graph: The unit's facts graph, mutated in place.
        findings: The unit's findings after the gap pass.
        quarantined: The unit's quarantined literals.
        contract: The catalog's statement contract.
    """
    by_action: dict[ExitAction, list[FactsUnitFinding]] = {}
    for finding in findings:
        if finding.mandatory and finding.kind in EXIT_ACTIONS:
            by_action.setdefault(EXIT_ACTIONS[finding.kind], []).append(finding)
    state = _Exit(graph=graph, contract=contract, quarantined=list(quarantined))
    for action, handler in EXIT_HANDLERS.items():
        handler(state, by_action.get(action, []))
    return UnitExit(records=state.records, quarantined=state.marked)


__all__ = [
    "ABSENCE_KINDS",
    "EXIT_ACTIONS",
    "EXIT_HANDLERS",
    "NOT_FINDINGS",
    "ExitAction",
    "UnitExit",
    "exit_unit",
]
