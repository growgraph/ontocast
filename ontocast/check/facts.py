"""Check a user-supplied facts graph against a catalog, without a model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Literal as TypingLiteral

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, URIRef
from rdflib.term import Node

from ontocast.check import messages
from ontocast.check.report import (
    FactsCheckFinding,
    FactsCheckKind,
    FactsCheckMetrics,
    FactsCheckReport,
)
from ontocast.check.scope import catalog_schema_terms, declared_terms, instances
from ontocast.onto.model import FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph, copy_triples
from ontocast.onto.scaffolding import FactScope
from ontocast.onto.well_known import well_known_terms
from ontocast.onto.xsd_lexical import TEMPORAL_DATATYPES
from ontocast.tool.facts_validation.connectivity import (
    isolated_instances,
    silent_instances,
    twin_labels,
)
from ontocast.tool.facts_validation.gate import validate_aggregated_facts
from ontocast.tool.facts_validation.grounding import node_of
from ontocast.tool.facts_validation.literal_repair import (
    range_mismatches,
    schema_retypings,
)
from ontocast.tool.facts_validation.shacl import run_shacl
from ontocast.tool.facts_validation.statements import StatementContract
from ontocast.tool.facts_validation.term_verdict import (
    TermPosition,
    TermVerdict,
    TermVerdictKind,
    term_verdict,
    used_terms,
)
from ontocast.tool.facts_validation.terms import (
    CatalogRoles,
    ValidationPolicy,
    _alias_candidates,
    _namespace_of,
    collect_catalog_terms,
    collect_declared_namespaces,
    value_class_base,
)
from ontocast.tool.facts_validation.unit_findings import (
    domain_violation_findings,
    empty_value_findings,
    role_outside_statement_findings,
)

_SEVERITY_ORDER = {"error": 0, "warning": 1}
_KIND_ORDER = {kind: index for index, kind in enumerate(FactsCheckKind)}
# Types that say nothing about what an instance is.
_VACUOUS_TYPES = frozenset({OWL.Thing, RDFS.Resource, OWL.NamedIndividual})

Vocabulary = TypingLiteral["strict", "open"]

# Shown when neither graph binds a prefix for these namespaces.
_DEFAULT_PREFIXES = {
    "rdf": str(RDF),
    "rdfs": str(RDFS),
    "owl": str(OWL),
    "xsd": str(XSD),
}


def check_facts(
    graph: Graph,
    ontology_graph: Graph | None,
    shapes_graph: Graph | None = None,
    *,
    vocabulary: Vocabulary = "strict",
    additional_standard_namespaces: Sequence[str] = (),
    instance_namespaces: Sequence[str] | None = None,
    max_findings: int | None = None,
    shacl_max_triples: int = 0,
) -> FactsCheckReport:
    """Check a facts graph against a catalog ontology and report what is wrong.

    The graph is never modified: every check that rewrites runs on a copy.
    Only the facts are checked: OntoCast's provenance (chunk nodes, reifiers,
    :meth:`~ontocast.onto.scaffolding.FactScope.facts`) is left out, and the
    document node is not an instance. With an empty or missing catalog,
    the checks that need one are skipped and the report says so
    (``catalog_empty``); the well-known-term checks and literal syntax still
    run.

    Args:
        graph: Facts graph, in the user's own namespaces.
        ontology_graph: Merged catalog ontology; ``None`` or empty when there
            is none.
        shapes_graph: Merged SHACL shapes, or ``None``.
        vocabulary: ``strict`` makes a term outside the catalog an error,
            ``open`` a warning.
        additional_standard_namespaces: Namespaces treated like the standard
            vocabularies.
        instance_namespaces: When given, only subjects under these namespaces
            are instances.
        max_findings: Cap on the returned findings; ``counts`` and
            ``findings_total`` stay complete.
        shacl_max_triples: Skip SHACL validation above this many triples;
            0 means no limit.

    Returns:
        The report.
    """
    work = FactScope.of(graph, None).facts()
    catalog = _as_rdf_graph(ontology_graph)
    catalog_empty = catalog is None
    policy = ValidationPolicy(
        additional_standard_namespaces=tuple(additional_standard_namespaces),
        vocabulary=vocabulary,
    )
    # Any IRI of the catalog is acceptable vocabulary; only its classes and
    # properties are schema terms, so its individuals stay instances.
    catalog_terms = collect_catalog_terms(catalog)
    schema_terms = catalog_schema_terms(catalog)
    found = instances(
        work,
        catalog_terms=schema_terms,
        standard_namespaces=policy.standard_namespaces(),
        restrict_to=instance_namespaces,
    )
    instance_iris = {str(instance) for instance in found}
    names = _Names(work, catalog)

    findings: list[FactsCheckFinding] = []
    roles = CatalogRoles.of([catalog] if catalog is not None else [])
    contract = roles.statement_contract(value_class_base(None, {}))
    term_findings, complete = _term_findings(
        work, catalog, catalog_terms, policy, names, roles, instance_namespaces
    )
    findings.extend(term_findings)
    findings.extend(_subject_findings(work, schema_terms, names))
    findings.extend(_literal_findings(work, catalog, roles, names))
    if catalog is not None:
        findings.extend(_domain_findings(work, catalog, instance_iris, names))
        findings.extend(
            _statement_shape_findings(
                work, catalog, contract, instance_namespaces, names
            )
        )
        findings.extend(
            _functional_findings(
                work, catalog, catalog_terms, instance_iris, policy, names
            )
        )

    structure = _structure_findings(
        work, catalog, catalog_terms, instance_iris, policy, names, contract
    )
    findings.extend(structure.findings)
    shacl_evaluated: bool | None = None
    shapes = _as_rdf_graph(shapes_graph)
    if shapes is not None:
        shacl_findings = _shacl_findings(
            work, catalog, shapes, shacl_max_triples, names
        )
        if shacl_findings is not None:
            findings.extend(shacl_findings)
            shacl_evaluated = True
    findings.sort(key=_sort_key)

    counts: dict[str, dict[str, int]] = {}
    for finding in findings:
        by_severity = counts.setdefault(finding.kind.value, {})
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
    total = len(findings)
    if max_findings is not None:
        findings = findings[:max_findings]
    # Suggestions cost a ranking against the catalog each: only the findings
    # returned get them.
    findings = [
        complete[id(finding)]() if id(finding) in complete else finding
        for finding in findings
    ]
    return FactsCheckReport(
        findings=findings,
        findings_total=total,
        counts=counts,
        metrics=FactsCheckMetrics(
            triples=len(work),
            instances=len(found),
            isolated=structure.isolated,
            untyped=structure.untyped,
            twin_labels=structure.twin_labels,
        ),
        shacl_evaluated=shacl_evaluated,
        catalog_empty=catalog_empty,
    )


def _rdf_copy(graph: Graph) -> RDFGraph:
    if isinstance(graph, RDFGraph):
        return graph.copy()
    copy = RDFGraph()
    copy_triples(graph, copy)
    for prefix, namespace in graph.namespaces():
        copy.bind(prefix, namespace)
    return copy


def _as_rdf_graph(graph: Graph | None) -> RDFGraph | None:
    """The catalog as an ``RDFGraph``, or ``None`` when it is missing or empty.

    The checks only read the catalog, so an ``RDFGraph`` is used as is.
    """
    if graph is None or not len(graph):
        return None
    if isinstance(graph, RDFGraph):
        return graph
    return _rdf_copy(graph)


class _Names:
    """Compact display names from the prefixes both graphs bind."""

    def __init__(self, *graphs: Graph | None) -> None:
        prefixes = dict(_DEFAULT_PREFIXES)
        for graph in graphs:
            if graph is None:
                continue
            for prefix, namespace in graph.namespaces():
                if prefix:
                    prefixes.setdefault(prefix, str(namespace))
        # Longest namespace first, so the most specific prefix wins.
        self._prefixes = sorted(
            ((namespace, prefix) for prefix, namespace in prefixes.items()),
            key=lambda item: (-len(item[0]), item[1]),
        )

    def __call__(self, iri: str) -> str:
        for namespace, prefix in self._prefixes:
            if iri.startswith(namespace):
                local = iri[len(namespace) :]
                if local and not any(char in local for char in "/#:"):
                    return f"{prefix}:{local}"
        return f"<{iri}>"


#: Which position a term used in several is judged in: as a predicate, else
#: as a class, else as an individual.
_POSITION_ORDER: dict[TermPosition, int] = {"predicate": 0, "type": 1, "object": 2}


def _term_findings(
    graph: RDFGraph,
    catalog: RDFGraph | None,
    catalog_terms: set[str],
    policy: ValidationPolicy,
    names: _Names,
    roles: CatalogRoles,
    instance_namespaces: Sequence[str] | None,
) -> tuple[list[FactsCheckFinding], dict[int, Callable[[], FactsCheckFinding]]]:
    """``unknown_term`` and ``non_catalog_vocabulary``, once per term.

    Every IRI the facts use (:func:`used_terms`: predicates, ``rdf:type``
    objects and other IRI objects) is judged by :func:`term_verdict`, as
    extraction judges it: an IRI object only by the closed vocabularies, an
    individual of a value also by its property's closed range (``contract``). It is asked for the strict verdict in both modes,
    so a term outside every known vocabulary is found under ``open`` too and
    reported there as a warning. Terms are judged by IRI: no namespace the
    catalog borrows from admits a term the catalog does not name, and a term
    missing from a namespace the catalog declares terms in is
    ``non_catalog_vocabulary`` too. A term the catalog defines is accepted
    first, so a catalog in a placeholder namespace accepts its own terms.

    Returns:
        The findings without suggestions, and for each (by ``id``) a function
        returning it with its suggestions and the message naming them.
    """
    declared_namespaces = collect_declared_namespaces(catalog)
    fallback_terms = policy.exempt_terms(graph, catalog)
    standard_namespaces = policy.standard_namespaces()
    contract = roles.statement_contract(value_class_base(None, {}))
    catalog_roles = (set(roles.properties), set(roles.classes))
    no_roles: tuple[set[str], set[str]] = (set(), set())

    def acceptable(text: str) -> bool:
        return text in catalog_terms or (
            term_verdict(
                text,
                catalog_terms=catalog_terms,
                known_terms=catalog_terms,
                fallback_terms=fallback_terms,
                declared_namespaces=declared_namespaces,
                catalog_namespaces=frozenset(),
                standard_namespaces=standard_namespaces,
                vocabulary="strict",
            )
            is None
        )

    uses: dict[str, int] = {}
    positions: dict[str, TermPosition] = {}
    closed: dict[str, frozenset[str]] = {}
    for term in used_terms(FactScope.of(graph, instance_namespaces), contract):
        uses[term.iri] = uses.get(term.iri, 0) + 1
        # A term used as a predicate is judged as one; a class next.
        if (
            term.iri not in positions
            or _POSITION_ORDER[term.position] < (_POSITION_ORDER[positions[term.iri]])
        ):
            positions[term.iri] = term.position
        if term.closed_range:
            closed[term.iri] = closed.get(term.iri, frozenset()) | term.closed_range

    findings: list[FactsCheckFinding] = []
    complete: dict[int, Callable[[], FactsCheckFinding]] = {}
    for text in sorted(uses):
        if text in catalog_terms:
            continue
        position = positions[text]
        judged = term_verdict(
            text,
            catalog_terms=catalog_terms,
            known_terms=catalog_terms,
            fallback_terms=fallback_terms,
            declared_namespaces=declared_namespaces,
            catalog_namespaces=frozenset(),
            standard_namespaces=standard_namespaces,
            vocabulary="strict",
            position=position,
            closed_range=closed.get(text, frozenset()),
        )
        if judged is None:
            continue
        if judged.kind is TermVerdictKind.PLACEHOLDER_NAMESPACE:
            kind, severity = FactsCheckKind.UNKNOWN_TERM, "error"
            pool, with_catalog = catalog_terms or None, True

            def message(suggestions: list[str], text: str = text) -> str:
                return messages.unknown_placeholder_term(names(text), suggestions)

        elif judged.kind is TermVerdictKind.WELL_KNOWN_MISSING:
            kind, severity = FactsCheckKind.UNKNOWN_TERM, "error"
            pool, with_catalog = well_known_terms(_namespace_of(text)), False

            def message(
                suggestions: list[str], text: str = text, judged: TermVerdict = judged
            ) -> str:
                return messages.unknown_well_known_term(
                    names(text), judged.vocabulary_title or "", suggestions
                )

        elif judged.kind is TermVerdictKind.CLOSED_RANGE_MISSING:
            kind, severity = FactsCheckKind.UNKNOWN_TERM, "error"
            pool, with_catalog = catalog_terms, True

            def message(suggestions: list[str], text: str = text) -> str:
                return messages.unknown_closed_range_term(names(text), suggestions)

        else:
            kind = FactsCheckKind.NON_CATALOG_VOCABULARY
            severity = "error" if policy.vocabulary == "strict" else "warning"
            pool, with_catalog = catalog_terms, True

            def message(suggestions: list[str], text: str = text) -> str:
                return messages.non_catalog_vocabulary(
                    names(text), uses[text], suggestions
                )

        finding = FactsCheckFinding(
            kind=kind, severity=severity, message=message([]), predicate=text
        )

        def with_suggestions(
            finding: FactsCheckFinding = finding,
            text: str = text,
            position: str = position,
            pool: set[str] | None = pool,
            with_catalog: bool = with_catalog,
            message: Callable[[list[str]], str] = message,
        ) -> FactsCheckFinding:
            candidates = (
                _alias_candidates(
                    URIRef(text),
                    graph,
                    pool,
                    ontology_graph=catalog if with_catalog else None,
                    position=position,
                    roles=catalog_roles if with_catalog else no_roles,
                )
                if pool is not None
                else []
            )
            # Only a term that is itself acceptable is worth offering.
            suggestions = [c for c in candidates if acceptable(c)]
            return finding.model_copy(
                update={
                    "suggestions": suggestions,
                    "message": message([names(c) for c in suggestions]),
                }
            )

        findings.append(finding)
        complete[id(finding)] = with_suggestions
    return findings, complete


def _subject_findings(
    graph: RDFGraph, catalog_terms: AbstractSet[str], names: _Names
) -> list[FactsCheckFinding]:
    """``schema_in_facts`` per declared term, ``catalog_subject`` per catalog term.

    A catalog term the upload redeclares is reported once, as
    ``schema_in_facts``.
    """
    declared = declared_terms(graph)
    findings: list[FactsCheckFinding] = []
    for term in sorted(declared):
        iri = URIRef(term)
        is_class = (iri, RDF.type, OWL.Class) in graph or (
            iri,
            RDF.type,
            RDFS.Class,
        ) in graph
        findings.append(
            FactsCheckFinding(
                kind=FactsCheckKind.SCHEMA_IN_FACTS,
                severity="warning",
                message=messages.schema_in_facts(
                    names(term), "class" if is_class else "property"
                ),
                subject=term,
            )
        )
    subjects = {str(s) for s in graph.subjects() if isinstance(s, URIRef)}
    for term in sorted((subjects & catalog_terms) - declared):
        findings.append(
            FactsCheckFinding(
                kind=FactsCheckKind.CATALOG_SUBJECT,
                severity="warning",
                message=messages.catalog_subject(names(term)),
                subject=term,
            )
        )
    return findings


def _domain_findings(
    graph: RDFGraph,
    catalog: RDFGraph,
    instance_iris: AbstractSet[str],
    names: _Names,
) -> list[FactsCheckFinding]:
    findings: list[FactsCheckFinding] = []
    for found in domain_violation_findings(graph, catalog):
        if found.subject not in instance_iris or found.predicate is None:
            continue
        types = sorted(
            str(value)
            for value in graph.objects(URIRef(found.subject), RDF.type)
            if isinstance(value, URIRef)
        )
        findings.append(
            FactsCheckFinding(
                kind=FactsCheckKind.DOMAIN_VIOLATION,
                severity="error",
                message=messages.domain_violation(
                    names(found.subject),
                    names(found.predicate),
                    [names(value) for value in types],
                    [names(value) for value in found.suggestions],
                ),
                subject=found.subject,
                predicate=found.predicate,
                suggestions=list(found.suggestions),
            )
        )
    return findings


def _statement_shape_findings(
    graph: RDFGraph,
    catalog: RDFGraph,
    contract: StatementContract,
    instance_namespaces: Sequence[str] | None,
    names: _Names,
) -> list[FactsCheckFinding]:
    """``role_outside_statement`` and ``empty_value``, as extraction finds them."""
    scope = FactScope.of(graph, instance_namespaces)
    findings = [
        FactsCheckFinding(
            kind=FactsCheckKind.ROLE_OUTSIDE_STATEMENT,
            severity="error",
            message=messages.role_outside_statement(
                _node_name(node_of(found.subject), names),
                names(found.predicate),
                found.facet,
            ),
            subject=found.subject,
            predicate=found.predicate,
            value=found.value,
            suggestions=list(found.suggestions),
        )
        for found in role_outside_statement_findings(scope, contract, catalog)
    ]
    findings.extend(
        FactsCheckFinding(
            kind=FactsCheckKind.EMPTY_VALUE,
            severity="warning",
            message=messages.empty_value(_node_name(node_of(found.subject), names)),
            subject=found.subject,
        )
        for found in empty_value_findings(scope, contract)
    )
    return findings


def _functional_findings(
    graph: RDFGraph,
    catalog: RDFGraph,
    catalog_terms: set[str],
    instance_iris: AbstractSet[str],
    policy: ValidationPolicy,
    names: _Names,
) -> list[FactsCheckFinding]:
    """``functional_violation`` from the gate, kept for instances only."""
    report = validate_aggregated_facts(
        graph,
        catalog,
        fact_namespaces=sorted({_namespace_of(iri) for iri in instance_iris}),
        confirmed_subjects=None,
        merge_provenance=None,
        non_catalog_severity="error" if policy.vocabulary == "strict" else "warning",
        additional_standard_namespaces=policy.additional_standard_namespaces,
        catalog_terms=catalog_terms,
    )
    findings: list[FactsCheckFinding] = []
    for found in report.findings:
        if (
            found.kind is not FactsValidationFindingKind.FUNCTIONAL_VIOLATION
            or found.subject not in instance_iris
            or found.predicate is None
        ):
            continue
        findings.append(
            FactsCheckFinding(
                kind=FactsCheckKind.FUNCTIONAL_VIOLATION,
                severity="error",
                message=messages.functional_violation(
                    names(found.subject), names(found.predicate), len(found.values)
                ),
                subject=found.subject,
                predicate=found.predicate,
            )
        )
    return findings


def _literal_findings(
    graph: RDFGraph, catalog: RDFGraph | None, roles: CatalogRoles, names: _Names
) -> list[FactsCheckFinding]:
    """``invalid_literal``, ``untyped_value`` and ``temporal_range_mismatch``.

    Each literal gets at most one finding. A literal that is no value of its
    own datatype is ``invalid_literal`` whatever the catalog says. The rest is
    judged against the declared datatype ranges, the catalog's and the
    well-known vocabularies', on a copy retyped as at parse time: a value the
    retyping fixes is ``untyped_value``; a literal that is no value of the
    range (:func:`~ontocast.tool.facts_validation.literal_repair.range_mismatches`)
    is ``temporal_range_mismatch`` when it is a typed value of another type on
    a date, time or duration range, and ``invalid_literal`` otherwise. Without
    a catalog only the first check runs.
    """
    findings: list[FactsCheckFinding] = []
    clean = RDFGraph()
    for triple in graph:
        subject, predicate, obj = triple
        # The read-only validity test: the quarantining partition logs a
        # warning meant for extraction.
        if isinstance(obj, Literal) and not RDFGraph.is_well_formed_literal(obj):
            findings.append(
                _literal_finding(
                    FactsCheckKind.INVALID_LITERAL,
                    "error",
                    subject,
                    predicate,
                    obj,
                    messages.invalid_literal(
                        _node_name(subject, names),
                        names(str(predicate)),
                        str(obj),
                        names(str(obj.datatype)),
                    ),
                )
            )
            continue
        clean.add(triple)
    if catalog is None:
        return findings

    for old, new in schema_retypings(clean, catalog):
        clean.remove(old)
        clean.add(new)
        subject, predicate, before = old
        after = new[2]
        # A number promoted to a wider numeric type was already typed.
        if (
            not isinstance(before, Literal)
            or not isinstance(after, Literal)
            or before.datatype not in (None, XSD.string)
        ):
            continue
        findings.append(
            _literal_finding(
                FactsCheckKind.UNTYPED_VALUE,
                "warning",
                subject,
                predicate,
                before,
                messages.untyped_value(
                    _node_name(subject, names),
                    names(str(predicate)),
                    str(before),
                    names(str(after.datatype)),
                ),
                value=after.n3(),
            )
        )

    # Every literal that is no value of its property's declared range, as the
    # library judges it at extraction.
    for triple, record in range_mismatches(clean, catalog, roles):
        subject, predicate, obj = triple
        if not isinstance(obj, Literal):
            continue
        temporal = any(
            URIRef(dt) in TEMPORAL_DATATYPES for dt in record.expected_datatypes
        )
        is_text = obj.datatype in (None, XSD.string)
        if temporal and not is_text and obj.language is None:
            # A value of another type where a date, time or duration belongs.
            kind = FactsCheckKind.TEMPORAL_RANGE_MISMATCH
            message = messages.temporal_range_mismatch(
                _node_name(subject, names),
                names(str(predicate)),
                str(obj),
                names(str(obj.datatype)),
                " or ".join(names(dt) for dt in record.expected_datatypes),
            )
        else:
            kind = FactsCheckKind.INVALID_LITERAL
            message = messages.invalid_literal(
                _node_name(subject, names),
                names(str(predicate)),
                str(obj),
                " or ".join(names(dt) for dt in record.expected_datatypes),
                obj.language,
            )
        findings.append(
            _literal_finding(kind, "error", subject, predicate, obj, message)
        )
    return findings


def _literal_finding(
    kind: FactsCheckKind,
    severity: TypingLiteral["error", "warning"],
    subject: Node,
    predicate: Node,
    obj: Node,
    message: str,
    *,
    value: str | None = None,
) -> FactsCheckFinding:
    return FactsCheckFinding(
        kind=kind,
        severity=severity,
        message=message,
        subject=str(subject),
        predicate=str(predicate),
        value=str(obj) if value is None else value,
    )


def _node_name(node: Node, names: _Names) -> str:
    return names(str(node)) if isinstance(node, URIRef) else f"_:{node}"


def _sort_key(finding: FactsCheckFinding) -> tuple[int, int, str, str]:
    return (
        _SEVERITY_ORDER[finding.severity],
        _KIND_ORDER[finding.kind],
        finding.subject or "",
        finding.predicate or "",
    )


def _shacl_findings(
    graph: RDFGraph,
    catalog: RDFGraph | None,
    shapes: RDFGraph,
    max_triples: int,
    names: _Names,
) -> list[FactsCheckFinding] | None:
    """``shacl`` findings, or ``None`` when validation did not run."""
    # pyshacl adds axioms to the graphs it is given: hand it copies, so the
    # caller's shapes and catalog (possibly one graph) stay as they were.
    violations = run_shacl(
        graph,
        shapes.copy(),
        ontology_graph=catalog.copy() if catalog is not None else None,
        max_triples=max_triples,
    )
    if violations is None:
        return None
    findings: list[FactsCheckFinding] = []
    for violation in violations:
        focus = violation.focus
        findings.append(
            FactsCheckFinding(
                kind=FactsCheckKind.SHACL,
                severity=violation.severity,
                message=messages.shacl_violation(
                    _node_name(focus, names) if focus is not None else "The graph",
                    violation.message,
                ),
                subject=str(focus) if focus is not None else None,
                predicate=str(violation.path) if violation.path else None,
                value=str(violation.value) if violation.value is not None else None,
                source_shape=(
                    str(violation.source_shape)
                    if violation.source_shape is not None
                    else None
                ),
            )
        )
    return findings


@dataclass
class _Structure:
    findings: list[FactsCheckFinding] = field(default_factory=list)
    isolated: int = 0
    untyped: int = 0
    twin_labels: int = 0

    def warn(
        self,
        kind: FactsCheckKind,
        message: str,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> None:
        self.findings.append(
            FactsCheckFinding(
                kind=kind,
                severity="warning",
                message=message,
                subject=subject,
                predicate=predicate,
            )
        )


def _structure_findings(
    graph: RDFGraph,
    catalog: RDFGraph | None,
    catalog_terms: AbstractSet[str],
    instance_iris: AbstractSet[str],
    policy: ValidationPolicy,
    names: _Names,
    contract: StatementContract,
) -> _Structure:
    """Untyped, isolated, silent, dangling and twin-label findings.

    The connectivity helpers scope by namespace; their results are kept only
    for subjects the per-IRI instance set holds. Typing is judged directly,
    since a class may share the instances' namespace; a catalog individual
    keeps the types the catalog gives it.
    """
    result = _Structure()
    namespaces = sorted({_namespace_of(iri) for iri in instance_iris})
    standard = policy.standard_namespaces()
    declared = declared_terms(graph)

    for iri in sorted(instance_iris):
        stated = list(graph.objects(URIRef(iri), RDF.type))
        if catalog is not None:
            stated.extend(catalog.objects(URIRef(iri), RDF.type))
        types = [
            str(t) for t in stated if isinstance(t, URIRef) and t not in _VACUOUS_TYPES
        ]
        if catalog is None:
            typed = bool(types)
        else:
            typed = any(
                t in catalog_terms or t in declared or t.startswith(standard)
                for t in types
            )
        if not typed:
            result.untyped += 1
            result.warn(
                FactsCheckKind.UNTYPED_INSTANCE,
                messages.untyped_instance(names(iri), []),
                iri,
            )

    if namespaces:
        isolated = {str(n) for n in isolated_instances(graph, namespaces)}
        silent = {str(n) for n in silent_instances(graph, namespaces)}
        for iri in sorted(isolated & set(instance_iris)):
            if iri in silent:
                message = messages.silent_instance(names(iri))
                kind = FactsCheckKind.SILENT_INSTANCE
            else:
                message = messages.isolated_instance(names(iri))
                kind = FactsCheckKind.ISOLATED_INSTANCE
            result.isolated += 1
            result.warn(kind, message, iri)
        twins = twin_labels(graph, namespaces, contract=contract)
        result.twin_labels = len(twins)
        for label, count in twins.items():
            result.warn(FactsCheckKind.TWIN_LABELS, messages.twin_labels(label, count))

    subjects = {s for s in graph.subjects() if isinstance(s, URIRef)}
    dangling: dict[str, set[str]] = {}
    for subject, predicate, obj in graph:
        if (
            predicate in (RDF.type, OWL.sameAs)
            or not isinstance(obj, URIRef)
            or not isinstance(predicate, URIRef)
            or str(subject) not in instance_iris
            or obj in subjects
            or str(obj) in catalog_terms
            or str(obj).startswith(standard)
            or not str(obj).startswith(tuple(namespaces))
        ):
            continue
        dangling.setdefault(str(obj), set()).add(str(predicate))
    for obj in sorted(dangling):
        predicates = sorted(dangling[obj])
        result.warn(
            FactsCheckKind.DANGLING_REFERENCE,
            messages.dangling_reference(names(obj), [names(p) for p in predicates]),
            obj,
            ", ".join(predicates),
        )
    return result
