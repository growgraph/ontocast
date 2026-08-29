"""SHACL execution, shape assembly, autofix repairs, and the catalog lint.

``shacl_catalog_contradictions`` cross-checks the shapes against the term
validator: a property the shapes require but the validator would flag as
unknown silently destroys extracted data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal as TypingLiteral

from pydantic import BaseModel, Field
from rdflib import RDF, RDFS, SKOS, Literal, URIRef
from rdflib.namespace import SH
from rdflib.term import Node

from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsValidationFinding,
    FactsValidationFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import (
    RDFGraph,
    copy_triples,
    drop_reifiers_mentioning,
    retarget_reifiers,
)
from ontocast.tool.facts_validation.literal_repair import (
    _literal_parses_as,
)
from ontocast.tool.facts_validation.terms import (
    ValidationPolicy,
    _in_fact_scope,
    _namespace_of,
    build_surface_index,
    collect_catalog_terms,
    collect_declared_namespaces,
    resolve_unique_surface,
)

logger = logging.getLogger(__name__)


class ShaclViolation(BaseModel):
    """One SHACL validation result, in the form the repair pass needs.

    ``FactsValidationFinding`` is the reporting shape and deliberately flat;
    this keeps the RDF terms (focus node, path, offending value, constraint
    component) so a repair can act on them.
    """

    model_config = {"arbitrary_types_allowed": True}

    focus: Node | None = None
    path: URIRef | None = None
    value: Node | None = None
    component: URIRef | None = None
    # Node, not URIRef: the common authoring style is an inline
    # ``sh:property [ sh:path … ; sh:datatype … ]``, whose shape is a blank
    # node. Narrowing to URIRef dropped it and left every such violation
    # unrepairable. pyshacl reports the same BNode the shapes graph holds, so
    # the datatype lookup resolves.
    source_shape: Node | None = None
    severity: TypingLiteral["error", "warning"] = "error"
    message: str = "SHACL constraint violated."

    def as_finding(self) -> FactsValidationFinding:
        """Project onto the reported finding shape."""
        return FactsValidationFinding(
            kind=FactsValidationFindingKind.SHACL,
            severity=self.severity,
            message=self.message,
            subject=str(self.focus) if self.focus is not None else "",
            predicate=str(self.path) if self.path is not None else "",
            values=[str(self.value)] if self.value is not None else [],
            component=str(self.component) if self.component is not None else "",
            source_shape=(
                str(self.source_shape) if self.source_shape is not None else ""
            ),
        )


def run_shacl(
    graph: RDFGraph,
    shapes_graph: RDFGraph,
    *,
    ontology_graph: RDFGraph | None = None,
    inference: str = "rdfs",
    advanced: bool = True,
    max_triples: int = 0,
) -> list[ShaclViolation] | None:
    """Validate ``graph`` against ``shapes_graph``, returning the violations.

    Reaching here means shapes were found, so the caller expects validation to
    happen: a missing extra or a skipped run is reported at warning level, not
    debug. Silently returning "no violations" is indistinguishable from
    "conforms", so those cases return ``None``.

    The ontology context is mixed in (``ont_graph``) rather than left out. A
    facts graph states that a value uses ``unit:DAY``; that the individual *is*
    a ``qudt:Unit`` is stated only in the catalog. Validating the facts alone
    therefore fails every ``sh:class`` constraint pointing at a catalog
    individual — violations that describe the missing schema, not the data.

    RDFS inference is the default for the same reason. SHACL resolves class
    targets through ``rdfs:subClassOf`` on its own, but property paths carry no
    entailment: a shape on ``obs:hasResult`` does not see the
    ``life:hasStorageResult`` the renderer emitted, and reports the more
    specific statement as a missing one, so turning inference off raises the
    violation count rather than lowering it.

    Args:
        graph: Data graph to validate.
        shapes_graph: Shapes to validate against.
        ontology_graph: Schema mixed into the data graph for validation.
        inference: pyshacl pre-inference (``none`` / ``rdfs`` / ``owlrl``).
        advanced: Enable SHACL Advanced Features.
        max_triples: Skip validation above this graph size; 0 disables.

    Returns:
        Violations in report order, or ``None`` when validation did not run.
    """
    try:
        import pyshacl
    except ImportError:
        logger.warning(
            "SHACL shapes are configured but pyshacl is not installed; "
            "skipping SHACL validation. Install the extra: uv sync --extra shacl"
        )
        return None

    if max_triples and len(graph) > max_triples:
        logger.warning(
            "Skipping SHACL validation: %d triples exceeds "
            "FACTS_SHACL_MAX_TRIPLES=%d. The graph is unvalidated, not conformant.",
            len(graph),
            max_triples,
        )
        return None

    # pyshacl clones and mixes the data graph through plain rdflib graphs,
    # which cannot hold the RDF 1.2 triple terms an oxigraph-backed aggregated
    # graph carries (rdflib ``Graph.add`` asserts on them). Hand pyshacl a
    # sanitised copy; the dropped reification provenance carries no shape
    # targets, so validation loses nothing.
    data_graph = RDFGraph()
    copy_triples(graph, data_graph, origin="run_shacl")
    for prefix, namespace in graph.namespaces():
        data_graph.bind(prefix, namespace, override=True)

    conforms, results_graph, _ = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=(
            ontology_graph
            if ontology_graph is not None and len(ontology_graph)
            else None
        ),
        inference=inference,
        advanced=advanced,
        abort_on_first=False,
    )
    if conforms:
        return []

    violations: list[ShaclViolation] = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity_iri = results_graph.value(result, SH.resultSeverity)
        message = results_graph.value(result, SH.resultMessage)
        path = results_graph.value(result, SH.resultPath)
        component = results_graph.value(result, SH.sourceConstraintComponent)
        source_shape = results_graph.value(result, SH.sourceShape)
        violations.append(
            ShaclViolation(
                focus=results_graph.value(result, SH.focusNode),
                path=path if isinstance(path, URIRef) else None,
                value=results_graph.value(result, SH.value),
                component=component if isinstance(component, URIRef) else None,
                source_shape=source_shape,
                severity=("error" if severity_iri == SH.Violation else "warning"),
                message=str(message) if message else "SHACL constraint violated.",
            )
        )
    return violations


def collect_shacl_shapes(
    ontology_graph: RDFGraph | None, stored_shapes: RDFGraph | None
) -> RDFGraph | None:
    """Assemble the SHACL shapes graph for the validation gate.

    Sources: the deployment's shapes partition (``stored_shapes``, resolved by
    :class:`~ontocast.tool.shapes_catalog.ShapesCatalog` -- seeded from
    ``FACTS_SHAPES_DIR`` and mutable over ``/shapes``), plus the ontology
    context itself when it already carries ``sh:NodeShape`` declarations inline
    -- the zero-config path for catalogs that ship shapes next to their schema.

    Args:
        ontology_graph: Ontology context offered to the renderer.
        stored_shapes: Merged shapes graph from the shapes partition.

    Returns:
        RDFGraph | None: The shapes to validate against, or ``None`` when there
        are none -- which is what keeps ``shacl_evaluated`` at ``None``
        ("never checked") rather than reporting a clean run.
    """
    shapes = RDFGraph()
    if stored_shapes is not None and len(stored_shapes):
        shapes += stored_shapes
    node_shape = SH.NodeShape
    if ontology_graph is not None and (None, RDF.type, node_shape) in ontology_graph:
        shapes += ontology_graph
    return shapes if len(shapes) else None


def shacl_catalog_contradictions(
    shapes_graph: RDFGraph | None,
    ontology_graph: RDFGraph | None,
    *,
    policy: ValidationPolicy | None = None,
) -> list[str]:
    """Property paths the shapes require but the unit validator would flag.

    A SHACL property shape with ``sh:minCount >= 1`` demands a property that
    the deterministic UNKNOWN_TERM check — same closure rules, same
    exemptions — would report as not existing. Data cannot satisfy both: the
    renderer is ordered to remove exactly what validation requires. Found live
    in practice, where shapes required ``qudt:numericValue``
    while the validator's mandatory findings drove repair renders to delete
    it. Callers log the returned IRIs as configuration errors.
    """
    if shapes_graph is None or ontology_graph is None:
        return []
    catalog_terms = collect_catalog_terms(ontology_graph)
    if not catalog_terms:
        return []
    policy = policy or ValidationPolicy()
    declared_namespaces = collect_declared_namespaces(ontology_graph)
    standard_namespaces = policy.standard_namespaces()
    fallback_terms = policy.exempt_terms(shapes_graph, ontology_graph)
    required: set[str] = set()
    for shape, path in shapes_graph.subject_objects(SH.path):
        if not isinstance(path, URIRef):
            continue
        min_count = next(shapes_graph.objects(shape, SH.minCount), None)
        try:
            if min_count is None or int(str(min_count)) < 1:
                continue
        except ValueError:
            continue
        required.add(str(path))
    contradictions = [
        term
        for term in sorted(required)
        if _namespace_of(term) in declared_namespaces
        and term not in catalog_terms
        and term not in fallback_terms
        and not _namespace_of(term).startswith(standard_namespaces)
    ]
    return contradictions


# --- LLM-free repair of SHACL violations -------------------------------------
#
# The contract for everything below: a repair either rewrites a term the
# catalog already declares, or removes a node that asserts nothing. No repair
# invents a value. A node carrying real data but missing a required property is
# left alone and reported -- filling it in would be fabrication, and dropping it
# would be data loss.

# Predicates that carry no assertion about the world: a node holding only these
# is a placeholder for an extraction that did not happen.
_EMPTY_NODE_PREDICATES = frozenset({RDF.type, RDFS.label, SKOS.prefLabel})

_RETYPABLE_COMPONENTS = frozenset({SH.DatatypeConstraintComponent})
_IRI_RESOLVABLE_COMPONENTS = frozenset(
    {SH.ClassConstraintComponent, SH.NodeKindConstraintComponent}
)
_PRUNABLE_COMPONENTS = frozenset({SH.MinCountConstraintComponent})


class ShaclRepairResult(BaseModel):
    """Outcome of the LLM-free SHACL repair pass."""

    model_config = {"arbitrary_types_allowed": True}

    graph: RDFGraph
    records: list[GraphRepairRecord] = Field(default_factory=list)
    violations_before: int = 0
    violations_after: int = 0
    passes_applied: int = 0
    reverted: bool = False
    ran: bool = False


def _node_in_graph(graph: RDFGraph, node: Node) -> bool:
    """True when ``node`` appears in ``graph`` as a subject or an object.

    With the ontology context mixed into validation, pyshacl reports focus
    nodes that live only in the catalog. Those are not the gate's to repair,
    and "absent from the facts graph" must never read as "asserts nothing".
    """
    return (node, None, None) in graph or (None, None, node) in graph


def _violation_in_fact_scope(
    graph: RDFGraph, focus: Node | None, fact_namespaces: Sequence[str]
) -> bool:
    """True when a violation on ``focus`` belongs to the facts graph.

    The gate's boundary, stated once. An IRI is scoped by namespace; a blank
    node carries none, so presence in the facts graph is the test — the same
    rule the repair pass applies. Splitting these apart is what let a blank-node
    violation be repaired while being filtered out of the report, so the
    findings under-counted exactly the nodes the gate had acted on.
    """
    if focus is None:
        return True
    if isinstance(focus, URIRef):
        # _in_fact_scope admits everything when no namespaces are configured.
        return _in_fact_scope(focus, [ns for ns in fact_namespaces if ns])
    return _node_in_graph(graph, focus)


def _node_asserts_nothing(graph: RDFGraph, node: Node) -> bool:
    """True when ``node`` is in the graph but carries nothing beyond typing/labels."""
    outgoing = list(graph.predicate_objects(node))
    if not outgoing:
        # Only a node the graph actually references is an empty placeholder;
        # a node absent from the graph entirely is simply not ours.
        return (None, None, node) in graph
    return all(predicate in _EMPTY_NODE_PREDICATES for predicate, _ in outgoing)


def _plan_retarget(
    retargets: dict[tuple, tuple], removed: tuple, replacement: tuple
) -> None:
    """Record that ``removed``'s reifier should follow it onto ``replacement``.

    First writer wins. Two violations can fire on one triple — a ``sh:datatype``
    and a ``sh:nodeKind`` report on the same literal — and the pass appends both
    replacements, but a reifier reifies exactly one statement. Keeping the first
    makes the retarget deterministic (violation order) without changing which
    replacements are applied.
    """
    if removed in retargets and retargets[removed] != replacement:
        logger.debug(
            "SHACL autofix: %s is already retargeted to %s; keeping the first",
            removed,
            retargets[removed],
        )
        return
    retargets[removed] = replacement


class _ShaclRepairPlan(BaseModel):
    """One round's derived edits, before the accept/revert test."""

    model_config = {"arbitrary_types_allowed": True}

    removals: list[tuple] = Field(default_factory=list)
    additions: list[tuple] = Field(default_factory=list)
    records: list[GraphRepairRecord] = Field(default_factory=list)
    #: Nodes deleted outright. Referenced from inside reification triple terms,
    #: which the removal list cannot express, so their reifiers are swept.
    pruned: set[Node] = Field(default_factory=set)
    #: Removed triple -> its replacement, for repairs that rewrite a statement
    #: rather than delete it. Their reifiers are moved, not dropped.
    retargets: dict[tuple, tuple] = Field(default_factory=dict)


def _shacl_repairs_for(
    graph: RDFGraph,
    shapes_graph: RDFGraph,
    violations: Sequence[ShaclViolation],
    *,
    mode: str,
    surface_index: dict[str, set[str]],
    fact_namespaces: Sequence[str],
) -> _ShaclRepairPlan:
    """Derive one round's edits from a set of violations."""
    removals: list[tuple] = []
    additions: list[tuple] = []
    records: list[GraphRepairRecord] = []
    pruned: set[Node] = set()
    retargets: dict[tuple, tuple] = {}

    for violation in violations:
        if violation.severity != "error" or violation.focus is None:
            continue
        # Ontology entities are not the gate's business to rewrite, and catalog
        # blank nodes (OWL restrictions, property shapes) reported via the
        # mixed-in ontology stay untouched.
        if not _violation_in_fact_scope(graph, violation.focus, fact_namespaces):
            continue
        component = violation.component
        shape = violation.source_shape

        if (
            component in _RETYPABLE_COMPONENTS
            and violation.path is not None
            and isinstance(violation.value, Literal)
            and shape is not None
        ):
            target = shapes_graph.value(shape, SH.datatype)
            lexical = str(violation.value).strip()
            if (
                isinstance(target, URIRef)
                and violation.value.datatype != target
                and _literal_parses_as(lexical, target)
            ):
                retyped = Literal(lexical, datatype=target)
                removed = (violation.focus, violation.path, violation.value)
                replacement = (violation.focus, violation.path, retyped)
                removals.append(removed)
                additions.append(replacement)
                _plan_retarget(retargets, removed, replacement)
                records.append(
                    GraphRepairRecord(
                        kind=FactsGateRepairKind.SHACL_RETYPE,
                        source=f"{violation.path} {violation.value.n3()}",
                        target=retyped.n3(),
                    )
                )
            continue

        if (
            component in _IRI_RESOLVABLE_COMPONENTS
            and violation.path is not None
            and isinstance(violation.value, Literal)
        ):
            resolved = resolve_unique_surface(surface_index, str(violation.value))
            if resolved is not None:
                removed = (violation.focus, violation.path, violation.value)
                replacement = (violation.focus, violation.path, resolved)
                removals.append(removed)
                additions.append(replacement)
                _plan_retarget(retargets, removed, replacement)
                records.append(
                    GraphRepairRecord(
                        kind=FactsGateRepairKind.SHACL_CODE_RESOLVED,
                        source=f"{violation.path} {violation.value.n3()}",
                        target=str(resolved),
                    )
                )
            continue

        if (
            mode == "prune"
            and component in _PRUNABLE_COMPONENTS
            and violation.focus not in pruned
            and _node_asserts_nothing(graph, violation.focus)
        ):
            referrers = {
                subject for subject, _ in graph.subject_predicates(violation.focus)
            }
            if len(referrers) > 1:
                # Shared by several subjects: removing it would silently change
                # statements that were never validated here.
                continue
            pruned.add(violation.focus)
            incoming = list(graph.triples((None, None, violation.focus)))
            outgoing = [
                (violation.focus, predicate, obj)
                for predicate, obj in graph.predicate_objects(violation.focus)
            ]
            removals.extend(incoming)
            removals.extend(outgoing)
            records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.SHACL_PRUNE,
                    source=str(violation.focus),
                    target="",
                    triple_count=len(incoming) + len(outgoing),
                )
            )

    return _ShaclRepairPlan(
        removals=removals,
        additions=additions,
        records=records,
        pruned=pruned,
        retargets=retargets,
    )


def _fact_scope_violations(
    graph: RDFGraph,
    violations: Sequence[ShaclViolation],
    fact_namespaces: Sequence[str],
) -> list[ShaclViolation]:
    """Violations that would survive the reporting filter.

    Shares :func:`_violation_in_fact_scope` with the report and the repair pass,
    so the ``violations_before``/``violations_after`` metrics count the same
    population as ``conforms`` does — with the ontology mixed in, the raw
    pyshacl count includes catalog nodes the report never shows.
    """
    return [
        violation
        for violation in violations
        if _violation_in_fact_scope(graph, violation.focus, fact_namespaces)
    ]


def apply_shacl_repairs(
    graph: RDFGraph,
    shapes_graph: RDFGraph | None,
    ontology_graph: RDFGraph | None,
    *,
    mode: str = "prune",
    passes: int = 1,
    fact_namespaces: Sequence[str] = (),
    code_predicates: Sequence[str] = (),
    inference: str = "rdfs",
    advanced: bool = True,
    max_triples: int = 0,
    initial_violations: Sequence[ShaclViolation] | None = None,
) -> ShaclRepairResult:
    """Repair SHACL violations in code, with no LLM round-trip.

    Bounded ``validate -> repair -> revalidate`` loop. A pass is kept only when
    it strictly reduces the violation count: a repair that trades triples for
    no conformance gain is reverted, the same discipline the un-merge repair
    uses.

    Repairs by constraint component:
        - ``sh:datatype``: retype a literal that parses as the declared
          datatype (``"2019"^^xsd:string`` -> ``"2019"^^xsd:gYear``).
        - ``sh:class`` / ``sh:nodeKind``: replace a string literal with the one
          catalog IRI declaring it as a surface form (``qudt:unit "meV"`` ->
          ``unit:MilliElectronVolt``). Ambiguous forms are left reported.
        - ``sh:minCount`` (mode ``prune`` only): drop a focus node that asserts
          nothing beyond ``rdf:type``/``rdfs:label`` and is referenced by at
          most one subject, together with that reference.

    Everything else -- ``sh:maxCount`` (owned by the functional-violation and
    un-merge machinery), ``sh:not``, ``sh:qualifiedValueShape``, SPARQL
    constraints -- is reported, never repaired.

    Args:
        graph: Aggregated facts graph, repaired **in place**: it may be
            oxigraph-backed and carry RDF 1.2 triple terms, which a copied
            rdflib graph would silently drop. A pass that fails the accept
            test is rolled back triple-for-triple instead.
        shapes_graph: Shapes to validate against; ``None`` disables the pass.
        ontology_graph: Merged ontology context, indexed for surface forms.
        mode: ``off`` | ``rewrite`` (rewrites only) | ``prune`` (also prunes).
        passes: Maximum repair rounds.
        fact_namespaces: Only nodes under these namespaces are repaired.
        code_predicates: Code-bearing predicates for surface resolution.
        inference: pyshacl pre-inference mode.
        advanced: Enable SHACL Advanced Features.
        max_triples: Skip validation above this graph size; 0 disables.
        initial_violations: Violations already computed for ``graph`` with the
            same parameters (e.g. by the reporting pass), reused to skip the
            redundant first validation.

    Returns:
        The repaired graph, the applied repair records, and fact-scoped
        violation counts before and after (the population ``conforms`` is
        judged on; the loop's accept test uses the raw count internally).
    """
    if mode == "off" or shapes_graph is None or not len(shapes_graph) or passes <= 0:
        return ShaclRepairResult(graph=graph)

    def _validate(target: RDFGraph) -> list[ShaclViolation] | None:
        return run_shacl(
            target,
            shapes_graph,
            ontology_graph=ontology_graph,
            inference=inference,
            advanced=advanced,
            max_triples=max_triples,
        )

    violations = (
        list(initial_violations) if initial_violations is not None else _validate(graph)
    )
    if violations is None:
        return ShaclRepairResult(graph=graph)

    def _scoped_count(candidates: Sequence[ShaclViolation]) -> int:
        return len(_fact_scope_violations(graph, candidates, fact_namespaces))

    result = ShaclRepairResult(
        graph=graph,
        violations_before=_scoped_count(violations),
        violations_after=_scoped_count(violations),
        ran=True,
    )
    surface_index = build_surface_index(ontology_graph, code_predicates)

    def _rollback(added: Sequence[tuple], removed: Sequence[tuple]) -> None:
        for triple in added:
            graph.remove(triple)
        for triple in removed:
            graph.add(triple)

    for _ in range(passes):
        if not violations:
            break
        plan = _shacl_repairs_for(
            graph,
            shapes_graph,
            violations,
            mode=mode,
            surface_index=surface_index,
            fact_namespaces=fact_namespaces,
        )
        records = plan.records
        if not records:
            break

        applied_removals: list[tuple] = []
        applied_removal_set: set[tuple] = set()
        seen_removals: set[tuple] = set()
        for triple in plan.removals:
            if triple in seen_removals:
                continue
            seen_removals.add(triple)
            if triple in graph:
                graph.remove(triple)
                applied_removals.append(triple)
                applied_removal_set.add(triple)
        applied_additions: list[tuple] = []
        for triple in plan.additions:
            if triple not in graph:
                graph.add(triple)
                applied_additions.append(triple)

        candidate_violations = _validate(graph)
        if candidate_violations is None:
            _rollback(applied_additions, applied_removals)
            break
        if len(candidate_violations) >= len(violations):
            logger.warning(
                "SHACL autofix: pass did not reduce violations (%d -> %d); "
                "keeping the pre-repair graph",
                len(violations),
                len(candidate_violations),
            )
            _rollback(applied_additions, applied_removals)
            result.reverted = True
            break

        # Only once the pass is accepted, and deliberately after the accept test
        # rather than alongside the removals: reification quads cannot travel
        # through _rollback (rdflib cannot add or remove a triple-term triple),
        # and validation runs on a copy with triple terms stripped, so this
        # changes no count and needs no undo.
        #
        # Retarget before sweeping. A statement that is retyped *and* then
        # pruned in the same pass has to be swept at its new triple term, which
        # the retarget has already installed -- so prune still wins, by the
        # sweep matching rather than by the retarget happening to miss.
        retargeted = retarget_reifiers(
            graph,
            {
                removed: replacement
                for removed, replacement in plan.retargets.items()
                if removed in applied_removal_set and replacement in graph
            },
        )
        swept = drop_reifiers_mentioning(graph, plan.pruned)

        provenance_note = ", ".join(
            note
            for note in (
                f"{retargeted} provenance quad(s) retargeted" if retargeted else "",
                f"{swept} orphaned provenance quad(s) swept" if swept else "",
            )
            if note
        )
        logger.info(
            "SHACL autofix: %d repair(s) applied, violations %d -> %d%s",
            len(records),
            len(violations),
            len(candidate_violations),
            f", {provenance_note}" if provenance_note else "",
        )
        violations = candidate_violations
        result.records.extend(records)
        result.passes_applied += 1
        result.violations_after = _scoped_count(candidate_violations)

    return result


# Scaffolding every facts graph uses regardless of catalog: flagging rdfs:label
