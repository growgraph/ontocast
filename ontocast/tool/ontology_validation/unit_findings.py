"""Deterministic per-unit findings on a unit's ontology delta.

Everything here runs against the unit's net **insert/delete delta**
(:meth:`~ontocast.onto.unit_states.UnitOntologyState.build_delta`), never the
whole working graph: the working graph is ``snapshot + delta``, so validating
it would test the shared catalog context against itself and attribute every
pre-existing third-party defect to this unit. Two facts-side rules are
deliberately absent for the same reason:

- ``UNKNOWN_TERM`` is semantically inverted here — minting new terms in a
  writable namespace is the ontology renderer's entire job;
- connectivity is not checked — a per-unit delta is by construction a few
  terms connecting to the *snapshot* rather than to each other, and the
  document-level ``STRUCTURAL_CHECK`` node already owns that concern where the
  context exists.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rdflib import OWL, RDF, RDFS, SKOS, BNode, Literal, URIRef

from ontocast.onto.graph_prune import (
    MIN_MEANINGFUL_RESTRICTION_PREDICATES,
    count_meaningful_restriction_predicates,
)
from ontocast.onto.model import (
    OntologyUnitFinding,
    OntologyUnitFindingKind,
    TripleFix,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_vocabulary
from ontocast.tool.agg.signatures import harvest_max_one_predicates
from ontocast.tool.facts_validation.terms import (
    _FORBIDDEN_NAMESPACES,
    ValidationPolicy,
    _catalog_term_roles,
    _namespace_of,
    build_surface_index,
    collect_declared_namespaces,
    superclass_closure,
)
from ontocast.tool.ontology_validation.reconcile import (
    match_catalog_term,
    minted_roles,
    surfaces_of,
)
from ontocast.tool.ontology_validation.shape import (
    CatalogSurfaceIndex,
    first_label,
    has_own_structure,
    minted_class_iris,
    named_class_depth,
    specific_label_reason,
    value_set_families,
)

logger = logging.getLogger(__name__)

#: Restrictions carrying a minimum this high contradict any max-1 declaration.
_MIN_CARDINALITY_PREDICATES = (OWL.minCardinality, OWL.minQualifiedCardinality)

#: Predicates whose literal object names a term (used for label collisions
#: and the missing-label check).
_LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)


def _new_subjects(inserts: RDFGraph) -> list[URIRef]:
    """URIRef subjects the delta asserts anything about, sorted."""
    return sorted({s for s in inserts.subjects() if isinstance(s, URIRef)}, key=str)


def _declared_class_or_property_subjects(inserts: RDFGraph) -> set[URIRef]:
    """Insert subjects the delta itself declares as a class or a property."""
    declared: set[URIRef] = set()
    class_types = (OWL.Class, RDFS.Class)
    property_types = (
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        RDF.Property,
    )
    for type_iri in (*class_types, *property_types):
        for subject in inserts.subjects(RDF.type, type_iri):
            if isinstance(subject, URIRef):
                declared.add(subject)
    for predicate in (RDFS.subClassOf, RDFS.domain, RDFS.range, RDFS.subPropertyOf):
        for subject in inserts.subjects(predicate, None):
            if isinstance(subject, URIRef):
                declared.add(subject)
    return declared


def _namespace_findings(
    inserts: RDFGraph,
    declared_namespaces: set[str],
    fact_namespaces: Sequence[str],
    standard_namespaces: tuple[str, ...],
) -> list[OntologyUnitFinding]:
    """Terms minted where no context ontology owns them.

    The reduce step attributes delta triples to writable catalog IRIs by
    namespace ownership and **silently drops** what it cannot attribute
    (``partition_triples_by_namespace``'s ``unattributed`` counter, DEBUG-only
    until now). A subject under a namespace the snapshot declares no terms in
    is that drop, predicted per-unit while a render can still fix it.
    """
    findings: list[OntologyUnitFinding] = []
    normalized_fact_namespaces = [ns for ns in fact_namespaces if ns]
    for subject in _new_subjects(inserts):
        text = str(subject)
        if text.startswith(_FORBIDDEN_NAMESPACES):
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.FOREIGN_NAMESPACE,
                    message=(
                        f"<{text}> is minted under the example.org placeholder "
                        "namespace; declare the term under one of the domain "
                        "ontology namespaces from the ontology chapter."
                    ),
                    subject=text,
                )
            )
            continue
        if any(text.startswith(ns) for ns in normalized_fact_namespaces):
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.FOREIGN_NAMESPACE,
                    message=(
                        f"<{text}> is an ontology term minted in the "
                        "facts/document namespace; facts namespaces hold "
                        "instances only — declare classes and properties under "
                        "a domain ontology namespace."
                    ),
                    subject=text,
                )
            )
            continue
        if not declared_namespaces:
            # Fresh-create path: the seed is empty, so there is no namespace
            # authority to check against.
            continue
        namespace = _namespace_of(text)
        vocabulary = well_known_vocabulary(text)
        if vocabulary is not None and not vocabulary.has_term(text[len(namespace) :]):
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.FOREIGN_NAMESPACE,
                    message=(
                        f"<{text}> is minted under <{namespace}>, the namespace "
                        f"of {vocabulary.title}, which has no such term. "
                        "Declare the term under one of the context ontology "
                        "namespaces."
                    ),
                    subject=text,
                )
            )
            continue
        if namespace in declared_namespaces or namespace.startswith(
            standard_namespaces
        ):
            continue
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.FOREIGN_NAMESPACE,
                message=(
                    f"<{text}> is minted under <{namespace}>, a namespace no "
                    "ontology in this unit's context declares terms under. "
                    "The catalog apply step will drop these triples as "
                    "unattributable — re-declare the term under one of the "
                    "context ontology namespaces."
                ),
                subject=text,
            )
        )
    return findings


def _degenerate_restriction_findings(
    inserts: RDFGraph,
) -> list[OntologyUnitFinding]:
    """Restriction blank nodes that constrain nothing."""
    findings: list[OntologyUnitFinding] = []
    seen: set[BNode] = set()
    for predicate in (RDFS.subClassOf, OWL.equivalentClass):
        for subject, obj in inserts.subject_objects(predicate):
            if not isinstance(obj, BNode) or obj in seen:
                continue
            seen.add(obj)
            meaningful = count_meaningful_restriction_predicates(inserts, obj)
            if meaningful >= MIN_MEANINGFUL_RESTRICTION_PREDICATES:
                continue
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.DEGENERATE_RESTRICTION,
                    message=(
                        f"<{subject}> is linked via {predicate.n3()} to a "
                        "restriction blank node that states nothing "
                        "(fewer than 2 meaningful predicates). Complete the "
                        "restriction with owl:onProperty plus a constraint "
                        "(owl:someValuesFrom, a cardinality, ...) or drop the "
                        "stub axiom."
                    ),
                    subject=str(subject),
                )
            )
    return findings


def _missing_label_findings(
    inserts: RDFGraph, snapshot_graph: RDFGraph | None
) -> list[OntologyUnitFinding]:
    """Newly declared classes/properties without a human-readable name.

    Single pass over the delta plus targeted lookups — deliberately not the
    ``validate_predicates`` loop, which rescans the whole graph per predicate.
    """
    findings: list[OntologyUnitFinding] = []
    for subject in sorted(_declared_class_or_property_subjects(inserts), key=str):
        labeled = any(
            True
            for predicate in _LABEL_PREDICATES
            for _ in inserts.objects(subject, predicate)
        )
        if not labeled and snapshot_graph is not None:
            labeled = any(
                True
                for predicate in _LABEL_PREDICATES
                for _ in snapshot_graph.objects(subject, predicate)
            )
        if labeled:
            continue
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.MISSING_LABEL,
                message=(
                    f"Newly declared term <{subject}> has no rdfs:label or "
                    "skos:prefLabel; add a concise English label."
                ),
                subject=str(subject),
            )
        )
    return findings


def _subclass_cycle_findings(
    inserts: RDFGraph, merged_graph: RDFGraph
) -> list[OntologyUnitFinding]:
    """Insert edges that close a subclass cycle through snapshot+delta."""
    findings: list[OntologyUnitFinding] = []
    for subject, obj in sorted(
        inserts.subject_objects(RDFS.subClassOf), key=lambda pair: tuple(map(str, pair))
    ):
        if not isinstance(subject, URIRef) or not isinstance(obj, URIRef):
            continue
        if subject == obj:
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.SUBCLASS_CYCLE,
                    message=f"<{subject}> is declared rdfs:subClassOf itself.",
                    subject=str(subject),
                )
            )
            continue
        if subject in superclass_closure(obj, merged_graph):
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.SUBCLASS_CYCLE,
                    message=(
                        f"Declaring <{subject}> rdfs:subClassOf <{obj}> closes "
                        "a subclass cycle — <{0}> is already an ancestor of "
                        "<{1}>. Point the edge at the intended superclass "
                        "instead.".format(subject, obj)
                    ),
                    subject=str(subject),
                    value=str(obj),
                )
            )
    return findings


def _role_confusion_findings(
    inserts: RDFGraph, snapshot_graph: RDFGraph | None
) -> list[OntologyUnitFinding]:
    """Catalog classes used as properties in the delta, and vice versa."""
    if snapshot_graph is None or not len(snapshot_graph):
        return []
    properties, classes = _catalog_term_roles(snapshot_graph)
    findings: list[OntologyUnitFinding] = []
    reported: set[str] = set()
    for _, predicate, _ in inserts:
        text = str(predicate)
        if text in reported or not isinstance(predicate, URIRef):
            continue
        if text in classes and text not in properties:
            reported.add(text)
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.ROLE_CONFUSION,
                    message=(
                        f"<{text}> is a class in the context ontology but is "
                        "used here as a predicate; use an appropriate property "
                        "or declare a new one."
                    ),
                    predicate=text,
                )
            )
    class_position_objects = [
        obj
        for predicate in (RDF.type, RDFS.subClassOf)
        for obj in inserts.objects(None, predicate)
    ]
    for obj in class_position_objects:
        text = str(obj)
        if text in reported or not isinstance(obj, URIRef):
            continue
        if text in properties and text not in classes:
            reported.add(text)
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.ROLE_CONFUSION,
                    message=(
                        f"<{text}> is a property in the context ontology but "
                        "is used here in class position (rdf:type / "
                        "rdfs:subClassOf object); use a class instead."
                    ),
                    value=text,
                )
            )
    return findings


def _label_collision_findings(
    inserts: RDFGraph, snapshot_graph: RDFGraph | None
) -> list[OntologyUnitFinding]:
    """New terms whose label duplicates an existing catalog surface form.

    Advisory: two distinct concepts may legitimately share a short label, so
    this asks for verification instead of blocking. The likely fix — reuse the
    existing term — is offered as a suggestion.
    """
    if snapshot_graph is None or not len(snapshot_graph):
        return []
    index = build_surface_index(snapshot_graph)
    if not index:
        return []
    findings: list[OntologyUnitFinding] = []
    for predicate in _LABEL_PREDICATES:
        for subject, value in sorted(
            inserts.subject_objects(predicate), key=lambda pair: tuple(map(str, pair))
        ):
            if not isinstance(subject, URIRef) or not isinstance(value, Literal):
                continue
            text = str(value).strip()
            owners = {iri for iri in index.get(text, set()) if iri != str(subject)}
            if not owners:
                continue
            findings.append(
                OntologyUnitFinding(
                    kind=OntologyUnitFindingKind.LABEL_COLLISION,
                    mandatory=False,
                    message=(
                        f"New term <{subject}> is labeled “{text}”, "
                        "which the context ontology already uses for another "
                        "term. If this is the same concept, reuse the existing "
                        "term instead of minting a duplicate."
                    ),
                    subject=str(subject),
                    value=text,
                    suggestions=sorted(owners),
                )
            )
    return findings


def _cardinality_contradiction_findings(
    inserts: RDFGraph, merged_graph: RDFGraph
) -> list[OntologyUnitFinding]:
    """Max-1 declarations contradicted by a min-cardinality >= 2."""
    max_one = harvest_max_one_predicates(merged_graph)
    if not max_one:
        return []
    insert_subjects = {s for s in inserts.subjects()}
    findings: list[OntologyUnitFinding] = []
    reported: set[URIRef] = set()
    for min_predicate in _MIN_CARDINALITY_PREDICATES:
        for restriction, cardinality in merged_graph.subject_objects(min_predicate):
            if not isinstance(cardinality, Literal):
                continue
            try:
                if int(cardinality) < 2:
                    continue
            except (TypeError, ValueError):
                continue
            for on_property in merged_graph.objects(restriction, OWL.onProperty):
                if (
                    not isinstance(on_property, URIRef)
                    or on_property not in max_one
                    or on_property in reported
                ):
                    continue
                # Only contradictions this unit participates in: a min>=2 vs
                # max-1 conflict entirely inside the snapshot is a pre-existing
                # catalog defect, not this unit's.
                if (
                    restriction not in insert_subjects
                    and on_property not in insert_subjects
                ):
                    continue
                reported.add(on_property)
                findings.append(
                    OntologyUnitFinding(
                        kind=OntologyUnitFindingKind.CARDINALITY_CONTRADICTION,
                        message=(
                            f"<{on_property}> is constrained to at most one "
                            f"value (functional / max-cardinality 1) but a "
                            f"restriction demands a minimum of "
                            f"{int(cardinality)}. Reconcile the two "
                            "declarations."
                        ),
                        predicate=str(on_property),
                    )
                )
    return findings


def _foreign_delete_findings(
    deletes: RDFGraph, snapshot_graph: RDFGraph | None, inserts: RDFGraph
) -> list[OntologyUnitFinding]:
    """Deletes of catalog content the unit does not redeclare.

    Ontology deletes propagate onto shared, versioned catalog terminals
    cross-document (``ontology_apply``), where facts deletes were unit-local.
    A unit may refine a term it is rewriting — its subject then reappears in
    the inserts — but a bare delete of snapshot content is destructive far
    beyond this unit and has no guard anywhere else.
    """
    if snapshot_graph is None or len(deletes) == 0:
        return []
    insert_subjects = {str(s) for s in inserts.subjects() if isinstance(s, URIRef)}
    findings: list[OntologyUnitFinding] = []
    reported: set[str] = set()
    for subject in deletes.subjects():
        if not isinstance(subject, URIRef):
            continue
        text = str(subject)
        if text in reported or text in insert_subjects:
            continue
        if (subject, None, None) not in snapshot_graph:
            continue
        reported.add(text)
        removed = sum(1 for _ in deletes.triples((subject, None, None)))
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.FOREIGN_DELETE,
                message=(
                    f"This update deletes {removed} statement(s) about "
                    f"<{text}>, a term the context ontology declares, without "
                    "redeclaring it. Catalog deletes propagate to every "
                    "document sharing the ontology — restate the corrected "
                    "term, or drop the delete."
                ),
                subject=text,
            )
        )
    return findings


def _minted_classes(inserts: RDFGraph, snapshot_graph: RDFGraph | None) -> list[URIRef]:
    """Classes the delta declares that the snapshot does not already hold."""
    minted = []
    for iri in sorted(minted_class_iris(inserts)):
        subject = URIRef(iri)
        if snapshot_graph is not None and (subject, None, None) in snapshot_graph:
            continue
        minted.append(subject)
    return minted


def _hierarchy_depth_findings(
    inserts: RDFGraph,
    snapshot_graph: RDFGraph | None,
    merged_graph: RDFGraph,
    max_class_depth: int,
) -> list[OntologyUnitFinding]:
    """Minted classes deeper below their root than the configured bound."""
    findings: list[OntologyUnitFinding] = []
    memo: dict = {}
    for subject in _minted_classes(inserts, snapshot_graph):
        depth, chain = named_class_depth(subject, merged_graph, memo)
        if depth <= max_class_depth:
            continue
        rendered = " ⊂ ".join(f"<{node}>" for node in chain)
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.HIERARCHY_TOO_DEEP,
                message=(
                    f"<{subject}> sits {depth} named classes below its root "
                    f"({rendered}); the bound is {max_class_depth}. Attach it "
                    "to a higher class, or collapse the intermediate classes "
                    "another document in the domain would not reuse."
                ),
                subject=str(subject),
                value=str(depth),
            )
        )
    return findings


def _value_set_sibling_findings(
    inserts: RDFGraph, snapshot_graph: RDFGraph | None, merged_graph: RDFGraph
) -> list[OntologyUnitFinding]:
    """Minted sibling classes one word apart that a property value would do."""
    siblings: dict[str, list[tuple[str, str]]] = {}
    for child in _minted_classes(inserts, snapshot_graph):
        if has_own_structure(child, merged_graph):
            continue
        label = first_label(child, inserts, merged_graph)
        if label is None:
            continue
        for parent in inserts.objects(child, RDFS.subClassOf):
            if isinstance(parent, URIRef):
                siblings.setdefault(str(parent), []).append((str(child), label))
    findings: list[OntologyUnitFinding] = []
    for family in value_set_families(siblings):
        labels = ", ".join(f"“{label}”" for _, label in family.members)
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.VALUE_SET_SIBLINGS,
                message=(
                    f"{len(family.members)} new subclasses of <{family.parent}> "
                    f"differ in one word only ({labels}) and add nothing of "
                    "their own. That is a value set, not a hierarchy: declare "
                    f"one property on <{family.parent}> whose values are "
                    "individuals (or SKOS concepts), and drop the subclasses."
                ),
                subject=family.parent,
                value=family.template,
                suggestions=[iri for iri, _ in family.members],
            )
        )
    return findings


def _specific_label_leaf_findings(
    inserts: RDFGraph, snapshot_graph: RDFGraph | None, merged_graph: RDFGraph
) -> list[OntologyUnitFinding]:
    """Minted leaves whose label names one document's case. Advisory."""
    findings: list[OntologyUnitFinding] = []
    for subject in _minted_classes(inserts, snapshot_graph):
        if any(True for _ in merged_graph.subjects(RDFS.subClassOf, subject)):
            continue
        if any(
            True
            for predicate in (RDFS.domain, RDFS.range)
            for _ in merged_graph.subjects(predicate, subject)
        ):
            continue
        label = first_label(subject, inserts, merged_graph)
        if label is None:
            continue
        reason = specific_label_reason(label)
        if reason is None:
            continue
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.SPECIFIC_LABEL_LEAF,
                mandatory=False,
                message=(
                    f"<{subject}> (“{label}”) is a leaf no property uses, and "
                    "its label names a particular case. Mint a class at the "
                    "level another document in the domain would reuse; a "
                    "particular case is an individual or a property value."
                ),
                subject=str(subject),
                value=label,
                facet=reason,
            )
        )
    return findings


def _catalog_duplicate_findings(
    inserts: RDFGraph,
    snapshot_graph: RDFGraph | None,
    catalog_index: CatalogSurfaceIndex | None,
) -> list[OntologyUnitFinding]:
    """Minted terms that name a catalog term the unit's context did not show.

    Matched against the whole catalog, case-insensitively, by label, preferred
    label or notation, then by local name. An exact label match against the
    snapshot is left to the label-collision rule, which already reports it.
    """
    if catalog_index is None or catalog_index.is_empty():
        return []
    exact_snapshot = build_surface_index(snapshot_graph) if snapshot_graph else {}
    roles = minted_roles(inserts)
    findings: list[OntologyUnitFinding] = []
    for subject in _new_subjects(inserts):
        if snapshot_graph is not None and (subject, None, None) in snapshot_graph:
            continue
        if any(
            exact_snapshot.get(surface) for surface in surfaces_of(subject, inserts)
        ):
            continue
        match = match_catalog_term(
            subject, inserts, catalog_index, roles.get(subject, "unknown")
        )
        if match is None:
            continue
        name = "local name" if match.key == "local_name" else "name"
        findings.append(
            OntologyUnitFinding(
                kind=OntologyUnitFindingKind.CATALOG_DUPLICATE,
                message=(
                    f"New term <{match.minted_iri}> has the same {name} "
                    f"(“{match.surface}”) as the catalog term "
                    f"<{match.catalog_iri}>, which the context did not show. "
                    f"Reuse <{match.catalog_iri}>: state the new statements "
                    f"on it and drop <{match.minted_iri}>."
                ),
                subject=match.minted_iri,
                value=match.surface,
                facet=match.key,
                suggestions=[match.catalog_iri],
            )
        )
    return findings


def collect_ontology_unit_findings(
    *,
    inserts: RDFGraph,
    deletes: RDFGraph,
    snapshot_graph: RDFGraph | None,
    merged_graph: RDFGraph | None = None,
    fact_namespaces: Sequence[str] = (),
    policy: ValidationPolicy | None = None,
    max_class_depth: int = 6,
    catalog_index: CatalogSurfaceIndex | None = None,
) -> list[OntologyUnitFinding]:
    """Assemble all deterministic findings for one unit's ontology delta.

    Args:
        inserts: Net new triples this unit adds (``build_delta().inserts``).
        deletes: Snapshot triples this unit removes.
        snapshot_graph: The prompt ontology context; ``None``/empty means the
            fresh-create path, where every catalog-relative check is skipped.
        merged_graph: ``snapshot + delta`` when the caller already has it
            (``working_graph`` after updates apply) — passing it avoids a
            snapshot copy. Built here when omitted.
        fact_namespaces: Facts/document namespaces; ontology terms minted
            there are mandatory findings.
        policy: Deployment namespace exemptions; ``None`` (tests) uses the
            built-in standard namespaces only.
        max_class_depth: Deepest named-superclass chain a minted class may
            sit on.
        catalog_index: Names of every catalog term, for the duplicate check;
            ``None`` skips it.

    Returns:
        Findings, mandatory first is *not* guaranteed — order follows the
        check sequence; callers filter on ``mandatory``.
    """
    active_policy = policy or ValidationPolicy()
    if merged_graph is None:
        merged_graph = RDFGraph()
        if snapshot_graph is not None:
            for triple in snapshot_graph:
                merged_graph.add(triple)
        for triple in inserts:
            merged_graph.add(triple)

    declared = collect_declared_namespaces(snapshot_graph)
    findings: list[OntologyUnitFinding] = [
        *_namespace_findings(
            inserts, declared, fact_namespaces, active_policy.standard_namespaces()
        ),
        *_degenerate_restriction_findings(inserts),
        *_missing_label_findings(inserts, snapshot_graph),
        *_subclass_cycle_findings(inserts, merged_graph),
        *_role_confusion_findings(inserts, snapshot_graph),
        *_cardinality_contradiction_findings(inserts, merged_graph),
        *_foreign_delete_findings(deletes, snapshot_graph, inserts),
        *_label_collision_findings(inserts, snapshot_graph),
        *_hierarchy_depth_findings(
            inserts, snapshot_graph, merged_graph, max_class_depth
        ),
        *_value_set_sibling_findings(inserts, snapshot_graph, merged_graph),
        *_specific_label_leaf_findings(inserts, snapshot_graph, merged_graph),
        *_catalog_duplicate_findings(inserts, snapshot_graph, catalog_index),
    ]
    return findings


def count_fixes_targeting_snapshot(
    fixes: Sequence[TripleFix],
    snapshot_graph: RDFGraph | None,
    insert_subjects: set[str],
) -> int:
    """Critic fixes aimed at catalog content this unit's delta never touched.

    The ontology critic is shown ``snapshot + delta`` and can reject a unit
    for pre-existing catalog defects the renderer cannot own. This counts the
    proposed fixes whose ``incorrect_value`` names a snapshot-declared subject
    absent from the unit's inserts — by full-IRI or prefixed-name substring
    match, so it is a lower bound, recorded as telemetry rather than used for
    control flow.
    """
    if snapshot_graph is None or not fixes:
        return 0
    prefix_map = [
        (prefix, str(namespace))
        for prefix, namespace in snapshot_graph.namespaces()
        if prefix
    ]
    snapshot_only: set[str] = set()
    qnames: set[str] = set()
    for subject in snapshot_graph.subjects():
        if not isinstance(subject, URIRef):
            continue
        text = str(subject)
        if text in insert_subjects or text in snapshot_only:
            continue
        snapshot_only.add(text)
        for prefix, namespace in prefix_map:
            if text.startswith(namespace) and len(text) > len(namespace):
                qnames.add(f"{prefix}:{text[len(namespace) :]}")
    count = 0
    for fix in fixes:
        haystack = fix.incorrect_value or ""
        if not haystack:
            continue
        if any(iri in haystack for iri in snapshot_only) or any(
            qname in haystack for qname in qnames
        ):
            count += 1
    return count
