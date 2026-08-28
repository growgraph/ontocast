"""LLM-free parse-time repairs on rendered facts graphs.

Every rewrite here either retypes a literal the schema grounds, resolves an
unambiguous near-miss or code, or collapses a degenerate encoding — no repair
invents a value.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher

from rdflib import RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.term import Node

from ontocast.onto.model import (
    FactsGateRepairKind,
    FactsUnitFinding,
    FactsUnitFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import (
    RDFGraph,
)
from ontocast.tool.facts_validation.terms import (
    _alias_candidates,
    _declared_domains,
    _local_name,
    _name_tokens,
    _namespace_of,
    _resolve_type_literal,
    _superclass_closure,
    _vocabulary_role_subset,
    collect_catalog_terms,
    collect_declared_namespaces,
    expand_vocabulary_terms,
    resolve_unique_surface,
)
from ontocast.util.numeric_inventory import canonical_number

logger = logging.getLogger(__name__)

_NUMERIC_RANGE_DATATYPES = {
    XSD.decimal,
    XSD.integer,
    XSD.float,
    XSD.double,
    XSD.nonNegativeInteger,
    XSD.positiveInteger,
}

# Datatypes whose rdflib value parser actually rejects a wrong lexical form, so
# ``_literal_parses_as`` is evidence. Deliberately excluded, each measured:
#   xsd:string / xsd:anyURI -- every lexical form parses;
#   xsd:boolean -- Literal("2019", datatype=xsd:boolean).value is False, not None;
#   xsd:time    -- "2019" parses as 20:19, so a time range would mangle years.
# Admitting any of those lets one sloppy range declaration rewrite unrelated
# literals on that predicate, including correctly typed ones.
_PARSE_CHECKED_RANGE_DATATYPES = {XSD.date, XSD.dateTime, XSD.duration}

# The gregorian datatypes have no rdflib value parser at all -- ``.value`` is
# always None -- so they need explicit lexical validation or they could never be
# retyped, which is precisely the "xsd:gYear range receiving a string" case this
# widening exists for. Patterns are the XSD lexical spaces, timezone included.
_GREGORIAN_RANGE_PATTERNS = {
    XSD.gYear: re.compile(r"^-?\d{4,}(Z|[+-]\d{2}:\d{2})?$"),
    XSD.gYearMonth: re.compile(r"^-?\d{4,}-\d{2}(Z|[+-]\d{2}:\d{2})?$"),
}

# Ranges a declared ``rdfs:range`` may retype a literal *to*. An allowlist, not
# "any XSD datatype" -- see the two sets above for why.
_RETYPABLE_RANGE_DATATYPES = (
    _NUMERIC_RANGE_DATATYPES
    | _PARSE_CHECKED_RANGE_DATATYPES
    | set(_GREGORIAN_RANGE_PATTERNS)
)

# Meta-vocabularies: the RDF/OWL substrate plus the annotation and provenance
# terms every facts graph carries regardless of catalog. Exempting these from
# UNKNOWN_TERM keeps the signal readable -- flagging rdfs:label would bury it.
#
# Domain vocabularies do NOT belong here. SOSA/SSN (sensor observations),
# CSVW (tabular metadata), FOAF and schema.org (people, organizations,
# creative works) model a subject area; a catalog that does not declare them
# should hear about it. They are exempted by configuration
# (FACTS_ADDITIONAL_STANDARD_NAMESPACES) when a deployment wants them, not by


def normalize_literals_against_schema(
    graph: RDFGraph, ontology_graph: RDFGraph | None
) -> int:
    """Retype literals whose predicate declares a compatible ``rdfs:range``.

    Fixes the ``qudt:numericValue 230`` vs ``"230"^^xsd:decimal`` drift at parse
    time, and the same drift for the date-like datatypes: when the schema
    declares a range in :data:`_RETYPABLE_RANGE_DATATYPES` and the lexical form
    parses as that datatype, the literal is rewritten with it.

    A literal is only retyped from an untyped, ``xsd:string``, or numeric source
    -- a string range must never be able to clobber a correctly typed value --
    and language-tagged literals are left alone, since they are
    ``rdf:langString`` and retyping would discard the tag.

    Returns:
        Number of retyped literals.
    """
    if ontology_graph is None:
        return 0
    declared_ranges: dict[URIRef, URIRef] = {}
    for predicate, range_iri in ontology_graph.subject_objects(RDFS.range):
        if (
            isinstance(predicate, URIRef)
            and isinstance(range_iri, URIRef)
            and range_iri in _RETYPABLE_RANGE_DATATYPES
        ):
            declared_ranges[predicate] = range_iri

    if not declared_ranges:
        return 0

    replacements: list[tuple[tuple, tuple]] = []
    for subject, predicate, obj in graph:
        if not isinstance(obj, Literal) or not isinstance(predicate, URIRef):
            continue
        target_datatype = declared_ranges.get(predicate)
        if target_datatype is None or obj.datatype == target_datatype:
            continue
        if obj.language is not None:
            continue
        numeric_target = target_datatype in _NUMERIC_RANGE_DATATYPES
        source_admissible = obj.datatype is None or obj.datatype == XSD.string
        if numeric_target:
            # Keep the pre-existing numeric->numeric promotion (integer to
            # decimal, say), which a source-side "untyped or string" rule alone
            # would silently drop.
            source_admissible = (
                source_admissible or obj.datatype in _NUMERIC_RANGE_DATATYPES
            )
        if not source_admissible:
            continue
        lexical = str(obj).strip()
        gregorian = _GREGORIAN_RANGE_PATTERNS.get(target_datatype)
        if numeric_target:
            parses = canonical_number(lexical) is not None
        elif gregorian is not None:
            parses = gregorian.match(lexical) is not None
        else:
            parses = _literal_parses_as(lexical, target_datatype)
        if not parses:
            continue
        replacements.append(
            (
                (subject, predicate, obj),
                (subject, predicate, Literal(lexical, datatype=target_datatype)),
            )
        )

    for old, new in replacements:
        graph.remove(old)
        graph.add(new)
    return len(replacements)


def repair_literal_type_objects(
    graph: RDFGraph,
) -> tuple[int, list[FactsUnitFinding], list[GraphRepairRecord]]:
    """Coerce literal ``rdf:type`` objects into IRIs.

    The renderer sometimes emits ``a "prefix:Class"^^xsd:string`` instead of
    ``a prefix:Class`` (JSON-LD bare-string type values parse the same way).
    A literal-typed node is invisible to SPARQL class queries, reasoning, and
    the aggregator's URI minting/entity matching, all of which guard on
    ``isinstance(obj, URIRef)``. Absolute IRIs and compact IRIs bound in the
    graph are rewritten deterministically; unresolvable forms become MANDATORY
    findings.

    Returns:
        Tuple of (number of rewritten triples, unresolved findings,
        applied-repair records).
    """
    prefix_map = {
        prefix: str(namespace) for prefix, namespace in graph.namespaces() if prefix
    }
    rewritten = 0
    findings: list[FactsUnitFinding] = []
    applied: list[GraphRepairRecord] = []
    for subject, predicate, obj in list(graph.triples((None, RDF.type, None))):
        if not isinstance(obj, Literal):
            continue
        lexical = str(obj).strip()
        resolved = _resolve_type_literal(lexical, prefix_map)
        if resolved is not None:
            graph.remove((subject, predicate, obj))
            graph.add((subject, RDF.type, URIRef(resolved)))
            rewritten += 1
            applied.append(
                GraphRepairRecord(
                    kind=FactsUnitFindingKind.LITERAL_TYPE_OBJECT,
                    source=lexical,
                    target=resolved,
                )
            )
            logger.info(
                "Repaired literal rdf:type object %r -> <%s>", lexical, resolved
            )
            continue
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.LITERAL_TYPE_OBJECT,
                message=(
                    f"rdf:type object '{lexical}' is a string literal, not an "
                    "IRI; assert the type as a catalog class IRI "
                    "(`a prefix:Class`), never as a quoted string."
                ),
                subject=str(subject),
                value=lexical,
            )
        )
    return rewritten, findings, applied


def repair_property_aliases(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    *,
    min_ratio: float = 0.85,
    exempt_terms: set[str] | None = None,
) -> tuple[int, list[FactsUnitFinding], list[GraphRepairRecord]]:
    """Rewrite near-miss predicates in catalog namespaces; report ambiguity.

    A predicate whose namespace belongs to the ontology context but which is
    not itself a catalog term is a near-miss (``qqval:lowerBound`` for
    ``qqval:hasLowerBound``). When exactly one candidate scores above
    ``min_ratio`` (token containment counts as 1.0) the rewrite is applied
    deterministically; otherwise a mandatory finding carries the top
    suggestions.

    Only namespaces the catalog *declares* terms in are eligible (see
    :func:`collect_declared_namespaces`); ``exempt_terms`` (expanded fallback
    vocabulary) are never treated as near-misses.

    Returns:
        Tuple of (number of rewritten triples, unresolved findings,
        applied-repair records).
    """
    catalog_terms = collect_catalog_terms(ontology_graph)
    if not catalog_terms:
        return 0, [], []
    declared_namespaces = collect_declared_namespaces(ontology_graph)
    exempt = exempt_terms or set()

    findings: list[FactsUnitFinding] = []
    applied: list[GraphRepairRecord] = []
    rewritten = 0
    predicates = {
        predicate
        for predicate in graph.predicates()
        if isinstance(predicate, URIRef)
        and str(predicate) not in catalog_terms
        and str(predicate) not in exempt
        and _namespace_of(str(predicate)) in declared_namespaces
    }
    for alias in sorted(predicates, key=str):
        candidates = _alias_candidates(
            alias, graph, catalog_terms, ontology_graph=ontology_graph
        )
        strong = [
            candidate
            for candidate in candidates
            if _name_tokens(_local_name(str(alias)))
            and (
                _name_tokens(_local_name(str(alias)))
                <= _name_tokens(_local_name(candidate))
                or _name_tokens(_local_name(candidate))
                <= _name_tokens(_local_name(str(alias)))
                or SequenceMatcher(
                    None,
                    _local_name(str(alias)).lower(),
                    _local_name(candidate).lower(),
                ).ratio()
                >= min_ratio
            )
        ]
        if len(strong) == 1:
            replacement = URIRef(strong[0])
            alias_triples = 0
            for subject, predicate, obj in list(graph.triples((None, alias, None))):
                graph.remove((subject, predicate, obj))
                graph.add((subject, replacement, obj))
                rewritten += 1
                alias_triples += 1
            applied.append(
                GraphRepairRecord(
                    kind=FactsUnitFindingKind.PROPERTY_ALIAS,
                    source=str(alias),
                    target=str(replacement),
                    triple_count=alias_triples,
                )
            )
            logger.info("Repaired property alias %s -> %s", alias, replacement)
            continue
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.PROPERTY_ALIAS,
                message=(
                    f"Predicate <{alias}> is not defined in its ontology; "
                    "replace it with the correct catalog property."
                ),
                predicate=str(alias),
                suggestions=candidates,
            )
        )
    return rewritten, findings, applied


def _collect_linking_evidence(
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    closure: Callable[[URIRef], set[URIRef]],
) -> list[tuple[set[URIRef], set[URIRef], URIRef]]:
    """One scan of the graph's IRI-object links, with type closures resolved.

    Collected once per :func:`resolve_code_literals` call so the per-literal
    fallback in :func:`_observed_linking_predicates` filters in memory instead
    of re-walking the graph and recomputing closures for every code literal.
    """
    evidence: list[tuple[set[URIRef], set[URIRef], URIRef]] = []
    subject_closures: dict[Node, set[URIRef]] = {}
    for subject, predicate, obj in graph:
        if not isinstance(predicate, URIRef) or not isinstance(obj, URIRef):
            continue
        if predicate == RDF.type:
            continue
        obj_types: set[URIRef] = set()
        for type_iri in ontology_graph.objects(obj, RDF.type):
            if isinstance(type_iri, URIRef):
                obj_types |= closure(type_iri)
        if not obj_types:
            continue
        if subject not in subject_closures:
            own_types: set[URIRef] = set()
            for type_iri in graph.objects(subject, RDF.type):
                if isinstance(type_iri, URIRef):
                    own_types |= closure(type_iri)
            subject_closures[subject] = own_types
        evidence.append((subject_closures[subject], obj_types, predicate))
    return evidence


def _observed_linking_predicates(
    evidence: Sequence[tuple[set[URIRef], set[URIRef], URIRef]],
    subject_types: set[URIRef],
    object_types: set[URIRef],
) -> list[URIRef]:
    """Predicates the graph already uses to link these two kinds of node.

    Evidence, not schema: used only where the schema declares no range, and
    only when the evidence is unambiguous (exactly one such predicate).
    """
    observed = {
        predicate
        for own_types, obj_types, predicate in evidence
        if obj_types & object_types and (not subject_types or own_types & subject_types)
    }
    return sorted(observed, key=str)


def resolve_code_literals(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    code_predicates: Sequence[str] = (),
) -> tuple[int, list[GraphRepairRecord]]:
    """Link nodes to the catalog individual whose code they already carry.

    A renderer that reads ``4-15 days`` often annotates the value node with the
    code it saw — ``qudt:ucumCode "d"`` — instead of the object property that
    points at the individual — ``qudt:unit unit:DAY``. The graph is well-formed,
    so no range check fires, but every query reading the object property gets
    an unbound result. The code came from the text and the individual is in the
    catalog, so the link is recoverable without asking the model again.

    Fully schema-driven, no vocabulary compiled in: the connecting property is
    whichever object property the ontology context declares with a range the
    resolved individual is typed as, and a domain the subject satisfies. If the
    schema offers several such properties, or none, nothing is added.

    Args:
        graph: Rendered facts graph, repaired in place.
        ontology_graph: Merged ontology context, read-only.
        code_predicates: Predicates carrying machine-resolvable codes.

    Returns:
        Tuple of (number of added triples, applied-repair records).
    """
    if ontology_graph is None or not code_predicates:
        return 0, []
    code_terms = [URIRef(predicate) for predicate in code_predicates]
    # Only the code predicates themselves resolve here: a label match is a
    # different, much weaker signal and belongs to the shapes-driven pass.
    code_index: dict[str, set[str]] = {}
    for predicate in code_terms:
        for subject, value in ontology_graph.subject_objects(predicate):
            if isinstance(subject, URIRef) and isinstance(value, Literal):
                text = str(value).strip()
                if text:
                    code_index.setdefault(text, set()).add(str(subject))
    if not code_index:
        return 0, []

    domains = _declared_domains(ontology_graph)
    ranges: dict[URIRef, set[URIRef]] = {}
    for predicate, _, range_iri in ontology_graph.triples((None, RDFS.range, None)):
        if isinstance(predicate, URIRef) and isinstance(range_iri, URIRef):
            ranges.setdefault(predicate, set()).add(range_iri)

    # Superclass closures repeat heavily across literals; memoise per call.
    closures: dict[URIRef, set[URIRef]] = {}

    def closure(class_iri: URIRef) -> set[URIRef]:
        if class_iri not in closures:
            closures[class_iri] = _superclass_closure(class_iri, ontology_graph)
        return closures[class_iri]

    # The usage-evidence scan is a full graph walk; build it lazily, once,
    # only if some literal actually needs the no-declared-range fallback.
    linking_evidence: list[tuple[set[URIRef], set[URIRef], URIRef]] | None = None

    added = 0
    records: list[GraphRepairRecord] = []
    for code_predicate in code_terms:
        for subject, value in list(graph.subject_objects(code_predicate)):
            if not isinstance(subject, URIRef) or not isinstance(value, Literal):
                continue
            resolved = resolve_unique_surface(code_index, str(value))
            if resolved is None:
                continue
            resolved_types: set[URIRef] = set()
            for type_iri in ontology_graph.objects(resolved, RDF.type):
                if isinstance(type_iri, URIRef):
                    resolved_types |= closure(type_iri)
            if not resolved_types:
                continue
            subject_types: set[URIRef] = set()
            for type_iri in graph.objects(subject, RDF.type):
                if isinstance(type_iri, URIRef):
                    subject_types |= closure(type_iri)

            candidates = [
                predicate
                for predicate, range_set in ranges.items()
                if range_set & resolved_types
                and (
                    predicate not in domains
                    or not domains[predicate]
                    or domains[predicate] & subject_types
                )
            ]
            if not candidates:
                # Vendored vocabulary projections often declare individuals and
                # their codes but no rdfs:range (the shipped QUDT unit subset is
                # one). Fall back to how the graph already links this kind of
                # subject to this kind of individual -- the same
                # induce-from-usage move the functional-predicate harvest makes.
                if linking_evidence is None:
                    linking_evidence = _collect_linking_evidence(
                        graph, ontology_graph, closure
                    )
                candidates = _observed_linking_predicates(
                    linking_evidence, subject_types, resolved_types
                )
            # Already linked, ambiguous, or unsupported by the schema.
            candidates = [
                predicate
                for predicate in candidates
                if (subject, predicate, None) not in graph
            ]
            if len(candidates) != 1:
                continue
            predicate = candidates[0]
            graph.add((subject, predicate, resolved))
            added += 1
            records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.CODE_RESOLVED,
                    source=f"{code_predicate} {value.n3()}",
                    target=f"{predicate} {resolved}",
                )
            )
            logger.info(
                "Resolved code %s on <%s> to <%s %s>",
                value.n3(),
                subject,
                predicate,
                resolved,
            )
    return added, records


def promote_degenerate_bounds_from_vocabulary(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    vocabulary: dict[str, str] | None,
) -> int:
    """Run :func:`promote_degenerate_bounds` with properties from configuration.

    Active only when the quantity vocabulary names all three roles —
    ``numeric_value``, ``lower_bound``, ``upper_bound`` (roles containing
    ``inclusive`` supply the optional bound flags). The default vocabulary
    carries no bound roles, so this is off unless a deployment configures its
    range encoding.
    """
    vocabulary = vocabulary or {}
    numeric_terms = expand_vocabulary_terms(
        {"numeric_value": vocabulary.get("numeric_value", "")}, graph, ontology_graph
    )
    lower_terms = expand_vocabulary_terms(
        {"lower_bound": vocabulary.get("lower_bound", "")}, graph, ontology_graph
    )
    upper_terms = expand_vocabulary_terms(
        {"upper_bound": vocabulary.get("upper_bound", "")}, graph, ontology_graph
    )
    inclusive_terms = expand_vocabulary_terms(
        _vocabulary_role_subset(vocabulary, "inclusive"), graph, ontology_graph
    )
    if len(numeric_terms) != 1 or len(lower_terms) != 1 or len(upper_terms) != 1:
        return 0
    return promote_degenerate_bounds(
        graph,
        numeric_value_property=next(iter(numeric_terms)),
        lower_bound_property=next(iter(lower_terms)),
        upper_bound_property=next(iter(upper_terms)),
        inclusive_flag_properties=sorted(inclusive_terms),
    )


def promote_degenerate_bounds(
    graph: RDFGraph,
    *,
    numeric_value_property: str,
    lower_bound_property: str,
    upper_bound_property: str,
    inclusive_flag_properties: Sequence[str] = (),
) -> int:
    """Rewrite equal lower/upper bounds into a single scalar value, in place.

    A node whose lower and upper bounds carry the same canonical numeric value
    encodes an exact scalar as a fake range. The rewrite fires only when the
    encoding is unambiguous: exactly one literal per bound property, equal
    canonical values, no existing scalar on the node, and no exclusive-bound
    flag (an exclusive equal bound denotes an empty interval — malformed, and
    left for findings). Property IRIs are injected by the caller from
    configuration; nothing is hardcoded.

    Returns:
        Number of nodes rewritten.
    """
    lower_ref = URIRef(lower_bound_property)
    upper_ref = URIRef(upper_bound_property)
    value_ref = URIRef(numeric_value_property)
    flag_refs = [URIRef(term) for term in inclusive_flag_properties]
    promoted = 0
    for subject in sorted(set(graph.subjects(lower_ref, None)), key=str):
        lowers = [obj for obj in graph.objects(subject, lower_ref)]
        uppers = [obj for obj in graph.objects(subject, upper_ref)]
        if len(lowers) != 1 or len(uppers) != 1:
            continue
        if not isinstance(lowers[0], Literal) or not isinstance(uppers[0], Literal):
            continue
        if (subject, value_ref, None) in graph:
            continue
        low = canonical_number(str(lowers[0]).strip())
        high = canonical_number(str(uppers[0]).strip())
        if low is None or low != high:
            continue
        if any(
            str(flag_value).strip().lower() == "false"
            for flag_ref in flag_refs
            for flag_value in graph.objects(subject, flag_ref)
        ):
            continue
        graph.remove((subject, lower_ref, lowers[0]))
        graph.remove((subject, upper_ref, uppers[0]))
        for flag_ref in flag_refs:
            for flag_value in list(graph.objects(subject, flag_ref)):
                graph.remove((subject, flag_ref, flag_value))
        graph.add((subject, value_ref, Literal(low, datatype=XSD.decimal)))
        promoted += 1
    if promoted:
        logger.info(
            "Promoted %d degenerate bound pair(s) to <%s>",
            promoted,
            numeric_value_property,
        )
    return promoted


def _literal_parses_as(lexical: str, datatype: URIRef) -> bool:
    """True when ``lexical`` is a well-formed literal of ``datatype``."""
    return Literal(lexical, datatype=datatype).value is not None
