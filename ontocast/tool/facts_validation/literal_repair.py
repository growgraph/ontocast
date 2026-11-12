"""LLM-free parse-time repairs on rendered facts graphs.

Every rewrite here either retypes a literal the schema grounds, resolves an
unambiguous near-miss or code, or collapses a degenerate encoding — no repair
invents a value.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from difflib import SequenceMatcher

from rdflib import RDF, RDFS, BNode, Literal, URIRef
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
    RejectedLiteralTriple,
)
from ontocast.onto.well_known import (
    is_well_known_term,
    well_known_datatype_ranges,
    well_known_role,
)
from ontocast.onto.xsd_lexical import TEMPORAL_DATATYPES, is_valid_temporal_lexical
from ontocast.tool.facts_validation.date_values import parse_date_label
from ontocast.tool.facts_validation.terms import (
    _COMPACT_IRI,
    _VACUOUS_TYPES,
    CatalogRoles,
    _alias_candidates,
    _catalog_term_roles,
    _declared_domains,
    _local_name,
    _name_tokens,
    _namespace_of,
    _resolve_type_literal,
    _vocabulary_role_subset,
    collect_catalog_terms,
    collect_declared_namespaces,
    declared_class_ranges,
    expand_vocabulary_terms,
    resolve_unique_surface,
    superclass_closure,
)
from ontocast.tool.llm import record_active_count
from ontocast.tool.validate import _predicate_expects_iri_object
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

# The date, time and duration datatypes, decided by their XSD lexical spaces
# (:func:`~ontocast.onto.xsd_lexical.is_valid_temporal_lexical`), not by
# rdflib's lax ISO parsers. A text that fails is no value of the type, so these
# are also the ranges text is set aside on; a number in prose is left to the
# numeric checks. Other datatypes are not retyped: every lexical form is an
# ``xsd:string`` or ``xsd:anyURI``, and rdflib reads ``"2019"`` as a boolean,
# so admitting them lets one sloppy range declaration rewrite unrelated literals.
_TEMPORAL_RANGE_DATATYPES = set(TEMPORAL_DATATYPES)

# Ranges a declared ``rdfs:range`` may retype a literal *to*.
_RETYPABLE_RANGE_DATATYPES = _NUMERIC_RANGE_DATATYPES | _TEMPORAL_RANGE_DATATYPES


def literal_ranges(
    ontology_graph: RDFGraph | None, catalog_roles: CatalogRoles | None = None
) -> dict[URIRef, URIRef]:
    """Each predicate with one declared retypable datatype range, with it.

    Read from the unit's ``ontology_graph``, the whole catalog
    (``catalog_roles``) and the well-known vocabularies, so a property the
    unit's snapshot did not retrieve is still known. A predicate declared with
    two different datatypes anywhere is left out: no single value fits both.
    Ranges inherited through ``rdfs:subPropertyOf`` are not followed.
    """
    declared: dict[str, set[str]] = {}
    sources: list[tuple[str, str]] = list(well_known_datatype_ranges().items())
    if catalog_roles is not None:
        sources.extend(
            (predicate, datatype)
            for predicate, datatypes in catalog_roles.datatype_ranges.items()
            for datatype in datatypes
        )
    if ontology_graph is not None:
        sources.extend(
            (str(predicate), str(range_iri))
            for predicate, range_iri in ontology_graph.subject_objects(RDFS.range)
            if isinstance(predicate, URIRef)
            and isinstance(range_iri, URIRef)
            and str(range_iri).startswith(str(XSD))
        )
    for predicate, datatype in sources:
        declared.setdefault(predicate, set()).add(datatype)
    return {
        URIRef(predicate): URIRef(next(iter(datatypes)))
        for predicate, datatypes in declared.items()
        if len(datatypes) == 1
        and URIRef(next(iter(datatypes))) in _RETYPABLE_RANGE_DATATYPES
    }


# A date with no zone: the one form widened to midnight on a date-time range.
_ZONELESS_DATE = re.compile(r"-?[0-9]{4,}-[0-9]{2}-[0-9]{2}")


def _typed_value(lexical: str, datatype: URIRef) -> Literal | None:
    """``lexical`` as a well-formed literal of a retypable ``datatype``, or ``None``.

    A date, time or duration must be a value of ``datatype`` in its XSD lexical
    space as written: rdflib's readers are laxer (``"2019"`` reads as the
    ``xsd:time`` 20:19). One widening matches what parsing does to a typed
    literal: an ISO date with no zone on an ``xsd:dateTime`` range is that day
    at midnight.
    A calendar text (``"29 June 2021"``) is first read as the ISO value it
    names exactly; ``"June 2021"`` names no day, so it is no date.
    """
    if datatype in _NUMERIC_RANGE_DATATYPES:
        if canonical_number(lexical) is None:
            return None
        return Literal(lexical, datatype=datatype)
    candidates = [lexical]
    named = parse_date_label(lexical, allow_year=True)
    if named is not None:
        candidates.append(str(named))
    for candidate in candidates:
        if datatype == XSD.dateTime and _ZONELESS_DATE.fullmatch(candidate):
            candidate = f"{candidate}T00:00:00"
        if is_valid_temporal_lexical(candidate, datatype):
            # As written: rdflib's normal form can move a zone into the hour.
            with _quiet_rdflib_casts():
                return Literal(candidate, datatype=datatype, normalize=False)
    return None


def _fits_range(lexical: str, datatype: URIRef) -> bool:
    """Whether ``lexical`` is a value of a retypable ``datatype``."""
    return _typed_value(lexical, datatype) is not None


def normalize_literals_against_schema(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    catalog_roles: CatalogRoles | None = None,
) -> int:
    """Retype literals whose predicate declares a compatible datatype range.

    Fixes the ``qudt:numericValue 230`` vs ``"230"^^xsd:decimal`` drift at parse
    time, and the same drift for the date, time and duration datatypes: when
    the range (:func:`literal_ranges`) is in :data:`_RETYPABLE_RANGE_DATATYPES`
    and the lexical form is a value of it, the literal is rewritten with it. A
    calendar text that names exactly one value of the range (``"29 June
    2021"`` on an ``xsd:date`` range) is rewritten to that value.

    A literal is only retyped from an untyped, ``xsd:string``, or numeric source
    -- a string range must never be able to clobber a correctly typed value --
    and language-tagged literals are left alone, since they are
    ``rdf:langString`` and retyping would discard the tag.

    Returns:
        Number of retyped literals.
    """
    declared_ranges = literal_ranges(ontology_graph, catalog_roles)
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
        value = _typed_value(str(obj).strip(), target_datatype)
        if value is None:
            continue
        replacements.append(((subject, predicate, obj), (subject, predicate, value)))

    for old, new in replacements:
        graph.remove(old)
        graph.add(new)
    return len(replacements)


def temporal_range_mismatches(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    catalog_roles: CatalogRoles | None = None,
) -> list[tuple[tuple[Node, Node, Node], RejectedLiteralTriple]]:
    """Text on a date, time or duration range that is not a value of it.

    Run after :func:`normalize_literals_against_schema`, so a text that is such
    a value has already been retyped: what is left untyped, ``xsd:string`` or
    language-tagged on a predicate whose range (:func:`literal_ranges`) is a
    temporal datatype is not one -- a recurrence or a phrase where a date
    belongs. Numeric ranges are left to the numeric checks.

    Returns:
        Each such triple with its quarantine record (``expected_datatype``
        names the range).
    """
    temporal_ranges = {
        predicate: datatype
        for predicate, datatype in literal_ranges(ontology_graph, catalog_roles).items()
        if datatype in _TEMPORAL_RANGE_DATATYPES
    }
    if not temporal_ranges:
        return []
    found: list[tuple[tuple[Node, Node, Node], RejectedLiteralTriple]] = []
    for subject, predicate, obj in graph:
        if not isinstance(obj, Literal) or obj.datatype not in (None, XSD.string):
            continue
        datatype = temporal_ranges.get(predicate)
        if datatype is None or _fits_range(str(obj).strip(), datatype):
            continue
        found.append(
            (
                (subject, predicate, obj),
                RejectedLiteralTriple(
                    subject=str(subject),
                    predicate=str(predicate),
                    object_lexical=str(obj),
                    datatype=str(obj.datatype or ""),
                    reason=(
                        "this text is not a value of the property's datatype; "
                        "write a value of that datatype, or state the text with "
                        "a property that takes text"
                    ),
                    expected_datatype=str(datatype),
                ),
            )
        )
    return found


def partition_literals_outside_temporal_ranges(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    catalog_roles: CatalogRoles | None = None,
) -> tuple[RDFGraph, list[RejectedLiteralTriple]]:
    """Split off :func:`temporal_range_mismatches`, quarantined for the critic.

    Returns:
        Tuple of (clean graph, quarantined triples).
    """
    found = temporal_range_mismatches(graph, ontology_graph, catalog_roles)
    if not found:
        return graph, []
    set_aside = {triple for triple, _ in found}
    clean = RDFGraph()
    for prefix, namespace_uri in graph.namespaces():
        if prefix:
            clean.bind(prefix, namespace_uri)
    for triple in graph:
        if triple not in set_aside:
            clean.add(triple)
    return clean, [record for _, record in found]


def _quarantined_subject(graph: RDFGraph, item: RejectedLiteralTriple) -> Node:
    iri = URIRef(item.subject)
    return iri if (iri, None, None) in graph else BNode(item.subject)


def _states_text(graph: RDFGraph, subject: Node, text: str) -> bool:
    return any(
        isinstance(obj, Literal) and str(obj) == text
        for obj in graph.objects(subject, None)
    )


def _replaces(obj: Node, item: RejectedLiteralTriple) -> bool:
    """Whether ``obj`` is a value of the kind ``item`` was set aside for lacking."""
    if item.expected_datatype:
        return (
            isinstance(obj, Literal)
            and str(obj.datatype) == item.expected_datatype
            and RDFGraph._is_valid_typed_literal(obj)
        )
    if item.reason:
        return isinstance(obj, URIRef | BNode)
    return (
        isinstance(obj, Literal)
        and obj.datatype is not None
        and RDFGraph._is_valid_typed_literal(obj)
    )


def unresolved_quarantine(
    graph: RDFGraph, items: Sequence[RejectedLiteralTriple]
) -> list[RejectedLiteralTriple]:
    """The quarantined triples ``graph`` has not since made good.

    One is resolved when its subject now states its predicate with a value of
    the kind it was set aside for lacking (an IRI for an object property, a
    valid literal of the expected or declared datatype), or states its text
    under any predicate.
    """
    unresolved: list[RejectedLiteralTriple] = []
    for item in items:
        subject = _quarantined_subject(graph, item)
        if _states_text(graph, subject, item.object_lexical):
            continue
        if any(
            _replaces(obj, item)
            for obj in graph.objects(subject, URIRef(item.predicate))
        ):
            continue
        unresolved.append(item)
    return unresolved


def write_back_temporal_quarantine(
    graph: RDFGraph, items: Sequence[RejectedLiteralTriple]
) -> list[GraphRepairRecord]:
    """Put back, as ``rdfs:comment``, date text the unit still holds aside.

    A text set aside for not being a value of a date, time or duration
    datatype -- written plain on such a range, or typed with such a datatype
    it is not a value of -- is kept verbatim on its subject unless the subject
    already states it, so nothing is lost and no false date is stated.
    Quarantined strings on object properties and bad numbers are not written
    back.

    Returns:
        One ``quarantine_as_comment`` record per text written back.
    """
    records: list[GraphRepairRecord] = []
    for item in items:
        datatype = item.expected_datatype or item.datatype
        if URIRef(datatype) not in _TEMPORAL_RANGE_DATATYPES:
            continue
        subject = _quarantined_subject(graph, item)
        if _states_text(graph, subject, item.object_lexical):
            continue
        graph.add((subject, RDFS.comment, Literal(item.object_lexical)))
        records.append(
            GraphRepairRecord(
                kind=FactsGateRepairKind.QUARANTINE_AS_COMMENT,
                source=f"<{item.predicate}> {item.object_lexical!r}",
                target=item.subject,
            )
        )
    return records


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


def repair_compact_iri_literals(
    graph: RDFGraph,
    ontology_context_graph: RDFGraph,
    known_terms: set[str],
) -> tuple[int, list[GraphRepairRecord]]:
    """Coerce plain-string compact IRIs on IRI-valued positions into IRIs.

    A JSON-LD bare string (``"qudt:unit": "unit:NanoM"``) parses as a literal,
    so the node loses its link and reads as missing the property. A literal is
    rewritten only when every condition holds: it is plain (no language tag,
    no datatype or ``xsd:string``); the predicate is not ``rdf:type`` (that is
    :func:`repair_literal_type_objects`); the lexical form is a compact IRI
    whose prefix the graph or the known-prefix context binds; and either the
    schema says the predicate takes an IRI or the expanded IRI is a known term.
    Prose such as ``"time: 10 minutes"`` never matches the compact-IRI form.

    Args:
        graph: Facts graph, repaired in place.
        ontology_context_graph: Schema that says which predicates take IRIs.
        known_terms: IRIs the catalog and fallback vocabulary declare; covers
            predicates whose range the vendored vocabulary leaves undeclared.

    Returns:
        Tuple of (number of rewritten triples, applied-repair records).
    """
    prefix_map = dict(RDFGraph.get_known_prefixes() or {})
    prefix_map.update(
        {prefix: str(namespace) for prefix, namespace in graph.namespaces() if prefix}
    )
    replacements: list[tuple[tuple[Node, Node, Node], URIRef, str]] = []
    expects_iri: dict[URIRef, bool] = {}
    for subject, predicate, obj in graph:
        if predicate == RDF.type or not isinstance(obj, Literal):
            continue
        if obj.language or obj.datatype not in (None, XSD.string):
            continue
        lexical = str(obj).strip()
        match = _COMPACT_IRI.match(lexical)
        if match is None:
            continue
        namespace = prefix_map.get(match.group(1))
        if not namespace:
            continue
        expanded = f"{namespace}{match.group(2)}"
        if expanded not in known_terms:
            if not isinstance(predicate, URIRef):
                continue
            if predicate not in expects_iri:
                expects_iri[predicate] = _predicate_expects_iri_object(
                    predicate, ontology_context_graph
                )[0]
            if not expects_iri[predicate]:
                continue
        replacements.append(((subject, predicate, obj), URIRef(expanded), lexical))

    applied: list[GraphRepairRecord] = []
    for (subject, predicate, obj), iri, lexical in replacements:
        graph.remove((subject, predicate, obj))
        graph.add((subject, predicate, iri))
        applied.append(
            GraphRepairRecord(
                kind=FactsUnitFindingKind.COMPACT_IRI_LITERAL,
                source=lexical,
                target=str(iri),
            )
        )
        logger.info("Repaired compact-IRI literal %r -> <%s>", lexical, iri)
    if applied:
        record_active_count("repair/compact_iri_literal", len(applied))
    return len(applied), applied


def materialize_entailed_types(
    graph: RDFGraph, ontology_graph: RDFGraph | None
) -> tuple[int, list[GraphRepairRecord]]:
    """Assert the types the ontology context entails on the graph's nodes.

    For every triple ``x p y``: when the context declares ``p rdfs:domain D``
    with ``D`` a named class, ``x a D`` is added; when it declares
    ``p rdfs:range R`` with ``R`` a named class and ``y`` an IRI, ``y a R``
    is added. ``owl:Thing`` and the other vacuous classes are never asserted,
    nor is a type on a node the context itself types (a catalog individual).

    A type is added only where it is consistent with what the node already
    asserts: when the node has no type, or when the entailed class
    specialises one of its asserted types. Nothing is added when the entailed
    class is already in the asserted types' superclass closure (redundant),
    or when every asserted type is unrelated to it -- that contradiction is
    the ``DOMAIN_VIOLATION`` finding's to report, and materialising the
    domain there would hide it.

    Args:
        graph: Facts graph, repaired in place.
        ontology_graph: Ontology context declaring domains and ranges.

    Returns:
        Tuple of (number of added triples, applied-repair records).
    """
    if ontology_graph is None or not len(ontology_graph):
        return 0, []
    domains = _declared_domains(ontology_graph)
    ranges = declared_class_ranges(ontology_graph)
    if not domains and not ranges:
        return 0, []

    closures: dict[URIRef, set[URIRef]] = {}

    def closure(class_iri: URIRef) -> set[URIRef]:
        if class_iri not in closures:
            closures[class_iri] = superclass_closure(class_iri, ontology_graph)
        return closures[class_iri]

    def admissible(node: Node, entailed: URIRef) -> bool:
        if entailed in _VACUOUS_TYPES or not isinstance(node, (URIRef, BNode)):
            return False
        if (node, RDF.type, None) in ontology_graph:
            return False
        asserted = {
            value
            for value in graph.objects(node, RDF.type)
            if isinstance(value, URIRef) and value not in _VACUOUS_TYPES
        }
        if not asserted:
            return True
        if any(entailed in closure(value) for value in asserted):
            return False
        return any(value in closure(entailed) for value in asserted)

    additions: dict[tuple[Node, URIRef], URIRef] = {}
    for subject, predicate, obj in graph:
        if not isinstance(predicate, URIRef) or predicate == RDF.type:
            continue
        for entailed in domains.get(predicate, ()):
            if admissible(subject, entailed):
                additions.setdefault((subject, entailed), predicate)
        if not isinstance(obj, URIRef):
            continue
        for entailed in ranges.get(predicate, ()):
            if admissible(obj, entailed):
                additions.setdefault((obj, entailed), predicate)

    applied: list[GraphRepairRecord] = []
    for (node, entailed), predicate in sorted(
        additions.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        if (node, RDF.type, entailed) in graph:
            continue
        graph.add((node, RDF.type, entailed))
        applied.append(
            GraphRepairRecord(
                kind=FactsUnitFindingKind.ENTAILED_TYPE,
                source=f"{node} {predicate}",
                target=str(entailed),
            )
        )
        logger.info("Materialised entailed type <%s> on <%s>", entailed, node)
    if applied:
        record_active_count("repair/entailed_type", len(applied))
    return len(applied), applied


def _folded_name(local: str) -> str:
    """A local name with case and separators removed, for spelling equality."""
    return re.sub(r"[_\-]", "", local).lower()


def _alias_qualifies(alias_local: str, candidate_local: str) -> bool:
    """Whether a candidate is a near-miss of the alias rather than a look-alike.

    Token containment either way (``value`` in ``numericValue``, ``lowerBound``
    in ``hasLowerBound``), or the same spelling once case and separators are
    folded (``numericvalue`` for ``numericValue``). A candidate that swaps one
    token for another never qualifies, however close the two strings are:
    that is a different term, not a misspelt one.
    """
    alias_tokens = _name_tokens(alias_local)
    candidate_tokens = _name_tokens(candidate_local)
    if alias_tokens and (
        alias_tokens <= candidate_tokens or candidate_tokens <= alias_tokens
    ):
        return True
    folded = _folded_name(alias_local)
    return bool(folded) and folded == _folded_name(candidate_local)


def _select_alias_target(
    alias: str, candidates: Sequence[str], min_ratio: float
) -> str | None:
    """The one candidate ``alias`` may be rewritten to, or None.

    Exactly one qualifying candidate (see :func:`_alias_qualifies`) wins
    outright. Among several, the ``SequenceMatcher`` ratio breaks the tie only
    when the best-scoring candidate clears ``min_ratio`` and no other
    candidate matches its score; the ratio never admits a candidate that did
    not qualify on its own.
    """
    alias_local = _local_name(alias)
    qualifying = [
        candidate
        for candidate in candidates
        if _alias_qualifies(alias_local, _local_name(candidate))
    ]
    if len(qualifying) == 1:
        return qualifying[0]
    if not qualifying:
        return None
    scored = sorted(
        (
            (
                SequenceMatcher(
                    None, alias_local.lower(), _local_name(candidate).lower()
                ).ratio(),
                candidate,
            )
            for candidate in qualifying
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_ratio, best = scored[0]
    if best_ratio < min_ratio or scored[1][0] == best_ratio:
        return None
    return best


def _same_name_elsewhere(alias: URIRef, properties: set[str]) -> str | None:
    """The one catalog property in another namespace spelled exactly like ``alias``.

    A property name written under the wrong prefix (``a:usedMethod`` for
    ``b:usedMethod``) means that property: the name is exact, only the prefix
    slipped. Two such properties are ambiguous, so neither is taken; only a
    term the catalog states is a property can replace a predicate.
    """
    namespace = _namespace_of(str(alias))
    local = _local_name(str(alias))
    matches = {
        term
        for term in properties
        if _namespace_of(term) != namespace and _local_name(term) == local
    }
    return next(iter(matches)) if len(matches) == 1 else None


def repair_property_aliases(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    *,
    min_ratio: float = 0.95,
    exempt_terms: set[str] | None = None,
    full_catalog_terms: set[str] | None = None,
    catalog_roles: CatalogRoles | None = None,
) -> tuple[int, list[FactsUnitFinding], list[GraphRepairRecord]]:
    """Rewrite near-miss predicates in catalog namespaces; report ambiguity.

    A predicate that is not a catalog term and not a real term of a
    well-known vocabulary (:func:`~ontocast.onto.well_known.is_well_known_term`)
    exists nowhere. When exactly one catalog property (``catalog_roles``, the
    whole catalog, else the snapshot) in another namespace has its very local
    name, the prefix slipped and that property is the target; this holds for
    a predicate in a catalog namespace or a well-known one. A predicate in a
    namespace the catalog declares terms in is otherwise a near-miss
    (``ex:lowerBound`` for ``ex:hasLowerBound``), and the rewrite fires only when
    exactly one candidate *qualifies*: its name tokens contain the alias's,
    the alias's contain its, or the two spell the same once case and
    separators are folded. Several qualifying candidates are a tie that the
    ``SequenceMatcher`` ratio may break -- the best-scoring one wins when it
    clears ``min_ratio`` and nothing ties it. Similarity alone never licenses
    a rewrite: a high ratio also joins names that differ by exactly one token,
    and rewriting across that gap substitutes one property for another.
    Anything else becomes a mandatory finding carrying the top suggestions.

    Only namespaces the catalog *declares* terms in are eligible for the
    near-miss rewrite (see :func:`collect_declared_namespaces`); ``exempt_terms`` (expanded fallback
    vocabulary) are never treated as near-misses, and neither is a predicate
    in ``full_catalog_terms``: a term the whole catalog declares is real even
    when this unit's snapshot did not retrieve it, and the snapshot's
    look-alike is then a different property, not the intended one. Such a
    predicate is left untouched here; whether it is reported is the term
    checks' business.

    Returns:
        Tuple of (number of rewritten triples, unresolved findings,
        applied-repair records).
    """
    catalog_terms = collect_catalog_terms(ontology_graph)
    if not catalog_terms:
        return 0, [], []
    declared_namespaces = collect_declared_namespaces(ontology_graph)
    exempt = exempt_terms or set()
    known_elsewhere = full_catalog_terms or set()

    findings: list[FactsUnitFinding] = []
    applied: list[GraphRepairRecord] = []
    rewritten = 0
    nowhere = {
        predicate
        for predicate in graph.predicates()
        if isinstance(predicate, URIRef)
        and str(predicate) not in catalog_terms
        and str(predicate) not in exempt
        and str(predicate) not in known_elsewhere
        and is_well_known_term(str(predicate)) is not True
    }
    snapshot_properties, _ = _catalog_term_roles(ontology_graph)
    # A well-known property the catalog only references is a property too.
    properties = (
        set(snapshot_properties)
        | (set(catalog_roles.properties) if catalog_roles is not None else set())
        | {
            term
            for term in catalog_terms | known_elsewhere
            if well_known_role(term) == "property"
        }
    )
    for alias in sorted(nowhere, key=str):
        near_miss = _namespace_of(str(alias)) in declared_namespaces
        if not near_miss and is_well_known_term(str(alias)) is None:
            continue
        candidates = (
            _alias_candidates(
                alias, graph, catalog_terms, ontology_graph=ontology_graph
            )
            if near_miss
            else []
        )
        target = _same_name_elsewhere(alias, properties) or (
            _select_alias_target(str(alias), candidates, min_ratio)
            if near_miss
            else None
        )
        if target is not None:
            replacement = URIRef(target)
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
        if not near_miss:
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
            closures[class_iri] = superclass_closure(class_iri, ontology_graph)
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


def _not_a_failed_cast(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith("Failed to convert Literal lexical")


@contextmanager
def _quiet_rdflib_casts() -> Generator[None]:
    """Filter rdflib's warning and traceback for a lexical form that does not parse.

    Building a literal is how a lexical form is tried here, so a failure is an
    answer, not an error.
    """
    rdflib_term = logging.getLogger("rdflib.term")
    rdflib_term.addFilter(_not_a_failed_cast)
    try:
        yield
    finally:
        rdflib_term.removeFilter(_not_a_failed_cast)


def _literal_parses_as(lexical: str, datatype: URIRef) -> bool:
    """True when ``lexical`` is a well-formed literal of ``datatype``.

    Date, time and duration datatypes are judged in rdflib's normal form
    against their XSD lexical spaces; the others by whether rdflib reads a
    value from the form.
    """
    with _quiet_rdflib_casts():
        value = Literal(lexical, datatype=datatype)
    temporal = is_valid_temporal_lexical(str(value), datatype)
    if temporal is not None:
        return temporal
    return value.value is not None


def dedupe_literal_variants(
    graph: RDFGraph,
    fact_namespaces: Sequence[str] | None = None,
) -> list[GraphRepairRecord]:
    """Collapse duplicate literals differing only in language tag or datatype.

    The renderer emits the same value inconsistently across chunks —
    ``"X"@en`` in one unit, ``"X"^^xsd:string`` in another, a plain ``"X"``
    in a third — and after aggregation one ``(subject, predicate)`` carries
    all three as distinct RDF terms. One survives per lexical form: the
    language-tagged form (each distinct language kept — those are distinct
    assertions), else the plain form, else the ``xsd:string`` form. Reified
    provenance follows the survivor.

    Args:
        graph: Aggregated facts graph, mutated in place.
        fact_namespaces: When set, only subjects under these namespaces are
            touched.

    Returns:
        One :class:`GraphRepairRecord` per removed literal variant.
    """
    from ontocast.onto.rdfgraph import retarget_reifiers
    from ontocast.tool.facts_validation.terms import _in_fact_scope

    namespaces = [ns for ns in (fact_namespaces or []) if ns]
    groups: dict[tuple[URIRef, URIRef, str], list[Literal]] = {}
    for subject, predicate, obj in graph:
        if not isinstance(subject, URIRef) or not isinstance(predicate, URIRef):
            continue
        if not isinstance(obj, Literal):
            continue
        if namespaces and not _in_fact_scope(subject, namespaces):
            continue
        groups.setdefault((subject, predicate, str(obj)), []).append(obj)

    records: list[GraphRepairRecord] = []
    replacements: dict[tuple[Node, Node, Node], tuple[Node, Node, Node]] = {}
    for (subject, predicate, _lexical), literals in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2])
    ):
        if len(literals) < 2:
            continue
        tagged = sorted(
            (lit for lit in literals if lit.language is not None),
            key=lambda lit: str(lit.language),
        )
        plain = [
            lit for lit in literals if lit.language is None and lit.datatype is None
        ]
        typed = [
            lit
            for lit in literals
            if lit.language is None and lit.datatype == XSD.string
        ]
        if tagged:
            keep = tagged[0]
            drop = plain + typed
        elif plain:
            keep = plain[0]
            drop = typed
        else:
            # Distinct datatypes beyond xsd:string are not variants of one
            # value; leave them to the datatype-aware repairs.
            continue
        for variant in drop:
            graph.remove((subject, predicate, variant))
            replacements[(subject, predicate, variant)] = (subject, predicate, keep)
            records.append(
                GraphRepairRecord(
                    kind=FactsGateRepairKind.LITERAL_VARIANT_PRUNED,
                    source=f"{subject} {predicate} {variant.n3()}",
                    target=keep.n3(),
                )
            )
    if replacements:
        retarget_reifiers(graph, replacements)
        logger.info(
            "Collapsed %d literal variant(s) differing only in language tag "
            "or datatype",
            len(records),
        )
    return records
