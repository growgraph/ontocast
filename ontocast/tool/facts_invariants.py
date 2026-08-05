"""Deterministic normalization, repair, and findings for rendered facts.

The renderer LLM is treated as a transcriber, not a guarantor: every check
here detects or repairs a concrete violation in code, and only unresolved
items go back to the LLM as mandatory fix instructions. All mechanisms are
schema-driven and domain-agnostic — alias tables and known-term sets are
derived from the ontology context at runtime, never hardcoded per
vocabulary.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef
from rdflib.namespace import DCTERMS, PROV, XSD

from ontocast.onto.model import (
    FactsUnitFinding,
    FactsUnitFindingKind,
    FactsValidationFinding,
    FactsValidationFindingKind,
    GraphRepairRecord,
)
from ontocast.onto.rdfgraph import RDFGraph, RejectedLiteralTriple
from ontocast.tool.agg.signatures import canonical_literal, harvest_max_one_predicates
from ontocast.util.numeric_inventory import canonical_number, missing_numeric_mentions

logger = logging.getLogger(__name__)

_FORBIDDEN_NAMESPACES = ("http://example.org/", "https://example.org/")

_NUMERIC_RANGE_DATATYPES = {
    XSD.decimal,
    XSD.integer,
    XSD.float,
    XSD.double,
    XSD.nonNegativeInteger,
    XSD.positiveInteger,
}

# Meta-vocabularies: the RDF/OWL substrate plus the annotation and provenance
# terms every facts graph carries regardless of catalog. Exempting these from
# UNKNOWN_TERM keeps the signal readable -- flagging rdfs:label would bury it.
#
# Domain vocabularies do NOT belong here. SOSA/SSN (sensor observations),
# CSVW (tabular metadata), FOAF and schema.org (people, organizations,
# creative works) model a subject area; a catalog that does not declare them
# should hear about it. They are exempted by configuration
# (FACTS_ADDITIONAL_STANDARD_NAMESPACES) when a deployment wants them, not by
# being compiled in.
_STANDARD_NAMESPACES = (
    str(RDF),
    str(RDFS),
    str(OWL),
    str(XSD),
    str(SKOS),
    str(DCTERMS),
    "http://purl.org/dc/elements/1.1/",
    "http://www.w3.org/ns/prov#",
    "http://www.w3.org/XML/1998/namespace",
)

_LOCAL_SPLIT = re.compile(r"[#/]")

# Bounds on the advisory alias-candidate list shown to the renderer. Not
# applied to the graph, so these trim prompt noise rather than gate a rewrite.
_ALIAS_MIN_SCORE = 0.5
_ALIAS_MAX_SUGGESTIONS = 3


def _namespace_of(iri: str) -> str:
    """Split an IRI into its namespace part (up to the last '#' or '/')."""
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return head + sep
    return iri


def _local_name(iri: str) -> str:
    for separator in ("#", "/"):
        head, sep, local = iri.rpartition(separator)
        if sep and local:
            return local
    return iri


def _name_tokens(local: str) -> set[str]:
    tokens = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
    tokens = re.sub(r"[_\-]", " ", tokens)
    return {token.lower() for token in tokens.split() if token}


def collect_catalog_terms(ontology_graph: RDFGraph | None) -> set[str]:
    """All IRIs appearing anywhere in the ontology context."""
    terms: set[str] = set()
    if ontology_graph is None:
        return terms
    for triple in ontology_graph:
        for term in triple:
            if isinstance(term, URIRef):
                terms.add(str(term))
    return terms


def collect_catalog_namespaces(ontology_graph: RDFGraph | None) -> set[str]:
    """Namespaces of every IRI appearing in the ontology context."""
    return {_namespace_of(term) for term in collect_catalog_terms(ontology_graph)}


def normalize_literals_against_schema(
    graph: RDFGraph, ontology_graph: RDFGraph | None
) -> int:
    """Retype untyped numeric literals whose predicate declares a numeric range.

    Fixes the ``qudt:numericValue 230`` vs ``"230"^^xsd:decimal`` drift at
    parse time: when the schema says the range is numeric and the lexical form
    parses as a number, the literal is rewritten with the declared datatype.

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
            and range_iri in _NUMERIC_RANGE_DATATYPES
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
        if obj.datatype is not None and obj.datatype not in _NUMERIC_RANGE_DATATYPES:
            continue
        if canonical_number(str(obj).strip()) is None:
            continue
        replacements.append(
            (
                (subject, predicate, obj),
                (
                    subject,
                    predicate,
                    Literal(str(obj).strip(), datatype=target_datatype),
                ),
            )
        )

    for old, new in replacements:
        graph.remove(old)
        graph.add(new)
    return len(replacements)


_COMPACT_IRI = re.compile(r"^([A-Za-z][\w.-]*):([^\s/][^\s]*)$")


def _resolve_type_literal(lexical: str, prefix_map: dict[str, str]) -> str | None:
    """Resolve a literal ``rdf:type`` object to a full IRI, when unambiguous.

    Accepts absolute IRIs and compact IRIs whose prefix is bound in the graph.
    Returns None when the lexical form cannot be resolved deterministically.
    """
    if lexical.startswith(("http://", "https://", "urn:")) and " " not in lexical:
        return lexical
    match = _COMPACT_IRI.match(lexical)
    if match is None:
        return None
    namespace = prefix_map.get(match.group(1))
    if not namespace:
        return None
    return f"{namespace}{match.group(2)}"


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


def _alias_candidates(
    alias: URIRef,
    graph: RDFGraph,
    catalog_terms: set[str],
) -> list[str]:
    """Rank replacement candidates for a near-miss predicate.

    Candidate pool: catalog terms sharing the alias namespace, plus predicates
    of that namespace the graph itself uses on >= 2 subjects (the renderer's
    own dominant usage defines the alias target — this is how
    ``qudt:value`` resolves to ``qudt:numericValue`` even when the snapshot
    does not spell the property out).
    """
    namespace = _namespace_of(str(alias))
    pool: set[str] = {
        term for term in catalog_terms if _namespace_of(term) == namespace
    }
    predicate_subjects: dict[str, set[str]] = {}
    for subject, predicate, _ in graph:
        text = str(predicate)
        if text != str(alias) and _namespace_of(text) == namespace:
            predicate_subjects.setdefault(text, set()).add(str(subject))
    pool.update(
        predicate
        for predicate, subjects in predicate_subjects.items()
        if len(subjects) >= 2
    )
    pool.discard(str(alias))

    alias_local = _local_name(str(alias))
    alias_tokens = _name_tokens(alias_local)
    scored: list[tuple[float, str]] = []
    for candidate in pool:
        candidate_local = _local_name(candidate)
        candidate_tokens = _name_tokens(candidate_local)
        if alias_tokens and (
            alias_tokens <= candidate_tokens or candidate_tokens <= alias_tokens
        ):
            score = 1.0
        else:
            score = SequenceMatcher(
                None, alias_local.lower(), candidate_local.lower()
            ).ratio()
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    # Suggestions only: these are shown to the renderer as candidates, never
    # applied. The 0.5 floor drops noise and the top-3 cap keeps the prompt
    # readable -- both are presentation bounds on an advisory list, so
    # truncation here cannot silently change the graph.
    qualifying = [candidate for score, candidate in scored if score >= _ALIAS_MIN_SCORE]
    return qualifying[:_ALIAS_MAX_SUGGESTIONS]


def repair_property_aliases(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    *,
    min_ratio: float = 0.85,
) -> tuple[int, list[FactsUnitFinding], list[GraphRepairRecord]]:
    """Rewrite near-miss predicates in catalog namespaces; report ambiguity.

    A predicate whose namespace belongs to the ontology context but which is
    not itself a catalog term is a near-miss (``qqval:lowerBound`` for
    ``qqval:hasLowerBound``). When exactly one candidate scores above
    ``min_ratio`` (token containment counts as 1.0) the rewrite is applied
    deterministically; otherwise a mandatory finding carries the top
    suggestions.

    Returns:
        Tuple of (number of rewritten triples, unresolved findings,
        applied-repair records).
    """
    catalog_terms = collect_catalog_terms(ontology_graph)
    if not catalog_terms:
        return 0, [], []
    catalog_namespaces = {_namespace_of(term) for term in catalog_terms}

    findings: list[FactsUnitFinding] = []
    applied: list[GraphRepairRecord] = []
    rewritten = 0
    predicates = {
        predicate
        for predicate in graph.predicates()
        if isinstance(predicate, URIRef)
        and str(predicate) not in catalog_terms
        and _namespace_of(str(predicate)) in catalog_namespaces
    }
    for alias in sorted(predicates, key=str):
        candidates = _alias_candidates(alias, graph, catalog_terms)
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


def _closed_range_suggestions(
    rejected: RejectedLiteralTriple, ontology_graph: RDFGraph | None
) -> list[str]:
    """Suggest named individuals of the expected range matching the literal."""
    if ontology_graph is None or not rejected.expected_range:
        return []
    range_ref = URIRef(rejected.expected_range)
    token = rejected.object_lexical.strip()
    suggestions: list[str] = []
    for individual in ontology_graph.subjects(RDF.type, range_ref):
        if not isinstance(individual, URIRef):
            continue
        surfaces = {
            str(value)
            for predicate in (RDFS.label, SKOS.notation, SKOS.altLabel)
            for value in ontology_graph.objects(individual, predicate)
        }
        surfaces.add(_local_name(str(individual)))
        # Character-for-character only: the facts prompt instructs that a
        # lowercase symbol and its uppercase variant denote DIFFERENT
        # individuals, so the suggester must not propose `unit:M` for "m".
        if token in surfaces:
            suggestions.append(str(individual))
    return sorted(suggestions)[:3]


def collect_unit_findings(
    *,
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    quarantined: list[RejectedLiteralTriple],
    extraction_text: str,
    fact_namespaces: list[str],
    coverage_limit: int = 30,
    additional_standard_namespaces: Sequence[str] = (),
) -> list[FactsUnitFinding]:
    """Assemble all deterministic findings for one rendered unit graph.

    Mandatory: quarantined literals (with closed-range individual
    suggestions), forbidden-namespace terms (``example.org``), doc-namespace
    predicates, and unresolved catalog near-misses. Advisory-strong: numeric
    mentions of the source text absent from the graph — the renderer decides
    per item whether each is an extractable quantity or an artifact.
    """
    findings: list[FactsUnitFinding] = []
    standard_namespaces = (
        *_STANDARD_NAMESPACES,
        *additional_standard_namespaces,
    )

    for rejected in quarantined:
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.QUARANTINED_LITERAL,
                message=(
                    f"Triple excluded ({rejected.reason}): the object of "
                    f"<{rejected.predicate}> must be an IRI/valid literal, got "
                    f"'{rejected.object_lexical}'."
                ),
                subject=rejected.subject,
                predicate=rejected.predicate,
                value=rejected.object_lexical,
                suggestions=_closed_range_suggestions(rejected, ontology_graph),
            )
        )

    catalog_terms = collect_catalog_terms(ontology_graph)
    catalog_namespaces = {_namespace_of(term) for term in catalog_terms}
    normalized_fact_namespaces = [ns for ns in fact_namespaces if ns]

    prefix_map = {
        prefix: str(namespace) for prefix, namespace in graph.namespaces() if prefix
    }
    flagged_terms: set[str] = set()
    for subject, predicate, obj in graph:
        if predicate == RDF.type and isinstance(obj, Literal):
            lexical = str(obj).strip()
            if lexical in flagged_terms:
                continue
            flagged_terms.add(lexical)
            resolved = _resolve_type_literal(lexical, prefix_map)
            findings.append(
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.LITERAL_TYPE_OBJECT,
                    message=(
                        f"rdf:type object '{lexical}' is a string literal, not "
                        "an IRI; assert the type as a catalog class IRI "
                        "(`a prefix:Class`), never as a quoted string."
                    ),
                    subject=str(subject),
                    value=lexical,
                    suggestions=[resolved]
                    if resolved and resolved in catalog_terms
                    else [],
                )
            )
            continue
        for position, term in (("predicate", predicate), ("type", obj)):
            if not isinstance(term, URIRef):
                continue
            if position == "type" and predicate != RDF.type:
                continue
            text = str(term)
            if text in flagged_terms:
                continue
            if text.startswith(_FORBIDDEN_NAMESPACES):
                flagged_terms.add(text)
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=(
                            f"<{text}> uses the example.org placeholder namespace; "
                            "replace it with a catalog term or express the "
                            "statement with catalog/standard vocabulary."
                        ),
                        predicate=text,
                        suggestions=_alias_candidates(term, graph, catalog_terms)
                        if catalog_terms
                        else [],
                    )
                )
                continue
            if position == "predicate" and any(
                text.startswith(ns) for ns in normalized_fact_namespaces
            ):
                flagged_terms.add(text)
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=(
                            f"Predicate <{text}> is minted in the facts/document "
                            "namespace; facts namespaces hold instances only — "
                            "use a catalog or standard-vocabulary property."
                        ),
                        predicate=text,
                        suggestions=_alias_candidates(term, graph, catalog_terms)
                        if catalog_terms
                        else [],
                    )
                )
                continue
            namespace = _namespace_of(text)
            if (
                catalog_terms
                and namespace in catalog_namespaces
                and text not in catalog_terms
                and not namespace.startswith(standard_namespaces)
            ):
                flagged_terms.add(text)
                findings.append(
                    FactsUnitFinding(
                        kind=FactsUnitFindingKind.UNKNOWN_TERM,
                        message=(
                            f"<{text}> does not exist in its ontology; use one of "
                            "the suggested catalog terms or the closest correct "
                            "term from the ontology chapter."
                        ),
                        predicate=text,
                        suggestions=_alias_candidates(term, graph, catalog_terms),
                    )
                )

    missing = missing_numeric_mentions(extraction_text, graph, limit=coverage_limit)
    if missing:
        findings.append(
            FactsUnitFinding(
                kind=FactsUnitFindingKind.NUMERIC_COVERAGE,
                mandatory=False,
                message=(
                    "These numbers appear in the source text but not in the "
                    "graph. For each: extract it as a typed literal on an "
                    "appropriate node (verbatim value and source unit — never "
                    "convert units) if it is a factual quantity, or ignore it "
                    "if it is a page/citation/figure artifact: " + ", ".join(missing)
                ),
                value=", ".join(missing),
            )
        )

    return findings


class FactsValidationReport(BaseModel):
    """Invariant findings over one aggregated facts graph."""

    findings: list[FactsValidationFinding] = Field(default_factory=list)

    @property
    def error_findings(self) -> list[FactsValidationFinding]:
        """Findings that justify a deterministic un-merge repair pass."""
        return [finding for finding in self.findings if finding.severity == "error"]


def _in_fact_scope(subject: URIRef, fact_namespaces: list[str]) -> bool:
    if not fact_namespaces:
        return True
    text = str(subject)
    return any(text.startswith(namespace) for namespace in fact_namespaces if namespace)


def _distinct_object_keys(objects: set) -> set[str]:
    """Distinct-value keys for a mixed object set (canonical for literals)."""
    keys: set[str] = set()
    for obj in objects:
        if isinstance(obj, Literal):
            canonical = canonical_literal(obj)
            keys.add(f"lit:{canonical[0]}" if canonical else f"lex:{obj}")
        else:
            keys.add(f"iri:{obj}")
    return keys


def _dominant_single_valued_predicates(
    iri_groups: dict[tuple[URIRef, URIRef], set[URIRef]],
    *,
    min_single_support: int,
) -> set[URIRef]:
    """Predicates whose IRI objects are single-valued for a dominant majority.

    Unlike the strict corpus inference used by merge guards, this tolerates a
    minority of multi-valued subjects — those are exactly the violation
    candidates the gate reports (e.g. one quantity node carrying two
    ``qudt:unit`` IRIs after a bad merge, while every other quantity node
    carries one).
    """
    single: dict[URIRef, int] = {}
    multi: dict[URIRef, int] = {}
    for (_, predicate), objects in iri_groups.items():
        bucket = single if len(objects) == 1 else multi
        bucket[predicate] = bucket.get(predicate, 0) + 1
    return {
        predicate
        for predicate, count in single.items()
        if count >= min_single_support and count >= 3 * multi.get(predicate, 0)
    }


def _shacl_findings(
    graph: RDFGraph, shapes_graph: RDFGraph
) -> list[FactsValidationFinding]:
    """Run pyshacl when available.

    Reaching here means shapes were found, so the caller expects validation to
    happen: a missing extra is reported at warning level, not debug. Silently
    returning "no findings" is indistinguishable from "conforms".
    """
    try:
        import pyshacl
    except ImportError:
        logger.warning(
            "SHACL shapes are configured but pyshacl is not installed; "
            "skipping SHACL validation. Install the extra: uv sync --extra shacl"
        )
        return []

    shacl = "http://www.w3.org/ns/shacl#"
    conforms, results_graph, _ = pyshacl.validate(
        graph, shacl_graph=shapes_graph, inference="none", abort_on_first=False
    )
    if conforms:
        return []
    findings: list[FactsValidationFinding] = []
    for result in results_graph.subjects(RDF.type, URIRef(shacl + "ValidationResult")):
        severity_iri = results_graph.value(result, URIRef(shacl + "resultSeverity"))
        focus = results_graph.value(result, URIRef(shacl + "focusNode"))
        path = results_graph.value(result, URIRef(shacl + "resultPath"))
        message = results_graph.value(result, URIRef(shacl + "resultMessage"))
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.SHACL,
                severity=(
                    "error" if str(severity_iri) == shacl + "Violation" else "warning"
                ),
                message=str(message) if message else "SHACL constraint violated.",
                subject=str(focus) if focus else "",
                predicate=str(path) if path else "",
            )
        )
    return findings


def collect_shacl_shapes(
    ontology_graph: RDFGraph | None, shapes_dir: str | None
) -> RDFGraph | None:
    """Assemble the SHACL shapes graph for the validation gate.

    Sources: every ``.ttl`` file under ``shapes_dir`` (when configured), plus
    the ontology context itself when it already carries ``sh:NodeShape``
    declarations inline — the zero-config path for catalogs that ship shapes
    next to their schema.
    """
    shapes = RDFGraph()
    if shapes_dir:
        directory = Path(shapes_dir)
        if not directory.is_dir():
            # Configuring a shapes directory that does not exist must not read
            # as "validated cleanly": glob() on a missing path yields nothing.
            logger.warning(
                "FACTS_SHAPES_DIR points at %s, which is not a directory; "
                "no SHACL shapes loaded",
                shapes_dir,
            )
        else:
            files = sorted(directory.glob("**/*.ttl"))
            if not files:
                logger.warning(
                    "FACTS_SHAPES_DIR %s contains no .ttl shape files", shapes_dir
                )
            for path in files:
                try:
                    shapes.parse(path.as_posix(), format="turtle")
                except Exception as error:
                    logger.warning("Failed to parse shapes file %s: %s", path, error)
    node_shape = URIRef("http://www.w3.org/ns/shacl#NodeShape")
    if ontology_graph is not None and (None, RDF.type, node_shape) in ontology_graph:
        shapes += ontology_graph
    return shapes if len(shapes) else None


# Scaffolding every facts graph uses regardless of catalog: flagging rdfs:label
# or rdf:type as "not in the ontology context" would bury the signal in noise.
_SCAFFOLDING_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(XSD))


def _non_catalog_vocabulary_findings(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    namespaces: list[str],
    quantity_fallback_vocabulary: dict[str, str] | None = None,
) -> list[FactsValidationFinding]:
    """Report terms the facts graph uses that the ontology context never supplied.

    When the renderer cannot find a term for something the text states, the
    prompt's documented fallback is to reach for a well-known vocabulary
    instead. That is the right behaviour -- the alternative is dropping the
    fact -- but it is silent, and it looks identical to success: a clean graph
    scoring in the nineties, expressed in vocabulary the catalog does not
    contain and downstream queries will not match. A fallback firing is
    evidence of a *retrieval* miss, so it is worth a finding even though the
    facts graph itself is well-formed.

    Warning severity: this is telemetry about context assembly, not the
    signature of a bad merge, and must not drive the un-merge repair.

    Args:
        graph: Aggregated facts graph.
        ontology_graph: Merged ontology context that was offered to the renderer.
        namespaces: Fact namespaces; terms minted there are the extraction, not
            borrowed vocabulary.

    Returns:
        list: One finding per non-catalog term, ordered by IRI.
    """
    # A pure graph-vs-graph check: with nothing to compare against there is no
    # per-term finding to make. The *deployment* condition this used to mask --
    # extraction proceeding with no catalog vocabulary at all -- is reported by
    # the validate node, which knows whether an empty context was expected.
    if ontology_graph is None or not len(ontology_graph):
        return []

    fallback_namespaces = _fallback_namespaces(graph, quantity_fallback_vocabulary)
    # Must match UNKNOWN_TERM's notion of "in the catalog" (collect_catalog_terms,
    # all three triple positions). A subjects()-only view calls a term supplied
    # solely as an rdfs:range object non-catalog -- exactly the terms the
    # domain/range schema closure is designed to admit.
    supplied = collect_catalog_terms(ontology_graph)
    used: dict[str, set[str]] = {}
    for subject, predicate, obj in graph:
        if not isinstance(subject, URIRef) or str(predicate).startswith(str(PROV)):
            continue
        if predicate == RDF.type:
            if isinstance(obj, URIRef):
                used.setdefault(str(obj), set()).add(str(subject))
            continue
        used.setdefault(str(predicate), set()).add(str(subject))

    findings: list[FactsValidationFinding] = []
    for term in sorted(used):
        if term in supplied:
            continue
        if term.startswith(_SCAFFOLDING_NAMESPACES):
            continue
        if any(term.startswith(ns) for ns in namespaces):
            continue
        subjects = sorted(used[term])
        # A term the deployment itself named as the fallback is a configured
        # choice the prompt instructed. Still a retrieval miss worth reporting,
        # but not the same signal as vocabulary invented out of nowhere -- the
        # message must not accuse the renderer of improvising what it was told.
        configured = _is_configured_fallback(term, fallback_namespaces)
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.NON_CATALOG_VOCABULARY,
                severity="warning",
                message=(
                    f"<{term}> is used by {len(subjects)} subject(s) but was "
                    "absent from the ontology context — the renderer took the "
                    "configured fallback vocabulary, which points at a "
                    "retrieval miss rather than a modeling choice."
                    if configured
                    else f"<{term}> is used by {len(subjects)} subject(s) but "
                    "was absent from the ontology context — the renderer fell "
                    "back to outside vocabulary, which points at a retrieval "
                    "miss rather than a modeling choice."
                ),
                subject=subjects[0],
                predicate=term,
                values=subjects,
            )
        )
    return findings


def _fallback_namespaces(
    graph: RDFGraph, fallback_vocabulary: dict[str, str] | None
) -> tuple[str, ...]:
    """Resolve configured fallback terms to namespace prefixes.

    Configured terms are CURIEs (``qudt:QuantityValue``) or full IRIs. CURIEs
    are expanded against the graph's own prefix bindings, which is where the
    renderer declared them.

    Args:
        graph: Facts graph, used for its prefix bindings.
        fallback_vocabulary: Configured role -> term mapping.

    Returns:
        tuple: Namespace IRIs, usable with ``str.startswith``.
    """
    if not fallback_vocabulary:
        return ()
    bindings = {prefix: str(uri) for prefix, uri in graph.namespaces()}
    namespaces: set[str] = set()
    for term in fallback_vocabulary.values():
        if not term:
            continue
        if term.startswith("http://") or term.startswith("https://"):
            namespaces.add(_namespace_of(term))
            continue
        prefix, separator, _ = term.partition(":")
        if separator and prefix in bindings:
            namespaces.add(bindings[prefix])
    return tuple(sorted(namespaces))


def _is_configured_fallback(term: str, fallback_namespaces: tuple[str, ...]) -> bool:
    """True when *term* lives in a namespace the deployment configured."""
    return bool(fallback_namespaces) and term.startswith(fallback_namespaces)


def validate_aggregated_facts(
    graph: RDFGraph,
    ontology_graph: RDFGraph | None,
    *,
    shapes_graph: RDFGraph | None = None,
    fact_namespaces: list[str] | None = None,
    suspect_multi_value_severity: str = "error",
    functional_min_single_support: int = 3,
    quantity_fallback_vocabulary: dict[str, str] | None = None,
) -> FactsValidationReport:
    """Check post-merge invariants over the aggregated facts graph.

    Deterministic defense-in-depth behind the merge guards: violations here
    are almost always the signature of a bad identity merge, and
    error-severity findings on merged subjects drive the un-merge repair.

    Checks:
        - ``FUNCTIONAL_VIOLATION``: >= 2 distinct objects on a predicate the
          schema constrains to at most one value (``owl:FunctionalProperty``
          or an OWL max-cardinality-1 restriction).
        - ``SUSPECT_MULTI_VALUE``: >= 2 distinct canonical numeric values on
          one (subject, predicate); or >= 2 IRI objects on a predicate that is
          single-valued for a dominant majority of other subjects. Severity is
          configurable — legitimate multi-value modeling exists, bad merges
          are far more common.
        - ``DEGENERATE_COREFERENCE``: one IRI object shared by >= 2 distinct
          functional-ish predicates of one subject (collapsed range bounds).
        - ``SHACL``: optional, when ``pyshacl`` is installed and shapes exist.
        - ``NON_CATALOG_VOCABULARY``: warning-only telemetry for terms the
          ontology context never supplied, which mark a retrieval miss the
          renderer papered over with a documented fallback.

    Args:
        graph: Aggregated facts graph.
        ontology_graph: Merged ontology context (functionality harvest).
        shapes_graph: Optional SHACL shapes graph.
        fact_namespaces: When set, only subjects under these namespaces are
            reported (ontology entities are not the gate's business).
        suspect_multi_value_severity: ``"error"`` or ``"warning"`` for
            SUSPECT_MULTI_VALUE findings.
        functional_min_single_support: Minimum single-valued subjects before a
            predicate counts as dominantly single-valued.

    Returns:
        Report with all findings, ordered by subject.
    """
    namespaces = [ns for ns in (fact_namespaces or []) if ns]
    functional = harvest_max_one_predicates(ontology_graph)

    # Provenance machinery (chunk nodes, derivation annotations) is
    # legitimately multi-valued and never the gate's business.
    provenance_subjects = {
        subject
        for subject in graph.subjects(RDF.type, PROV.Entity)
        if isinstance(subject, URIRef)
    }

    object_groups: dict[tuple[URIRef, URIRef], set] = {}
    iri_groups: dict[tuple[URIRef, URIRef], set[URIRef]] = {}
    for subject, predicate, obj in graph:
        if (
            not isinstance(subject, URIRef)
            or not isinstance(predicate, URIRef)
            or predicate == RDF.type
            # owl:sameAs is the aggregator's own merge bookkeeping: the rewriter
            # emits `canonical owl:sameAs original` per remapped entity, so it
            # carries 1 object for an unmerged entity and N-1 for a cluster.
            # Left in, it reads as "dominantly single-valued" and every large
            # cluster becomes an error that drives the repair to un-merge it.
            or predicate == OWL.sameAs
            or subject in provenance_subjects
            or str(predicate).startswith(str(PROV))
        ):
            continue
        object_groups.setdefault((subject, predicate), set()).add(obj)
        if isinstance(obj, URIRef):
            iri_groups.setdefault((subject, predicate), set()).add(obj)

    dominant_single = _dominant_single_valued_predicates(
        iri_groups, min_single_support=functional_min_single_support
    )
    functional_ish = functional | dominant_single

    findings: list[FactsValidationFinding] = []
    flagged_pairs: set[tuple[URIRef, URIRef]] = set()

    for (subject, predicate), objects in sorted(
        object_groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        if not _in_fact_scope(subject, namespaces):
            continue
        distinct = _distinct_object_keys(objects)
        if predicate in functional and len(distinct) >= 2:
            flagged_pairs.add((subject, predicate))
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.FUNCTIONAL_VIOLATION,
                    message=(
                        f"<{subject}> holds {len(distinct)} distinct values for "
                        f"<{predicate}>, which the schema constrains to at most "
                        "one."
                    ),
                    subject=str(subject),
                    predicate=str(predicate),
                    values=sorted(str(obj) for obj in objects),
                )
            )
            continue

        numeric_values = {
            canonical[0]
            for obj in objects
            if isinstance(obj, Literal)
            and (canonical := canonical_literal(obj)) is not None
            and canonical[1] == "numeric"
        }
        if len(numeric_values) >= 2:
            flagged_pairs.add((subject, predicate))
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
                    severity=(
                        "error"
                        if suspect_multi_value_severity == "error"
                        else "warning"
                    ),
                    message=(
                        f"<{subject}> carries {len(numeric_values)} distinct "
                        f"numeric values on <{predicate}> — the signature of "
                        "distinct quantities collapsed into one node."
                    ),
                    subject=str(subject),
                    predicate=str(predicate),
                    values=sorted(numeric_values),
                )
            )
            continue

        iri_objects = iri_groups.get((subject, predicate), set())
        if (
            len(iri_objects) >= 2
            and predicate not in functional
            and predicate in dominant_single
        ):
            flagged_pairs.add((subject, predicate))
            findings.append(
                FactsValidationFinding(
                    kind=FactsValidationFindingKind.SUSPECT_MULTI_VALUE,
                    severity=(
                        "error"
                        if suspect_multi_value_severity == "error"
                        else "warning"
                    ),
                    message=(
                        f"<{subject}> points at {len(iri_objects)} objects via "
                        f"<{predicate}>, which is single-valued for every other "
                        "subject in this graph."
                    ),
                    subject=str(subject),
                    predicate=str(predicate),
                    values=sorted(str(obj) for obj in iri_objects),
                )
            )

    coreference: dict[tuple[URIRef, URIRef], set[URIRef]] = {}
    for (subject, predicate), objects in iri_groups.items():
        if predicate not in functional_ish or not _in_fact_scope(subject, namespaces):
            continue
        for obj in objects:
            coreference.setdefault((subject, obj), set()).add(predicate)
    for (subject, obj), predicates in sorted(
        coreference.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        if len(predicates) < 2:
            continue
        findings.append(
            FactsValidationFinding(
                kind=FactsValidationFindingKind.DEGENERATE_COREFERENCE,
                message=(
                    f"<{subject}> reaches <{obj}> through "
                    f"{len(predicates)} single-valued predicates "
                    f"({', '.join(sorted(str(p) for p in predicates))}) — "
                    "distinct endpoints (e.g. range bounds) likely merged."
                ),
                subject=str(subject),
                predicate=", ".join(sorted(str(p) for p in predicates)),
                values=[str(obj)],
            )
        )

    if shapes_graph is not None and len(shapes_graph):
        findings.extend(
            finding
            for finding in _shacl_findings(graph, shapes_graph)
            if not finding.subject
            or _in_fact_scope(URIRef(finding.subject), namespaces)
        )

    findings.extend(
        _non_catalog_vocabulary_findings(
            graph, ontology_graph, namespaces, quantity_fallback_vocabulary
        )
    )

    return FactsValidationReport(findings=findings)


def format_findings_for_prompt(findings: list[FactsUnitFinding]) -> str:
    """Render findings as MANDATORY-fixes + coverage blocks for the renderer."""
    mandatory = [finding for finding in findings if finding.mandatory]
    coverage = [finding for finding in findings if not finding.mandatory]
    sections: list[str] = []
    if mandatory:
        lines = ["## MANDATORY fixes (deterministic validation — apply every item)"]
        for index, finding in enumerate(mandatory, 1):
            line = f"{index}. {finding.message}"
            if finding.suggestions:
                line += " Candidates: " + ", ".join(
                    f"<{suggestion}>" for suggestion in finding.suggestions
                )
            lines.append(line)
        sections.append("\n".join(lines))
    if coverage:
        lines = ["## Verify numeric coverage"]
        lines.extend(finding.message for finding in coverage)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
