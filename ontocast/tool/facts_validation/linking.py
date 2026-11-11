"""What the link pass shows the model, and what it accepts back.

After the gate, an instance can hold a type and a value yet link to no other
instance: the text relates it to something, but the unit that extracted it
missed the relation, or the other end sits in another unit. The link pass
shows each such instance with the passage that mentions it, a bounded list of
instances it may link to, and the object properties that could fit; the
model proposes links the passage states. Nothing it says is trusted:
:func:`validate_link_proposals` keeps a proposal only when its ends are
listed, one is isolated, its property is a listed object property whose
domain and range admit the ends, a functional property is still free, and its
quoted evidence occurs in the source text.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.term import Node

from ontocast.onto.annotation import is_annotation
from ontocast.onto.constants import PROV
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.enum import FactsLinkRejection
from ontocast.onto.model import FactsLinkProposal, FactsLinkRecord
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_role
from ontocast.prompt.term_sheet import partition_terms, qname_for
from ontocast.tool.agg.coreference import source_excerpt
from ontocast.tool.agg.signatures import (
    harvest_max_one_predicates,
    normalize_string_value,
)
from ontocast.tool.facts_validation.connectivity import instance_links
from ontocast.tool.facts_validation.terms import (
    _declared_domains,
    _described_classes,
    declared_class_ranges,
    fact_instances,
    superclass_closure,
    types_compatible,
)

#: Targets offered per isolated instance, and in all.
_MAX_TARGETS_PER_ISOLATED = 8
_MAX_TARGETS = 60
#: The document's best-linked instances offered to every isolated one.
_HUBS = 3
_MAX_PROPERTIES = 40
_MAX_PASSAGE = 600
#: Shortest label or value that locates a passage or a target in it.
_MIN_NEEDLE = 3

_CLASS_TYPES = frozenset({OWL.Class, RDFS.Class})
_NON_OBJECT_PROPERTY_TYPES = frozenset({OWL.DatatypeProperty, OWL.AnnotationProperty})
#: Well-known classes of structured values. A value belongs to its carrier
#: through a property made for it, so a link may reach one only through a
#: property whose declared domain or range is shown to fit it.
_VALUE_CLASSES: frozenset[URIRef] = frozenset(
    {
        URIRef(namespace + name)
        for namespace in ("https://schema.org/", "http://schema.org/")
        for name in (
            "StructuredValue",
            "QuantitativeValue",
            "PropertyValue",
            "MonetaryAmount",
        )
    }
    | {URIRef("http://qudt.org/schema/qudt/QuantityValue")}
)


class LinkCandidates(BaseModel):
    """What one link pass shows: entities numbered isolated first, then targets.

    Attributes:
        isolated: Isolated instance IRIs, numbered 1..k.
        targets: Target instance IRIs, numbered k+1..n.
        properties: Property name as listed -> its IRI, in listing order.
        passages: Isolated IRI -> the passage shown for it.
        sources: Isolated IRI -> the IRIs of its source units, by unit index.
    """

    isolated: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)
    passages: dict[str, str] = Field(default_factory=dict)
    sources: dict[str, list[str]] = Field(default_factory=dict)

    def entities(self) -> list[str]:
        """Every listed entity, in numbering order (entity n is index n-1)."""
        return [*self.isolated, *self.targets]


def statement_sources(graph: RDFGraph) -> dict[Node, set[URIRef]]:
    """Node -> the units its statements derive from, read from the reifiers."""
    sources: dict[Node, set[URIRef]] = {}
    for reifier, (subject, _, obj) in graph.reifications:
        units = {
            unit
            for unit in graph.objects(reifier, PROV.wasDerivedFrom)
            if isinstance(unit, URIRef)
        }
        if not units:
            continue
        sources.setdefault(subject, set()).update(units)
        sources.setdefault(obj, set()).update(units)
    return sources


def is_object_property(predicate: URIRef, ontology_graph: RDFGraph) -> bool:
    """Whether ``predicate`` can link two instances.

    False for typing, aliasing and annotation predicates, a term the context
    types as a datatype or annotation property or as a class, a property whose
    declared range is a datatype, and a well-known class or datatype.
    """
    if predicate in (RDF.type, OWL.sameAs) or is_annotation(predicate):
        return False
    types = set(ontology_graph.objects(predicate, RDF.type))
    if types & _NON_OBJECT_PROPERTY_TYPES or types & _CLASS_TYPES:
        return False
    datatypes = set(ontology_graph.subjects(RDF.type, RDFS.Datatype))
    for range_iri in ontology_graph.objects(predicate, RDFS.range):
        if (
            range_iri in datatypes
            or range_iri in (RDFS.Literal, RDF.langString)
            or str(range_iri).startswith(str(XSD))
        ):
            return False
    return well_known_role(str(predicate)) not in ("class", "datatype")


def _labels(graph: RDFGraph, node: URIRef) -> list[str]:
    return sorted(
        str(label) for label in graph.objects(node, RDFS.label) if str(label).strip()
    )


def _types(graph: RDFGraph, node: URIRef) -> set[URIRef]:
    return {t for t in graph.objects(node, RDF.type) if isinstance(t, URIRef)}


def _needles(graph: RDFGraph, node: URIRef) -> list[str]:
    """Labels and literal values of ``node`` long enough to find in text."""
    values = [
        str(obj)
        for predicate, obj in graph.predicate_objects(node)
        if isinstance(obj, Literal) and not is_annotation(predicate)
    ]
    return sorted(
        {text for text in [*_labels(graph, node), *values] if len(text) >= _MIN_NEEDLE}
    )


def select_link_candidates(
    graph: RDFGraph,
    isolated: Sequence[str],
    units: Sequence[ContentUnit],
    ontology_graph: RDFGraph,
    fact_namespaces: Sequence[str],
    *,
    max_isolated: int,
    excluded_predicates: Collection[str] = (),
) -> LinkCandidates | None:
    """Choose what the link pass shows; ``None`` when nothing can be linked.

    An isolated instance is shown when a source unit outside the citation
    metadata names it, with the sentences that do (earliest unit first, at
    most ``max_isolated``). Its targets, at most eight, are the labelled
    instances its passage names, then those of its units by how linked they
    are, then the document's best-linked ones. The properties are the
    context's object properties whose domain and range admit some shown pair
    with an isolated end, and the linking predicates the document already
    uses, except ``excluded_predicates`` (vocabulary the gate refused).

    Args:
        graph: The served facts graph, after the gate.
        isolated: Instances the gate found isolated.
        units: The document's content units.
        ontology_graph: The facts ontology context.
        fact_namespaces: Namespaces holding minted instances.
        max_isolated: Most isolated instances shown.
        excluded_predicates: Predicates never offered.
    """
    sources = statement_sources(graph)
    unit_by_iri = {unit.iri_absolute: unit for unit in units}

    def units_of(node: URIRef) -> list[ContentUnit]:
        found = [
            unit_by_iri[str(iri)]
            for iri in sources.get(node, set())
            if str(iri) in unit_by_iri
        ]
        return sorted(
            (unit for unit in found if not unit.is_citation_metadata),
            key=lambda unit: unit.index,
        )

    shown: list[tuple[int, str, str, list[str]]] = []
    for iri in sorted(set(isolated)):
        node = URIRef(iri)
        node_units = units_of(node)
        if not node_units:
            continue
        needles = _needles(graph, node)
        excerpts = [
            excerpt
            for unit in node_units
            if (excerpt := source_excerpt(unit.text, needles, _MAX_PASSAGE))
        ]
        if not excerpts:
            continue
        passage = " … ".join(excerpts)
        if len(passage) > _MAX_PASSAGE:
            passage = f"{passage[: _MAX_PASSAGE - 1]}…"
        shown.append(
            (
                node_units[0].index,
                iri,
                passage,
                [unit.iri_absolute for unit in node_units],
            )
        )
    shown.sort(key=lambda entry: (entry[0], entry[1]))
    shown = shown[:max_isolated]
    if not shown:
        return None
    isolated_iris = [iri for _, iri, _, _ in shown]
    isolated_set = set(isolated_iris)

    degree: Counter[str] = Counter()
    for subject, _, obj in instance_links(graph, fact_namespaces):
        degree[str(subject)] += 1
        degree[str(obj)] += 1
    pool = sorted(
        str(node)
        for node in fact_instances(graph, fact_namespaces)
        if _labels(graph, node) and _types(graph, node)
    )
    pool_labels = {
        iri: [
            normalize_string_value(label)
            for label in _labels(graph, URIRef(iri))
            if len(label) >= _MIN_NEEDLE
        ]
        for iri in pool
    }
    by_degree = sorted(pool, key=lambda iri: (-degree[iri], iri))
    hubs = [iri for iri in by_degree if degree[iri] > 0][:_HUBS]

    targets: list[str] = []
    for _, iri, passage, unit_iris in shown:
        flat = normalize_string_value(passage)
        named = sorted(
            (
                (min(flat.find(label) for label in labels if label in flat), other)
                for other, labels in pool_labels.items()
                if any(label and label in flat for label in labels)
            ),
        )
        same_unit = [
            other
            for other in by_degree
            if any(str(unit) in unit_iris for unit in sources.get(URIRef(other), set()))
        ]
        chosen: list[str] = []
        for other in [*(other for _, other in named), *same_unit, *hubs]:
            if len(chosen) >= _MAX_TARGETS_PER_ISOLATED:
                break
            if other == iri or other in isolated_set or other in chosen:
                continue
            chosen.append(other)
        for other in chosen:
            if other not in targets and len(targets) < _MAX_TARGETS:
                targets.append(other)

    properties = _select_properties(
        graph,
        ontology_graph,
        fact_namespaces,
        isolated_iris,
        targets,
        excluded=set(excluded_predicates),
    )
    if not properties:
        return None
    return LinkCandidates(
        isolated=isolated_iris,
        targets=targets,
        properties=properties,
        passages={iri: passage for _, iri, passage, _ in shown},
        sources={iri: unit_iris for _, iri, _, unit_iris in shown},
    )


def _select_properties(
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    fact_namespaces: Sequence[str],
    isolated: Sequence[str],
    targets: Sequence[str],
    *,
    excluded: set[str],
) -> dict[str, str]:
    """Listed name -> IRI of the properties offered, best fitting first."""
    _, context_properties, _ = partition_terms(ontology_graph)
    used = {predicate for _, predicate, _ in instance_links(graph, fact_namespaces)}
    candidates = sorted(
        {
            predicate
            for predicate in [*context_properties, *used]
            if isinstance(predicate, URIRef)
            and str(predicate) not in excluded
            and is_object_property(predicate, ontology_graph)
        },
        key=str,
    )
    domains = _declared_domains(ontology_graph)
    ranges = declared_class_ranges(ontology_graph)
    described = _described_classes(ontology_graph)
    closures: dict[URIRef, set[URIRef]] = {}
    entities = [*isolated, *targets]
    isolated_set = set(isolated)
    types = {iri: _types(graph, URIRef(iri)) for iri in entities}

    def fit(iri: str, declared: set[URIRef] | None) -> bool | None:
        if not declared:
            return None
        return types_compatible(
            types[iri],
            declared,
            ontology_graph,
            described=described,
            closures=closures,
        )

    def top_two(scores: dict[str, int], pool: Sequence[str]) -> list[tuple[int, str]]:
        return sorted(((scores[iri], iri) for iri in pool), reverse=True)[:2]

    def best_other(top: list[tuple[int, str]], subject: str) -> int:
        return next((score for score, iri in top if iri != subject), -1)

    scored: list[tuple[int, str, URIRef]] = []
    for predicate in candidates:
        domain_fit = {iri: fit(iri, domains.get(predicate)) for iri in entities}
        range_fit = {iri: fit(iri, ranges.get(predicate)) for iri in entities}
        # A fitting end scores 1, an undecidable one 0; a misfit is excluded.
        objects = {iri: int(bool(range_fit[iri])) for iri in entities}
        allowed = [iri for iri in entities if range_fit[iri] is not False]
        top_any = top_two(objects, allowed)
        top_isolated = top_two(objects, [iri for iri in allowed if iri in isolated_set])
        best = -1
        for subject in entities:
            if domain_fit[subject] is False:
                continue
            # One end must be isolated: any object for an isolated subject.
            top = top_any if subject in isolated_set else top_isolated
            other = best_other(top, subject)
            if other >= 0:
                best = max(best, int(bool(domain_fit[subject])) + other)
        if best >= 0:
            scored.append(
                (best, _listed_name(graph, ontology_graph, predicate), predicate)
            )
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return {name: str(predicate) for _, name, predicate in scored[:_MAX_PROPERTIES]}


def _local_name(name: str) -> str:
    """The part of a prefixed name or IRI after its last ``:``, ``#`` or ``/``."""
    for separator in ("#", "/", ":"):
        name = name.rsplit(separator, 1)[-1]
    return name


def _listed_name(graph: RDFGraph, ontology_graph: RDFGraph, predicate: URIRef) -> str:
    """The property's prefixed name in the context, else in the facts, else its IRI."""
    for source in (ontology_graph, graph):
        name = qname_for(source, predicate)
        if name != str(predicate):
            return name
    return str(predicate)


def validate_link_proposals(
    proposals: Sequence[FactsLinkProposal],
    candidates: LinkCandidates,
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    units: Sequence[ContentUnit],
) -> tuple[list[FactsLinkRecord], Counter[FactsLinkRejection]]:
    """Keep the proposals every check admits; count the others by reason.

    The first failing check names the rejection: both ends listed, one end
    isolated, two different ends, a listed property (by its listed name, its
    IRI, or a bare local name only one listed property has), an object
    property, a domain and a range that admit the ends (undecidable counts as
    admitted), a structured value reached only through a domain or range shown
    to fit it, a functional property still free, evidence found in a source unit of an
    isolated end, and a link not already stated.

    Returns:
        The accepted links, in proposal order, and the rejection counts.
    """
    entities = candidates.entities()
    isolated = set(candidates.isolated)
    names = {name: iri for name, iri in candidates.properties.items()}
    names.update({iri: iri for iri in candidates.properties.values()})
    names.update({f"<{iri}>": iri for iri in candidates.properties.values()})
    domains = _declared_domains(ontology_graph)
    ranges = declared_class_ranges(ontology_graph)
    described = _described_classes(ontology_graph)
    closures: dict[URIRef, set[URIRef]] = {}
    functional = harvest_max_one_predicates(ontology_graph)
    unit_text = {unit.iri_absolute: normalize_string_value(unit.text) for unit in units}

    def fits(node: URIRef, declared: set[URIRef] | None) -> bool | None:
        if not declared:
            return None
        return types_compatible(
            _types(graph, node),
            declared,
            ontology_graph,
            described=described,
            closures=closures,
        )

    def is_value(node: URIRef) -> bool:
        for type_iri in _types(graph, node):
            if type_iri not in closures:
                closures[type_iri] = superclass_closure(type_iri, ontology_graph)
            if closures[type_iri] & _VALUE_CLASSES:
                return True
        return False

    local_names: dict[str, list[str]] = {}
    for iri in candidates.properties.values():
        local_names.setdefault(_local_name(iri), []).append(iri)

    def by_local_name(name: str) -> str | None:
        """A listed property named by its bare local name, when only one has it."""
        matches = local_names.get(_local_name(name.strip()), [])
        return matches[0] if len(matches) == 1 else None

    accepted: list[FactsLinkRecord] = []
    stated: set[tuple[str, str, str]] = set()
    taken: set[tuple[str, str]] = set()
    rejected: Counter[FactsLinkRejection] = Counter()

    for proposal in proposals:
        if not (
            1 <= proposal.subject <= len(entities)
            and 1 <= proposal.object <= len(entities)
        ):
            rejected[FactsLinkRejection.UNKNOWN_ENTITY] += 1
            continue
        subject = entities[proposal.subject - 1]
        obj = entities[proposal.object - 1]
        if subject not in isolated and obj not in isolated:
            rejected[FactsLinkRejection.NO_ISOLATED_ENDPOINT] += 1
            continue
        if subject == obj:
            rejected[FactsLinkRejection.SELF_LINK] += 1
            continue
        predicate_iri = names.get(proposal.property.strip()) or by_local_name(
            proposal.property
        )
        if predicate_iri is None:
            rejected[FactsLinkRejection.UNKNOWN_PROPERTY] += 1
            continue
        predicate = URIRef(predicate_iri)
        if not is_object_property(predicate, ontology_graph):
            rejected[FactsLinkRejection.NOT_OBJECT_PROPERTY] += 1
            continue
        subject_node, object_node = URIRef(subject), URIRef(obj)
        domain_fit = fits(subject_node, domains.get(predicate))
        if domain_fit is False:
            rejected[FactsLinkRejection.DOMAIN_MISMATCH] += 1
            continue
        range_fit = fits(object_node, ranges.get(predicate))
        if range_fit is False:
            rejected[FactsLinkRejection.RANGE_MISMATCH] += 1
            continue
        if (is_value(subject_node) and domain_fit is not True) or (
            is_value(object_node) and range_fit is not True
        ):
            rejected[FactsLinkRejection.VALUE_ENDPOINT] += 1
            continue
        if predicate in functional and (
            (subject, predicate_iri) in taken
            or any(True for _ in graph.objects(subject_node, predicate))
        ):
            rejected[FactsLinkRejection.FUNCTIONAL_TAKEN] += 1
            continue
        evidence = normalize_string_value(proposal.evidence)
        source_units = [
            unit_iri
            for end in (subject, obj)
            if end in isolated
            for unit_iri in candidates.sources.get(end, [])
        ]
        unit = next(
            (
                unit_iri
                for unit_iri in source_units
                if evidence and evidence in unit_text.get(unit_iri, "")
            ),
            None,
        )
        if unit is None:
            rejected[FactsLinkRejection.EVIDENCE_MISSING] += 1
            continue
        key = (subject, predicate_iri, obj)
        if key in stated or (subject_node, predicate, object_node) in graph:
            rejected[FactsLinkRejection.DUPLICATE] += 1
            continue
        stated.add(key)
        taken.add((subject, predicate_iri))
        accepted.append(
            FactsLinkRecord(
                subject=subject,
                predicate=predicate_iri,
                object=obj,
                unit=unit,
                evidence=proposal.evidence,
            )
        )
    return accepted, rejected


__all__ = [
    "LinkCandidates",
    "is_object_property",
    "select_link_candidates",
    "statement_sources",
    "validate_link_proposals",
]
