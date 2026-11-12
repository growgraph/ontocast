"""Catalog term inventory, namespace closure rules, and alias candidates.

What the catalog *declares* versus merely *references* decides which
namespaces the term checks may treat as closed; ``ValidationPolicy`` carries
the deployment-level exemptions every check honours.
"""

from __future__ import annotations

import logging
import re
import typing
from collections.abc import Sequence
from difflib import SequenceMatcher

from pydantic import BaseModel
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef
from rdflib.namespace import DCTERMS, XSD
from rdflib.term import Node

from ontocast.onto.rdfgraph import (
    RDFGraph,
)
from ontocast.onto.scaffolding import (
    document_nodes,
    is_scaffolding,
    scaffolding_nodes,
)
from ontocast.onto.well_known import well_known_namespaces, well_known_value_classes

logger = logging.getLogger(__name__)

_FORBIDDEN_NAMESPACES = ("http://example.org/", "https://example.org/")


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


def collect_declared_namespaces(ontology_graph: RDFGraph | None) -> set[str]:
    """Namespaces the catalog *declares* terms in (subject-position IRIs).

    The UNKNOWN_TERM check treats a namespace as closed — flagging members the
    catalog does not list — only when the catalog actually declares terms
    there. A namespace the catalog merely *references* (``qudt:QuantityValue``
    in an ``rdfs:subClassOf``, ``qudt:unit`` in an ``owl:onProperty``) is an
    external vocabulary the catalog borrows from, and the catalog is not an
    authority on its membership. Treating referenced-only namespaces as closed
    produced mandatory findings against canonical external properties
    (``qudt:numericValue``), which repair renders then obeyed by deleting
    correct data.
    """
    namespaces: set[str] = set()
    if ontology_graph is None:
        return namespaces
    for subject in ontology_graph.subjects():
        if isinstance(subject, URIRef):
            namespaces.add(_namespace_of(str(subject)))
    return namespaces


def expand_vocabulary_terms(
    vocabulary: dict[str, str] | None,
    *graphs: RDFGraph | None,
) -> set[str]:
    """Expand configured vocabulary terms (CURIEs or full IRIs) to IRI strings.

    CURIEs are expanded against the prefix bindings of every graph given, in
    order; a CURIE whose prefix no graph binds is dropped rather than guessed.
    """
    terms: set[str] = set()
    if not vocabulary:
        return terms
    bindings: dict[str, str] = {}
    for graph in graphs:
        if graph is None:
            continue
        for prefix, uri in graph.namespaces():
            if prefix:
                bindings.setdefault(prefix, str(uri))
    for term in vocabulary.values():
        if not term:
            continue
        if term.startswith("http://") or term.startswith("https://"):
            terms.add(term)
            continue
        prefix, separator, local = term.partition(":")
        if separator and local and prefix in bindings:
            terms.add(bindings[prefix] + local)
    return terms


class ValidationPolicy(BaseModel):
    """Deployment-level exemptions and vocabulary for deterministic validation.

    One object instead of a parameter per concern: the namespaces a deployment
    shares across catalogs, the sanctioned quantity fallback vocabulary, and
    the code predicates — everything the term checks must never flag, because
    configuration explicitly blessed it.
    """

    additional_standard_namespaces: tuple[str, ...] = ()
    #: ``strict`` closes the vocabulary to the catalog, the vocabularies it
    #: borrows from, the well-known vocabularies and the terms named here:
    #: any other class or property is UNKNOWN_TERM. ``open`` lets any other
    #: vocabulary pass, so only a member of a closed namespace is unknown.
    vocabulary: typing.Literal["strict", "open"] = "strict"
    quantity_fallback_vocabulary: dict[str, str] | None = None
    code_predicates: tuple[str, ...] = ()
    #: IRIs the shapes-derived conformance chapter instructs the renderer to
    #: emit. Same class of exemption as the fallback vocabulary: the prompt
    #: recommends these terms, so flagging them UNKNOWN_TERM would have the
    #: repair lane delete exactly what the prompt required. Threaded per unit
    #: from the tenancy's shapes catalog.
    contract_exempt_terms: tuple[str, ...] = ()
    #: Keep identifier digit groups out of the numeric-coverage inventory.
    #: Same class as the exemptions above -- something configuration says the
    #: deterministic checks must not put in front of the renderer.
    numeric_identifier_guard: bool = False
    #: Floor on the share of schema terms a render must take from the
    #: catalog before DOMAIN_ADHERENCE fires. 0 disables the check.
    domain_adherence_min_share: float = 0.15
    #: Fewest distinct schema terms a render must use before its catalog
    #: share is judged at all. A share over a handful of terms is noise: a
    #: front-matter unit typing an identifier and an author with generic
    #: vocabulary is not a render that abandoned the catalog.
    domain_adherence_min_terms: int = 4

    def standard_namespaces(self) -> tuple[str, ...]:
        """Namespaces no catalog owns: the meta-vocabularies, the well-known
        vocabularies (:mod:`ontocast.onto.well_known`) and the configured
        ones. A term of a well-known vocabulary must still exist in it."""
        return (
            *_STANDARD_NAMESPACES,
            *well_known_namespaces(),
            *self.additional_standard_namespaces,
        )

    def exempt_terms(self, *graphs: RDFGraph | None) -> set[str]:
        """Exact IRIs configuration blessed: fallback vocabulary, code
        predicates, and the shapes-contract terms."""
        terms = expand_vocabulary_terms(self.quantity_fallback_vocabulary, *graphs)
        terms.update(self.code_predicates)
        terms.update(self.contract_exempt_terms)
        return terms


_PROPERTY_TYPES = (OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty)


def _catalog_term_roles(
    ontology_graph: RDFGraph | None,
) -> tuple[set[str], set[str]]:
    """Split catalog terms into (known properties, known classes).

    Only what the catalog states is used: explicit property/class typing,
    ``rdfs:domain``/``rdfs:range`` (property evidence), and
    ``rdfs:subClassOf`` / ``rdf:type``-object position (class evidence). Terms
    with no evidence land in neither set and pass every role filter.
    """
    properties: set[str] = set()
    classes: set[str] = set()
    if ontology_graph is None:
        return properties, classes
    for property_type in _PROPERTY_TYPES:
        for subject in ontology_graph.subjects(RDF.type, property_type):
            if isinstance(subject, URIRef):
                properties.add(str(subject))
    for subject in ontology_graph.subjects(RDF.type, RDF.Property):
        if isinstance(subject, URIRef):
            properties.add(str(subject))
    for predicate in (RDFS.domain, RDFS.range):
        for subject in ontology_graph.subjects(predicate, None):
            if isinstance(subject, URIRef):
                properties.add(str(subject))
    for class_type in (OWL.Class, RDFS.Class):
        for subject in ontology_graph.subjects(RDF.type, class_type):
            if isinstance(subject, URIRef):
                classes.add(str(subject))
    for subject, obj in ontology_graph.subject_objects(RDFS.subClassOf):
        for term in (subject, obj):
            if isinstance(term, URIRef):
                classes.add(str(term))
    for obj in ontology_graph.objects(None, RDF.type):
        if isinstance(obj, URIRef):
            classes.add(str(obj))
    # A term the catalog types both ways is contradictory evidence; trust
    # neither and let it through every filter.
    ambiguous = properties & classes
    return properties - ambiguous, classes - ambiguous


def _alias_candidates(
    alias: URIRef,
    graph: RDFGraph,
    catalog_terms: set[str],
    *,
    ontology_graph: RDFGraph | None = None,
    position: str = "predicate",
) -> list[str]:
    """Rank replacement candidates for a near-miss predicate.

    Candidate pool: catalog terms sharing the alias namespace, plus predicates
    of that namespace the graph itself uses on >= 2 subjects (the renderer's
    own dominant usage defines the alias target — this is how
    ``qudt:value`` resolves to ``qudt:numericValue`` even when the snapshot
    does not spell the property out).

    Candidates are role-filtered against the catalog's own declarations: a
    term in ``predicate`` position never gets a known class suggested, and a
    term in ``type`` position never gets a known property. Suggesting
    ``qudt:QuantityValue`` (a class) as the replacement for the property
    ``qudt:numericValue`` produced repair renders that wrote the class as a
    predicate — or deleted the statement outright.
    """
    namespace = _namespace_of(str(alias))
    pool: set[str] = {
        term for term in catalog_terms if _namespace_of(term) == namespace
    }
    known_properties, known_classes = _catalog_term_roles(ontology_graph)
    if position == "predicate":
        pool -= known_classes
    elif position == "type":
        pool -= known_properties
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


def _vocabulary_role_subset(
    vocabulary: dict[str, str] | None, role_fragment: str
) -> dict[str, str]:
    """Entries of a role -> term vocabulary whose role names the fragment."""
    if not vocabulary:
        return {}
    return {role: term for role, term in vocabulary.items() if role_fragment in role}


def build_surface_index(
    ontology_graph: RDFGraph | None,
    code_predicates: Sequence[str] = (),
) -> dict[str, set[str]]:
    """Map exact catalog surface forms to the IRIs declaring them.

    Case-sensitive and exact: these are codes and names a model may have
    transcribed verbatim (``"d"``, ``"meV"``, ``"CsPbBr3"``), not free text to
    be fuzzy-matched. A form claimed by more than one IRI stays in the index and
    is rejected at lookup time — an ambiguous code is not a repairable one.

    Args:
        ontology_graph: Merged ontology context to index.
        code_predicates: Extra code-bearing predicates (UCUM codes, symbols,
            notations) on top of the standard name predicates.

    Returns:
        Surface form -> set of IRIs declaring it.
    """
    index: dict[str, set[str]] = {}
    if ontology_graph is None:
        return index
    predicates: list[URIRef] = [RDFS.label, SKOS.prefLabel, SKOS.notation]
    predicates.extend(URIRef(predicate) for predicate in code_predicates)
    for predicate in predicates:
        for subject, value in ontology_graph.subject_objects(predicate):
            if not isinstance(subject, URIRef) or not isinstance(value, Literal):
                continue
            text = str(value).strip()
            if text:
                index.setdefault(text, set()).add(str(subject))
    return index


def resolve_unique_surface(index: dict[str, set[str]], text: str) -> URIRef | None:
    """The single IRI declaring ``text`` as a surface form, if exactly one does."""
    candidates = index.get(text.strip(), set())
    if len(candidates) != 1:
        return None
    return URIRef(next(iter(candidates)))


def superclass_closure(class_iri: URIRef, ontology_graph: RDFGraph) -> set[URIRef]:
    """Asserted supertypes of *class_iri*, including itself.

    Walks ``rdfs:subClassOf`` and steps through ``owl:equivalentClass``
    intersections, because a class defined only as an intersection
    (``LeadHalidePerovskite ≡ Perovskite ⊓ …``) has no asserted subClassOf edge
    to its genus and would otherwise look unrelated to it.
    """
    seen: set[URIRef] = set()
    frontier: list[Node] = [class_iri]
    while frontier:
        current = frontier.pop()
        if not isinstance(current, URIRef) or current in seen:
            continue
        seen.add(current)
        frontier.extend(ontology_graph.objects(current, RDFS.subClassOf))
        for equivalent in ontology_graph.objects(current, OWL.equivalentClass):
            for collection in ontology_graph.objects(equivalent, OWL.intersectionOf):
                frontier.extend(ontology_graph.items(collection))
    return seen


#: Statements that place a class in the hierarchy; the catalog-wide hierarchy
#: graph keeps only these, so a closure over it equals one over the catalog.
_HIERARCHY_PREDICATES = (
    RDFS.subClassOf,
    OWL.equivalentClass,
    OWL.intersectionOf,
    RDF.first,
    RDF.rest,
)


class CatalogRoles:
    """What the whole catalog says each term is, read once per catalog state.

    A unit sees a retrieved snapshot; the role of a term (property or class)
    and a class's place in the hierarchy are read from every held ontology, so
    a term the snapshot omits is still known for what it is.

    Attributes:
        properties: Terms the catalog states are properties.
        classes: Terms the catalog states are classes.
        hierarchy: The catalog's subclass statements, for closures.
        datatype_ranges: Each property's declared ``rdfs:range`` IRIs in the
            XSD namespace, across every graph; more than one means the catalog
            contradicts itself.
    """

    def __init__(
        self,
        properties: frozenset[str] = frozenset(),
        classes: frozenset[str] = frozenset(),
        hierarchy: RDFGraph | None = None,
        datatype_ranges: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self.properties = properties
        self.classes = classes
        self.hierarchy = hierarchy if hierarchy is not None else RDFGraph()
        self.datatype_ranges = datatype_ranges or {}
        self._value_classes: dict[frozenset[str], frozenset[str]] = {}

    @classmethod
    def of(cls, graphs: typing.Iterable[RDFGraph]) -> "CatalogRoles":
        """The roles and hierarchy stated across ``graphs``."""
        properties: set[str] = set()
        classes: set[str] = set()
        hierarchy = RDFGraph()
        ranges: dict[str, set[str]] = {}
        for graph in graphs:
            known_properties, known_classes = _catalog_term_roles(graph)
            properties |= known_properties
            classes |= known_classes
            for predicate in _HIERARCHY_PREDICATES:
                for triple in graph.triples((None, predicate, None)):
                    hierarchy.add(triple)
            for subject, range_iri in graph.subject_objects(RDFS.range):
                if isinstance(subject, URIRef) and str(range_iri).startswith(str(XSD)):
                    ranges.setdefault(str(subject), set()).add(str(range_iri))
        # A term typed both ways across files is contradictory; trust neither.
        ambiguous = properties & classes
        return cls(
            frozenset(properties - ambiguous),
            frozenset(classes - ambiguous),
            hierarchy,
            {iri: frozenset(found) for iri, found in ranges.items()},
        )

    def value_classes(self, base: frozenset[str]) -> frozenset[str]:
        """``base`` and every catalog class whose supertypes meet it.

        Args:
            base: Classes of structured values: the well-known ones and the
                deployment's configured value class.
        """
        found = self._value_classes.get(base)
        if found is None:
            candidates = self.classes | {
                str(s) for s in self.hierarchy.subjects(RDFS.subClassOf, None)
            }
            found = frozenset(base) | {
                iri
                for iri in candidates
                if {str(c) for c in superclass_closure(URIRef(iri), self.hierarchy)}
                & base
            }
            self._value_classes[base] = found
        return found


def value_class_base(
    configured: dict[str, str] | None, bindings: dict[str, str]
) -> frozenset[str]:
    """The well-known value classes plus the deployment's configured one.

    Args:
        configured: ``FACTS_QUANTITY_FALLBACK_VOCABULARY``; its ``value_class``
            role, a CURIE or an IRI, names the deployment's value class.
        bindings: Prefix -> namespace, to expand a CURIE.
    """
    base = set(well_known_value_classes())
    term = (configured or {}).get("value_class", "")
    if term.startswith(("http://", "https://")):
        base.add(term)
    elif term:
        prefix, _, local = term.partition(":")
        if local and prefix in bindings:
            base.add(bindings[prefix] + local)
    return frozenset(base)


def is_value_node(graph: RDFGraph, node: Node, value_classes: frozenset[str]) -> bool:
    """Whether a type of ``node`` is a value class (see :meth:`CatalogRoles.value_classes`)."""
    return any(str(t) in value_classes for t in graph.objects(node, RDF.type))


def _described_classes(ontology_graph: RDFGraph) -> set[URIRef]:
    """Classes whose position in the hierarchy the context actually states.

    A class named only as the object of an ``rdfs:domain`` or an ``rdf:type``
    carries no subclass edges here -- typically because it belongs to an
    imported vocabulary (SOSA, QUDT) that the context does not vendor, since
    ``owl:imports`` is not dereferenced. Its ancestors are unknown, so nothing
    can be concluded about whether it relates to another class.
    """
    described: set[URIRef] = set()
    for subject in ontology_graph.subjects(RDF.type, OWL.Class):
        if isinstance(subject, URIRef):
            described.add(subject)
    for subject in ontology_graph.subjects(RDFS.subClassOf, None):
        if isinstance(subject, URIRef):
            described.add(subject)
    return described


def _declared_domains(ontology_graph: RDFGraph) -> dict[URIRef, set[URIRef]]:
    """Predicate -> its declared ``rdfs:domain`` classes, named classes only."""
    domains: dict[URIRef, set[URIRef]] = {}
    for predicate, _, domain in ontology_graph.triples((None, RDFS.domain, None)):
        # Anonymous domains are class expressions (unions, restrictions); they
        # need a reasoner to decide membership, so they are out of scope here.
        if isinstance(predicate, URIRef) and isinstance(domain, URIRef):
            domains.setdefault(predicate, set()).add(domain)
    return domains


def declared_class_ranges(ontology_graph: RDFGraph) -> dict[URIRef, set[URIRef]]:
    """Predicate -> its declared ``rdfs:range`` classes, named classes only.

    A datatype range (XSD, ``rdfs:Literal``, a declared ``rdfs:Datatype``)
    types a literal, not a node, and is left out.
    """
    datatypes = {
        subject
        for subject in ontology_graph.subjects(RDF.type, RDFS.Datatype)
        if isinstance(subject, URIRef)
    }
    ranges: dict[URIRef, set[URIRef]] = {}
    for predicate, _, range_iri in ontology_graph.triples((None, RDFS.range, None)):
        if not isinstance(predicate, URIRef) or not isinstance(range_iri, URIRef):
            continue
        if range_iri in datatypes or range_iri in (RDFS.Literal, RDF.langString):
            continue
        if str(range_iri).startswith(str(XSD)):
            continue
        ranges.setdefault(predicate, set()).add(range_iri)
    return ranges


def types_compatible(
    asserted: set[URIRef],
    declared: set[URIRef],
    ontology_graph: RDFGraph,
    *,
    described: set[URIRef],
    closures: dict[URIRef, set[URIRef]],
) -> bool | None:
    """Whether a node typed ``asserted`` fits classes ``declared`` (a domain or range).

    Compatible in either direction: an asserted type specializes a declared
    class, or a declared class specializes an asserted type. ``None`` when the
    context cannot decide: no declared class it places in a hierarchy
    (``described``), or an asserted type it does not describe.

    Args:
        asserted: The node's asserted types.
        declared: The declared domain or range classes.
        ontology_graph: The ontology context.
        described: :func:`_described_classes` of ``ontology_graph``.
        closures: Cache of :func:`superclass_closure` results, filled in place.
    """
    declared = {value for value in declared if value in described}
    if not declared or not asserted or not asserted <= described:
        return None

    def closure(class_iri: URIRef) -> set[URIRef]:
        if class_iri not in closures:
            closures[class_iri] = superclass_closure(class_iri, ontology_graph)
        return closures[class_iri]

    declared_closure = set().union(*(closure(value) for value in declared))
    return any(
        closure(asserted_type) & declared or asserted_type in declared_closure
        for asserted_type in asserted
    )


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


def _in_fact_scope(subject: URIRef, fact_namespaces: list[str]) -> bool:
    if not fact_namespaces:
        return True
    text = str(subject)
    return any(text.startswith(namespace) for namespace in fact_namespaces if namespace)


# Classes that type nothing: every resource is one of these, so an instance
# whose only ``rdf:type`` is here is untyped for the facts contract.
_VACUOUS_TYPES: frozenset[URIRef] = frozenset(
    {
        OWL.Thing,
        RDFS.Resource,
        URIRef("http://schema.org/Thing"),
        URIRef("https://schema.org/Thing"),
    }
)

# Schema metaclasses: a fact-namespace node typed only as one of these is a
# class or property the renderer tried to declare, not a typed instance.
_SCHEMA_METACLASSES: frozenset[URIRef] = frozenset(
    {
        RDFS.Class,
        OWL.Class,
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.NamedIndividual,
    }
)

# Objects of these predicates are alias handles, not instances.
_HANDLE_PREDICATES: frozenset[URIRef] = frozenset({OWL.sameAs})


def fact_instances(graph: RDFGraph, fact_namespaces: Sequence[str]) -> set[URIRef]:
    """The instances a facts graph asserts.

    An instance is a ``URIRef`` in one of ``fact_namespaces`` that occurs as a
    subject, or as the IRI object of a triple other than ``rdf:type``,
    ``owl:sameAs``. Blank nodes are never instances, and neither are the
    nodes the pipeline mints: chunks, reifiers and the document node
    (:mod:`ontocast.onto.scaffolding`). Triples touching a chunk or reifier
    are skipped, whatever their vocabulary.

    Args:
        graph: Facts graph for one unit or the aggregated document.
        fact_namespaces: Namespaces holding minted instances.

    Returns:
        set[URIRef]: The instances, in no particular order.
    """
    namespaces = [ns for ns in fact_namespaces if ns]
    if not namespaces:
        return set()
    scaffolding = scaffolding_nodes(graph)
    minted = scaffolding | document_nodes(graph)
    instances: set[URIRef] = set()
    for subject, predicate, obj in graph:
        if is_scaffolding((subject, predicate, obj), scaffolding):
            continue
        if isinstance(subject, URIRef) and _in_fact_scope(subject, namespaces):
            instances.add(subject)
        if (
            isinstance(obj, URIRef)
            and predicate != RDF.type
            and predicate not in _HANDLE_PREDICATES
            and _in_fact_scope(obj, namespaces)
        ):
            instances.add(obj)
    return instances - minted


def concrete_types(
    graph: RDFGraph, node: Node, fact_namespaces: Sequence[str]
) -> set[URIRef]:
    """The asserted ``rdf:type`` objects of ``node`` that count as a type.

    A type counts when it is an IRI outside ``fact_namespaces`` (a class
    minted beside the instances is no class), not a schema metaclass
    (``rdfs:Class``, ``owl:Class``, ``rdf:Property``, the OWL property
    classes, ``owl:NamedIndividual``) and not vacuous (``owl:Thing``,
    ``rdfs:Resource``, ``schema:Thing``).
    """
    namespaces = [ns for ns in fact_namespaces if ns]
    return {
        value
        for value in graph.objects(node, RDF.type)
        if isinstance(value, URIRef)
        and value not in _VACUOUS_TYPES
        and value not in _SCHEMA_METACLASSES
        and not (namespaces and _in_fact_scope(value, namespaces))
    }


def untyped_instances(graph: RDFGraph, fact_namespaces: Sequence[str]) -> list[URIRef]:
    """Instances of ``graph`` (:func:`fact_instances`) with no concrete type.

    Returns:
        list[URIRef]: The untyped instances, ordered by IRI.
    """
    return sorted(
        (
            instance
            for instance in fact_instances(graph, fact_namespaces)
            if not concrete_types(graph, instance, fact_namespaces)
        ),
        key=str,
    )
