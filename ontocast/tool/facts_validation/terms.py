"""Catalog term inventory, namespace closure rules, and alias candidates.

What the catalog *declares* versus merely *references* decides which
namespaces the term checks may treat as closed; ``ValidationPolicy`` carries
the deployment-level exemptions every check honours.
"""

from __future__ import annotations

import logging
import re
import typing
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher

from pydantic import BaseModel
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef
from rdflib.namespace import DCTERMS, XSD
from rdflib.term import Node

from ontocast.onto.property_sides import (
    SIDE_PREDICATES,
    Side,
    SideSource,
    is_datatype_value,
    sides_by_property,
)
from ontocast.onto.rdfgraph import (
    RDFGraph,
)
from ontocast.onto.scaffolding import FactScope
from ontocast.onto.well_known import (
    StatementRoleName,
    well_known_anchor_properties,
    well_known_closed_ranges,
    well_known_datatype_ranges,
    well_known_member_classes,
    well_known_namespaces,
    well_known_statement_classes,
    well_known_statement_roles,
    well_known_value_classes,
)
from ontocast.tool.agg.signatures import NUMERIC_DATATYPES
from ontocast.tool.facts_validation.statements import RoleLinks, StatementContract

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


def _folded_local_name(iri: str) -> str:
    """The local name of ``iri`` with case and ``_``/``-`` separators folded away."""
    return re.sub(r"[_\-]", "", _local_name(iri)).lower()


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


def catalog_term_evidence(
    ontology_graph: RDFGraph | None,
) -> tuple[set[str], set[str]]:
    """Catalog terms with property evidence and with class evidence.

    Only what the catalog states is used: explicit property/class typing,
    a domain or a range (``rdfs`` or ``schema:*Includes``, property evidence), and
    ``rdfs:subClassOf`` / ``rdf:type``-object position (class evidence). A
    term may land in both sets; :func:`_catalog_term_roles` drops those.
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
    for predicate in SIDE_PREDICATES:
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
    return properties, classes


def _catalog_term_roles(
    ontology_graph: RDFGraph | None,
) -> tuple[set[str], set[str]]:
    """Split catalog terms into (known properties, known classes).

    Evidence as :func:`catalog_term_evidence` collects it. Terms with no
    evidence land in neither set and pass every role filter.
    """
    properties, classes = catalog_term_evidence(ontology_graph)
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
    roles: tuple[set[str], set[str]] | None = None,
) -> list[str]:
    """Rank replacement candidates for a near-miss predicate.

    ``roles`` is ``_catalog_term_roles(ontology_graph)`` computed once by a
    caller that ranks many aliases against one catalog.

    Candidate pool: catalog terms sharing the alias namespace, plus predicates
    of that namespace the graph itself uses on >= 2 subjects (the renderer's
    own dominant usage defines the alias target — this is how
    ``qudt:value`` resolves to ``qudt:numericValue`` even when the snapshot
    does not spell the property out), plus catalog terms of the alias's kind
    (a property in ``predicate`` position, a class in ``type`` position) in
    any namespace whose local name is the alias's once case and separators
    are folded. Those last are suggestions only: a repair that applies a
    candidate must keep to the alias's namespace.

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
    known_properties, known_classes = (
        roles if roles is not None else _catalog_term_roles(ontology_graph)
    )
    if position == "predicate":
        pool -= known_classes
    elif position == "type":
        pool -= known_properties
    same_kind = (
        known_properties
        if position == "predicate"
        else known_classes
        if position == "type"
        else set()
    )
    folded = _folded_local_name(str(alias))
    if folded:
        pool.update(term for term in same_kind if _folded_local_name(term) == folded)
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


#: What a term's domain, range and listed classes say, kept catalog-wide so a
#: repair does not depend on which axioms retrieval kept for a unit.
_AXIOM_PREDICATES = tuple(sorted(SIDE_PREDICATES, key=str))
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
        axioms: The catalog's domains, ranges, listed classes and the types
            of its terms, with the hierarchy.
        datatype_ranges: Each property's declared ``rdfs:range`` IRIs in the
            XSD namespace, across every graph; more than one means the catalog
            contradicts itself.
        restrictions: Class -> the properties a restriction on it requires
            (``owl:someValuesFrom``, or a minimum or exact cardinality, plain
            or qualified, of at least one), read from its direct
            ``rdfs:subClassOf`` restrictions.
    """

    def __init__(
        self,
        properties: frozenset[str] = frozenset(),
        classes: frozenset[str] = frozenset(),
        hierarchy: RDFGraph | None = None,
        datatype_ranges: dict[str, frozenset[str]] | None = None,
        axioms: RDFGraph | None = None,
        restrictions: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self.properties = properties
        self.classes = classes
        self.hierarchy = hierarchy if hierarchy is not None else RDFGraph()
        self.axioms = axioms if axioms is not None else RDFGraph()
        self.datatype_ranges = datatype_ranges or {}
        self.restrictions = restrictions or {}
        self._declared_ranges: dict[str, frozenset[str]] | None = None
        self._role_links: dict[str, RoleLinks] = {}
        self._contracts: dict[frozenset[str], StatementContract] = {}
        self._classes_under: dict[frozenset[str], frozenset[str]] = {}

    @classmethod
    def of(cls, graphs: typing.Iterable[RDFGraph]) -> "CatalogRoles":
        """The roles and hierarchy stated across ``graphs``."""
        properties: set[str] = set()
        classes: set[str] = set()
        hierarchy = RDFGraph()
        axioms = RDFGraph()
        ranges: dict[str, set[str]] = {}
        restrictions: dict[str, set[str]] = {}
        for graph in graphs:
            for predicate in _AXIOM_PREDICATES:
                for triple in graph.triples((None, predicate, None)):
                    axioms.add(triple)
            # Every catalog term's own type: what is a class, a datatype, a
            # symmetric property, and which IRIs are the catalog's individuals,
            # which entailment must never type.
            for triple in graph.triples((None, RDF.type, None)):
                if isinstance(triple[0], URIRef):
                    axioms.add(triple)
            known_properties, known_classes = _catalog_term_roles(graph)
            properties |= known_properties
            classes |= known_classes
            for predicate in (
                *_HIERARCHY_PREDICATES,
                RDFS.subPropertyOf,
                OWL.inverseOf,
                OWL.oneOf,
            ):
                for triple in graph.triples((None, predicate, None)):
                    hierarchy.add(triple)
            for subject, range_iri in graph.subject_objects(RDFS.range):
                if isinstance(subject, URIRef) and str(range_iri).startswith(str(XSD)):
                    ranges.setdefault(str(subject), set()).add(str(range_iri))
            for cls_iri, prop in _required_restrictions(graph):
                restrictions.setdefault(cls_iri, set()).add(prop)
        # A term typed both ways across files is contradictory; trust neither.
        ambiguous = properties & classes
        return cls(
            frozenset(properties - ambiguous),
            frozenset(classes - ambiguous),
            hierarchy,
            {iri: frozenset(found) for iri, found in ranges.items()},
            axioms + hierarchy,
            {iri: frozenset(found) for iri, found in restrictions.items()},
        )

    def with_axioms(self, graph: RDFGraph) -> RDFGraph:
        """``graph`` together with the catalog's axioms, as a new graph.

        A unit's snapshot may keep a property's domain and drop its range, or a
        class without its place in the hierarchy; domain and range repairs read
        this union so the answer does not depend on what retrieval kept.
        """
        if not len(self.axioms):
            return graph
        return graph + self.axioms

    def declared_ranges(self) -> dict[str, frozenset[str]]:
        """Each property's declared XSD ranges: the catalog's and the well-known.

        A literal on the property fits when it is a value of one of them
        (:func:`~ontocast.tool.facts_validation.literal_repair.range_mismatches`).
        Read once per catalog state; a unit adds only its snapshot's.
        """
        if self._declared_ranges is None:
            merged: dict[str, set[str]] = {}
            for prop, datatypes in well_known_datatype_ranges().items():
                merged.setdefault(prop, set()).update(datatypes)
            for prop, datatypes in self.datatype_ranges.items():
                merged.setdefault(prop, set()).update(datatypes)
            self._declared_ranges = {
                prop: frozenset(found) for prop, found in merged.items()
            }
        return self._declared_ranges

    def role_links(self, role: StatementRoleName) -> RoleLinks:
        """The properties playing ``role`` for a statement, each direction.

        The well-known ones (:func:`~ontocast.onto.well_known.well_known_statement_roles`),
        every catalog sub-property of one, and every catalog inverse of one
        in the other direction, to a fixed point.
        """
        found = self._role_links.get(role)
        if found is None:
            seeds = [
                (iri, link.direction)
                for iri, link in well_known_statement_roles().items()
                if link.role == role
            ]
            forward = {iri for iri, direction in seeds if direction == "forward"}
            inverse = {iri for iri, direction in seeds if direction == "inverse"}
            changed = True
            while changed:
                before = len(forward) + len(inverse)
                for side, other in ((forward, inverse), (inverse, forward)):
                    for prop in list(side):
                        node = URIRef(prop)
                        side |= {
                            str(s)
                            for s in self.hierarchy.subjects(RDFS.subPropertyOf, node)
                            if isinstance(s, URIRef)
                        }
                        other |= {
                            str(term)
                            for term in (
                                *self.hierarchy.subjects(OWL.inverseOf, node),
                                *self.hierarchy.objects(node, OWL.inverseOf),
                            )
                            if isinstance(term, URIRef)
                        }
                changed = len(forward) + len(inverse) != before
            found = RoleLinks(forward=frozenset(forward), inverse=frozenset(inverse))
            self._role_links[role] = found
        return found

    def statement_contract(self, value_base: frozenset[str]) -> StatementContract:
        """The :class:`~ontocast.tool.facts_validation.statements.StatementContract`
        of this catalog, memoised per ``value_base``.

        Args:
            value_base: The value classes to anchor on (:func:`value_class_base`).
        """
        contract = self._contracts.get(value_base)
        if contract is None:
            contract = StatementContract(
                statement_classes=self._statement_classes(),
                required_links=self._required_links(),
                value_classes=self.classes_under(value_base),
                roles={
                    role: self.role_links(role)
                    for role in typing.get_args(StatementRoleName)
                },
                super_properties=self._super_properties(),
                closed_ranges=self._closed_ranges(),
                link_classes=self._link_classes(),
                anchor_links=self._anchor_links(),
                numeric_properties=frozenset(
                    prop
                    for prop, datatypes in self.declared_ranges().items()
                    if any(URIRef(dt) in NUMERIC_DATATYPES for dt in datatypes)
                ),
                property_classes=self._property_classes(),
            )
            self._contracts[value_base] = contract
        return contract

    def _link_classes(self) -> dict[str, frozenset[str]]:
        """Each role link with the statement classes its declared side admits.

        The side is the link's domain when it runs from the statement, its
        range when it runs to it: the catalog's own declaration
        (``rdfs:*`` or ``schema:*Includes``), else the well-known one, else
        that of its nearest super-property. A class is admitted when it is
        under a declared class or above one. A link with no declaration
        anywhere is left out: it admits every class.
        """
        statement_classes = self._statement_classes()
        seeds = well_known_statement_roles()
        sides = {
            "forward": sides_by_property(self.axioms, "domain"),
            "inverse": sides_by_property(self.axioms, "range"),
        }
        closures: dict[str, set[str]] = {}

        def closure(cls: str) -> set[str]:
            if cls not in closures:
                closures[cls] = {
                    str(c) for c in superclass_closure(URIRef(cls), self.hierarchy)
                }
            return closures[cls]

        def declared(prop: str, direction: str) -> set[str] | None:
            frontier = [prop]
            seen: set[str] = set()
            while frontier:
                current = frontier.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                own = sides[direction].get(URIRef(current))
                if own:
                    return {str(cls) for cls in own}
                seed = seeds.get(current)
                if seed is not None and seed.classes:
                    namespace = _namespace_of(current)
                    return {namespace + name for name in seed.classes}
                frontier.extend(
                    str(parent)
                    for parent in self.hierarchy.objects(
                        URIRef(current), RDFS.subPropertyOf
                    )
                    if isinstance(parent, URIRef)
                )
            return None

        found: dict[str, frozenset[str]] = {}
        for role in typing.get_args(StatementRoleName):
            links = self.role_links(role)
            for direction, props in (
                ("forward", links.forward),
                ("inverse", links.inverse),
            ):
                for prop in props:
                    side = declared(prop, direction)
                    if side is None:
                        continue
                    found[prop] = frozenset(
                        cls
                        for cls in statement_classes
                        if closure(cls) & side or any(cls in closure(d) for d in side)
                    )
        return found

    def _statement_classes(self) -> dict[str, frozenset[StatementRoleName]]:
        """Each statement class, catalog subclasses included, with its roles."""
        anchors = well_known_statement_classes()
        found: dict[str, set[StatementRoleName]] = {}
        for anchor, roles in anchors.items():
            for cls in self.classes_under(frozenset({anchor})):
                found.setdefault(cls, set()).update(roles)
        return {cls: frozenset(roles) for cls, roles in found.items()}

    def _required_links(self) -> dict[str, frozenset[str]]:
        """Each statement class with the properties its restrictions require."""
        found: dict[str, frozenset[str]] = {}
        for cls in self._statement_classes():
            required: set[str] = set()
            for ancestor in superclass_closure(URIRef(cls), self.hierarchy):
                required |= self.restrictions.get(str(ancestor), frozenset())
            if required:
                found[cls] = frozenset(required)
        return found

    def _property_classes(self) -> frozenset[str]:
        """The classes a statement's ``property`` link names, with those under them.

        The range each forward ``property`` link declares: the catalog's own,
        else the well-known one (:attr:`StatementRole.range_class`).
        """
        seeds = well_known_statement_roles()
        declared = declared_class_ranges(self.axioms)
        base: set[str] = set()
        for prop in self.role_links("property").forward:
            own = declared.get(URIRef(prop))
            if own:
                base |= {str(cls) for cls in own}
                continue
            seed = seeds.get(prop)
            if seed is not None and seed.range_class is not None:
                base.add(_namespace_of(prop) + seed.range_class)
        return self.classes_under(frozenset(base))

    def _anchor_links(self) -> frozenset[str]:
        """The well-known anchor properties and every catalog sub-property of one."""
        found = set(well_known_anchor_properties())
        frontier = list(found)
        while frontier:
            prop = frontier.pop()
            for sub in self.hierarchy.subjects(RDFS.subPropertyOf, URIRef(prop)):
                if isinstance(sub, URIRef) and str(sub) not in found:
                    found.add(str(sub))
                    frontier.append(str(sub))
        return frozenset(found)

    def _super_properties(self) -> dict[str, frozenset[str]]:
        """Each catalog sub-property with its transitive super-properties."""
        found: dict[str, frozenset[str]] = {}
        for prop in {
            s
            for s in self.hierarchy.subjects(RDFS.subPropertyOf, None)
            if isinstance(s, URIRef)
        }:
            seen: set[URIRef] = set()
            frontier = [prop]
            while frontier:
                current = frontier.pop()
                for parent in self.hierarchy.objects(current, RDFS.subPropertyOf):
                    if isinstance(parent, URIRef) and parent not in seen:
                        seen.add(parent)
                        frontier.append(parent)
            seen.discard(prop)
            if seen:
                found[str(prop)] = frozenset(str(p) for p in seen)
        return found

    def _closed_ranges(self) -> dict[str, frozenset[str]]:
        """Each property with the range classes whose members are known.

        A class is closed when the catalog names an individual of it or of a
        subclass, enumerates it (``owl:oneOf``), or a well-known vocabulary
        ships its term list. Ranges are the catalog's and the well-known
        vocabularies' published ones.
        """
        listed = set(well_known_member_classes())
        individuals = {
            s
            for s in self.axioms.subjects(RDF.type, None)
            if isinstance(s, URIRef)
            and str(s) not in self.classes
            and str(s) not in self.properties
        }
        typed = {
            str(cls)
            for individual in individuals
            for cls in self.axioms.objects(individual, RDF.type)
        }

        def closed(cls: str) -> bool:
            if cls in listed or (URIRef(cls), OWL.oneOf, None) in self.hierarchy:
                return True
            return bool(typed & self.classes_under(frozenset({cls})))

        found: dict[str, set[str]] = {
            prop: set(classes) for prop, classes in well_known_closed_ranges().items()
        }
        for prop, classes in declared_class_ranges(self.axioms).items():
            for cls in classes:
                if closed(str(cls)):
                    found.setdefault(str(prop), set()).add(str(cls))
        return {prop: frozenset(classes) for prop, classes in found.items()}

    def classes_under(self, base: frozenset[str]) -> frozenset[str]:
        """``base`` and every catalog class whose supertypes meet it.

        Args:
            base: Anchor classes, such as the classes of structured values
                (:func:`value_class_base`) or of statements.
        """
        found = self._classes_under.get(base)
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
            self._classes_under[base] = found
        return found


_REQUIRING_CARDINALITIES = (
    OWL.minCardinality,
    OWL.cardinality,
    OWL.minQualifiedCardinality,
    OWL.qualifiedCardinality,
)


def _required_restrictions(graph: RDFGraph) -> typing.Iterator[tuple[str, str]]:
    """``(class, property)`` for each direct restriction of a class requiring a link.

    A restriction requires one when it is ``owl:someValuesFrom``, or a minimum
    or exact cardinality (plain or qualified) of at least one. A restriction
    inside a union or an intersection requires nothing on its own.
    """
    for cls, restriction in graph.subject_objects(RDFS.subClassOf):
        if not isinstance(cls, URIRef):
            continue
        prop = graph.value(restriction, OWL.onProperty)
        if not isinstance(prop, URIRef):
            continue
        requires = (restriction, OWL.someValuesFrom, None) in graph or any(
            _at_least_one(graph.value(restriction, cardinality))
            for cardinality in _REQUIRING_CARDINALITIES
        )
        if requires:
            yield str(cls), str(prop)


def _at_least_one(value: Node | None) -> bool:
    if not isinstance(value, Literal):
        return False
    try:
        return int(str(value)) >= 1
    except ValueError:
        return False


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

    A class named only as the object of a domain or an ``rdf:type``
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


def _declared_domains(
    ontology_graph: RDFGraph, *, source: SideSource | None = None
) -> dict[URIRef, set[URIRef]]:
    """Predicate -> the named classes its domain names.

    Both ``rdfs:domain`` and ``schema:domainIncludes`` by default: the classes
    a subject is expected to fit. ``source="rdfs"`` keeps the entailing ones.
    Anonymous domains are class expressions (unions, restrictions); they need
    a reasoner to decide membership, so they are out of scope here.
    """
    return sides_by_property(ontology_graph, "domain", source=source)


def declared_class_ranges(
    ontology_graph: RDFGraph, *, source: SideSource | None = None
) -> dict[URIRef, set[URIRef]]:
    """Predicate -> the named classes its range names, datatypes left out.

    Both ``rdfs:range`` and ``schema:rangeIncludes`` by default;
    ``source="rdfs"`` keeps the entailing ones. A datatype range (XSD,
    ``rdfs:Literal``, a declared ``rdfs:Datatype``) types a literal, not a
    node, and is left out.
    """
    ranges: dict[URIRef, set[URIRef]] = {}
    for predicate, values in sides_by_property(
        ontology_graph, "range", source=source
    ).items():
        classes = {v for v in values if not is_datatype_value(v, ontology_graph)}
        if classes:
            ranges[predicate] = classes
    return ranges


def sole_listed_classes(
    ontology_graph: RDFGraph, side: Side, declared: Mapping[URIRef, set[URIRef]]
) -> dict[URIRef, URIRef]:
    """Predicate -> the one class its ``schema:<side>Includes`` lists.

    Listed classes are alternatives, so they entail nothing in general; a
    predicate listing exactly one named class, and declaring no ``rdfs``
    counterpart, leaves one reading.

    Args:
        ontology_graph: Ontology context.
        side: ``domain`` or ``range``.
        declared: The predicate's ``rdfs`` classes of that side; a predicate
            present here is left out.

    Returns:
        The sole listed class per predicate; datatypes are never a class.
    """
    listed = sides_by_property(ontology_graph, side, source="includes")
    return {
        predicate: next(iter(classes))
        for predicate, classes in listed.items()
        if len(classes) == 1
        and predicate not in declared
        and not is_datatype_value(next(iter(classes)), ontology_graph)
    }


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


def domain_fitting_properties(
    subject: URIRef,
    obj: Node,
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    *,
    replacing: URIRef,
    limit: int = 8,
) -> list[tuple[URIRef, set[URIRef]]]:
    """Context properties that may link ``subject`` to ``obj`` in place of ``replacing``.

    A candidate declares a domain (``rdfs:domain`` or ``schema:domainIncludes``)
    that the subject's asserted types fit, and no range the object's types
    contradict; a literal object takes only a property with no class range.
    Ranked by the local-name tokens shared with ``replacing``, then by IRI.

    Returns:
        Up to ``limit`` (property, its domain classes) pairs.
    """
    asserted = {t for t in graph.objects(subject, RDF.type) if isinstance(t, URIRef)}
    if not asserted:
        return []
    described = _described_classes(ontology_graph)
    closures: dict[URIRef, set[URIRef]] = {}
    domains = _declared_domains(ontology_graph)
    ranges = declared_class_ranges(ontology_graph)
    object_types = (
        {t for t in graph.objects(obj, RDF.type) if isinstance(t, URIRef)}
        if isinstance(obj, URIRef)
        else set()
    )
    wanted = _name_tokens(_local_name(str(replacing)))
    scored: list[tuple[int, str, URIRef, set[URIRef]]] = []
    for predicate, declared in domains.items():
        if predicate == replacing:
            continue
        fits = types_compatible(
            asserted, declared, ontology_graph, described=described, closures=closures
        )
        if fits is not True:
            continue
        declared_range = ranges.get(predicate)
        if isinstance(obj, Literal):
            if declared_range:
                continue
        elif declared_range and object_types:
            if (
                types_compatible(
                    object_types,
                    declared_range,
                    ontology_graph,
                    described=described,
                    closures=closures,
                )
                is False
            ):
                continue
        shared = len(wanted & _name_tokens(_local_name(str(predicate))))
        scored.append((-shared, str(predicate), predicate, declared))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(predicate, declared) for _, _, predicate, declared in scored[:limit]]


def link_properties(
    node: URIRef,
    graph: RDFGraph,
    ontology_graph: RDFGraph,
    *,
    limit: int = 8,
) -> list[tuple[URIRef, str]]:
    """Context object properties that may link ``node`` to another instance.

    A property qualifies with ``node`` as its subject when its declared domain
    (``rdfs:domain`` or ``schema:domainIncludes``) fits ``node``'s asserted
    types and it has a class range; as its object when its class range fits
    them. Ordered by IRI, subject direction first.

    Returns:
        Up to ``limit`` (property, ``"from"`` or ``"to"``) pairs: ``from``
        when ``node`` is the subject.
    """
    asserted = {t for t in graph.objects(node, RDF.type) if isinstance(t, URIRef)}
    if not asserted:
        return []
    described = _described_classes(ontology_graph)
    closures: dict[URIRef, set[URIRef]] = {}
    domains = _declared_domains(ontology_graph)
    ranges = declared_class_ranges(ontology_graph)

    def fits(declared: set[URIRef] | None) -> bool:
        return bool(declared) and (
            types_compatible(
                asserted,
                declared or set(),
                ontology_graph,
                described=described,
                closures=closures,
            )
            is True
        )

    found: list[tuple[URIRef, str]] = []
    for predicate in sorted(ranges, key=str):
        if fits(domains.get(predicate)):
            found.append((predicate, "from"))
    for predicate in sorted(ranges, key=str):
        if fits(ranges.get(predicate)):
            found.append((predicate, "to"))
    return found[:limit]


def domain_fits(
    subject: URIRef, predicate: URIRef, graph: RDFGraph, ontology_graph: RDFGraph
) -> bool:
    """Whether ``subject``'s asserted types contradict no ``rdfs:domain`` of ``predicate``.

    The test :func:`domain_violation_findings` applies: only ``rdfs:domain``
    entails, so only it can be contradicted; a listed class is an expected
    one.
    """
    declared = _declared_domains(ontology_graph, source="rdfs").get(predicate)
    if not declared:
        return True
    asserted = {t for t in graph.objects(subject, RDF.type) if isinstance(t, URIRef)}
    return (
        types_compatible(
            asserted,
            declared,
            ontology_graph,
            described=_described_classes(ontology_graph),
            closures={},
        )
        is not False
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
    """Instances of ``graph`` (:attr:`FactScope.instances`) with no concrete type.

    Returns:
        list[URIRef]: The untyped instances, ordered by IRI.
    """
    return sorted(
        (
            instance
            for instance in FactScope.of(graph, fact_namespaces).instances
            if not concrete_types(graph, instance, fact_namespaces)
        ),
        key=str,
    )
