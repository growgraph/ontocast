"""Which subjects of a facts graph are instances."""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from rdflib import OWL, RDF, RDFS, Graph, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope
from ontocast.tool.facts_validation.terms import catalog_term_evidence

_DECLARING_TYPES = (
    OWL.Class,
    RDFS.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    RDF.Property,
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
)


def declared_terms(graph: Graph) -> set[str]:
    """Return IRIs the graph itself declares as a class or property."""
    return {
        str(subject)
        for declaring in _DECLARING_TYPES
        for subject in graph.subjects(RDF.type, declaring)
        if isinstance(subject, URIRef)
    }


def catalog_schema_terms(catalog: RDFGraph | None) -> set[str]:
    """Return the IRIs the catalog declares as a class or property.

    Evidence is explicit typing, a domain or range, ``rdfs:subClassOf``,
    ``rdfs:subPropertyOf`` and ``rdf:type``-object position. The catalog's
    individuals (``ex:Paris a ex:City``, SKOS concepts, units) are not terms:
    statements about them are facts.
    """
    if catalog is None:
        return set()
    properties, classes = catalog_term_evidence(catalog)
    terms = properties | classes | declared_terms(catalog)
    for subject, obj in catalog.subject_objects(RDFS.subPropertyOf):
        terms.update(str(term) for term in (subject, obj) if isinstance(term, URIRef))
    return terms


def instances(
    graph: Graph,
    *,
    catalog_terms: AbstractSet[str],
    standard_namespaces: tuple[str, ...],
    restrict_to: Sequence[str] | None = None,
) -> set[URIRef]:
    """Return the IRI subjects of `graph` that are instances.

    The fact subjects of `graph` (:attr:`~ontocast.onto.scaffolding.FactScope.subjects`:
    no scaffolding, no document node) that are not a catalog term (a class or
    property, :func:`catalog_schema_terms`), not declared in the graph, and
    not under a standard or well-known namespace. A node the graph only
    points at is not counted. With `restrict_to`, only IRIs under those
    namespaces are kept.
    """
    declared = declared_terms(graph)
    return {
        subject
        for subject in FactScope.of(graph, restrict_to).subjects
        if str(subject) not in catalog_terms
        and str(subject) not in declared
        and not str(subject).startswith(standard_namespaces)
    }
