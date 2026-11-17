"""Which subjects of a facts graph are instances."""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from rdflib import OWL, RDF, RDFS, Graph, URIRef

from ontocast.onto.rdfgraph import RDFGraph
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

    An instance is not a catalog term (a class or property,
    :func:`catalog_schema_terms`), not declared in the graph, and not under
    a standard or well-known namespace. Blank-node and literal subjects are
    ignored. With `restrict_to`, only IRIs under those namespaces are kept.
    """
    declared = declared_terms(graph)
    result: set[URIRef] = set()
    for subject in set(graph.subjects()):
        if not isinstance(subject, URIRef):
            continue
        iri = str(subject)
        if iri in catalog_terms or iri in declared:
            continue
        if iri.startswith(standard_namespaces):
            continue
        if restrict_to is not None and not iri.startswith(tuple(restrict_to)):
            continue
        result.add(subject)
    return result
