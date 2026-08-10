"""Seed-free graph pruning shared by induced-subgraph retrieval and prompt condensing.

These pruners and predicate vocabularies were written for the vector-retrieval
induced-subgraph builder in :mod:`ontocast.tool.sparql`, which is the only place
in the pipeline that ever bounded how much ontology reached the LLM. The prompt
condenser (:mod:`ontocast.onto.ontology_condense`) needs the same judgements
about which triples carry the schema and which are scaffolding, so they live
here rather than being duplicated: two copies of "what is safe to drop" would
drift, and the drift would be invisible until an extraction quietly lost a term.

Everything here operates on a materialized graph with no seed list, no relevance
scores and no triple store, which is what makes it reusable outside retrieval.
"""

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

#: RDF list plumbing and ontology-header bookkeeping. Disconnected from any
#: business entity, so dropping these is as close to lossless as pruning gets.
NOISY_EXPANSION_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDF.first,
        RDF.rest,
        OWL.imports,
        OWL.versionIRI,
        OWL.versionInfo,
        OWL.priorVersion,
        OWL.backwardCompatibleWith,
        OWL.incompatibleWith,
        DCTERMS.creator,
        DCTERMS.license,
        DCTERMS.created,
        DCTERMS.modified,
        DCTERMS.identifier,
        DCTERMS.publisher,
        DCTERMS.contributor,
    }
)

#: Types that say nothing beyond "this is a term", once a real type is present.
GENERIC_INDIVIDUAL_TYPES: frozenset[URIRef] = frozenset(
    {OWL.NamedIndividual, OWL.Class, RDFS.Class}
)

OWL_RESTRICTION_MEANINGFUL_PREDICATES: frozenset[URIRef] = frozenset(
    {
        OWL.onProperty,
        OWL.someValuesFrom,
        OWL.allValuesFrom,
        OWL.hasValue,
        OWL.cardinality,
        OWL.minCardinality,
        OWL.maxCardinality,
        OWL.qualifiedCardinality,
        OWL.minQualifiedCardinality,
        OWL.maxQualifiedCardinality,
        OWL.onDataRange,
        OWL.onClass,
        OWL.oneOf,
    }
)

#: Order in which triples are admitted when a budget cannot hold all of them --
#: and, read backwards, the order in which they may be dropped. Ordering by
#: ``str(triple)`` instead made the surviving facts effectively alphabetical: a
#: term could arrive labelled but unplaced in the hierarchy, or placed but
#: unnamed. Glosses rank last because a term the model cannot name is useless,
#: while a term it cannot read a comment about is merely harder to use.
BFS_PREDICATE_PRIORITY: tuple[frozenset[URIRef], ...] = (
    frozenset({RDFS.label, SKOS.prefLabel}),  # name it
    frozenset({RDF.type}),  # say what it is
    frozenset({RDFS.subClassOf, OWL.equivalentClass}),  # place it in the hierarchy
    frozenset({RDFS.domain, RDFS.range, RDFS.subPropertyOf}),  # connect properties
    frozenset(
        {RDFS.comment, SKOS.definition, SKOS.scopeNote, SKOS.altLabel}
    ),  # describe it (scope notes carry usage contracts)
)

#: Below this many meaningful predicates a restriction bnode is a stub that
#: states nothing, and its ``subClassOf`` edge points at noise.
MIN_MEANINGFUL_RESTRICTION_PREDICATES = 2


def bfs_triple_rank(triple: tuple) -> tuple[int, str]:
    """Sort key admitting defining triples before incidental ones, ties lexicographic."""
    predicate = triple[1]
    for rank, predicates in enumerate(BFS_PREDICATE_PRIORITY):
        if predicate in predicates:
            return (rank, str(triple))
    return (len(BFS_PREDICATE_PRIORITY), str(triple))


def count_meaningful_restriction_predicates(graph: Graph, bnode: BNode) -> int:
    """How many predicates of a restriction bnode actually constrain anything."""
    return sum(
        1
        for _, pred, _ in graph.triples((bnode, None, None))
        if pred in OWL_RESTRICTION_MEANINGFUL_PREDICATES
    )


def remove_bnode_subgraph(graph: Graph, bnode: BNode) -> None:
    """Remove every triple asserted about a blank node."""
    for triple in list(graph.triples((bnode, None, None))):
        graph.remove(triple)


def remove_subclassof_to_bnode(graph: Graph, bnode: BNode) -> None:
    """Remove the class-axiom edges pointing at a blank node."""
    for triple in list(graph.triples((None, RDFS.subClassOf, bnode))):
        graph.remove(triple)
    for triple in list(graph.triples((None, OWL.equivalentClass, bnode))):
        graph.remove(triple)


def prune_orphaned_bnode_subjects(graph: Graph) -> None:
    """Remove blank-node subjects that no triple in the graph references as object."""
    bnode_as_object: set[BNode] = {o for _, _, o in graph if isinstance(o, BNode)}
    for triple in list(graph):
        subj, _, _ = triple
        if isinstance(subj, BNode) and subj not in bnode_as_object:
            graph.remove(triple)


def strip_redundant_generic_types(graph: Graph) -> None:
    """Drop generic rdf:types when the subject has informative types or URI hierarchy."""
    for subj, pred, obj in list(graph):
        if pred != RDF.type or obj not in GENERIC_INDIVIDUAL_TYPES:
            continue
        other_types = [
            term
            for _, _, term in graph.triples((subj, RDF.type, None))
            if term not in GENERIC_INDIVIDUAL_TYPES
        ]
        has_subclass_uri = any(
            isinstance(parent, URIRef)
            for _, _, parent in graph.triples((subj, RDFS.subClassOf, None))
        )
        if other_types or has_subclass_uri:
            graph.remove((subj, pred, obj))


def prune_degenerate_restriction_bnodes(result: Graph) -> int:
    """Remove stub restriction blank nodes and subClassOf edges pointing to them."""
    dropped = 0
    bnode_objects = sorted(
        {
            obj
            for _, _, obj in result.triples((None, RDFS.subClassOf, None))
            if isinstance(obj, BNode)
        },
        key=str,
    )
    for bnode in bnode_objects:
        if (
            count_meaningful_restriction_predicates(result, bnode)
            >= MIN_MEANINGFUL_RESTRICTION_PREDICATES
        ):
            continue
        remove_subclassof_to_bnode(result, bnode)
        remove_bnode_subgraph(result, bnode)
        dropped += 1
    return dropped
