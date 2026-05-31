"""SPARQL tool for incremental graph updates.

This module provides functionality for executing SPARQL operations on RDF graphs,
enabling incremental updates instead of full graph replacement.
"""

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Mapping

from rdflib import BNode, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS
from rdflib.plugins.sparql import prepareQuery

from ontocast.onto.enum import SPARQLOperationType
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.sparql_models import SPARQLOperationModel
from ontocast.tool.representation_text import ROLE_PREDICATE
from ontocast.tool.triple_manager.core import TripleStoreManager

logger = logging.getLogger(__name__)

# Predicates treated as human-facing descriptions for seed entities (always merged in first).
_SEED_DESCRIPTION_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDFS.label,
        RDFS.comment,
        SKOS.prefLabel,
        SKOS.altLabel,
        SKOS.definition,
        URIRef("http://purl.org/dc/terms/description"),
        URIRef("http://purl.org/dc/elements/1.1/description"),
    }
)

# RDF list expansion and ontology header predicates tend to introduce low-value
# triples that are disconnected from business entities.
_NOISY_EXPANSION_PREDICATES: frozenset[URIRef] = frozenset(
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

_PROPERTY_DEFINITION_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDF.type,
        RDFS.label,
        RDFS.comment,
        SKOS.definition,
        RDFS.domain,
        RDFS.range,
        RDFS.subPropertyOf,
        OWL.inverseOf,
        OWL.equivalentProperty,
    }
)

_CONNECTIVITY_PREDICATES: frozenset[URIRef] = frozenset(
    {RDFS.subClassOf, RDFS.domain, RDFS.range}
)

_CLASS_SCHEMA_PREDICATES: frozenset[URIRef] = frozenset(
    {RDFS.subClassOf, OWL.equivalentClass}
)

_GENERIC_INDIVIDUAL_TYPES: frozenset[URIRef] = frozenset(
    {OWL.NamedIndividual, OWL.Class, RDFS.Class}
)

_SCHEMA_URI_CONNECTIVITY_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDFS.subClassOf,
        RDFS.domain,
        RDFS.range,
        OWL.equivalentClass,
        OWL.equivalentProperty,
        RDFS.subPropertyOf,
    }
)

_OWL_RESTRICTION_MEANINGFUL_PREDICATES: frozenset[URIRef] = frozenset(
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

_OWL_RESTRICTION_SHELL_PREDICATES: frozenset[URIRef] = (
    _OWL_RESTRICTION_MEANINGFUL_PREDICATES | frozenset({RDF.type})
)

_RESTRICTION_SHELL_MAX_TRIPLES = 16
_SCHEMA_PATH_MAX_DEPTH = 4
_MIN_MEANINGFUL_RESTRICTION_PREDICATES = 2

_PROPERTY_TYPES: frozenset[URIRef] = frozenset(
    {OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property}
)


def _filter_overbroad_namespace_map(ns_map: dict[str, str]) -> dict[str, str]:
    """Drop namespace bindings whose URI is a strict prefix of another in the map."""
    all_ns_uris = set(ns_map.values())
    return {
        prefix: uri
        for prefix, uri in ns_map.items()
        if not any(other != uri and other.startswith(uri) for other in all_ns_uris)
    }


def _prune_orphaned_bnode_subjects(graph: RDFGraph) -> None:
    """Remove blank-node subjects that no triple in the graph references as object."""
    bnode_as_object: set[BNode] = {o for _, _, o in graph if isinstance(o, BNode)}
    for triple in list(graph):
        subj, _, _ = triple
        if isinstance(subj, BNode) and subj not in bnode_as_object:
            graph.remove(triple)


def _strip_redundant_generic_types(graph: RDFGraph) -> None:
    """Drop generic rdf:types when the subject has informative types or URI hierarchy."""
    for subj, pred, obj in list(graph):
        if pred != RDF.type or obj not in _GENERIC_INDIVIDUAL_TYPES:
            continue
        other_types = [
            term
            for _, _, term in graph.triples((subj, RDF.type, None))
            if term not in _GENERIC_INDIVIDUAL_TYPES
        ]
        has_subclass_uri = any(
            isinstance(parent, URIRef)
            for _, _, parent in graph.triples((subj, RDFS.subClassOf, None))
        )
        if other_types or has_subclass_uri:
            graph.remove((subj, pred, obj))


def _strip_redundant_named_individual_types(graph: RDFGraph) -> None:
    """Drop rdf:type owl:NamedIndividual when the subject has other informative types."""
    _strip_redundant_generic_types(graph)


def _promoted_type_iris(
    merged_graph: RDFGraph,
    ref: URIRef,
    ontology_subjects: frozenset[str],
) -> list[str]:
    return [
        str(type_iri)
        for _, _, type_iri in merged_graph.triples((ref, RDF.type, None))
        if isinstance(type_iri, URIRef)
        and type_iri not in _GENERIC_INDIVIDUAL_TYPES
        and str(type_iri) not in ontology_subjects
    ]


def _classify_and_promote_seeds(
    seed_uris_ranked: list[str],
    merged_graph: RDFGraph,
    entity_roles: Mapping[str, str | None],
    ontology_subjects: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Split retrieval seeds into concept (class) and property seeds; promote individuals."""
    concept_seeds: list[str] = []
    property_seeds: list[str] = []
    for uri in seed_uris_ranked:
        ref = URIRef(uri)
        role = entity_roles.get(uri)
        has_outgoing = any(merged_graph.triples((ref, None, None)))
        has_incoming = any(merged_graph.triples((None, None, ref)))
        if role == ROLE_PREDICATE or (has_incoming and not has_outgoing):
            property_seeds.append(uri)
            continue
        promoted = _promoted_type_iris(merged_graph, ref, ontology_subjects)
        if promoted:
            concept_seeds.extend(promoted)
        else:
            concept_seeds.append(uri)
    return list(dict.fromkeys(concept_seeds)), list(dict.fromkeys(property_seeds))


def _crosslink_property_seeds(
    merged_graph: RDFGraph,
    concept_seeds: list[str],
    property_seeds: list[str],
    ontology_subjects: frozenset[str],
) -> list[str]:
    """Add properties whose domain or range is a retrieved concept class."""
    linked = list(property_seeds)
    for class_uri in concept_seeds:
        ref = URIRef(class_uri)
        for prop, _, _ in merged_graph.triples((None, RDFS.domain, ref)):
            if isinstance(prop, URIRef) and str(prop) not in ontology_subjects:
                linked.append(str(prop))
        for prop, _, _ in merged_graph.triples((None, RDFS.range, ref)):
            if isinstance(prop, URIRef) and str(prop) not in ontology_subjects:
                linked.append(str(prop))
    return list(dict.fromkeys(linked))


def _build_concept_relevance(
    seed_uris_ranked: list[str],
    merged_graph: RDFGraph,
    relevance: dict[str, float],
    ontology_subjects: frozenset[str],
) -> dict[str, float]:
    """Map retrieval scores onto promoted class IRIs (max score per class)."""
    concept_relevance: dict[str, float] = {}
    for orig_uri in seed_uris_ranked:
        score = float(relevance.get(orig_uri, 0.0))
        ref = URIRef(orig_uri)
        promoted = _promoted_type_iris(merged_graph, ref, ontology_subjects)
        targets = promoted if promoted else [orig_uri]
        for target in targets:
            concept_relevance[target] = max(concept_relevance.get(target, 0.0), score)
    return concept_relevance


def _uri_entities_in_graph(graph: RDFGraph) -> set[URIRef]:
    entities: set[URIRef] = set()
    for subj, _, obj in graph:
        if isinstance(subj, URIRef):
            entities.add(subj)
        if isinstance(obj, URIRef):
            entities.add(obj)
    return entities


def _build_uri_adjacency(graph: RDFGraph) -> dict[URIRef, set[URIRef]]:
    adjacency: dict[URIRef, set[URIRef]] = defaultdict(set)
    for subj, _, obj in graph:
        if isinstance(subj, URIRef) and isinstance(obj, URIRef):
            adjacency[subj].add(obj)
            adjacency[obj].add(subj)
    return adjacency


def _find_uri_connected_components(
    graph: RDFGraph,
) -> list[set[URIRef]]:
    entities = _uri_entities_in_graph(graph)
    adjacency = _build_uri_adjacency(graph)
    visited: set[URIRef] = set()
    components: list[set[URIRef]] = []
    for entity in sorted(entities, key=str):
        if entity in visited:
            continue
        component: set[URIRef] = set()
        queue: deque[URIRef] = deque([entity])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if component:
            components.append(component)
    return components


def _build_schema_uri_adjacency(graph: RDFGraph) -> dict[URIRef, set[URIRef]]:
    adjacency: dict[URIRef, set[URIRef]] = defaultdict(set)
    for subj, pred, obj in graph:
        if pred not in _SCHEMA_URI_CONNECTIVITY_PREDICATES:
            continue
        if isinstance(subj, URIRef) and isinstance(obj, URIRef):
            adjacency[subj].add(obj)
            adjacency[obj].add(subj)
    return adjacency


_SCHEMA_VOCABULARY_URIS: frozenset[URIRef] = frozenset(
    {
        OWL.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.NamedIndividual,
        OWL.Restriction,
        OWL.Ontology,
        RDFS.Class,
        RDF.Property,
    }
)


def _schema_relevant_uri_entities(graph: RDFGraph) -> set[URIRef]:
    """URI nodes that participate in schema connectivity edges (excludes bare vocabulary IRIs)."""
    entities: set[URIRef] = set()
    for subj, pred, obj in graph:
        if pred not in _SCHEMA_URI_CONNECTIVITY_PREDICATES:
            continue
        if isinstance(subj, URIRef) and subj not in _SCHEMA_VOCABULARY_URIS:
            entities.add(subj)
        if isinstance(obj, URIRef) and obj not in _SCHEMA_VOCABULARY_URIS:
            entities.add(obj)
    return entities


def _find_schema_uri_connected_components(
    graph: RDFGraph,
) -> list[set[URIRef]]:
    entities = _schema_relevant_uri_entities(graph)
    if not entities:
        return []
    adjacency = _build_schema_uri_adjacency(graph)
    visited: set[URIRef] = set()
    components: list[set[URIRef]] = []
    for entity in sorted(entities, key=str):
        if entity in visited:
            continue
        component: set[URIRef] = set()
        queue: deque[URIRef] = deque([entity])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited and neighbor in entities:
                    queue.append(neighbor)
        if component:
            components.append(component)
    return components


def _count_meaningful_restriction_predicates(
    graph: RDFGraph,
    bnode: BNode,
) -> int:
    return sum(
        1
        for _, pred, _ in graph.triples((bnode, None, None))
        if pred in _OWL_RESTRICTION_MEANINGFUL_PREDICATES
    )


def _remove_bnode_subgraph(graph: RDFGraph, bnode: BNode) -> None:
    for triple in list(graph.triples((bnode, None, None))):
        graph.remove(triple)


def _remove_subclassof_to_bnode(graph: RDFGraph, bnode: BNode) -> None:
    for triple in list(graph.triples((None, RDFS.subClassOf, bnode))):
        graph.remove(triple)
    for triple in list(graph.triples((None, OWL.equivalentClass, bnode))):
        graph.remove(triple)


def _materialize_owl_restriction_shell(
    merged_graph: RDFGraph,
    bnode: BNode,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
    max_shell_triples: int = _RESTRICTION_SHELL_MAX_TRIPLES,
) -> int:
    """Copy restriction interior from merged graph; return meaningful predicate count."""
    shell_added = 0
    nested: list[BNode] = []
    for pred, obj in sorted(
        merged_graph.predicate_objects(bnode),
        key=lambda pair: str(pair),
    ):
        if pred not in _OWL_RESTRICTION_SHELL_PREDICATES:
            continue
        if shell_added >= max_shell_triples or len(result) >= max_total_triples:
            break
        triple = (bnode, pred, obj)
        if not _add_triple_if_room(
            result,
            triple,
            max_total_triples=max_total_triples,
            should_include=should_include,
        ):
            break
        shell_added += 1
        if isinstance(obj, BNode) and pred in _OWL_RESTRICTION_MEANINGFUL_PREDICATES:
            nested.append(obj)

    for nested_bnode in nested:
        if shell_added >= max_shell_triples or len(result) >= max_total_triples:
            break
        for pred, obj in sorted(
            merged_graph.predicate_objects(nested_bnode),
            key=lambda pair: str(pair),
        ):
            if pred not in _OWL_RESTRICTION_SHELL_PREDICATES:
                continue
            if shell_added >= max_shell_triples or len(result) >= max_total_triples:
                break
            triple = (nested_bnode, pred, obj)
            if _add_triple_if_room(
                result,
                triple,
                max_total_triples=max_total_triples,
                should_include=should_include,
            ):
                shell_added += 1

    return _count_meaningful_restriction_predicates(result, bnode)


def _add_subclassof_bnode_or_drop(
    merged_graph: RDFGraph,
    class_node: URIRef,
    bnode: BNode,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
) -> bool:
    """Materialize restriction subClassOf target or omit the edge entirely."""
    edge = (class_node, RDFS.subClassOf, bnode)
    if not _add_triple_if_room(
        result,
        edge,
        max_total_triples=max_total_triples,
        should_include=should_include,
    ):
        return False
    meaningful = _materialize_owl_restriction_shell(
        merged_graph,
        bnode,
        result,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    if meaningful >= _MIN_MEANINGFUL_RESTRICTION_PREDICATES:
        return True
    _remove_subclassof_to_bnode(result, bnode)
    _remove_bnode_subgraph(result, bnode)
    return False


def _add_triple_if_room(
    result: RDFGraph,
    triple: tuple,
    *,
    max_total_triples: int,
    should_include,
) -> bool:
    if len(result) >= max_total_triples:
        return False
    subj, pred, obj = triple
    if not should_include(subj, pred, obj):
        return False
    if triple in result:
        return True
    result.add(triple)
    return True


def _add_class_schema_triples_for_node(
    merged_graph: RDFGraph,
    node: URIRef,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
) -> None:
    """Outgoing class hierarchy axioms (subClassOf, equivalentClass) for one URI."""
    for pred in _CLASS_SCHEMA_PREDICATES:
        for triple in sorted(
            merged_graph.triples((node, pred, None)),
            key=lambda t: str(t),
        ):
            if len(result) >= max_total_triples:
                return
            _, _, obj = triple
            if isinstance(obj, URIRef):
                _add_triple_if_room(
                    result,
                    triple,
                    max_total_triples=max_total_triples,
                    should_include=should_include,
                )
            elif isinstance(obj, BNode) and pred == RDFS.subClassOf:
                _add_subclassof_bnode_or_drop(
                    merged_graph,
                    node,
                    obj,
                    result,
                    max_total_triples=max_total_triples,
                    should_include=should_include,
                )


def _materialize_class_node_in_snapshot(
    merged_graph: RDFGraph,
    node: URIRef,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
    include_types: bool = True,
) -> None:
    """Priority order: hierarchy axioms, then glosses, then informative types."""
    _add_class_schema_triples_for_node(
        merged_graph,
        node,
        result,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    _add_description_triples_for_node(
        merged_graph,
        node,
        result,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    if not include_types:
        return
    for triple in sorted(
        merged_graph.triples((node, RDF.type, None)),
        key=lambda t: str(t),
    ):
        if len(result) >= max_total_triples:
            return
        _, _, type_iri = triple
        if type_iri in _GENERIC_INDIVIDUAL_TYPES:
            continue
        _add_triple_if_room(
            result,
            triple,
            max_total_triples=max_total_triples,
            should_include=should_include,
        )


def _ensure_class_hierarchy_axioms_pass(
    merged_graph: RDFGraph,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
) -> None:
    """Backfill subClassOf on URI subjects already present without hierarchy axioms."""
    if len(result) >= max_total_triples:
        return
    subjects = sorted(
        {s for s, _, _ in result if isinstance(s, URIRef)},
        key=str,
    )
    for subj in subjects:
        if len(result) >= max_total_triples:
            break
        has_hierarchy = any(
            pred in _CLASS_SCHEMA_PREDICATES
            for _, pred, _ in result.triples((subj, None, None))
        )
        if has_hierarchy:
            continue
        if not any(merged_graph.triples((subj, RDFS.subClassOf, None))):
            continue
        _add_class_schema_triples_for_node(
            merged_graph,
            subj,
            result,
            max_total_triples=max_total_triples,
            should_include=should_include,
        )


def _add_description_triples_for_node(
    merged_graph: RDFGraph,
    node: URIRef,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
) -> None:
    for pred in _SEED_DESCRIPTION_PREDICATES:
        outgoing = sorted(
            merged_graph.triples((node, pred, None)),
            key=lambda triple: str(triple),
        )
        incoming = sorted(
            merged_graph.triples((None, pred, node)),
            key=lambda triple: str(triple),
        )
        for triple in outgoing + incoming:
            if len(result) >= max_total_triples:
                return
            _add_triple_if_room(
                result,
                triple,
                max_total_triples=max_total_triples,
                should_include=should_include,
            )


def _add_subclass_ancestor_closure(
    merged_graph: RDFGraph,
    seed: URIRef,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
    ancestor_closure_depth: int,
) -> None:
    if ancestor_closure_depth <= 0:
        return
    frontier: set[URIRef] = {seed}
    visited: set[URIRef] = set()
    for _ in range(ancestor_closure_depth):
        if not frontier:
            break
        next_frontier: set[URIRef] = set()
        for node in sorted(frontier, key=str):
            if node in visited:
                continue
            visited.add(node)
            _materialize_class_node_in_snapshot(
                merged_graph,
                node,
                result,
                max_total_triples=max_total_triples,
                should_include=should_include,
                include_types=False,
            )
            for _, _, parent in sorted(
                merged_graph.triples((node, RDFS.subClassOf, None)),
                key=lambda t: str(t[2]),
            ):
                if isinstance(parent, URIRef) and parent not in visited:
                    next_frontier.add(parent)
        frontier = next_frontier


def _schema_shell_for_concept_seeds(
    merged_graph: RDFGraph,
    concept_seeds: list[str],
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
    ancestor_closure_depth: int,
) -> None:
    for seed_uri in concept_seeds:
        if len(result) >= max_total_triples:
            break
        seed = URIRef(seed_uri)
        _materialize_class_node_in_snapshot(
            merged_graph,
            seed,
            result,
            max_total_triples=max_total_triples,
            should_include=should_include,
            include_types=True,
        )
        _add_subclass_ancestor_closure(
            merged_graph,
            seed,
            result,
            max_total_triples=max_total_triples,
            should_include=should_include,
            ancestor_closure_depth=ancestor_closure_depth,
        )


def _bfs_expand_from_seed(
    merged_graph: RDFGraph,
    seed_uri: str,
    result: RDFGraph,
    *,
    max_total_triples: int,
    should_include,
    depth: int,
    quota: int,
) -> None:
    if quota <= 0 or depth < 0:
        return
    seed = URIRef(seed_uri)
    candidates: list[tuple] = []
    seen: set[tuple] = set()

    def append_candidate(triple: tuple) -> None:
        if triple in seen:
            return
        seen.add(triple)
        candidates.append(triple)

    frontier: set[URIRef] = {seed}
    visited: set[URIRef] = set()
    for _ in range(depth + 1):
        if not frontier:
            break
        next_frontier: set[URIRef] = set()
        for node in sorted(frontier, key=str):
            if node in visited:
                continue
            visited.add(node)
            _materialize_class_node_in_snapshot(
                merged_graph,
                node,
                result,
                max_total_triples=max_total_triples,
                should_include=should_include,
                include_types=False,
            )
            outgoing = sorted(
                merged_graph.triples((node, None, None)),
                key=lambda triple: str(triple),
            )
            incoming = sorted(
                merged_graph.triples((None, None, node)),
                key=lambda triple: str(triple),
            )
            for triple in outgoing + incoming:
                subj, pred, obj = triple
                if not should_include(subj, pred, obj):
                    continue
                append_candidate(triple)
                if isinstance(subj, URIRef) and subj not in visited:
                    next_frontier.add(subj)
                if isinstance(obj, URIRef) and obj not in visited:
                    next_frontier.add(obj)
        frontier = next_frontier

    selected = 0
    for triple in candidates:
        if len(result) >= max_total_triples:
            break
        if triple in result:
            continue
        result.add(triple)
        selected += 1
        if selected >= quota:
            break


def _prune_degenerate_restriction_bnodes(result: RDFGraph) -> int:
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
            _count_meaningful_restriction_predicates(result, bnode)
            >= _MIN_MEANINGFUL_RESTRICTION_PREDICATES
        ):
            continue
        _remove_subclassof_to_bnode(result, bnode)
        _remove_bnode_subgraph(result, bnode)
        dropped += 1
    return dropped


def _class_has_property_incidence(graph: RDFGraph, class_ref: URIRef) -> bool:
    for subj, pred, obj in graph:
        if pred not in (RDFS.domain, RDFS.range) or obj != class_ref:
            continue
        if isinstance(subj, URIRef):
            return True
    return False


def _find_property_for_class(
    merged_graph: RDFGraph,
    class_ref: URIRef,
    preferred_properties: set[str],
) -> URIRef | None:
    candidates: list[URIRef] = []
    for prop, _, _ in merged_graph.triples((None, RDFS.domain, class_ref)):
        if isinstance(prop, URIRef):
            candidates.append(prop)
    for prop, _, _ in merged_graph.triples((None, RDFS.range, class_ref)):
        if isinstance(prop, URIRef) and prop not in candidates:
            candidates.append(prop)
    if not candidates:
        return None
    preferred = [p for p in candidates if str(p) in preferred_properties]
    pool = preferred if preferred else candidates
    return sorted(pool, key=str)[0]


def _ensure_property_schema_links(
    merged_graph: RDFGraph,
    result: RDFGraph,
    concept_seeds: list[str],
    property_seeds: list[str],
    *,
    max_total_triples: int,
    should_include,
) -> None:
    """Add at least one domain/range property bridge for classes lacking property incidence."""
    preferred = set(property_seeds)
    targets = list(dict.fromkeys(concept_seeds))
    for subj, pred, obj in result:
        if pred in (RDFS.domain, RDFS.range) and isinstance(obj, URIRef):
            uri = str(obj)
            if uri not in targets:
                targets.append(uri)
    for class_uri in targets:
        if len(result) >= max_total_triples:
            break
        class_ref = URIRef(class_uri)
        if not _class_has_property_incidence(result, class_ref):
            prop = _find_property_for_class(merged_graph, class_ref, preferred)
            if prop is not None:
                for triple in sorted(
                    merged_graph.triples((prop, None, None)),
                    key=lambda t: str(t),
                ):
                    if len(result) >= max_total_triples:
                        break
                    _, pred, _ = triple
                    if pred not in _PROPERTY_DEFINITION_PREDICATES:
                        continue
                    _add_triple_if_room(
                        result,
                        triple,
                        max_total_triples=max_total_triples,
                        should_include=should_include,
                    )
        _materialize_class_node_in_snapshot(
            merged_graph,
            class_ref,
            result,
            max_total_triples=max_total_triples,
            should_include=should_include,
            include_types=False,
        )


def _find_schema_path_in_merged_graph(
    merged_graph: RDFGraph,
    start: URIRef,
    goal: URIRef,
    *,
    max_depth: int = _SCHEMA_PATH_MAX_DEPTH,
) -> list[tuple] | None:
    """Shortest URI path via schema predicates in merged graph (list of triples)."""
    if start == goal:
        return []
    queue: deque[tuple[URIRef, list[tuple]]] = deque([(start, [])])
    visited: set[URIRef] = {start}
    while queue:
        current, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for pred in _SCHEMA_URI_CONNECTIVITY_PREDICATES:
            for neighbor in merged_graph.objects(current, pred):
                if not isinstance(neighbor, URIRef):
                    continue
                triple = (current, pred, neighbor)
                if neighbor == goal:
                    return path + [triple]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [triple]))
            for subj in merged_graph.subjects(pred, current):
                if not isinstance(subj, URIRef):
                    continue
                triple = (subj, pred, current)
                if subj == goal:
                    return path + [triple]
                if subj not in visited:
                    visited.add(subj)
                    queue.append((subj, path + [triple]))
    return None


def _apply_schema_path_to_result(
    merged_graph: RDFGraph,
    result: RDFGraph,
    path: list[tuple],
    *,
    max_total_triples: int,
    should_include,
) -> None:
    for triple in path:
        if len(result) >= max_total_triples:
            return
        _add_triple_if_room(
            result,
            triple,
            max_total_triples=max_total_triples,
            should_include=should_include,
        )
        subj, _, obj = triple
        if isinstance(subj, URIRef):
            _materialize_class_node_in_snapshot(
                merged_graph,
                subj,
                result,
                max_total_triples=max_total_triples,
                should_include=should_include,
                include_types=False,
            )
        if isinstance(obj, URIRef):
            _materialize_class_node_in_snapshot(
                merged_graph,
                obj,
                result,
                max_total_triples=max_total_triples,
                should_include=should_include,
                include_types=False,
            )


def _prune_disconnected_uri_entities(
    result: RDFGraph,
    protected_uris: set[str],
) -> int:
    """Drop URI subjects in schema components that do not intersect protected seeds."""
    components = _find_schema_uri_connected_components(result)
    if not components:
        return 0
    protected_refs = {URIRef(uri) for uri in protected_uris}
    seed_components = [c for c in components if c & protected_refs]
    if not seed_components:
        return 0
    if len(seed_components) == 1:
        keep = seed_components[0]
    else:
        keep = max(seed_components, key=lambda c: len(c & protected_refs))

    all_uri_subjects = {s for s, _, _ in result if isinstance(s, URIRef)}
    drop_uris = all_uri_subjects - keep

    pruned = 0
    for uri in sorted(drop_uris, key=str):
        pruned += 1
        for triple in list(result.triples((uri, None, None))):
            result.remove(triple)
        for subj, pred, obj in list(result):
            if obj == uri and pred in _SCHEMA_URI_CONNECTIVITY_PREDICATES:
                result.remove((subj, pred, obj))
    return pruned


def _finalize_induced_subgraph_snapshot(
    merged_graph: RDFGraph,
    result: RDFGraph,
    sorted_seed_uris: list[str],
    concept_seeds: list[str],
    property_seeds: list[str],
    protected_uris: set[str],
    *,
    max_total_triples: int,
    should_include,
) -> dict[str, int]:
    """Post-process snapshot for schema connectivity and cleanliness."""
    _ensure_property_schema_links(
        merged_graph,
        result,
        concept_seeds,
        property_seeds,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    _ensure_class_hierarchy_axioms_pass(
        merged_graph,
        result,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    _connectivity_repair_pass(
        merged_graph,
        result,
        sorted_seed_uris,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    dropped_restrictions = _prune_degenerate_restriction_bnodes(result)
    _strip_redundant_generic_types(result)
    pruned_uris = _prune_disconnected_uri_entities(result, protected_uris)
    _prune_orphaned_bnode_subjects(result)
    components = _find_schema_uri_connected_components(result)
    return {
        "snapshot_uri_components": len(components),
        "snapshot_pruned_uri_count": pruned_uris,
        "snapshot_dropped_restriction_count": dropped_restrictions,
    }


def _connectivity_repair_seed_local_pass(
    merged_graph: RDFGraph,
    result: RDFGraph,
    sorted_seed_uris: list[str],
    *,
    max_total_triples: int,
    should_include,
) -> None:
    """Add outgoing schema triples from seeds to bridge URI components."""
    if len(result) >= max_total_triples:
        return
    components = _find_uri_connected_components(result)
    if len(components) <= 1:
        return

    seed_refs = [URIRef(uri) for uri in sorted_seed_uris]
    component_for: dict[URIRef, int] = {}
    for idx, component in enumerate(components):
        for node in component:
            component_for[node] = idx

    for seed in seed_refs:
        if len(result) >= max_total_triples:
            return
        if seed not in component_for:
            continue
        origin = component_for[seed]
        for _, pred, obj in merged_graph.triples((seed, None, None)):
            if pred not in _CONNECTIVITY_PREDICATES or not isinstance(obj, URIRef):
                continue
            if len(result) >= max_total_triples:
                return
            for triple in merged_graph.triples((seed, pred, obj)):
                _add_triple_if_room(
                    result,
                    triple,
                    max_total_triples=max_total_triples,
                    should_include=should_include,
                )
            _materialize_class_node_in_snapshot(
                merged_graph,
                obj,
                result,
                max_total_triples=max_total_triples,
                should_include=should_include,
                include_types=False,
            )
            if obj in component_for and component_for[obj] != origin:
                _add_class_schema_triples_for_node(
                    merged_graph,
                    obj,
                    result,
                    max_total_triples=max_total_triples,
                    should_include=should_include,
                )


def _connectivity_repair_cross_component_pass(
    merged_graph: RDFGraph,
    result: RDFGraph,
    sorted_seed_uris: list[str],
    *,
    max_total_triples: int,
    should_include,
) -> None:
    """Bridge remaining schema components via shortest paths in merged graph."""
    components = _find_schema_uri_connected_components(result)
    if len(components) <= 1:
        return
    seed_refs = sorted({URIRef(uri) for uri in sorted_seed_uris}, key=str)
    representatives: list[URIRef] = []
    for component in components:
        seeds_in_comp = [seed for seed in seed_refs if seed in component]
        if seeds_in_comp:
            representatives.append(seeds_in_comp[0])
    if len(representatives) < 2:
        return
    hub = representatives[0]
    for other in representatives[1:]:
        if len(result) >= max_total_triples:
            return
        path = _find_schema_path_in_merged_graph(merged_graph, hub, other)
        if path is None:
            path = _find_schema_path_in_merged_graph(merged_graph, other, hub)
        if path is None:
            continue
        domain_range_first = sorted(
            path,
            key=lambda triple: (
                0 if triple[1] in (RDFS.domain, RDFS.range) else 1,
                str(triple),
            ),
        )
        _apply_schema_path_to_result(
            merged_graph,
            result,
            domain_range_first,
            max_total_triples=max_total_triples,
            should_include=should_include,
        )


def _connectivity_repair_pass(
    merged_graph: RDFGraph,
    result: RDFGraph,
    sorted_seed_uris: list[str],
    *,
    max_total_triples: int,
    should_include,
) -> None:
    """Add schema triples linking disconnected URI components when budget remains."""
    _connectivity_repair_seed_local_pass(
        merged_graph,
        result,
        sorted_seed_uris,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )
    _connectivity_repair_cross_component_pass(
        merged_graph,
        result,
        sorted_seed_uris,
        max_total_triples=max_total_triples,
        should_include=should_include,
    )


def _protected_uris_for_snapshot(
    seed_uris_ranked: list[str],
    concept_seeds: list[str],
    property_seeds: list[str],
    merged_graph: RDFGraph,
) -> set[str]:
    protected = set(seed_uris_ranked) | set(concept_seeds) | set(property_seeds)
    for prop_uri in property_seeds:
        prop_ref = URIRef(prop_uri)
        for _, pred, obj in merged_graph.triples((prop_ref, None, None)):
            if pred in (RDFS.domain, RDFS.range) and isinstance(obj, URIRef):
                protected.add(str(obj))
    return protected


class SPARQLTool:
    """Tool for executing SPARQL operations on RDF graphs."""

    def __init__(self, triple_store_manager: TripleStoreManager | None = None):
        """Initialize SPARQL tool.

        Args:
            triple_store_manager: Optional triple store manager for persistent storage.
        """
        self.triple_store_manager = triple_store_manager
        self.operation_history = []
        self.last_finalize_metrics: dict[str, int] = {}

    def execute_operations(
        self, graph: RDFGraph, operations: list[SPARQLOperationModel]
    ) -> RDFGraph:
        """Execute a list of SPARQL operations on a graph.

        Args:
            graph: The RDF graph to operate on.
            operations: List of SPARQL operations to execute.

        Returns:
            RDFGraph: Updated graph after applying operations.
        """
        logger.info(f"Executing {len(operations)} SPARQL operations")

        for operation in operations:
            try:
                self._execute_single_operation(graph, operation)
                self.operation_history.append(operation)
                logger.debug(
                    f"Executed {operation.operation_type} operation: {operation.description}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to execute {operation.operation_type} operation: {str(e)}"
                )
                raise

        return graph

    def execute_operation(self, operation: SPARQLOperationModel) -> None:
        """Execute a single SPARQL operation.

        Args:
            operation: The SPARQL operation to execute.
        """
        # For now, we'll use a simple approach - in a real implementation,
        # you might want to track which graph this operation should be applied to
        logger.info(
            f"Executing {operation.operation_type} operation: {operation.description}"
        )
        # This is a placeholder - in practice, you'd need to specify which graph to operate on
        # or maintain a default graph in the tool

    def _execute_single_operation(
        self, graph: RDFGraph, operation: SPARQLOperationModel
    ):
        """Execute a single SPARQL operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The SPARQL operation to execute.
        """
        if operation.operation_type == SPARQLOperationType.INSERT:
            self._execute_insert(graph, operation)
        elif operation.operation_type == SPARQLOperationType.DELETE:
            self._execute_delete(graph, operation)
        elif operation.operation_type == SPARQLOperationType.UPDATE:
            self._execute_update(graph, operation)
        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")

    def _execute_insert(self, graph: RDFGraph, operation: SPARQLOperationModel):
        """Execute INSERT operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The INSERT operation to execute.
        """
        # Parse the INSERT query
        query = prepareQuery(operation.query)

        # For INSERT DATA, we need to parse the triples and add them to the graph
        if "INSERT DATA" in operation.query.upper():
            # Extract triples from INSERT DATA query
            triples = self._parse_insert_data_triples(operation.query)
            for triple in triples:
                graph.add(triple)
        else:
            # For other INSERT queries, execute against the graph
            graph.query(query)
            # INSERT queries typically don't return results, but we execute them

    def _execute_delete(self, graph: RDFGraph, operation: SPARQLOperationModel):
        """Execute DELETE operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The DELETE operation to execute.
        """
        # Parse the DELETE query
        query = prepareQuery(operation.query)

        # For DELETE DATA, we need to parse the triples and remove them from the graph
        if "DELETE DATA" in operation.query.upper():
            # Extract triples from DELETE DATA query
            triples = self._parse_delete_data_triples(operation.query)
            for triple in triples:
                graph.remove(triple)
        else:
            # For other DELETE queries, execute against the graph
            graph.query(query)
            # DELETE queries typically don't return results, but we execute them

    def _execute_update(self, graph: RDFGraph, operation: SPARQLOperationModel):
        """Execute UPDATE operation.

        Args:
            graph: The RDF graph to operate on.
            operation: The UPDATE operation to execute.
        """
        # Parse the UPDATE query
        query = prepareQuery(operation.query)

        # Execute the UPDATE query
        graph.query(query)
        # UPDATE queries typically don't return results, but we execute them

    def _parse_insert_data_triples(self, query: str) -> list[tuple]:
        """Parse triples from INSERT DATA query.

        Args:
            query: The INSERT DATA query string.

        Returns:
            List of triples to insert.
        """
        # This is a simplified parser - in practice, you'd want a more robust parser
        triples = []

        # Extract the content between INSERT DATA { ... }
        start = query.upper().find("INSERT DATA {")
        if start == -1:
            return triples

        start += len("INSERT DATA {")
        end = query.rfind("}")

        if end == -1:
            return triples

        data_content = query[start:end].strip()

        # Split by lines and parse each triple
        lines = [line.strip() for line in data_content.split("\n") if line.strip()]

        for line in lines:
            if line.endswith("."):
                line = line[:-1]  # Remove trailing period

            # Parse the triple (simplified - assumes standard N3 format)
            parts = line.split()
            if len(parts) >= 3:
                subject = self._parse_term(parts[0])
                predicate = self._parse_term(parts[1])
                object_part = self._parse_term(" ".join(parts[2:]))

                if subject and predicate and object_part:
                    triples.append((subject, predicate, object_part))

        return triples

    def _parse_delete_data_triples(self, query: str) -> list[tuple]:
        """Parse triples from DELETE DATA query.

        Args:
            query: The DELETE DATA query string.

        Returns:
            List of triples to delete.
        """
        # Similar to INSERT DATA parsing
        return self._parse_insert_data_triples(
            query.replace("DELETE DATA", "INSERT DATA")
        )

    def _parse_term(self, term: str):
        """Parse a SPARQL term (subject, predicate, or object).

        Args:
            term: The term string to parse.

        Returns:
            Parsed RDF term (URIRef, Literal, or BNode).
        """
        term = term.strip()

        if term.startswith("<") and term.endswith(">"):
            # URI
            return URIRef(term[1:-1])
        elif term.startswith('"') and term.endswith('"'):
            # Literal
            return Literal(term[1:-1])
        elif term.startswith("_:"):
            # Blank node
            return BNode(term[2:])
        elif term.startswith('"') and '"^^' in term:
            # Typed literal
            value, datatype = term.split('"^^')
            return Literal(value[1:], datatype=URIRef(datatype))
        else:
            # Assume it's a URI without angle brackets
            return URIRef(term)

    def validate_operation(self, operation: SPARQLOperationModel) -> bool:
        """Validate a SPARQL operation.

        Args:
            operation: The operation to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            prepareQuery(operation.query)
            return True
        except Exception as e:
            logger.error(f"Invalid SPARQL operation: {str(e)}")
            return False

    def get_operation_history(self) -> list[SPARQLOperationModel]:
        """Get the history of executed operations.

        Returns:
            List of executed operations.
        """
        return self.operation_history.copy()

    def clear_history(self):
        """Clear the operation history."""
        self.operation_history.clear()

    @staticmethod
    def _build_induced_subgraph(
        ontologies: list[Ontology],
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None,
        ontology_iris: list[str] | None,
        depth: int,
        max_total_triples: int,
        estimated_triples_per_query: int,
        ontology_version_filters: dict[str, set[str]] | None,
        ontology_hash_filters: dict[str, set[str]] | None,
        entity_roles: Mapping[str, str | None] | None = None,
        hub_seed_count: int = 8,
        ancestor_closure_depth: int = 3,
    ) -> tuple[RDFGraph, dict[str, int]]:
        """Merge filtered graphs; schema shell, hub BFS, and connectivity repair."""

        ontology_filter = set(ontology_iris or [])
        relevant_graphs: list[RDFGraph] = []
        for ontology in ontologies:
            if ontology_filter and ontology.iri not in ontology_filter:
                continue
            if ontology_version_filters and ontology.iri in ontology_version_filters:
                ontology_version = (
                    str(ontology.version) if ontology.version is not None else None
                )
                if ontology_version not in ontology_version_filters[ontology.iri]:
                    continue
            if ontology_hash_filters and ontology.iri in ontology_hash_filters:
                if ontology.hash not in ontology_hash_filters[ontology.iri]:
                    continue
            relevant_graphs.append(ontology.graph)
        if not relevant_graphs:
            return RDFGraph(), {}

        all_ns_map: dict[str, str] = {}
        for graph in relevant_graphs:
            for prefix, namespace in graph.namespaces():
                if prefix:
                    all_ns_map[prefix] = str(namespace)
        filtered_ns = _filter_overbroad_namespace_map(all_ns_map)

        merged_graph = RDFGraph()
        for prefix, uri in filtered_ns.items():
            merged_graph.bind(prefix, Namespace(uri))
        for graph in relevant_graphs:
            merged_graph += graph

        ontology_subjects: frozenset[str] = frozenset(
            str(s) for s, _, _ in merged_graph.triples((None, RDF.type, OWL.Ontology))
        )

        def should_include_expansion_triple(
            subj: object,
            pred: object,
            obj: object,
        ) -> bool:
            if not isinstance(pred, URIRef):
                return False
            if pred in _NOISY_EXPANSION_PREDICATES:
                return False
            if isinstance(subj, BNode) and isinstance(obj, BNode):
                return False
            if isinstance(subj, URIRef) and str(subj) in ontology_subjects:
                return False
            return True

        if not entity_uris:
            return RDFGraph(), {}
        seed_uris_ranked = list(dict.fromkeys(uri for uri in entity_uris if uri))
        if not seed_uris_ranked:
            return RDFGraph(), {}
        result = RDFGraph()
        for prefix, uri in filtered_ns.items():
            result.bind(prefix, Namespace(uri))

        if max_total_triples <= 0 or estimated_triples_per_query <= 0:
            return result, {}

        relevance = entity_relevance or {}
        roles = entity_roles or {}
        concept_seeds, property_seeds = _classify_and_promote_seeds(
            seed_uris_ranked, merged_graph, roles, ontology_subjects
        )
        property_seeds = _crosslink_property_seeds(
            merged_graph, concept_seeds, property_seeds, ontology_subjects
        )

        property_triple_budget = min(
            max(32, max_total_triples // 6),
            max_total_triples // 4,
        )
        property_triples_start = len(result)
        for prop_uri in property_seeds:
            if len(result) >= max_total_triples:
                break
            if len(result) - property_triples_start >= property_triple_budget:
                break
            for triple in merged_graph.triples((URIRef(prop_uri), None, None)):
                subj, pred, obj = triple
                if pred not in _PROPERTY_DEFINITION_PREDICATES:
                    continue
                if not should_include_expansion_triple(subj, pred, obj):
                    continue
                if triple in result:
                    continue
                result.add(triple)

        protected_uris = _protected_uris_for_snapshot(
            seed_uris_ranked, concept_seeds, property_seeds, merged_graph
        )

        if not concept_seeds:
            concept_for_finalize = list(
                dict.fromkeys(
                    str(obj)
                    for prop_uri in property_seeds
                    for _, pred, obj in merged_graph.triples(
                        (URIRef(prop_uri), None, None)
                    )
                    if pred in (RDFS.domain, RDFS.range) and isinstance(obj, URIRef)
                )
            )
            metrics = _finalize_induced_subgraph_snapshot(
                merged_graph,
                result,
                concept_for_finalize,
                concept_for_finalize,
                property_seeds,
                protected_uris,
                max_total_triples=max_total_triples,
                should_include=should_include_expansion_triple,
            )
            return result, metrics

        concept_relevance = _build_concept_relevance(
            seed_uris_ranked, merged_graph, relevance, ontology_subjects
        )
        sorted_seed_uris = sorted(
            concept_seeds,
            key=lambda uri: (-float(concept_relevance.get(uri, 0.0)), uri),
        )

        _schema_shell_for_concept_seeds(
            merged_graph,
            sorted_seed_uris,
            result,
            max_total_triples=max_total_triples,
            should_include=should_include_expansion_triple,
            ancestor_closure_depth=ancestor_closure_depth,
        )

        score_by_seed: dict[str, float] = {
            uri: float(concept_relevance.get(uri, 0.0)) for uri in sorted_seed_uris
        }
        score_total = sum(max(score, 0.0) for score in score_by_seed.values())
        if score_total <= 0.0:
            score_by_seed = {uri: 1.0 for uri in sorted_seed_uris}
            score_total = float(len(sorted_seed_uris))

        remaining = max_total_triples - len(result)
        per_entity_cap = max(1, estimated_triples_per_query)
        hub_count = (
            len(sorted_seed_uris)
            if hub_seed_count <= 0
            else min(hub_seed_count, len(sorted_seed_uris))
        )
        hub_seeds = sorted_seed_uris[:hub_count]
        tail_seeds = sorted_seed_uris[hub_count:]

        hub_budget = int(remaining * 0.65) if remaining > 0 else 0
        tail_budget = remaining - hub_budget

        if hub_seeds and hub_budget > 0:
            hub_quota_base = max(1, hub_budget // len(hub_seeds))
            for seed_uri in hub_seeds:
                if len(result) >= max_total_triples:
                    break
                quota = min(per_entity_cap, hub_quota_base)
                _bfs_expand_from_seed(
                    merged_graph,
                    seed_uri,
                    result,
                    max_total_triples=max_total_triples,
                    should_include=should_include_expansion_triple,
                    depth=depth,
                    quota=quota,
                )

        if tail_seeds and tail_budget > 0:
            tail_quota_total = tail_budget
            for seed_uri in tail_seeds:
                if tail_quota_total <= 0 or len(result) >= max_total_triples:
                    break
                weight = max(score_by_seed.get(seed_uri, 0.0), 0.0) / score_total
                quota = max(1, int(tail_quota_total * weight))
                quota = min(quota, per_entity_cap)
                _bfs_expand_from_seed(
                    merged_graph,
                    seed_uri,
                    result,
                    max_total_triples=max_total_triples,
                    should_include=should_include_expansion_triple,
                    depth=max(0, depth - 1),
                    quota=quota,
                )
                tail_quota_total -= quota

        metrics = _finalize_induced_subgraph_snapshot(
            merged_graph,
            result,
            sorted_seed_uris,
            concept_seeds,
            property_seeds,
            protected_uris,
            max_total_triples=max_total_triples,
            should_include=should_include_expansion_triple,
        )
        return result, metrics

    def get_induced_subgraph(
        self,
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None = None,
        entity_roles: Mapping[str, str | None] | None = None,
        ontology_iris: list[str] | None = None,
        depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
        ontology_version_filters: dict[str, set[str]] | None = None,
        ontology_hash_filters: dict[str, set[str]] | None = None,
        hub_seed_count: int = 8,
        ancestor_closure_depth: int = 3,
    ) -> RDFGraph:
        """Fetch a deterministic induced subgraph around selected entities."""
        if self.triple_store_manager is None:
            return RDFGraph()
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if max_total_triples <= 0:
            return RDFGraph()
        if estimated_triples_per_query <= 0:
            return RDFGraph()

        ontologies = self.triple_store_manager.fetch_ontologies()
        result, metrics = SPARQLTool._build_induced_subgraph(
            ontologies,
            entity_uris,
            entity_relevance,
            ontology_iris,
            depth,
            max_total_triples,
            estimated_triples_per_query,
            ontology_version_filters,
            ontology_hash_filters,
            entity_roles,
            hub_seed_count,
            ancestor_closure_depth,
        )
        self.last_finalize_metrics = metrics
        return result

    async def aget_induced_subgraph(
        self,
        entity_uris: list[str],
        entity_relevance: dict[str, float] | None = None,
        entity_roles: Mapping[str, str | None] | None = None,
        ontology_iris: list[str] | None = None,
        depth: int = 1,
        max_total_triples: int = 300,
        estimated_triples_per_query: int = 24,
        ontology_version_filters: dict[str, set[str]] | None = None,
        ontology_hash_filters: dict[str, set[str]] | None = None,
        hub_seed_count: int = 8,
        ancestor_closure_depth: int = 3,
    ) -> RDFGraph:
        """Like ``get_induced_subgraph`` but uses ``afetch_ontologies`` for I/O."""
        if self.triple_store_manager is None:
            return self.get_induced_subgraph(
                entity_uris=entity_uris,
                entity_relevance=entity_relevance,
                entity_roles=entity_roles,
                ontology_iris=ontology_iris,
                depth=depth,
                max_total_triples=max_total_triples,
                estimated_triples_per_query=estimated_triples_per_query,
                ontology_version_filters=ontology_version_filters,
                ontology_hash_filters=ontology_hash_filters,
                hub_seed_count=hub_seed_count,
                ancestor_closure_depth=ancestor_closure_depth,
            )
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if max_total_triples <= 0:
            return RDFGraph()
        if estimated_triples_per_query <= 0:
            return RDFGraph()

        ontologies = await self.triple_store_manager.afetch_ontologies()
        result, metrics = await asyncio.to_thread(
            SPARQLTool._build_induced_subgraph,
            ontologies,
            entity_uris,
            entity_relevance,
            ontology_iris,
            depth,
            max_total_triples,
            estimated_triples_per_query,
            ontology_version_filters,
            ontology_hash_filters,
            entity_roles,
            hub_seed_count,
            ancestor_closure_depth,
        )
        self.last_finalize_metrics = metrics
        return result

    def create_insert_operation(
        self, query: str, description: str = ""
    ) -> SPARQLOperationModel:
        """Create an INSERT operation.

        Args:
            query: The SPARQL INSERT query.
            description: Optional description of the operation.

        Returns:
            SPARQLOperationModel: The created operation.
        """
        return SPARQLOperationModel(
            operation_type=SPARQLOperationType.INSERT,
            query=query,
            description=description,
        )

    def create_delete_operation(
        self, query: str, description: str = ""
    ) -> SPARQLOperationModel:
        """Create a DELETE operation.

        Args:
            query: The SPARQL DELETE query.
            description: Optional description of the operation.

        Returns:
            SPARQLOperationModel: The created operation.
        """
        return SPARQLOperationModel(
            operation_type=SPARQLOperationType.DELETE,
            query=query,
            description=description,
        )

    def create_update_operation(
        self, query: str, description: str = ""
    ) -> SPARQLOperationModel:
        """Create an UPDATE operation.

        Args:
            query: The SPARQL UPDATE query.
            description: Optional description of the operation.

        Returns:
            SPARQLOperationModel: The created operation.
        """
        return SPARQLOperationModel(
            operation_type=SPARQLOperationType.UPDATE,
            query=query,
            description=description,
        )
