import logging
from datetime import datetime, timezone

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)


def merge_ontologies(onto1: Ontology, onto2: Ontology) -> Ontology:
    """Merge two ontologies algorithmically.

    This performs a union merge of the two ontology graphs, mapping contradictions.
    The result has both ontologies as parents. This is similar to a git merge:
    - Takes union of all triples from both ontologies
    - Detects contradictions (same subject-predicate with different objects)
    - Creates a new ontology with both parents
    - Sets created_at to merge time

    Args:
        onto1: First ontology to merge
        onto2: Second ontology to merge

    Returns:
        Ontology: Merged ontology with both parents

    Raises:
        ValueError: If ontologies have different IRIs
    """
    # Validate that both ontologies have the same IRI
    if onto1.iri != onto2.iri:
        raise ValueError(
            f"Cannot merge ontologies with different IRIs: {onto1.iri} != {onto2.iri}"
        )

    # Ensure both ontologies have hashes
    if not onto1.hash:
        onto1._compute_and_set_hash()
    if not onto2.hash:
        onto2._compute_and_set_hash()

    if not onto1.hash or not onto2.hash:
        raise ValueError("Cannot merge ontologies without hashes")

    # Create merged graph (union) - use RDFGraph's __add__ operator
    merged_graph = onto1.graph + onto2.graph

    # Map contradictions (same subject-predicate with different objects)
    contradictions = _find_contradictions(onto1.graph, onto2.graph)
    if contradictions:
        logger.warning(f"Found {len(contradictions)} contradictions in merge")
        for (s, p), (obj1_set, obj2) in contradictions.items():
            logger.debug(f"Contradiction: {s} {p} -> {obj1_set} vs {obj2}")
            # For now, keep both objects (RDF allows multiple values)
            # In future LLM-based merge, this would be resolved intelligently

    # Create merged ontology
    # Note: sync_properties_to_graph() will remove existing versionInfo triples,
    # so we need to preserve them manually after creation
    merged_ontology = Ontology(
        graph=merged_graph,
        iri=onto1.iri,  # Use IRI from first ontology (should be same)
        title=onto1.title or onto2.title,
        description=onto1.description or onto2.description,
        ontology_id=onto1.ontology_id or onto2.ontology_id,
        version=onto1.version or onto2.version or "1.0.0",
        parent_hashes=[onto1.hash, onto2.hash],
        created_at=datetime.now(timezone.utc),
    )

    # Compute hash for merged ontology
    # Note: Hash excludes metadata (version, title, description, created_at, hash, parent_hash)
    # so it only reflects the actual ontology content (classes, properties, etc.)
    merged_ontology._compute_and_set_hash()

    logger.info(
        f"Merged ontologies {onto1.hash[:8]}... "
        f"and {onto2.hash[:8]}... "
        f"-> {merged_ontology.hash[:8] if merged_ontology.hash else 'None'}..."
    )

    return merged_ontology


def _find_contradictions(graph1: RDFGraph, graph2: RDFGraph) -> dict:
    """Find contradictions between two graphs.

    Contradictions are triples with the same subject-predicate but different objects.
    Note: RDF allows multiple values for the same property, so this detects potential
    conflicts that might need resolution in an LLM-based merge.

    Args:
        graph1: First graph
        graph2: Second graph

    Returns:
        dict: Dictionary mapping (subject, predicate) to (set of objects from graph1, object from graph2) tuples
    """
    contradictions = {}

    # Build index of graph1: (subject, predicate) -> set of objects
    graph1_index: dict[tuple, set] = {}
    for s, p, o in graph1:
        key = (s, p)
        if key not in graph1_index:
            graph1_index[key] = set()
        # Use string representation for comparison
        graph1_index[key].add(str(o))

    # Check graph2 against graph1
    for s, p, o in graph2:
        key = (s, p)
        if key in graph1_index:
            # Check if objects differ
            graph1_objects = graph1_index[key]
            obj2_str = str(o)
            if obj2_str not in graph1_objects:
                # Contradiction found - different object values
                contradictions[key] = (graph1_objects, obj2_str)

    return contradictions
