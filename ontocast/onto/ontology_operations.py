import logging
from datetime import datetime, timezone
from pathlib import Path

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool import FusekiTripleStoreManager, OntologyManager

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


def plot_ontology_graph(
    ontology_manager: OntologyManager,
    output_path: Path,
    iri: str | None = None,
) -> None:
    """Plot the ontology version graph using pygraphviz.

    Args:
        ontology_manager: The ontology manager containing ontologies
        output_path: Path to save the graph image
        iri: Optional IRI to plot (if None, plots all ontologies)
    """
    try:
        import pygraphviz as pgv  # type: ignore
    except ImportError:
        logger.error("pygraphviz not installed. Cannot plot graph.")
        logger.info("Install with: pip install pygraphviz")
        return

    # Get ontologies to plot
    if iri:
        if iri not in ontology_manager.ontology_versions:
            logger.warning(f"No ontologies found for IRI: {iri}")
            return
        ontologies = ontology_manager.ontology_versions[iri]
    else:
        ontologies = [
            o
            for versions in ontology_manager.ontology_versions.values()
            for o in versions
        ]

    if not ontologies:
        logger.warning("No ontologies to plot")
        return

    # Create graph
    viz = pgv.AGraph(directed=True, nodesep=0.7, ranksep=0.5)

    # Add nodes
    for onto in ontologies:
        if not onto.hash:
            continue
        node_id = onto.hash[:12]  # Use first 12 chars of hash as node ID
        label = f"{onto.ontology_id or 'ont'}\n{onto.hash[:8]}..."
        if onto.created_at:
            label += f"\n{onto.created_at.strftime('%Y-%m-%d')}"
        viz.add_node(
            node_id,
            label=label,
            style="filled",
            fillcolor="#a9cca9",
            fontsize=10,
        )

    # Add edges (parent relationships)
    for onto in ontologies:
        if not onto.hash:
            continue
        node_id = onto.hash[:12]
        for parent_hash in onto.parent_hashes:
            # Find parent node
            parent_onto = None
            for o in ontologies:
                if o.hash == parent_hash:
                    parent_onto = o
                    break
            if parent_onto:
                parent_id = parent_hash[:12]
                viz.add_edge(parent_id, node_id, style="solid")

    # Highlight terminal ontologies
    terminals = (
        ontology_manager.get_terminal_ontologies_by_iri(iri)
        if iri
        else ontology_manager.get_terminal_ontologies_by_iri(None)
    )
    for terminal in terminals:
        if terminal.hash:
            node_id = terminal.hash[:12]
            node = viz.get_node(node_id)
            if node:
                node.attr["fillcolor"] = "#ffdb99"  # Orange for terminals

    # Save graph
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz.draw(str(output_path), format="png", prog="dot", args="-Gdpi=300")
    logger.info(f"Saved ontology graph to {output_path}")


async def merge_terminal_ontologies(
    fuseki_manager: FusekiTripleStoreManager,
    ontology_manager: OntologyManager,
    iri: str,
) -> Ontology | None:
    """Merge terminal ontologies for a given IRI.

    Fetches all terminal ontologies from Fuseki and merges them pair-wise
    until only one remains.

    Args:
        fuseki_manager: Fuseki triple store manager
        ontology_manager: Ontology manager to add merged ontologies to
        iri: IRI of the ontology to merge

    Returns:
        Ontology: The final merged ontology, or None if no ontologies found
    """
    # Fetch all ontologies from Fuseki
    logger.info(f"Fetching ontologies for IRI: {iri}")
    all_ontologies = await fuseki_manager.afetch_ontologies()

    # Filter by IRI and add to ontology manager
    matching_ontologies = [o for o in all_ontologies if o.iri == iri]
    if not matching_ontologies:
        logger.warning(f"No ontologies found for IRI: {iri}")
        return None

    logger.info(f"Found {len(matching_ontologies)} ontologies for IRI: {iri}")

    # Add all to ontology manager
    for onto in matching_ontologies:
        ontology_manager.add_ontology(onto)

    # Get terminal ontologies
    terminals = ontology_manager.get_terminal_ontologies_by_iri(iri)
    logger.info(f"Found {len(terminals)} terminal ontologies")

    # Merge pair-wise until only one remains
    while len(terminals) > 1:
        # Sort by created_at (oldest first)
        terminals_with_time = [t for t in terminals if t.created_at is not None]
        terminals_without_time = [t for t in terminals if t.created_at is None]

        # Sort terminals with time by created_at
        terminals_with_time.sort(key=lambda x: x.created_at)

        # Combine: terminals with time (sorted) + terminals without time
        sorted_terminals = terminals_with_time + terminals_without_time

        if len(sorted_terminals) < 2:
            break

        # Take the two oldest
        onto1 = sorted_terminals[0]
        onto2 = sorted_terminals[1]

        logger.info(
            f"Merging ontologies: {onto1.hash[:8] if onto1.hash else 'None'}... "
            f"and {onto2.hash[:8] if onto2.hash else 'None'}..."
        )

        # Merge
        merged = merge_ontologies(onto1, onto2)

        # Add merged ontology to manager
        ontology_manager.add_ontology(merged)

        # Update terminals list
        terminals = ontology_manager.get_terminal_ontologies_by_iri(iri)
        logger.info(f"After merge: {len(terminals)} terminal ontologies remaining")

    if terminals:
        logger.info(
            f"Final terminal ontology: {terminals[0].hash[:8] if terminals[0].hash else 'None'}..."
        )
        return terminals[0]
    else:
        logger.warning("No terminal ontologies remaining after merge")
        return None
