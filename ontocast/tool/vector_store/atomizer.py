"""Ontology atomization into neighborhood patches for vector indexing."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from rdflib import BNode, URIRef
from rdflib.term import Node

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.vector_store.core import OntologyAtom
from ontocast.util import render_text_hash


class OntologyAtomizer(Tool):
    """Extract small neighborhood subgraphs around ontology nodes."""

    def atomize(self, ontology: Ontology, depth: int = 1) -> list[OntologyAtom]:
        """Generate deterministic ontology atoms from local graph neighborhoods."""
        if depth < 0:
            raise ValueError("Atomizer depth must be >= 0")

        graph = ontology.graph
        subjects = sorted({subject for subject in graph.subjects()}, key=str)
        generated_at = datetime.now(timezone.utc)

        atoms_by_id: dict[str, OntologyAtom] = {}
        for subject in subjects:
            if not isinstance(subject, URIRef):
                continue
            patch_graph = self._build_neighborhood_graph(
                graph=graph, root=subject, depth=depth
            )
            if len(patch_graph) == 0:
                continue
            turtle = patch_graph.serialize(format="turtle")
            atom_id = render_text_hash(turtle, digits=None)
            if atom_id in atoms_by_id:
                continue
            atoms_by_id[atom_id] = OntologyAtom(
                atom_id=atom_id,
                ontology_iri=ontology.iri,
                ontology_id=ontology.ontology_id,
                ontology_hash=ontology.hash,
                node_uri=str(subject),
                turtle=turtle,
                created_at=generated_at,
            )
        return list(atoms_by_id.values())

    def _build_neighborhood_graph(
        self, graph: RDFGraph, root: URIRef, depth: int
    ) -> RDFGraph:
        """Build a local subgraph by bounded BFS over URI/BNode neighbors."""
        result = RDFGraph()
        self._copy_namespaces(graph=graph, result=result)
        queue: deque[tuple[Node, int]] = deque([(root, 0)])
        visited: set[Node] = {root}

        while queue:
            node, node_depth = queue.popleft()

            for triple in graph.triples((node, None, None)):
                result.add(triple)
                _, _, obj = triple
                if node_depth < depth and isinstance(obj, (URIRef, BNode)):
                    if obj not in visited:
                        visited.add(obj)
                        queue.append((obj, node_depth + 1))

            for triple in graph.triples((None, None, node)):
                result.add(triple)
                subj, _, _ = triple
                if node_depth < depth and isinstance(subj, (URIRef, BNode)):
                    if subj not in visited:
                        visited.add(subj)
                        queue.append((subj, node_depth + 1))

        return result

    def _copy_namespaces(self, graph: RDFGraph, result: RDFGraph) -> None:
        """Preserve namespace bindings in derived patch graphs."""
        for prefix, namespace in graph.namespaces():
            if prefix:
                result.bind(prefix, namespace)
