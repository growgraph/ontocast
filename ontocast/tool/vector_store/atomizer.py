"""Graph atomization into neighborhood patches for vector indexing.

This module atomizes both ontologies and extracted facts graphs into
embedding-ready neighborhood representations.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone
from typing import Protocol, cast

from rdflib import DCTERMS, RDF, RDFS, SKOS, BNode, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.embedding_policy import (
    strip_provenance_triples_for_embedding,
)
from ontocast.onto.facts import Facts
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.onto import Tool
from ontocast.tool.representation_text import (
    normalize_text,
    normalize_uri_local_name,
    render_term_for_text,
    role_from_predicate_usage,
    stable_sorted_triples,
)
from ontocast.tool.vector_store.core import GraphAtom
from ontocast.util import render_text_hash


class GraphAtomizer(Tool):
    """Extract natural-language atoms around graph focal entities."""

    class _VectorizationSource(Protocol):
        graph: RDFGraph
        iri: str
        ontology_id: str | None
        hash: str | None
        version: str | None

    def atomize(self, source: _VectorizationSource, depth: int = 1) -> list[GraphAtom]:
        """Generate deterministic atoms from local graph neighborhoods."""
        if depth < 0:
            raise ValueError("Atomizer depth must be >= 0")

        raw_graph = source.graph
        embedding_graph = strip_provenance_triples_for_embedding(raw_graph)
        focal_namespace = source.facts_namespace if isinstance(source, Facts) else None
        entities = self._collect_focal_entities(
            graph=embedding_graph, focal_namespace=focal_namespace
        )
        generated_at = datetime.now(timezone.utc)

        atoms_by_id: dict[str, GraphAtom] = {}
        for entity in entities:
            role = self._detect_entity_role(entity=entity, graph=embedding_graph)
            patch_graph = self._build_neighborhood_graph(
                graph=embedding_graph, root=entity, depth=depth
            )
            if len(patch_graph) == 0:
                continue

            core_representation = self._build_core_representation(
                entity=entity, graph=patch_graph, role=role
            )
            neighborhood_variants = self._build_neighborhood_variants(
                entity=entity, graph=patch_graph
            )
            if not neighborhood_variants:
                neighborhood_variants = ["no neighborhood facts available"]

            for variant_index, neighborhood_representation in enumerate(
                neighborhood_variants
            ):
                atom_key = (
                    f"{source.iri}|{source.hash}|{source.version}|{entity}|"
                    f"{variant_index}|{core_representation}|{neighborhood_representation}"
                )
                atom_id = render_text_hash(atom_key, digits=None)
                if atom_id in atoms_by_id:
                    continue
                atoms_by_id[atom_id] = GraphAtom(
                    atom_id=atom_id,
                    ontology_iri=source.iri,
                    ontology_id=source.ontology_id,
                    ontology_hash=source.hash,
                    ontology_version=source.version,
                    iri=str(entity),
                    entity_role=role,
                    core_representation=core_representation,
                    neighborhood_representation=neighborhood_representation,
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

    def _collect_focal_entities(
        self, graph: RDFGraph, focal_namespace: str | None = None
    ) -> list[URIRef]:
        entities: set[URIRef] = set()
        for subj, pred, obj in graph:
            if isinstance(subj, URIRef):
                entities.add(subj)
            if isinstance(pred, URIRef):
                entities.add(pred)
            if isinstance(obj, URIRef):
                entities.add(obj)

        if focal_namespace is not None:
            ns = focal_namespace.rstrip("/")
            entities = {e for e in entities if str(e).startswith(ns)}

        return cast(
            list[URIRef],
            sorted(entities, key=lambda entity: str(entity)),
        )

    def _detect_entity_role(self, entity: URIRef, graph: RDFGraph) -> str:
        is_predicate = any(True for _ in graph.triples((None, entity, None)))
        return role_from_predicate_usage(is_predicate=is_predicate)

    def _build_core_representation(
        self, entity: URIRef, graph: RDFGraph, role: str
    ) -> str:
        labels = self._collect_literals(
            graph=graph,
            subject=entity,
            predicates=[RDFS.label, SKOS.prefLabel, DCTERMS.title],
            max_items=3,
        )
        descriptions = self._collect_literals(
            graph=graph,
            subject=entity,
            predicates=[RDFS.comment, DCTERMS.description, SKOS.definition],
            max_items=2,
        )
        types = [
            self._normalize_uri(obj)
            for _, _, obj in graph.triples((entity, RDF.type, None))
            if isinstance(obj, URIRef)
        ][:3]
        # Prefer grammatical text over schema-like field labels to improve
        # embedding alignment with natural-language queries.
        entity_name = self._normalize_uri(entity)
        sentences: list[str] = [entity_name, f"It is used as a {role}"]
        if labels:
            sentences.append(f"It is labeled {', '.join(labels)}")
        if descriptions:
            sentences.append(f"It is described as {', '.join(descriptions)}")
        if types:
            sentences.append(f"It has type {', '.join(types)}")
        return ". ".join(sentences)

    def _build_neighborhood_variants(
        self, entity: URIRef, graph: RDFGraph
    ) -> list[str]:
        by_role: dict[str, list[str]] = {
            "as_subject": [],
            "as_object": [],
            "as_predicate": [],
        }
        seen_by_role: dict[str, set[str]] = {
            "as_subject": set(),
            "as_object": set(),
            "as_predicate": set(),
        }

        triples_sorted = stable_sorted_triples(list(graph))
        for subj, pred, obj in triples_sorted:
            if subj == entity:
                sentence = self._subject_sentence(subj=subj, pred=pred, obj=obj)
                self._append_unique(by_role, seen_by_role, "as_subject", sentence)
            if obj == entity:
                sentence = self._object_sentence(subj=subj, pred=pred, obj=obj)
                self._append_unique(by_role, seen_by_role, "as_object", sentence)
            if pred == entity:
                sentence = self._predicate_sentence(subj=subj, pred=pred, obj=obj)
                self._append_unique(by_role, seen_by_role, "as_predicate", sentence)

        role_cap = 3
        max_variants = min(
            3,
            max(
                1,
                max(
                    math.ceil(len(by_role["as_subject"]) / role_cap),
                    math.ceil(len(by_role["as_object"]) / role_cap),
                    math.ceil(len(by_role["as_predicate"]) / role_cap),
                ),
            ),
        )
        variants: list[str] = []
        for variant_index in range(max_variants):
            selected: list[str] = []
            for role in ("as_subject", "as_object", "as_predicate"):
                start = variant_index * role_cap
                selected.extend(by_role[role][start : start + role_cap])
            if not selected:
                continue
            variants.append(". ".join(selected))
        return variants

    def _append_unique(
        self,
        by_role: dict[str, list[str]],
        seen_by_role: dict[str, set[str]],
        role: str,
        sentence: str,
    ) -> None:
        if sentence in seen_by_role[role]:
            return
        seen_by_role[role].add(sentence)
        by_role[role].append(sentence)

    def _subject_sentence(self, subj: Node, pred: Node, obj: Node) -> str:
        return (
            f"{self._render_term(subj)} has relation {self._render_term(pred)} "
            f"to {self._render_term(obj)}"
        )

    def _object_sentence(self, subj: Node, pred: Node, obj: Node) -> str:
        return (
            f"{self._render_term(subj)} relates via {self._render_term(pred)} "
            f"to this entity {self._render_term(obj)}"
        )

    def _predicate_sentence(self, subj: Node, pred: Node, obj: Node) -> str:
        return (
            f"predicate {self._render_term(pred)} links {self._render_term(subj)} "
            f"and {self._render_term(obj)}"
        )

    def _collect_literals(
        self, graph: RDFGraph, subject: URIRef, predicates: list[URIRef], max_items: int
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for predicate in predicates:
            for _, _, obj in graph.triples((subject, predicate, None)):
                if not isinstance(obj, Literal):
                    continue
                normalized = self._normalize_string(str(obj))
                if not normalized or normalized in seen:
                    continue
                values.append(normalized)
                seen.add(normalized)
                if len(values) >= max_items:
                    return values
        return values

    def _render_term(self, term: Node) -> str:
        return render_term_for_text(term)

    def _normalize_uri(self, uri: URIRef) -> str:
        return normalize_uri_local_name(uri)

    def _normalize_string(self, text: str) -> str:
        return normalize_text(text)
