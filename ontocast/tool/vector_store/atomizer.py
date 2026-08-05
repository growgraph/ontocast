"""Graph atomization into neighborhood patches for vector indexing.

This module atomizes both ontologies and extracted facts graphs into
embedding-ready neighborhood representations.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Protocol

from pydantic import Field
from rdflib import DCTERMS, OWL, RDF, RDFS, SKOS, BNode, Literal, Namespace, URIRef
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
    role_from_declaration,
    stable_sorted_triples,
)
from ontocast.tool.vector_store.core import GraphAtom
from ontocast.tool.vector_store.lexical_trigger import (
    dedupe_preserve_case,
    looks_like_lexical_code,
)
from ontocast.util.hash import render_text_hash

# rdf:type values that add little embedding signal (OWL/RDFS scaffolding).
_GENERIC_TYPE_IRIS: frozenset[URIRef] = frozenset(
    {
        RDF.Property,
        RDFS.Class,
        RDFS.Resource,
        OWL.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.Ontology,
        OWL.NamedIndividual,
        OWL.DeprecatedClass,
        OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty,
        OWL.TransitiveProperty,
        OWL.SymmetricProperty,
        OWL.AsymmetricProperty,
        OWL.ReflexiveProperty,
        OWL.IrreflexiveProperty,
    }
)

# rdf:type values that declare an entity to be a property. OWL property
# characteristics (functional, transitive, …) are included: they are only ever
# asserted of properties, and an ontology that states a characteristic without
# also stating owl:ObjectProperty is still describing a predicate.
_PROPERTY_TYPE_IRIS: frozenset[URIRef] = frozenset(
    {
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty,
        OWL.TransitiveProperty,
        OWL.SymmetricProperty,
        OWL.AsymmetricProperty,
        OWL.ReflexiveProperty,
        OWL.IrreflexiveProperty,
    }
)

QUDT = Namespace("http://qudt.org/schema/qudt/")

# Declared surface forms, in descending priority.
_LABEL_PREDICATES: list[URIRef] = [
    RDFS.label,
    SKOS.prefLabel,
    DCTERMS.title,
    SKOS.altLabel,
]

# QUDT publishes an authoritative symbol and UCUM code per unit. Those are the forms
# prose actually uses — "meV", not "millielectronvolt" — so a corpus reporting
# measurements is queried by symbol. They are collected against their own budget rather
# than queued behind labels: a QUDT unit may declare a dozen labels (one per language),
# which would exhaust the surface-form cap before any symbol is reached.
_SYMBOL_PREDICATES: list[URIRef] = [
    SKOS.notation,
    QUDT.symbol,
    QUDT.ucumCode,
]


def _language_rank(value: Literal) -> int:
    """Sort key putting English and untagged literals ahead of other languages.

    Sorting literals alphabetically makes atomization reproducible, but it also picks
    the display name: a term declaring one label per language is named by whichever
    language happens to sort first (QUDT's ``unit:DEG_C`` declares 23 and would be named
    in Hungarian). Ranking by language first keeps the ordering total and deterministic
    while giving an English-language corpus a readable name, and spends the
    surface-form cap on forms a reader might actually type.

    Args:
        value: Literal whose language tag is inspected.

    Returns:
        int: ``0`` for untagged or English literals, ``1`` otherwise.
    """
    language = value.language
    return 0 if not language or language.lower().startswith("en") else 1


# Predicates whose objects are usually literal glosses — kept out of neighborhood clues.
_ANNOTATION_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDFS.label,
        RDFS.comment,
        RDFS.seeAlso,
        RDFS.isDefinedBy,
        SKOS.prefLabel,
        SKOS.altLabel,
        SKOS.definition,
        SKOS.hiddenLabel,
        SKOS.note,
        DCTERMS.title,
        DCTERMS.description,
        DCTERMS.abstract,
        SKOS.notation,
        QUDT.symbol,
        QUDT.ucumCode,
    }
)

# Covered by dedicated phrasing so the generic edge loop skips them.
_STRUCTURAL_PREDICATES: frozenset[URIRef] = frozenset(
    {
        RDF.type,
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        RDFS.domain,
        RDFS.range,
        OWL.inverseOf,
        OWL.equivalentClass,
        OWL.equivalentProperty,
        OWL.disjointWith,
    }
)

# Standard RDF/OWL scaffolding namespaces: focal entities under these IRIs are not embedded
# by default (ontology sources only) so casual uploads need no namespace configuration.
STANDARD_VOCABULARY_NAMESPACE_PREFIXES: frozenset[str] = frozenset(
    {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2001/XMLSchema#",
        "http://www.w3.org/2004/02/skos/core#",
        "http://purl.org/dc/elements/1.1",
        "http://purl.org/dc/terms",
        "http://www.w3.org/ns/prov#",
        "http://xmlns.com/foaf/0.1",
        "http://www.w3.org/ns/shacl#",
        "https://schema.org",
        "http://schema.org",
    }
)


def _normalize_vocab_exclude_prefix(prefix: str) -> str:
    return prefix.strip().rstrip("/")


class GraphAtomizer(Tool):
    """Extract natural-language atoms around graph focal entities.

    Two defaults narrow what an ontology contributes, both restorable:

    * Focal IRIs in common W3C and DC vocabulary namespaces are skipped (see
      module-level exclusions); ``embed_standard_vocab_iris=True`` embeds them.
    * An IRI is atomized only when this graph *describes* it — a subject-position triple
      or a label. ``index_undescribed_iris=True`` restores atomizing every URIRef,
      object-position references included.

    Facts sources are restricted to ``facts_namespace`` only; neither narrowing applies
    to them.
    """

    embed_standard_vocab_iris: bool = Field(
        default=False,
        description="If True, do not exclude standard vocabulary namespace IRIs as focal entities.",
    )
    extra_excluded_namespace_prefixes: list[str] = Field(
        default_factory=list,
        description="Additional IRI prefixes excluded from focal entities (ontology sources).",
    )
    index_undescribed_iris: bool = Field(
        default=False,
        description=(
            "If True, atomize every IRI in the graph, including ones appearing only in "
            "object or predicate position. Default False: an ontology mints an atom "
            "only for terms it describes (a subject-position triple, or a label). A "
            "referenced IRI has no local text, so its atom is its mangled local name -- "
            "'a0e0l2i0m1h0t 3d0' for a QUDT dimension vector -- and such strings embed "
            "near the corpus centroid, making them hubs that rank against every query. "
            "Measured on the 8-module matsci catalog: 247 of 690 atoms (36%) were "
            "undescribed references, and dimension vectors alone took 51 of 140 dense "
            "retrieval slots on one document, crowding four ontologies out entirely. "
            "Referenced IRIs are still reachable -- induced-subgraph expansion walks "
            "into them from seeds; they just stop being seeds themselves. Changing this "
            "changes which atoms exist and requires a reindex."
        ),
    )
    minimal_representation_label_limit: int = Field(
        default=5,
        ge=0,
        description=(
            "Maximum declared surface forms (label/prefLabel/title/altLabel) folded "
            "into the sparse BM25 representation. A vocabulary may declare more "
            "aliases than this -- symbol aliases in particular sort last and are the "
            "first to be dropped -- so raising it widens what the sparse lane can "
            "match. Changing it changes stored vectors and requires a reindex."
        ),
    )
    lexical_trigger_enabled: bool = Field(
        default=True,
        description="Collect case-preserved lexical triggers on each atom.",
    )
    lexical_trigger_predicates: list[str] = Field(
        default_factory=lambda: [
            "http://www.w3.org/2004/02/skos/core#notation",
            "http://qudt.org/schema/qudt/symbol",
            "http://qudt.org/schema/qudt/ucumCode",
        ],
        description="Predicate IRIs whose literal objects become lexical triggers.",
    )
    lexical_trigger_heuristic_enabled: bool = Field(
        default=True,
        description=(
            "Promote code-shaped labels/altLabels when no notation is declared."
        ),
    )
    lexical_trigger_min_len: int = Field(default=2, ge=1)
    lexical_trigger_max_len: int = Field(default=24, ge=1)
    lexical_trigger_heuristic_max_per_entity: int = Field(default=2, ge=0)

    class _VectorizationSource(Protocol):
        graph: RDFGraph
        iri: str
        ontology_id: str | None
        hash: str | None
        version: str | None

    def _merged_excluded_vocab_prefixes(self) -> frozenset[str]:
        extra = (
            _normalize_vocab_exclude_prefix(p)
            for p in self.extra_excluded_namespace_prefixes
        )
        return frozenset(STANDARD_VOCABULARY_NAMESPACE_PREFIXES).union(
            frozenset(p for p in extra if p)
        )

    def atomize(self, source: _VectorizationSource, depth: int = 1) -> list[GraphAtom]:
        """Generate deterministic atoms from local graph neighborhoods."""
        if depth < 0:
            raise ValueError("Atomizer depth must be >= 0")

        raw_graph = source.graph
        embedding_graph = strip_provenance_triples_for_embedding(raw_graph)
        focal_namespace = source.facts_namespace if isinstance(source, Facts) else None
        is_ontology_source = not isinstance(source, Facts)
        excluded_vocab: frozenset[str] | None = None
        if is_ontology_source and not self.embed_standard_vocab_iris:
            excluded_vocab = self._merged_excluded_vocab_prefixes()
        entities = self._collect_focal_entities(
            graph=embedding_graph,
            focal_namespace=focal_namespace,
            excluded_vocab_prefixes=excluded_vocab,
            # Facts are already confined to ``facts_namespace``, where every individual
            # is a subject; the describes-only rule targets ontology cross-references.
            require_description=is_ontology_source and not self.index_undescribed_iris,
        )
        predicate_uris = {p for (_, p, _) in embedding_graph if isinstance(p, URIRef)}
        declared_property_uris = {
            subject
            for property_type in _PROPERTY_TYPE_IRIS
            for subject in embedding_graph.subjects(RDF.type, property_type)
            if isinstance(subject, URIRef)
        }
        generated_at = datetime.now(timezone.utc)

        atoms_by_id: dict[str, GraphAtom] = {}
        seen_payload_keys: set[tuple[str, str, str, str | None, str | None]] = set()
        for entity in entities:
            role = role_from_declaration(
                is_declared_property=entity in declared_property_uris,
                is_predicate=entity in predicate_uris,
            )
            patch_graph = self._build_neighborhood_graph(
                graph=embedding_graph, root=entity, depth=depth
            )
            if len(patch_graph) == 0:
                continue

            core_representation = self._build_core_representation(
                entity=entity, graph=patch_graph, role=role
            )
            minimal_representation = self._build_minimal_representation(
                entity, embedding_graph
            )
            lexical_triggers = self._build_lexical_triggers(entity, embedding_graph)
            neighborhood_variants = self._build_neighborhood_variants(
                entity=entity, graph=patch_graph, entity_role=role
            )
            if not neighborhood_variants:
                neighborhood_variants = [""]
            # Keep first occurrence while removing repeated textual variants.
            neighborhood_variants = list(dict.fromkeys(neighborhood_variants))

            for variant_index, neighborhood_representation in enumerate(
                neighborhood_variants
            ):
                payload_key = (
                    source.iri,
                    str(entity),
                    core_representation,
                    neighborhood_representation,
                    role,
                )
                if payload_key in seen_payload_keys:
                    continue
                seen_payload_keys.add(payload_key)
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
                    minimal_representation=minimal_representation,
                    neighborhood_representation=neighborhood_representation,
                    lexical_triggers=lexical_triggers,
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

    def _describes(self, graph: RDFGraph, entity: URIRef) -> bool:
        """True when this graph says something *about* ``entity``, not merely with it.

        Subject-position triples are the primary evidence. A label alone also counts:
        a vocabulary may name a term it otherwise only references, and that name is
        exactly what retrieval needs.
        """
        for _ in graph.triples((entity, None, None)):
            return True
        return any(
            next(graph.objects(entity, predicate), None) is not None
            for predicate in _LABEL_PREDICATES
        )

    def _collect_focal_entities(
        self,
        graph: RDFGraph,
        focal_namespace: str | None = None,
        excluded_vocab_prefixes: frozenset[str] | None = None,
        require_description: bool = False,
    ) -> list[URIRef]:
        ns_prefix = focal_namespace.rstrip("/") if focal_namespace is not None else None
        entities: set[URIRef] = set()
        for subj, pred, obj in graph:
            for term in (subj, pred, obj):
                if isinstance(term, URIRef):
                    if ns_prefix is None or str(term).startswith(ns_prefix):
                        entities.add(term)

        if ns_prefix is not None:
            entities = {e for e in entities if str(e).startswith(ns_prefix)}

        if excluded_vocab_prefixes:
            entities = {
                e
                for e in entities
                if not any(str(e).startswith(p) for p in excluded_vocab_prefixes)
            }

        if require_description:
            entities = {e for e in entities if self._describes(graph, e)}

        return sorted(entities, key=lambda entity: str(entity))

    def _parent_resource_phrase(self, graph: RDFGraph, parent: URIRef) -> str:
        """Local name plus optional label gloss when it adds information."""
        base = self._normalize_uri(parent)
        literals = self._collect_surface_forms(graph, parent, 1)
        if not literals:
            return base
        gloss = literals[0]
        if gloss == base:
            return base
        return f'{base} (also described as "{gloss}")'

    def _subclass_parent_index(self, graph: RDFGraph) -> dict[URIRef, set[URIRef]]:
        parent_to_children: dict[URIRef, set[URIRef]] = defaultdict(
            lambda: set[URIRef]()
        )
        for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(child, URIRef) and isinstance(parent, URIRef):
                parent_to_children[parent].add(child)
        return parent_to_children

    def _incident_triples(
        self, graph: RDFGraph, entity: URIRef
    ) -> list[tuple[Node, Node, Node]]:
        raw: list[tuple[Node, Node, Node]] = []
        seen: set[tuple[Node, Node, Node]] = set()
        for triple in graph.triples((entity, None, None)):
            if triple not in seen:
                seen.add(triple)
                raw.append(triple)
        for triple in graph.triples((None, None, entity)):
            if triple not in seen:
                seen.add(triple)
                raw.append(triple)
        for triple in graph.triples((None, entity, None)):
            if triple not in seen:
                seen.add(triple)
                raw.append(triple)
        return stable_sorted_triples(raw)

    def _is_generic_type(self, type_uri: URIRef) -> bool:
        return type_uri in _GENERIC_TYPE_IRIS

    def _is_annotation_predicate(self, pred: URIRef) -> bool:
        return pred in _ANNOTATION_PREDICATES

    def _collect_domain_labels(
        self, entity: URIRef, graph: RDFGraph, max_items: int
    ) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for _, _, o in sorted(
            graph.triples((entity, RDFS.domain, None)), key=lambda t: str(t[2])
        ):
            if not isinstance(o, URIRef):
                continue
            text = self._normalize_uri(o)
            if text not in seen:
                seen.add(text)
                labels.append(text)
            if len(labels) >= max_items:
                break
        return labels

    def _collect_range_labels(
        self, entity: URIRef, graph: RDFGraph, max_items: int
    ) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for _, _, o in sorted(
            graph.triples((entity, RDFS.range, None)), key=lambda t: str(t[2])
        ):
            if not isinstance(o, URIRef):
                continue
            text = self._normalize_uri(o)
            if text not in seen:
                seen.add(text)
                labels.append(text)
            if len(labels) >= max_items:
                break
        return labels

    def _append_inverse_of_clues_for_property(
        self, prop_ref: URIRef, graph: RDFGraph, clues: list[str]
    ) -> None:
        for _, _, inv in sorted(
            graph.triples((prop_ref, OWL.inverseOf, None)),
            key=lambda tr: str(tr[2]),
        ):
            if isinstance(inv, URIRef):
                inv_phrase = self._parent_resource_phrase(graph, inv)
                clues.append(
                    f"{self._normalize_uri(prop_ref)} is the reverse of {inv_phrase}"
                )

    def _append_property_domain_range_clues_for_subject_resource(
        self,
        entity: URIRef,
        graph: RDFGraph,
        clues: list[str],
        *,
        max_properties: int,
        endpoint_label_cap: int,
    ) -> None:
        props_with_domain = sorted(
            {
                p
                for p, _, _ in graph.triples((None, RDFS.domain, entity))
                if isinstance(p, URIRef)
            },
            key=str,
        )[:max_properties]
        for prop in props_with_domain:
            prop_verb = self._normalize_uri(prop)  # bare verb for SPO
            ranges = self._collect_range_labels(
                prop, graph, max_items=endpoint_label_cap
            )
            for r_label in ranges or ["something"]:
                clues.append(f"it {prop_verb} {r_label}")
            self._append_inverse_of_clues_for_property(prop, graph, clues)

        props_with_range = sorted(
            {
                p
                for p, _, _ in graph.triples((None, RDFS.range, entity))
                if isinstance(p, URIRef)
            },
            key=str,
        )[:max_properties]
        for prop in props_with_range:
            prop_verb = self._normalize_uri(prop)  # bare verb for SPO
            domains = self._collect_domain_labels(
                prop, graph, max_items=endpoint_label_cap
            )
            for d_label in domains or ["something"]:
                clues.append(f"{d_label} {prop_verb} it")
            self._append_inverse_of_clues_for_property(prop, graph, clues)

    def _build_minimal_representation(
        self, entity: URIRef, graph: RDFGraph | None = None
    ) -> str:
        """Keyword-oriented text for the sparse BM25 lane.

        The IRI local name (camelCase/PascalCase split, see ``normalize_uri_local_name``)
        plus any human labels. Lexical match is the strongest available signal for
        technical vocabulary that appears near-verbatim in source text, but an IRI local
        name is often an opaque identifier — Wikidata-derived ``Q36834`` carries no
        tokens at all, and the term is only findable through its ``rdfs:label``.
        Descriptions are deliberately excluded: they would dominate term frequency
        without naming the entity.
        """
        local_name = normalize_uri_local_name(entity)
        if graph is None:
            return local_name
        labels = self._collect_surface_forms(
            graph,
            entity,
            self.minimal_representation_label_limit,
            lead_with_symbol=True,
        )
        parts = [local_name, *labels]
        seen: set[str] = set()
        tokens: list[str] = []
        for part in parts:
            normalized = normalize_text(part)
            if normalized and normalized not in seen:
                seen.add(normalized)
                tokens.append(normalized)
        return " ".join(tokens)

    def _build_core_representation(
        self, entity: URIRef, graph: RDFGraph, role: str
    ) -> str:
        labels = self._collect_surface_forms(graph, entity, 5)
        descriptions = self._collect_literals(
            graph, entity, [RDFS.comment, DCTERMS.description, SKOS.definition], 2
        )
        informative_types = []
        for _, _, obj in sorted(
            graph.triples((entity, RDF.type, None)), key=lambda t: str(t[2])
        ):
            if not isinstance(obj, URIRef) or self._is_generic_type(obj):
                continue
            informative_types.append(self._normalize_uri(obj))
            if len(informative_types) >= 3:
                break

        entity_name = labels[0] if labels else self._normalize_uri(entity)
        parts: list[str] = [entity_name]

        if informative_types:
            parts[0] += f" ({', '.join(informative_types)})"

        if len(labels) > 1:
            parts.append(f"Also known as {', '.join(labels[1:])}")

        parts.extend(descriptions)

        domains = self._collect_domain_labels(entity, graph, max_items=4)
        ranges = self._collect_range_labels(entity, graph, max_items=4)
        if domains:
            parts.append(f"Applies to: {', '.join(domains)}")
        if ranges:
            parts.append(f"Values restricted to: {', '.join(ranges)}")

        return ". ".join(parts)

    def _collect_structural_clues(
        self, entity: URIRef, graph: RDFGraph, entity_role: str
    ) -> list[str]:
        clues: list[str] = []
        focal_is_property = entity_role == "predicate"

        for _, _, t in sorted(
            graph.triples((entity, RDF.type, None)), key=lambda tr: str(tr[2])
        ):
            if isinstance(t, URIRef) and not self._is_generic_type(t):
                clues.append(f"it is a {self._normalize_uri(t)}")

        if not focal_is_property:
            parent_to_children = self._subclass_parent_index(graph)

            for _, _, parent in sorted(
                graph.triples((entity, RDFS.subClassOf, None)),
                key=lambda tr: str(tr[2]),
            ):
                if isinstance(parent, URIRef):
                    clues.append(
                        f"it is a kind of {self._parent_resource_phrase(graph, parent)}"
                    )

            for child, _, _ in sorted(
                graph.triples((None, RDFS.subClassOf, entity)),
                key=lambda tr: str(tr[0]),
            ):
                if isinstance(child, URIRef):
                    clues.append(f"{self._normalize_uri(child)} is a kind of it")

            parents = [
                o
                for _, _, o in graph.triples((entity, RDFS.subClassOf, None))
                if isinstance(o, URIRef)
            ]
            for par in sorted(set(parents), key=str):
                siblings = sorted(
                    (
                        sib
                        for sib in parent_to_children.get(par, set[URIRef]())
                        if sib != entity
                    ),
                    key=str,
                )
                for sib in siblings[:6]:
                    clues.append(
                        f"{self._normalize_uri(sib)} is also a kind of "
                        f"{self._parent_resource_phrase(graph, par)}"
                    )

            self._append_property_domain_range_clues_for_subject_resource(
                entity=entity,
                graph=graph,
                clues=clues,
                max_properties=8,
                endpoint_label_cap=3,
            )

        for _, _, other in sorted(
            graph.triples((entity, OWL.equivalentClass, None)),
            key=lambda tr: str(tr[2]),
        ):
            if isinstance(other, URIRef):
                clues.append(f"it means the same as {self._normalize_uri(other)}")

        for _, _, other in sorted(
            graph.triples((entity, OWL.disjointWith, None)),
            key=lambda tr: str(tr[2]),
        ):
            if isinstance(other, URIRef):
                clues.append(f"it never overlaps with {self._normalize_uri(other)}")

        for _, _, other in sorted(
            graph.triples((entity, OWL.equivalentProperty, None)),
            key=lambda tr: str(tr[2]),
        ):
            if isinstance(other, URIRef):
                clues.append(f"it means the same as {self._normalize_uri(other)}")

        if focal_is_property:
            for _, _, parent in sorted(
                graph.triples((entity, RDFS.subPropertyOf, None)),
                key=lambda tr: str(tr[2]),
            ):
                if isinstance(parent, URIRef):
                    clues.append(
                        f"it is a narrower form of {self._parent_resource_phrase(graph, parent)}"
                    )

            for child, _, _ in sorted(
                graph.triples((None, RDFS.subPropertyOf, entity)),
                key=lambda tr: str(tr[0]),
            ):
                if isinstance(child, URIRef):
                    clues.append(
                        f"{self._normalize_uri(child)} is a narrower form of it"
                    )

            for _, _, inv in sorted(
                graph.triples((entity, OWL.inverseOf, None)),
                key=lambda tr: str(tr[2]),
            ):
                if isinstance(inv, URIRef):
                    clues.append(f"it is the reverse of {self._normalize_uri(inv)}")

            for d in self._collect_domain_labels(entity, graph, max_items=3):
                clues.append(f"it applies to {d}")
            for r in self._collect_range_labels(entity, graph, max_items=3):
                clues.append(f"it yields {r}")

        for subj, pred, obj in self._incident_triples(graph, entity):
            if not isinstance(pred, URIRef):
                continue
            if self._is_annotation_predicate(pred):
                continue
            if pred in _STRUCTURAL_PREDICATES:
                continue

            pred_phrase = self._normalize_uri(pred)

            if pred == entity:
                if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                    clues.append(
                        f"{self._normalize_uri(subj)} it {self._normalize_uri(obj)}"
                    )
                continue

            if subj == entity:
                if not isinstance(obj, URIRef):
                    continue
                clues.append(f"it {pred_phrase} {self._normalize_uri(obj)}")
            elif obj == entity:
                if not isinstance(subj, URIRef):
                    continue
                clues.append(f"{self._normalize_uri(subj)} {pred_phrase} it")

        return sorted(set(clues))

    def _build_neighborhood_variants(
        self, entity: URIRef, graph: RDFGraph, entity_role: str
    ) -> list[str]:
        clues = self._collect_structural_clues(
            entity=entity, graph=graph, entity_role=entity_role
        )
        if not clues:
            return []
        # Temporary simplification: emit a single deterministic neighborhood view.
        return [". ".join(clues)]

    def _collect_literals(
        self, graph: RDFGraph, subject: URIRef, predicates: list[URIRef], max_items: int
    ) -> list[str]:
        """Collect literal surface forms, deterministically, in predicate priority order.

        ``graph.triples`` yields in unspecified order, so truncating its output at
        ``max_items`` picked an arbitrary subset of a term's labels: a term declaring
        more aliases than the cap allows would embed differently between runs over
        identical input, which makes retrieval measurements irreproducible. Values are
        sorted within each predicate before truncation; predicate order is still
        honoured, keeping ``rdfs:label`` ahead of ``skos:altLabel``.

        Args:
            graph: Graph to read literals from.
            subject: Subject whose literals are collected.
            predicates: Predicates to read, in descending priority.
            max_items: Maximum number of distinct values to return.

        Returns:
            list[str]: Normalized literal values, at most ``max_items``.
        """
        values: list[str] = []
        seen: set[str] = set()
        for predicate in predicates:
            candidates = sorted(
                {
                    (_language_rank(obj), normalized)
                    for _, _, obj in graph.triples((subject, predicate, None))
                    if isinstance(obj, Literal)
                    and (normalized := self._normalize_string(str(obj)))
                }
            )
            for _, normalized in candidates:
                if normalized in seen:
                    continue
                values.append(normalized)
                seen.add(normalized)
                if len(values) >= max_items:
                    return values
        return values

    def _collect_surface_forms(
        self,
        graph: RDFGraph,
        subject: URIRef,
        max_items: int,
        *,
        lead_with_symbol: bool = False,
    ) -> list[str]:
        """Declared labels plus QUDT symbols, with symbols guaranteed a slot.

        ``_collect_literals`` honours predicate priority, so appending the symbol
        predicates to the label list would let a term that declares many labels crowd
        the symbols out entirely — and QUDT units routinely declare one label per
        language. The two families are therefore collected against separate budgets and
        merged, so a unit stays findable by the symbol a reader actually types.

        Args:
            graph: Graph to read literals from.
            subject: Entity whose surface forms are collected.
            max_items: Maximum number of distinct values to return.
            lead_with_symbol: Put symbols first, for the sparse lexical lane. When
                ``False`` the primary label leads so the entity keeps a readable name.

        Returns:
            list[str]: Normalized surface forms, at most ``max_items``.
        """
        labels = self._collect_literals(graph, subject, _LABEL_PREDICATES, max_items)
        symbols = self._collect_literals(graph, subject, _SYMBOL_PREDICATES, max_items)
        if not symbols:
            return labels[:max_items]
        if lead_with_symbol:
            ordered = [*symbols, *labels]
        else:
            ordered = [*labels[:1], *symbols, *labels[1:]]

        merged: list[str] = []
        seen: set[str] = set()
        for value in ordered:
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
            if len(merged) >= max_items:
                break
        return merged

    def _resolved_lexical_trigger_predicates(self) -> list[URIRef]:
        return [URIRef(iri) for iri in self.lexical_trigger_predicates if iri.strip()]

    def _collect_raw_literals(
        self,
        graph: RDFGraph,
        subject: URIRef,
        predicates: list[URIRef],
        max_items: int,
    ) -> list[str]:
        """Collect literal values preserving original case (for lexical triggers)."""
        values: list[str] = []
        seen: set[str] = set()
        for predicate in predicates:
            candidates = sorted(
                {
                    (_language_rank(obj), str(obj).strip())
                    for _, _, obj in graph.triples((subject, predicate, None))
                    if isinstance(obj, Literal) and str(obj).strip()
                }
            )
            for _, raw in candidates:
                if raw in seen:
                    continue
                values.append(raw)
                seen.add(raw)
                if len(values) >= max_items:
                    return values
        return values

    def _build_lexical_triggers(self, entity: URIRef, graph: RDFGraph) -> list[str]:
        if not self.lexical_trigger_enabled:
            return []
        predicate_iris = self._resolved_lexical_trigger_predicates()
        declared = self._collect_raw_literals(
            graph, entity, predicate_iris, max_items=16
        )
        if declared:
            return dedupe_preserve_case(declared)

        if not self.lexical_trigger_heuristic_enabled:
            return []

        heuristic: list[str] = []
        for candidate in self._collect_raw_literals(
            graph,
            entity,
            [RDFS.label, SKOS.altLabel],
            max_items=self.lexical_trigger_heuristic_max_per_entity + 4,
        ):
            if looks_like_lexical_code(
                candidate,
                min_len=self.lexical_trigger_min_len,
                max_len=self.lexical_trigger_max_len,
            ):
                heuristic.append(candidate)
            if len(heuristic) >= self.lexical_trigger_heuristic_max_per_entity:
                break
        return dedupe_preserve_case(heuristic)

    def _normalize_uri(self, uri: URIRef) -> str:
        return normalize_uri_local_name(uri)

    def _normalize_string(self, text: str) -> str:
        return normalize_text(text)
