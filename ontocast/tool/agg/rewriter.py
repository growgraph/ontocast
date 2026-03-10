"""Graph rewriting for entity disambiguation.

This module handles the robust application of entity mappings to RDF graphs,
replacing all occurrences of entities according to the mapping.

Provenance is tracked using `RDF 1.2 reification
<https://www.w3.org/TR/rdf12-concepts/>`_ together with the
`PROV-O <https://www.w3.org/TR/prov-o/>`_ vocabulary.  For every asserted
fact triple a **reifier** blank node is created::

    _:r  rdf:reifies  <<( s p o )>> .
    _:r  prov:wasDerivedFrom  <chunk_uri> .

When the same triple originates from multiple chunks the reifier
accumulates several ``prov:wasDerivedFrom`` arcs.  Chunk metadata
(``index``, ``hid``) is recorded as separate triples on the chunk URI.

The merged graph is backed by the *oxigraph* store so that RDF 1.2
triple-term syntax (``<<( s p o )>>``) is serialised correctly via
``pyoxigraph``.
"""

import logging
from collections import defaultdict

import pyoxigraph as ox
from oxrdflib._converter import to_ox
from rdflib import Literal, Node, URIRef
from rdflib.namespace import OWL, RDF, XSD

from ontocast.onto.constants import PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

# Local alias for readability
_PROV = PROV
_SCHEMA = SCHEMA


class GraphRewriter:
    """Rewrites RDF graphs by applying entity mappings.

    This class handles the robust replacement of entity URIs in RDF graphs
    according to a mapping, while preserving graph structure and metadata.
    """

    def __init__(
        self,
        add_sameas_links: bool = True,
        blocked_sameas_namespaces: tuple[str, ...] = (),
    ):
        """Initialize the graph rewriter.

        Args:
            add_sameas_links: Whether to add owl:sameAs links for merged entities
            blocked_sameas_namespaces: Namespace prefixes that should never appear
                as subject or object in emitted owl:sameAs links.
        """
        self.add_sameas_links = add_sameas_links
        self.blocked_sameas_namespaces = blocked_sameas_namespaces

    @staticmethod
    def _in_namespace(entity: URIRef, namespace: str) -> bool:
        entity_str = str(entity)
        if entity_str.startswith(namespace):
            return True
        slash_variant = namespace.rstrip("/") + "/"
        hash_variant = namespace.rstrip("#") + "#"
        return entity_str.startswith(slash_variant) or entity_str.startswith(
            hash_variant
        )

    def should_emit_sameas(self, original: URIRef, canonical: URIRef) -> bool:
        """Return whether a sameAs link is valid for emission."""
        if original == canonical:
            return False
        for namespace in self.blocked_sameas_namespaces:
            if self._in_namespace(original, namespace) or self._in_namespace(
                canonical, namespace
            ):
                return False
        return True

    def _emit_sameas_links(
        self,
        target_graph: RDFGraph,
        merged_entities: dict[URIRef, set[URIRef]],
    ) -> None:
        if not self.add_sameas_links:
            return
        for canonical, originals in merged_entities.items():
            for original in originals:
                if self.should_emit_sameas(original, canonical):
                    target_graph.add((canonical, OWL.sameAs, original))

    def apply_mapping_to_triple(
        self,
        subject: Node,
        predicate: Node,
        obj: Node,
        mapping: dict[URIRef, URIRef],
    ) -> tuple[Node, Node, Node]:
        """Apply entity mapping to a single triple.

        Args:
            subject: Triple subject
            predicate: Triple predicate
            obj: Triple object
            mapping: Entity mapping

        Returns:
            Mapped triple (subject, predicate, object)
        """
        # Map subject if needed
        new_subject = (
            mapping.get(subject, subject) if isinstance(subject, URIRef) else subject
        )

        # Map predicate if needed
        new_predicate = (
            mapping.get(predicate, predicate)
            if isinstance(predicate, URIRef)
            else predicate
        )

        # Map object if needed
        new_obj = mapping.get(obj, obj) if isinstance(obj, URIRef) else obj

        return new_subject, new_predicate, new_obj

    def rewrite_graph(self, graph: RDFGraph, mapping: dict[URIRef, URIRef]) -> RDFGraph:
        """Rewrite a graph by applying entity mapping.

        Args:
            graph: Original RDF graph
            mapping: Entity mapping (e -> e')

        Returns:
            New RDF graph with entities replaced according to mapping
        """
        rewritten = RDFGraph()

        # Copy namespace bindings
        for prefix, namespace in graph.namespaces():
            rewritten.bind(prefix, namespace)

        # Track which entities were merged for owl:sameAs links
        merged_entities: dict[URIRef, set[URIRef]] = defaultdict(set)
        for original, mapped in mapping.items():
            if original != mapped:
                merged_entities[mapped].add(original)

        # Rewrite all triples
        processed_triples = set()

        for s, p, o in graph:
            # Apply mapping
            new_s, new_p, new_o = self.apply_mapping_to_triple(s, p, o, mapping)

            # Create triple signature to avoid duplicates
            triple_sig = (new_s, new_p, new_o)

            # Skip if we've already added this triple
            if triple_sig in processed_triples:
                continue

            # Add rewritten triple
            rewritten.add(triple_sig)
            processed_triples.add(triple_sig)

        # Add owl:sameAs links for merged entities
        self._emit_sameas_links(rewritten, merged_entities)

        logger.info(
            f"Rewrote graph: {len(graph)} -> {len(rewritten)} triples "
            f"({len(merged_entities)} entities merged)"
        )

        return rewritten

    @staticmethod
    def _merge_sameas_links(
        merged_entities: dict[URIRef, set[URIRef]],
        extra_sameas_links: dict[URIRef, set[URIRef]] | None,
    ) -> dict[URIRef, set[URIRef]]:
        """Merge mapping-derived sameAs links with explicitly provided aliases."""
        if not extra_sameas_links:
            return merged_entities
        for canonical, originals in extra_sameas_links.items():
            merged_entities[canonical].update(
                original for original in originals if original != canonical
            )
        return merged_entities

    def merge_graphs(
        self,
        graphs: list[RDFGraph],
        mapping: dict[URIRef, URIRef],
        base_namespace: str,
        extra_sameas_links: dict[URIRef, set[URIRef]] | None = None,
        suppress_sameas_origins: set[URIRef] | None = None,
    ) -> RDFGraph:
        """Merge multiple graphs into one, applying entity mapping.

        Args:
            graphs: List of RDF graphs to merge.
            mapping: Entity mapping (e -> e').
            base_namespace: Base namespace for the merged graph (bound as ``facts:``).

        Returns:
            Single merged and rewritten graph.
        """
        merged = RDFGraph()

        # Bind base namespace
        merged.bind("facts", base_namespace)

        # Collect all namespaces from all graphs
        all_namespaces = {}
        for graph in graphs:
            for prefix, namespace in graph.namespaces():
                if prefix not in all_namespaces:
                    all_namespaces[prefix] = namespace
                elif all_namespaces[prefix] != namespace:
                    # Handle prefix conflicts
                    new_prefix = f"{prefix}_{len(all_namespaces)}"
                    all_namespaces[new_prefix] = namespace

        # Bind all namespaces
        for prefix, namespace in all_namespaces.items():
            merged.bind(prefix, namespace)

        # Track processed triples to avoid duplicates
        processed_triples = set()

        # Track merged entities for owl:sameAs
        merged_entities: dict[URIRef, set[URIRef]] = defaultdict(set)
        suppressed_origins = suppress_sameas_origins or set()
        for original, mapped in mapping.items():
            if original != mapped:
                if original in suppressed_origins:
                    continue
                merged_entities[mapped].add(original)

        # Merge all graphs
        for graph in graphs:
            for s, p, o in graph:
                # Apply mapping
                new_s, new_p, new_o = self.apply_mapping_to_triple(s, p, o, mapping)

                triple_sig = (new_s, new_p, new_o)

                if triple_sig not in processed_triples:
                    merged.add(triple_sig)
                    processed_triples.add(triple_sig)

        merged_entities = self._merge_sameas_links(merged_entities, extra_sameas_links)

        # Add owl:sameAs links
        self._emit_sameas_links(merged, merged_entities)

        total_original_triples = sum(len(g) for g in graphs)
        logger.info(
            f"Merged {len(graphs)} graphs: "
            f"{total_original_triples} -> {len(merged)} triples "
            f"({len(merged_entities)} entities merged)"
        )

        return merged

    # ------------------------------------------------------------------
    # provenance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ox_term(
        node: Node,
    ) -> ox.NamedNode | ox.BlankNode | ox.Literal:
        """Convert an rdflib term to a pyoxigraph term via oxrdflib.

        The ``oxrdflib._converter.to_ox`` function has a broad return
        type, but for RDF *terms* (``URIRef``, ``Literal``, ``BNode``)
        it always produces the corresponding pyoxigraph type.
        """
        result = to_ox(node)
        assert isinstance(result, (ox.NamedNode, ox.BlankNode, ox.Literal))
        return result

    def _add_unit_metadata(
        self,
        graph: RDFGraph,
        unit: ContentUnit,
    ) -> URIRef:
        """Add source-unit metadata triples and return the source unit URI.

        Emitted triples::

        <unit_iri> a prov:Entity, schema:Text ;
            schema:position <index> ;
            schema:identifier <hid> ;
            prov:generatedAtTime <datetime> .
        """

        unit_uri = URIRef(unit.iri_absolute)

        graph.add((unit_uri, RDF.type, _PROV.Entity))
        graph.add((unit_uri, RDF.type, _SCHEMA.text))
        graph.add(
            (
                unit_uri,
                _PROV.generatedAtTime,
                Literal(f"{unit.generated_at_iso}", datatype=XSD.dateTime),
            )
        )
        graph.add(
            (
                unit_uri,
                _SCHEMA.position,
                Literal(unit.index, datatype=XSD.integer),
            )
        )
        graph.add((unit_uri, _SCHEMA.identifier, Literal(unit.hid)))
        return unit_uri

    def _add_reified_provenance(
        self,
        graph: RDFGraph,
        s: Node,
        p: Node,
        o: Node,
        chunk_uri: URIRef,
        reifier: ox.BlankNode | None = None,
    ) -> ox.BlankNode:
        """Attach provenance to a triple using RDF 1.2 reification.

        Creates (or reuses) a reifier blank node and emits::

            _:r  rdf:reifies  <<( s p o )>> .
            _:r  prov:wasDerivedFrom  <chunk_uri> .

        The ``rdf:reifies`` quad is only added when a *new* reifier is
        created.  The ``prov:wasDerivedFrom`` quad is always added so that
        a shared triple accumulates one arc per source chunk.

        Args:
            graph: Oxigraph-backed :class:`RDFGraph`.
            s: Triple subject (rdflib term).
            p: Triple predicate (rdflib term).
            o: Triple object (rdflib term).
            chunk_uri: URI of the source :class:`ContentUnit`.
            reifier: Existing reifier to reuse.  When *None* a fresh
                blank node is created.

        Returns:
            The reifier blank node (for later reuse).
        """
        # Access the underlying pyoxigraph Store and graph context so
        # that triples added here are visible through the rdflib API.
        ox_store: ox.Store = graph.store._inner  # type: ignore[attr-defined]
        graph_ctx_raw = to_ox(graph.identifier)
        assert isinstance(graph_ctx_raw, (ox.NamedNode, ox.BlankNode, ox.DefaultGraph))
        graph_ctx: ox.NamedNode | ox.BlankNode | ox.DefaultGraph = graph_ctx_raw

        # Convert rdflib terms → pyoxigraph terms
        s_ox = self._to_ox_term(s)
        p_ox = self._to_ox_term(p)
        o_ox = self._to_ox_term(o)

        # Narrow types for ox.Triple (subject: NamedNode|BlankNode|Triple,
        # predicate: NamedNode, object: any ox term).
        assert isinstance(s_ox, (ox.NamedNode, ox.BlankNode))
        assert isinstance(p_ox, ox.NamedNode)

        # RDF 1.2 triple term
        triple_term = ox.Triple(s_ox, p_ox, o_ox)

        if reifier is None:
            reifier = ox.BlankNode()
            # rdf:reifies is emitted only once per reifier
            ox_store.add(
                ox.Quad(
                    reifier,
                    ox.NamedNode(str(RDF_REIFIES)),
                    triple_term,
                    graph_ctx,
                )
            )

        # prov:wasDerivedFrom — one arc per source chunk
        ox_store.add(
            ox.Quad(
                reifier,
                ox.NamedNode(str(_PROV.wasDerivedFrom)),
                ox.NamedNode(str(chunk_uri)),
                graph_ctx,
            )
        )

        return reifier

    def merge_graphs_with_provenance(
        self,
        units: list[ContentUnit],
        mapping: dict[URIRef, URIRef],
        extra_sameas_links: dict[URIRef, set[URIRef]] | None = None,
        suppress_sameas_origins: set[URIRef] | None = None,
    ) -> RDFGraph:
        """Merge multiple chunk graphs with per-triple provenance tracking.

        This method extends :meth:`merge_graphs` by:

        1. Recording **chunk metadata** (``index``, ``hid``) as separate
           triples using ``prov:Entity`` / ``schema:position`` /
           ``schema:identifier``.
        2. Creating an **RDF 1.2 reifier** node for every asserted fact
           triple using ``rdf:reifies`` with a triple term
           ``<<( s p o )>>``, and linking it back to its source chunk via
           ``prov:wasDerivedFrom``.  If the same triple is produced by
           several chunks the reifier accumulates multiple
           ``prov:wasDerivedFrom`` arcs.

        The merged graph is backed by the *oxigraph* store so that
        RDF 1.2 triple-term serialisation is available natively.

        Args:
            units: Content units whose graphs are to be merged.
            mapping: Entity mapping ``e → e'``.

        Returns:
            Merged RDF graph with RDF 1.2 provenance annotations.
        """
        merged = RDFGraph(store="oxigraph")

        # Bind well-known namespaces
        merged.bind("prov", str(_PROV))
        merged.bind("schema", str(_SCHEMA))

        # Collect all unique doc_iri namespaces and bind them
        doc_iris: set[str] = set()
        for unit in units:
            if unit.doc_iri:
                doc_iris.add(unit.doc_iri)
        for idx, doc_iri in enumerate(sorted(doc_iris)):
            prefix = f"doc{idx}" if len(doc_iris) > 1 else "doc"
            merged.bind(prefix, doc_iri.rstrip("/") + "/")

        # Collect namespaces from all source graphs
        all_namespaces: dict[str, str] = {}
        for unit in units:
            if unit.graph is None:
                continue
            for prefix, namespace in unit.graph.namespaces():
                if prefix not in all_namespaces and namespace != unit.iri:
                    all_namespaces[prefix] = namespace

        for prefix, namespace in all_namespaces.items():
            merged.bind(prefix, namespace)

        # Track processed fact triples
        processed_triples: set[tuple[Node, Node, Node]] = set()

        # Track reifier blank nodes keyed by triple signature so that a
        # shared triple accumulates multiple prov:wasDerivedFrom arcs on
        # the *same* reifier.
        reifier_map: dict[tuple[Node, Node, Node], ox.BlankNode] = {}

        # Merged-entity tracking for owl:sameAs
        merged_entities: dict[URIRef, set[URIRef]] = defaultdict(set)
        suppressed_origins = suppress_sameas_origins or set()
        for original, mapped in mapping.items():
            if original != mapped:
                if original in suppressed_origins:
                    continue
                merged_entities[mapped].add(original)

        for unit in units:
            if unit.graph is None:
                continue

            # 1. Chunk metadata
            chunk_uri = self._add_unit_metadata(merged, unit)

            # 2. Merge triples with provenance
            for s, p, o in unit.graph:
                new_s, new_p, new_o = self.apply_mapping_to_triple(
                    s,
                    p,
                    o,
                    mapping,
                )
                triple_sig = (new_s, new_p, new_o)

                # Assert fact (deduplicated)
                if triple_sig not in processed_triples:
                    merged.add(triple_sig)
                    processed_triples.add(triple_sig)

                # Attach RDF 1.2 reified provenance.
                # Reuse the existing reifier when the triple was already
                # seen from a previous chunk so that prov:wasDerivedFrom
                # arcs accumulate on the same blank node.
                existing_reifier = reifier_map.get(triple_sig)
                reifier = self._add_reified_provenance(
                    merged,
                    new_s,
                    new_p,
                    new_o,
                    chunk_uri,
                    reifier=existing_reifier,
                )
                if triple_sig not in reifier_map:
                    reifier_map[triple_sig] = reifier

        merged_entities = self._merge_sameas_links(merged_entities, extra_sameas_links)

        # owl:sameAs links
        self._emit_sameas_links(merged, merged_entities)

        total_original = sum(len(u.graph) for u in units if u.graph is not None)
        logger.info(
            f"Merged {len(units)} unit graphs with provenance: "
            f"{total_original} -> {len(merged)} triples "
            f"({len(merged_entities)} entities merged)"
        )

        return merged
