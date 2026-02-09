"""Tests for the embedding-based aggregator pipeline.

Isolated tests for each stage of the aggregation pipeline:
1. Entity normalisation
2. Clustering & representative selection
3. URI building (naming conventions)
4. Graph rewriting
5. End-to-end aggregation
6. Multi-chunk aggregation with provenance & doc_iri mapping (RDF 1.2)
"""

from typing import cast
from unittest.mock import Mock

from rdflib import OWL, RDF, RDFS, Literal, Namespace, URIRef
from rdflib.namespace import XSD

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.clustering import ClusterRepresentativeSelector
from ontocast.tool.agg.normalizer import EntityNormalizer, EntityRepresentation
from ontocast.tool.agg.rewriter import GraphRewriter
from ontocast.tool.agg.uri_builder import (
    EntityRole,
    URIBuilder,
    detect_role,
    format_structured_id,
    has_structured_id,
    normalize_local_name,
    to_lower_camel_case,
    to_pascal_case,
)

# ---------------------------------------------------------------------------
# Stage 1: Entity Normalisation
# ---------------------------------------------------------------------------


class TestEntityNormalizer:
    """Test entity normalisation (stage 1)."""

    def test_normalize_string_camel_case(self, normalizer: EntityNormalizer) -> None:
        """CamelCase is split into lowercase tokens."""
        assert normalizer.normalize_string("PLRedShift") == "pl red shift"

    def test_normalize_string_snake_case(self, normalizer: EntityNormalizer) -> None:
        """Snake_case is split into lowercase tokens."""
        assert normalizer.normalize_string("PL_red_shift_value") == "pl red shift value"

    def test_normalize_string_diacritics(self, normalizer: EntityNormalizer) -> None:
        """Diacritics are removed."""
        assert normalizer.normalize_string("Café") == "cafe"

    def test_normalize_uri_camel(self, normalizer: EntityNormalizer) -> None:
        """URI local part is normalised (camelCase)."""
        uri = URIRef("http://example.org/PLRedShift")
        assert normalizer.normalize_uri(uri) == "pl red shift"

    def test_normalize_uri_snake(self, normalizer: EntityNormalizer) -> None:
        """URI local part is normalised (snake_case)."""
        uri = URIRef("http://example.org/PL_red_shift_value")
        assert normalizer.normalize_uri(uri) == "pl red shift value"

    def test_is_ontology_entity(self, normalizer: EntityNormalizer) -> None:
        """Ontology namespace detection works correctly."""
        assert (
            normalizer.is_ontology_entity(URIRef("http://ontology.org/Thing")) is True
        )
        assert normalizer.is_ontology_entity(URIRef(f"{DEFAULT_IRI}/entity")) is False

    def test_create_representation_has_types(
        self, normalizer: EntityNormalizer
    ) -> None:
        """Representation captures rdf:type information."""
        g = RDFGraph()
        EX = Namespace("http://example.org/")
        ONT = Namespace("http://ontology.org/")

        entity = EX.TestEntity
        g.add((entity, RDF.type, ONT.Thing))
        g.add((entity, RDFS.label, Literal("Test Entity")))
        g.add((entity, EX.hasValue, Literal("123")))

        rep = normalizer.create_representation(entity, g)

        assert rep.entity == entity
        assert "test entity" in rep.normal_form
        assert len(rep.types) == 1
        assert rep.types[0] == ONT.Thing
        assert "Test Entity" in rep.labels
        assert EX.hasValue in rep.properties
        assert "type" in rep.representation

    def test_create_representation_ontology_flag(
        self, normalizer: EntityNormalizer
    ) -> None:
        """Ontology entities are flagged correctly."""
        g = RDFGraph()
        ONT = Namespace("http://ontology.org/")
        entity = ONT.SomeClass
        g.add((entity, RDF.type, RDFS.Class))

        rep = normalizer.create_representation(entity, g)
        assert rep.is_ontology_entity is True


# ---------------------------------------------------------------------------
# Stage 2: Clustering & Representative Selection
# ---------------------------------------------------------------------------


class TestClusterRepresentativeSelector:
    """Test representative selection (stage 2)."""

    def test_simplicity_score_ordering(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Simple URIs score lower than complex URIs."""
        simple = URIRef("http://ex.org/Thing")
        complex_ = URIRef("http://example.org/deeply/nested/path/ComplexEntity_123")

        s_simple = cluster_representative_selector.compute_simplicity_score(simple)
        s_complex = cluster_representative_selector.compute_simplicity_score(complex_)
        assert s_simple < s_complex

    def test_prefers_ontology_entity(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Ontology entities are preferred over chunk entities."""
        ont_entity = URIRef("http://ontology.org/Thing")
        chunk_entity = URIRef(f"{DEFAULT_IRI}/entity_long_name")

        ont_rep = Mock(is_ontology_entity=True)
        chunk_rep = Mock(is_ontology_entity=False)

        reps = cast(
            dict[URIRef, EntityRepresentation],
            {ont_entity: ont_rep, chunk_entity: chunk_rep},
        )
        selected = cluster_representative_selector.select_representative(
            [ont_entity, chunk_entity], reps
        )
        assert selected == ont_entity

    def test_prefers_simpler_when_no_ontology(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Among non-ontology entities, simpler URIs win."""
        simple = URIRef("http://chunk1.org/Thing")
        complex_ = URIRef("http://chunk2.org/very_long_complex_entity_name_123")

        simple_rep = Mock(is_ontology_entity=False)
        complex_rep = Mock(is_ontology_entity=False)

        reps = cast(
            dict[URIRef, EntityRepresentation],
            {simple: simple_rep, complex_: complex_rep},
        )
        selected = cluster_representative_selector.select_representative(
            [simple, complex_], reps
        )
        assert selected == simple

    def test_singleton_cluster(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Singleton clusters return the only entity."""
        entity = URIRef("http://chunk1.org/Only")
        rep = Mock(is_ontology_entity=False)
        reps = cast(dict[URIRef, EntityRepresentation], {entity: rep})

        selected = cluster_representative_selector.select_representative([entity], reps)
        assert selected == entity

    def test_create_mapping(
        self, cluster_representative_selector: ClusterRepresentativeSelector
    ) -> None:
        """Mapping assigns every entity to its cluster representative."""
        e1 = URIRef("http://chunk1.org/A")
        e2 = URIRef("http://chunk1.org/B")
        e3 = URIRef("http://chunk2.org/C")

        rep1 = Mock(is_ontology_entity=False)
        rep2 = Mock(is_ontology_entity=False)
        rep3 = Mock(is_ontology_entity=False)

        reps = cast(
            dict[URIRef, EntityRepresentation],
            {e1: rep1, e2: rep2, e3: rep3},
        )
        clusters = [[e1, e2], [e3]]
        mapping = cluster_representative_selector.create_mapping(clusters, reps)

        # e1 and e2 map to the same representative
        assert mapping[e1] == mapping[e2]
        # e3 maps to itself
        assert mapping[e3] == e3


# ---------------------------------------------------------------------------
# Stage 3: URI Building (Naming Conventions)
# ---------------------------------------------------------------------------


class TestNamingConventions:
    """Test pure naming-convention functions."""

    def test_to_pascal_case(self) -> None:
        assert to_pascal_case("judicial decision") == "JudicialDecision"
        assert to_pascal_case("french court of cassation") == "FrenchCourtOfCassation"
        assert to_pascal_case("case") == "Case"
        assert to_pascal_case("") == ""

    def test_to_lower_camel_case(self) -> None:
        assert to_lower_camel_case("has decision") == "hasDecision"
        assert to_lower_camel_case("date published") == "datePublished"
        assert to_lower_camel_case("name") == "name"
        assert to_lower_camel_case("") == ""

    def test_has_structured_id(self) -> None:
        assert has_structured_id(URIRef("http://ex.org/Case_2023_456")) is True
        assert has_structured_id(URIRef("http://ex.org/Decision_2021_09_15")) is True
        assert has_structured_id(URIRef("http://ex.org/Person")) is False
        assert (
            has_structured_id(URIRef("http://ex.org/Item123")) is False
        )  # no underscore

    def test_format_structured_id(self) -> None:
        assert (
            format_structured_id(URIRef("http://ex.org/case_2023_456"))
            == "Case_2023_456"
        )
        assert (
            format_structured_id(URIRef("http://ex.org/Decision_2021_09_15"))
            == "Decision_2021_09_15"
        )


class TestDetectRole:
    """Test entity role detection from graph context."""

    def test_detect_class(self) -> None:
        g = RDFGraph()
        entity = URIRef("http://ex.org/Person")
        g.add((entity, RDF.type, RDFS.Class))
        assert detect_role(entity, g) == EntityRole.CLASS

    def test_detect_owl_class(self) -> None:
        g = RDFGraph()
        entity = URIRef("http://ex.org/Person")
        g.add((entity, RDF.type, OWL.Class))
        assert detect_role(entity, g) == EntityRole.CLASS

    def test_detect_property_by_type(self) -> None:
        g = RDFGraph()
        entity = URIRef("http://ex.org/hasAge")
        g.add((entity, RDF.type, OWL.DatatypeProperty))
        assert detect_role(entity, g) == EntityRole.PROPERTY

    def test_detect_property_by_usage(self) -> None:
        """Entity used as predicate is detected as property."""
        g = RDFGraph()
        subj = URIRef("http://ex.org/Alice")
        pred = URIRef("http://ex.org/knows")
        obj = URIRef("http://ex.org/Bob")
        g.add((subj, pred, obj))
        assert detect_role(pred, g) == EntityRole.PROPERTY

    def test_detect_instance(self) -> None:
        g = RDFGraph()
        entity = URIRef("http://ex.org/Alice")
        g.add((entity, RDF.type, URIRef("http://ex.org/Person")))
        assert detect_role(entity, g) == EntityRole.INSTANCE

    def test_detect_instance_no_type(self) -> None:
        """Entity with no type info defaults to instance."""
        g = RDFGraph()
        entity = URIRef("http://ex.org/Unknown")
        assert detect_role(entity, g) == EntityRole.INSTANCE


class TestNormalizeLocalName:
    """Test normalize_local_name which combines role detection with naming."""

    def _make_rep(self, uri: str, normal_form: str) -> EntityRepresentation:
        return EntityRepresentation(
            entity=URIRef(uri),
            normal_form=normal_form,
            types=[],
            properties=[],
            labels=[],
            representation=normal_form,
            is_ontology_entity=False,
        )

    def test_class_gets_pascal_case(self) -> None:
        rep = self._make_rep("http://ex.org/JudicialDecision", "judicial decision")
        assert normalize_local_name(rep, EntityRole.CLASS) == "JudicialDecision"

    def test_property_gets_lower_camel_case(self) -> None:
        rep = self._make_rep("http://ex.org/hasDecision", "has decision")
        assert normalize_local_name(rep, EntityRole.PROPERTY) == "hasDecision"

    def test_instance_natural_name_gets_pascal_case(self) -> None:
        rep = self._make_rep("http://ex.org/FrenchCourt", "french court")
        assert normalize_local_name(rep, EntityRole.INSTANCE) == "FrenchCourt"

    def test_instance_structured_id_preserves_structure(self) -> None:
        rep = self._make_rep("http://ex.org/Case_2023_456", "case 2023 456")
        assert normalize_local_name(rep, EntityRole.INSTANCE) == "Case_2023_456"


class TestURIBuilder:
    """Test the URIBuilder class."""

    def test_build_uri_fact_entity(self, uri_builder: URIBuilder) -> None:
        """Fact entities (under base_iri) get URIs under DEFAULT_IRI."""
        rep = EntityRepresentation(
            entity=URIRef(f"{DEFAULT_IRI}/testEntity"),
            normal_form="test entity",
            types=[],
            properties=[],
            labels=[],
            representation="test entity",
            is_ontology_entity=False,
        )
        result = uri_builder.build_uri(rep.entity, rep, EntityRole.INSTANCE)
        assert str(result).startswith(DEFAULT_IRI)
        assert "TestEntity" in str(result)

    def test_build_uri_ontology_entity_preserved(self, uri_builder: URIBuilder) -> None:
        """Ontology entities are returned unchanged."""
        ont_entity = URIRef("http://ontology.org/Thing")
        rep = EntityRepresentation(
            entity=ont_entity,
            normal_form="thing",
            types=[],
            properties=[],
            labels=[],
            representation="thing",
            is_ontology_entity=True,
        )
        result = uri_builder.build_uri(ont_entity, rep, "class")
        assert result == ont_entity

    def test_build_uri_uniqueness(self, uri_builder: URIBuilder) -> None:
        """Duplicate names get suffixed for uniqueness."""
        rep1 = EntityRepresentation(
            entity=URIRef("http://chunk1.org/person"),
            normal_form="person",
            types=[],
            properties=[],
            labels=[],
            representation="person",
            is_ontology_entity=False,
        )
        rep2 = EntityRepresentation(
            entity=URIRef("http://chunk2.org/person"),
            normal_form="person",
            types=[],
            properties=[],
            labels=[],
            representation="person",
            is_ontology_entity=False,
        )
        uri1 = uri_builder.build_uri(rep1.entity, rep1, "class")
        uri2 = uri_builder.build_uri(rep2.entity, rep2, "class")
        assert uri1 != uri2
        assert "Person" in str(uri1)
        assert "Person" in str(uri2)

    def test_create_uri_mapping(self, uri_builder: URIBuilder) -> None:
        """create_uri_mapping produces a complete mapping."""
        EX = Namespace("http://chunk1.org/")
        entity = EX.TestClass

        rep = EntityRepresentation(
            entity=entity,
            normal_form="test class",
            types=[RDFS.Class],
            properties=[RDF.type],
            labels=[],
            representation="test class type class",
            is_ontology_entity=False,
            role=EntityRole.CLASS,
        )
        mapping = uri_builder.create_uri_mapping([entity], {entity: rep})
        assert entity in mapping
        assert "TestClass" in str(mapping[entity])

    def test_compose_mappings(self) -> None:
        """Mapping composition chains clustering → URI building."""
        e1 = URIRef("http://chunk1.org/A")
        e2 = URIRef("http://chunk2.org/B")
        rep = URIRef("http://chunk1.org/A")
        final = URIRef(f"{DEFAULT_IRI}/SomeEntity")

        clustering = {e1: rep, e2: rep}
        uri_map = {rep: final}

        composed = URIBuilder.compose_mappings(clustering, uri_map)
        assert composed[e1] == final
        assert composed[e2] == final


# ---------------------------------------------------------------------------
# Stage 4: Graph Rewriting
# ---------------------------------------------------------------------------


class TestGraphRewriter:
    """Test graph rewriting (stage 4)."""

    def test_apply_mapping_to_triple(self, graph_rewriter: GraphRewriter) -> None:
        """Individual triple mapping works."""
        e1 = URIRef("http://chunk1.org/e1")
        p1 = URIRef("http://chunk1.org/p1")
        e2 = URIRef("http://chunk1.org/e2")

        e1p = URIRef(f"{DEFAULT_IRI}/Entity1")
        p1p = URIRef(f"{DEFAULT_IRI}/property1")
        e2p = URIRef(f"{DEFAULT_IRI}/Entity2")

        mapping = {e1: e1p, p1: p1p, e2: e2p}
        new_s, new_p, new_o = graph_rewriter.apply_mapping_to_triple(
            e1, p1, e2, mapping
        )
        assert new_s == e1p
        assert new_p == p1p
        assert new_o == e2p

    def test_preserve_ontology_types(self, graph_rewriter: GraphRewriter) -> None:
        """Ontology types are preserved in rdf:type triples."""
        entity = URIRef("http://chunk1.org/entity")
        ontology_type = URIRef("http://ontology.org/Thing")
        entity_p = URIRef(f"{DEFAULT_IRI}/Entity")

        mapping = {entity: entity_p}

        new_s, new_p, new_o = graph_rewriter.apply_mapping_to_triple(
            entity, RDF.type, ontology_type, mapping
        )
        assert new_s == entity_p
        assert new_p == RDF.type
        assert new_o == ontology_type  # preserved

    def test_rewrite_graph(self, graph_rewriter: GraphRewriter) -> None:
        """Full graph rewriting applies mapping to all triples."""
        g = RDFGraph()
        e1 = URIRef("http://chunk1.org/e1")
        e2 = URIRef("http://chunk1.org/e2")
        p = URIRef("http://chunk1.org/p")
        ont_type = URIRef("http://ontology.org/Thing")

        g.add((e1, p, e2))
        g.add((e1, RDF.type, ont_type))

        e1p = URIRef(f"{DEFAULT_IRI}/Entity1")
        e2p = URIRef(f"{DEFAULT_IRI}/Entity2")
        pp = URIRef(f"{DEFAULT_IRI}/relatesTo")

        mapping = {e1: e1p, e2: e2p, p: pp}
        rewritten = graph_rewriter.rewrite_graph(g, mapping)

        assert (e1p, pp, e2p) in rewritten
        assert (e1p, RDF.type, ont_type) in rewritten

    def test_merge_graphs_deduplicates(self, graph_rewriter: GraphRewriter) -> None:
        """Merging two graphs with same mapped triples removes duplicates."""
        g1 = RDFGraph()
        g2 = RDFGraph()
        e = URIRef("http://chunk1.org/e")
        p = URIRef("http://chunk1.org/p")
        o = Literal("value")

        ep = URIRef(f"{DEFAULT_IRI}/Entity")
        pp = URIRef(f"{DEFAULT_IRI}/hasValue")

        g1.add((e, p, o))
        g2.add((e, p, o))

        mapping = {e: ep, p: pp}
        merged = graph_rewriter.merge_graphs([g1, g2], mapping, DEFAULT_IRI)

        # Should have exactly 1 triple (deduplicated)
        triples = list(merged.triples((ep, pp, o)))
        assert len(triples) == 1

    def test_sameas_links_added(self, graph_rewriter: GraphRewriter) -> None:
        """owl:sameAs links are added for merged entities."""
        g = RDFGraph()
        e1 = URIRef("http://chunk1.org/e1")
        e2 = URIRef("http://chunk2.org/e2")
        p = URIRef("http://chunk1.org/p")

        g.add((e1, p, Literal("a")))
        g.add((e2, p, Literal("b")))

        canonical = URIRef(f"{DEFAULT_IRI}/Entity")
        mapping = {e1: canonical, e2: canonical}
        rewritten = graph_rewriter.rewrite_graph(g, mapping)

        sameas_triples = list(rewritten.triples((canonical, OWL.sameAs, None)))
        assert len(sameas_triples) >= 1


# ---------------------------------------------------------------------------
# Stage 5: End-to-End Aggregation
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Test end-to-end aggregation with ContentUnit types."""

    def _make_fact_unit(
        self, text: str, hid: str, doc_iri: str, ttl: str
    ) -> ContentUnit:
        """Create a fact-type ContentUnit with a graph."""
        graph = RDFGraph()
        graph.parse(data=ttl, format="turtle")
        return ContentUnit(
            text=text,
            index=0,
            hid=hid,
            doc_iri=doc_iri,
            graph=graph,
            type=OutputType.FACTS,
        )

    def _make_ontology_unit(
        self, text: str, hid: str, doc_iri: str, ttl: str
    ) -> ContentUnit:
        """Create an ontology-type ContentUnit with a graph."""
        graph = RDFGraph()
        graph.parse(data=ttl, format="turtle")
        return ContentUnit(
            text=text,
            index=0,
            hid=hid,
            doc_iri=doc_iri,
            graph=graph,
            type=OutputType.ONTOLOGIES,
        )

    def test_empty_units(self) -> None:
        """Aggregating zero units returns empty graph."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs([])
        assert len(result) == 0

    def test_fact_units_produce_default_iri_namespace(self) -> None:
        """Fact-type units get URIs under DEFAULT_IRI."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        ttl = f"""
        @prefix facts: <{DEFAULT_IRI}/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        facts:Alice rdf:type facts:Person .
        facts:Alice rdfs:label "Alice" .
        """
        unit = self._make_fact_unit("Alice is a person", "h1", DEFAULT_IRI, ttl)
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs([unit])

        # Result should have triples
        assert len(result) > 0

        # All subject URIs should be under DEFAULT_IRI or standard namespaces
        for s, _p, _o in result:
            s_str = str(s)
            if not s_str.startswith("http://www.w3.org"):
                assert s_str.startswith(DEFAULT_IRI), (
                    f"Expected URI under {DEFAULT_IRI}, got {s_str}"
                )

    def test_content_unit_type_field(self) -> None:
        """ContentUnit type field distinguishes facts from ontology."""
        fact = ContentUnit(
            text="t",
            index=0,
            hid="h1",
            doc_iri="http://ex.org/doc1",
            type=OutputType.FACTS,
        )
        onto = ContentUnit(
            text="t",
            index=0,
            hid="h2",
            doc_iri="http://ex.org/doc1",
            type=OutputType.ONTOLOGIES,
        )
        assert fact.type == OutputType.FACTS
        assert onto.type == OutputType.ONTOLOGIES


# ---------------------------------------------------------------------------
# Stage 6: Multi-chunk Aggregation, Provenance & doc_iri Mapping
# ---------------------------------------------------------------------------


def _make_fact_unit(
    text: str,
    index: int,
    hid: str,
    doc_iri: str,
    ttl: str,
) -> ContentUnit:
    """Helper: create a fact-type ContentUnit with a parsed Turtle graph."""
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return ContentUnit(
        text=text,
        index=index,
        hid=hid,
        doc_iri=doc_iri,
        graph=graph,
        type=OutputType.FACTS,
    )


class TestMultiChunkAggregation:
    """Test aggregate_graphs with multiple chunks from the same document."""

    DOC_IRI = "https://example.org/docs/report1"

    def _chunk_units(self) -> list[ContentUnit]:
        """Two chunks from the same document with overlapping entities."""
        ttl_chunk0 = f"""
        @prefix facts: <{DEFAULT_IRI}/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        facts:Apple rdf:type facts:Company .
        facts:Apple rdfs:label "Apple" .
        facts:Apple facts:foundedIn "1976" .
        """
        ttl_chunk1 = f"""
        @prefix facts: <{DEFAULT_IRI}/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        facts:Apple rdf:type facts:Company .
        facts:Apple facts:headquarters "Cupertino" .
        facts:TimCook rdf:type facts:Person .
        facts:TimCook facts:ceoOf facts:Apple .
        """
        return [
            _make_fact_unit(
                "Apple was founded in 1976.", 0, "chunk0hash", self.DOC_IRI, ttl_chunk0
            ),
            _make_fact_unit(
                "Apple HQ is in Cupertino. Tim Cook is CEO.",
                1,
                "chunk1hash",
                self.DOC_IRI,
                ttl_chunk1,
            ),
        ]

    def test_merged_graph_contains_triples_from_both_chunks(self) -> None:
        """Aggregation merges facts from all chunks into one graph."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        units = self._chunk_units()
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs(units)

        # Must contain triples originating from both chunks
        assert len(result) > 0

        # Serialise for human-readable inspection and check key facts
        ttl = result.serialize(format="turtle")
        # "1976" appears in chunk 0, "Cupertino" in chunk 1
        assert "1976" in ttl
        assert "Cupertino" in ttl

    def test_chunk_metadata_present(self) -> None:
        """Each chunk is described with prov:Entity, position and hid."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        units = self._chunk_units()
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs(units)

        chunk_uris = list(result.subjects(RDF.type, PROV.Entity))
        assert len(chunk_uris) == 2, (
            f"Expected 2 prov:Entity chunks, got {len(chunk_uris)}"
        )

        for chunk_uri in chunk_uris:
            # Must have schema:position
            positions = list(result.objects(chunk_uri, SCHEMA.position))
            assert len(positions) == 1, f"Missing schema:position for {chunk_uri}"
            # Must have schema:identifier (hid)
            hids = list(result.objects(chunk_uri, SCHEMA.identifier))
            assert len(hids) == 1, f"Missing schema:identifier for {chunk_uri}"

    def test_provenance_reification_links_triples_to_chunks(self) -> None:
        """Every asserted fact triple has an RDF 1.2 reifier with prov:wasDerivedFrom."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        units = self._chunk_units()
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs(units)

        # Collect all RDF 1.2 reifier nodes (subjects of rdf:reifies)
        stmt_nodes = set(result.subjects(RDF_REIFIES, None))
        assert len(stmt_nodes) > 0, "No rdf:reifies reifier nodes found"

        # Every statement must link to at least one chunk via prov:wasDerivedFrom
        for stmt in stmt_nodes:
            sources = list(result.objects(stmt, PROV.wasDerivedFrom))
            assert len(sources) >= 1, f"Reification {stmt} has no prov:wasDerivedFrom"
            # Each source must be a prov:Entity (chunk)
            for src in sources:
                assert (src, RDF.type, PROV.Entity) in result, (
                    f"prov:wasDerivedFrom target {src} is not a prov:Entity"
                )

    def test_shared_triple_has_multiple_provenance_sources(self) -> None:
        """A triple appearing in two chunks accumulates two wasDerivedFrom arcs."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        units = self._chunk_units()
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs(units)

        # "Apple rdf:type Company" appears in both chunks.
        # Its RDF 1.2 reifier should have wasDerivedFrom pointing to both chunk URIs.
        stmt_nodes = set(result.subjects(RDF_REIFIES, None))

        multi_source_found = False
        for stmt in stmt_nodes:
            sources = list(result.objects(stmt, PROV.wasDerivedFrom))
            if len(sources) > 1:
                multi_source_found = True
                break

        assert multi_source_found, (
            "Expected at least one reifier linked to multiple chunks"
        )

    def test_deduplication_of_content_triples(self) -> None:
        """Identical triples from different chunks appear only once."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        units = self._chunk_units()
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs(units)

        # Count how many times a literal "1976" appears as an object
        # in asserted triples (RDF 1.2 provenance uses triple terms,
        # so no rdf:object triples to exclude).
        asserted_triples_with_1976 = [
            (s, p, o) for s, p, o in result if str(o) == "1976"
        ]
        assert len(asserted_triples_with_1976) == 1, (
            f"Expected exactly 1 asserted triple with '1976', "
            f"got {len(asserted_triples_with_1976)}"
        )


class TestDocIriNamespace:
    """Test that fact entities are placed under the ContentUnit's doc_iri."""

    CUSTOM_DOC_IRI = "https://my-org.io/reports/annual2025"

    def test_fact_entities_under_doc_iri(self) -> None:
        """Fact entities get URIs under the chunk's doc_iri, not DEFAULT_IRI."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        ttl = f"""
        @prefix facts: <{DEFAULT_IRI}/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        facts:Revenue rdf:type facts:FinancialMetric .
        facts:Revenue rdfs:label "Revenue" .
        facts:Revenue facts:amount "42000000" .
        """
        unit = _make_fact_unit(
            "Revenue was $42M.",
            0,
            "rev01",
            self.CUSTOM_DOC_IRI,
            ttl,
        )
        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs([unit])

        assert len(result) > 0

        # Fact subjects must live under CUSTOM_DOC_IRI, not DEFAULT_IRI
        fact_subjects = {
            str(s)
            for s, _p, _o in result
            if isinstance(s, URIRef)
            and not str(s).startswith("http://www.w3.org")
            and not str(s).startswith("https://schema.org")
            # Exclude reifier and chunk-meta nodes
            and "/stmt/" not in str(s)
            and "/chunk/" not in str(s)
        }

        for s_str in fact_subjects:
            assert s_str.startswith(self.CUSTOM_DOC_IRI), (
                f"Expected fact URI under {self.CUSTOM_DOC_IRI}, got {s_str}"
            )
            assert not s_str.startswith(DEFAULT_IRI), (
                f"Fact URI should NOT be under DEFAULT_IRI: {s_str}"
            )

    def test_mixed_doc_iris_preserved(self) -> None:
        """Chunks from different documents keep their respective doc_iri namespaces."""
        from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

        doc_a = "https://org-a.io/doc/alpha"
        doc_b = "https://org-b.io/doc/beta"

        ttl_a = f"""
        @prefix facts: <{DEFAULT_IRI}/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        facts:Sensor rdf:type facts:Device .
        facts:Sensor facts:measures "temperature" .
        """
        ttl_b = f"""
        @prefix facts: <{DEFAULT_IRI}/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        facts:Actuator rdf:type facts:Device .
        facts:Actuator facts:controls "valve" .
        """
        unit_a = _make_fact_unit("Sensor measures temp.", 0, "sa01", doc_a, ttl_a)
        unit_b = _make_fact_unit("Actuator controls valve.", 0, "sb01", doc_b, ttl_b)

        agg = EmbeddingBasedAggregator()
        result = agg.aggregate_graphs([unit_a, unit_b])

        assert len(result) > 0

        # Collect fact-entity subject URIs (skip reifier / chunk / std ns)
        fact_subjects = {
            str(s)
            for s, _p, _o in result
            if isinstance(s, URIRef)
            and not str(s).startswith("http://www.w3.org")
            and not str(s).startswith("https://schema.org")
            and "/stmt/" not in str(s)
            and "/chunk/" not in str(s)
        }

        # At least some facts should be under doc_a or doc_b (not DEFAULT_IRI)
        under_doc_a = [s for s in fact_subjects if s.startswith(doc_a)]
        under_doc_b = [s for s in fact_subjects if s.startswith(doc_b)]
        under_default = [s for s in fact_subjects if s.startswith(DEFAULT_IRI)]

        assert len(under_doc_a) > 0 or len(under_doc_b) > 0, (
            "Expected at least some fact URIs under doc_a or doc_b"
        )
        assert len(under_default) == 0, (
            f"No fact URIs should remain under DEFAULT_IRI, found: {under_default}"
        )


class TestGraphRewriterProvenance:
    """Unit tests for GraphRewriter.merge_graphs_with_provenance."""

    def test_chunk_metadata_triples(self, graph_rewriter: GraphRewriter) -> None:
        """Chunk metadata (prov:Entity, position, hid) is emitted."""
        g = RDFGraph()
        e = URIRef(f"{DEFAULT_IRI}/Entity1")
        g.add((e, RDF.type, URIRef(f"{DEFAULT_IRI}/Thing")))

        unit = ContentUnit(
            text="test",
            index=5,
            hid="xyz687",
            doc_iri="https://example.org/doc/abc123",
            graph=g,
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit],
            mapping={},
        )

        unit_uri = URIRef(unit.iri_absolute)
        assert (unit_uri, RDF.type, PROV.Entity) in merged
        assert (unit_uri, SCHEMA.position, Literal(5, datatype=XSD.integer)) in merged
        assert (unit_uri, SCHEMA.identifier, Literal("abc123")) in merged

    def test_rdf12_reifier_nodes_created(self, graph_rewriter: GraphRewriter) -> None:
        """Each content triple gets an RDF 1.2 reifier with rdf:reifies."""
        g = RDFGraph()
        s = URIRef(f"{DEFAULT_IRI}/Alice")
        p = URIRef(f"{DEFAULT_IRI}/knows")
        o = URIRef(f"{DEFAULT_IRI}/Bob")
        g.add((s, p, o))

        unit = ContentUnit(
            text="Alice knows Bob",
            index=0,
            hid="h1",
            doc_iri="https://example.org/doc",
            graph=g,
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit],
            mapping={},
        )

        # Find RDF 1.2 reifier nodes (subjects of rdf:reifies)
        stmts = list(merged.subjects(RDF_REIFIES, None))
        assert len(stmts) == 1

        stmt = stmts[0]
        # The object of rdf:reifies should be a triple term (tuple)
        reified = list(merged.objects(stmt, RDF_REIFIES))
        assert len(reified) == 1
        qt = reified[0]
        assert isinstance(qt, tuple), f"Expected tuple, got {type(qt)}"
        assert qt == (s, p, o)
        assert (stmt, PROV.wasDerivedFrom, URIRef(unit.iri_absolute)) in merged

    def test_mapping_applied_before_reification(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """RDF 1.2 reifier captures the *mapped* (not original) triple."""
        g = RDFGraph()
        old_s = URIRef("http://chunk.org/OldEntity")
        p = URIRef("http://chunk.org/prop")
        o = Literal("value")
        g.add((old_s, p, o))

        new_s = URIRef(f"{DEFAULT_IRI}/NewEntity")
        new_p = URIRef(f"{DEFAULT_IRI}/prop")
        mapping = {old_s: new_s, p: new_p}

        unit = ContentUnit(
            text="test",
            index=0,
            hid="h1",
            doc_iri="https://example.org/doc",
            graph=g,
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit],
            mapping,
        )

        stmts = list(merged.subjects(RDF_REIFIES, None))
        assert len(stmts) == 1

        stmt = stmts[0]
        # Triple term must reference the *mapped* URIs
        reified = list(merged.objects(stmt, RDF_REIFIES))
        assert len(reified) == 1
        qt = reified[0]
        assert isinstance(qt, tuple)
        assert qt[0] == new_s
        assert qt[1] == new_p
        # Literal value is preserved (pyoxigraph may normalise plain
        # literals to explicit xsd:string, so compare by value).
        assert str(qt[2]) == str(o)

    # ------------------------------------------------------------------
    # Multi-unit tests
    # ------------------------------------------------------------------

    def test_multiple_units_chunk_metadata(self, graph_rewriter: GraphRewriter) -> None:
        """Each unit contributes its own prov:Entity chunk metadata."""
        g1 = RDFGraph()
        g1.add(
            (
                URIRef(f"{DEFAULT_IRI}/Alice"),
                URIRef(f"{DEFAULT_IRI}/knows"),
                URIRef(f"{DEFAULT_IRI}/Bob"),
            )
        )
        g2 = RDFGraph()
        g2.add(
            (
                URIRef(f"{DEFAULT_IRI}/Carol"),
                URIRef(f"{DEFAULT_IRI}/knows"),
                URIRef(f"{DEFAULT_IRI}/Dave"),
            )
        )

        doc_iri = "https://example.org/doc"
        unit_a = ContentUnit(
            text="Alice knows Bob",
            index=0,
            hid="ha",
            doc_iri=doc_iri,
            graph=g1,
            type=OutputType.FACTS,
        )
        unit_b = ContentUnit(
            text="Carol knows Dave",
            index=1,
            hid="hb",
            doc_iri=doc_iri,
            graph=g2,
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit_a, unit_b],
            mapping={},
        )

        chunk_uris = list(merged.subjects(RDF.type, PROV.Entity))
        assert len(chunk_uris) == 2

        # Verify each chunk has position + identifier
        for chunk_uri in chunk_uris:
            assert len(list(merged.objects(chunk_uri, SCHEMA.position))) == 1
            assert len(list(merged.objects(chunk_uri, SCHEMA.identifier))) == 1

    def test_shared_triple_accumulates_provenance(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """Same triple from two units gets two prov:wasDerivedFrom arcs."""
        shared_triple = (
            URIRef(f"{DEFAULT_IRI}/Alice"),
            URIRef(f"{DEFAULT_IRI}/knows"),
            URIRef(f"{DEFAULT_IRI}/Bob"),
        )
        g1 = RDFGraph()
        g1.add(shared_triple)
        g2 = RDFGraph()
        g2.add(shared_triple)

        doc_iri = "https://example.org/doc"
        unit_a = ContentUnit(
            text="chunk 0",
            index=0,
            hid="h0",
            doc_iri=doc_iri,
            graph=g1,
            type=OutputType.FACTS,
        )
        unit_b = ContentUnit(
            text="chunk 1",
            index=1,
            hid="h1",
            doc_iri=doc_iri,
            graph=g2,
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit_a, unit_b],
            mapping={},
        )

        # Only one RDF 1.2 reifier node for the shared triple
        stmts = list(merged.subjects(RDF_REIFIES, None))
        assert len(stmts) == 1

        # But it must link to both chunks
        sources = list(merged.objects(stmts[0], PROV.wasDerivedFrom))
        assert len(sources) == 2
        source_strs = {str(s) for s in sources}
        assert str(URIRef(unit_a.iri_absolute)) in source_strs
        assert str(URIRef(unit_b.iri_absolute)) in source_strs

    def test_content_triples_deduplicated_across_units(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """Identical triples from different units appear only once in merged graph."""
        triple = (
            URIRef(f"{DEFAULT_IRI}/X"),
            RDF.type,
            URIRef(f"{DEFAULT_IRI}/Thing"),
        )
        g1 = RDFGraph()
        g1.add(triple)
        g2 = RDFGraph()
        g2.add(triple)

        units = [
            ContentUnit(
                text="a",
                index=0,
                hid="h0",
                doc_iri="https://example.org/doc",
                graph=g1,
                type=OutputType.FACTS,
            ),
            ContentUnit(
                text="b",
                index=1,
                hid="h1",
                doc_iri="https://example.org/doc",
                graph=g2,
                type=OutputType.FACTS,
            ),
        ]

        merged = graph_rewriter.merge_graphs_with_provenance(
            units,
            mapping={},
        )

        # The asserted content triple should appear exactly once
        # (skip rdf:reifies triples whose object is a triple term tuple)
        matches = [
            (s, p, o)
            for s, p, o in merged
            if not isinstance(o, tuple) and (s, p, o) == triple
        ]
        assert len(matches) == 1

    def test_multiple_units_different_documents(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """Units from different doc_iris get separate chunk URIs."""
        g1 = RDFGraph()
        g1.add(
            (
                URIRef(f"{DEFAULT_IRI}/Sensor"),
                RDF.type,
                URIRef(f"{DEFAULT_IRI}/Device"),
            )
        )
        g2 = RDFGraph()
        g2.add(
            (
                URIRef(f"{DEFAULT_IRI}/Motor"),
                RDF.type,
                URIRef(f"{DEFAULT_IRI}/Device"),
            )
        )

        unit_a = ContentUnit(
            text="sensor info",
            index=0,
            hid="sa01",
            doc_iri="https://org-a.io/doc/alpha",
            graph=g1,
            type=OutputType.FACTS,
        )
        unit_b = ContentUnit(
            text="motor info",
            index=0,
            hid="sb01",
            doc_iri="https://org-b.io/doc/beta",
            graph=g2,
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit_a, unit_b],
            mapping={},
        )

        chunk_uris = {str(u) for u in merged.subjects(RDF.type, PROV.Entity)}
        assert str(URIRef(unit_a.iri_absolute)) in chunk_uris
        assert str(URIRef(unit_b.iri_absolute)) in chunk_uris

        # Both documents' content triples are present
        assert (
            URIRef(f"{DEFAULT_IRI}/Sensor"),
            RDF.type,
            URIRef(f"{DEFAULT_IRI}/Device"),
        ) in merged
        assert (
            URIRef(f"{DEFAULT_IRI}/Motor"),
            RDF.type,
            URIRef(f"{DEFAULT_IRI}/Device"),
        ) in merged

    def test_mapping_applied_across_multiple_units(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """A shared mapping rewrites entities consistently across all units."""
        old_entity = URIRef("http://chunk.org/Sensor")
        new_entity = URIRef(f"{DEFAULT_IRI}/Sensor")
        mapping: dict[URIRef, URIRef] = {old_entity: new_entity}

        g1 = RDFGraph()
        g1.add((old_entity, RDF.type, URIRef(f"{DEFAULT_IRI}/Device")))

        g2 = RDFGraph()
        g2.add((old_entity, URIRef(f"{DEFAULT_IRI}/measures"), Literal("temp")))

        units = [
            ContentUnit(
                text="sensor type",
                index=0,
                hid="h0",
                doc_iri="https://example.org/doc",
                graph=g1,
                type=OutputType.FACTS,
            ),
            ContentUnit(
                text="sensor measures",
                index=1,
                hid="h1",
                doc_iri="https://example.org/doc",
                graph=g2,
                type=OutputType.FACTS,
            ),
        ]

        merged = graph_rewriter.merge_graphs_with_provenance(
            units,
            mapping=mapping,
        )

        # Old entity must not appear in any asserted triple (except owl:sameAs
        # and quoted-triple tuples)
        for s, p, o in merged:
            if p == OWL.sameAs:
                continue
            if isinstance(o, tuple):
                # Skip triple term objects (rdf:reifies)
                continue
            assert s != old_entity, f"Old entity found as subject: {s}"
            assert o != old_entity, f"Old entity found as object: {o}"

        # Mapped entity is used instead
        assert (new_entity, RDF.type, URIRef(f"{DEFAULT_IRI}/Device")) in merged
        assert (
            new_entity,
            URIRef(f"{DEFAULT_IRI}/measures"),
            Literal("temp"),
        ) in merged

        # RDF 1.2 reifier also references the mapped entity
        found_mapped = False
        for stmt in merged.subjects(RDF_REIFIES, None):
            reified = list(merged.objects(stmt, RDF_REIFIES))
            for qt in reified:
                if isinstance(qt, tuple) and qt[0] == new_entity:
                    found_mapped = True
                    break
            if found_mapped:
                break
        assert found_mapped, "No RDF 1.2 reifier references the mapped entity"

    def test_unit_with_empty_graph_contributes_no_content(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """Units with an empty graph contribute metadata but no content triples."""
        g = RDFGraph()
        g.add(
            (
                URIRef(f"{DEFAULT_IRI}/A"),
                URIRef(f"{DEFAULT_IRI}/rel"),
                URIRef(f"{DEFAULT_IRI}/B"),
            )
        )
        unit_ok = ContentUnit(
            text="valid",
            index=0,
            hid="h0",
            doc_iri="https://example.org/doc",
            graph=g,
            type=OutputType.FACTS,
        )
        unit_empty = ContentUnit(
            text="empty",
            index=1,
            hid="h1",
            doc_iri="https://example.org/doc",
            type=OutputType.FACTS,
        )

        merged = graph_rewriter.merge_graphs_with_provenance(
            [unit_ok, unit_empty],
            mapping={},
        )

        # Both chunks get prov:Entity metadata
        chunk_uris = list(merged.subjects(RDF.type, PROV.Entity))
        assert len(chunk_uris) == 2

        # Only 1 RDF 1.2 reifier node (from the non-empty unit)
        stmts = list(merged.subjects(RDF_REIFIES, None))
        assert len(stmts) == 1

        # The valid triple is present
        assert (
            URIRef(f"{DEFAULT_IRI}/A"),
            URIRef(f"{DEFAULT_IRI}/rel"),
            URIRef(f"{DEFAULT_IRI}/B"),
        ) in merged

    def test_owl_sameas_with_mapping_multiple_units(
        self, graph_rewriter: GraphRewriter
    ) -> None:
        """owl:sameAs links are emitted for mapped entities across units."""
        old = URIRef("http://chunk.org/Entity")
        new = URIRef(f"{DEFAULT_IRI}/Entity")
        mapping: dict[URIRef, URIRef] = {old: new}

        g1 = RDFGraph()
        g1.add((old, RDF.type, URIRef(f"{DEFAULT_IRI}/Class")))
        g2 = RDFGraph()
        g2.add((old, URIRef(f"{DEFAULT_IRI}/label"), Literal("Entity")))

        units = [
            ContentUnit(
                text="a",
                index=0,
                hid="h0",
                doc_iri="https://example.org/doc",
                graph=g1,
                type=OutputType.FACTS,
            ),
            ContentUnit(
                text="b",
                index=1,
                hid="h1",
                doc_iri="https://example.org/doc",
                graph=g2,
                type=OutputType.FACTS,
            ),
        ]

        merged = graph_rewriter.merge_graphs_with_provenance(
            units,
            mapping=mapping,
        )

        assert (new, OWL.sameAs, old) in merged
