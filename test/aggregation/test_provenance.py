from rdflib import RDF, Literal, URIRef
from rdflib.namespace import DCTERMS, FOAF, RDFS, XSD

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.iri_policy import join_namespace_local, normalize_namespace_iri
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import apply_document_metadata_provenance
from ontocast.tool.agg.rewriter import GraphRewriter
from ontocast.tool.triple_manager.core import TripleStoreManager


def test_merge_graphs_with_provenance_adds_chunk_metadata(
    graph_rewriter: GraphRewriter,
) -> None:
    graph = RDFGraph()
    entity = URIRef(f"{DEFAULT_IRI}/Entity1")
    graph.add((entity, RDF.type, URIRef(f"{DEFAULT_IRI}/Thing")))

    unit = ContentUnit(
        text="test",
        index=5,
        doc_iri=URIRef("https://example.org/doc/abc123"),
        graph=graph,
        type=OutputType.FACTS,
    )
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    unit_uri = URIRef(unit.iri_absolute)

    assert (unit_uri, RDF.type, PROV.Entity) in merged
    assert (unit_uri, SCHEMA.position, Literal(5, datatype=XSD.integer)) in merged
    assert (unit_uri, SCHEMA.identifier, Literal(unit.hid)) in merged

    namespaces = {prefix: str(namespace) for prefix, namespace in merged.namespaces()}

    assert namespaces["prov"] == str(PROV)
    assert namespaces["schema"] == str(SCHEMA)
    assert namespaces["doc"] == "https://example.org/doc/abc123/"


def test_merge_graphs_with_provenance_reifies_mapped_triple(
    graph_rewriter: GraphRewriter,
) -> None:
    graph = RDFGraph()
    old_subject = URIRef("http://chunk.org/OldEntity")
    old_predicate = URIRef("http://chunk.org/prop")
    value = Literal("value")
    graph.add((old_subject, old_predicate, value))

    new_subject = URIRef(f"{DEFAULT_IRI}/NewEntity")
    new_predicate = URIRef(f"{DEFAULT_IRI}/prop")
    unit = ContentUnit(
        text="test",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
        graph=graph,
        type=OutputType.FACTS,
    )

    merged = graph_rewriter.merge_graphs_with_provenance(
        [unit],
        {old_subject: new_subject, old_predicate: new_predicate},
    )
    stmt_nodes = list(merged.subjects(RDF_REIFIES, None))
    assert len(stmt_nodes) == 1

    reified = list(merged.objects(stmt_nodes[0], RDF_REIFIES))
    assert len(reified) == 1
    quoted = reified[0]
    assert isinstance(quoted, tuple)
    assert quoted[0] == new_subject
    assert quoted[1] == new_predicate
    assert str(quoted[2]) == str(value)


def test_shared_triple_accumulates_multiple_provenance_sources(
    graph_rewriter: GraphRewriter,
) -> None:
    triple = (
        URIRef(f"{DEFAULT_IRI}/Alice"),
        URIRef(f"{DEFAULT_IRI}/knows"),
        URIRef(f"{DEFAULT_IRI}/Bob"),
    )
    graph_a = RDFGraph()
    graph_b = RDFGraph()
    graph_a.add(triple)
    graph_b.add(triple)

    unit_a = ContentUnit(
        text="chunk 0",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
        graph=graph_a,
        type=OutputType.FACTS,
    )
    unit_b = ContentUnit(
        text="chunk 1",
        index=1,
        doc_iri=URIRef("https://example.org/doc"),
        graph=graph_b,
        type=OutputType.FACTS,
    )

    merged = graph_rewriter.merge_graphs_with_provenance([unit_a, unit_b], mapping={})
    statements = list(merged.subjects(RDF_REIFIES, None))
    assert len(statements) == 1

    sources = {str(src) for src in merged.objects(statements[0], PROV.wasDerivedFrom)}
    assert str(URIRef(unit_a.iri_absolute)) in sources
    assert str(URIRef(unit_b.iri_absolute)) in sources


def test_merge_graphs_with_provenance_skips_empty_unit_graph(
    graph_rewriter: GraphRewriter,
) -> None:
    unit = ContentUnit(
        text="test",
        index=0,
        doc_iri=URIRef("https://example.org/doc"),
        graph=RDFGraph(),
        type=OutputType.FACTS,
    )
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    assert len(merged) == 0


def test_apply_document_metadata_provenance_emits_identity_triples() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    apply_document_metadata_provenance(
        doc_iri,
        {
            "title": "Annual Report",
            "doi": "10.1234/example",
            "identifiers": [{"scheme": "erp:doc", "value": "INV-2024-001"}],
        },
        graph,
    )

    assert (doc_iri, RDF.type, PROV.Entity) in graph
    assert (doc_iri, RDF.type, FOAF.Document) in graph
    assert (doc_iri, DCTERMS.title, Literal("Annual Report")) in graph
    assert (doc_iri, DCTERMS.identifier, Literal("10.1234/example")) in graph

    structured = list(graph.objects(doc_iri, DCTERMS.identifier))
    bnodes = [n for n in structured if not isinstance(n, Literal)]
    assert len(bnodes) == 1
    schemes = {str(o) for b in bnodes for o in graph.objects(b, DCTERMS.type)}
    assert "erp:doc" in schemes

    cleaned = TripleStoreManager.strip_provenance(graph)
    assert (doc_iri, DCTERMS.title, Literal("Annual Report")) in cleaned
    assert (doc_iri, DCTERMS.identifier, Literal("10.1234/example")) in cleaned


def test_apply_document_metadata_provenance_noop_when_empty() -> None:
    graph = RDFGraph()
    apply_document_metadata_provenance(
        URIRef("https://example.org/doc/abc"),
        {},
        graph,
    )
    assert len(graph) == 0


def test_author_string_mints_schema_person() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"author": "Jane Doe"},
        graph,
        entity_namespace=ns,
    )
    person = URIRef(join_namespace_local(ns, "janeDoe", context="facts"))
    assert (doc_iri, DCTERMS.creator, person) in graph
    assert (person, RDF.type, SCHEMA.Person) in graph
    assert (person, RDFS.label, Literal("Jane Doe")) in graph

    cleaned = TripleStoreManager.strip_provenance(graph)
    assert (person, RDF.type, SCHEMA.Person) in cleaned
    assert (doc_iri, DCTERMS.creator, person) in cleaned


def test_authors_list_mints_two_persons() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"authors": ["Jane Doe", "John Smith"]},
        graph,
        entity_namespace=ns,
    )
    jane = URIRef(join_namespace_local(ns, "janeDoe", context="facts"))
    john = URIRef(join_namespace_local(ns, "johnSmith", context="facts"))
    assert (doc_iri, DCTERMS.creator, jane) in graph
    assert (doc_iri, DCTERMS.creator, john) in graph
    assert (jane, RDF.type, SCHEMA.Person) in graph
    assert (john, RDF.type, SCHEMA.Person) in graph


def test_author_dict_type_override() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"author": {"name": "Acme Corp", "type": "schema:Organization"}},
        graph,
        entity_namespace=ns,
    )
    org = URIRef(join_namespace_local(ns, "acmeCorp", context="facts"))
    assert (doc_iri, DCTERMS.creator, org) in graph
    assert (org, RDF.type, URIRef("http://schema.org/Organization")) in graph
    assert (org, RDFS.label, Literal("Acme Corp")) in graph


def test_project_string_mints_prov_entity() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"project": "Perovskite Survey"},
        graph,
        entity_namespace=ns,
    )
    project = URIRef(join_namespace_local(ns, "perovskiteSurvey", context="facts"))
    assert (doc_iri, DCTERMS.relation, project) in graph
    assert (project, RDF.type, PROV.Entity) in graph
    assert (project, RDFS.label, Literal("Perovskite Survey")) in graph
    # Not a blank-node identifier
    bnodes = [
        n
        for n in graph.objects(doc_iri, DCTERMS.identifier)
        if not isinstance(n, Literal)
    ]
    assert bnodes == []


def test_project_dict_with_identifier() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {
            "project": {
                "name": "Perovskite Survey",
                "identifier": "PRJ-2024-07",
            }
        },
        graph,
        entity_namespace=ns,
    )
    project = URIRef(join_namespace_local(ns, "perovskiteSurvey", context="facts"))
    assert (doc_iri, DCTERMS.relation, project) in graph
    assert (project, RDF.type, PROV.Entity) in graph
    assert (project, RDFS.label, Literal("Perovskite Survey")) in graph
    assert (project, DCTERMS.identifier, Literal("PRJ-2024-07")) in graph


def test_custom_key_string_mints_prov_entity() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"department": "R&D"},
        graph,
        entity_namespace=ns,
    )
    dept = URIRef(join_namespace_local(ns, "rd", context="facts"))
    assert (doc_iri, DCTERMS.relation, dept) in graph
    assert (dept, RDF.type, PROV.Entity) in graph
    assert (dept, RDFS.label, Literal("R&D")) in graph


def test_custom_key_dict_type_override() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {
            "department": {
                "name": "R&D",
                "type": "schema:Organization",
            }
        },
        graph,
        entity_namespace=ns,
    )
    dept = URIRef(join_namespace_local(ns, "rd", context="facts"))
    assert (doc_iri, DCTERMS.relation, dept) in graph
    assert (dept, RDF.type, URIRef("http://schema.org/Organization")) in graph
