import pytest
from rdflib import RDF, Literal, URIRef
from rdflib.namespace import DCTERMS, FOAF, RDFS, XSD

from ontocast.onto.constants import DEFAULT_IRI, ONTOCAST, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.enum import SectionLabelSource
from ontocast.onto.iri_policy import join_namespace_local, normalize_namespace_iri
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import (
    _resolve_metadata_key,
    _split_identifier_affix,
    apply_document_metadata_provenance,
)
from ontocast.tool.agg.rewriter import GraphRewriter
from ontocast.tool.triple_manager.core import TripleStoreManager

pytestmark = pytest.mark.unit


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
    # schema:Text, the class -- schema:text is the *property* of that name, and
    # typing every provenance node with a property IRI was simply wrong.
    assert (unit_uri, RDF.type, SCHEMA.Text) in merged
    assert (unit_uri, SCHEMA.position, Literal(5, datatype=XSD.integer)) in merged
    assert (unit_uri, SCHEMA.identifier, Literal(unit.hid)) in merged

    namespaces = {prefix: str(namespace) for prefix, namespace in merged.namespaces()}

    assert namespaces["prov"] == str(PROV)
    assert namespaces["schema"] == str(SCHEMA)
    assert namespaces["ontocast"] == str(ONTOCAST)
    assert namespaces["doc"] == "https://example.org/doc/abc123/"


def _labeled_unit(**overrides) -> ContentUnit:
    graph = RDFGraph()
    entity = URIRef(f"{DEFAULT_IRI}/Entity1")
    graph.add((entity, RDF.type, URIRef(f"{DEFAULT_IRI}/Thing")))
    fields: dict = {
        "text": "test",
        "index": 5,
        "doc_iri": URIRef("https://example.org/doc/abc123"),
        "graph": graph,
        "type": OutputType.FACTS,
        "section_label": "results",
        "section_label_source": SectionLabelSource.HEADING_KEYWORD,
        "section_label_confidence": 0.75,
    }
    fields.update(overrides)
    return ContentUnit(**fields)


def test_section_label_reaches_the_provenance_artifact(
    graph_rewriter: GraphRewriter,
) -> None:
    """Which part of the document a fact came from must be auditable.

    The label reached the summarizer and ``ontocast inspect-sections`` and
    stopped there, so a finished run could not answer the question at all.
    """
    unit = _labeled_unit()
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    unit_uri = URIRef(unit.iri_absolute)

    assert (unit_uri, SCHEMA.articleSection, Literal("results")) in merged
    assert (
        unit_uri,
        ONTOCAST.sectionLabelSource,
        Literal("heading_keyword"),
    ) in merged
    assert (
        unit_uri,
        ONTOCAST.sectionLabelConfidence,
        Literal("0.75", datatype=XSD.decimal),
    ) in merged


def test_unlabeled_unit_emits_no_section_triples(
    graph_rewriter: GraphRewriter,
) -> None:
    """A bare confidence of 0.0 would read as certainty about nothing."""
    unit = _labeled_unit(
        section_label=None, section_label_source=None, section_label_confidence=0.0
    )
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    unit_uri = URIRef(unit.iri_absolute)

    assert list(merged.objects(unit_uri, SCHEMA.articleSection)) == []
    assert list(merged.objects(unit_uri, ONTOCAST.sectionLabelSource)) == []
    assert list(merged.objects(unit_uri, ONTOCAST.sectionLabelConfidence)) == []


def test_label_without_a_source_still_emits_label_and_confidence(
    graph_rewriter: GraphRewriter,
) -> None:
    unit = _labeled_unit(section_label_source=None)
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    unit_uri = URIRef(unit.iri_absolute)

    assert (unit_uri, SCHEMA.articleSection, Literal("results")) in merged
    assert list(merged.objects(unit_uri, ONTOCAST.sectionLabelSource)) == []
    assert len(list(merged.objects(unit_uri, ONTOCAST.sectionLabelConfidence))) == 1


def test_strip_provenance_removes_the_chunk_metadata_it_is_given(
    graph_rewriter: GraphRewriter,
) -> None:
    """The stripper keys on the unit node, so new metadata is covered for free.

    Pinned rather than assumed: ``strip_provenance`` matches chunk nodes by
    their ``prov:Entity`` + ``schema:Text`` typing, so it silently stops working
    the moment the rewriter and the matcher disagree on that class -- which
    nothing checked until now.
    """
    unit = _labeled_unit()
    merged = graph_rewriter.merge_graphs_with_provenance([unit], mapping={})
    unit_uri = URIRef(unit.iri_absolute)

    cleaned = TripleStoreManager.strip_provenance(merged)

    assert list(cleaned.predicate_objects(unit_uri)) == []
    assert (URIRef(f"{DEFAULT_IRI}/Entity1"), RDF.type, None) in cleaned


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
    assert (org, RDF.type, URIRef("https://schema.org/Organization")) in graph
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
    assert (dept, RDF.type, URIRef("https://schema.org/Organization")) in graph


@pytest.mark.parametrize(
    ("raw_key", "canonical"),
    [
        ("doi", "doi"),
        ("DOI", "doi"),
        ("doi_id", "doi"),
        ("id_doi", "doi"),
        ("doi-id", "doi"),
        ("arxiv_id", "arxiv_id"),
        ("arxivId", "arxiv_id"),
        ("arxiv-id", "arxiv_id"),
        ("arxiv", "arxiv_id"),
        ("sourceUrl", "source_url"),
        ("source-uri", "source_uri"),
        ("Title", "title"),
        ("Identifiers", "identifiers"),
        ("department", "department"),
        ("department_id", "department_id"),
        ("project_id", "project_id"),
        ("title_id", "title_id"),
    ],
)
def test_resolve_metadata_key_aliases(raw_key: str, canonical: str) -> None:
    assert _resolve_metadata_key(raw_key) == canonical


@pytest.mark.parametrize(
    ("raw_key", "stem", "affix"),
    [
        ("department_id", "department", "id"),
        ("id_department", "department", "id"),
        ("case_no", "case", "no"),
        ("invoice_ref", "invoice", "ref"),
        ("sku_code", "sku", "code"),
        ("asset_uid", "asset", "uid"),
        ("doc_slug", "doc", "slug"),
        ("seq_accession", "seq", "accession"),
        ("key_finding", "finding", "key"),
        ("finding_key", "finding", "key"),
        ("project_id", "project", "id"),
        ("num_invoice", "invoice", "num"),
        ("invoice_number", "invoice", "number"),
        ("uuid_batch", "batch", "uuid"),
        ("batch_guid", "batch", "guid"),
        ("item_reference", "item", "reference"),
    ],
)
def test_split_identifier_affix(raw_key: str, stem: str, affix: str) -> None:
    assert _split_identifier_affix(raw_key) == (stem, affix)


@pytest.mark.parametrize(
    "raw_key",
    ["department", "title", "doi", "id", "key"],
)
def test_split_identifier_affix_rejects_non_affixed(raw_key: str) -> None:
    assert _split_identifier_affix(raw_key) is None


@pytest.mark.parametrize(
    ("raw_key", "value"),
    [
        ("DOI", "10.1234/example"),
        ("doi_id", "10.1234/example"),
        ("id_doi", "10.1234/example"),
        ("doi-id", "10.1234/example"),
        ("arxivId", "2401.00001"),
        ("arxiv", "2401.00001"),
    ],
)
def test_identifier_key_aliases_emit_dcterms_identifier(
    raw_key: str, value: str
) -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    apply_document_metadata_provenance(doc_iri, {raw_key: value}, graph)
    assert (doc_iri, DCTERMS.identifier, Literal(value)) in graph


@pytest.mark.parametrize(
    ("raw_key", "value"),
    [
        ("sourceUrl", "https://example.org/paper"),
        ("source-uri", "https://example.org/paper"),
    ],
)
def test_source_key_aliases_emit_dcterms_source(raw_key: str, value: str) -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    apply_document_metadata_provenance(doc_iri, {raw_key: value}, graph)
    assert (doc_iri, DCTERMS.source, URIRef(value)) in graph


def test_title_alias_emits_dcterms_title() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    apply_document_metadata_provenance(doc_iri, {"Title": "Annual Report"}, graph)
    assert (doc_iri, DCTERMS.title, Literal("Annual Report")) in graph


def _structured_identifier_schemes(graph: RDFGraph, doc_iri: URIRef) -> dict[str, str]:
    """Map dcterms:type scheme -> rdf:value for structured identifiers on doc."""
    out: dict[str, str] = {}
    for node in graph.objects(doc_iri, DCTERMS.identifier):
        if isinstance(node, Literal):
            continue
        schemes = list(graph.objects(node, DCTERMS.type))
        values = list(graph.objects(node, RDF.value))
        if schemes and values:
            out[str(schemes[0])] = str(values[0])
    return out


@pytest.mark.parametrize(
    ("raw_key", "scheme", "value"),
    [
        ("department_id", "department", "D-42"),
        ("case_no", "case", "C-9"),
        ("invoice_ref", "invoice", "INV-1"),
        ("sku_code", "sku", "SKU-7"),
        ("asset_uid", "asset", "A-1"),
        ("doc_slug", "doc", "annual-report"),
        ("seq_accession", "seq", "ACC1"),
        ("item_num", "item", "42"),
        ("batch_number", "batch", "B-3"),
    ],
)
def test_affix_keys_emit_structured_identifier(
    raw_key: str, scheme: str, value: str
) -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    apply_document_metadata_provenance(doc_iri, {raw_key: value}, graph)
    schemes = _structured_identifier_schemes(graph, doc_iri)
    assert schemes.get(scheme) == value
    # Value must not be minted as a labeled companion entity
    labeled = [s for s in graph.subjects(RDFS.label, Literal(value)) if s != doc_iri]
    assert labeled == []


def test_department_id_emits_structured_identifier_not_entity() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"department_id": "D-42"},
        graph,
        entity_namespace=ns,
    )
    schemes = _structured_identifier_schemes(graph, doc_iri)
    assert schemes == {"department": "D-42"}
    dept = URIRef(join_namespace_local(ns, "d42", context="facts"))
    assert (doc_iri, DCTERMS.relation, dept) not in graph
    assert (dept, RDFS.label, Literal("D-42")) not in graph


def test_project_id_companion_attaches_to_project_entity() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"project": "Perovskite Survey", "project_id": "PRJ-1"},
        graph,
        entity_namespace=ns,
    )
    project = URIRef(join_namespace_local(ns, "perovskiteSurvey", context="facts"))
    assert (doc_iri, DCTERMS.relation, project) in graph
    assert (project, RDF.type, PROV.Entity) in graph
    assert (project, RDFS.label, Literal("Perovskite Survey")) in graph
    assert (project, DCTERMS.identifier, Literal("PRJ-1")) in graph
    assert _structured_identifier_schemes(graph, doc_iri) == {}


def test_project_id_alone_emits_structured_identifier() -> None:
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    ns = normalize_namespace_iri(str(doc_iri), context="facts")
    apply_document_metadata_provenance(
        doc_iri,
        {"project_id": "PRJ-1"},
        graph,
        entity_namespace=ns,
    )
    assert _structured_identifier_schemes(graph, doc_iri) == {"project": "PRJ-1"}
    project = URIRef(join_namespace_local(ns, "prj1", context="facts"))
    assert (doc_iri, DCTERMS.relation, project) not in graph


def test_key_finding_tradeoff_emits_structured_identifier() -> None:
    """Accepted trade-off: ``key`` affix treats non-id fields as structured ids."""
    graph = RDFGraph()
    doc_iri = URIRef("https://example.org/doc/abc")
    apply_document_metadata_provenance(
        doc_iri,
        {"key_finding": "Important result"},
        graph,
    )
    assert _structured_identifier_schemes(graph, doc_iri) == {
        "finding": "Important result"
    }
