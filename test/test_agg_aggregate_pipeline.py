from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator


def make_fact_unit(
    text: str,
    index: int,
    hid: str,
    doc_iri: URIRef | str,
    ttl: str,
) -> ContentUnit:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return ContentUnit(
        text=text,
        index=index,
        hid=hid,
        doc_iri=URIRef(str(doc_iri)),
        graph=graph,
        type=OutputType.FACTS,
    )


def test_aggregate_graphs_returns_empty_graph_for_no_units() -> None:
    aggregator = EmbeddingBasedAggregator()
    result = aggregator.aggregate_graphs([])
    assert len(result) == 0


def test_fact_entities_use_doc_iri_namespace() -> None:
    doc_iri = "https://my-org.io/reports/annual2025"
    ttl = f"""
    @prefix facts: <{DEFAULT_IRI}/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    facts:Revenue rdf:type facts:FinancialMetric .
    facts:Revenue rdfs:label "Revenue" .
    facts:Revenue facts:amount "42000000" .
    """
    unit = make_fact_unit("Revenue was $42M.", 0, "rev01", doc_iri, ttl)

    result = EmbeddingBasedAggregator().aggregate_graphs([unit])
    assert len(result) > 0

    fact_subjects = {
        str(subject)
        for subject, predicate, _ in result
        if isinstance(subject, URIRef)
        and predicate != RDF.type
        and not str(subject).startswith("http://www.w3.org")
        and not str(subject).startswith("https://schema.org")
        and "/stmt/" not in str(subject)
        and "/chunk/" not in str(subject)
    }
    assert fact_subjects
    assert any(subject.startswith(doc_iri) for subject in fact_subjects)


def test_aggregate_graphs_merges_overlapping_facts(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/report1"
    ttl_chunk_0 = f"""
    @prefix facts: <{DEFAULT_IRI}/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    facts:UnitedStates rdf:type facts:Country .
    facts:UnitedStates rdfs:label "United States" .
    facts:UnitedStates facts:capitalCity "Washington, D.C." .
    facts:UnitedStates facts:currency "USD" .
    """
    ttl_chunk_1 = f"""
    @prefix facts: <{DEFAULT_IRI}/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    facts:united_states rdf:type facts:Country .
    facts:united_states rdfs:label "United States" .
    facts:united_states facts:population "331000000" .
    """
    ttl_chunk_2 = f"""
    @prefix facts: <{DEFAULT_IRI}/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    facts:UnitedStatesBank rdf:type facts:Company .
    facts:UnitedStatesBank rdfs:label "United States Bank" .
    facts:UnitedStatesBank facts:headquarters "Portland" .
    """

    units = [
        make_fact_unit(
            "The United States has capital Washington, D.C. and uses USD.",
            0,
            "chunk0hash",
            doc_iri,
            ttl_chunk_0,
        ),
        make_fact_unit(
            "In another section, united_states is described with population data.",
            1,
            "chunk1hash",
            doc_iri,
            ttl_chunk_1,
        ),
        make_fact_unit(
            "United States Bank is headquartered in Portland.",
            2,
            "chunk2hash",
            doc_iri,
            ttl_chunk_2,
        ),
    ]
    aggregator = EmbeddingBasedAggregator()

    def cluster_by_normal_form(representations):
        clusters_by_key: dict[str, list[URIRef]] = {}
        for entity, representation in representations.items():
            clusters_by_key.setdefault(representation.normal_form, []).append(entity)
        return list(clusters_by_key.values()), {}

    monkeypatch.setattr(
        aggregator.clusterer, "cluster_entities", cluster_by_normal_form
    )
    result = aggregator.aggregate_graphs(units)
    result.bind("unused", "https://unused.example/")
    turtle = result.serialize(format="turtle")

    assert "Washington, D.C." in turtle
    assert "USD" in turtle
    assert "331000000" in turtle
    assert "Portland" in turtle
    assert "@prefix doc:" in turtle
    assert "@prefix unused:" not in turtle
    assert len(list(result.triples((None, RDFS.label, None)))) >= 2

    us_subjects = {
        subject
        for subject in result.subjects(RDFS.label, Literal("United States"))
        if isinstance(subject, URIRef)
    }
    assert len(us_subjects) == 1
    us_entity = next(iter(us_subjects))

    bank_subjects = {
        subject
        for subject in result.subjects(RDFS.label, Literal("United States Bank"))
        if isinstance(subject, URIRef)
    }
    assert len(bank_subjects) == 1
    bank_entity = next(iter(bank_subjects))

    assert us_entity != bank_entity
    assert str(us_entity).startswith(doc_iri)
    assert str(bank_entity).startswith(doc_iri)

    assert (us_entity, None, Literal("USD")) in result
    assert (us_entity, None, Literal("331000000")) in result
    assert (bank_entity, None, Literal("Portland")) in result

    original_camel = URIRef(f"{DEFAULT_IRI}/UnitedStates")
    original_snake = URIRef(f"{DEFAULT_IRI}/united_states")
    assert (us_entity, OWL.sameAs, original_camel) in result
    assert (us_entity, OWL.sameAs, original_snake) in result

    statement_nodes = list(result.subjects(RDF_REIFIES, None))
    assert statement_nodes
    assert any(
        len(set(result.objects(stmt, PROV.wasDerivedFrom))) >= 2
        for stmt in statement_nodes
    )

    chunk_ids = {str(value) for value in result.objects(None, SCHEMA.identifier)}
    assert {"chunk0hash", "chunk1hash", "chunk2hash"} <= chunk_ids
