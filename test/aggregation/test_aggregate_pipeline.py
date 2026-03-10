from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI, PROV, RDF_REIFIES, SCHEMA
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator
from ontocast.util import render_text_hash


def make_fact_unit(
    text: str,
    index: int,
    doc_iri: URIRef | str,
    ttl: str,
) -> ContentUnit:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return ContentUnit(
        text=text,
        index=index,
        doc_iri=URIRef(str(doc_iri)),
        graph=graph,
        type=OutputType.FACTS,
    )


def make_ontology_unit(
    text: str,
    index: int,
    doc_iri: URIRef | str,
    ttl: str,
) -> ContentUnit:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return ContentUnit(
        text=text,
        index=index,
        doc_iri=URIRef(str(doc_iri)),
        graph=graph,
        type=OutputType.ONTOLOGIES,
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
    unit = make_fact_unit("Revenue was $42M.", 0, doc_iri, ttl)

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

    text_0 = "The United States has capital Washington, D.C. and uses USD."
    text_1 = "In another section, united_states is described with population data."
    text_2 = "United States Bank is headquartered in Portland."
    units = [
        make_fact_unit(
            text_0,
            0,
            doc_iri,
            ttl_chunk_0,
        ),
        make_fact_unit(
            text_1,
            1,
            doc_iri,
            ttl_chunk_1,
        ),
        make_fact_unit(
            text_2,
            2,
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
    assert (us_entity, OWL.sameAs, original_camel) not in result
    assert (us_entity, OWL.sameAs, original_snake) not in result

    statement_nodes = list(result.subjects(RDF_REIFIES, None))
    assert statement_nodes
    assert all(
        len(set(result.objects(stmt, PROV.wasDerivedFrom))) >= 1
        for stmt in statement_nodes
    )

    chunk_ids = {str(value) for value in result.objects(None, SCHEMA.identifier)}
    expected_ids = {
        render_text_hash(text_0),
        render_text_hash(text_1),
        render_text_hash(text_2),
    }
    assert expected_ids <= chunk_ids


def test_aggregate_graphs_preserves_ontology_uris_and_provenance(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/report1"
    ttl_chunk_0 = """
    @prefix ex: <http://example.org/onto#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    ex:Person rdf:type rdfs:Class .
    ex:Person rdfs:label "Person" .
    """
    ttl_chunk_1 = """
    @prefix ex: <http://example.org/onto#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    ex:Persno rdf:type rdfs:Class .
    ex:Persno rdfs:label "Person" .
    """
    units = [
        make_ontology_unit("Defines Person class.", 0, doc_iri, ttl_chunk_0),
        make_ontology_unit("Repeats class with typo URI.", 1, doc_iri, ttl_chunk_1),
    ]

    aggregator = EmbeddingBasedAggregator()

    def force_typo_and_canonical_in_one_cluster(representations):
        canonical = URIRef("http://example.org/onto#Person")
        typo = URIRef("http://example.org/onto#Persno")
        entities = set(representations.keys())
        if canonical in entities and typo in entities:
            return [[canonical, typo]], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_typo_and_canonical_in_one_cluster,
    )

    result = aggregator.aggregate_graphs(units)

    canonical = URIRef("http://example.org/onto#Person")
    typo = URIRef("http://example.org/onto#Persno")

    assert (canonical, RDFS.label, Literal("Person")) in result
    assert (typo, RDFS.label, Literal("Person")) in result
    assert (canonical, OWL.sameAs, typo) in result or (
        typo,
        OWL.sameAs,
        canonical,
    ) in result
    assert str(canonical).startswith("http://example.org/onto#")

    statement_nodes = list(result.subjects(RDF_REIFIES, None))
    assert statement_nodes
    assert all(
        len(set(result.objects(stmt, PROV.wasDerivedFrom))) >= 1
        for stmt in statement_nodes
    )


def test_facts_doc_entity_does_not_replace_ontology_entity(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/case-42"
    ontology_court = URIRef("https://growgraph.dev/fcaont#CourAppelRouen")
    doc_court = URIRef(f"{doc_iri}/CourAppelRouen")
    heard_at = URIRef("https://growgraph.dev/fcaont#heardAt")
    court_type = URIRef("https://growgraph.dev/fcaont#Court")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    doc:Case1 fcaont:heardAt doc:CourAppelRouen .
    doc:Case2 fcaont:heardAt fcaont:CourAppelRouen .
    doc:CourAppelRouen rdf:type fcaont:Court .
    fcaont:CourAppelRouen rdf:type fcaont:Court .
    """
    unit = make_fact_unit("Rouen court references.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((ontology_court, RDF.type, court_type))

    def force_doc_and_ontology_court_together(representations):
        entities = set(representations.keys())
        if doc_court in entities and ontology_court in entities:
            remaining = [e for e in entities if e not in {doc_court, ontology_court}]
            return [[doc_court, ontology_court], *[[e] for e in remaining]], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_doc_and_ontology_court_together,
    )

    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    assert (ontology_court, RDF.type, court_type) in result
    assert (doc_court, RDF.type, court_type) in result
    assert (ontology_court, OWL.sameAs, doc_court) not in result
    assert (doc_court, OWL.sameAs, ontology_court) not in result

    heard_at_targets = set(result.objects(None, heard_at))
    assert ontology_court in heard_at_targets
    assert doc_court in heard_at_targets


def test_ontology_entities_in_same_cluster_keep_original_iris(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/case-43"
    court_fr = URIRef("https://growgraph.dev/fcaont#CourAppelRouen")
    court_en = URIRef("https://growgraph.dev/fcaont#AppealCourt_Rouen")
    heard_at = URIRef("https://growgraph.dev/fcaont#heardAt")
    same_as = OWL.sameAs
    rdfs_label = RDFS.label

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    doc:Case1 fcaont:heardAt fcaont:CourAppelRouen .
    doc:Case2 fcaont:heardAt fcaont:AppealCourt_Rouen .
    fcaont:CourAppelRouen rdfs:label "Cour d'appel de Rouen" .
    fcaont:AppealCourt_Rouen rdfs:label "Rouen Court of Appeal" .
    """
    unit = make_fact_unit("Rouen court variants.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((court_fr, rdfs_label, Literal("Cour d'appel de Rouen")))
    ontology_graph.add((court_en, rdfs_label, Literal("Rouen Court of Appeal")))

    def force_ontology_variants_together(representations):
        entities = set(representations.keys())
        if court_fr in entities and court_en in entities:
            remaining = [
                entity for entity in entities if entity not in {court_fr, court_en}
            ]
            return [[court_fr, court_en], *[[entity] for entity in remaining]], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_ontology_variants_together,
    )

    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    assert (court_fr, rdfs_label, Literal("Cour d'appel de Rouen")) in result
    assert (court_en, rdfs_label, Literal("Rouen Court of Appeal")) in result
    assert (court_fr, heard_at, None) not in result
    assert (court_en, heard_at, None) not in result

    heard_at_targets = set(result.objects(None, heard_at))
    assert court_fr in heard_at_targets
    assert court_en in heard_at_targets

    assert (court_fr, same_as, court_en) in result or (
        court_en,
        same_as,
        court_fr,
    ) in result


def test_tentative_ontology_like_alias_maps_to_known_ontology(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/case-44"
    known_court = URIRef("https://growgraph.dev/fcaont#AppealCourtRouen")
    invented_court = URIRef("https://growgraph.dev/fcaont#AppealCourt_Rouen")
    heard_at = URIRef("https://growgraph.dev/fcaont#heardAt")
    court_type = URIRef("https://growgraph.dev/fcaont#Court")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    doc:Case1 fcaont:heardAt fcaont:AppealCourt_Rouen .
    fcaont:AppealCourt_Rouen rdf:type fcaont:Court .
    """
    unit = make_fact_unit("Invented ontology-like alias.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((known_court, RDF.type, court_type))
    ontology_graph.add((known_court, RDFS.label, Literal("Rouen Court of Appeal")))

    def force_known_and_invented_together(representations):
        entities = set(representations.keys())
        if known_court in entities and invented_court in entities:
            remaining = [
                entity
                for entity in entities
                if entity not in {known_court, invented_court}
            ]
            return [
                [known_court, invented_court],
                *[[entity] for entity in remaining],
            ], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_known_and_invented_together,
    )

    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    heard_at_targets = set(result.objects(None, heard_at))
    assert known_court in heard_at_targets
    assert invented_court not in heard_at_targets
    assert (known_court, OWL.sameAs, invented_court) not in result


def test_tentative_only_ontology_like_entities_are_preserved(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/case-45"
    invented_court_1 = URIRef("https://growgraph.dev/fcaont#AppealCourt_Rouen")
    invented_court_2 = URIRef("https://growgraph.dev/fcaont#CourtOfAppealRouen")
    heard_at = URIRef("https://growgraph.dev/fcaont#heardAt")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    doc:Case1 fcaont:heardAt fcaont:AppealCourt_Rouen .
    doc:Case2 fcaont:heardAt fcaont:CourtOfAppealRouen .
    """
    unit = make_fact_unit("Tentative ontology-like terms only.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()

    def force_tentatives_together(representations):
        entities = set(representations.keys())
        if invented_court_1 in entities and invented_court_2 in entities:
            remaining = [
                entity
                for entity in entities
                if entity not in {invented_court_1, invented_court_2}
            ]
            return [
                [invented_court_1, invented_court_2],
                *[[entity] for entity in remaining],
            ], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_tentatives_together,
    )

    result = aggregator.aggregate_graphs([unit])

    heard_at_targets = set(result.objects(None, heard_at))
    assert invented_court_1 in heard_at_targets
    assert invented_court_2 in heard_at_targets


def test_unused_ontology_entities_do_not_create_spurious_sameas() -> None:
    doc_iri = "https://example.org/docs/case-46"
    court_in_facts = URIRef("https://growgraph.dev/fcaont#CourAppelRouen")
    heard_at = URIRef("https://growgraph.dev/fcaont#heardAt")
    court_type = URIRef("https://growgraph.dev/fcaont#AppealCourt")
    unused_a = URIRef("https://growgraph.dev/fcaont#CourAppelParis")
    unused_b = URIRef("https://growgraph.dev/fcaont#CourAppelLyon")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    doc:Case1 fcaont:heardAt fcaont:CourAppelRouen .
    fcaont:CourAppelRouen rdf:type fcaont:AppealCourt .
    """
    unit = make_fact_unit("Case heard at Rouen court of appeal.", 0, doc_iri, ttl)
    ontology_graph = RDFGraph()
    ontology_graph.add((court_in_facts, RDF.type, court_type))
    ontology_graph.add((unused_a, RDF.type, court_type))
    ontology_graph.add((unused_b, RDF.type, court_type))

    result = EmbeddingBasedAggregator().aggregate_graphs(
        [unit], ontology_graph=ontology_graph
    )

    assert (unused_a, OWL.sameAs, unused_b) not in result
    assert (unused_b, OWL.sameAs, unused_a) not in result
    assert court_in_facts in set(result.objects(None, heard_at))


def test_tentative_with_incompatible_type_does_not_merge_to_known_ontology(
    monkeypatch,
) -> None:
    doc_iri = "https://example.org/docs/case-47"
    known_conviction = URIRef("https://growgraph.dev/fcaont#Conviction")
    tentative_person = URIRef("https://growgraph.dev/fcaont#Conviction1")
    associated_with = URIRef("https://growgraph.dev/fcaont#isAssociatedWith")
    conviction_type = URIRef("https://growgraph.dev/fcaont#Conviction")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix schema: <https://schema.org/> .
    doc:Judgment1 fcaont:isAssociatedWith fcaont:Conviction1 .
    fcaont:Conviction1 rdf:type schema:Person .
    """
    unit = make_fact_unit("Person associated with judgment.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((known_conviction, RDF.type, conviction_type))
    ontology_graph.add((known_conviction, RDFS.label, Literal("Conviction")))

    def force_known_and_tentative_together(representations):
        entities = set(representations.keys())
        if known_conviction in entities and tentative_person in entities:
            remaining = [
                entity
                for entity in entities
                if entity not in {known_conviction, tentative_person}
            ]
            return [
                [known_conviction, tentative_person],
                *[[entity] for entity in remaining],
            ], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_known_and_tentative_together,
    )

    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    assert tentative_person in set(result.objects(None, associated_with))
    assert known_conviction not in set(result.objects(None, associated_with))
    assert (known_conviction, OWL.sameAs, tentative_person) not in result


def test_tentative_alias_merged_without_sameas_leak(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/case-47b"
    known_conviction = URIRef("https://growgraph.dev/fcaont#Conviction")
    tentative_alias = URIRef("https://growgraph.dev/fcaont#Conviction1")
    associated_with = URIRef("https://growgraph.dev/fcaont#isAssociatedWith")
    class_type = URIRef("http://www.w3.org/2000/01/rdf-schema#Class")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    doc:Judgment1 fcaont:isAssociatedWith fcaont:Conviction1 .
    fcaont:Conviction1 rdf:type fcaont:Conviction .
    """
    unit = make_fact_unit("Ontology-like alias mention.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((known_conviction, RDF.type, class_type))

    def force_known_and_tentative_together(representations):
        entities = set(representations.keys())
        if known_conviction in entities and tentative_alias in entities:
            remaining = [
                entity
                for entity in entities
                if entity not in {known_conviction, tentative_alias}
            ]
            return [
                [known_conviction, tentative_alias],
                *[[entity] for entity in remaining],
            ], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_known_and_tentative_together,
    )

    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    assert known_conviction in set(result.objects(None, associated_with))
    assert tentative_alias not in set(result.objects(None, associated_with))
    assert (known_conviction, OWL.sameAs, tentative_alias) not in result


def test_non_alias_ontology_terms_do_not_emit_sameas(monkeypatch) -> None:
    doc_iri = "https://example.org/docs/case-48"
    appeal = URIRef("https://growgraph.dev/fcaont#Appeal")
    appeal_decision = URIRef("https://growgraph.dev/fcaont#AppealDecision")
    type_class = URIRef("http://www.w3.org/2000/01/rdf-schema#Class")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    fcaont:Appeal rdf:type rdfs:Class .
    fcaont:AppealDecision rdf:type rdfs:Class .
    """
    unit = make_fact_unit("Ontology class references.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((appeal, RDF.type, type_class))
    ontology_graph.add((appeal_decision, RDF.type, type_class))

    def force_together(representations):
        entities = set(representations.keys())
        if appeal in entities and appeal_decision in entities:
            remaining = [
                entity for entity in entities if entity not in {appeal, appeal_decision}
            ]
            return [[appeal, appeal_decision], *[[entity] for entity in remaining]], {}
        return [list(entities)], {}

    monkeypatch.setattr(aggregator.clusterer, "cluster_entities", force_together)
    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    assert (appeal, OWL.sameAs, appeal_decision) not in result
    assert (appeal_decision, OWL.sameAs, appeal) not in result


def test_entity_in_namespace_accepts_exact_prefix_namespace() -> None:
    entity = URIRef("https://growgraph.dev/factsConviction1")
    assert EmbeddingBasedAggregator._entity_in_namespace(
        entity, "https://growgraph.dev/facts"
    )


def test_fact_entity_forced_with_known_ontology_uses_identity_guard(
    monkeypatch,
) -> None:
    doc_iri = "https://example.org/docs/case-49"
    known_conviction = URIRef("https://growgraph.dev/fcaont#Conviction")
    fact_conviction = URIRef("https://growgraph.dev/factsConviction1")
    associated_with = URIRef("https://growgraph.dev/fcaont#isAssociatedWith")
    class_type = URIRef("http://www.w3.org/2000/01/rdf-schema#Class")

    ttl = f"""
    @prefix doc: <{doc_iri}/> .
    @prefix cd: <https://growgraph.dev/facts> .
    @prefix fcaont: <https://growgraph.dev/fcaont#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix schema: <https://schema.org/> .
    doc:Judgment1 fcaont:isAssociatedWith cd:Conviction1 .
    cd:Conviction1 rdf:type schema:Person .
    """
    unit = make_fact_unit("Forced mixed cluster.", 0, doc_iri, ttl)
    aggregator = EmbeddingBasedAggregator()
    ontology_graph = RDFGraph()
    ontology_graph.add((known_conviction, RDF.type, class_type))

    def force_known_and_fact_together(representations):
        entities = set(representations.keys())
        if known_conviction in entities and fact_conviction in entities:
            remaining = [
                entity
                for entity in entities
                if entity not in {known_conviction, fact_conviction}
            ]
            return [
                [known_conviction, fact_conviction],
                *[[entity] for entity in remaining],
            ], {}
        return [list(entities)], {}

    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        force_known_and_fact_together,
    )

    result = aggregator.aggregate_graphs([unit], ontology_graph=ontology_graph)

    associated_targets = {
        obj for obj in result.objects(None, associated_with) if isinstance(obj, URIRef)
    }
    assert associated_targets
    assert all(str(obj).startswith(doc_iri) for obj in associated_targets)
    assert known_conviction not in associated_targets

    uri_nodes = {
        term for s, _, o in result for term in (s, o) if isinstance(term, URIRef)
    }
    assert all(not str(node).startswith(DEFAULT_IRI) for node in uri_nodes)
