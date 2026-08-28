"""Regressions from the art6 (ECHR) benchmark run of 2026-08-11.

`benchmarking/art6/result_c` exhibited simultaneous over- and under-merging:

- four judges ("Mrs E. Palm", "Mrs M. Tsatsa-Nikolovska", "Mrs N. Vajić",
  "Mrs W. Thomassen") collapsed into one node — a shared honorific literal
  acted as label agreement, and pairwise vetoes were chained around by the
  union-find's transitive closure;
- three companies ("French company S.", "French company T.", "Italian
  company T.I.") collapsed — the tokenizer dropped the distinguishing
  initials;
- ESA merged with its own Appeals Board, producing a self-referential
  ``schema:isPartOf``;
- two case nodes sharing the identical ``hasApplicationNumber`` literal were
  left split — identical identifiers carried no positive identity evidence.

Each test pins the corresponding fix. Clustering is stubbed (single cluster
or singletons) so no embedding model loads.
"""

from rdflib import RDF, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator
from ontocast.tool.agg.rewriter import GraphRewriter
from ontocast.tool.agg.signatures import labels_differ_only_by_initials

CD = f"{DEFAULT_IRI}/"
DOC = "https://x.org/doc/1"
ECHR = "https://x.org/echr#"
SCHEMA = "https://schema.org/"

_PREFIXES = f"""
@prefix cd: <{CD}> .
@prefix echr: <{ECHR}> .
@prefix schema: <{SCHEMA}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _fact_unit(index: int, ttl: str) -> ContentUnit:
    graph = RDFGraph()
    graph.parse(data=_PREFIXES + ttl, format="turtle")
    return ContentUnit(
        text="text",
        index=index,
        doc_iri=URIRef(DOC),
        graph=graph,
        type=OutputType.FACTS,
    )


def _single_cluster_aggregator(monkeypatch) -> EmbeddingBasedAggregator:
    """All entities of a role in one candidate cluster, no embeddings."""
    aggregator = EmbeddingBasedAggregator()
    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        lambda representations: ([list(representations.keys())], {}),
    )
    return aggregator


def _singleton_cluster_aggregator(
    monkeypatch, aggregator: EmbeddingBasedAggregator | None = None
) -> EmbeddingBasedAggregator:
    """Every entity alone — nothing merges unless key evidence bridges."""
    aggregator = aggregator or EmbeddingBasedAggregator()
    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        lambda representations: ([[entity] for entity in representations], {}),
    )
    return aggregator


def _cluster_members(result) -> list[set[str]]:
    return [set(members) for members in result.merged_clusters.values()]


def _never_together(result, left: str, right: str) -> bool:
    return all(
        not {CD + left, CD + right} <= members for members in _cluster_members(result)
    )


# --- four judges must stay four -----------------------------------------------


def test_shared_honorific_is_not_label_agreement_and_chains_do_not_close(
    monkeypatch,
) -> None:
    """Distinct 'Mrs X' judges stay distinct despite a label-only bridge.

    Every judge carries the honorific literal that used to be harvested as an
    alt-label ('Mrs' passes the 3-char floor where 'Mr' did not — which is
    exactly why only the female judges collapsed in result_c). The label-only
    bridge node is mergeable with Palm; the chain must not pull Thomassen in.
    """
    units = [
        _fact_unit(
            0,
            """
            cd:mrsEPalm a echr:JudicialOfficer ; rdfs:label "Mrs E. Palm"@en ;
                echr:hasHonorific "Mrs"@en ;
                echr:hasPersonName "Mrs E. Palm"^^xsd:string .
            """,
        ),
        _fact_unit(
            1,
            """
            cd:mrsWThomassen a echr:JudicialOfficer ;
                rdfs:label "Mrs W. Thomassen"@en ;
                echr:hasHonorific "Mrs"@en ;
                echr:hasPersonName "Mrs W. Thomassen"^^xsd:string .
            """,
        ),
        _fact_unit(
            2,
            """
            cd:mrsPalmMention a echr:JudicialOfficer ;
                rdfs:label "Mrs E. Palm"@en ;
                echr:hasHonorific "Mrs"@en .
            """,
        ),
    ]
    aggregator = _single_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(units, ontology_graph=RDFGraph())

    assert _never_together(result, "mrsEPalm", "mrsWThomassen")
    # The genuine re-mention still merges.
    assert any(
        {CD + "mrsEPalm", CD + "mrsPalmMention"} <= members
        for members in _cluster_members(result)
    )


# --- companies differing only by initials -------------------------------------


def test_labels_differ_only_by_initials_helper() -> None:
    assert labels_differ_only_by_initials({"french company s"}, {"french company t"})
    # An initial expanding or matching the other side is spelling variance.
    assert not labels_differ_only_by_initials({"u.s. government"}, {"us government"})
    assert not labels_differ_only_by_initials({"mr beer"}, {"mr karlheinz beer"})
    assert not labels_differ_only_by_initials({"j. r. smith"}, {"j. smith"})


def test_companies_differing_only_by_initial_do_not_merge(monkeypatch) -> None:
    units = [
        _fact_unit(
            0,
            """
            cd:companyS a schema:Organization ; rdfs:label "French company S."@en .
            """,
        ),
        _fact_unit(
            1,
            """
            cd:companyT a schema:Organization ; rdfs:label "French company T."@en .
            """,
        ),
    ]
    aggregator = _single_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(units, ontology_graph=RDFGraph())
    assert _never_together(result, "companyS", "companyT")


# --- part-of self-loops --------------------------------------------------------


def test_direct_relation_veto_holds_through_a_bridge(monkeypatch) -> None:
    """ESA and its Appeals Board must not merge via an intermediate alias."""
    units = [
        _fact_unit(
            0,
            """
            cd:esaAppealsBoard a schema:Organization ;
                rdfs:label "ESA Appeals Board"@en ;
                schema:isPartOf cd:esa .
            cd:esa a schema:Organization ; rdfs:label "ESA"@en .
            """,
        ),
        _fact_unit(
            1,
            """
            cd:esaAgency a schema:Organization ;
                rdfs:label "ESA"@en, "ESA Appeals Board"@en .
            """,
        ),
    ]
    aggregator = _single_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(units, ontology_graph=RDFGraph())

    assert _never_together(result, "esa", "esaAppealsBoard")
    is_part_of = URIRef(f"{SCHEMA}isPartOf")
    assert not [
        (s, o) for s, _, o in result.graph.triples((None, is_part_of, None)) if s == o
    ]


def test_rewriter_drops_merge_created_self_loops_only() -> None:
    graph = RDFGraph()
    part = URIRef(CD + "board")
    whole = URIRef(CD + "agency")
    reflexive = URIRef(CD + "self_ref")
    predicate = URIRef(f"{SCHEMA}isPartOf")
    graph.add((part, predicate, whole))
    graph.add((reflexive, predicate, reflexive))
    unit = ContentUnit(
        text="text",
        index=0,
        doc_iri=URIRef(DOC),
        graph=graph,
        type=OutputType.FACTS,
    )
    rewriter = GraphRewriter(add_sameas_links=False)
    merged = rewriter.merge_graphs_with_provenance([unit], {part: whole})

    # The merge-created loop is dropped; the author-asserted one survives.
    assert (whole, predicate, whole) not in merged
    assert (reflexive, predicate, reflexive) in merged


# --- natural-key identity evidence ---------------------------------------------


def _case_units(number_a: str, number_b: str) -> list[ContentUnit]:
    return [
        _fact_unit(
            0,
            f"""
            cd:case_36760_06 a echr:CaseDocument ;
                rdfs:label "Application no. {number_a}"@en ;
                echr:hasApplicationNumber "{number_a}"^^xsd:string .
            """,
        ),
        _fact_unit(
            1,
            f"""
            cd:caseStanevVBulgaria a echr:CaseDocument ;
                rdfs:label "CASE OF STANEV v. BULGARIA"@en ;
                echr:hasApplicationNumber "{number_b}"^^xsd:string .
            """,
        ),
    ]


def test_shared_application_number_merges_the_two_case_nodes(monkeypatch) -> None:
    """Identical values on a single-valued identifier predicate are identity.

    The two nodes share no label vocabulary and (with singleton candidate
    clusters) no embedding agreement — the key value alone must bridge them.
    """
    aggregator = _singleton_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(
        _case_units("36760/06", "36760/06"), ontology_graph=RDFGraph()
    )
    assert any(
        {CD + "case_36760_06", CD + "caseStanevVBulgaria"} <= members
        for members in _cluster_members(result)
    )


def test_differing_application_numbers_do_not_merge(monkeypatch) -> None:
    aggregator = _singleton_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(
        _case_units("36760/06", "28934/95"), ontology_graph=RDFGraph()
    )
    assert _never_together(result, "case_36760_06", "caseStanevVBulgaria")


def test_natural_key_merge_can_be_disabled(monkeypatch) -> None:
    from ontocast.config import AggregationConfig

    aggregator = _singleton_cluster_aggregator(
        monkeypatch,
        EmbeddingBasedAggregator(AggregationConfig(natural_key_merge=False)),
    )
    result = aggregator.aggregate_graphs(
        _case_units("36760/06", "36760/06"), ontology_graph=RDFGraph()
    )
    assert _never_together(result, "case_36760_06", "caseStanevVBulgaria")


# --- prose payloads are not keys ------------------------------------------------


def test_long_shared_prose_is_not_key_evidence(monkeypatch) -> None:
    note = "The Labour Court declared the action inadmissible, finding immunity."
    units = [
        _fact_unit(
            0,
            f"""
            cd:proceedingFirst a echr:DomesticProceeding ;
                rdfs:label "labour court proceeding - first applicant"@en ;
                echr:hasExtractionNote "{note}"^^xsd:string .
            """,
        ),
        _fact_unit(
            1,
            f"""
            cd:proceedingSecond a echr:DomesticProceeding ;
                rdfs:label "labour court proceeding - second applicant"@en ;
                echr:hasExtractionNote "{note}"^^xsd:string .
            """,
        ),
    ]
    aggregator = _singleton_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(units, ontology_graph=RDFGraph())
    assert _never_together(result, "proceedingFirst", "proceedingSecond")


# --- gate visibility of label explosions ----------------------------------------


def test_string_multi_value_finding_flags_collapsed_names() -> None:
    from ontocast.tool.facts_validation import validate_aggregated_facts

    graph = RDFGraph()
    name = URIRef(f"{ECHR}hasPersonName")
    # Dominance: three well-behaved person nodes with one name each.
    for index in range(3):
        person = URIRef(CD + f"person_{index}")
        graph.add((person, RDF.type, URIRef(f"{ECHR}NaturalPerson")))
        graph.add((person, name, Literal(f"Judge Number {index}")))
    merged = URIRef(CD + "mrsEPalm")
    graph.add((merged, RDF.type, URIRef(f"{ECHR}NaturalPerson")))
    for value in (
        "Mrs E. Palm",
        "Mrs M. Tsatsa-Nikolovska",
        "Mrs N. Vajić",
        "Mrs W. Thomassen",
    ):
        graph.add((merged, name, Literal(value)))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])
    flagged = [
        finding
        for finding in report.findings
        if finding.subject == str(merged) and finding.predicate == str(name)
    ]
    assert flagged and flagged[0].severity == "error"


def test_compatible_name_variants_are_not_flagged() -> None:
    from ontocast.tool.facts_validation import validate_aggregated_facts

    graph = RDFGraph()
    name = URIRef(f"{ECHR}hasPersonName")
    for index in range(3):
        person = URIRef(CD + f"person_{index}")
        graph.add((person, name, Literal(f"Judge Number {index}")))
    beer = URIRef(CD + "personBeer")
    graph.add((beer, name, Literal("Mr Beer")))
    graph.add((beer, name, Literal("Mr Karlheinz Beer")))

    report = validate_aggregated_facts(graph, None, fact_namespaces=[CD])
    assert not [finding for finding in report.findings if finding.subject == str(beer)]


# --- literal variant dedupe -----------------------------------------------------


def test_dedupe_literal_variants_keeps_the_language_tagged_form() -> None:
    from ontocast.tool.facts_validation import dedupe_literal_variants

    graph = RDFGraph()
    subject = URIRef(CD + "personBeer")
    name = URIRef(f"{ECHR}hasPersonName")
    graph.add((subject, name, Literal("Mr Karlheinz Beer", lang="en")))
    graph.add((subject, name, Literal("Mr Karlheinz Beer", datatype=XSD.string)))
    graph.add((subject, name, Literal("Mr Karlheinz Beer")))
    graph.add((subject, name, Literal("Mr Beer", lang="en")))

    records = dedupe_literal_variants(graph, [CD])

    values = set(graph.objects(subject, name))
    assert values == {
        Literal("Mr Karlheinz Beer", lang="en"),
        Literal("Mr Beer", lang="en"),
    }
    assert len(records) == 2
    assert all(str(record.kind) == "literal_variant_pruned" for record in records)


def test_key_supported_merge_label_variance_is_warning_not_error() -> None:
    from ontocast.tool.facts_validation import validate_aggregated_facts

    graph = RDFGraph()
    name = URIRef(f"{ECHR}hasCaseName")
    for index in range(3):
        case = URIRef(CD + f"case_{index}")
        graph.add((case, name, Literal(f"Case Number {index}")))
    merged = URIRef(CD + "case_36760_06")
    graph.add((merged, name, Literal("Application no. 36760/06")))
    graph.add((merged, name, Literal("Case of Stanev v. Bulgaria")))

    flagged = [
        finding
        for finding in validate_aggregated_facts(
            graph, None, fact_namespaces=[CD]
        ).findings
        if finding.subject == str(merged)
    ]
    assert flagged and flagged[0].severity == "error"

    downgraded = [
        finding
        for finding in validate_aggregated_facts(
            graph,
            None,
            fact_namespaces=[CD],
            key_supported_subjects=[str(merged)],
        ).findings
        if finding.subject == str(merged)
    ]
    assert downgraded and downgraded[0].severity == "warning"


def test_natural_key_merge_reports_key_supported_cluster(monkeypatch) -> None:
    aggregator = _singleton_cluster_aggregator(monkeypatch)
    result = aggregator.aggregate_graphs(
        _case_units("36760/06", "36760/06"), ontology_graph=RDFGraph()
    )
    assert result.key_supported_clusters
    assert set(result.key_supported_clusters) <= set(result.merged_clusters)
