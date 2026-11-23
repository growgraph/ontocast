"""Identity from evidence: what merges, in which order, and what keeps two apart.

Equal names merge unless distinctness is shown; a name only overlapping
another waits for a coreference verdict. Distinctness is read on the identity
decided so far: objects by their current cluster, statements by the property
they report, their results and their context. Clustering is stubbed (one
candidate cluster, no embeddings), so no embedding model loads.
"""

import pytest
from rdflib import BNode, Literal, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.well_known import well_known_value_classes
from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator
from ontocast.tool.agg.signatures import normalize_string_value
from ontocast.tool.agg.statement_conflict import (
    StatementIndex,
    dimension_conflict,
    statement_conflict,
    statement_conflict_findings,
)
from ontocast.tool.agg.value_names import merge_equal_values
from ontocast.tool.facts_validation.terms import CatalogRoles

pytestmark = pytest.mark.unit

CD = f"{DEFAULT_IRI}/"
DOC = "https://x.org/doc/1"
EX = "https://x.org/onto#"
QUDT = "http://qudt.org/schema/qudt/"
PREFIXES = f"""
@prefix cd: <{CD}> .
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix qudt: <{QUDT}> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=PREFIXES + body, format="turtle")
    return graph


def _unit(index: int, body: str) -> ContentUnit:
    return ContentUnit(
        text=f"unit {index}",
        index=index,
        doc_iri=URIRef(DOC),
        graph=_graph(body),
        type=OutputType.FACTS,
    )


def _aggregate(monkeypatch, units, ontology: str = "", confirmed=()):
    aggregator = EmbeddingBasedAggregator()
    monkeypatch.setattr(
        aggregator.clusterer,
        "cluster_entities",
        lambda representations: ([list(representations)], {}),
    )
    return aggregator.postprocess_facts_units(
        units,
        _graph(ontology),
        doc_iri=URIRef(DOC),
        confirmed_pairs={
            frozenset((URIRef(CD + left), URIRef(CD + right)))
            for left, right in confirmed
        },
    )


def _together(result, *locals_: str) -> bool:
    wanted = {CD + local for local in locals_}
    return any(wanted <= set(members) for members in result.merged_clusters.values())


# --- distinctness on merged identity ---------------------------------------


def _city_units() -> list[ContentUnit]:
    """One place named in two units, its region minted under two names."""
    return [
        _unit(
            0,
            'cd:city a ex:Place ; rdfs:label "city a" ; ex:inRegion cd:north .\n'
            'cd:north a ex:Region ; rdfs:label "north region" .',
        ),
        _unit(
            1,
            'cd:city a ex:Place ; rdfs:label "city a" ;'
            " ex:inRegion cd:northern_region .\n"
            'cd:northern_region a ex:Region ; rdfs:label "north region" .',
        ),
    ]


def test_functional_objects_compare_by_the_cluster_decided_in_the_same_pass(
    monkeypatch,
) -> None:
    result = _aggregate(
        monkeypatch,
        _city_units(),
        ontology="ex:inRegion a owl:FunctionalProperty .",
    )

    assert _together(result, "north__u0", "northern_region__u1")
    assert _together(result, "city__u0", "city__u1")


def test_objects_that_stay_apart_still_conflict(monkeypatch) -> None:
    units = _city_units()
    units[1] = _unit(
        1,
        'cd:city a ex:Place ; rdfs:label "city a" ; ex:inRegion cd:south .\n'
        'cd:south a ex:Region ; rdfs:label "south region" .',
    )
    result = _aggregate(
        monkeypatch, units, ontology="ex:inRegion a owl:FunctionalProperty ."
    )

    assert not _together(result, "city__u0", "city__u1")
    assert result.rejected_by_reason.get("functional_iri_conflict", 0) >= 1


# --- statement conflict ------------------------------------------------------


def _observation(name: str, subject: str, number: str, extra: str = "") -> str:
    return (
        f'cd:{name} a sosa:Observation ; rdfs:label "{name}" ;\n'
        f"    sosa:hasFeatureOfInterest cd:{subject} ;\n"
        "    sosa:observedProperty ex:size ;\n"
        f'    sosa:hasResult [ a qudt:QuantityValue ; qudt:numericValue "{number}"^^xsd:decimal ] {extra}.\n'
    )


def _hub_units(first: str, second: str) -> list[ContentUnit]:
    """A generic mention G and two specific samples whose sizes conflict."""
    return [
        _unit(
            0,
            f'cd:{first} a ex:Sample ; rdfs:label "first sample" .\n'
            + _observation("obs_a", first, "6.8"),
        ),
        _unit(
            1,
            f'cd:{second} a ex:Sample ; rdfs:label "second sample" .\n'
            + _observation("obs_b", second, "8.5"),
        ),
        _unit(2, 'cd:generic a ex:Sample ; rdfs:label "sample" .'),
    ]


@pytest.mark.parametrize(("first", "second"), [("s1", "s2"), ("s2", "s1")])
def test_a_hub_confirmed_with_two_conflicting_samples_chains_nothing(
    monkeypatch, first: str, second: str
) -> None:
    """Whichever IRI sorts first, the two samples never share a node."""
    result = _aggregate(
        monkeypatch,
        _hub_units(first, second),
        confirmed=[("generic__u2", f"{first}__u0"), ("generic__u2", f"{second}__u1")],
    )

    assert not _together(result, f"{first}__u0", f"{second}__u1")
    assert result.rejected_by_reason.get("statement_conflict", 0) >= 1


def test_a_time_series_of_one_sample_is_not_a_conflict(monkeypatch) -> None:
    units = [
        _unit(
            0,
            'cd:s a ex:Sample ; rdfs:label "sample a" .\n'
            + _observation(
                "obs_a",
                "s",
                "6.8",
                '; sosa:resultTime "2020-01-01T00:00:00"^^xsd:dateTime ',
            ),
        ),
        _unit(
            1,
            'cd:s a ex:Sample ; rdfs:label "sample a" .\n'
            + _observation(
                "obs_b",
                "s",
                "8.5",
                '; sosa:resultTime "2020-02-01T00:00:00"^^xsd:dateTime ',
            ),
        ),
    ]

    assert _together(_aggregate(monkeypatch, units), "s__u0", "s__u1")


def test_the_same_samples_without_a_differing_context_stay_apart(monkeypatch) -> None:
    units = [
        _unit(
            0,
            'cd:s a ex:Sample ; rdfs:label "sample a" .\n'
            + _observation("obs_a", "s", "6.8"),
        ),
        _unit(
            1,
            'cd:s a ex:Sample ; rdfs:label "sample a" .\n'
            + _observation("obs_b", "s", "8.5"),
        ),
    ]

    result = _aggregate(monkeypatch, units)

    assert not _together(result, "s__u0", "s__u1")
    assert result.rejected_by_reason.get("statement_conflict", 0) >= 1


def test_the_gate_check_finds_nothing_on_a_served_graph_and_a_conflict_on_a_merged_one() -> (
    None
):
    graph = _graph(
        'cd:s a ex:Sample ; rdfs:label "sample a" .\n'
        + _observation("obs_a", "s", "6.8")
        + _observation("obs_b", "s", "8.5")
    )
    contract = CatalogRoles().statement_contract(frozenset())
    index = StatementIndex.of([graph], contract)

    assert statement_conflict(index, URIRef(CD + "s"), URIRef(CD + "s")) == URIRef(
        EX + "size"
    )
    findings = statement_conflict_findings(graph, None, [CD])
    assert [f.subject for f in findings] == [CD + "s"]
    assert (
        statement_conflict_findings(
            _graph(_observation("obs_a", "s", "6.8")), None, [CD]
        )
        == []
    )


# --- dimension ---------------------------------------------------------------


def _reading(name: str, prop: str, unit: str | None) -> str:
    """A statement reporting the property node ``cd:<prop>`` with one result."""
    unit_link = f" ; qudt:unit unit:{unit}" if unit else ""
    return (
        f"cd:{name} a sosa:Observation ;\n"
        f"    sosa:observedProperty cd:{prop} ;\n"
        "    sosa:hasResult [ a qudt:QuantityValue ;"
        f' qudt:numericValue "4"^^xsd:decimal{unit_link} ] .\n'
        f'cd:{prop} a sosa:ObservableProperty ; rdfs:label "extent" .\n'
    )


def _property_units(first: str | None, second: str | None) -> list[ContentUnit]:
    """One property label in two units, its results in the given units."""
    return [
        _unit(0, _reading("reading_a", "extent", first)),
        _unit(1, _reading("reading_b", "extent", second)),
    ]


def test_properties_whose_results_differ_in_dimension_stay_apart(monkeypatch) -> None:
    result = _aggregate(monkeypatch, _property_units("M", "SEC"))

    assert not _together(result, "extent__u0", "extent__u1")
    assert result.rejected_by_reason.get("dimension_conflict", 0) >= 1


def test_properties_whose_results_share_a_dimension_merge(monkeypatch) -> None:
    result = _aggregate(monkeypatch, _property_units("M", "KiloM"))

    assert _together(result, "extent__u0", "extent__u1")


@pytest.mark.parametrize(
    "units",
    [(None, None), ("M", None), (None, "SEC")],
    ids=["no unit", "one side without a unit", "other side without a unit"],
)
def test_the_dimension_veto_is_inert_without_a_unit(monkeypatch, units) -> None:
    result = _aggregate(monkeypatch, _property_units(*units))

    assert _together(result, "extent__u0", "extent__u1")
    assert "dimension_conflict" not in result.rejected_by_reason


def test_a_unit_of_no_known_dimension_decides_nothing() -> None:
    graph = _graph(
        _reading("reading_a", "first", "M")
        + _reading("reading_b", "second", "M").replace("unit:M", "ex:ownUnit")
    )
    index = StatementIndex.of([graph], CatalogRoles().statement_contract(frozenset()))

    assert index.dimensions_of(URIRef(CD + "first"))
    assert not index.dimensions_of(URIRef(CD + "second"))
    assert (
        dimension_conflict(index, URIRef(CD + "first"), URIRef(CD + "second")) is None
    )


# --- names -------------------------------------------------------------------


def test_text_keeps_case_changes_so_no_false_initial_appears() -> None:
    assert normalize_string_value("large ABcd set") == "large abcd set"


def test_a_case_change_inside_a_word_is_no_initial(monkeypatch) -> None:
    units = [
        _unit(0, 'cd:x a ex:Item ; rdfs:label "ABcd" .'),
        _unit(1, 'cd:y a ex:Item ; rdfs:label "alpha bcd" .'),
    ]

    assert not _together(_aggregate(monkeypatch, units), "x__u0", "y__u1")


def test_labels_differing_in_one_marked_component_do_not_merge(monkeypatch) -> None:
    units = [
        _unit(0, 'cd:x a ex:Part ; rdfs:label "unit A-site component" .'),
        _unit(1, 'cd:y a ex:Part ; rdfs:label "unit B-site component" .'),
    ]

    assert not _together(_aggregate(monkeypatch, units), "x__u0", "y__u1")


def test_an_overlapping_name_merges_on_a_same_verdict(monkeypatch) -> None:
    units = [
        _unit(0, 'cd:batch a ex:Sample ; rdfs:label "sample batch" .'),
        _unit(1, 'cd:large a ex:Sample ; rdfs:label "large sample batch" .'),
    ]

    assert not _together(_aggregate(monkeypatch, units), "batch__u0", "large__u1")
    assert _together(
        _aggregate(monkeypatch, units, confirmed=[("batch__u0", "large__u1")]),
        "batch__u0",
        "large__u1",
    )


# --- values -----------------------------------------------------------------


def _results(result, local: str) -> list:
    return [
        obj
        for subject, obj in result.graph.subject_objects(
            URIRef("http://www.w3.org/ns/sosa/hasResult")
        )
        if str(subject).endswith(local)
    ]


def test_two_equal_values_from_two_units_become_one(monkeypatch) -> None:
    units = [
        _unit(
            0,
            'cd:obs a sosa:Observation ; rdfs:label "obs" ; sosa:observedProperty ex:size ;\n'
            '  sosa:hasResult [ a qudt:QuantityValue ; qudt:numericValue "12.5"^^xsd:decimal ;'
            " qudt:unit unit:M ] .",
        ),
        _unit(
            1,
            'cd:obs a sosa:Observation ; rdfs:label "obs" ; sosa:observedProperty ex:size ;\n'
            '  sosa:hasResult [ a qudt:QuantityValue ; qudt:numericValue "12.5"^^xsd:decimal ;'
            ' rdfs:comment "m" ] .',
        ),
    ]

    result = _aggregate(monkeypatch, units)

    values = _results(result, "/obs")
    assert len(values) == 1
    value = values[0]
    assert (value, URIRef(QUDT + "unit"), None) in result.graph
    assert (value, URIRef("http://www.w3.org/2000/01/rdf-schema#comment"), None) in (
        result.graph
    )


def _two_values(first: str, second: str) -> RDFGraph:
    return _graph(
        f"cd:obs sosa:hasResult [ a qudt:QuantityValue ; {first} ] ,"
        f" [ a qudt:QuantityValue ; {second} ] ."
    )


def _merge(graph: RDFGraph) -> int:
    return merge_equal_values(
        graph,
        CatalogRoles().statement_contract(frozenset(well_known_value_classes())),
    )


def test_an_estimate_and_its_interval_stay_two_values() -> None:
    graph = _two_values(
        'qudt:numericValue "2.0"^^xsd:decimal',
        'ex:lower "1.2"^^xsd:decimal ; ex:upper "2.8"^^xsd:decimal',
    )

    assert _merge(graph) == 0
    assert len(list(graph.objects(URIRef(CD + "obs"), None))) == 2


def test_one_number_in_two_measurement_units_is_two_values() -> None:
    graph = _two_values(
        'qudt:numericValue "12.5"^^xsd:decimal ; qudt:unit unit:K',
        'qudt:numericValue "12.5"^^xsd:decimal ; qudt:unit unit:DEG_C',
    )

    assert _merge(graph) == 0


def test_a_value_merged_away_keeps_its_triples_on_the_survivor() -> None:
    graph = _two_values(
        'qudt:numericValue "12.5"^^xsd:decimal',
        'qudt:numericValue "12.50"^^xsd:decimal ; ex:qualifier ex:exact',
    )

    assert _merge(graph) == 1
    (value,) = list(graph.objects(URIRef(CD + "obs"), None))
    assert isinstance(value, BNode)
    assert (value, URIRef(EX + "qualifier"), URIRef(EX + "exact")) in graph
    # One number, one lexical form: the shortest stays.
    assert set(graph.objects(value, URIRef(QUDT + "numericValue"))) == {
        Literal("12.5", datatype=URIRef("http://www.w3.org/2001/XMLSchema#decimal"))
    }


# --- a veto read on the identity one round builds ---------------------------


def _place_unit(index: int, place: str, measure: str, count: str) -> ContentUnit:
    return _unit(
        index,
        f'cd:{place} a ex:Place ; rdfs:label "place one" .\n'
        f'cd:{measure} a sosa:ObservableProperty ; rdfs:label "head count" .\n'
        f'cd:obs{index} a sosa:Observation ; rdfs:label "obs {index}" ;\n'
        f"    sosa:hasFeatureOfInterest cd:{place} ;"
        f" sosa:observedProperty cd:{measure} ;\n"
        "    sosa:hasResult [ a qudt:QuantityValue ;"
        f' qudt:numericValue "{count}"^^xsd:decimal ] .',
    )


@pytest.mark.parametrize(
    ("place", "measure", "counts"),
    [
        ("place", "measure", ("100", "200")),
        ("place", "measure", ("200", "100")),
        # IRI order reversed: the place sorts after the measure.
        ("zz_place", "aa_measure", ("100", "200")),
        ("aa_place", "zz_measure", ("200", "100")),
    ],
)
def test_two_places_reporting_one_measure_differently_stay_two(
    monkeypatch, place: str, measure: str, counts: tuple[str, str]
) -> None:
    """Both pairs merge on equal labels in one round. Once the measures are
    one, the places report it with conflicting results: the union of the
    places is undone, whatever the IRI order."""
    result = _aggregate(
        monkeypatch,
        [
            _place_unit(0, place, measure, counts[0]),
            _place_unit(1, place, measure, counts[1]),
        ],
    )

    assert _together(result, f"{measure}__u0", f"{measure}__u1")
    assert not _together(result, f"{place}__u0", f"{place}__u1")
    assert statement_conflict_findings(result.graph, None, [CD, DOC + "/"]) == []


# --- only values merge, and only when every text agrees ---------------------


def test_amounts_in_two_currencies_stay_two_values() -> None:
    graph = _graph(
        "@prefix schema: <https://schema.org/> ."
        " cd:co schema:revenue"
        ' [ a schema:MonetaryAmount ; schema:value 5 ; schema:currency "AAA" ] ,'
        ' [ a schema:MonetaryAmount ; schema:value 5 ; schema:currency "BBB" ] .'
    )
    assert _merge(graph) == 0


def test_two_addresses_sharing_a_postal_code_stay_two() -> None:
    graph = _graph(
        "@prefix schema: <https://schema.org/> ."
        " cd:co schema:address"
        ' [ a schema:PostalAddress ; schema:postalCode "10115" ;'
        ' schema:streetAddress "First road 1" ] ,'
        ' [ a schema:PostalAddress ; schema:postalCode "10115" ;'
        ' schema:streetAddress "Second road 2" ] .'
    )
    assert _merge(graph) == 0


def test_blank_nodes_of_no_value_class_never_merge() -> None:
    graph = _graph(
        "@prefix schema: <https://schema.org/> ."
        ' cd:co ex:member [ schema:name "first" ; ex:age 40 ] ,'
        ' [ schema:name "second" ; ex:age 40 ] , [ ex:age 40 ] .'
    )
    assert _merge(graph) == 0
    assert len(list(graph.objects(URIRef(CD + "co"), URIRef(EX + "member")))) == 3


def test_a_value_merged_twice_ends_on_one_survivor() -> None:
    """A value merged into one that is itself merged away follows it."""
    graph = _graph(
        "cd:a ex:has _:v1 , _:v2 . cd:b ex:has _:v2 , _:v3 ."
        ' _:v1 a qudt:QuantityValue ; qudt:numericValue "7"^^xsd:decimal .'
        ' _:v2 a qudt:QuantityValue ; qudt:numericValue "7"^^xsd:decimal ;'
        " qudt:unit unit:M ."
        ' _:v3 a qudt:QuantityValue ; qudt:numericValue "7.0"^^xsd:decimal .'
    )
    _merge(graph)

    values = {
        value
        for carrier in ("a", "b")
        for value in graph.objects(URIRef(CD + carrier), URIRef(EX + "has"))
    }
    assert len(values) == 1
    (value,) = values
    assert (value, URIRef(QUDT + "unit"), None) in graph
    assert len(set(graph.subjects(URIRef(QUDT + "numericValue"), None))) == 1
