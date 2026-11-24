"""One statement model and one fact scope that every facts rule reads.

``StatementContract`` says what a statement and a value are over the catalog;
``value_components`` reads a value, blank or named; ``FactScope`` says which
triples and nodes are facts. Test data is neutral: ``ex:`` terms only, with
the well-known statement classes as the registry ships them.
"""

from __future__ import annotations

import pytest
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SOSA

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.model import FactsUnitFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.scaffolding import FactScope
from ontocast.tool.facts_validation import ValidationPolicy
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    value_components,
)
from ontocast.tool.facts_validation.terms import CatalogRoles
from ontocast.tool.facts_validation.unit_findings import collect_unit_findings

pytestmark = pytest.mark.unit

EX = "https://ex.org/v#"
CD = DEFAULT_IRI
DATA = "https://ex.org/data/"

_PREFIXES = f"""
@prefix ex:   <{EX}> .
@prefix cd:   <{CD}> .
@prefix data: <{DATA}> .
@prefix sosa: <{SOSA}> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix schema: <https://schema.org/> .
"""

CATALOG = """
ex:Reading a owl:Class ; rdfs:subClassOf sosa:Observation ,
    [ a owl:Restriction ; owl:onProperty ex:measures ; owl:someValuesFrom ex:Item ] ,
    [ a owl:Restriction ; owl:onProperty ex:note ; owl:minCardinality "0"^^xsd:nonNegativeInteger ] ,
    [ a owl:Class ; owl:unionOf (
        [ a owl:Restriction ; owl:onProperty ex:low ; owl:someValuesFrom ex:Amount ]
        [ a owl:Restriction ; owl:onProperty ex:high ; owl:someValuesFrom ex:Amount ] ) ] .
ex:SpotReading a owl:Class ; rdfs:subClassOf ex:Reading ,
    [ a owl:Restriction ; owl:onProperty ex:place ; owl:cardinality "1"^^xsd:nonNegativeInteger ] .
ex:Item a owl:Class .
ex:Amount a owl:Class ; rdfs:subClassOf ex:Value .
ex:Scale a owl:Class .
ex:Grade a owl:Class ; owl:oneOf ( ex:Fine ex:Coarse ) .
ex:degree a ex:Scale .
ex:Fine a ex:Grade .
ex:measures a owl:ObjectProperty ; rdfs:subPropertyOf sosa:observedProperty .
ex:about a owl:ObjectProperty ; owl:inverseOf ex:hasReading .
ex:hasReading a owl:ObjectProperty ; rdfs:subPropertyOf sosa:isFeatureOfInterestOf .
ex:amount a owl:DatatypeProperty ; rdfs:range xsd:decimal .
ex:central a owl:DatatypeProperty ; rdfs:subPropertyOf ex:amount .
ex:scale a owl:ObjectProperty ; rdfs:range ex:Scale .
ex:grade a owl:ObjectProperty ; rdfs:range ex:Grade .
ex:item a owl:ObjectProperty ; rdfs:range ex:Item .
"""


def _graph(body: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=_PREFIXES + body, format="turtle")
    return graph


def _contract() -> StatementContract:
    return CatalogRoles.of([_graph(CATALOG)]).statement_contract(
        frozenset({EX + "Value"})
    )


# -- the contract ----------------------------------------------------------------


def test_catalog_subclasses_inherit_the_required_roles() -> None:
    contract = _contract()

    assert contract.statement_classes[EX + "Reading"] == {"subject", "property"}
    assert contract.required_roles([URIRef(EX + "SpotReading")]) == {
        "subject",
        "property",
    }
    assert contract.required_roles([URIRef(EX + "Item")]) == frozenset()
    assert EX + "Amount" in contract.value_classes


def test_role_links_close_over_sub_properties_and_inverses() -> None:
    contract = _contract()

    assert EX + "measures" in contract.links("property").forward
    assert EX + "hasReading" in contract.links("subject").inverse
    assert EX + "about" in contract.links("subject").forward
    assert contract.links("context").inverse == frozenset()


def test_a_restriction_requiring_a_link_adds_it_and_subclasses_inherit_it() -> None:
    contract = _contract()

    assert contract.required_links[EX + "Reading"] == {EX + "measures"}
    # Inherited, plus its own exact cardinality; a minimum of zero and the
    # members of a union require nothing.
    assert contract.required_links[EX + "SpotReading"] == {
        EX + "measures",
        EX + "place",
    }


def test_a_range_is_closed_when_the_catalog_names_or_enumerates_its_members() -> None:
    contract = _contract()

    assert contract.closed_ranges[EX + "scale"] == {EX + "Scale"}
    assert contract.closed_ranges[EX + "grade"] == {EX + "Grade"}
    assert EX + "item" not in contract.closed_ranges


# -- value components ------------------------------------------------------------


_VALUES = """
data:r1 a ex:Reading ; ex:result [ a ex:Amount ; rdfs:label "about 5 degrees" ;
    ex:central "5"^^xsd:decimal ; ex:scale ex:degree ; ex:grade ex:Fine ] .
data:v2 a ex:Amount ; ex:central 5.0 ; ex:scale ex:degree ; ex:grade ex:Fine .
"""


def test_blank_and_named_values_have_the_same_components() -> None:
    graph = _graph(_VALUES)
    contract = _contract()
    blank = graph.value(URIRef(DATA + "r1"), URIRef(EX + "result"))
    assert isinstance(blank, BNode)

    def summary(node):
        parts = value_components(graph, node, contract)
        return (
            [(c.predicate, c.number) for c in parts.numbers],
            [(c.predicate, str(c.value), c.closed_range) for c in parts.individuals],
        )

    assert summary(blank) == summary(URIRef(DATA + "v2"))
    numbers, individuals = summary(blank)
    assert numbers == [(EX + "central", "5")]
    assert (EX + "scale", EX + "degree", frozenset({EX + "Scale"})) in individuals


def test_a_number_is_read_under_the_catalog_sub_property_closure() -> None:
    graph = _graph(_VALUES)
    parts = value_components(graph, URIRef(DATA + "v2"), _contract())

    assert [c.number for c in parts.numbers_on({EX + "amount"})] == ["5"]
    assert parts.numbers_on({EX + "count"}) == ()


def test_types_labels_and_text_are_no_components() -> None:
    graph = _graph('data:v a ex:Amount ; rdfs:label "5 units" ; ex:note "five" .')
    parts = value_components(graph, URIRef(DATA + "v"), _contract())

    assert parts.numbers == ()
    assert parts.individuals == ()


# -- the fact scope ----------------------------------------------------------------


_PROVENANCE = """
data:doc a prov:Entity , foaf:Document ; ex:title "A document" .
data:chunk a prov:Entity , schema:Text ; ex:position 0 .
data:item a ex:Item ; ex:partOf data:other ; ex:refers data:elsewhere .
[ a rdf:Statement ; rdf:subject data:item ; rdf:predicate ex:partOf ;
  rdf:object data:other ; prov:wasDerivedFrom data:chunk ] .
"""


def test_the_scope_leaves_out_scaffolding_and_the_document_node() -> None:
    scope = FactScope.of(_graph(_PROVENANCE), [DATA])

    assert scope.instances == {
        URIRef(DATA + "item"),
        URIRef(DATA + "other"),
        URIRef(DATA + "elsewhere"),
    }
    assert scope.subjects == {URIRef(DATA + "item")}
    assert not list(scope.triples((URIRef(DATA + "chunk"), None, None)))
    assert (URIRef(DATA + "doc"), URIRef(EX + "title"), Literal("A document")) in (
        scope.facts()
    )


def test_without_namespaces_only_described_nodes_are_instances() -> None:
    scope = FactScope.of(_graph(_PROVENANCE), None)

    assert scope.instances == {URIRef(DATA + "item")}
    assert FactScope.of(_graph(_PROVENANCE), []).instances == frozenset()


def test_a_blank_value_is_held_and_a_blank_reifier_is_not() -> None:
    graph = _graph(_PROVENANCE + _VALUES)
    scope = FactScope.of(graph, [DATA])
    value = graph.value(URIRef(DATA + "r1"), URIRef(EX + "result"))
    reifier = next(graph.subjects(RDF.type, RDF.Statement))

    assert isinstance(value, BNode)
    assert scope.holds(value)
    assert not scope.holds(reifier)


# -- the value checks see blank-node values ------------------------------------


_VOCABULARY = {
    "value_class": EX + "Amount",
    "numeric_value": EX + "amount",
    "unit": EX + "scale",
}


def _findings(graph: RDFGraph, catalog: RDFGraph, **kwargs):
    return collect_unit_findings(
        graph=graph,
        ontology_graph=catalog,
        quarantined=[],
        extraction_text=kwargs.pop("text", ""),
        fact_namespaces=[DATA],
        coverage_limit=0,
        policy=ValidationPolicy(
            quantity_fallback_vocabulary=kwargs.pop("vocabulary", _VOCABULARY),
            vocabulary="open",
        ),
        contract=kwargs.pop("contract", _contract()),
    )


_VALUE_KINDS = {
    FactsUnitFindingKind.LABEL_ONLY_NUMBER,
    FactsUnitFindingKind.SCALAR_AS_BOUNDS,
    FactsUnitFindingKind.UNGROUNDED_NUMBER,
    FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
    FactsUnitFindingKind.MISSING_ROLE,
}


@pytest.mark.parametrize(
    "value",
    [
        # Named, in the fact namespace.
        'data:r1 ex:result data:v . data:v a ex:Amount ; rdfs:label "5 degrees" ;'
        " ex:scale ex:degree .",
        # Blank, as the prompt writes values.
        'data:r1 ex:result [ a ex:Amount ; rdfs:label "5 degrees" ;'
        " ex:scale ex:degree ] .",
    ],
    ids=["named", "blank"],
)
def test_a_label_only_number_is_found_on_a_blank_value_as_on_a_named_one(
    value,
) -> None:
    graph = _graph("data:r1 a ex:Reading ; ex:measures data:item .\n" + value)

    (finding,) = [
        f
        for f in _findings(graph, _graph(CATALOG))
        if f.kind == FactsUnitFindingKind.LABEL_ONLY_NUMBER
    ]
    assert finding.value == "5"
    assert finding.mandatory


def test_a_blank_value_is_named_by_its_carrier_in_the_message() -> None:
    graph = _graph(
        "data:r1 ex:result [ a ex:Amount ; rdfs:label '5 degrees' ;"
        " ex:scale ex:degree ] ."
    )

    (finding,) = [
        f
        for f in _findings(graph, _graph(CATALOG))
        if f.kind == FactsUnitFindingKind.LABEL_ONLY_NUMBER
    ]
    assert finding.message.startswith(f"the value of <{DATA}r1> on <{EX}result>")


def test_equal_bounds_are_found_on_a_blank_value() -> None:
    catalog = _graph(
        CATALOG
        + """
        ex:low a owl:DatatypeProperty , owl:FunctionalProperty .
        ex:high a owl:DatatypeProperty , owl:FunctionalProperty ."""
    )
    graph = _graph("data:r1 ex:result [ a ex:Amount ; ex:low 5 ; ex:high 5.0 ] .")

    kinds = [f.kind for f in _findings(graph, catalog)]
    assert kinds.count(FactsUnitFindingKind.SCALAR_AS_BOUNDS) == 1


def test_value_checks_are_inert_without_a_quantity_vocabulary() -> None:
    """Plain entity relations, numbers without any unit, no quantity terms."""
    catalog = _graph(
        """
        ex:Item a owl:Class .
        ex:partOf a owl:ObjectProperty .
        ex:count a owl:DatatypeProperty , owl:FunctionalProperty .
        ex:year a owl:DatatypeProperty , owl:FunctionalProperty ."""
    )
    graph = _graph(
        """
        data:a a ex:Item ; rdfs:label "item 3" ; ex:partOf data:b ;
            ex:count 3 ; ex:year 2021 .
        data:b a ex:Item ; rdfs:label "item 4" ; ex:count 4 .
        data:a ex:extra [ rdfs:label "ratio 0.5" ; ex:count 7 ] ."""
    )
    contract = CatalogRoles.of([catalog]).statement_contract(frozenset())

    findings = _findings(
        graph, catalog, vocabulary=None, contract=contract, text="3 items in 2021"
    )

    assert not [f for f in findings if f.kind in _VALUE_KINDS]
    assert contract.statement_classes.keys() == {
        str(SOSA.Observation),
        str(SOSA.Actuation),
        str(SOSA.Sampling),
    }
    assert contract.required_links == {}
    assert contract.closed_ranges.keys().isdisjoint(
        {EX + "partOf", EX + "count", EX + "year"}
    )
    assert RDFS.label not in {
        URIRef(c.predicate)
        for c in value_components(graph, URIRef(DATA + "a"), contract).numbers
    }
