"""One shown prefix per namespace: the table's rules, and storage keeping them."""

from __future__ import annotations

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import SKOS

from ontocast.onto.prefixes import (
    PrefixTable,
    build_prefix_table,
    valid_shown_prefix,
)
from ontocast.onto.rdfgraph import RDFGraph
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit

EVT = "https://ex.org/eventlog#"
REG = "https://ex.org/registry#"
AST = "https://ex.org/assets#"
UNIT = "http://qudt.org/vocab/unit/"

EVENTLOG_TTL = f"""
@prefix ev: <{EVT}> .
@prefix reg:  <{REG}> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://ex.org/eventlog> a owl:Ontology .
ev:Incident a owl:Class ; rdfs:subClassOf reg:Event .
ev:hasSpan a owl:ObjectProperty ; rdfs:domain reg:Event .
"""
REGISTRY_TTL = f"""
@prefix reg: <{REG}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<https://ex.org/registry> a owl:Ontology .
reg:Event a owl:Class .
reg:hasCondition a owl:ObjectProperty .
"""
UNITS_TTL = f"""
@prefix unit: <{UNIT}> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix quantitykind: <http://qudt.org/vocab/quantitykind/> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://ex.org/units> a owl:Ontology .
unit:DEG_C a qudt:Unit ; rdfs:label "degree Celsius" ;
    qudt:hasQuantityKind quantitykind:Temperature .
unit:NanoM a qudt:Unit ; rdfs:label "nanometre" .
"""
ASSETS_TTL = f"""
@prefix el:   <{EVT}> .
@prefix ast:  <{AST}> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://ex.org/assets> a owl:Ontology .
ast:Asset a owl:Class .
ast:affectedBy a owl:ObjectProperty ; rdfs:range el:Incident .
"""


def _catalog():
    return [
        stamped_ontology(EVENTLOG_TTL, ontology_id="eventlog"),
        stamped_ontology(REGISTRY_TTL, ontology_id="registry"),
        stamped_ontology(UNITS_TTL, ontology_id="units"),
        stamped_ontology(ASSETS_TTL, ontology_id="assets"),
    ]


def test_adoption_shows_author_prefixes_and_standard_ones() -> None:
    eventlog, registry, units, assets = _catalog()

    assert (eventlog.ontology_id, eventlog.prefix) == ("eventlog", "ev")
    assert (registry.ontology_id, registry.prefix) == ("registry", "reg")
    # Terms in a well-known namespace keep its standard prefix, not the id.
    assert units.namespace == UNIT
    assert (units.ontology_id, units.prefix) == ("units", "unit")
    assert assets.prefix == "ast"


def test_the_table_shows_one_prefix_per_namespace() -> None:
    table = build_prefix_table(_catalog())

    assert table.prefix_for(EVT) == "ev"
    assert table.prefix_for(REG) == "reg"
    assert table.prefix_for(AST) == "ast"
    assert table.prefix_for(UNIT) == "unit"
    assert table.prefix_for("http://www.w3.org/ns/sosa/") == "sosa"
    # The id and a borrowing file's prefix are aliases, never shown.
    assert table.aliases["eventlog"] == EVT
    assert table.aliases["el"] == EVT
    assert "ev" not in table.aliases
    assert table.qname(f"{EVT}Incident") == "ev:Incident"
    assert table.qname(f"{UNIT}DEG_C") == "unit:DEG_C"


def test_the_parse_map_accepts_shown_prefixes_and_aliases() -> None:
    parse = build_prefix_table(_catalog()).parse_map()

    for prefix in ("ev", "eventlog", "el"):
        assert parse[prefix] == EVT
    assert parse["unit"] == UNIT
    assert parse["units"] == UNIT


def test_an_alias_two_namespaces_claim_is_dropped() -> None:
    other = stamped_ontology(
        """
        @prefix el:  <https://ex.org/lightcurve#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://ex.org/lightcurve> a owl:Ontology .
        el:Curve a owl:Class .
        """,
        ontology_id="lightcurve",
    )
    table = build_prefix_table([*_catalog(), other])

    # ``el`` is the light-curve ontology's own prefix, so it is shown there and
    # is no longer an alias of the event-log namespace.
    assert table.prefix_for("https://ex.org/lightcurve#") == "el"
    assert "el" not in table.aliases


def test_a_clashing_author_prefix_falls_back_to_the_id() -> None:
    first = stamped_ontology(
        """
        @prefix ex:  <https://ex.org/one#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://ex.org/one> a owl:Ontology .
        ex:A a owl:Class .
        """,
        ontology_id="one",
    )
    second = stamped_ontology(
        """
        @prefix ex:  <https://ex.org/two#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://ex.org/two> a owl:Ontology .
        ex:B a owl:Class .
        """,
        ontology_id="two",
    )
    table = build_prefix_table([first, second])

    # ``ex`` is the standard prefix of example.org, so neither may show it.
    assert table.prefix_for("https://ex.org/one#") == "one"
    assert table.prefix_for("https://ex.org/two#") == "two"


@pytest.mark.parametrize(
    ("prefix", "namespace", "valid"),
    [
        ("ev", EVT, True),
        ("time", EVT, False),
        ("time1", EVT, False),
        ("ns1", EVT, False),
        ("cd", EVT, False),
        ("unit", UNIT, True),
        ("1x", EVT, False),
    ],
)
def test_valid_shown_prefix(prefix: str, namespace: str, valid: bool) -> None:
    assert valid_shown_prefix(prefix, namespace) is valid


def test_rewrite_touches_only_curies_of_known_prefixes() -> None:
    table = build_prefix_table(_catalog())
    text = (
        "Use eventlog:hasSpan, not el:Incident. See "
        "https://ex.org/eventlog#Incident at 10:30; cd:x and onto:Trial "
        "stay; ev:hasSpan stays."
    )

    rewritten = table.rewrite_curies(text)

    assert rewritten == (
        "Use ev:hasSpan, not ev:Incident. See "
        "https://ex.org/eventlog#Incident at 10:30; cd:x and onto:Trial "
        "stay; ev:hasSpan stays."
    )


def test_rewrite_reads_the_source_graphs_own_bindings() -> None:
    table = build_prefix_table(_catalog())

    assert table.rewrite_curies("needs evl:hasSpan", {"evl": EVT}) == "needs ev:hasSpan"


def test_normalize_rebinds_and_rewrites_notes() -> None:
    table = build_prefix_table(_catalog())
    graph = RDFGraph()
    graph.bind("eventlog", EVT)
    term = URIRef(f"{EVT}Incident")
    graph.add((term, SKOS.scopeNote, Literal("Pair with eventlog:hasSpan.")))

    table.normalize(graph)

    assert {p for p, ns in graph.namespaces() if str(ns) == EVT} == {"ev"}
    assert graph.value(term, SKOS.scopeNote) == Literal("Pair with ev:hasSpan.")


def test_an_empty_table_still_shows_standard_prefixes() -> None:
    table = PrefixTable({}, {})

    assert table.prefix_for(UNIT) == "unit"
    assert table.qname("http://www.w3.org/2006/time#Instant") == "time:Instant"


@pytest.mark.anyio
async def test_author_and_standard_prefixes_survive_the_store() -> None:
    from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager

    manager = InMemoryTripleStoreManager()
    for ontology in (
        stamped_ontology(EVENTLOG_TTL, ontology_id="eventlog"),
        stamped_ontology(UNITS_TTL, ontology_id="units"),
    ):
        await manager.aserialize(ontology)
    fetched = {o.ontology_id: o for o in await manager.afetch_ontologies()}

    eventlog, units = fetched["eventlog"], fetched["units"]
    assert eventlog.prefix == "ev"
    assert units.prefix == "unit"
    bound = {p: str(ns) for p, ns in units.graph.namespaces() if p}
    assert bound["qudt"] == "http://qudt.org/schema/qudt/"
    # No synthetic ``<id>_<slug>`` names for well-known namespaces.
    assert not [
        p for p, ns in bound.items() if p.startswith("units_") and "qudt.org" in ns
    ]
