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

LIFE = "https://ex.org/lifecycle#"
OBS = "https://ex.org/observation#"
MAT = "https://ex.org/matter#"
UNIT = "http://qudt.org/vocab/unit/"

LIFECYCLE_TTL = f"""
@prefix life: <{LIFE}> .
@prefix obs:  <{OBS}> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://ex.org/lifecycle> a owl:Ontology .
life:AgingProcess a owl:Class ; rdfs:subClassOf obs:Process .
life:hasDuration a owl:ObjectProperty ; rdfs:domain obs:Process .
"""
OBSERVATION_TTL = f"""
@prefix obs: <{OBS}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<https://ex.org/observation> a owl:Ontology .
obs:Process a owl:Class .
obs:hasProcessCondition a owl:ObjectProperty .
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
MATTER_TTL = f"""
@prefix lc:   <{LIFE}> .
@prefix mat:  <{MAT}> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://ex.org/matter> a owl:Ontology .
mat:Sample a owl:Class .
mat:agedBy a owl:ObjectProperty ; rdfs:range lc:AgingProcess .
"""


def _catalog():
    return [
        stamped_ontology(LIFECYCLE_TTL, ontology_id="lifecycle"),
        stamped_ontology(OBSERVATION_TTL, ontology_id="observation"),
        stamped_ontology(UNITS_TTL, ontology_id="units"),
        stamped_ontology(MATTER_TTL, ontology_id="matter"),
    ]


def test_adoption_shows_author_prefixes_and_standard_ones() -> None:
    lifecycle, observation, units, matter = _catalog()

    assert (lifecycle.ontology_id, lifecycle.prefix) == ("lifecycle", "life")
    assert (observation.ontology_id, observation.prefix) == ("observation", "obs")
    # Terms in a well-known namespace keep its standard prefix, not the id.
    assert units.namespace == UNIT
    assert (units.ontology_id, units.prefix) == ("units", "unit")
    assert matter.prefix == "mat"


def test_the_table_shows_one_prefix_per_namespace() -> None:
    table = build_prefix_table(_catalog())

    assert table.prefix_for(LIFE) == "life"
    assert table.prefix_for(OBS) == "obs"
    assert table.prefix_for(MAT) == "mat"
    assert table.prefix_for(UNIT) == "unit"
    assert table.prefix_for("http://www.w3.org/ns/sosa/") == "sosa"
    # The id and a borrowing file's prefix are aliases, never shown.
    assert table.aliases["lifecycle"] == LIFE
    assert table.aliases["lc"] == LIFE
    assert "life" not in table.aliases
    assert table.qname(f"{LIFE}AgingProcess") == "life:AgingProcess"
    assert table.qname(f"{UNIT}DEG_C") == "unit:DEG_C"


def test_the_parse_map_accepts_shown_prefixes_and_aliases() -> None:
    parse = build_prefix_table(_catalog()).parse_map()

    for prefix in ("life", "lifecycle", "lc"):
        assert parse[prefix] == LIFE
    assert parse["unit"] == UNIT
    assert parse["units"] == UNIT


def test_an_alias_two_namespaces_claim_is_dropped() -> None:
    other = stamped_ontology(
        """
        @prefix lc:  <https://ex.org/lightcurve#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://ex.org/lightcurve> a owl:Ontology .
        lc:Curve a owl:Class .
        """,
        ontology_id="lightcurve",
    )
    table = build_prefix_table([*_catalog(), other])

    # ``lc`` is the light-curve ontology's own prefix, so it is shown there and
    # is no longer an alias of the lifecycle namespace.
    assert table.prefix_for("https://ex.org/lightcurve#") == "lc"
    assert "lc" not in table.aliases


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
        ("life", LIFE, True),
        ("time", LIFE, False),
        ("time1", LIFE, False),
        ("ns1", LIFE, False),
        ("cd", LIFE, False),
        ("unit", UNIT, True),
        ("1x", LIFE, False),
    ],
)
def test_valid_shown_prefix(prefix: str, namespace: str, valid: bool) -> None:
    assert valid_shown_prefix(prefix, namespace) is valid


def test_rewrite_touches_only_curies_of_known_prefixes() -> None:
    table = build_prefix_table(_catalog())
    text = (
        "Use lifecycle:hasDuration, not lc:AgingProcess. See "
        "https://ex.org/lifecycle#AgingProcess at 10:30; cd:x and onto:Trial "
        "stay; life:hasDuration stays."
    )

    rewritten = table.rewrite_curies(text)

    assert rewritten == (
        "Use life:hasDuration, not life:AgingProcess. See "
        "https://ex.org/lifecycle#AgingProcess at 10:30; cd:x and onto:Trial "
        "stay; life:hasDuration stays."
    )


def test_rewrite_reads_the_source_graphs_own_bindings() -> None:
    table = build_prefix_table(_catalog())

    assert (
        table.rewrite_curies("needs lifec:hasDuration", {"lifec": LIFE})
        == "needs life:hasDuration"
    )


def test_normalize_rebinds_and_rewrites_notes() -> None:
    table = build_prefix_table(_catalog())
    graph = RDFGraph()
    graph.bind("lifecycle", LIFE)
    term = URIRef(f"{LIFE}AgingProcess")
    graph.add((term, SKOS.scopeNote, Literal("Pair with lifecycle:hasDuration.")))

    table.normalize(graph)

    assert {p for p, ns in graph.namespaces() if str(ns) == LIFE} == {"life"}
    assert graph.value(term, SKOS.scopeNote) == Literal("Pair with life:hasDuration.")


def test_an_empty_table_still_shows_standard_prefixes() -> None:
    table = PrefixTable({}, {})

    assert table.prefix_for(UNIT) == "unit"
    assert table.qname("http://www.w3.org/2006/time#Instant") == "time:Instant"


@pytest.mark.anyio
async def test_author_and_standard_prefixes_survive_the_store() -> None:
    from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager

    manager = InMemoryTripleStoreManager()
    for ontology in (
        stamped_ontology(LIFECYCLE_TTL, ontology_id="lifecycle"),
        stamped_ontology(UNITS_TTL, ontology_id="units"),
    ):
        await manager.aserialize(ontology)
    fetched = {o.ontology_id: o for o in await manager.afetch_ontologies()}

    lifecycle, units = fetched["lifecycle"], fetched["units"]
    assert lifecycle.prefix == "life"
    assert units.prefix == "unit"
    bound = {p: str(ns) for p, ns in units.graph.namespaces() if p}
    assert bound["qudt"] == "http://qudt.org/schema/qudt/"
    # No synthetic ``<id>_<slug>`` names for well-known namespaces.
    assert not [
        p for p, ns in bound.items() if p.startswith("units_") and "qudt.org" in ns
    ]
