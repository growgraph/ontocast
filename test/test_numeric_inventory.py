"""Numeric-mention inventory tests."""

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import XSD

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.util.numeric_inventory import (
    extract_numeric_tokens,
    missing_numeric_mentions,
    numeric_literals_in_graph,
)

pytestmark = pytest.mark.unit


def test_extract_numeric_tokens_basic_and_ranges() -> None:
    text = "a red shift of ~10-15 meV, up to 96 meV, edge length 8.5 ± 0.5 nm"
    tokens = extract_numeric_tokens(text)
    assert {"10", "15", "96", "8.5", "0.5"} <= tokens


def test_extract_numeric_tokens_drops_year_like_integers() -> None:
    tokens = extract_numeric_tokens("published in 2019, measured 2019.5 K at 77 K")
    assert "77" in tokens
    assert "2019.5" in tokens
    assert "2019" not in tokens


def test_extract_numeric_tokens_canonicalizes() -> None:
    assert extract_numeric_tokens("value 230.0 and 230") == {"230"}


def _typed_and_labelled() -> RDFGraph:
    graph = RDFGraph()
    subject = URIRef("http://x/s")
    graph.add(
        (
            subject,
            URIRef("http://qudt.org/schema/qudt/numericValue"),
            Literal("12.5", datatype=XSD.decimal),
        )
    )
    graph.add(
        (
            subject,
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
            Literal("range 50 - 300 nm"),
        )
    )
    return graph


def test_numeric_literals_in_graph_ignores_label_numbers_by_default() -> None:
    """A number that exists only inside a label has not been extracted.

    Counting it let a placeholder labelled with the missing number silence
    the coverage finding that asked for it.
    """
    values = numeric_literals_in_graph(_typed_and_labelled())
    assert "12.5" in values
    assert not {"50", "300"} & values


def test_numeric_literals_in_graph_can_include_annotations() -> None:
    values = numeric_literals_in_graph(_typed_and_labelled(), include_annotations=True)
    assert {"12.5", "50", "300"} <= values


def test_missing_numeric_mentions() -> None:
    graph = RDFGraph()
    graph.add(
        (
            URIRef("http://x/s"),
            URIRef("http://qudt.org/schema/qudt/numericValue"),
            Literal("96", datatype=XSD.decimal),
        )
    )
    missing = missing_numeric_mentions("shifts of 96 meV and 12.5 meV", graph)
    assert missing == ["12.5"]


# ---------------------------------------------------------------------------
# Identifier guard (FACTS_NUMERIC_IDENTIFIER_GUARD)
# ---------------------------------------------------------------------------


def test_identifier_fragments_kept_by_default() -> None:
    """The guard is opt-in; without it the current inventory is unchanged."""
    assert "92" in extract_numeric_tokens("case 7 AZR 600/92 was decided")


def test_identifier_fragments_dropped_under_guard() -> None:
    """A file number is one identifier, not three quantities missing from the graph."""
    guarded = extract_numeric_tokens(
        "case 7 AZR 600/92 was decided", ignore_identifier_fragments=True
    )
    assert "600" not in guarded
    assert "92" not in guarded


def test_guard_keeps_a_value_with_its_unit() -> None:
    """A magnitude followed by a unit is one digit group, not a code.

    ``5mg`` with no space matches nothing even without the guard -- the number
    pattern's trailing ``(?![\\w])`` already excludes it -- so the case that
    matters is the spaced form the pattern does read.
    """
    guarded = extract_numeric_tokens(
        "the sample held 5 mg", ignore_identifier_fragments=True
    )
    assert "5" in guarded


def test_guard_keeps_plain_decimals_and_exponents() -> None:
    guarded = extract_numeric_tokens(
        "values 3.14 and 1.4e3 and 250", ignore_identifier_fragments=True
    )
    assert "3.14" in guarded
    assert "250" in guarded
    assert any(token.startswith("1400") for token in guarded)


def test_guard_drops_doi_stems() -> None:
    guarded = extract_numeric_tokens(
        "recorded in 10.1234/example", ignore_identifier_fragments=True
    )
    assert "10.1234" not in guarded


def test_guard_keeps_hyphenated_ranges() -> None:
    """A hyphen joins two real values, so it is not an identifier separator.

    The module reads each side of a range separately by design; treating "-"
    as a code separator would drop both sides of "10-15 meV".
    """
    guarded = extract_numeric_tokens(
        "a red shift of 10-15 meV", ignore_identifier_fragments=True
    )
    assert {"10", "15"} <= guarded


def test_guard_does_not_reach_standalone_digit_groups() -> None:
    """Stated limitation: a lone token carries no evidence either way.

    "7" in a file number is indistinguishable from a small quantity without
    looking outside the token, and guessing there would cost real values.
    """
    guarded = extract_numeric_tokens(
        "case 7 AZR 600/92 was decided", ignore_identifier_fragments=True
    )
    assert "7" in guarded


def test_missing_numeric_mentions_threads_the_guard() -> None:
    graph = RDFGraph()
    unguarded = missing_numeric_mentions("file 600/92 refers", graph)
    guarded = missing_numeric_mentions(
        "file 600/92 refers", graph, ignore_identifier_fragments=True
    )
    assert "600" in unguarded
    assert guarded == []


# ---------------------------------------------------------------------------
# Unit-aware inventory: measurements vs bare numbers
# ---------------------------------------------------------------------------


def test_inventory_splits_measurements_from_bare_numbers() -> None:
    from ontocast.util.numeric_inventory import inventory_numeric_mentions

    inventory = inventory_numeric_mentions(
        "a red shift of 10-15 meV in sample 7, stored 30 days; see ref 42"
    )

    assert [(m.value, m.unit) for m in inventory.measurements] == [
        ("10", "meV"),
        ("15", "meV"),
        ("30", "days"),
    ]
    assert inventory.unclassified == ["7", "42"]
    assert "a red shift of 10-15 meV" in inventory.measurements[0].context


def test_inventory_keeps_one_mention_per_value_in_text_order() -> None:
    from ontocast.util.numeric_inventory import inventory_numeric_mentions

    inventory = inventory_numeric_mentions("96 meV here, and 96 meV again, 5 nm")

    assert [m.value for m in inventory.measurements] == ["96", "5"]


def test_snapshot_unit_surfaces_widen_the_measurement_lane() -> None:
    from ontocast.util.numeric_inventory import inventory_numeric_mentions

    without = inventory_numeric_mentions("irradiated at 1 sun and 3 suns for 2 cycles")
    with_units = inventory_numeric_mentions(
        "aged for 12 pulses at 2 cycles", unit_surfaces={"pulse"}
    )

    assert [m.unit for m in without.measurements] == ["sun", "suns", "cycles"]
    assert [(m.value, m.unit) for m in with_units.measurements] == [
        ("12", "pulses"),
        ("2", "cycles"),
    ]


def test_missing_inventory_lists_measurements_first_then_bare_numbers() -> None:
    from ontocast.util.numeric_inventory import missing_numeric_inventory

    graph = RDFGraph()
    graph.add(
        (
            URIRef("http://x/s"),
            URIRef("http://qudt.org/schema/qudt/numericValue"),
            Literal("96", datatype=XSD.decimal),
        )
    )
    inventory = missing_numeric_inventory(
        "shifts of 96 meV and 12.5 meV over 3 samples at 250 K", graph
    )

    assert [m.value for m in inventory.measurements] == ["12.5", "250"]
    assert inventory.unclassified == ["3"]
    assert missing_numeric_mentions(
        "shifts of 96 meV and 12.5 meV over 3 samples at 250 K", graph
    ) == ["12.5", "250", "3"]


def test_the_cap_drops_bare_numbers_before_measurements() -> None:
    from ontocast.util.numeric_inventory import missing_numeric_inventory

    inventory = missing_numeric_inventory(
        "1 and 2 and 3 and 4 then 5 nm and 6 nm", RDFGraph(), limit=3
    )

    assert [m.value for m in inventory.measurements] == ["5", "6"]
    assert inventory.unclassified == ["1"]


def test_unit_surface_index_reads_unit_individuals_and_is_memoised() -> None:
    from rdflib import OWL, RDF, RDFS, SKOS

    from ontocast.util.numeric_inventory import (
        unit_surface_index,
        unit_surfaces_in_ontology,
    )

    QUDT = "http://qudt.org/schema/qudt/"
    UNIT = "http://qudt.org/vocab/unit/"
    onto = RDFGraph()
    onto.add((URIRef(f"{QUDT}unit"), RDFS.range, URIRef(f"{QUDT}Unit")))
    onto.add((URIRef(f"{UNIT}SUN"), RDF.type, URIRef(f"{QUDT}Unit")))
    onto.add((URIRef(f"{UNIT}SUN"), RDFS.label, Literal("sun equivalent")))
    onto.add((URIRef(f"{UNIT}SUN"), URIRef(f"{QUDT}symbol"), Literal("sun")))
    onto.add((URIRef(f"{UNIT}CYCLE"), RDF.type, URIRef(f"{QUDT}Unit")))
    onto.add((URIRef(f"{UNIT}CYCLE"), SKOS.altLabel, Literal("cycles")))
    onto.add((URIRef(f"{UNIT}CYCLE"), SKOS.notation, Literal("12")))
    onto.add((URIRef("http://x/Thing"), RDF.type, OWL.Class))
    onto.add((URIRef("http://x/t1"), RDF.type, URIRef("http://x/Thing")))
    onto.add((URIRef("http://x/t1"), RDFS.label, Literal("nope")))

    index = unit_surface_index(onto, {f"{QUDT}unit"})

    assert index == {"sun": (f"{UNIT}SUN",), "cycles": (f"{UNIT}CYCLE",)}, (
        "a spaced label, a numeric notation and a non-unit individual stay out"
    )
    assert unit_surface_index(onto, {f"{QUDT}unit"}) is index, "memoised per graph"
    assert unit_surfaces_in_ontology(onto, {f"{QUDT}unit"}) == {"sun", "cycles"}
    assert unit_surface_index(None) == {}
