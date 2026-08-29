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


def test_numeric_literals_in_graph_covers_typed_and_label_numbers() -> None:
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
    values = numeric_literals_in_graph(graph)
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
