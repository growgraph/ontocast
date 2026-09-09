"""The lexicon behind numeric coverage and the completion pass.

Every "missing measurement" finding, and every unit the completion pass is
asked to recover, starts as a :func:`unit_adjacent_numbers` mention. The module
shipped untested, so the two rules that decide whether a number is a
measurement -- compound-unit acceptance, and the figure-label/initial guards --
were unpinned in the release whose headline feature depends on them.
"""

import pytest

from ontocast.util.measurement_lexicon import (
    is_unit_surface,
    unit_adjacent_numbers,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "token",
    ["nm", "meV", "K", "ps", "°C", "%"],
)
def test_builtin_surfaces_are_recognized(token: str) -> None:
    assert is_unit_surface(token)


@pytest.mark.parametrize(
    "token",
    ["mW/cm2", "g·mol⁻¹", "μJ·cm-2", "J/mol/K"],
)
def test_a_compound_is_accepted_when_every_factor_is_known(token: str) -> None:
    """Compounds are decomposed rather than enumerated.

    Splitting on the factor separators and stripping an exponent is what lets
    the lexicon stay small while covering the units a materials corpus actually
    writes.
    """
    assert is_unit_surface(token)


@pytest.mark.parametrize("token", ["widgets/cm2", "mW/frobs", "", "  ", "the"])
def test_an_unknown_factor_rejects_the_whole_compound(token: str) -> None:
    assert not is_unit_surface(token)


def test_extra_surfaces_extend_the_lexicon_without_replacing_it() -> None:
    """A caller's ontology units must add to the built-ins, not shadow them."""
    assert not is_unit_surface("furlong")
    assert is_unit_surface("furlong", extra_surfaces={"furlong"})
    assert is_unit_surface("nm", extra_surfaces={"furlong"})


def test_a_trailing_separator_does_not_hide_a_unit() -> None:
    assert is_unit_surface("nm,")
    assert is_unit_surface("K.")


def test_a_plain_number_next_to_a_unit_is_a_mention() -> None:
    mentions = unit_adjacent_numbers("the shift was 96 meV overall")
    assert [(m.value, m.unit) for m in mentions] == [("96", "meV")]
    assert "96 meV" in mentions[0].context


def test_each_side_of_a_range_is_its_own_mention() -> None:
    """A range holds two values the graph is expected to carry, not one.

    Emitting one mention per number is what makes the coverage inventory ask
    for both endpoints rather than counting the range as satisfied by either.
    """
    mentions = unit_adjacent_numbers("aged 10-15 meV above the peak")
    assert [(m.value, m.unit) for m in mentions] == [("10", "meV"), ("15", "meV")]


def test_an_uncertainty_pair_shares_its_unit() -> None:
    mentions = unit_adjacent_numbers("thickness 8.5 ± 0.5 nm")
    assert [(m.value, m.unit) for m in mentions] == [("8.5", "nm"), ("0.5", "nm")]


@pytest.mark.parametrize(
    "text",
    ["see Figure 2 K for the trend", "Fig. 3 K shows the decay", "Table 2 K"],
)
def test_a_figure_or_table_label_is_not_a_measurement(text: str) -> None:
    """ "Figure 2 K" is a panel name, and K is not measuring anything.

    Without this guard every figure cross-reference in a paper becomes a
    missing-measurement finding, and the completion pass is sent to recover it.
    """
    assert unit_adjacent_numbers(text) == []


def test_a_number_with_no_unit_yields_nothing() -> None:
    assert unit_adjacent_numbers("we repeated it 12 times") == []


def test_mentions_come_back_in_text_order() -> None:
    mentions = unit_adjacent_numbers("first 5 nm, then 300 K, then 2 ps")
    assert [m.value for m in mentions] == ["5", "300", "2"]
    assert [m.start for m in mentions] == sorted(m.start for m in mentions)


def test_offsets_bracket_the_number_and_its_unit() -> None:
    text = "the shift was 96 meV overall"
    (mention,) = unit_adjacent_numbers(text)
    assert text[mention.start : mention.end] == "96 meV"
