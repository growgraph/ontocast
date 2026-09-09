"""Unit-adjacent numbers in free text: the measurement side of the inventory.

A number standing next to a unit token ("96 meV", "8.5 ± 0.5 nm", "0.5 %",
"77 K") is a stated measurement; a bare number is not classifiable from the
text alone -- it may be a value whose unit sits elsewhere in the sentence, or
a citation, page, figure or equation token. The numeric-coverage lane and the
density-aware chunk split both need that distinction, and it has to be drawn
the same way in both places, so the lexicon and the scanner live here with no
dependency on either caller.

Two vocabularies feed the match: the built-in lexicon below (SI base and
derived units, the scale prefixes a scientific text uses, percent forms,
time words) and whatever unit surfaces the caller passes in -- typically the
labels and symbols of the unit individuals in the unit's ontology snapshot,
so a catalog-specific unit ("sun", "cycles") counts once the catalog
declares it. Compound tokens ("mW/cm2", "cm⁻¹", "g/mol") are matched
structurally: every factor, stripped of its exponent, must be a known
surface, so the lexicon does not have to enumerate products.

Latin-script and English-centric, like the query-side signal it mirrors
(``tool/vector_store/query_signals.number_adjacent_tokens``): the plural rule
strips a trailing "s" and the time words are English. Both only widen the
match, so a non-English corpus loses recall on this lane rather than
misclassifying.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

#: Characters of context kept on each side of a mention.
CONTEXT_CHARS = 40

#: Unit surfaces a measurement may be written with. Exact-case entries; the
#: matcher adds a case-insensitive lookup for tokens of three or more letters
#: and a singular form for plurals, so "Days"/"days"/"day" all reach "day"
#: while "m"/"M" and "mV"/"MV" stay distinct.
BUILTIN_UNIT_SURFACES: frozenset[str] = frozenset(
    {
        # SI base
        "m",
        "kg",
        "s",
        "A",
        "K",
        "mol",
        "cd",
        # SI derived and their common scales
        "Hz",
        "kHz",
        "MHz",
        "GHz",
        "THz",
        "N",
        "mN",
        "kN",
        "Pa",
        "hPa",
        "kPa",
        "MPa",
        "GPa",
        "bar",
        "mbar",
        "atm",
        "Torr",
        "mTorr",
        "psi",
        "J",
        "mJ",
        "µJ",
        "μJ",
        "uJ",
        "kJ",
        "MJ",
        "nJ",
        "pJ",
        "W",
        "mW",
        "µW",
        "μW",
        "uW",
        "kW",
        "MW",
        "nW",
        "C",
        "mC",
        "µC",
        "μC",
        "nC",
        "pC",
        "V",
        "mV",
        "µV",
        "μV",
        "kV",
        "MV",
        "F",
        "mF",
        "µF",
        "μF",
        "nF",
        "pF",
        "Ω",
        "ohm",
        "Ohm",
        "kΩ",
        "MΩ",
        "mΩ",
        "kOhm",
        "MOhm",
        "S",
        "mS",
        "µS",
        "μS",
        "T",
        "mT",
        "µT",
        "μT",
        "G",
        "kG",
        "Wb",
        "H",
        "mH",
        "µH",
        "μH",
        "nH",
        "lm",
        "lx",
        "Bq",
        "Gy",
        "Sv",
        "mSv",
        "µSv",
        "μSv",
        "kat",
        # energy
        "eV",
        "meV",
        "keV",
        "MeV",
        "GeV",
        "µeV",
        "μeV",
        "cal",
        "kcal",
        "Wh",
        "kWh",
        "mAh",
        "Ah",
        # length
        "Å",
        "pm",
        "nm",
        "µm",
        "μm",
        "um",
        "mm",
        "cm",
        "dm",
        "km",
        "inch",
        "ft",
        "mi",
        # time
        "fs",
        "ps",
        "ns",
        "µs",
        "μs",
        "us",
        "ms",
        "sec",
        "second",
        "min",
        "minute",
        "h",
        "hr",
        "hour",
        "d",
        "day",
        "wk",
        "week",
        "month",
        "yr",
        "year",
        # mass and amount
        "ng",
        "µg",
        "μg",
        "ug",
        "mg",
        "g",
        "t",
        "u",
        "Da",
        "kDa",
        "mmol",
        "µmol",
        "μmol",
        "umol",
        "nmol",
        "pmol",
        "M",
        "mM",
        "µM",
        "μM",
        "uM",
        "nM",
        "pM",
        # volume
        "L",
        "l",
        "mL",
        "ml",
        "µL",
        "μL",
        "uL",
        "nL",
        "pL",
        "dL",
        # temperature
        "°C",
        "°F",
        "°",
        "degC",
        "deg",
        # dimensionless forms
        "%",
        "‰",
        "wt%",
        "wt.%",
        "mol%",
        "at%",
        "at.%",
        "vol%",
        "v/v",
        "w/w",
        "w/v",
        "ppm",
        "ppb",
        "ppt",
        "rpm",
        "dB",
        "dBm",
        "px",
        "fold",
        # current
        "mA",
        "µA",
        "μA",
        "uA",
        "nA",
        "pA",
        "kA",
        # frequently used compound/other surfaces
        "cm-1",
        "cm⁻¹",
        "cm−1",
        "cm^-1",
        "sun",
        "suns",
        "cycle",
        "cycles",
        "rad",
        "sr",
        "mrad",
    }
)

_EXPONENT_TAIL = re.compile(r"(?:\^?[-−]?\d+|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)$")
_FACTOR_SPLIT = re.compile(r"[/·⋅*]")

#: One number, mirroring the inventory's pattern so both lanes read the same
#: digit groups: an integer or decimal with an optional exponent, not glued to
#: a word character on either side ("CsPbBr3" and "3D" carry no number).
_NUMBER = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER)

#: A run of numbers joined by range/uncertainty separators, then the unit
#: token they share. The token shape follows the query-side pattern: percent
#: and degree glyphs, or a letter-led token of up to twelve characters that
#: may carry scale letters, digits, factor separators and exponent glyphs.
_MENTION = re.compile(
    rf"(?<![\w.])(?P<numbers>{_NUMBER}"
    rf"(?:\s*(?:[-–—−~]|±|\+/-|to)\s*{_NUMBER})*)"
    r"(?:\s*[-‑]\s*|\s*)"
    r"(?P<token>[%‰]|°[A-Za-z]?|"
    r"[A-Za-zµμΩÅ][A-Za-z0-9µμΩÅ/·⋅*⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹%°^\-−.]{0,11})"
)


@dataclass(frozen=True)
class Mention:
    """One number written next to a unit.

    Attributes:
        value: The number as written (unsigned, verbatim lexical form).
        unit: The unit token as written.
        start: Offset of the number in the text.
        end: Offset just past the unit token.
        context: Whitespace-collapsed window of ``CONTEXT_CHARS`` on each side.
    """

    value: str
    unit: str
    start: int
    end: int
    context: str


def _lookup_keys(token: str) -> list[str]:
    """Exact form first, then the case-folded and singular forms it may stand for."""
    keys = [token]
    if len(token) >= 3 and token.isalpha():
        lowered = token.lower()
        keys.append(lowered)
        if len(lowered) > 3 and lowered.endswith("s"):
            keys.append(lowered[:-1])
    return keys


def _surface_matches(
    token: str, surfaces: Collection[str], folded: Collection[str]
) -> bool:
    for key in _lookup_keys(token):
        if key in surfaces or key in folded:
            return True
    return False


def _fold(surfaces: Collection[str]) -> frozenset[str]:
    return frozenset(s.lower() for s in surfaces if len(s) >= 3 and s.isalpha())


def is_unit_surface(token: str, extra_surfaces: Collection[str] = frozenset()) -> bool:
    """Whether ``token`` names a unit, built-in or supplied.

    A compound token is accepted when every factor -- split on ``/``, ``·``,
    ``⋅`` or ``*`` and stripped of a trailing exponent -- is itself a known
    surface, so "mW/cm2" and "g·mol⁻¹" pass without being listed.

    Args:
        token: Candidate unit token as it appears after a number.
        extra_surfaces: Additional surfaces, usually the labels/symbols of the
            unit individuals in the caller's ontology context.

    Returns:
        True when the token is a unit surface.
    """
    token = token.strip().rstrip(".,;:")
    if not token:
        return False
    surfaces: Collection[str] = (
        BUILTIN_UNIT_SURFACES
        if not extra_surfaces
        else BUILTIN_UNIT_SURFACES | frozenset(extra_surfaces)
    )
    folded = _fold(surfaces)
    if _surface_matches(token, surfaces, folded):
        return True
    factors = [factor for factor in _FACTOR_SPLIT.split(token) if factor]
    if not factors:
        return False
    for factor in factors:
        bare = _EXPONENT_TAIL.sub("", factor)
        if not bare or not _surface_matches(bare, surfaces, folded):
            return False
    return True


#: A number that labels a figure, table or equation rather than measuring
#: anything: "Figure 2A", "Fig. 3a", "Table 2 K". Read from the words just
#: before the number.
_LABEL_HEAD = re.compile(
    r"(?:fig(?:ure|s)?\.?|tables?|eq(?:uation)?s?\.?|schemes?|sections?|sec\.|"
    r"refs?\.?|chapters?|panels?|steps?)\s*$",
    re.IGNORECASE,
)
#: A single-letter unit followed by an initial: "Smith,1 A. B. Jones" is a
#: fused citation superscript and an author, not one ampere.
_INITIAL_TAIL = re.compile(r"\.\s+[A-Z]")


def _is_label_number(text: str, numbers_start: int) -> bool:
    head = text[max(0, numbers_start - 12) : numbers_start]
    return _LABEL_HEAD.search(head) is not None


def _is_initial(text: str, unit: str, end: int) -> bool:
    return (
        len(unit) == 1 and unit.isalpha() and _INITIAL_TAIL.match(text, end) is not None
    )


def _context(text: str, start: int, end: int) -> str:
    window = text[max(0, start - CONTEXT_CHARS) : min(len(text), end + CONTEXT_CHARS)]
    return " ".join(window.split())


def _token_candidates(token: str) -> list[str]:
    """The token as written, then progressively less of a hyphenated tail.

    "10 nm-thick" reads as the unit "nm" followed by prose; the tail is only
    kept when it is an exponent ("cm-2"), which the surface matcher handles.
    """
    stripped = token.rstrip(".,;:")
    candidates = [stripped]
    head = re.split(r"[-−](?!\d)", stripped, maxsplit=1)[0]
    if head and head != stripped:
        candidates.append(head)
    return candidates


def unit_adjacent_numbers(
    text: str, extra_surfaces: Collection[str] = frozenset()
) -> list[Mention]:
    """Scan ``text`` for numbers written next to a unit, in text order.

    A range or uncertainty pair ("10-15 meV", "8.5 ± 0.5 nm") yields one
    mention per number, all carrying the shared unit, because each side is a
    value the graph is expected to hold.

    Args:
        text: Source text of a unit or window.
        extra_surfaces: Unit surfaces beyond the built-in lexicon.

    Returns:
        Mentions in order of appearance.
    """
    mentions: list[Mention] = []
    for match in _MENTION.finditer(text):
        token = match.group("token")
        unit = next(
            (c for c in _token_candidates(token) if is_unit_surface(c, extra_surfaces)),
            None,
        )
        if unit is None:
            continue
        numbers_start = match.start("numbers")
        end = match.start("token") + len(unit)
        if _is_label_number(text, numbers_start) or _is_initial(text, unit, end):
            continue
        for number in _NUMBER_RE.finditer(match.group("numbers")):
            start = numbers_start + number.start()
            mentions.append(
                Mention(
                    value=number.group(0),
                    unit=unit,
                    start=start,
                    end=end,
                    context=_context(text, start, end),
                )
            )
    return mentions
