"""Lexical forms of the XSD date, time and duration datatypes.

rdflib reads these datatypes with Python's ISO parsers, which also accept forms
XSD does not (``2021-06-29`` or ``20210629`` as ``xsd:dateTime``, week dates,
a space for ``T``, a date-time as ``xsd:date``). A triple store treats such a
literal as ill-typed, so validity is decided here by the XSD 1.1 lexical
spaces instead, with the day checked against its month.
"""

from __future__ import annotations

import re

from rdflib import URIRef
from rdflib.namespace import XSD

_YEAR = r"(?P<year>-?(?:[1-9][0-9]{3,}|0[0-9]{3}))"
_MONTH = r"(?P<month>0[1-9]|1[0-2])"
_DAY = r"(?P<day>0[1-9]|[12][0-9]|3[01])"
_TIME = r"(?:(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?|24:00:00(?:\.0+)?)"
_ZONE = r"(?:Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))"

_DT_TIME = r"T(?:[0-9]+H(?:[0-9]+M)?(?:[0-9]+(?:\.[0-9]+)?S)?|[0-9]+M(?:[0-9]+(?:\.[0-9]+)?S)?|[0-9]+(?:\.[0-9]+)?S)"
_YM = r"(?:[0-9]+Y(?:[0-9]+M)?|[0-9]+M)"

_PATTERNS: dict[URIRef, re.Pattern[str]] = {
    XSD.dateTime: re.compile(rf"{_YEAR}-{_MONTH}-{_DAY}T{_TIME}{_ZONE}?"),
    XSD.dateTimeStamp: re.compile(rf"{_YEAR}-{_MONTH}-{_DAY}T{_TIME}{_ZONE}"),
    XSD.date: re.compile(rf"{_YEAR}-{_MONTH}-{_DAY}{_ZONE}?"),
    XSD.time: re.compile(rf"{_TIME}{_ZONE}?"),
    XSD.gYear: re.compile(rf"{_YEAR}{_ZONE}?"),
    XSD.gYearMonth: re.compile(rf"{_YEAR}-{_MONTH}{_ZONE}?"),
    XSD.gMonth: re.compile(rf"--{_MONTH}{_ZONE}?"),
    XSD.gDay: re.compile(rf"---{_DAY}{_ZONE}?"),
    XSD.gMonthDay: re.compile(rf"--{_MONTH}-{_DAY}{_ZONE}?"),
    XSD.duration: re.compile(
        rf"-?P(?:{_YM}(?:[0-9]+D)?(?:{_DT_TIME})?|[0-9]+D(?:{_DT_TIME})?|{_DT_TIME})"
    ),
    XSD.dayTimeDuration: re.compile(rf"-?P(?:[0-9]+D(?:{_DT_TIME})?|{_DT_TIME})"),
    XSD.yearMonthDuration: re.compile(rf"-?P{_YM}"),
}

#: The datatypes :func:`is_valid_temporal_lexical` decides.
TEMPORAL_DATATYPES: frozenset[URIRef] = frozenset(_PATTERNS)

#: The temporal datatypes whose value is a point or span on the calendar: a
#: node holding one says which date it is. Durations and recurring days are
#: not dates.
CALENDAR_DATATYPES: frozenset[URIRef] = frozenset(
    {XSD.date, XSD.dateTime, XSD.dateTimeStamp, XSD.gYearMonth, XSD.gYear}
)


def _days_in_month(year: int | None, month: int) -> int:
    if month == 2:
        if year is None:
            return 29
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31


def is_valid_temporal_lexical(lexical: str, datatype: URIRef) -> bool | None:
    """Whether ``lexical`` is in the XSD lexical space of ``datatype``.

    A day must exist in its month (``2021-02-30`` is not a date; ``--02-29``
    is a month-day).

    Returns:
        ``None`` when ``datatype`` is not a date, time or duration datatype.
    """
    pattern = _PATTERNS.get(datatype)
    if pattern is None:
        return None
    # These datatypes collapse whitespace before their lexical space applies.
    match = pattern.fullmatch(" ".join(lexical.split()))
    if match is None:
        return False
    groups = match.groupdict()
    if groups.get("day") is None or groups.get("month") is None:
        return True
    year = int(groups["year"]) if groups.get("year") is not None else None
    return int(groups["day"]) <= _days_in_month(year, int(groups["month"]))


__all__ = ["CALENDAR_DATATYPES", "TEMPORAL_DATATYPES", "is_valid_temporal_lexical"]
