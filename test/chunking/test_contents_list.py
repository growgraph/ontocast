"""Tables of contents and lists of figures, recognised from their content."""

from __future__ import annotations

from typing import Any, cast

import pytest

from ontocast.config import ChunkConfig
from ontocast.tool.chunk.contents import looks_like_contents_list
from ontocast.tool.chunk.routing import RouteReason, UnitRoute, route_unit
from test.chunking.test_section_routing import _chunk

pytestmark = pytest.mark.unit

_DOTTED = """## Table of Contents:

Executive Summary: ............................................ 3
Background: ................................................... 5
6.1 Inks ...................................................... 5
6.1.2 Tin Oxide ............................................... 8
Project Objectives: ........................................... 17
Project Results and Discussion: ............................... 27"""

_TABLE = """| Chapter 1: Introduction to thin-film devices | 1 |
|---|---|
| 1.1 Background | 3 |
| 1.2 Device architectures | 7 |
| 3.2.6 X-ray diffraction | 84 |
| References | 185 |"""

_FIGURES = """Figure 1: Device stack of the inverted cell. 12
Figure 2: Aging curves under damp heat. 15
Figure 3: Photoluminescence maps after 500 h. 21
Figure 4: Cross-section micrographs. 24"""

_ROMAN = """Abstract iii
Acknowledgements v
List of Publications vii
Contents ix"""

_DATA_TABLE = """| Sample | Thickness |
|---|---|
| Sample A | 300 |
| Sample B | 450 |
| Sample C | 600 |
| Sample D | 750 |"""

_STEPS = """1. Clean the substrate with acetone and isopropanol for 10 min.
2. Spin-coat the precursor at 4000 rpm for 30 s.
3. Anneal the film at 100 °C for 10 min.
4. Evaporate a gold electrode of 80 nm."""

_PROSE = (
    "The devices were aged for 1000 h at 85 °C. Their efficiency fell by 12\n"
    "percent over the first week and by a further 3 percent afterwards, which "
    "matches 5"
)


@pytest.mark.parametrize("text", [_DOTTED, _TABLE, _FIGURES, _ROMAN])
def test_contents_lists_are_recognised(text: str) -> None:
    assert looks_like_contents_list(text)


@pytest.mark.parametrize("text", [_DATA_TABLE, _STEPS, _PROSE])
def test_data_tables_steps_and_prose_are_not_contents(text: str) -> None:
    assert not looks_like_contents_list(text)


def test_page_numbers_must_ascend() -> None:
    shuffled = "\n".join(
        ["Results ........ 40", "Methods ........ 12", "Outlook ........ 3"]
        + ["Summary ........ 2"]
    )
    assert not looks_like_contents_list(shuffled)


@pytest.mark.parametrize(
    ("mode", "route"), [("extract", UnitRoute.NON_CONTENT), ("skip", UnitRoute.SKIP)]
)
def test_a_contents_list_routes_as_non_content(mode: str, route: UnitRoute) -> None:
    decision = route_unit(
        _chunk(_DOTTED),
        schema=None,
        schema_fallback=False,
        exclude_sections=None,
        config=ChunkConfig(non_content_mode=cast(Any, mode)),
    )
    assert decision.route is route
    assert decision.reason is RouteReason.CONTENTS_LIST
