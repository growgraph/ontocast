"""Citation entries across reference styles, and the prose that must not read as one."""

from __future__ import annotations

import pytest

from ontocast.tool.chunk.bibliography import (
    citation_share,
    is_citation_entry,
    looks_like_bibliography,
    trim_citation_runs,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "entry",
    [
        # Full names before the year, proceedings venue with a page range.
        "Ana Lopez, Ben Okafor, and Chen Wei. 2018. Contextual embeddings for "
        "sequence tagging. In Proceedings of the 27th International Conference "
        "on Computational Linguistics, pages 1638-1649.",
        # Full names before the year, an acronym as the venue.
        "Dana Park and Eli Stone. 2015. A large annotated corpus for inference. "
        "In EMNLP. Association for Computational Linguistics.",
        # A bare number as the marker; volume with an issue.
        '17 Y. Hamada, "Recent progress of thin-film cells," Sol. Energy Mater. '
        "17(2), 87-96 (2018).",
        # A bracketed label as the marker.
        '[RFC9999] Doe, J. and R. Roe, "Problem Details for HTTP APIs", RFC 9999, '
        "DOI 10.17487/RFC9999, March 2016, <https://www.rfc-editor.org/info/rfc9999>.",
        # Year;volume(issue):pages.
        "3. Roe A, Poe B. Responsibilities of the investigator. Ann Med. "
        "1994;26(2):89-94.",
        # A web source.
        "3. National Laboratory. Best research-cell efficiency chart. Available "
        "online: https://example.org/chart (accessed on 30 June 2023).",
        # A percent sign inside a link target is not a measurement.
        "- [4] [A. Kim, B. Lee, Light harvesters in photovoltaics, Angew. Chem. 53 "
        "(2014) 2812-2824.](http://refs.example.org/S2405-8440%2821%2900316-9/bib4)",
    ],
)
def test_reference_styles_are_entries(entry: str) -> None:
    assert is_citation_entry(entry)


@pytest.mark.parametrize(
    "text",
    [
        "Unlike Radford et al. (2018), which uses unidirectional language models "
        "for pre-training, we use masked language models; in prior work, only "
        "sentence embeddings were transferred to downstream tasks.",
        "The benchmark (Wang et al., 2018a) is a collection of diverse natural "
        "language understanding tasks. In GLUE, pages of each task are paired "
        "with a held-out test set.",
        # A title that states a measurement keeps the entry out, as in prose.
        "56. Bush, K.A.; Yu, Z.J. 23.6%-efficient monolithic perovskite/silicon "
        "tandem solar cells. Nat. Energy 2017, 2, 17009.",
    ],
)
def test_prose_and_measured_titles_are_not_entries(text: str) -> None:
    assert not is_citation_entry(text)


_SPLIT_LIST = "\n\n".join(
    [
        '4 C. Weber, P. Green, and R. Lammers, "Global water resources:',
        'vulnerability from climate change," Science 289(5477), 284-288 (2000).',
        '5 D. Ito and K. Mori, "Optimal generation mix for island grids,"',
        "Energy Rep. 8(1), 101-112 (2022).",
        '6 F. Abe, "Coal combustion wastes and health,"',
        "Waste Manag. 17(2), 87-96 (2018).",
    ]
)


def test_an_entry_split_across_blocks_is_one_entry() -> None:
    """A page break or a wrapped line leaves the year or venue in the next block."""
    assert citation_share(_SPLIT_LIST) == 1.0
    assert looks_like_bibliography(_SPLIT_LIST)


def test_a_split_run_is_trimmed_whole() -> None:
    prose = (
        "In conclusion, the films kept most of their efficiency after aging, and "
        "the encapsulant slowed the formation of lead iodide at grain boundaries."
    )
    more = "\n\n".join(
        [
            '7 G. Ota, "Thin-film modules in the field,"',
            "Prog. Photovolt. 30(4), 300-310 (2021).",
            '8 H. Sato, "Encapsulation of flexible devices,"',
            "Adv. Mater. 33(9), 2004-2015 (2020).",
        ]
    )
    pieces = trim_citation_runs(prose + "\n\n" + _SPLIT_LIST + "\n\n" + more)
    assert [is_run for _, is_run in pieces] == [False, True]
    assert pieces[0][0] == prose


def test_prose_after_an_entry_opener_is_not_joined_into_an_entry() -> None:
    text = "\n\n".join(
        [
            "Smith, J. and colleagues examined the films.",
            "The films were stored in the dark and measured every week, and the "
            "procedure followed the consensus protocol published in 2019.",
        ]
    )
    assert citation_share(text) == 0.0
