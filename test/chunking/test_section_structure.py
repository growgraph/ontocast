"""Document structure: the heading tree, docling block kinds, one segmenter.

The tree decides membership where the document states it (numbering, varied
markdown levels); heading genericity decides only where it does not. Docling
items give each block of the markdown export its kind, so a caption is never
read as a cited work.
"""

from __future__ import annotations

import asyncio

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, GroupLabel

from ontocast.config import ChunkConfig
from ontocast.config.section_labels import load_section_label_schema
from ontocast.onto.enum import SectionLabelSource
from ontocast.tool.chunk.bibliography import citation_counts, is_citation_entry
from ontocast.tool.chunk.chunker import ChunkerTool
from ontocast.tool.chunk.outline import build_document_outline, heading_number
from ontocast.tool.chunk.prepare import PrepareOptions, prepare_content_units
from ontocast.tool.chunk.structure import (
    BlockKind,
    build_document_structure,
    split_blocks,
)
from test.docling_test_helpers import plain_doc

pytestmark = pytest.mark.unit

ACADEMIC = load_section_label_schema("academic")


def _labels(text: str) -> list[tuple[str, str | None]]:
    outline = build_document_outline(text, ACADEMIC)
    return [(node.normalised, node.label) for node in outline.nodes]


# --- heading numbers --------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "number"),
    [
        ("2.1 Synthesis of films", "2.1"),
        ("## 3 Results", "3"),
        ("1) Introduction", "1"),
        ("3.2.1. Film thickness", "3.2.1"),
        ("II. Methods", "II"),
        ("A.2 Derivation of the rate", "A.2"),
        ("Chapter 3: Aging of films", "3"),
        ("Results", None),
        ("2020 Annual report", None),
        ("A Framework for aging", None),
    ],
)
def test_heading_number_is_parsed_from_the_raw_heading(
    heading: str, number: str | None
) -> None:
    assert heading_number(heading) == number


# --- the tree from numbering ----------------------------------------------------------

_NUMBERED = (
    "# 1 Introduction\n\nBody.\n\n"
    "# 2 Results\n\nBody.\n\n"
    "# 2.1 Stability of films under humid air for weeks\n\nBody.\n\n"
    "# 2.2 Miscellany\n\nBody.\n\n"
    "# 3 Aging of encapsulated films stored in humid air\n\nBody.\n\n"
    "# Films degrade within days of exposure to humid air\n\nBody.\n"
)


def test_numbered_children_hang_under_their_parent() -> None:
    nodes = build_document_outline(_NUMBERED, ACADEMIC).nodes
    assert [node.number for node in nodes] == ["1", "2", "2.1", "2.2", "3", None]
    assert [node.parent for node in nodes[:5]] == [None, None, 1, 1, None]


def test_numbered_children_inherit_their_parents_label() -> None:
    labels = _labels(_NUMBERED)
    assert labels[2][1] == "results"
    # A section-like child the schema does not name is still inside Results.
    assert labels[3] == ("Miscellany", "results")
    outline = build_document_outline(_NUMBERED, ACADEMIC)
    assert outline.nodes[3].source is SectionLabelSource.HEADING_INHERITED


def test_a_numbered_top_level_heading_does_not_inherit() -> None:
    """Numbering says "3" is a sibling of Results, not a part of it."""
    outline = build_document_outline(_NUMBERED, ACADEMIC)
    sibling = outline.nodes[4]
    assert sibling.label is None
    assert sibling.source is SectionLabelSource.OUTLINE_UNRESOLVED


def test_an_unnumbered_descriptive_heading_follows_the_section_it_is_in() -> None:
    """After an unresolved top-level section, nothing is inherited."""
    assert _labels(_NUMBERED)[5][1] is None


def test_a_terminal_parent_has_no_children() -> None:
    labels = _labels(
        "# 4 Discussion\n\nBody.\n\n# 5 References\n\n1. A ref.\n\n"
        "# 5.1 Further reading on aging of perovskite films\n\nBody.\n"
    )
    assert labels[1][1] == "references"
    assert labels[2][1] is None


def test_a_child_names_its_own_section() -> None:
    labels = _labels(
        "# 2 Results\n\nBody.\n\n# 2.1 Discussion\n\nBody.\n\n"
        "# 2.2 Device stability under illumination for weeks\n\nBody.\n"
    )
    assert labels[1][1] == "discussion"
    assert labels[2][1] == "results"


def test_a_child_without_its_numbered_parent_falls_back_to_genericity() -> None:
    labels = _labels(
        "# Results\n\nBody.\n\n# 2.1 Stability of films under humid air\n\nBody.\n"
    )
    assert labels[1][1] == "results"


# --- the tree from markdown levels ---------------------------------------------------


def test_varied_markdown_levels_give_the_tree() -> None:
    nodes = build_document_outline(
        "# Halide perovskite films and their long-term stability in air\n\n"
        "## Results\n\nBody.\n\n"
        "### Stability of films under humid air\n\nBody.\n\n"
        "## Aging of encapsulated films stored in humid air\n\nBody.\n",
        ACADEMIC,
    ).nodes
    # A lone leading title is not a tree level: Results is not its child.
    assert [node.parent for node in nodes] == [None, None, 1, None]
    assert [node.label for node in nodes] == [None, "results", "results", None]


def test_uniform_levels_keep_the_genericity_rule() -> None:
    """Docling reports every PDF heading at one level: no tree from levels."""
    nodes = build_document_outline(
        "## Results\n\nBody.\n\n"
        "## Stability of films under humid air for weeks\n\nBody.\n",
        ACADEMIC,
    ).nodes
    assert [node.parent for node in nodes] == [None, None]
    assert nodes[1].label == "results"
    assert nodes[1].source is SectionLabelSource.HEADING_INHERITED


# --- block kinds from docling ------------------------------------------------------

_CAPTION = (
    "Figure 2. Reproduced from Smith, J. et al. Nano Lett. 2019, 19, 1234. "
    "Copyright 2019 American Chemical Society."
)
_REFERENCES = [
    "Smith, J.; Doe, A. Nano Lett. 2019, 19, 1234-1240.",
    "Roe, B.; Poe, C. Phys. Rev. B 2020, 101, 5521.",
    "Lee, D.; Kim, E. J. Am. Chem. Soc. 2018, 140, 3301.",
    "Wu, F.; Li, G. Adv. Mater. 2021, 33, 2007120.",
    "Chen, H.; Wang, I. ACS Energy Lett. 2017, 2, 889.",
]


def _paper() -> DoclingDocument:
    doc = DoclingDocument(name="paper")
    doc.add_heading("Results", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Films kept 92 % efficiency.")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text=_CAPTION)
    doc.add_picture(caption=caption)
    doc.add_text(label=DocItemLabel.FOOTNOTE, text="1 Key Laboratory, Shanghai.")
    doc.add_text(label=DocItemLabel.FORMULA, text="E = mc^2")
    doc.add_heading("References", level=1)
    group = doc.add_group(label=GroupLabel.LIST)
    for entry in _REFERENCES:
        doc.add_list_item(entry, parent=group)
    return doc


def test_docling_items_give_blocks_their_kind() -> None:
    structure = build_document_structure(_paper())
    kinds = {
        block.text.strip(): block.kind
        for block in split_blocks(structure.text, structure.block_kinds)
    }
    assert kinds[_CAPTION] is BlockKind.CAPTION
    assert kinds["1 Key Laboratory, Shanghai."] is BlockKind.FOOTNOTE
    assert kinds["$$E = mc^2$$"] is BlockKind.FORMULA
    assert kinds["<!-- image -->"] is BlockKind.PICTURE
    assert kinds[f"- {_REFERENCES[0]}"] is BlockKind.TEXT


def test_blocks_carry_the_item_ref() -> None:
    structure = build_document_structure(_paper())
    refs = [block.ref for block in split_blocks(structure.text, structure.block_kinds)]
    assert "#/texts/1" in refs  # the measurement paragraph
    assert all(ref is None or ref.startswith("#/") for ref in refs)


def test_a_caption_is_never_a_citation_entry() -> None:
    """A caption crediting a source reads like a reference; its kind says not."""
    assert is_citation_entry(_CAPTION)
    structure = build_document_structure(_paper())
    unit = f"{_CAPTION}\n\n<!-- image -->"
    assert citation_counts(unit) == (1, 1)
    assert citation_counts(unit, structure.block_kinds) == (0, 1)


def test_without_docling_kinds_a_placeholder_is_a_picture() -> None:
    blocks = split_blocks("Films kept 92 % efficiency.\n\n<!-- image -->")
    assert [block.kind for block in blocks] == [BlockKind.TEXT, BlockKind.PICTURE]


# --- preparation: one segmenter, item refs on units --------------------------------


def test_the_segmenter_option_is_gone() -> None:
    assert "segmenter" not in ChunkConfig.model_fields


def _prepare(doc: DoclingDocument):
    config = ChunkConfig(min_size=10, max_size=2000)
    chunker = ChunkerTool(chunk_config=config)
    return asyncio.run(
        prepare_content_units(
            doc, chunker, config, PrepareOptions(section_schema_id="academic")
        )
    )


def test_units_carry_the_refs_of_the_items_they_contain() -> None:
    document = _prepare(_paper())
    refs = [ref for unit in document.routed for ref in unit.doc_item_refs]
    assert "#/texts/1" in refs
    citations = [unit for unit in document.routed if unit.section_label == "references"]
    assert citations and all(unit.doc_item_refs for unit in citations)


def test_plain_text_input_still_prepares() -> None:
    document = _prepare(
        plain_doc(
            "Films were annealed at 100 °C for 10 min and stored in humid air. "
            "After 500 h they kept 92 % of their initial efficiency."
        )
    )
    assert len(document.units) == 1
