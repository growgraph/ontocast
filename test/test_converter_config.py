"""Tests for Docling converter configuration and temporary text repair."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

import pytest

from ontocast.config import Config, ConverterConfig, PathConfig, ToolConfig
from ontocast.onto.docling_helpers import (
    apply_text_sanitizers,
    plain_text_to_docling_doc,
    rejoin_flattened_exponents,
    repair_ligature_gaps,
    repair_numeric_artifacts,
    repair_single_sided_ligature_gaps,
)
from ontocast.tool.cache import Cacher
from ontocast.tool.converter import (
    CONVERTER_CACHE_FORMAT_VERSION,
    CONVERTER_REPAIR_RULES_VERSION,
    ConverterTool,
    build_document_converter,
)
from ontocast.tool.pdf_regime import pdf_has_text_layer
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def _pdf(pages: list[str], render_mode: int = 0) -> bytes:
    """A minimal PDF, one page per string; an empty string is a page with no text."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        content = (
            f"BT {render_mode} Tr /F1 10 Tf 72 720 Td ({text}) Tj ET".encode()
            if text
            else b""
        )
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
            f"<< /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objs.append(
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"
        )
    out, offsets = b"%PDF-1.4\n", []
    for k, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % k + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref,
    )
    return out


SENTENCE = "The quick brown fox jumps over the lazy dog near the riverbank."
TEXT_PDF = _pdf([SENTENCE, SENTENCE, SENTENCE])
IMAGE_ONLY_PDF = _pdf(["", "", ""])
HIDDEN_TEXT_PDF = _pdf([SENTENCE, SENTENCE], render_mode=3)


def test_default_profile_is_auto() -> None:
    assert ConverterConfig().profile == "auto"


@pytest.mark.parametrize(
    ("removed", "replacement"), [("born_digital", "fast"), ("default", "ocr")]
)
def test_removed_profiles_name_their_replacement(
    removed: str, replacement: str
) -> None:
    with pytest.raises(ValueError, match=f"'{replacement}'"):
        ConverterConfig(profile=removed)


def test_profile_presets() -> None:
    fast = ConverterConfig().resolved("fast")
    assert (fast.profile, fast.do_ocr, fast.table_mode, fast.do_formula_enrichment) == (
        "fast",
        False,
        "fast",
        False,
    )
    lean = ConverterConfig().resolved("lean")
    assert (lean.profile, lean.do_ocr, lean.table_mode, lean.do_formula_enrichment) == (
        "lean",
        False,
        "fast",
        True,
    )
    ocr = ConverterConfig().resolved("ocr")
    assert (ocr.profile, ocr.do_ocr, ocr.table_mode, ocr.do_formula_enrichment) == (
        "ocr",
        True,
        "accurate",
        False,
    )


def test_presets_leave_explicit_fields_alone(monkeypatch) -> None:
    assert (
        ConverterConfig(profile="lean", table_mode="accurate")
        .resolved("lean")
        .table_mode
        == "accurate"
    )
    monkeypatch.setenv("CONVERTER_DO_OCR", "true")
    assert ConverterConfig(profile="lean").resolved("lean").do_ocr is True


def test_fixed_profile_resolves_only_to_itself() -> None:
    with pytest.raises(ValueError, match="cannot resolve"):
        ConverterConfig(profile="ocr").resolved("lean")


@pytest.mark.parametrize(("profile", "formula"), [("fast", False), ("lean", True)])
def test_build_document_converter_applies_the_text_layer_presets(
    profile: Literal["fast", "lean"], formula: bool
) -> None:
    pytest.importorskip("docling")
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import TableFormerMode

    converter = build_document_converter(ConverterConfig().resolved(profile))

    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert options.do_ocr is False
    assert options.do_formula_enrichment is formula
    assert options.table_structure_options.mode == TableFormerMode.FAST


def test_pdf_text_layer_detection() -> None:
    assert pdf_has_text_layer(TEXT_PDF) is True
    assert pdf_has_text_layer(IMAGE_ONLY_PDF) is False
    # A scan's OCR layer is invisible text; it is read rather than re-recognised.
    assert pdf_has_text_layer(HIDDEN_TEXT_PDF) is True
    assert pdf_has_text_layer(b"%PDF-not-really") is False


def test_auto_routes_each_pdf_by_its_text_layer(monkeypatch, tmp_path: Path) -> None:
    builds: list[ConverterConfig] = []
    monkeypatch.setattr(
        "ontocast.tool.converter.build_document_converter", _fake_build(builds)
    )
    tool = ConverterTool(cache=Cacher(cache_dir=tmp_path))

    tool(TEXT_PDF, filename="paper.pdf")
    tool(IMAGE_ONLY_PDF)  # no name: recognised as a PDF from its bytes
    tool(b"# Title\n", filename="note.md")

    assert [c.profile for c in builds] == ["fast", "ocr"]
    assert set(tool._converters) == {"fast", "ocr"}
    assert tool.cache_config("a.pdf", "fast") != tool.cache_config("a.pdf", "ocr")


def test_fixed_profile_skips_detection(monkeypatch, tmp_path: Path) -> None:
    def fail(_data: bytes) -> bool:
        raise AssertionError("a fixed profile must not inspect the PDF")

    monkeypatch.setattr("ontocast.tool.converter.pdf_has_text_layer", fail)
    tool = ConverterTool(
        cache=Cacher(cache_dir=tmp_path),
        converter_config=ConverterConfig(profile="ocr"),
    )
    assert tool.resolve_profile(TEXT_PDF, "paper.pdf") == "ocr"


def test_build_document_converter_uses_configured_pdf_backend_and_options() -> None:
    from docling.datamodel.base_models import InputFormat

    config = ConverterConfig(
        pdf_backend="pypdfium2",
        do_ocr=False,
        force_backend_text=True,
        do_table_structure=False,
        table_cell_matching=False,
    )

    converter = build_document_converter(config)

    pdf_option = converter.format_to_options[InputFormat.PDF]
    assert pdf_option.backend.__name__ == "PyPdfiumDocumentBackend"
    assert pdf_option.pipeline_options.do_ocr is False
    assert pdf_option.pipeline_options.force_backend_text is True
    assert pdf_option.pipeline_options.do_table_structure is False
    assert pdf_option.pipeline_options.table_structure_options.do_cell_matching is False
    assert converter.format_to_options[InputFormat.PPTX].__class__.__name__ == (
        "PowerpointFormatOption"
    )


def test_converter_cache_key_includes_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        shared_cache = Cacher(cache_dir=tmp)
        content = b"%PDF-test-bytes"
        doc_json = plain_text_to_docling_doc("cached", "doc").model_dump_json()

        ocr_tool = ConverterTool(
            cache=shared_cache,
            converter_config=ConverterConfig(profile="ocr"),
        )
        fast_tool = ConverterTool(
            cache=shared_cache,
            converter_config=ConverterConfig(profile="fast"),
        )

        ocr_tool.cache.set(content, doc_json, config=ocr_tool.cache_config("a.pdf"))

        assert ocr_tool.cache.get(content, config=ocr_tool.cache_config("a.pdf"))
        assert (
            fast_tool.cache.get(content, config=fast_tool.cache_config("a.pdf")) is None
        )


def test_temp_repair_ligature_gaps_repairs_user_example_without_merging_nc_sls() -> (
    None
):
    text = (
        "we introduce a solvent di ff usion technique and nearly con fi ned "
        "CsPbBr3 on per fl uorodecalin with NC SLs."
    )

    repaired = repair_ligature_gaps(text)

    assert "diffusion" in repaired
    assert "confined" in repaired
    assert "perfluorodecalin" in repaired
    assert "NC SLs" in repaired


def test_apply_text_sanitizers_repairs_docling_document_texts() -> None:
    doc = plain_text_to_docling_doc("di ff usion and con fi ned", "doc")

    sanitized = apply_text_sanitizers(doc, repair_ligature_gaps_enabled=True)

    assert sanitized.export_to_markdown().strip() == "diffusion and confined"


def test_toolbox_wires_converter_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        od = wd / "ontologies"
        od.mkdir()
        tool_config = ToolConfig(
            path_config=PathConfig(ontology_directory=od),
            converter_config=ConverterConfig(profile="ocr", repair_ligature_gaps=True),
        )

        toolbox = ToolBox(Config(tool_config=tool_config))

        assert toolbox.converter.converter_config.profile == "ocr"
        assert toolbox.converter.converter_config.repair_ligature_gaps is True
        # Docling converter is deferred until first conversion
        assert toolbox.converter._converter is None


def test_converter_tool_builds_document_converter_once(monkeypatch) -> None:
    builds: list[ConverterConfig] = []

    def fake_build(config: ConverterConfig):
        builds.append(config)

        class _Result:
            document = plain_text_to_docling_doc("ok", "doc")

        class _Converter:
            def convert(self, _src):
                return _Result()

        return _Converter()

    monkeypatch.setattr("ontocast.tool.converter.build_document_converter", fake_build)
    with tempfile.TemporaryDirectory() as tmp:
        tool = ConverterTool(
            cache=Cacher(cache_dir=tmp),
            converter_config=ConverterConfig(do_ocr=False),
        )
        assert tool._converter is None
        doc1 = tool(b"%PDF-unique-lazy-1%")
        doc2 = tool(b"%PDF-unique-lazy-2%")
        assert doc1 is not None and doc2 is not None
        assert len(builds) == 1
        assert len(tool._converters) == 1


# --- CONVERTER_REPAIR_NUMERIC_ARTIFACTS --------------------------------------


def test_repair_numeric_artifacts_default_is_off() -> None:
    assert ConverterConfig().repair_numeric_artifacts is False
    assert ConverterConfig(profile="fast").repair_numeric_artifacts is False


def test_unescapes_only_named_entities_with_semicolons() -> None:
    assert repair_numeric_artifacts("T &lt; 300 K &amp; p &gt; 1 bar") == (
        "T < 300 K & p > 1 bar"
    )
    assert repair_numeric_artifacts("&quot;x&quot; &apos;y&apos;") == "\"x\" 'y'"
    # No html.unescape: a semicolon-less entity in running text stays text.
    assert repair_numeric_artifacts("see &para 3 and R&D") == "see &para 3 and R&D"


def test_carriage_return_wraps_collapse_but_paragraph_breaks_survive() -> None:
    assert repair_numeric_artifacts("shift of\r  \n20 meV") == "shift of\n20 meV"
    assert repair_numeric_artifacts("one.\r \n\ntwo") == "one.\n\ntwo"


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        ("2 × 10 6 cm-3", "2 × 10^6 cm-3"),
        ("1.5 x 10 19 cm-3", "1.5 × 10^19 cm-3"),
        ("3 × 10 −3 S/cm", "3 × 10^-3 S/cm"),
        ("~10 6 cycles", "~10^6 cycles"),
        ("≈ 10 5", "≈ 10^5"),
        ("on the order of 10 4", "on the order of 10^4"),
    ],
)
def test_flattened_exponents_are_rejoined(raw: str, repaired: str) -> None:
    assert rejoin_flattened_exponents(raw) == repaired


@pytest.mark.parametrize(
    "text",
    [
        "10 6-membered rings",
        "on the order of 10 6-membered rings",
        "5 × 10 6-membered rings",
        "10 6 samples",
        "2 × 10 100",
        "2 × 10^6 cm-3",
        "10 cm × 10 cm",
        # A dash between two numbers behind a bare cue is a range, not a
        # negative exponent -- publishers typeset ranges with U+2212 too.
        "∼10−15 meV",
        "∼ 10−15 meV",
        "~10–50 meV",
        "on the order of 10-15 meV",
    ],
)
def test_exponent_rejoin_leaves_other_number_pairs_alone(text: str) -> None:
    """A bare '10 6' without a cue, or one starting a hyphenated word, is text."""
    assert rejoin_flattened_exponents(text) == text


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        ("a ffected", "affected"),
        ("di fferent", "different"),
        ("e fficient", "efficient"),
        ("switched o ff.", "switched off."),
        ("signifi cant", "significant"),
        ("confi ned", "confined"),
        ("refl ected", "reflected"),
        ("suffi cient", "sufficient"),
    ],
)
def test_single_sided_ligature_gaps_with_one_reading_are_closed(
    raw: str, repaired: str
) -> None:
    assert repair_single_sided_ligature_gaps(raw) == repaired


@pytest.mark.parametrize(
    "text",
    [
        "the field",
        "a flat film",
        "of it",
        "we find",
        "fl oz",
        "cliff face",
        "if fed",
        "of fish",
        "NC SLs",
    ],
)
def test_single_sided_ligature_rule_leaves_two_word_phrases_alone(text: str) -> None:
    assert repair_single_sided_ligature_gaps(text) == text


def test_apply_text_sanitizers_numeric_flag_repairs_docling_document_texts() -> None:
    doc = plain_text_to_docling_doc("T &lt; 300 K at 2 × 10 6 cm-3, a ffected", "doc")

    sanitized = apply_text_sanitizers(doc, repair_numeric_artifacts_enabled=True)

    # The markdown export re-escapes "<" and "&"; the item text is what the
    # chunker and the prompts see.
    assert sanitized.texts[0].text == "T < 300 K at 2 × 10^6 cm-3, affected"


def test_apply_text_sanitizers_flags_compose() -> None:
    doc = plain_text_to_docling_doc("the e ff ect was a ffected", "doc")

    sanitized = apply_text_sanitizers(
        doc, repair_ligature_gaps_enabled=True, repair_numeric_artifacts_enabled=True
    )

    assert sanitized.export_to_markdown().strip() == "the effect was affected"


def test_apply_text_sanitizers_is_identity_when_both_flags_are_off() -> None:
    doc = plain_text_to_docling_doc("a ffected &lt;", "doc")
    assert apply_text_sanitizers(doc).texts[0].text == "a ffected &lt;"


def _fake_build(builds: list[ConverterConfig]):
    def fake_build(config: ConverterConfig):
        builds.append(config)

        class _Result:
            document = plain_text_to_docling_doc("ok", "doc")

        class _Converter:
            def convert(self, _src):
                return _Result()

        return _Converter()

    return fake_build


def test_numeric_repair_flag_joins_the_converter_cache_key(monkeypatch) -> None:
    builds: list[ConverterConfig] = []
    monkeypatch.setattr(
        "ontocast.tool.converter.build_document_converter", _fake_build(builds)
    )
    content = b"%PDF-numeric-key%"
    with tempfile.TemporaryDirectory() as tmp:
        shared_cache = Cacher(cache_dir=tmp)

        ConverterTool(cache=shared_cache, converter_config=ConverterConfig())(content)
        assert len(builds) == 1

        # Enabling the flag changes the text, so it must miss the cached entry.
        ConverterTool(
            cache=shared_cache,
            converter_config=ConverterConfig(repair_numeric_artifacts=True),
        )(content)
        assert len(builds) == 2

        # A fresh flag-off tool hits the first entry again.
        ConverterTool(cache=shared_cache, converter_config=ConverterConfig())(content)
        assert len(builds) == 2


def test_repair_rules_version_joins_the_key_only_when_repair_is_on(
    monkeypatch,
) -> None:
    """A rule fix must invalidate repaired entries and leave the rest alone."""
    builds: list[ConverterConfig] = []
    monkeypatch.setattr(
        "ontocast.tool.converter.build_document_converter", _fake_build(builds)
    )
    content = b"%PDF-rules-version%"
    with tempfile.TemporaryDirectory() as tmp:
        shared_cache = Cacher(cache_dir=tmp)
        repairing = ConverterConfig(repair_numeric_artifacts=True)

        ConverterTool(cache=shared_cache, converter_config=repairing)(content)
        assert len(builds) == 1
        # Same rules version: the repaired conversion is still served.
        ConverterTool(cache=shared_cache, converter_config=repairing)(content)
        assert len(builds) == 1

        # Bumping the rules version must re-convert rather than serve text
        # repaired by the superseded rule.
        monkeypatch.setattr(
            "ontocast.tool.converter.CONVERTER_REPAIR_RULES_VERSION",
            CONVERTER_REPAIR_RULES_VERSION + 1,
        )
        ConverterTool(cache=shared_cache, converter_config=repairing)(content)
        assert len(builds) == 2

        # With repair off no rule ran, so the version must stay out of the key.
        off = ConverterConfig()
        ConverterTool(cache=shared_cache, converter_config=off)(content)
        assert len(builds) == 3
        monkeypatch.setattr(
            "ontocast.tool.converter.CONVERTER_REPAIR_RULES_VERSION",
            CONVERTER_REPAIR_RULES_VERSION + 2,
        )
        ConverterTool(cache=shared_cache, converter_config=off)(content)
        assert len(builds) == 3


def test_flag_off_keeps_pre_flag_cache_entries_valid(monkeypatch) -> None:
    """The flag joins the key only when on, so old conversions stay cached."""
    builds: list[ConverterConfig] = []
    monkeypatch.setattr(
        "ontocast.tool.converter.build_document_converter", _fake_build(builds)
    )
    content = b"%PDF-legacy-key%"
    with tempfile.TemporaryDirectory() as tmp:
        shared_cache = Cacher(cache_dir=tmp)
        tool = ConverterTool(cache=shared_cache, converter_config=ConverterConfig())

        # These bytes are no readable PDF, so ``auto`` resolves them to ``ocr``.
        legacy_key = tool.converter_config.resolved("ocr").model_dump(mode="json")
        legacy_key.pop("repair_numeric_artifacts")
        # Written before the field existed; it never joins the key.
        legacy_key.pop("supported_extensions")
        legacy_key["cache_format_version"] = CONVERTER_CACHE_FORMAT_VERSION
        tool.cache.set(
            content,
            plain_text_to_docling_doc("cached", "doc").model_dump_json(),
            config=legacy_key,
        )

        doc = tool(content)

        assert builds == []
        assert doc.export_to_markdown().strip() == "cached"


# -- supported formats -------------------------------------------------------

#: The shipped default. Changing it changes what ``/process`` accepts and what
#: ``/info`` advertises, so it is pinned here deliberately.
SHIPPED_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".csv",
    ".adoc",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
]


def test_default_supported_extensions_are_pinned() -> None:
    assert ConverterConfig().supported_extensions == SHIPPED_EXTENSIONS


def test_supported_extensions_are_normalised() -> None:
    config = ConverterConfig(supported_extensions=["PDF", " .Docx "])
    assert config.supported_extensions == [".pdf", ".docx"]


@pytest.mark.parametrize("suffix", [".txt", ".json", ".jsonl"])
def test_suffixes_read_without_docling_are_refused(suffix: str) -> None:
    with pytest.raises(ValueError, match=suffix):
        ConverterConfig(supported_extensions=[".pdf", suffix])


def test_supported_extensions_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CONVERTER_SUPPORTED_EXTENSIONS", '[".pdf", ".md"]')
    assert ConverterConfig().supported_extensions == [".pdf", ".md"]


def test_tool_accepts_the_configured_set_when_docling_is_installed(
    monkeypatch, tmp_path: Path
) -> None:
    from ontocast.tool import converter as module

    monkeypatch.setattr(module, "is_available", lambda _name: True)
    tool = ConverterTool(
        cache=Cacher(cache_dir=tmp_path),
        converter_config=ConverterConfig(supported_extensions=[".pdf", ".md"]),
    )
    assert tool.supported_extensions == {".pdf", ".md"}


def test_tool_accepts_nothing_without_docling(monkeypatch, tmp_path: Path) -> None:
    from ontocast.tool import converter as module

    monkeypatch.setattr(module, "is_available", lambda _name: False)
    tool = ConverterTool(cache=Cacher(cache_dir=tmp_path))
    assert tool.supported_extensions == set()


def test_cache_key_ignores_the_extension_set(tmp_path: Path) -> None:
    """Widening or narrowing the accepted set must not re-convert anything."""
    narrow = ConverterTool(
        cache=Cacher(cache_dir=tmp_path),
        converter_config=ConverterConfig(supported_extensions=[".pdf"]),
    )
    wide = ConverterTool(cache=Cacher(cache_dir=tmp_path))
    assert narrow.cache_config("paper.pdf") == wide.cache_config("paper.pdf")
    assert "supported_extensions" not in wide.cache_config("paper.pdf")


def test_cache_key_names_the_format_only_beyond_pdf_and_pptx(tmp_path: Path) -> None:
    """PDF/PPTX keys stay as they were; other formats are keyed by suffix,
    since the same bytes parse differently as Markdown and as AsciiDoc."""
    tool = ConverterTool(cache=Cacher(cache_dir=tmp_path))
    assert tool.cache_config("a.pdf") == tool.cache_config(None)
    assert tool.cache_config("a.pptx") == tool.cache_config(None)
    assert tool.cache_config("a.md") != tool.cache_config("a.adoc")


def test_document_converter_allows_only_the_configured_formats() -> None:
    pytest.importorskip("docling")
    from docling.datamodel.base_models import InputFormat

    converter = build_document_converter(
        ConverterConfig(supported_extensions=[".pdf", ".md"])
    )
    assert set(converter.allowed_formats) == {InputFormat.PDF, InputFormat.MD}


def test_markdown_bytes_convert_when_named(tmp_path: Path) -> None:
    """Docling cannot sniff Markdown from bytes; the upload's name must reach it."""
    pytest.importorskip("docling")
    tool = ConverterTool(cache=Cacher(cache_dir=tmp_path))
    doc = tool(b"# Title\n\nSome text.\n", filename="note.md")
    assert "Some text." in doc.export_to_markdown()
