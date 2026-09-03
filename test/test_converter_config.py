"""Tests for Docling converter configuration and temporary text repair."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
    ConverterTool,
    build_document_converter,
)
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit


def test_converter_config_born_digital_profile_applies_expected_defaults() -> None:
    config = ConverterConfig(profile="born_digital")

    assert config.pdf_backend == "pypdfium2"
    assert config.do_ocr is False
    assert config.force_backend_text is True
    assert config.repair_ligature_gaps is True


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

        default_tool = ConverterTool(
            cache=shared_cache,
            converter_config=ConverterConfig(do_ocr=True),
        )
        born_digital_tool = ConverterTool(
            cache=shared_cache,
            converter_config=ConverterConfig(profile="born_digital"),
        )

        default_tool.cache.set(
            content,
            doc_json,
            config=default_tool.converter_config.model_dump(mode="json"),
        )

        assert (
            default_tool.cache.get(
                content,
                config=default_tool.converter_config.model_dump(mode="json"),
            )
            is not None
        )
        assert (
            born_digital_tool.cache.get(
                content,
                config=born_digital_tool.converter_config.model_dump(mode="json"),
            )
            is None
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
            converter_config=ConverterConfig(profile="born_digital"),
        )

        toolbox = ToolBox(Config(tool_config=tool_config))

        assert toolbox.converter.converter_config.profile == "born_digital"
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
        assert tool._converter is not None


# --- CONVERTER_REPAIR_NUMERIC_ARTIFACTS --------------------------------------


def test_repair_numeric_artifacts_default_is_off() -> None:
    assert ConverterConfig().repair_numeric_artifacts is False
    assert ConverterConfig(profile="born_digital").repair_numeric_artifacts is False


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

        legacy_key = tool.converter_config.model_dump(mode="json")
        legacy_key.pop("repair_numeric_artifacts")
        legacy_key["cache_format_version"] = CONVERTER_CACHE_FORMAT_VERSION
        tool.cache.set(
            content,
            plain_text_to_docling_doc("cached", "doc").model_dump_json(),
            config=legacy_key,
        )

        doc = tool(content)

        assert builds == []
        assert doc.export_to_markdown().strip() == "cached"
