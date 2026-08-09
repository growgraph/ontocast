"""Tests for Docling converter configuration and temporary text repair."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ontocast.config import Config, ConverterConfig, PathConfig, ToolConfig
from ontocast.onto.docling_helpers import (
    apply_text_sanitizers,
    plain_text_to_docling_doc,
    repair_ligature_gaps,
)
from ontocast.tool.cache import Cacher
from ontocast.tool.converter import ConverterTool, build_document_converter
from ontocast.toolbox import ToolBox


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
