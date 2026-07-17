"""Document conversion tools for OntoCast.

This module provides functionality for converting various document formats
into structured data that can be processed by the OntoCast system.
"""

import importlib
import logging
import pathlib
import threading
from io import BytesIO
from typing import Any

from docling_core.types.doc import DoclingDocument
from pydantic import Field

from ontocast.config import ConverterConfig
from ontocast.onto.docling_helpers import apply_text_sanitizers

from .cache import Cacher, ToolCacher
from .onto import Tool

logger = logging.getLogger(__name__)


def _build_layout_options(config: ConverterConfig) -> Any:
    pipeline_options_module = importlib.import_module(
        "docling.datamodel.pipeline_options"
    )
    layout_specs_module = importlib.import_module(
        "docling.datamodel.layout_model_specs"
    )
    LayoutOptions = getattr(pipeline_options_module, "LayoutOptions")
    model_spec_map = {
        "heron": getattr(layout_specs_module, "DOCLING_LAYOUT_HERON"),
        "heron_101": getattr(layout_specs_module, "DOCLING_LAYOUT_HERON_101"),
        "egret_medium": getattr(layout_specs_module, "DOCLING_LAYOUT_EGRET_MEDIUM"),
        "egret_large": getattr(layout_specs_module, "DOCLING_LAYOUT_EGRET_LARGE"),
        "egret_xlarge": getattr(layout_specs_module, "DOCLING_LAYOUT_EGRET_XLARGE"),
        "v2": getattr(layout_specs_module, "DOCLING_LAYOUT_V2"),
    }
    return LayoutOptions(model_spec=model_spec_map[config.layout_model])


def _build_ocr_options(config: ConverterConfig) -> Any:
    pipeline_options_module = importlib.import_module(
        "docling.datamodel.pipeline_options"
    )
    ocr_kwargs = {
        "lang": config.ocr_lang,
        "force_full_page_ocr": config.force_full_page_ocr,
        "bitmap_area_threshold": config.ocr_bitmap_area_threshold,
    }
    if config.ocr_engine == "auto":
        OcrAutoOptions = getattr(pipeline_options_module, "OcrAutoOptions")
        return OcrAutoOptions(**ocr_kwargs)
    if config.ocr_engine == "easyocr":
        EasyOcrOptions = getattr(pipeline_options_module, "EasyOcrOptions")
        return EasyOcrOptions(**ocr_kwargs)
    if config.ocr_engine == "rapidocr":
        RapidOcrOptions = getattr(pipeline_options_module, "RapidOcrOptions")
        return RapidOcrOptions(**ocr_kwargs)
    if config.ocr_engine == "tesseract_cli":
        TesseractCliOcrOptions = getattr(
            pipeline_options_module, "TesseractCliOcrOptions"
        )
        return TesseractCliOcrOptions(**ocr_kwargs)

    TesseractOcrOptions = getattr(pipeline_options_module, "TesseractOcrOptions")
    return TesseractOcrOptions(**ocr_kwargs)


def build_document_converter(config: ConverterConfig) -> Any:
    """Build a Docling DocumentConverter from OntoCast converter settings."""
    base_models_module = importlib.import_module("docling.datamodel.base_models")
    document_converter_module = importlib.import_module("docling.document_converter")
    pipeline_options_module = importlib.import_module(
        "docling.datamodel.pipeline_options"
    )
    parse_backend_module = importlib.import_module(
        "docling.backend.docling_parse_backend"
    )
    pypdfium_backend_module = importlib.import_module(
        "docling.backend.pypdfium2_backend"
    )

    InputFormat = getattr(base_models_module, "InputFormat")
    DocumentConverter = getattr(document_converter_module, "DocumentConverter")
    PdfFormatOption = getattr(document_converter_module, "PdfFormatOption")
    PdfPipelineOptions = getattr(pipeline_options_module, "PdfPipelineOptions")
    TableStructureOptions = getattr(
        pipeline_options_module, "TableStructureOptions", None
    ) or getattr(pipeline_options_module, "BaseTableStructureOptions")
    DoclingParseDocumentBackend = getattr(
        parse_backend_module, "DoclingParseDocumentBackend"
    )
    PyPdfiumDocumentBackend = getattr(
        pypdfium_backend_module, "PyPdfiumDocumentBackend"
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=config.do_ocr,
        do_table_structure=config.do_table_structure,
        force_backend_text=config.force_backend_text,
        ocr_options=_build_ocr_options(config),
        layout_options=_build_layout_options(config),
        table_structure_options=TableStructureOptions(
            do_cell_matching=config.table_cell_matching
        ),
    )
    backend_map = {
        "docling_parse": DoclingParseDocumentBackend,
        "pypdfium2": PyPdfiumDocumentBackend,
    }
    pdf_format_option = PdfFormatOption(
        pipeline_options=pipeline_options,
        backend=backend_map[config.pdf_backend],
    )

    default_converter = DocumentConverter()
    format_options = dict(default_converter.format_to_options)
    format_options[InputFormat.PDF] = pdf_format_option
    return DocumentConverter(format_options=format_options)


class ConverterTool(Tool):
    """Tool for converting documents to native DoclingDocument format.

    This class provides functionality for converting various document formats
    into DoclingDocument objects that can be processed by the OntoCast system.
    It includes caching to avoid re-converting the same documents.

    Attributes:
        supported_extensions: Set of supported file extensions.
        cache: Cacher instance for caching conversion results.
    """

    supported_extensions: set[str] = Field(
        default={".pdf", ".pptx"},
        description="Set of supported file extensions",
    )
    cache: Any = Field(default=None, exclude=True)
    converter_config: ConverterConfig = Field(default_factory=ConverterConfig)

    def __init__(
        self,
        cache: Cacher | None = None,
        converter_config: ConverterConfig | None = None,
        **kwargs,
    ):
        """Initialize the converter tool.

        Args:
            cache: Optional shared Cacher instance. If None, creates a new one.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.converter_config = converter_config or ConverterConfig()
        self._converter = None
        self._converter_lock = threading.Lock()  # Lock for thread-safe converter access

        # Initialize cache - use shared cacher or create new one
        if cache is not None:
            self.cache = ToolCacher(cache, "converter_v3")
        else:
            # Fallback for backward compatibility
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, "converter_v3")

        try:
            self._converter = build_document_converter(self.converter_config)
        except ImportError as e:
            logger.error(f"Could not import DocumentConverter: {e}")

    def __call__(self, file_input: bytes | str | pathlib.Path) -> DoclingDocument:
        """Convert a document to a DoclingDocument.

        Args:
            file_input: The input file as either bytes, string, or pathlib.Path.

        Returns:
            DoclingDocument: The converted document.
        """
        # Prepare content for caching
        if isinstance(file_input, bytes):
            content_for_cache = file_input
        elif isinstance(file_input, pathlib.Path):
            content_for_cache = file_input.read_bytes()
        elif isinstance(file_input, str):
            raise TypeError(
                "ConverterTool expects bytes or pathlib.Path; "
                "use plain_text_to_docling_doc for raw text."
            )
        else:
            raise TypeError(f"Unsupported file input type: {type(file_input).__name__}")

        # Check cache first
        config_dict = self.converter_config.model_dump(mode="json")
        cached_result = self.cache.get(content_for_cache, config=config_dict)
        if cached_result is not None:
            logger.debug("Cache hit for document conversion")
            if isinstance(cached_result, DoclingDocument):
                return cached_result
            if isinstance(cached_result, str):
                return DoclingDocument.model_validate_json(cached_result)
            if isinstance(cached_result, dict):
                return DoclingDocument.model_validate(cached_result)

        # Convert document (with thread-safe access to converter)
        with self._converter_lock:
            if isinstance(file_input, bytes):
                if self._converter is None:
                    raise ImportError("DocumentConverter not available")
                try:
                    base_models_module = importlib.import_module(
                        "docling.datamodel.base_models"
                    )
                    DocumentStream = getattr(base_models_module, "DocumentStream")
                    ds = DocumentStream(name="doc", stream=BytesIO(file_input))
                except ImportError:
                    raise ImportError(
                        f"Could not import DocumentConverter: {file_input}"
                    )
                result = self._converter.convert(ds)
                converted_result = result.document
            elif isinstance(file_input, pathlib.Path):
                if self._converter is None:
                    raise ImportError(
                        f"Could not import DocumentConverter: {file_input}"
                    )
                result = self._converter.convert(file_input)
                converted_result = result.document
            else:
                raise TypeError(
                    f"Unsupported file input type: {type(file_input).__name__}"
                )

        converted_result = apply_text_sanitizers(
            converted_result,
            repair_ligature_gaps_enabled=self.converter_config.repair_ligature_gaps,
        )

        # Cache the result as JSON for stable serialization
        self.cache.set(
            content_for_cache,
            converted_result.model_dump_json(),
            config=config_dict,
        )
        logger.debug("Cached document conversion result")

        return converted_result
