"""Document conversion tools for OntoCast.

This module provides functionality for converting various document formats
into structured data that can be processed by the OntoCast system.
"""

from __future__ import annotations

import importlib
import logging
import pathlib
import threading
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from ontocast.config import ConverterConfig
from ontocast.onto.docling_helpers import apply_text_sanitizers
from ontocast.util.optional import is_available, require

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

from .cache import CONVERTER_CACHE_SUBDIR, Cacher, ToolCacher
from .onto import Tool
from .pdf_regime import pdf_has_text_layer

logger = logging.getLogger(__name__)

# Bumped when the cached DoclingDocument shape changes, replacing the older
# practice of renaming the cache subdirectory. 2: keys carry the resolved
# profile and its preset fields.
CONVERTER_CACHE_FORMAT_VERSION = 2

# Bumped whenever a numeric-artifact repair rule changes the text it produces.
# Without it a rule fix is inert wherever conversions are cached: the key is
# the same, so the cache keeps serving text repaired by the old rule and the
# fix looks like it did nothing. It joins the key only when the repair flag is
# set -- with repair off no rule ran, so no rule version can have changed the
# output, and pre-existing entries stay valid.
CONVERTER_REPAIR_RULES_VERSION = 2


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
    """Build a Docling DocumentConverter from OntoCast converter settings.

    ``config`` is taken as given; resolve an ``auto`` profile first
    (``ConverterConfig.resolved``).
    """
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
    FormatToExtensions = getattr(base_models_module, "FormatToExtensions")
    DocumentConverter = getattr(document_converter_module, "DocumentConverter")
    PdfFormatOption = getattr(document_converter_module, "PdfFormatOption")
    PdfPipelineOptions = getattr(
        pipeline_options_module, "ThreadedPdfPipelineOptions", None
    ) or getattr(pipeline_options_module, "PdfPipelineOptions")
    TableFormerMode = getattr(pipeline_options_module, "TableFormerMode")
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
            do_cell_matching=config.table_cell_matching,
            mode=TableFormerMode(config.table_mode),
        ),
        do_formula_enrichment=config.do_formula_enrichment,
    )
    backend_map = {
        "docling_parse": DoclingParseDocumentBackend,
        "pypdfium2": PyPdfiumDocumentBackend,
    }
    pdf_format_option = PdfFormatOption(
        pipeline_options=pipeline_options,
        backend=backend_map[config.pdf_backend],
    )

    # Docling fills the options of every other allowed format from its
    # defaults; only PDF is configured here.
    format_options = {InputFormat.PDF: pdf_format_option}
    wanted = {suffix.lstrip(".") for suffix in config.supported_extensions}
    allowed_formats = [
        input_format
        for input_format, suffixes in FormatToExtensions.items()
        if wanted.intersection(suffixes)
    ]
    return DocumentConverter(
        allowed_formats=allowed_formats or None, format_options=format_options
    )


#: A profile a document actually converts with (``auto`` resolves to one).
FixedProfile = Literal["fast", "lean", "ocr"]

#: Suffixes Docling recognises from content, so their cache keys carry no suffix.
_SNIFFED_SUFFIXES = frozenset({".pdf", ".pptx"})


class ConverterTool(Tool):
    """Tool for converting documents to native DoclingDocument format.

    This class provides functionality for converting various document formats
    into DoclingDocument objects that can be processed by the OntoCast system.
    It includes caching to avoid re-converting the same documents.

    Attributes:
        supported_extensions: Suffixes this install converts:
            ``CONVERTER_SUPPORTED_EXTENSIONS``, or none when Docling is not
            installed.
        cache: Cacher instance for caching conversion results.
    """

    supported_extensions: set[str] = Field(
        default_factory=set,
        description="File suffixes this install converts",
    )
    cache: Any = Field(default=None, exclude=True)
    converter_config: ConverterConfig = Field(default_factory=ConverterConfig)

    def __init__(
        self,
        cache: Cacher | None = None,
        converter_config: ConverterConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize the converter tool.

        Args:
            cache: Optional shared Cacher instance. If None, creates a new one.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.converter_config = converter_config or ConverterConfig()
        if "supported_extensions" not in kwargs:
            # Computed from the install, so /info never advertises a format
            # whose conversion would fail on import.
            self.supported_extensions = (
                set(self.converter_config.supported_extensions)
                if is_available("docling")
                else set()
            )
        # One Docling converter per resolved profile, each built on first use.
        self._converters: dict[str, Any] = {}
        self._converter_lock = threading.Lock()  # Lock for thread-safe converter access

        # Initialize cache - use shared cacher or create new one
        if cache is not None:
            self.cache = ToolCacher(cache, CONVERTER_CACHE_SUBDIR)
        else:
            # Standalone use (CLI helpers, direct library use): fall back to a
            # private Cacher on the configured/default directory.
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, CONVERTER_CACHE_SUBDIR)

    @property
    def _converter(self) -> Any:
        """The converter for the default resolved profile, or None before first use."""
        return self._converters.get(self._default_profile())

    def _default_profile(self) -> FixedProfile:
        profile = self.converter_config.profile
        return "fast" if profile == "auto" else profile

    def resolve_profile(self, content: bytes, filename: str | None) -> FixedProfile:
        """The fixed profile a document converts with.

        ``auto`` checks a PDF's text layer: ``fast`` when it has one, ``ocr``
        when its pages are images only. Other formats take ``fast``; it differs
        from ``ocr`` only in PDF options.
        """
        if self.converter_config.profile != "auto":
            return self.converter_config.profile
        suffix = pathlib.Path(filename).suffix.lower() if filename else ""
        if suffix == ".pdf" or (not suffix and content[:5] == b"%PDF-"):
            return "fast" if pdf_has_text_layer(content) else "ocr"
        return "fast"

    def ensure_converter(self, profile: FixedProfile | None = None) -> Any:
        """Return the Docling converter for ``profile``, building it on first use.

        Exposed so a server can warm the models at startup instead of making the
        first request pay for loading the layout, OCR and table-structure models.
        Without ``profile``, the configured one (``fast`` under ``auto``).

        Returns:
            Any: The shared docling ``DocumentConverter``. Untyped because
            docling is an optional dependency resolved lazily.
        """
        profile = profile or self._default_profile()
        converter = self._converters.get(profile)
        if converter is not None:
            return converter
        with self._converter_lock:
            if profile not in self._converters:
                logger.info("Building Docling DocumentConverter (profile %s)", profile)
                try:
                    self._converters[profile] = build_document_converter(
                        self.converter_config.resolved(profile)
                    )
                except ImportError as e:
                    logger.error("Could not import DocumentConverter: %s", e)
                    raise
            return self._converters[profile]

    def cache_config(
        self, filename: str | None, profile: FixedProfile | None = None
    ) -> dict[str, Any]:
        """Config part of the conversion cache key for a file called ``filename``.

        Keyed by the resolved config, so a document converted under ``auto``
        shares its entry with the same document under the profile ``auto``
        chose. The accepted suffix set is left out, so widening or narrowing it
        re-converts nothing. The suffix joins the key only beyond PDF and PPTX,
        whose keys predate it: other formats are chosen by name, and the same
        bytes parse differently as Markdown and as AsciiDoc.
        """
        profile = profile or self._default_profile()
        config_dict = self.converter_config.resolved(profile).model_dump(mode="json")
        config_dict.pop("supported_extensions", None)
        # Off is the pre-existing output, so the flag joins the key only when
        # it changes the text: enabling it re-converts, leaving it off keeps
        # every conversion cached before the flag existed.
        if config_dict.get("repair_numeric_artifacts"):
            config_dict["repair_rules_version"] = CONVERTER_REPAIR_RULES_VERSION
        else:
            config_dict.pop("repair_numeric_artifacts", None)
        config_dict["cache_format_version"] = CONVERTER_CACHE_FORMAT_VERSION
        suffix = pathlib.Path(filename).suffix.lower() if filename else ""
        if suffix and suffix not in _SNIFFED_SUFFIXES:
            config_dict["input_suffix"] = suffix
        return config_dict

    def __call__(
        self, file_input: bytes | str | pathlib.Path, *, filename: str | None = None
    ) -> DoclingDocument:
        """Convert a document to a DoclingDocument.

        Args:
            file_input: The input file as either bytes, string, or pathlib.Path.
            filename: Name of an uploaded file given as bytes. Docling picks
                text formats (Markdown, AsciiDoc) by name, not content.

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

        if filename is None and isinstance(file_input, pathlib.Path):
            filename = file_input.name
        profile = self.resolve_profile(content_for_cache, filename)
        # Check cache first. The format version lives in the key, so bumping it
        # orphans stale entries in place rather than stranding a whole directory.
        config_dict = self.cache_config(filename, profile)
        cached_result = self.cache.get(content_for_cache, config=config_dict)
        if cached_result is not None:
            logger.debug("Cache hit for document conversion")
            docling_document = require(
                "docling_core.types.doc", feature="Document conversion"
            ).DoclingDocument
            if isinstance(cached_result, docling_document):
                return cached_result
            if isinstance(cached_result, str):
                return docling_document.model_validate_json(cached_result)
            if isinstance(cached_result, dict):
                return docling_document.model_validate(cached_result)

        converter = self.ensure_converter(profile)

        # Deliberately outside the lock: conversion is the multi-second part, and
        # holding the build lock across it serialised every concurrent document
        # in the process behind one another. Docling's convert() is a per-call
        # pipeline over its own result objects.
        if isinstance(file_input, bytes):
            try:
                base_models_module = importlib.import_module(
                    "docling.datamodel.base_models"
                )
                DocumentStream = getattr(base_models_module, "DocumentStream")
                ds = DocumentStream(name=filename or "doc", stream=BytesIO(file_input))
            except ImportError:
                raise ImportError(f"Could not import DocumentConverter: {file_input}")
            result = converter.convert(ds)
            converted_result = result.document
        elif isinstance(file_input, pathlib.Path):
            result = converter.convert(file_input)
            converted_result = result.document
        else:
            raise TypeError(f"Unsupported file input type: {type(file_input).__name__}")

        converted_result = apply_text_sanitizers(
            converted_result,
            repair_ligature_gaps_enabled=self.converter_config.repair_ligature_gaps,
            repair_numeric_artifacts_enabled=(
                self.converter_config.repair_numeric_artifacts
            ),
        )

        # Cache the result as JSON for stable serialization
        self.cache.set(
            content_for_cache,
            converted_result.model_dump_json(),
            config=config_dict,
        )
        logger.debug("Cached document conversion result")

        return converted_result
