"""Document conversion tools for OntoCast.

This module provides functionality for converting various document formats
into structured data that can be processed by the OntoCast system.
"""

import importlib
import logging
import pathlib
import threading
from io import BytesIO
from typing import Any, Union

from docling_core.types.doc import DoclingDocument
from pydantic import Field

from .cache import Cacher, ToolCacher
from .onto import Tool

logger = logging.getLogger(__name__)


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
        default={".pdf", ".ppt", ".pptx"},
        description="Set of supported file extensions",
    )
    cache: Any = Field(default=None, exclude=True)

    def __init__(
        self,
        cache: Cacher | None = None,
        **kwargs,
    ):
        """Initialize the converter tool.

        Args:
            cache: Optional shared Cacher instance. If None, creates a new one.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._converter = None
        self._converter_lock = threading.Lock()  # Lock for thread-safe converter access

        # Initialize cache - use shared cacher or create new one
        if cache is not None:
            self.cache = ToolCacher(cache, "converter_v2")
        else:
            # Fallback for backward compatibility
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, "converter_v2")

        try:
            document_converter_module = importlib.import_module(
                "docling.document_converter"
            )
            DocumentConverter = getattr(document_converter_module, "DocumentConverter")
            self._converter = DocumentConverter()
        except ImportError as e:
            logger.error(f"Could not import DocumentConverter: {e}")

    def __call__(self, file_input: Union[bytes, str, pathlib.Path]) -> DoclingDocument:
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
        cached_result = self.cache.get(content_for_cache)
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

        # Cache the result as JSON for stable serialization
        self.cache.set(content_for_cache, converted_result.model_dump_json())
        logger.debug("Cached document conversion result")

        return converted_result
