"""Document conversion tools for OntoCast.

This module provides functionality for converting various document formats
into structured data that can be processed by the OntoCast system.
"""

import logging
import pathlib
import threading
from io import BytesIO
from typing import Any, Union

from pydantic import Field

from .cache import Cacher, ToolCacher
from .onto import Tool

logger = logging.getLogger(__name__)


class ConverterTool(Tool):
    """Tool for converting documents to structured data.

    This class provides functionality for converting various document formats
    into structured data that can be processed by the OntoCast system.
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
            self.cache = ToolCacher(cache, "converter")
        else:
            # Fallback for backward compatibility
            shared_cache = Cacher()
            self.cache = ToolCacher(shared_cache, "converter")

        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            self._converter = DocumentConverter()
        except ImportError as e:
            logger.error(f"Could not import DocumentConverter: {e}")

    def __call__(self, file_input: Union[bytes, str, pathlib.Path]) -> dict[str, Any]:
        """Convert a document to structured data.

        Args:
            file_input: The input file as either bytes, string, or pathlib.Path.

        Returns:
            dict[str, Any]: The converted document data.
        """
        # For plain text input, no caching needed
        if isinstance(file_input, str):
            return {"text": file_input}

        # Prepare content for caching
        if isinstance(file_input, bytes):
            content_for_cache = file_input
        elif isinstance(file_input, pathlib.Path):
            content_for_cache = file_input.read_bytes()
        else:
            # Fallback for other types
            return {"text": str(file_input)}

        # Check cache first
        cached_result = self.cache.get(content_for_cache)
        if cached_result is not None:
            logger.debug("Cache hit for document conversion")
            return cached_result

        # Convert document (with thread-safe access to converter)
        with self._converter_lock:
            if isinstance(file_input, bytes):
                if self._converter is None:
                    raise ImportError("DocumentConverter not available")
                try:
                    from docling.datamodel.base_models import (  # type: ignore
                        DocumentStream,
                    )

                    ds = DocumentStream(name="doc", stream=BytesIO(file_input))
                except ImportError:
                    raise ImportError(
                        f"Could not import DocumentConverter: {file_input}"
                    )
                result = self._converter.convert(ds)
                doc = result.document.export_to_markdown()
                converted_result = {"text": doc}
            elif isinstance(file_input, pathlib.Path):
                if self._converter is None:
                    raise ImportError(
                        f"Could not import DocumentConverter: {file_input}"
                    )
                result = self._converter.convert(file_input)
                doc = result.document.export_to_markdown()
                converted_result = {"text": doc}
            else:
                # Fallback for other types
                converted_result = {"text": str(file_input)}

        # Cache the result
        self.cache.set(content_for_cache, converted_result)
        logger.debug("Cached document conversion result")

        return converted_result
