"""Document conversion tools for OntoCast.

This module provides functionality for converting various document formats
into structured data that can be processed by the OntoCast system.
"""

import logging
import pathlib
from io import BytesIO
from typing import Any, Union

from docling.datamodel.base_models import (
    DocumentStream,
)

from .onto import Tool

logger = logging.getLogger(__name__)


class ConverterTool(Tool):
    """Tool for converting documents to structured data.

    This class provides functionality for converting various document formats
    into structured data that can be processed by the OntoCast system.

    Attributes:
        supported_extensions: Set of supported file extensions.
    """

    supported_extensions: set[str] = {".pdf", ".ppt", ".pptx"}

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the converter tool.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        try:
            from docling.document_converter import DocumentConverter

            super().__init__(**kwargs)
            self._converter: DocumentConverter | None = DocumentConverter()
        except ImportError as e:
            logger.error(f"Could not import DocumentConverter: {e}")

    def __call__(self, file_input: Union[bytes, str, pathlib.Path]) -> dict[str, Any]:
        """Convert a document to structured data.

        Args:
            file_input: The input file as either bytes, string, or pathlib.Path.

        Returns:
            dict[str, Any]: The converted document data.
        """
        if isinstance(file_input, bytes):
            ds = DocumentStream(name="doc", stream=BytesIO(file_input))
            if self._converter is None:
                raise ImportError(f"Could not import DocumentConverter: {file_input}")
            result = self._converter.convert(ds)
            doc = result.document.export_to_markdown()
            return {"text": doc}
        elif isinstance(file_input, pathlib.Path):
            if self._converter is None:
                raise ImportError(f"Could not import DocumentConverter: {file_input}")
            result = self._converter.convert(file_input)
            doc = result.document.export_to_markdown()
            return {"text": doc}
        else:
            # For non-BytesIO input (like plain text), return as is
            return {"text": file_input}
