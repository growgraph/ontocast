"""Document conversion tools for OntoCast.

This module provides functionality for converting various document formats
into structured data that can be processed by the OntoCast system.
"""

from io import BytesIO
from typing import Any, Union

from docling.datamodel.base_models import (
    DocumentStream,
)
from docling.document_converter import DocumentConverter

from .onto import Tool


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
        super().__init__(**kwargs)
        self._converter = DocumentConverter()

    def __call__(self, file_input: Union[bytes, str]) -> dict[str, Any]:
        """Convert a document to structured data.

        Args:
            file_input: The input file as either a BytesIO object or file path.

        Returns:
            dict[str, Any]: The converted document data.
        """
        if isinstance(file_input, bytes):
            ds = DocumentStream(name="doc", stream=BytesIO(file_input))
            result = self._converter.convert(ds)
            doc = result.document.export_to_markdown()
            return {"text": doc}
        else:
            # For non-BytesIO input (like plain text), return as is
            return {"text": file_input}
