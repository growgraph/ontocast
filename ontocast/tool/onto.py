"""Base tool class for OntoCast tools.

This module provides the base Tool class that serves as a foundation for all
tools in the OntoCast system. It provides common functionality and interface
for tool implementations.
"""

from typing import Any

from ontocast.onto.model import BasePydanticModel


class Tool(BasePydanticModel):
    """Base class for all OntoCast tools.

    This class serves as the foundation for all tools in the OntoCast system.
    It provides common functionality and interface that all tools must implement.
    Tools should inherit from this class and implement their specific
    functionality. All attributes are inherited from
    :class:`~ontocast.onto.model.BasePydanticModel`; this class adds none.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the tool.

        Args:
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
