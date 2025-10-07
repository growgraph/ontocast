"""Base tool class for OntoCast tools.

This module provides the base Tool class that serves as a foundation for all
tools in the OntoCast system. It provides common functionality and interface
for tool implementations.
"""

from pydantic import BaseModel, Field
from rdflib import URIRef

from ontocast.onto.model import BasePydanticModel


class Tool(BasePydanticModel):
    """Base class for all OntoCast tools.

    This class serves as the foundation for all tools in the OntoCast system.
    It provides common functionality and interface that all tools must implement.
    Tools should inherit from this class and implement their specific functionality.

    Attributes:
        Inherits all attributes from BasePydanticModel.
    """

    def __init__(self, **kwargs):
        """Initialize the tool.

        Args:
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)


class EntityMetadata(BaseModel):
    """Metadata for an entity in the graph."""

    model_config = {"arbitrary_types_allowed": True}

    local_name: str = Field(description="The local name of the entity")
    label: str | None = Field(
        default=None, description="Optional human-readable label for the entity"
    )
    comment: str | None = Field(
        default=None, description="Optional comment describing the entity"
    )
    types: set[URIRef] = Field(
        default_factory=set, description="Set of RDF types for this entity"
    )


class PredicateMetadata(BaseModel):
    """Metadata for a predicate in the graph."""

    model_config = {"arbitrary_types_allowed": True}

    local_name: str = Field(description="The local name of the predicate")
    label: str | None = Field(
        default=None, description="Optional human-readable label for the predicate"
    )
    comment: str | None = Field(
        default=None, description="Optional comment describing the predicate"
    )
    domain: None | URIRef = Field(
        default=None, description="Optional domain of the predicate"
    )
    range: None | URIRef = Field(
        default=None, description="Optional range of the predicate"
    )
    is_explicit_property: bool = Field(
        default=False, description="Whether this is an explicit property"
    )
