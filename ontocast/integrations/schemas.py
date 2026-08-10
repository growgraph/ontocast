"""Argument schemas for the LangChain tool wrappers.

Each model becomes a tool's ``args_schema``, which providers turn into a JSON
Schema for tool calling. Two constraints follow from that and shape everything
here: every field must be a JSON-primitive type (no ``RDFGraph``, no
``Ontology``), and every field needs a description, because the description is
the only instruction the model gets about how to fill it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NoArgs(BaseModel):
    """Schema for tools that take no arguments."""


class GetOntologyArgs(BaseModel):
    """Fetch one ontology by IRI."""

    iri: str = Field(
        description=(
            "Full ontology IRI, exactly as returned by ontocast_list_ontologies "
            "(not a prefix or short name)."
        )
    )


class SearchOntologyTermsArgs(BaseModel):
    """Vector search over indexed ontology terms."""

    query: str = Field(
        description=(
            "Natural-language description of the concept to find, e.g. "
            "'measurement unit for electrical resistance'. Not a SPARQL query."
        )
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of terms to return. Defaults to the store setting.",
    )
    filter_iri: str | None = Field(
        default=None,
        description="Restrict results to a single ontology IRI.",
    )


class RetrieveOntologyContextArgs(BaseModel):
    """Retrieve a relevant ontology subgraph as Turtle."""

    query: str = Field(
        description=(
            "Natural-language description of the area of the ontology you need. "
            "Returns the surrounding subgraph, not just matching terms."
        )
    )
    top_k: int | None = Field(
        default=None, ge=1, le=100, description="Number of seed terms to expand from."
    )
    subgraph_depth: int | None = Field(
        default=None,
        ge=0,
        le=4,
        description="Neighbourhood expansion depth around each seed term.",
    )
    max_total_triples: int | None = Field(
        default=None,
        ge=1,
        description="Cap on triples in the returned subgraph.",
    )


class SparqlQueryArgs(BaseModel):
    """Run a read-only SPARQL query."""

    query: str = Field(
        description=(
            "A complete SPARQL query string. Read-only forms only: SELECT/ASK for "
            "ontocast_sparql_select, CONSTRUCT/DESCRIBE for ontocast_sparql_construct. "
            "Update forms (INSERT/DELETE/DROP/CLEAR) are rejected."
        )
    )
    use_ontologies_dataset: bool = Field(
        default=True,
        description=(
            "Query the ontology dataset (schema and reference individuals) when "
            "true, or the facts dataset (instances extracted from documents) "
            "when false."
        ),
    )


class ChunkTextArgs(BaseModel):
    """Split text into size-bounded chunks."""

    text: str = Field(description="The document text to split.")


class ExtractArgs(BaseModel):
    """Run the OntoCast extraction pipeline over a piece of text."""

    text: str = Field(description="The source text to extract from.")
    render_mode: str | None = Field(
        default=None,
        description=(
            "What to produce: 'ontology' for schema only, 'facts' for instances "
            "against the existing ontology, or 'ontology_and_facts' for both. "
            "Omit to use the server's configured RENDER_MODE."
        ),
    )
    instruction: str = Field(
        default="",
        description=(
            "Optional extra guidance for the extractor, e.g. 'focus on "
            "experimental conditions'. Appended to the built-in prompt."
        ),
    )
    domain: str | None = Field(
        default=None,
        description="Optional base IRI domain for minted instance identifiers.",
    )


class ApplyGraphUpdateArgs(BaseModel):
    """Apply an insert/delete patch to a graph."""

    insert_ttl: str = Field(
        default="",
        description=(
            "Turtle for triples to add. Include all prefixes you use. May be "
            "empty if you are only deleting."
        ),
    )
    delete_ttl: str = Field(
        default="",
        description=(
            "Turtle for triples to remove. Patterns must match existing triples "
            "exactly. May be empty if you are only inserting."
        ),
    )
    base_ttl: str | None = Field(
        default=None,
        description=(
            "Turtle for the graph to patch. Omit to patch the ontology "
            "currently held by the triple store."
        ),
    )
    target: str = Field(
        default="ontology",
        description="Which graph to patch: 'ontology' or 'facts'.",
    )
    persist: bool = Field(
        default=False,
        description=(
            "Write the result back to the triple store. When false (the "
            "default) the patch is applied and returned but nothing is stored."
        ),
    )


class IngestOntologyArgs(BaseModel):
    """Register a new ontology from Turtle."""

    ttl: str = Field(description="The complete ontology serialized as Turtle.")
    filename: str | None = Field(
        default=None,
        description="Optional filename to store it under in the ontology directory.",
    )


class DeleteOntologyArgs(BaseModel):
    """Remove an ontology and everything derived from it."""

    iri: str = Field(
        description=(
            "IRI of the ontology to delete. This drops its named graph, removes "
            "its file from the ontology directory, and deletes its vectors. "
            "It cannot be undone."
        )
    )


class ConvertDocumentArgs(BaseModel):
    """Convert a document file to markdown."""

    path: str = Field(description="Filesystem path to the document to convert.")


class TaggedGraphArg(BaseModel):
    """One named graph in an entity-alignment request."""

    name: str = Field(description="Label identifying this graph in the results.")
    ttl: str = Field(description="The graph serialized as Turtle.")


class AlignEntitiesArgs(BaseModel):
    """Find equivalent entities across graphs."""

    graphs: list[TaggedGraphArg] = Field(
        description="Two or more named graphs to align against each other."
    )
    regime: str = Field(
        default="ontology_loose",
        description="Matching strictness preset, e.g. 'ontology_loose'.",
    )
