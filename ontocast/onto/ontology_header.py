"""Lightweight ontology catalog metadata read without materializing graphs."""

from pydantic import Field, field_validator

from ontocast.onto.ontology import (
    Ontology,
    OntologyPropertiesWithLineage,
    normalize_semantic_version,
)


class OntologyHeader(OntologyPropertiesWithLineage):
    """Ontology header metadata for a single stored named graph.

    A header describes one *version* of an ontology (one named graph), carrying
    exactly the lineage fields terminal-version selection needs -- ``iri``,
    ``hash``, ``parent_hashes``, ``created_at``, ``version`` -- without the graph
    itself. This is deliberately not an :class:`~ontocast.onto.ontology.Ontology`:
    constructing one recomputes ``hash`` from the graph and defaults ``version``,
    so a graph-less ``Ontology`` would carry fabricated lineage values.

    Attributes:
        graph_uri: Named graph URI holding this ontology version. May be the base
            ontology IRI or a versioned form (``<iri>#<hash>``).
    """

    graph_uri: str = Field(
        default="",
        description="Named graph URI holding this ontology version.",
    )

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, value: str | None) -> str | None:
        """Coerce stored ``owl:versionInfo`` text into a semantic version.

        Stored version info is arbitrary text (``"1.0"``, ``"3"``, ...), while
        ``Ontology`` normalizes it on construction. Mirror that here so header and
        ontology versions agree -- terminal selection tie-breaks on the value.
        """
        if value is None or not isinstance(value, str):
            return value
        return normalize_semantic_version(value)

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> "OntologyHeader":
        """Build a header from a materialized ontology.

        Args:
            ontology: The ontology whose metadata should be captured.

        Returns:
            OntologyHeader: Header carrying the ontology's lineage metadata, with
            ``graph_uri`` set to the ontology's versioned IRI.
        """
        return cls(
            graph_uri=ontology.versioned_iri,
            iri=ontology.iri,
            ontology_id=ontology.ontology_id,
            title=ontology.title,
            description=ontology.description,
            version=ontology.version,
            hash=ontology.hash,
            parent_hashes=list(ontology.parent_hashes),
            created_at=ontology.created_at,
        )
