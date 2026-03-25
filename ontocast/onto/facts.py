from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.rdfgraph import RDFGraph


class Facts(BaseModel):
    """A lightweight wrapper around an RDF graph of extracted facts.

    Facts are expected to use the fixed `cd:` namespace as produced by the
    facts rendering prompts.
    """

    graph: RDFGraph = Field(default_factory=RDFGraph)

    # Keep naming aligned with GraphAtomizer's "source" contract so we can
    # reuse the atomizer for both Ontology and Facts.
    ontology_id: str | None = Field(
        default=None,
        description="Optional source identifier (e.g., document/chunk id).",
    )
    iri: str = Field(
        default=DEFAULT_IRI,
        description="Facts graph IRI (for stable source identity).",
    )
    hash: str | None = Field(
        default=None,
        description="Hash/version of the facts graph (optional).",
    )
    version: str | None = Field(
        default="1.0.0",
        description="Semantic version of the facts graph.",
    )

    facts_namespace: str = Field(
        default=DEFAULT_IRI,
        description="Base namespace for `cd:` entities in this facts graph.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_cd_only(self) -> "Facts":
        """Sanity-check that the facts graph contains `cd:`-namespaced URIRefs."""
        if len(self.graph) == 0:
            return self

        # Accept both `.../facts` and `.../facts/` forms.
        facts_ns = self.facts_namespace.rstrip("/")

        for subj, pred, obj in self.graph:
            for term in (subj, pred, obj):
                if isinstance(term, URIRef) and str(term).startswith(facts_ns):
                    return self

        raise ValueError(
            "Facts graph must contain at least one `cd:`-namespaced URIRef "
            f"(facts_namespace={self.facts_namespace!r})."
        )
