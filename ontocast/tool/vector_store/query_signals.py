"""Query-side unit/quantity signals for deterministic seed injection.

A number adjacent to a short token ("4-15 days", "200 kV", "0.5 %",
"77 K") is strong evidence of a unit mention — strong enough to justify a
more permissive match than the general lexical-trigger lane: catalog
surface forms (labels, symbols, UCUM codes) are compared
case-insensitively and plural-tolerantly, so text "days" reaches
``unit:DAY`` (label "Day", symbol "d") even though the case-sensitive
trigger lane never could.

Matched *values* come from the catalog at runtime. The surface *predicates*
are supplied by the caller (see ``VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES``);
the module-level defaults below are the same overridable defaults the
lexical-trigger lane uses, not a hardcoded vocabulary.

The token heuristics are Latin-script and English-centric: ``_STOP_TOKENS``
is an English function-word list, and the plural rule strips a trailing
"s". Both only ever suppress or widen *candidate* tokens, so a non-English
corpus loses recall on this additive lane rather than extracting wrongly.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

from rdflib import RDFS, SKOS, Literal, URIRef

from ontocast.onto.ontology import Ontology

logger = logging.getLogger(__name__)

# Name predicates every RDF vocabulary shares. Symbol/notation predicates are
# NOT listed here: they are catalog-specific and arrive via the configured
# `symbol_predicates` argument, so no domain vocabulary is compiled in.
DEFAULT_NAME_PREDICATES: tuple[URIRef, ...] = (
    RDFS.label,
    SKOS.prefLabel,
    SKOS.altLabel,
)

# number (with optional range/uncertainty tail) followed by a candidate unit
# token: letters with optional degree/percent/micro glyphs, up to ~12 chars.
_NUMBER_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[-–−±]\s*\d+(?:\.\d+)?\s*)?"
    r"(?P<token>[%‰°]|°[A-Za-z]|[A-Za-zµμ][A-Za-z0-9µμ/·⁻²³%°]{0,11})"
)

_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "per",
        "the",
        "to",
        "with",
        "x",
    }
)


def number_adjacent_tokens(text: str) -> set[str]:
    """Extract candidate unit tokens that directly follow numbers.

    Args:
        text: Source text of one content unit / query window.

    Returns:
        Set of case-preserved candidate tokens.
    """
    tokens: set[str] = set()
    for match in _NUMBER_TOKEN_PATTERN.finditer(text):
        token = match.group("token").strip()
        if not token or token.lower() in _STOP_TOKENS:
            continue
        tokens.add(token)
    return tokens


def _token_keys(token: str) -> set[str]:
    """Case-folded lookup keys for a token, incl. a singular variant."""
    lowered = token.lower()
    keys = {lowered}
    if len(lowered) > 3 and lowered.endswith("s"):
        keys.add(lowered[:-1])
    return keys


class CatalogSurfaceIndex:
    """Case-insensitive surface-form index over catalog ontologies.

    Maps lowered labels/symbols/UCUM codes to entity IRIs, cached per
    ontology (iri, hash) so repeat retrievals are dictionary lookups.
    """

    def __init__(self, symbol_predicates: Sequence[URIRef] | None = None) -> None:
        """Build the index.

        Args:
            symbol_predicates: Catalog symbol/notation predicates to index in
                addition to the standard name predicates. Supplied from
                configuration by the retriever; ``None`` indexes names only.
        """
        self._per_ontology: dict[tuple[str, str], dict[str, set[str]]] = {}
        self._surface_predicates: tuple[URIRef, ...] = (
            *DEFAULT_NAME_PREDICATES,
            *(symbol_predicates or ()),
        )

    def _index_ontology(self, ontology: Ontology) -> dict[str, set[str]]:
        key = (ontology.iri, ontology.hash or "")
        cached = self._per_ontology.get(key)
        if cached is not None:
            return cached
        surface: dict[str, set[str]] = {}
        for predicate in self._surface_predicates:
            for subject, value in ontology.graph.subject_objects(predicate):
                if not isinstance(subject, URIRef) or not isinstance(value, Literal):
                    continue
                text = str(value).strip()
                if not text or " " in text or len(text) > 24:
                    continue
                surface.setdefault(text.lower(), set()).add(str(subject))
        self._per_ontology[key] = surface
        return surface

    def match(
        self,
        tokens: Iterable[str],
        ontologies: Sequence[Ontology],
    ) -> dict[str, str]:
        """Match candidate tokens against catalog surface forms.

        Args:
            tokens: Number-adjacent candidate tokens.
            ontologies: Catalog ontologies to match against.

        Returns:
            Mapping of matched entity IRI -> owning ontology IRI.
        """
        matched: dict[str, str] = {}
        keys: set[str] = set()
        for token in tokens:
            keys |= _token_keys(token)
        if not keys:
            return matched
        for ontology in ontologies:
            if ontology.is_null():
                continue
            surface = self._index_ontology(ontology)
            for key in keys:
                for iri in surface.get(key, ()):
                    matched.setdefault(iri, ontology.iri)
        return matched
