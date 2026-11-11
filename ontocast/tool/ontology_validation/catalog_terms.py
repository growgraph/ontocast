"""The catalog's terms as the ontology prompts list them.

Shared by the document plan, consolidation and catalog curation: CURIE
prefixes for every held ontology, and each ontology's own classes and
properties with their kind and label.
"""

from __future__ import annotations

from collections.abc import Sequence

from rdflib import OWL, RDF, URIRef

from ontocast.onto.constants import prefix_lookup_for_ingest
from ontocast.onto.identity import OntologyIdentity
from ontocast.onto.ontology import Ontology
from ontocast.onto.prefixes import PrefixTable, build_prefix_table
from ontocast.tool.facts_validation.terms import _catalog_term_roles
from ontocast.tool.ontology_validation.plan import compact_iri
from ontocast.tool.ontology_validation.shape import (
    first_label,
    label_tokens,
    name_tokens,
)


class CatalogTerm:
    """One catalog term as the plan and consolidation prompts list it."""

    __slots__ = ("iri", "kind", "label")

    def __init__(self, iri: str, kind: str, label: str) -> None:
        self.iri = iri
        self.kind = kind
        self.label = label


def catalog_prefixes(
    heads: Sequence[Ontology],
    home: OntologyIdentity | None,
    table: PrefixTable | None = None,
) -> dict[str, str]:
    """CURIE prefixes for the catalog's ontologies, well-known ones and home.

    Each namespace gets the prefix ``table`` shows for it (the catalog's own
    table when none is given), so this lane names terms as the facts lane
    does. A home ontology not in the catalog yet is named by its id.
    """
    table = table if table is not None else build_prefix_table(heads)
    prefixes = dict(prefix_lookup_for_ingest())
    for namespace, prefix in sorted(table.shown.items()):
        prefixes.setdefault(prefix, namespace)
    if home is not None and home.namespace not in prefixes.values():
        prefixes[home.ontology_id] = home.namespace
    return prefixes


def catalog_terms(
    heads: Sequence[Ontology], *, first_iri: str | None = None
) -> list[CatalogTerm]:
    """Classes and properties each head declares, ``first_iri``'s head first."""
    ordered = sorted(
        (head for head in heads if not head.is_null()),
        key=lambda head: (head.iri != first_iri, head.ontology_id or head.iri or ""),
    )
    seen: set[str] = set()
    terms: list[CatalogTerm] = []
    for head in ordered:
        graph = head.graph
        properties, classes = _catalog_term_roles(graph)
        namespace = head.namespace or ""
        for iri in sorted(classes | properties):
            if iri in seen or (namespace and not iri.startswith(namespace)):
                continue
            seen.add(iri)
            subject = URIRef(iri)
            if iri in classes and iri not in properties:
                kind = "class"
            elif (subject, RDF.type, OWL.DatatypeProperty) in graph:
                kind = "datatype_property"
            else:
                kind = "object_property"
            terms.append(CatalogTerm(iri, kind, first_label(subject, graph) or ""))
    return terms


def format_term_sheet(
    terms: Sequence[CatalogTerm], prefixes: dict[str, str], max_terms: int
) -> str:
    """``CURIE | kind | label`` lines, at most ``max_terms`` of them."""
    return "\n".join(
        f"- {compact_iri(term.iri, prefixes)} | {term.kind} | {term.label}".rstrip(" |")
        for term in terms[:max_terms]
    )


#: A term name shorter than this is too generic to say a text is about the
#: ontology (``name``, ``type``).
_MIN_NAME_CHARS = 4


def catalog_coverage(
    heads: Sequence[Ontology], text: str
) -> list[tuple[int, Ontology]]:
    """Each head with the number of its terms the text names, most first.

    A term counts once when its label or its split local name occurs in the
    text as a whole-word phrase, case-insensitive, also with a plural ``s`` or
    ``es``.
    """
    words = " " + " ".join(label_tokens(" ".join(text.split()))) + " "
    covered: list[tuple[int, Ontology]] = []
    for head in heads:
        if head.is_null():
            continue
        hits = 0
        for term in catalog_terms([head]):
            phrases = {" ".join(name_tokens(term.iri))}
            if term.label:
                phrases.add(" ".join(label_tokens(term.label)))
            if any(
                len(phrase) >= _MIN_NAME_CHARS
                and any(f" {phrase}{plural} " in words for plural in ("", "s", "es"))
                for phrase in phrases
            ):
                hits += 1
        covered.append((hits, head))
    covered.sort(key=lambda pair: (-pair[0], pair[1].iri or ""))
    return covered


__all__ = [
    "CatalogTerm",
    "catalog_coverage",
    "catalog_prefixes",
    "catalog_terms",
    "format_term_sheet",
]
