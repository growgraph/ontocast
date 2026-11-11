"""One prefix per namespace: the rule every prompt, parse and export follows.

A namespace is shown under exactly one prefix, chosen by the first rule that
applies and leaves the prefix free:

1. a well-known namespace keeps its standard prefix (``unit:``, ``sosa:``);
2. a catalog ontology's own namespace takes its author's declared prefix, when
   that prefix is valid -- not a standard prefix of another namespace, not a
   library id;
3. else the ontology's id;
4. a namespace no catalog ontology owns takes the prefix its users bind most;
5. else a slug of the namespace.

Every other prefix seen bound to a namespace is an alias: accepted when the
model writes it and rewritten where authored prose uses it, never shown. The
ontology id stays the identity alias whatever is shown.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import cache
from typing import TYPE_CHECKING

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDFS, SKOS

from ontocast.onto.constants import prefix_lookup_for_ingest
from ontocast.onto.identity import LIBRARY_ONTOLOGY_IDS, STANDARD_PREFIX_NAMESPACES
from ontocast.onto.well_known import WELL_KNOWN_VOCABULARIES

if TYPE_CHECKING:
    from ontocast.onto.ontology import Ontology

#: A Turtle prefix name (``PN_PREFIX``, ASCII subset) that ends in a name char.
_PREFIX_NAME = re.compile(r"^[A-Za-z](?:[\w.-]*[\w-])?$")
#: Names rdflib invents for unbound namespaces; never an author's.
_GENERATED_PREFIX = re.compile(r"^(?:ns|default)\d*$")
#: ``prefix:Local`` in prose. The prefix may not follow a name, ``:``, ``/``,
#: ``#`` or ``<`` character, so IRIs and times never match; a local name may
#: not start with ``/``, so ``http://`` never does.
_CURIE = re.compile(r"(?<![\w:/#<.-])([A-Za-z][\w.-]*):([A-Za-z_][\w.-]*)")

#: Annotation predicates whose text a prompt shows as notes.
_PROSE_PREDICATES: tuple[URIRef, ...] = (
    SKOS.scopeNote,
    SKOS.definition,
    SKOS.note,
    SKOS.example,
    RDFS.comment,
)


@cache
def standard_namespace_prefixes() -> dict[str, str]:
    """Namespace IRI -> its standard prefix.

    The well-known vocabularies first, then the library's prefix tables, then
    the remaining standard prefixes, the lexically first where a namespace has
    several.
    """
    found: dict[str, str] = {}
    for vocabulary in WELL_KNOWN_VOCABULARIES:
        for namespace in vocabulary.namespaces:
            found.setdefault(namespace, vocabulary.prefix)
    for prefix, namespace in prefix_lookup_for_ingest().items():
        found.setdefault(namespace, prefix)
    for prefix in sorted(STANDARD_PREFIX_NAMESPACES):
        for namespace in sorted(STANDARD_PREFIX_NAMESPACES[prefix]):
            found.setdefault(namespace, prefix)
    return found


def standard_prefix_for(namespace: str) -> str | None:
    """The standard prefix of ``namespace``, if it is a standard namespace."""
    return standard_namespace_prefixes().get(str(namespace))


def valid_shown_prefix(prefix: str, namespace: str) -> bool:
    """Whether ``prefix`` may be shown for ``namespace``.

    It must be a prefix name, not one rdflib generated, not a library
    namespace's, and not the standard prefix of another namespace.
    """
    if not prefix or not _PREFIX_NAME.match(prefix):
        return False
    if _GENERATED_PREFIX.match(prefix) or prefix in LIBRARY_ONTOLOGY_IDS:
        return False
    stem = prefix.rstrip("0123456789")
    if stem != prefix and stem in STANDARD_PREFIX_NAMESPACES:
        # rdflib's rename of a clashing standard prefix (``time1``).
        return False
    standard = STANDARD_PREFIX_NAMESPACES.get(prefix)
    return standard is None or str(namespace) in standard


def _bound_prefix(graph: Graph, namespace: str) -> str | None:
    for prefix, bound in graph.namespaces():
        if prefix and str(bound) == namespace:
            return prefix
    return None


def _bind(graph: Graph, prefix: str, namespace: str) -> None:
    """Bind ``prefix`` to ``namespace``, moving it off any other namespace."""
    graph.namespace_manager.bind(
        prefix, Namespace(namespace), override=True, replace=True
    )


def bind_own_namespace(graph: Graph, namespace: str, ontology_id: str) -> str:
    """Bind the shown prefix of an ontology's own namespace on its graph.

    A standard namespace takes its standard prefix; else a valid prefix the
    graph already binds to it (the author's) is kept; else the id is bound.

    Returns:
        The prefix now bound to ``namespace``.
    """
    namespace = str(namespace)
    standard = standard_prefix_for(namespace)
    if standard is not None:
        _bind(graph, standard, namespace)
        return standard
    existing = _bound_prefix(graph, namespace)
    if existing is not None and valid_shown_prefix(existing, namespace):
        return existing
    graph.namespace_manager.bind(ontology_id, Namespace(namespace), override=True)
    return _bound_prefix(graph, namespace) or ontology_id


def _slug(namespace: str) -> str:
    tail = namespace.rstrip("#/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", tail).strip("_-").lower()
    if not slug or not slug[0].isalpha():
        slug = f"n{slug}" if slug else "ns"
    return slug


class PrefixTable:
    """The shown prefix of each namespace, and the aliases accepted for it.

    Build it with :func:`build_prefix_table`. Standard namespaces need no
    entry: :meth:`prefix_for` falls back to their standard prefix.
    """

    def __init__(self, shown: Mapping[str, str], aliases: Mapping[str, str]) -> None:
        self.shown: dict[str, str] = dict(shown)
        self.aliases: dict[str, str] = dict(aliases)
        known = {**standard_namespace_prefixes(), **self.shown}
        # Longest namespace first, so a namespace nested in another wins.
        self._by_length = sorted(known.items(), key=lambda item: -len(item[0]))

    def prefix_for(self, namespace: str) -> str | None:
        """The prefix shown for ``namespace``, if it has one."""
        namespace = str(namespace)
        return self.shown.get(namespace) or standard_prefix_for(namespace)

    def namespace_for(self, prefix: str) -> str | None:
        """The namespace a shown prefix or an alias stands for."""
        return self.parse_map().get(prefix)

    def qname(self, iri: str) -> str:
        """``iri`` as ``prefix:local`` under the shown prefixes, else as is."""
        text = str(iri)
        for namespace, prefix in self._by_length:
            if text.startswith(namespace):
                local = text[len(namespace) :]
                if local and re.fullmatch(r"[A-Za-z0-9_][\w.-]*", local):
                    return f"{prefix}:{local}"
        return text

    def parse_map(self) -> dict[str, str]:
        """Prefix -> namespace for every shown prefix and every alias."""
        result = {prefix: namespace for namespace, prefix in self._by_length if prefix}
        for prefix, namespace in self.aliases.items():
            result.setdefault(prefix, namespace)
        return result

    def _namespace_of(self, iri: str, cache: dict[str, str | None]) -> str | None:
        cut = max(iri.rfind("#"), iri.rfind("/")) + 1
        stem = iri[:cut]
        if stem not in cache:
            cache[stem] = next(
                (ns for ns, _ in self._by_length if stem.startswith(ns)), None
            )
        return cache[stem]

    def bind(self, graph: Graph) -> None:
        """Bind the shown prefix of every namespace ``graph`` binds or uses."""
        namespaces = {str(bound) for prefix, bound in graph.namespaces() if prefix}
        cache: dict[str, str | None] = {}
        for triple in graph:
            for term in triple:
                if isinstance(term, URIRef):
                    namespace = self._namespace_of(str(term), cache)
                    if namespace is not None:
                        namespaces.add(namespace)
        for namespace in sorted(namespaces):
            prefix = self.prefix_for(namespace)
            if prefix is not None and _bound_prefix(graph, namespace) != prefix:
                _bind(graph, prefix, namespace)

    def rewrite_curies(
        self, text: str, source_bindings: Mapping[str, str] | None = None
    ) -> str:
        """Show every CURIE in ``text`` under its namespace's shown prefix.

        A token ``prefix:Local`` is rewritten only when ``source_bindings``
        (the bindings of the graph the text comes from) or an unambiguous
        alias maps its prefix to a namespace shown under another prefix. Any
        other text, unknown prefixes included, is left as it is.
        """
        lookup = {**self.aliases, **(source_bindings or {})}

        def replace(match: re.Match[str]) -> str:
            prefix, local = match.group(1), match.group(2)
            namespace = lookup.get(prefix)
            shown = self.prefix_for(namespace) if namespace is not None else None
            if shown is None or shown == prefix:
                return match.group(0)
            return f"{shown}:{local}"

        return _CURIE.sub(replace, text)

    def normalize(self, graph: Graph) -> None:
        """Make ``graph`` read under this table, in place.

        Binds the shown prefixes and rewrites the CURIEs in its note text, so
        a prompt rendered from it shows one prefix per namespace. Use it on a
        prompt's copy of a context graph, never on a stored ontology.
        """
        source = {
            prefix: str(namespace) for prefix, namespace in graph.namespaces() if prefix
        }
        self.bind(graph)
        for predicate in _PROSE_PREDICATES:
            for subject, obj in list(graph.subject_objects(predicate)):
                if not isinstance(obj, Literal):
                    continue
                rewritten = self.rewrite_curies(str(obj), source)
                if rewritten != str(obj):
                    graph.remove((subject, predicate, obj))
                    graph.add(
                        (
                            subject,
                            predicate,
                            Literal(
                                rewritten, lang=obj.language, datatype=obj.datatype
                            ),
                        )
                    )


def build_prefix_table(
    ontologies: Iterable[Ontology], others: Iterable[Graph] = ()
) -> PrefixTable:
    """The prefix table of a catalog.

    Args:
        ontologies: The catalog's head ontologies.
        others: Further graphs whose bindings name namespaces (the shapes).

    Returns:
        The table: rules 2-5 of the module docstring, in that order; rule 1
        holds through :meth:`PrefixTable.prefix_for`.
    """
    heads = sorted(
        (o for o in ontologies if not o.is_null() and o.namespace),
        key=lambda o: o.iri,
    )
    graphs: list[Graph] = [o.graph for o in heads] + list(others)
    seen: dict[str, Counter[str]] = {}
    for graph in graphs:
        for prefix, namespace in graph.namespaces():
            if prefix and not _GENERATED_PREFIX.match(prefix):
                seen.setdefault(str(namespace), Counter())[prefix] += 1

    standard = standard_namespace_prefixes()
    taken: set[str] = set(STANDARD_PREFIX_NAMESPACES) | set(standard.values())
    taken |= LIBRARY_ONTOLOGY_IDS
    shown: dict[str, str] = {}

    def claim(namespace: str, *candidates: str | None) -> None:
        for candidate in candidates:
            if (
                candidate
                and candidate not in taken
                and valid_shown_prefix(candidate, namespace)
            ):
                shown[namespace] = candidate
                taken.add(candidate)
                return
        base = next((c for c in candidates if c), None) or _slug(namespace)
        number = 2
        while f"{base}{number}" in taken:
            number += 1
        shown[namespace] = f"{base}{number}"
        taken.add(shown[namespace])

    owners: dict[str, Ontology] = {}
    for head in heads:
        owners.setdefault(str(head.namespace), head)
    for namespace, head in sorted(owners.items()):
        if namespace in standard:
            continue
        claim(namespace, head.prefix, head.ontology_id)
    for namespace in sorted(seen):
        if namespace in standard or namespace in shown:
            continue
        ranked = sorted(seen[namespace].items(), key=lambda item: (-item[1], item[0]))
        claim(namespace, *(prefix for prefix, _ in ranked), _slug(namespace))

    claims: dict[str, set[str]] = {}
    for namespace, counter in seen.items():
        for prefix in counter:
            claims.setdefault(prefix, set()).add(namespace)
    for namespace, head in owners.items():
        if head.ontology_id:
            claims.setdefault(head.ontology_id, set()).add(namespace)
    shown_prefixes = set(shown.values()) | set(standard.values())
    aliases = {
        prefix: next(iter(namespaces))
        for prefix, namespaces in claims.items()
        if len(namespaces) == 1 and prefix not in shown_prefixes
    }
    return PrefixTable(shown, aliases)


__all__ = [
    "PrefixTable",
    "bind_own_namespace",
    "build_prefix_table",
    "standard_namespace_prefixes",
    "standard_prefix_for",
    "valid_shown_prefix",
]
