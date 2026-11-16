"""What the catalog declares about identity across documents.

Read from every catalog head, never from a document's retrieved context, so
every document sees the same declarations whatever retrieval chose for it.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field
from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef
from rdflib.collection import Collection

from ontocast.onto.constants import GLOBALLY_NAMED
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.signatures import (
    harvest_disjoint_classes,
    harvest_max_one_predicates,
)
from ontocast.tool.facts_validation.terms import collect_catalog_terms


class IdentityDeclarations(BaseModel):
    """The catalog's identity declarations, read once per catalog state.

    Attributes:
        globally_named: Each class with ``ontocast:globallyNamed`` stated, and
            the stated value. When catalog heads disagree, ``false`` wins.
        parents: Each class's ``rdfs:subClassOf`` parents.
        keys: Each class's ``owl:hasKey`` property lists.
        inverse_functional: Properties declared ``owl:InverseFunctionalProperty``.
        functional: Properties the catalog caps at one value
            (``owl:FunctionalProperty`` or a max-cardinality-1 restriction).
        disjoint: Class pairs declared disjoint.
        catalog_terms: Every IRI the catalog declares or references.
    """

    model_config = ConfigDict(frozen=True)

    globally_named: dict[str, bool] = Field(default_factory=dict)
    parents: dict[str, frozenset[str]] = Field(default_factory=dict)
    keys: dict[str, tuple[tuple[str, ...], ...]] = Field(default_factory=dict)
    inverse_functional: frozenset[str] = frozenset()
    functional: frozenset[str] = frozenset()
    disjoint: frozenset[frozenset[str]] = frozenset()
    catalog_terms: frozenset[str] = frozenset()

    @classmethod
    def of(cls, graphs: Iterable[RDFGraph]) -> IdentityDeclarations:
        """The declarations stated across ``graphs``."""
        globally_named: dict[str, bool] = {}
        parents: dict[str, set[str]] = {}
        keys: dict[str, set[tuple[str, ...]]] = {}
        inverse_functional: set[str] = set()
        functional: set[str] = set()
        disjoint: set[frozenset[str]] = set()
        catalog_terms: set[str] = set()
        for graph in graphs:
            for cls_iri, value in graph.subject_objects(GLOBALLY_NAMED):
                if not isinstance(cls_iri, URIRef) or not isinstance(value, Literal):
                    continue
                stated = _as_bool(value)
                if stated is None:
                    continue
                key = str(cls_iri)
                globally_named[key] = globally_named.get(key, True) and stated
            for child, parent in graph.subject_objects(RDFS.subClassOf):
                if isinstance(child, URIRef) and isinstance(parent, URIRef):
                    parents.setdefault(str(child), set()).add(str(parent))
            for cls_iri, members in graph.subject_objects(OWL.hasKey):
                if not isinstance(cls_iri, URIRef) or not isinstance(members, BNode):
                    continue
                props = tuple(
                    sorted(
                        str(m)
                        for m in Collection(graph, members)
                        if isinstance(m, URIRef)
                    )
                )
                if props:
                    keys.setdefault(str(cls_iri), set()).add(props)
            for prop in graph.subjects(RDF.type, OWL.InverseFunctionalProperty):
                if isinstance(prop, URIRef):
                    inverse_functional.add(str(prop))
            functional |= {str(p) for p in harvest_max_one_predicates(graph)}
            disjoint |= {
                frozenset(str(c) for c in pair)
                for pair in harvest_disjoint_classes(graph)
            }
            catalog_terms |= collect_catalog_terms(graph)
        return cls(
            globally_named=globally_named,
            parents={k: frozenset(v) for k, v in parents.items()},
            keys={k: tuple(sorted(v)) for k, v in keys.items()},
            inverse_functional=frozenset(inverse_functional),
            functional=frozenset(functional),
            disjoint=frozenset(disjoint),
            catalog_terms=frozenset(catalog_terms),
        )

    @property
    def declares_anything(self) -> bool:
        """Whether any class or property can produce a link."""
        return (
            any(self.globally_named.values())
            or bool(self.keys)
            or bool(self.inverse_functional)
        )

    def ancestors(self, class_iri: str) -> frozenset[str]:
        """``class_iri`` and every class above it."""
        seen: set[str] = set()
        frontier = [class_iri]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.parents.get(current, ()))
        return frozenset(seen)

    def declaring_classes(self, types: Iterable[str]) -> frozenset[str]:
        """The globally named classes an instance of ``types`` belongs to.

        Walks up from each type and stops at the first class on each path that
        states the annotation: ``true`` makes it a declaring class, ``false``
        stops the path.
        """
        found: set[str] = set()
        seen: set[str] = set()
        frontier = list(types)
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            stated = self.globally_named.get(current)
            if stated is not None:
                if stated:
                    found.add(current)
                continue
            frontier.extend(self.parents.get(current, ()))
        return frozenset(found)

    def keys_for(self, types: Iterable[str]) -> list[tuple[str, tuple[str, ...]]]:
        """Each ``(class, properties)`` key that applies to an instance of ``types``."""
        classes: set[str] = set()
        for type_iri in types:
            classes |= self.ancestors(type_iri)
        return sorted(
            (cls_iri, props)
            for cls_iri in classes
            for props in self.keys.get(cls_iri, ())
        )

    def are_disjoint(
        self, left_types: Iterable[str], right_types: Iterable[str]
    ) -> bool:
        """Whether some class of one typing is declared disjoint with one of the other."""
        if not self.disjoint:
            return False
        left = set().union(*(self.ancestors(t) for t in left_types))
        right = set().union(*(self.ancestors(t) for t in right_types))
        return any(frozenset((a, b)) in self.disjoint for a in left for b in right)


def _as_bool(value: Literal) -> bool | None:
    python = value.toPython()
    if isinstance(python, bool):
        return python
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None
