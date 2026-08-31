"""Render the deployment's SHACL shapes as a prompt chapter.

The shapes partition already judges every extracted graph; until now the
renderer was graded against a rulebook it was never shown, and the dominant
violation classes were exactly the rules no prompt stated. This module makes
the contract symmetric with the ontology chapter: whatever shapes the
deployment loaded are summarized into prose the renderer (and critic) see.

The module is domain-agnostic by construction -- it contains no vocabulary
of its own. Every line of the rendered chapter comes from the deployment's
shapes graph, preferring the shape author's ``sh:message`` (which states the
rule in prose, SPARQL constraints included) and synthesizing a line from the
constraint structure only when no message exists. A SPARQL constraint
without a message cannot be summarized mechanically and is omitted with a
warning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from rdflib import RDF, Graph, URIRef
from rdflib.namespace import SH

logger = logging.getLogger(__name__)

CHAPTER_HEADING = "# CONFORMANCE REQUIREMENTS"

_CHAPTER_PREAMBLE = (
    "The output graph is validated against these structural rules. When you "
    "instantiate one of the classes below, satisfy its rules in the same "
    "render -- a missing required property is a defect, not an omission to "
    "fill in later."
)

_CHAPTER_CLOSING = (
    "Only instantiate these structures for content actually stated in the "
    "text. Never mint a placeholder node or invent a property value to "
    "satisfy a rule."
)


@dataclass(frozen=True)
class ShapeRequirement:
    """One node shape, reduced to a target label and its rule lines.

    ``terms`` are the shape's own contract IRIs (targets, paths, classes,
    datatypes, ``sh:in`` members). They serve two consumers: the union over
    all requirements is the ``UNKNOWN_TERM`` exemption set, and the per-shape
    set is what context-join selection intersects with a unit's resolved
    ontology snapshot.
    """

    anchor: str
    lines: tuple[str, ...]
    terms: tuple[str, ...] = ()
    closed: bool = False


def _curie(graph: Graph, term) -> str:
    if not isinstance(term, URIRef):
        return str(term)
    try:
        prefix, _, local = graph.compute_qname(term, generate=False)
        return f"{prefix}:{local}"
    except Exception:  # noqa: BLE001 - unbindable IRIs render absolute
        return str(term)


def _synthesized_line(graph: Graph, prop) -> str | None:
    """A rule line composed from constraint structure, for message-less shapes."""
    path = graph.value(prop, SH.path)
    if path is None or not isinstance(path, URIRef):
        return None
    parts = []
    min_count = graph.value(prop, SH.minCount)
    max_count = graph.value(prop, SH.maxCount)
    klass = graph.value(prop, SH["class"])
    datatype = graph.value(prop, SH.datatype)
    if min_count is not None:
        parts.append(f"at least {min_count}")
    if max_count is not None:
        parts.append(f"at most {max_count}")
    if klass is not None:
        parts.append(f"of type {_curie(graph, klass)}")
    if datatype is not None:
        parts.append(f"as {_curie(graph, datatype)}")
    if not parts:
        return None
    return f"{_curie(graph, path)}: " + ", ".join(parts)


def derive_shape_requirements(shapes_graph: Graph) -> list[ShapeRequirement]:
    """Reduce every node shape to an anchor and prose rule lines.

    Preference order per constraint: the author's ``sh:message`` verbatim; a
    synthesized structural line; nothing (SPARQL constraints without a
    message, warned once per derivation).
    """
    requirements: list[ShapeRequirement] = []
    unrenderable = 0
    for shape in sorted(shapes_graph.subjects(RDF.type, SH.NodeShape), key=str):
        anchors = [
            _curie(shapes_graph, t) for t in shapes_graph.objects(shape, SH.targetClass)
        ]
        anchors.extend(
            f"subjects of {_curie(shapes_graph, p)}"
            for p in shapes_graph.objects(shape, SH.targetSubjectsOf)
        )
        anchors.extend(
            f"objects of {_curie(shapes_graph, p)}"
            for p in shapes_graph.objects(shape, SH.targetObjectsOf)
        )
        if not anchors:
            continue
        terms: set[str] = set()
        for predicate in (SH.targetClass, SH.targetSubjectsOf, SH.targetObjectsOf):
            terms.update(
                str(t)
                for t in shapes_graph.objects(shape, predicate)
                if isinstance(t, URIRef)
            )
        lines: list[str] = []
        for prop in shapes_graph.objects(shape, SH.property):
            for predicate in (SH.path, SH["class"], SH.datatype):
                value = shapes_graph.value(prop, predicate)
                if isinstance(value, URIRef):
                    terms.add(str(value))
            in_list = shapes_graph.value(prop, SH["in"])
            if in_list is not None:
                terms.update(
                    str(member)
                    for member in shapes_graph.items(in_list)
                    if isinstance(member, URIRef)
                )
            message = shapes_graph.value(prop, SH.message)
            if message is not None:
                lines.append(str(message))
                continue
            synthesized = _synthesized_line(shapes_graph, prop)
            if synthesized is not None:
                lines.append(synthesized)
        for sparql in shapes_graph.objects(shape, SH.sparql):
            message = shapes_graph.value(sparql, SH.message) or shapes_graph.value(
                shape, SH.message
            )
            if message is not None:
                lines.append(str(message))
            else:
                unrenderable += 1
        closed = bool(shapes_graph.value(shape, SH.closed))
        if closed:
            lines.append("Do not attach properties beyond the ones above.")
        if lines:
            requirements.append(
                ShapeRequirement(
                    anchor=" / ".join(anchors),
                    lines=tuple(dict.fromkeys(lines)),
                    terms=tuple(sorted(terms)),
                    closed=closed,
                )
            )
    if unrenderable:
        logger.warning(
            "Shapes prompt contract: %d SPARQL constraint(s) carry no "
            "sh:message and cannot be summarized; they are validated but "
            "not shown to the renderer",
            unrenderable,
        )
    return requirements


def format_conformance_chapter(
    requirements: Sequence[ShapeRequirement],
    *,
    max_lines: int = 40,
) -> str:
    """The prompt chapter, or "" when there is nothing to state.

    ``max_lines`` caps the total rule lines (the chapter is run-constant, so
    the cap is a prompt-size guard, not a relevance ranking); truncation is
    noted in the chapter itself so the model does not read the tail rules'
    absence as their nonexistence.
    """
    if not requirements:
        return ""
    body: list[str] = []
    used = 0
    truncated = False
    for requirement in requirements:
        remaining = max_lines - used
        if remaining <= 0:
            truncated = True
            break
        lines = requirement.lines[:remaining]
        if len(lines) < len(requirement.lines):
            truncated = True
        used += len(lines)
        body.append(f"- {requirement.anchor}:")
        body.extend(f"  - {line}" for line in lines)
    chapter = "\n".join(
        [CHAPTER_HEADING, _CHAPTER_PREAMBLE, "", *body, "", _CHAPTER_CLOSING]
    )
    if truncated:
        chapter += "\n(Further rules exist; validation checks all of them.)"
    return chapter


def select_requirements(
    requirements: Sequence[ShapeRequirement],
    context_terms: set[str],
) -> list[ShapeRequirement]:
    """The requirements whose terms intersect a unit's ontology context.

    Shape relevance is derivative of ontology-term relevance: a shape
    constrains classes and properties, and a unit can only instantiate the
    ones its resolved snapshot carries — so the snapshot's IRIs are the join
    key, and no separate retrieval decision is needed. ``context_terms``
    should be every IRI of the snapshot graph (subjects, predicates and
    objects: the schema closure carries superclass IRIs as objects, which is
    how a shape targeting a superclass joins a unit typed with the
    subclass). Order is preserved.
    """
    return [r for r in requirements if context_terms.intersection(r.terms)]


def contract_terms(shapes_graph: Graph) -> tuple[str, ...]:
    """Every IRI the shapes require the output to use.

    These join the deterministic validator's exempt set: a term the
    conformance chapter instructs the renderer to emit must not be flagged
    UNKNOWN_TERM when the unit's retrieved ontology context happens not to
    contain it -- otherwise the repair lane orders removal of exactly what
    the chapter required.
    """
    terms: set[str] = set()
    for predicate in (SH.targetClass, SH.targetSubjectsOf, SH.targetObjectsOf):
        terms.update(
            str(t)
            for t in shapes_graph.objects(None, predicate)
            if isinstance(t, URIRef)
        )
    for prop in shapes_graph.objects(None, SH.property):
        for predicate in (SH.path, SH["class"], SH.datatype):
            value = shapes_graph.value(prop, predicate)
            if isinstance(value, URIRef):
                terms.add(str(value))
        in_list = shapes_graph.value(prop, SH["in"])
        if in_list is not None:
            for member in shapes_graph.items(in_list):
                if isinstance(member, URIRef):
                    terms.add(str(member))
    return tuple(sorted(terms))
