"""Whether a value says what its text says.

A value (an instance of a value class, blank or named) states numbers and
names individuals (:func:`~ontocast.tool.facts_validation.statements.value_components`).
Two checks read it against the content unit's text:

* each number is *grounded*: written in digits in the value's label or in the
  text (:func:`~ontocast.util.numeric_inventory.stated_numbers`), as written
  or rescaled by a power of ten
  (:func:`~ontocast.util.numeric_inventory.grounds`); a number the text gives
  only in words, a fraction, a converted value, or a conventional value for a
  vague phrase, is not stated;
* each individual whose property's range is a closed class agrees with the
  catalog's *cues* written next to the value's numbers -- in the value's label
  when it writes the number, the value's own mention, else in the text: the
  ``skos:altLabel``, notation and symbol literals of that class's individuals
  (:func:`~ontocast.util.numeric_inventory.cue_index`). A cue of another
  individual, and none of its own, contradicts it, when the individual has
  cues of its own in the catalog.

Both act only on value nodes: a graph without values gets no finding, and a
catalog without cues gives no contradiction.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rdflib import RDF, RDFS, SKOS, BNode, Graph, Literal, URIRef
from rdflib.term import Node

from ontocast.onto.model import FactsUnitFinding, FactsUnitFindingKind
from ontocast.onto.scaffolding import FactScope
from ontocast.tool.facts_validation.statements import (
    StatementContract,
    value_components,
)
from ontocast.util.measurement_lexicon import is_unit_surface
from ontocast.util.numeric_inventory import cue_index, grounds, stated_number_spans

#: Predicates naming a value in words: where its verbatim text is kept.
_LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)

#: How far from a number a cue is looked for, in characters.
_CUE_WINDOW = 48


def node_key(node: Node) -> str:
    """How a finding names ``node``: its IRI, or ``_:`` and a blank node's id."""
    return f"_:{node}" if isinstance(node, BNode) else str(node)


def node_of(key: str) -> Node:
    """The node :func:`node_key` named."""
    return BNode(key[2:]) if key.startswith("_:") else URIRef(key)


def node_name(graph: Graph, node: Node) -> str:
    """How a message names ``node``: its IRI, or a blank value by its carrier.

    A blank node's id differs from run to run, so a value is named by the
    first carrier and property that point at it, in IRI order, up to the
    first named carrier.
    """
    path: list[str] = []
    seen: set[Node] = set()
    while isinstance(node, BNode) and node not in seen:
        seen.add(node)
        carriers = sorted(
            (
                (carrier, predicate)
                for carrier, predicate in graph.subject_predicates(node)
                if carrier not in seen
            ),
            key=lambda pair: (isinstance(pair[0], BNode), str(pair[0]), str(pair[1])),
        )
        if not carriers:
            return "a blank value node" + "".join(reversed(path))
        node, predicate = carriers[0]
        path.append(f" on <{predicate}>")
    if isinstance(node, BNode):
        return "a blank value node" + "".join(reversed(path))
    if not path:
        return f"<{node}>"
    return f"the value of <{node}>" + "".join(reversed(path))


def value_nodes(scope: FactScope, contract: StatementContract) -> list[Node]:
    """The fact nodes typed with a value class, blank or named, in a stable order."""
    if not contract.value_classes:
        return []
    found = {
        node
        for node, cls in scope.graph.subject_objects(RDF.type)
        if str(cls) in contract.value_classes and scope.holds(node)
    }
    return sorted(found, key=lambda node: (isinstance(node, BNode), str(node)))


def _label_text(graph: Graph, node: Node) -> str:
    return "\n".join(
        str(value)
        for predicate in _LABEL_PREDICATES
        for value in graph.objects(node, predicate)
        if isinstance(value, Literal)
    )


#: Characters that end a sentence or a clause: a cue followed by one ends there.
_CLOSING = frozenset(".,;:!?)]}\"'»”’")
#: The first word after a cue and its spacing.
_NEXT_WORD = re.compile(r"\s+(\S+)")


def _ends_token(text: str, cues: Sequence[str]) -> bool:
    """Whether a cue followed by ``text`` is a whole token on its own.

    It is at the end of the text, before closing punctuation, or before
    white space and a word that is no further unit token ("5 m long", not
    "5 m s-2"). A cue run on by a factor, an exponent, a digit or a sign
    ("m/s", "m2", "m-thick") is part of a compound.
    """
    if not text or text[0] in _CLOSING:
        return True
    if not text[0].isspace():
        return False
    match = _NEXT_WORD.match(text)
    if match is None:
        return True
    word = match.group(1).rstrip("".join(_CLOSING))
    return not word or not is_unit_surface(word, cues)


def _adjacent_cues(
    text: str, spans: Sequence[tuple[int, int]], cues: Sequence[str]
) -> set[str]:
    """The cues written right before or right after one of ``spans``.

    Only white space may stand between a cue and the number. A cue after the
    number counts only as a whole token (:func:`_ends_token`). A cue before
    the number starting or ending in a letter or digit must be a whole token
    there, and does not count when a number stands right before it: it is
    that number's.
    """
    found: set[str] = set()
    for start, end in spans:
        before = text[max(0, start - _CUE_WINDOW) : start].rstrip()
        after = text[end : end + _CUE_WINDOW].lstrip()
        for cue in cues:
            if after.startswith(cue) and _ends_token(after[len(cue) :], cues):
                found.add(cue)
            if before.endswith(cue):
                rest = before[: len(before) - len(cue)]
                if cue[0].isalnum() and rest and rest[-1].isalnum():
                    continue
                if rest.rstrip()[-1:].isdigit():
                    continue
                found.add(cue)
    return found


def value_grounding_findings(
    scope: FactScope,
    contract: StatementContract,
    text: str,
    ontology_graph: Graph | None,
) -> list[FactsUnitFinding]:
    """Ungrounded numbers and contradicted individuals of the values in ``scope``.

    ``ungrounded_number``: a number of a value that neither the value's label
    nor ``text`` writes in digits, as written or rescaled by a power of ten
    (``1e3`` is grounded by "1000", ``0.05`` by "5%"). Its
    ``subject`` is the value (:func:`node_key`), ``predicate`` and ``value``
    the number's property and lexical form.

    ``contradicted_individual``: an individual of a value, on a property whose
    range is a closed class, when a cue of another individual of that class,
    and none of its own, is written next to one of the value's numbers: in
    the value's label when the label writes the number, else in the text,
    and the individual has cues of its own. Its ``value`` is the individual,
    its suggestions the individuals the cue names. No individual is ever
    proposed for a value that names none.

    Needs the unit's ``text``: with none, nothing is judged.

    Args:
        scope: The unit's fact scope.
        contract: The catalog's statement contract (value classes, closed
            ranges).
        text: The content unit's text.
        ontology_graph: The unit's ontology context, where cues are read.

    Returns:
        Mandatory findings, ordered by value then property.
    """
    if not text.strip():
        return []
    graph = scope.graph
    text_spans = stated_number_spans(text)
    text_numbers = {value for value, _, _ in text_spans}
    findings: list[FactsUnitFinding] = []
    for node in value_nodes(scope, contract):
        components = value_components(graph, node, contract)
        if not components.numbers:
            continue
        label = _label_text(graph, node)
        label_spans = stated_number_spans(label)
        stated = text_numbers | {value for value, _, _ in label_spans}
        name = node_name(graph, node)
        magnitudes: set[str] = set()
        for number in components.numbers:
            magnitude = (number.number or "").lstrip("-")
            magnitudes.add(magnitude)
            if magnitude in stated or grounds(magnitude, stated):
                continue
            findings.append(
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.UNGROUNDED_NUMBER,
                    message=(
                        f"{name} records {number.value} on <{number.predicate}>, "
                        "a number neither its label nor the text writes. Record "
                        "only a number the text states, as written; a quantity "
                        "the text gives in words stays in the label."
                    ),
                    subject=node_key(node),
                    predicate=number.predicate,
                    value=str(number.value),
                )
            )
        for individual in components.individuals:
            if not individual.closed_range:
                continue
            cues = cue_index(ontology_graph, individual.closed_range)
            if not cues:
                continue
            # The label is the value's own mention when it writes the number;
            # elsewhere in the text the same number may belong to another one.
            in_label = [(s, e) for value, s, e in label_spans if value in magnitudes]
            adjacent = (
                _adjacent_cues(label, in_label, tuple(cues))
                if in_label
                else _adjacent_cues(
                    text,
                    [(s, e) for value, s, e in text_spans if value in magnitudes],
                    tuple(cues),
                )
            )
            carried = str(individual.value)
            if not adjacent or any(carried in cues[cue] for cue in adjacent):
                continue
            # An individual the catalog gives no cue can be named by none.
            if not any(carried in iris for iris in cues.values()):
                continue
            named = sorted({iri for cue in adjacent for iri in cues[cue]})
            written = ", ".join(f"'{cue}'" for cue in sorted(adjacent))
            findings.append(
                FactsUnitFinding(
                    kind=FactsUnitFindingKind.CONTRADICTED_INDIVIDUAL,
                    message=(
                        f"{name} names <{carried}> on <{individual.predicate}>, "
                        f"but the text writes {written} next to its number, "
                        "which the catalog declares for "
                        + ", ".join(f"<{iri}>" for iri in named)
                        + ". Name the individual the text's cue declares, or "
                        "none."
                    ),
                    subject=node_key(node),
                    predicate=individual.predicate,
                    value=carried,
                    suggestions=named,
                )
            )
    return findings


__all__ = [
    "node_key",
    "node_name",
    "node_of",
    "value_grounding_findings",
    "value_nodes",
]
