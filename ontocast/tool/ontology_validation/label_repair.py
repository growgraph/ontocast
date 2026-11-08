"""LLM-free labels for newly minted ontology terms.

A minted term the model left unlabelled almost always has a readable local
name (``PerovskiteSample``, ``hasPLRedShift``). Its label is that name split
into words, so deriving it costs nothing and spares the critic a call whose
only fix would be the label. An opaque local name (``Q42``, a hash) carries no
words; it is left unlabelled and the ``missing_label`` finding still fires.
"""

from __future__ import annotations

import re

from rdflib import RDFS, SKOS, Literal, URIRef

from ontocast.onto.iri_policy import is_opaque_local_name
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation.terms import _local_name
from ontocast.tool.ontology_validation.unit_findings import (
    _declared_class_or_property_subjects,
)

#: An acronym before a capitalised word (``PL`` in ``PLRedShift``), a word,
#: a run of capitals, or a number.
_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)


def label_from_local_name(local: str) -> str | None:
    """A readable label for a local name, or ``None`` when it has no words.

    Acronyms keep their capitals, other words are lower-cased, and a name that
    starts with a capital (a class) keeps a capital first letter:
    ``PerovskiteSample`` gives "Perovskite sample", ``hasPLRedShift`` gives
    "has PL red shift".
    """
    if not local or is_opaque_local_name(local):
        return None
    words: list[str] = []
    for chunk in re.split(r"[_\-\s]+", local):
        words.extend(_WORD_RE.findall(chunk))
    if not words or not any(word.isalpha() for word in words):
        return None
    text = " ".join(
        word if word.isupper() and len(word) > 1 else word.lower() for word in words
    )
    if local[0].isupper():
        text = text[0].upper() + text[1:]
    return text


def _has_label(subject: URIRef, *graphs: RDFGraph | None) -> bool:
    return any(
        True
        for graph in graphs
        if graph is not None
        for predicate in _LABEL_PREDICATES
        for _ in graph.objects(subject, predicate)
    )


def derive_missing_labels(
    inserts: RDFGraph, snapshot_graph: RDFGraph | None
) -> RDFGraph:
    """The ``rdfs:label`` triples the delta's unlabelled terms are missing.

    Covers every class or property the delta declares that has no label in the
    delta or the snapshot and whose local name has words.

    Returns:
        A graph of the label triples to insert, empty when none is missing.
    """
    labels = RDFGraph()
    for subject in sorted(_declared_class_or_property_subjects(inserts), key=str):
        if _has_label(subject, inserts, snapshot_graph):
            continue
        label = label_from_local_name(_local_name(str(subject)))
        if label is not None:
            labels.add((subject, RDFS.label, Literal(label, lang="en")))
    return labels


__all__ = ["derive_missing_labels", "label_from_local_name"]
