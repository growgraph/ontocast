"""Embedding heading tier: a heading no pattern or keyword named votes for its nearest label.

Each section-like heading left unnamed by the lexical tiers is embedded with
the chunker's model and compared with every label's prototypes (the label
name and its keywords). It takes the nearest label when that label is close
enough and clearly ahead of the runner-up; otherwise it stays unresolved. The
vote is weak: it can select a section but never drop one. Paraphrases and
translations of a section name ("Closing Thoughts", "Schlussfolgerung") are
what it recovers; topical headings sit far from every label and abstain.

It names content sections only. A heading nearest a boilerplate, back-matter
or citations label abstains: a weak label of that kind can neither drop nor
flag a unit, so it could only mislabel the content under a banner heading
("Access" is near "open access").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ontocast.config.section_labels import RouteClass
from ontocast.onto.enum import LabelStrength, SectionLabelSource
from ontocast.tool.chunk.embedding import cosine, unit_vector

if TYPE_CHECKING:
    from ontocast.config.section_labels import SectionLabelSchema
    from ontocast.onto.section_models import DocumentOutline
    from ontocast.tool.chunk.embedding import TextEmbedder

logger = logging.getLogger(__name__)

#: Least cosine similarity between a heading and its nearest label prototype.
HEADING_MIN_SIMILARITY = 0.70
#: Least lead of the nearest label over the runner-up label.
HEADING_MIN_MARGIN = 0.10


def label_prototypes(schema: SectionLabelSchema) -> list[tuple[str, str]]:
    """``(label, text)`` pairs a heading is compared with: each name and keyword."""
    return [
        (label.id, text)
        for label in schema.labels
        for text in dict.fromkeys([label.id.replace("_", " "), *label.keywords])
    ]


def nearest_label(
    vector: list[float], prototypes: list[tuple[str, list[float]]]
) -> str | None:
    """The label nearest ``vector``, or ``None`` when none is near and clear.

    Args:
        vector: Unit embedding of a heading.
        prototypes: ``(label, unit embedding)`` pairs.

    Returns:
        The nearest label when its similarity reaches
        :data:`HEADING_MIN_SIMILARITY` and leads every other label by
        :data:`HEADING_MIN_MARGIN`.
    """
    best: dict[str, float] = {}
    for label, prototype in prototypes:
        similarity = cosine(vector, prototype)
        if similarity > best.get(label, -1.0):
            best[label] = similarity
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None
    label, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
    if top < HEADING_MIN_SIMILARITY or top - runner_up < HEADING_MIN_MARGIN:
        return None
    return label


def embed_heading_labels(
    outline: DocumentOutline, schema: SectionLabelSchema, embed: TextEmbedder
) -> int:
    """Name unresolved section-like headings by their nearest label (mutates in place).

    Runs after the lexical tiers and before inheritance, so the headings a
    tree places under a heading named here inherit its label, weakly and as
    this tier's vote. A heading whose nearest label is not a content label
    abstains.

    Returns:
        How many headings were named.
    """
    nodes = [node for node in outline.nodes if node.label is None and node.sectionlike]
    content = {
        label.id for label in schema.labels if label.route_class is RouteClass.CONTENT
    }
    if not nodes:
        return 0
    prototypes = label_prototypes(schema)
    vectors = embed(
        [text for _, text in prototypes] + [node.normalised for node in nodes]
    )
    if vectors is None:
        return 0
    prototype_vectors = [
        (label, unit_vector(vector))
        for (label, _), vector in zip(prototypes, vectors[: len(prototypes)])
    ]
    named = 0
    for node, vector in zip(nodes, vectors[len(prototypes) :]):
        label = nearest_label(unit_vector(vector), prototype_vectors)
        if label not in content:
            continue
        node.label = label
        node.source = SectionLabelSource.HEADING_EMBEDDING
        node.strength = LabelStrength.WEAK
        named += 1
    if named:
        logger.debug("Embedding tier named %s heading(s)", named)
    return named


__all__ = [
    "HEADING_MIN_MARGIN",
    "HEADING_MIN_SIMILARITY",
    "embed_heading_labels",
    "label_prototypes",
    "nearest_label",
]
