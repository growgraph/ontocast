"""Route each prepared unit before extraction: extract, citations, non-content or skip.

A section label is evidence, not a decision. A unit is dropped only when a
**strong** label and the unit's content agree, or when the content alone is
conclusive; anything less is extracted. Selecting sections (``target_sections``)
is the caller's intent and happens earlier, on labels of any strength.

Each unit carries its label's strength, set where the label was assigned
(:mod:`ontocast.tool.chunk.assignment`): strong for an anchored heading, a
heading inside a strong section by its number or level, or the content itself;
weak otherwise. A weak label may help selection, never a drop.

Rules, first match wins:

1. a unit shorter than ``CHUNK_MIN_UNIT_CHARS`` is skipped;
2. a strong label the request's ``exclude_sections`` names is skipped;
3. unless ``CHUNK_BIBLIOGRAPHY_MODE=domain_facts``, a reference list -- by
   content alone, or by a strong citations-class label the content agrees
   with -- takes the bibliography route;
4. with the schema's default exclusions in force (no ``exclude_sections``, a
   schema that was chosen rather than fallen back to), a strong
   boilerplate-class label on text that states no measurement is skipped;
5. front/back matter -- a strong non-content-class label, or a heading the
   schema names as front/back matter, on text that states no measurement, or
   identifier and licence boilerplate recognised from the content alone --
   takes the non-content route. A fallback schema lends no vocabulary;
6. everything else is extracted, with a reason saying why a label that might
   have dropped it did not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ontocast.config.section_labels import (
    RouteClass,
    SectionLabelSchema,
    label_route_class,
)
from ontocast.onto.enum import LabelStrength, RouteReason, UnitRoute
from ontocast.tool.chunk.bibliography import (
    LABEL_AGREEMENT_SHARE,
    citation_share,
    looks_like_bibliography,
)
from ontocast.tool.chunk.non_content import is_non_content_unit, states_measurement
from ontocast.tool.chunk.structure import BlockKinds

if TYPE_CHECKING:
    from ontocast.config import ChunkConfig


@dataclass(frozen=True)
class RoutingDecision:
    """A unit's route and the reason for it."""

    route: UnitRoute
    reason: RouteReason


class RoutableUnit(Protocol):
    """What routing reads from a unit."""

    @property
    def text(self) -> str: ...

    @property
    def headings(self) -> list[str] | None: ...

    @property
    def section_label(self) -> str | None: ...

    @property
    def section_label_strength(self) -> LabelStrength | None: ...


def route_unit(
    unit: RoutableUnit,
    *,
    schema: SectionLabelSchema | None,
    schema_fallback: bool,
    exclude_sections: list[str] | None,
    config: ChunkConfig,
    block_kinds: BlockKinds | None = None,
) -> RoutingDecision:
    """Decide one unit's route from its label, the label's strength and its content.

    Args:
        unit: A prepared unit.
        schema: The schema the document was labelled against; ``None`` when
            section classification is off, which leaves the content rules only.
        schema_fallback: The schema is the manifest default because nothing
            chose one; its default exclusions do not apply.
        exclude_sections: The request's denylist: ``None`` keeps the schema's
            boilerplate exclusions, a list (``[]`` included) replaces them.
        config: Chunk configuration (bibliography and non-content modes, the
            unit-size floor).
        block_kinds: Docling block kinds of the document, when it was
            converted: a caption or formula is never a citation entry.

    Returns:
        The route and its reason.
    """
    text = unit.text
    if config.min_unit_chars and len(text) < config.min_unit_chars:
        return RoutingDecision(UnitRoute.SKIP, RouteReason.UNDERSIZED)

    label = unit.section_label.strip().lower() if unit.section_label else None
    strong = unit.section_label_strength is LabelStrength.STRONG
    route_class = (
        label_route_class(label, schema) if schema is not None else RouteClass.CONTENT
    )

    if exclude_sections is not None and label is not None:
        denied = {section.strip().lower() for section in exclude_sections}
        if label in denied and strong:
            return RoutingDecision(UnitRoute.SKIP, RouteReason.EXCLUDED_BY_REQUEST)

    if config.bibliography_mode != "domain_facts":
        reason: RouteReason | None = None
        if looks_like_bibliography(text, block_kinds):
            reason = RouteReason.CITATION_LIST
        elif (
            route_class is RouteClass.CITATIONS
            and strong
            and citation_share(text, block_kinds) >= LABEL_AGREEMENT_SHARE
        ):
            reason = RouteReason.CITATION_LABEL
        if reason is not None:
            route = (
                UnitRoute.SKIP
                if config.bibliography_mode == "skip"
                else UnitRoute.CITATIONS
            )
            return RoutingDecision(route, reason)

    if route_class is RouteClass.BOILERPLATE and strong:
        if (
            exclude_sections is None
            and not schema_fallback
            and not states_measurement(text)
        ):
            return RoutingDecision(UnitRoute.SKIP, RouteReason.BOILERPLATE)

    vocabulary = None if schema_fallback else schema
    strong_non_content = (
        route_class is RouteClass.NON_CONTENT
        and strong
        and not schema_fallback
        and not states_measurement(text)
    )
    if strong_non_content or is_non_content_unit(text, unit.headings, vocabulary):
        route = (
            UnitRoute.SKIP
            if config.non_content_mode == "skip"
            else UnitRoute.NON_CONTENT
        )
        return RoutingDecision(route, RouteReason.NON_CONTENT)

    return RoutingDecision(
        UnitRoute.EXTRACT,
        _extract_reason(
            route_class,
            strong=strong,
            schema_fallback=schema_fallback,
            exclusions_replaced=exclude_sections is not None,
            denied=label is not None
            and exclude_sections is not None
            and label in {s.strip().lower() for s in exclude_sections},
        ),
    )


def _extract_reason(
    route_class: RouteClass,
    *,
    strong: bool,
    schema_fallback: bool,
    exclusions_replaced: bool,
    denied: bool,
) -> RouteReason:
    """Why a unit is extracted although a label might have dropped it."""
    if denied and not strong:
        return RouteReason.WEAK_LABEL
    if route_class is RouteClass.CONTENT:
        return RouteReason.CONTENT
    if not strong:
        return RouteReason.WEAK_LABEL
    if route_class is RouteClass.BOILERPLATE and exclusions_replaced:
        return RouteReason.EXCLUSION_OVERRIDDEN
    if route_class is not RouteClass.CITATIONS and schema_fallback:
        return RouteReason.FALLBACK_SCHEMA
    return RouteReason.LABEL_CONTRADICTED


__all__ = [
    "RouteReason",
    "RoutingDecision",
    "UnitRoute",
    "route_unit",
]
