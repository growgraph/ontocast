"""Best-effort condensing of an ontology graph before it becomes prompt text.

Only the vector-retrieval mode ever bounded how much ontology reached the LLM.
``selected_single_ontology`` (the default) and ``fixed_single_ontology`` serialize
the whole selected catalog ontology into every prompt, and the facts fan-out
serializes the union of every ontology artifact -- all with no cap, no sampling
and no truncation. On a large catalog that is the context blow-up; nothing else
in the pipeline notices.

This module trims a graph toward a triple budget by dropping the
least-load-bearing triples first, in the order established by
:data:`~ontocast.onto.graph_prune.BFS_PREDICATE_PRIORITY`. It is deliberately
**best-effort**: it will never drop labels, types, hierarchy or domain/range to
hit a number. A budget that cannot be met without cutting into those is reported
as a warning and the graph is passed through oversized, because silently
removing the schema the model needed produces a bad extraction that looks like a
bad model -- the most expensive failure mode this pipeline has.
"""

import logging

from pydantic import BaseModel, Field
from rdflib import RDFS, SKOS, Literal, URIRef

from ontocast.onto.graph_prune import (
    BFS_PREDICATE_PRIORITY,
    NOISY_EXPANSION_PREDICATES,
    prune_degenerate_restriction_bnodes,
    prune_orphaned_bnode_subjects,
    strip_redundant_generic_types,
)
from ontocast.onto.rdfgraph import RDFGraph

logger = logging.getLogger(__name__)

#: Predicates carrying human-facing description rather than structure. Last tier
#: of the shared priority ordering, and the only tier this module will drop.
GLOSS_PREDICATES: frozenset[URIRef] = BFS_PREDICATE_PRIORITY[-1]

#: Tiers that are never dropped: a term the model cannot name, type, place in the
#: hierarchy, or connect is not usable context -- it is an invitation to invent.
LOAD_BEARING_PREDICATES: frozenset[URIRef] = frozenset().union(
    *BFS_PREDICATE_PRIORITY[:-1]
)


#: Marker left where a literal was cut, so the model can tell a clipped
#: definition from a complete one.
CLIP_MARKER = "\u2026"

#: Text roles, by predicate. Naming carries the surface forms the model matches
#: document text against; contract carries the usage rules a term is only safe
#: to apply under; prose is description a reader wants and an extractor does not.
NAMING_TEXT_PREDICATES: frozenset[URIRef] = frozenset(
    {RDFS.label, SKOS.prefLabel, SKOS.altLabel}
)
CONTRACT_TEXT_PREDICATES: frozenset[URIRef] = frozenset(
    {SKOS.scopeNote, SKOS.definition}
)
PROSE_TEXT_PREDICATES: frozenset[URIRef] = frozenset(
    {RDFS.comment, SKOS.example, SKOS.note, SKOS.editorialNote, SKOS.historyNote}
)


class TextCaps(BaseModel):
    """Per-role character caps on the text literals reaching a prompt.

    Nothing else in the pipeline bounds a single literal, so chapter size is
    otherwise proportional to how chatty a catalog's authors were rather than to
    how many terms it offers. These caps make it proportional to the term count,
    which the retrieval budget already controls. On a tersely authored catalog
    they are a no-op; that is the intended shape -- a bound, not a reduction.

    Clipping rather than dropping is what keeps a usage contract available at a
    predictable price: the first sentence of a scope note is the part that says
    when a term applies.
    """

    naming: int | None = Field(
        default=None,
        ge=1,
        description="Cap on rdfs:label / skos:prefLabel / skos:altLabel.",
    )
    contract: int | None = Field(
        default=None,
        ge=1,
        description="Cap on skos:scopeNote / skos:definition.",
    )
    prose: int | None = Field(
        default=None, ge=1, description="Cap on rdfs:comment and other notes."
    )
    total_budget: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Ceiling on the summed length of all text literals in the chapter. "
            "Backstop for a catalog that defeats the per-role caps by holding "
            "very many short terms."
        ),
    )

    @property
    def active(self) -> bool:
        """Whether any cap is set at all."""
        return any(
            cap is not None
            for cap in (self.naming, self.contract, self.prose, self.total_budget)
        )

    def cap_for(self, predicate: URIRef) -> int | None:
        """The cap governing ``predicate``, or None when it governs no role."""
        if predicate in NAMING_TEXT_PREDICATES:
            return self.naming
        if predicate in CONTRACT_TEXT_PREDICATES:
            return self.contract
        if predicate in PROSE_TEXT_PREDICATES:
            return self.prose
        return None


#: Progressive tightening applied when the per-role caps still leave the chapter
#: over ``total_budget``, in increasing order of harm; ``None`` drops the role's
#: literals outright. Naming is never dropped -- a term the model cannot name is
#: an invitation to invent one -- only clipped to a floor.
_BUDGET_STAGES: tuple[tuple[frozenset[URIRef], int | None], ...] = (
    (PROSE_TEXT_PREDICATES, 80),
    (PROSE_TEXT_PREDICATES, None),
    (CONTRACT_TEXT_PREDICATES, 120),
    (CONTRACT_TEXT_PREDICATES, None),
    (NAMING_TEXT_PREDICATES, 48),
)

#: Predicates whose literals the caps govern at all.
_CAPPED_PREDICATES: frozenset[URIRef] = (
    NAMING_TEXT_PREDICATES | CONTRACT_TEXT_PREDICATES | PROSE_TEXT_PREDICATES
)


def clip_text(text: str, cap: int | None) -> str:
    """Clip ``text`` to ``cap`` characters on a word boundary, marking the cut.

    The retained text is at most ``cap`` characters; :data:`CLIP_MARKER` is
    appended on top of it, so a clipped literal reads as clipped. A ``cap`` of
    None, or text already within it, is returned unchanged -- byte-identical, so
    a disabled cap cannot perturb a prompt or its cache key.

    Args:
        text: Literal text to bound.
        cap: Maximum retained characters, or None to leave ``text`` alone.

    Returns:
        Either ``text`` itself or a clipped copy ending in the marker.
    """
    if cap is None or len(text) <= cap:
        return text
    head = text[:cap].rsplit(" ", 1)[0].rstrip()
    if not head:
        head = text[:cap].rstrip()
    return head + CLIP_MARKER


def _text_triples(graph: RDFGraph) -> list[tuple]:
    """Every triple whose object is a text literal a cap governs."""
    return [
        triple
        for triple in graph
        if triple[1] in _CAPPED_PREDICATES and isinstance(triple[2], Literal)
    ]


def _text_chars(graph: RDFGraph) -> int:
    """Summed length of the capped text literals in ``graph``."""
    return sum(len(str(triple[2])) for triple in _text_triples(graph))


def _apply_cap(
    graph: RDFGraph, predicates: frozenset[URIRef], cap: int | None
) -> tuple[int, int]:
    """Clip (or drop, when ``cap`` is None) ``predicates``' literals in place.

    Returns:
        Tuple of (literals clipped, literals dropped).
    """
    clipped = dropped = 0
    for subject, predicate, obj in _text_triples(graph):
        if predicate not in predicates:
            continue
        if cap is None:
            graph.remove((subject, predicate, obj))
            dropped += 1
            continue
        text = str(obj)
        shortened = clip_text(text, cap)
        if shortened == text:
            continue
        graph.remove((subject, predicate, obj))
        graph.add(
            (
                subject,
                predicate,
                Literal(shortened, lang=obj.language)
                if obj.language
                else Literal(shortened, datatype=obj.datatype),
            )
        )
        clipped += 1
    return clipped, dropped


class CondenseReport(BaseModel):
    """What condensing did, for telemetry and for explaining a warning."""

    triples_before: int = Field(description="Triple count on entry")
    triples_after: int = Field(description="Triple count after condensing")
    max_triples: int | None = Field(
        default=None, description="Budget applied; None means no budget"
    )
    dropped_noise: int = Field(default=0, description="Header/list plumbing removed")
    dropped_structural: int = Field(
        default=0, description="Generic types, stub restrictions, orphan bnodes"
    )
    dropped_glosses: int = Field(
        default=0, description="Comments, definitions, scope notes, alt labels"
    )
    over_budget: bool = Field(
        default=False,
        description="Still above budget after condensing; passed through oversized",
    )
    text_chars_before: int = Field(
        default=0, description="Summed length of capped text literals on entry"
    )
    text_chars_after: int = Field(
        default=0, description="Summed length of capped text literals after capping"
    )
    literals_clipped: int = Field(
        default=0, description="Text literals shortened to a per-role cap"
    )
    literals_dropped: int = Field(
        default=0, description="Text literals removed to meet the total budget"
    )
    text_over_budget: bool = Field(
        default=False,
        description="Still above the total text budget after every tightening stage",
    )

    @property
    def changed(self) -> bool:
        """Whether anything was removed or shortened at all."""
        return bool(
            self.triples_after != self.triples_before
            or self.literals_clipped
            or self.literals_dropped
        )

    def as_metrics(self) -> dict[str, int | bool | None]:
        """Flat mapping for the retrieval-metrics payload."""
        return {
            "triples_before": self.triples_before,
            "triples_after": self.triples_after,
            "max_triples": self.max_triples,
            "dropped_noise": self.dropped_noise,
            "dropped_structural": self.dropped_structural,
            "dropped_glosses": self.dropped_glosses,
            "over_budget": self.over_budget,
            "text_chars_before": self.text_chars_before,
            "text_chars_after": self.text_chars_after,
            "literals_clipped": self.literals_clipped,
            "literals_dropped": self.literals_dropped,
            "text_over_budget": self.text_over_budget,
        }


def _drop_predicates(graph: RDFGraph, predicates: frozenset[URIRef]) -> int:
    removed = 0
    for triple in list(graph):
        if triple[1] in predicates:
            graph.remove(triple)
            removed += 1
    return removed


def _cap_text_literals(graph: RDFGraph, caps: TextCaps, report: CondenseReport) -> None:
    """Apply ``caps`` to ``graph`` in place, recording what happened on ``report``.

    Per-role caps first, then the total budget as a backstop: the budget stages
    tighten prose before contracts and contracts before names, and stop as soon
    as the chapter fits. A chapter that still does not fit is passed through with
    a warning rather than having its names removed, matching how the triple
    budget refuses to cut into load-bearing structure.
    """
    report.text_chars_before = _text_chars(graph)
    for predicates, cap in (
        (NAMING_TEXT_PREDICATES, caps.naming),
        (CONTRACT_TEXT_PREDICATES, caps.contract),
        (PROSE_TEXT_PREDICATES, caps.prose),
    ):
        if cap is None:
            continue
        clipped, dropped = _apply_cap(graph, predicates, cap)
        report.literals_clipped += clipped
        report.literals_dropped += dropped

    budget = caps.total_budget
    if budget is not None:
        for predicates, cap in _BUDGET_STAGES:
            if _text_chars(graph) <= budget:
                break
            clipped, dropped = _apply_cap(graph, predicates, cap)
            report.literals_clipped += clipped
            report.literals_dropped += dropped
        report.text_over_budget = _text_chars(graph) > budget

    report.text_chars_after = _text_chars(graph)
    if report.text_over_budget:
        logger.warning(
            "Ontology text still exceeds the literal budget after every "
            "tightening stage (%d > %d chars). Passing it through rather than "
            "removing the labels the model needs to name a term at all.",
            report.text_chars_after,
            budget,
        )
    elif report.changed:
        logger.info(
            "Capped ontology text %d -> %d chars (%d clipped, %d dropped).",
            report.text_chars_before,
            report.text_chars_after,
            report.literals_clipped,
            report.literals_dropped,
        )


def condense_graph_for_prompt(
    graph: RDFGraph,
    max_triples: int | None,
    text_caps: TextCaps | None = None,
) -> tuple[RDFGraph, CondenseReport]:
    """Trim ``graph`` toward ``max_triples``, dropping the least useful triples first.

    Passes are applied in increasing order of harm, stopping as soon as the graph
    fits: header/list noise, then structural scaffolding, then glosses. Structure
    that lets the model name and place a term is never dropped.

    Text literals are bounded first and unconditionally: the triple budget is a
    count and says nothing about how long a single ``rdfs:comment`` may be, so a
    graph well under it can still carry an unbounded chapter. ``text_caps`` makes
    chapter size a function of term count rather than of prose volume.

    Args:
        graph: Ontology graph destined for a prompt. Not mutated.
        max_triples: Triple budget, or ``None`` to disable condensing entirely.
        text_caps: Per-role character caps on text literals, or ``None``/all-unset
            to leave every literal as authored.

    Returns:
        The condensed graph (or ``graph`` itself when nothing was done) and a
        :class:`CondenseReport` describing what happened.
    """
    before = len(graph)
    report = CondenseReport(
        triples_before=before, triples_after=before, max_triples=max_triples
    )
    capping = text_caps is not None and text_caps.active
    fits = max_triples is None or before <= max_triples
    if fits and not capping:
        return graph, report

    working = graph.copy()
    if capping:
        assert text_caps is not None
        _cap_text_literals(working, text_caps, report)
    if fits:
        report.triples_after = len(working)
        return working, report

    report.dropped_noise = _drop_predicates(working, NOISY_EXPANSION_PREDICATES)
    if len(working) > max_triples:
        structural_before = len(working)
        strip_redundant_generic_types(working)
        prune_degenerate_restriction_bnodes(working)
        prune_orphaned_bnode_subjects(working)
        report.dropped_structural = structural_before - len(working)

    if len(working) > max_triples:
        report.dropped_glosses = _drop_predicates(working, GLOSS_PREDICATES)

    report.triples_after = len(working)
    report.over_budget = len(working) > max_triples

    if report.over_budget:
        logger.warning(
            "Ontology context still exceeds the prompt budget after condensing "
            "(%d > %d triples; was %d). Passing it through rather than dropping "
            "labels, types, hierarchy or domain/range, which the model needs to "
            "use the schema at all. Reduce the catalog, split it, or switch to "
            "ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology, which "
            "retrieves only the relevant part.",
            report.triples_after,
            max_triples,
            before,
        )
    else:
        logger.info(
            "Condensed ontology context %d -> %d triples for a %d budget "
            "(noise=%d, structural=%d, glosses=%d).",
            before,
            report.triples_after,
            max_triples,
            report.dropped_noise,
            report.dropped_structural,
            report.dropped_glosses,
        )

    return working, report
