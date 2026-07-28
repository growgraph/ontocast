"""Ground-truth sets and funnel accounting for ontology retrieval recall.

Retrieval quality was previously unmeasured: every dense embedding in the suite is a
hash-based fake, so no test could tell whether a relevant catalog term actually reached
the prompt snapshot. This module supplies the missing ``{text -> expected IRIs}`` maps
plus the per-stage accounting needed to attribute a miss to a specific pipeline stage.

Two ground-truth tiers:

* **Text2KGBench** — an external corpus (see ``ontocast-validation``) where every row
  pairs a sentence with the triples it expresses. Each triple's ``rel`` is an
  ``rdfs:label`` of an ontology term, so the sentence-to-IRI map falls out by
  construction with no hand labelling. Large enough (29 ontologies, ~13.5k relation
  mentions) to discriminate between embedding models and to stress multi-ontology
  allocation.
* **Synthetic anchors** — in-repo fixtures whose labels contain nonsense phrases that
  also appear verbatim in the paired document. Small and near-verbatim, so they mostly
  exercise plumbing and the lexical lane, but they need no external data and are
  deterministic.

Retrieval is scored permissively: a term counts as retrieved when *any* IRI carrying the
expected label is found. Text2KGBench labels are occasionally shared by a class and a
property (e.g. ``composer``), and surfacing either means retrieval did its job; requiring
class/property disambiguation would measure something else.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdflib import RDFS

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph

DEFAULT_TEXT2KGBENCH_ROOT = Path.home() / "data" / "ontocast" / "validation"
TEXT2KGBENCH_ROOT_ENV = "ONTOCAST_RECALL_ROOT"

CORPUS_ROOT_ENV = "ONTOCAST_RECALL_CORPUS"

FIXTURES_DIR = Path(__file__).parent / "manual" / "fixtures"

_GT_TEXT_SUBDIRS = ("wikidata_tekgen", "dbpedia_webnlg")


def owner_index(ontologies: list[Ontology]) -> list[tuple[str, str]]:
    """Namespace-to-ontology-IRI index, longest namespace first.

    Mirrors the namespace-containment step production uses to attribute an entity to a
    catalog ontology (``patch_retriever._ontology_iri_for_entity``). Duplicated rather
    than imported: the harness must keep measuring term ownership the same way even if
    the production resolver changes, otherwise a recall regression and an attribution
    change would be indistinguishable.

    Args:
        ontologies: Catalog ontologies loaded for the run.

    Returns:
        list[tuple[str, str]]: ``(namespace_stem, ontology_iri)`` pairs, longest first
        so a nested namespace wins over its parent.
    """
    pairs = {
        (ontology.namespace or ontology.iri or "").rstrip("#/"): ontology.iri
        for ontology in ontologies
        if (ontology.namespace or ontology.iri)
    }
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def owner_of(iri: str, candidates: list[tuple[str, str]]) -> str | None:
    """Resolve which ontology owns ``iri`` by namespace containment.

    Args:
        iri: Entity IRI to attribute.
        candidates: Output of :func:`owner_index`.

    Returns:
        str | None: Owning ontology IRI, or None when no namespace matches.
    """
    for namespace, ontology_iri in candidates:
        if (
            iri == namespace
            or iri.startswith(f"{namespace}#")
            or iri.startswith(f"{namespace}/")
        ):
            return ontology_iri
    return None


@dataclass(frozen=True)
class RecallCase:
    """One query with the ontology terms retrieval is expected to surface.

    Args:
        case_id: Stable identifier, used to label failures.
        text: The query text handed to the retriever.
        expected_iris: IRIs that satisfy the case; surfacing any one counts as a hit.
        ontology_iri: Catalog ontology the expected terms belong to.
    """

    case_id: str
    text: str
    expected_iris: frozenset[str]
    ontology_iri: str


@dataclass
class StageCounts:
    """Funnel counters accumulated over a run of cases.

    Field names mirror the keys already emitted by
    ``OntologyPatchRetriever.last_retrieval_metrics`` and ``SPARQLTool.last_finalize_metrics``
    so the harness adds no new instrumentation to production code.
    """

    cases: int = 0
    seed_hits: int = 0
    snapshot_hits: int = 0
    expected_terms: int = 0
    seed_terms: int = 0
    snapshot_terms: int = 0
    atoms_after_dedupe: int = 0
    atoms_final: int = 0
    snapshot_triple_count: int = 0
    snapshot_pruned_uri_count: int = 0
    snapshot_uri_components: int = 0
    empty_snapshots: int = 0
    ontologies_per_case: int = 0
    on_topic_subjects: int = 0
    total_subjects: int = 0
    seeds_by_ontology: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    expected_terms_by_ontology: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    seed_terms_by_ontology: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    snapshot_terms_by_ontology: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def observe(
        self,
        *,
        expected: frozenset[str],
        seed_iris: set[str],
        snapshot_subjects: set[str],
        metrics: dict[str, Any],
        on_topic_subjects: int = 0,
        total_subjects: int = 0,
        expected_owner: Mapping[str, str] | None = None,
    ) -> None:
        """Fold one case's outcome and retrieval metrics into the running totals.

        Both a case-level and a term-level view are accumulated. Case level asks whether
        *any* expected term survived, which saturates once a case carries several expected
        terms; term level counts each one, and is the sensitive measure.
        """
        self.cases += 1
        self.on_topic_subjects += on_topic_subjects
        self.total_subjects += total_subjects

        seed_found = expected & seed_iris
        snapshot_found = expected & snapshot_subjects
        self.seed_hits += int(bool(seed_found))
        self.snapshot_hits += int(bool(snapshot_found))
        self.expected_terms += len(expected)
        self.seed_terms += len(seed_found)
        self.snapshot_terms += len(snapshot_found)

        # Aggregate recall hides a vocabulary that is never retrieved: a cross-cutting
        # ontology contributes few expected terms, so losing all of them barely moves
        # the total. Attribute each term to its owning ontology to expose that.
        if expected_owner:
            for iri in expected:
                owner = expected_owner.get(iri)
                if not owner:
                    continue
                self.expected_terms_by_ontology[owner] += 1
                if iri in seed_found:
                    self.seed_terms_by_ontology[owner] += 1
                if iri in snapshot_found:
                    self.snapshot_terms_by_ontology[owner] += 1

        self.atoms_after_dedupe += int(metrics.get("atoms_after_dedupe", 0))
        self.atoms_final += int(metrics.get("atoms_final", 0))
        triple_count = int(metrics.get("snapshot_triple_count", 0))
        self.snapshot_triple_count += triple_count
        if triple_count == 0:
            self.empty_snapshots += 1
        self.snapshot_pruned_uri_count += int(
            metrics.get("snapshot_pruned_uri_count", 0)
        )
        self.snapshot_uri_components += int(metrics.get("snapshot_uri_components", 0))
        by_ontology = metrics.get("seeds_by_ontology") or {}
        self.ontologies_per_case += len(by_ontology)
        for iri, count in by_ontology.items():
            self.seeds_by_ontology[iri] += int(count)

    @property
    def seed_recall(self) -> float:
        """Fraction of cases whose expected term reached the final seed set."""
        return self.seed_hits / self.cases if self.cases else 0.0

    @property
    def snapshot_recall(self) -> float:
        """Fraction of cases whose expected term is defined in the returned graph."""
        return self.snapshot_hits / self.cases if self.cases else 0.0

    @property
    def seed_term_recall(self) -> float:
        """Share of *all* expected terms that reached the final seed set."""
        return self.seed_terms / self.expected_terms if self.expected_terms else 0.0

    @property
    def snapshot_term_recall(self) -> float:
        """Share of *all* expected terms defined in the returned graph."""
        return self.snapshot_terms / self.expected_terms if self.expected_terms else 0.0

    @property
    def on_topic_precision(self) -> float:
        """Share of snapshot subjects owned by the case's own ontology.

        Recall alone cannot detect a snapshot that improved by simply including more of
        everything. This is a noise proxy, not a correctness measure: a genuinely useful
        snapshot may legitimately pull in parent terms from another ontology.
        """
        return (
            self.on_topic_subjects / self.total_subjects if self.total_subjects else 0.0
        )

    def ontologies_with_expected_but_no_seed_terms(self) -> set[str]:
        """Ontologies that were expected to contribute a term and contributed none.

        The stable signal in this harness. Recall percentages move run to run, but
        "this vocabulary was never once retrieved" does not, so it is the condition
        worth asserting on and the one that survives comparison across ablation arms.
        """
        return {
            iri
            for iri, expected in self.expected_terms_by_ontology.items()
            if expected > 0 and self.seed_terms_by_ontology.get(iri, 0) == 0
        }

    def _mean(self, total: int) -> float:
        return total / self.cases if self.cases else 0.0

    def render(self, title: str) -> str:
        """Human-readable funnel report.

        The seed-to-snapshot gap isolates damage done by induced-subgraph expansion
        (budget truncation, component pruning) from damage done by vector-stage filtering.
        """
        lines = [
            f"=== {title} ===",
            f"cases                     {self.cases}",
            f"seed recall               {self.seed_recall:.1%}  ({self.seed_hits}/{self.cases})",
            f"snapshot recall           {self.snapshot_recall:.1%}  ({self.snapshot_hits}/{self.cases})",
            # Case level saturates as soon as cases carry several expected terms; these
            # two are the numbers to compare variants on.
            f"seed TERM recall          {self.seed_term_recall:.1%}  ({self.seed_terms}/{self.expected_terms})",
            f"snapshot TERM recall      {self.snapshot_term_recall:.1%}  ({self.snapshot_terms}/{self.expected_terms})",
            # Expansion can also *recover* a term that was never a seed, by pulling it in
            # over subClassOf/domain/range, so this delta is signed rather than a pure loss.
            f"graph stage net           {self.snapshot_terms - self.seed_terms:+d} terms",
            f"on-topic precision        {self.on_topic_precision:.1%}  (snapshot subjects from the case ontology)",
            f"mean atoms_after_dedupe   {self._mean(self.atoms_after_dedupe):.1f}",
            f"mean atoms_final          {self._mean(self.atoms_final):.1f}",
            f"mean ontologies per case  {self._mean(self.ontologies_per_case):.1f}",
            f"mean snapshot triples     {self._mean(self.snapshot_triple_count):.1f}",
            f"mean pruned URIs          {self._mean(self.snapshot_pruned_uri_count):.2f}",
            f"mean URI components       {self._mean(self.snapshot_uri_components):.2f}",
            f"empty snapshots           {self.empty_snapshots}",
        ]
        if self.seeds_by_ontology:
            top = sorted(self.seeds_by_ontology.items(), key=lambda kv: -kv[1])[:8]
            lines.append("seeds by ontology (top 8):")
            lines.extend(f"    {count:6d}  {iri}" for iri, count in top)
        if self.expected_terms_by_ontology:
            lines.append("per-ontology TERM recall (expected terms owned by each):")
            lines.append(f"    {'ontology':<48}{'seed':>14}{'snapshot':>14}   status")
            ordered = sorted(
                self.expected_terms_by_ontology.items(), key=lambda kv: -kv[1]
            )
            starved = self.ontologies_with_expected_but_no_seed_terms()
            for iri, expected_count in ordered:
                seeded = self.seed_terms_by_ontology.get(iri, 0)
                snapped = self.snapshot_terms_by_ontology.get(iri, 0)
                seed_cell = f"{seeded}/{expected_count} {seeded / expected_count:.0%}"
                snap_cell = f"{snapped}/{expected_count} {snapped / expected_count:.0%}"
                status = "NO TERMS RETRIEVED" if iri in starved else ""
                short = iri if len(iri) <= 47 else "…" + iri[-46:]
                lines.append(
                    f"    {short:<48}{seed_cell:>14}{snap_cell:>14}   {status}"
                )
        return "\n".join(lines)


def _disambiguate_prefixes(graph: RDFGraph, stem: str) -> None:
    """Rebind generic author prefixes to a per-ontology unique prefix.

    Every Text2KGBench ontology binds ``onto:``, but ``OntologyManager`` registers the
    author prefix as a catalog alias and rejects a second ontology claiming one already
    bound. That is a property of this corpus rather than of the ontologies under test, so
    the collision is normalised away here instead of being worked around in the catalog.
    """
    unique = re.sub(r"[^a-z0-9]+", "", stem.lower()) or "onto"
    for prefix, namespace in list(graph.namespaces()):
        if prefix in {"onto", "ns1", ""}:
            graph.bind(unique, namespace, replace=True, override=True)


def _label_index(graph: RDFGraph) -> dict[str, set[str]]:
    """Map lowercased ``rdfs:label`` to the IRIs carrying it."""
    index: dict[str, set[str]] = defaultdict(set)
    for subject, _, obj in graph.triples((None, RDFS.label, None)):
        index[str(obj).strip().lower()].add(str(subject))
    return index


def text2kgbench_root() -> Path | None:
    """Corpus root from ``ONTOCAST_RECALL_ROOT``, else the conventional location."""
    override = os.getenv(TEXT2KGBENCH_ROOT_ENV)
    root = Path(override).expanduser() if override else DEFAULT_TEXT2KGBENCH_ROOT
    return root if (root / "a_ontologies").is_dir() else None


def _ground_truth_path(root: Path, stem: str) -> Path | None:
    for subdir in _GT_TEXT_SUBDIRS:
        candidate = root / "b_gt_text" / subdir / f"{stem}_ground_truth.jsonl"
        if candidate.exists():
            return candidate
    return None


def load_text2kgbench(
    root: Path,
    *,
    max_ontologies: int = 6,
    max_cases_per_ontology: int = 20,
) -> tuple[list[RecallCase], list[Ontology]]:
    """Load ``(cases, ontologies)`` from a Text2KGBench-style corpus.

    Only rows whose relation labels all resolve to ontology IRIs become cases, so a
    failure is always a retrieval failure and never a ground-truth gap.

    Args:
        root: Corpus root containing ``a_ontologies/`` and ``b_gt_text/``.
        max_ontologies: Cap on ontologies loaded into the catalog.
        max_cases_per_ontology: Cap on sentences taken from each ontology's GT file.
    """
    cases: list[RecallCase] = []
    ontologies: list[Ontology] = []

    for onto_path in sorted((root / "a_ontologies").glob("ont_*.ttl"))[:max_ontologies]:
        stem = onto_path.stem
        gt_path = _ground_truth_path(root, stem)
        if gt_path is None:
            continue

        graph = RDFGraph()
        graph.parse(str(onto_path), format="turtle")
        _disambiguate_prefixes(graph, stem)
        ontology = Ontology(graph=graph)
        labels = _label_index(graph)

        taken = 0
        with gt_path.open() as handle:
            for line in handle:
                if taken >= max_cases_per_ontology:
                    break
                row = json.loads(line)
                sentence = (row.get("sent") or "").strip()
                triples = row.get("triples") or []
                if not sentence or not triples:
                    continue
                expected: set[str] = set()
                for triple in triples:
                    expected |= labels.get(
                        (triple.get("rel") or "").strip().lower(), set()
                    )
                if not expected:
                    continue
                cases.append(
                    RecallCase(
                        case_id=row.get("id") or f"{stem}_{taken}",
                        text=sentence,
                        expected_iris=frozenset(expected),
                        ontology_iri=ontology.iri,
                    )
                )
                taken += 1

        if taken:
            ontologies.append(ontology)

    return cases, ontologies


def corpus_root() -> Path | None:
    """Prebuilt corpus directory from ``ONTOCAST_RECALL_CORPUS``, if it looks valid."""
    raw = os.getenv(CORPUS_ROOT_ENV)
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if (root / "cases.jsonl").is_file() else None


def load_corpus(root: Path) -> tuple[list[RecallCase], list[Ontology]]:
    """Load ``(cases, ontologies)`` from a prebuilt, domain-neutral recall corpus.

    Layout, as emitted by ``ontocast-validation/run/build_recall_corpus.py``::

        <root>/ontologies/*.ttl
        <root>/cases.jsonl   # {"id", "text", "expected_iris": [...], "ontology_iri"}

    Rows may carry extra keys (the builder writes a ``review`` block); they are ignored.
    Ground-truth defects raise rather than deflating recall silently: a case naming an
    ontology absent from the catalog would otherwise look like a retrieval miss.

    Args:
        root: Corpus directory containing ``ontologies/`` and ``cases.jsonl``.
    """
    ontology_dir = root / "ontologies"
    if not ontology_dir.is_dir():
        raise ValueError(f"{root}: missing ontologies/ directory")

    ontologies: list[Ontology] = []
    for path in sorted(ontology_dir.glob("*.ttl")):
        graph = RDFGraph()
        graph.parse(str(path), format="turtle")
        ontologies.append(Ontology(graph=graph))
    if not ontologies:
        raise ValueError(f"{ontology_dir}: no .ttl ontologies found")

    known_iris = {ontology.iri for ontology in ontologies}
    cases: list[RecallCase] = []
    with (root / "cases.jsonl").open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("id") or f"case_{line_number}")
            text = (row.get("text") or "").strip()
            expected = {str(iri) for iri in row.get("expected_iris") or []}
            ontology_iri = str(row.get("ontology_iri") or "")
            if not text or not expected:
                raise ValueError(f"{case_id}: needs both 'text' and 'expected_iris'")
            if ontology_iri not in known_iris:
                raise ValueError(
                    f"{case_id}: ontology_iri {ontology_iri!r} is not in the catalog"
                )
            cases.append(
                RecallCase(
                    case_id=case_id,
                    text=text,
                    expected_iris=frozenset(expected),
                    ontology_iri=ontology_iri,
                )
            )

    return cases, ontologies


def load_anchor_cases() -> tuple[list[RecallCase], list[Ontology]]:
    """Load in-repo anchor fixtures as ``(cases, ontologies)``.

    Each fixture document is split per line; a line becomes a case when it contains a
    phrase that is also an ``rdfs:label`` in the paired ontology.
    """
    cases: list[RecallCase] = []
    ontologies: list[Ontology] = []

    for domain in ("biomed", "finance"):
        onto_path = FIXTURES_DIR / f"{domain}_integration_ontology.ttl"
        doc_path = FIXTURES_DIR / f"{domain}_source_document.txt"
        if not onto_path.exists() or not doc_path.exists():
            continue

        graph = RDFGraph()
        graph.parse(str(onto_path), format="turtle")
        ontology = Ontology(graph=graph)
        labels = _label_index(graph)
        ontologies.append(ontology)

        for index, line in enumerate(doc_path.read_text().splitlines()):
            sentence = line.strip()
            if not sentence:
                continue
            haystack = re.sub(r"[^a-z0-9 ]+", " ", sentence.lower())
            haystack = re.sub(r"\s+", " ", haystack)
            expected: set[str] = set()
            for label, iris in labels.items():
                if len(label) > 3 and label in haystack:
                    expected |= iris
            if expected:
                cases.append(
                    RecallCase(
                        case_id=f"{domain}_{index}",
                        text=sentence,
                        expected_iris=frozenset(expected),
                        ontology_iri=ontology.iri,
                    )
                )

    return cases, ontologies
