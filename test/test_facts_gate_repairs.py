"""Regression tests for defects found in the pre-0.5.0 release audit.

Each test here pins a mechanism that was wired but could not fire, or that
degraded silently. They are grouped by the defect they guard.
"""

import logging
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD

from ontocast.config import FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.stategraph.node_factories import _vetoes_from_findings
from ontocast.tool.facts_invariants import (
    _shacl_findings,
    collect_shacl_shapes,
    repair_property_aliases,
    validate_aggregated_facts,
)
from ontocast.toolbox import ToolBox

CD = f"{DEFAULT_IRI}/"
Q = "https://x.org/schema#"


# --- DEGENERATE_COREFERENCE could never drive the un-merge repair ------------


def test_vetoes_from_findings_uses_iri_values_not_only_subject() -> None:
    """The over-merged node is in ``values``, not ``subject``.

    A collapsed range reports ``range1`` (the pointing node) as subject and the
    merged endpoint ``v1`` in values. ``range1`` is typically not a merged
    cluster at all, so reading only ``subject`` produced an empty veto set and
    the repair loop broke immediately on an error-severity finding.
    """
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.DEGENERATE_COREFERENCE,
        message="collapsed bounds",
        subject=CD + "range1",
        values=[CD + "merged_endpoint"],
    )
    clusters = {CD + "merged_endpoint": [CD + "lower", CD + "upper"]}

    assert _vetoes_from_findings([finding], clusters) == {
        frozenset((URIRef(CD + "lower"), URIRef(CD + "upper")))
    }


def test_vetoes_from_findings_ignores_literal_values() -> None:
    """Literal values in ``values`` must not be looked up as cluster keys."""
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.FUNCTIONAL_VIOLATION,
        message="two values",
        subject=CD + "merged",
        values=["12.5", "96"],
    )
    clusters = {CD + "merged": [CD + "a", CD + "b"]}

    assert _vetoes_from_findings([finding], clusters) == {
        frozenset((URIRef(CD + "a"), URIRef(CD + "b")))
    }


# --- owl:sameAs must not be read as a dominantly single-valued predicate -----


def _sameas_graph(cluster_size: int) -> RDFGraph:
    """Canonical entity with N-1 sameAs links, as the rewriter emits them."""
    graph = RDFGraph()
    canonical = URIRef("https://other.org/facts/canonical")
    for index in range(cluster_size - 1):
        graph.add((canonical, OWL.sameAs, URIRef(f"https://other.org/facts/m{index}")))
    # Two unmerged singletons give sameAs a "dominantly single-valued" profile.
    for index in range(4):
        singleton = URIRef(f"https://other.org/facts/s{index}")
        graph.add((singleton, OWL.sameAs, URIRef(f"https://other.org/facts/o{index}")))
    return graph


def test_sameas_does_not_produce_suspect_multi_value() -> None:
    """Merge bookkeeping must not be mistaken for a bad merge.

    ``owl:sameAs`` carries 1 object for an unmerged entity and N-1 for a
    cluster, so a graph with many singletons and one large cluster made it look
    single-valued -- and every large cluster then became an error that drove
    the repair to dissolve a legitimate merge.
    """
    report = validate_aggregated_facts(
        _sameas_graph(cluster_size=4),
        None,
        fact_namespaces=["https://other.org/facts/"],
    )
    assert [
        finding for finding in report.findings if finding.predicate == str(OWL.sameAs)
    ] == []
    assert not report.error_findings


# --- SHACL degraded silently -------------------------------------------------


def test_missing_shapes_dir_warns(caplog, tmp_path) -> None:
    """A configured directory that does not exist must not read as 'clean'."""
    with caplog.at_level(logging.WARNING):
        shapes = collect_shacl_shapes(None, str(tmp_path / "nope"))
    assert shapes is None
    assert "not a directory" in caplog.text


def test_empty_shapes_dir_warns(caplog, tmp_path) -> None:
    with caplog.at_level(logging.WARNING):
        shapes = collect_shacl_shapes(None, str(tmp_path))
    assert shapes is None
    assert "no .ttl shape files" in caplog.text


def test_missing_pyshacl_warns(caplog, monkeypatch) -> None:
    """Absent extra is reported at warning level, not debug.

    Reaching ``_shacl_findings`` means shapes were found, so the caller expects
    validation to run; returning [] quietly is indistinguishable from conforms.
    """
    import builtins

    real_import = builtins.__import__

    def fail_pyshacl(name, *args, **kwargs):
        if name == "pyshacl":
            raise ImportError("no pyshacl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pyshacl)
    with caplog.at_level(logging.WARNING):
        assert _shacl_findings(RDFGraph(), RDFGraph()) == []
    assert "pyshacl is not installed" in caplog.text


# --- repair_property_aliases rewrites against a partial snapshot -------------


def _alias_graph(predicate: str, subjects: int = 1) -> RDFGraph:
    graph = RDFGraph()
    for index in range(subjects):
        graph.add(
            (
                URIRef(f"{CD}q{index}"),
                URIRef(predicate),
                Literal("1.0", datatype=XSD.decimal),
            )
        )
    return graph


def test_alias_repair_does_not_invent_a_target_from_an_empty_catalog() -> None:
    """With no catalog terms there is nothing to rewrite toward."""
    graph = _alias_graph(Q + "hasResult", subjects=2)
    rewritten, _ = repair_property_aliases(graph, RDFGraph(), min_ratio=0.85)
    assert rewritten == 0
    assert (None, URIRef(Q + "hasResult"), None) in graph


def test_alias_repair_rewrites_toward_a_declared_catalog_term() -> None:
    """The intended case: a near-miss of a term the catalog actually declares."""
    ontology = RDFGraph()
    ontology.add((URIRef(Q + "numericValue"), RDF.type, OWL.DatatypeProperty))
    graph = _alias_graph(Q + "numericvalue")

    rewritten, _ = repair_property_aliases(graph, ontology, min_ratio=0.85)

    assert rewritten == 1
    assert (None, URIRef(Q + "numericValue"), None) in graph


# --- the repaired graph must not blindly replace the original ---------------


def _fake_tools(aggregator, **overrides) -> ToolBox:
    facts_validation = FactsValidationConfig(**overrides)
    return cast(
        ToolBox,
        SimpleNamespace(
            aggregator=aggregator,
            config=SimpleNamespace(
                get_tool_config=lambda: SimpleNamespace(
                    facts_validation=facts_validation
                )
            ),
        ),
    )


def test_repair_that_does_not_reduce_errors_is_reverted(monkeypatch, caplog) -> None:
    """Un-merging is destructive; a pass that does not help must be rolled back.

    The aggregator here returns a graph that still violates the invariant, so
    the repair buys nothing while dissolving the cluster.
    """
    from ontocast.onto.content_unit import ContentUnit, OutputType
    from ontocast.onto.state import AgentState
    from ontocast.stategraph.node_factories import make_validate_facts_node

    subject = URIRef(CD + "merged")

    def conflicting_graph() -> RDFGraph:
        graph = RDFGraph()
        graph.add((subject, URIRef(Q + "numericValue"), Literal(1)))
        graph.add((subject, URIRef(Q + "numericValue"), Literal(2)))
        graph.add((subject, RDFS.label, Literal("m")))
        return graph

    class _StubAggregator:
        def __init__(self) -> None:
            self.calls = 0

        def postprocess_facts_units(self, **kwargs):
            self.calls += 1
            # Same violation survives the re-aggregation.
            return SimpleNamespace(graph=conflicting_graph(), merged_clusters={})

    aggregator = _StubAggregator()
    tools = _fake_tools(aggregator)

    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    original = conflicting_graph()
    state.aggregated_facts = original
    state.aggregation_clusters = {str(subject): [CD + "a", CD + "b"]}
    state.facts_units = [
        ContentUnit(
            text="t",
            index=0,
            doc_iri=URIRef("https://x.org/doc/1"),
            graph=conflicting_graph(),
            type=OutputType.FACTS,
        )
    ]

    with caplog.at_level(logging.WARNING):
        make_validate_facts_node(tools)(state)

    assert aggregator.calls == 1
    assert state.aggregated_facts is original
    assert state.retrieval_metrics["facts_merge_repairs_rejected"] == 1
    assert state.retrieval_metrics["facts_merge_repair_passes"] == 0
    assert "did not reduce errors" in caplog.text


# --- functional_min_single_support is reachable from config -----------------


@pytest.mark.parametrize(
    ("min_support", "expect_error"),
    [(2, True), (99, False)],
)
def test_functional_min_single_support_is_honoured(
    min_support: int, expect_error: bool
) -> None:
    """The knob was exposed but never passed, pinning it at a hardcoded 3."""
    graph = RDFGraph()
    for index in range(3):
        graph.add((URIRef(f"{CD}s{index}"), URIRef(Q + "p"), URIRef(f"{CD}o{index}")))
    conflicted = URIRef(CD + "bad")
    graph.add((conflicted, URIRef(Q + "p"), URIRef(CD + "x")))
    graph.add((conflicted, URIRef(Q + "p"), URIRef(CD + "y")))

    report = validate_aggregated_facts(
        graph,
        None,
        fact_namespaces=[CD],
        functional_min_single_support=min_support,
    )
    assert bool(report.error_findings) is expect_error
