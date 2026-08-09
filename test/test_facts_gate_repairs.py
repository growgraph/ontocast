"""Regression tests for defects found in the pre-0.5.0 release audit.

Each test here pins a mechanism that was wired but could not fire, or that
degraded silently. They are grouped by the defect they guard.
"""

import logging
from types import SimpleNamespace
from typing import cast

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Literal, Namespace, URIRef
from rdflib.namespace import XSD

from ontocast.config import FactsValidationConfig
from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.stategraph.facts_gate import vetoes_from_findings
from ontocast.tool.facts_invariants import (
    apply_shacl_repairs,
    collect_shacl_shapes,
    repair_property_aliases,
    resolve_code_literals,
    run_shacl,
    summarize_conformance,
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

    assert vetoes_from_findings([finding], clusters) == {
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

    assert vetoes_from_findings([finding], clusters) == {
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
    """Absent extra is reported at warning level, and as "did not run".

    Reaching ``run_shacl`` means shapes were found, so the caller expects
    validation to run; returning [] quietly is indistinguishable from conforms.
    ``None`` is what keeps the two apart downstream.
    """
    import builtins

    real_import = builtins.__import__

    def fail_pyshacl(name, *args, **kwargs):
        if name == "pyshacl":
            raise ImportError("no pyshacl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pyshacl)
    with caplog.at_level(logging.WARNING):
        assert run_shacl(RDFGraph(), RDFGraph()) is None
    assert "pyshacl is not installed" in caplog.text


def test_oversized_graph_skips_shacl_with_a_warning(caplog) -> None:
    """A skipped run must not read as a clean one."""
    graph = RDFGraph()
    for index in range(5):
        graph.add((URIRef(f"{CD}s{index}"), RDFS.label, Literal(index)))
    with caplog.at_level(logging.WARNING):
        assert run_shacl(graph, _value_shapes(), max_triples=2) is None
    assert "unvalidated, not conformant" in caplog.text


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
    rewritten, _, _applied = repair_property_aliases(graph, RDFGraph(), min_ratio=0.85)
    assert rewritten == 0
    assert (None, URIRef(Q + "hasResult"), None) in graph


def test_alias_repair_rewrites_toward_a_declared_catalog_term() -> None:
    """The intended case: a near-miss of a term the catalog actually declares."""
    ontology = RDFGraph()
    ontology.add((URIRef(Q + "numericValue"), RDF.type, OWL.DatatypeProperty))
    graph = _alias_graph(Q + "numericvalue")

    rewritten, _, _applied = repair_property_aliases(graph, ontology, min_ratio=0.85)

    assert rewritten == 1
    assert (None, URIRef(Q + "numericValue"), None) in graph


# --- the repaired graph must not blindly replace the original ---------------


def _fake_tools(aggregator, **overrides) -> ToolBox:
    # model_construct: field defaults + overrides only, no environment reads --
    # a developer shell with FACTS_* exported must not steer these tests.
    facts_validation = FactsValidationConfig.model_construct(**overrides)
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
    assert "did not reduce merge-signature errors" in caplog.text


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


# --- SHACL findings must not drive the un-merge repair -----------------------


def test_shacl_findings_never_veto_a_merge() -> None:
    """A constraint violation is not evidence that two entities were confused.

    Un-merging cannot fix a missing required property, so letting SHACL reach
    the veto set dissolved legitimate clusters whenever the focus node happened
    to be merged.
    """
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.SHACL,
        message="missing qualifier",
        subject=CD + "merged",
    )
    clusters = {CD + "merged": [CD + "a", CD + "b"]}

    assert vetoes_from_findings([finding], clusters) == set()


# --- LLM-free SHACL repair ---------------------------------------------------

SH = Namespace("http://www.w3.org/ns/shacl#")
VALUE_CLASS = URIRef(Q + "QuantityValue")
NUMERIC = URIRef(Q + "numericValue")
UNIT = URIRef(Q + "unit")
UNIT_CLASS = URIRef(Q + "Unit")
UCUM = URIRef("http://qudt.org/schema/qudt/ucumCode")


def _value_shapes() -> RDFGraph:
    """Shapes for a quantity value: typed numeric, IRI unit, numeric required."""
    shapes = RDFGraph()
    shape = URIRef(Q + "ValueShape")
    numeric_prop = URIRef(Q + "p_numeric")
    unit_prop = URIRef(Q + "p_unit")
    shapes.add((shape, RDF.type, SH.NodeShape))
    shapes.add((shape, SH.targetClass, VALUE_CLASS))
    shapes.add((shape, SH.property, numeric_prop))
    shapes.add((numeric_prop, SH.path, NUMERIC))
    shapes.add((numeric_prop, SH.datatype, XSD.decimal))
    shapes.add((numeric_prop, SH.minCount, Literal(1)))
    shapes.add((shape, SH.property, unit_prop))
    shapes.add((unit_prop, SH.path, UNIT))
    shapes.add((unit_prop, SH.nodeKind, SH.IRI))
    return shapes


def _unit_catalog() -> RDFGraph:
    """One unit individual carrying a UCUM code."""
    ontology = RDFGraph()
    day = URIRef(Q + "DAY")
    ontology.add((UNIT_CLASS, RDF.type, OWL.Class))
    ontology.add((day, RDF.type, UNIT_CLASS))
    ontology.add((day, UCUM, Literal("d")))
    ontology.add((UNIT, RDF.type, OWL.ObjectProperty))
    ontology.add((UNIT, RDFS.range, UNIT_CLASS))
    return ontology


def _repair(graph: RDFGraph, *, mode: str = "prune", ontology=None):
    return apply_shacl_repairs(
        graph,
        _value_shapes(),
        ontology if ontology is not None else _unit_catalog(),
        mode=mode,
        passes=2,
        fact_namespaces=[CD],
        code_predicates=[str(UCUM)],
    )


def test_datatype_violation_is_retyped_not_reported() -> None:
    """An untyped literal that parses as the declared datatype is repairable."""
    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("230")))
    graph.add((node, UNIT, URIRef(Q + "DAY")))

    result = _repair(graph)

    assert result.violations_after == 0
    assert list(result.graph.objects(node, NUMERIC)) == [
        Literal("230", datatype=XSD.decimal)
    ]
    assert [record.kind for record in result.records] == ["shacl_retype"]


def test_literal_on_an_iri_path_resolves_to_the_unique_catalog_term() -> None:
    """``unit "d"`` becomes ``unit <DAY>`` when exactly one term declares "d"."""
    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("4", datatype=XSD.decimal)))
    graph.add((node, UNIT, Literal("d")))

    result = _repair(graph)

    assert list(result.graph.objects(node, UNIT)) == [URIRef(Q + "DAY")]
    assert result.violations_after == 0


def test_ambiguous_surface_form_is_left_reported() -> None:
    """Two terms claiming one code is not a repairable situation."""
    ontology = _unit_catalog()
    twin = URIRef(Q + "DAY_TWIN")
    ontology.add((twin, RDF.type, UNIT_CLASS))
    ontology.add((twin, UCUM, Literal("d")))

    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("4", datatype=XSD.decimal)))
    graph.add((node, UNIT, Literal("d")))

    result = _repair(graph, ontology=ontology)

    assert result.records == []
    assert list(result.graph.objects(node, UNIT)) == [Literal("d")]


def test_placeholder_node_is_pruned_with_its_incoming_edge() -> None:
    """A value node asserting only a label stands for an extraction that failed."""
    graph = RDFGraph()
    placeholder = URIRef(CD + "v_empty")
    observation = URIRef(CD + "obs")
    graph.add((placeholder, RDF.type, VALUE_CLASS))
    graph.add((placeholder, RDFS.label, Literal("efficiency")))
    graph.add((observation, URIRef(Q + "hasValue"), placeholder))

    result = _repair(graph)

    assert [record.kind for record in result.records] == ["shacl_prune"]
    assert len(result.graph) == 0
    assert result.violations_after == 0


def test_node_with_data_is_never_pruned_and_never_invented() -> None:
    """Missing a required property is reported, not filled in or deleted."""
    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, UNIT, URIRef(Q + "DAY")))
    graph.add((node, URIRef(Q + "comment"), Literal("source states < 1 um")))

    result = _repair(graph)

    assert result.records == []
    assert result.violations_after == result.violations_before == 1
    assert (
        node,
        URIRef(Q + "comment"),
        Literal("source states < 1 um"),
    ) in result.graph


def test_oxigraph_reified_graph_survives_the_shacl_gate() -> None:
    """The aggregated graph is oxigraph-backed and carries RDF 1.2 triple terms.

    ``rdflib.Graph.add`` asserts on a triple term, so the gate must neither
    crash handing such a graph to pyshacl nor drop the reification provenance
    from the repaired graph (repairs are applied in place, not on a copy).
    """
    ox = pytest.importorskip("pyoxigraph")
    from oxrdflib._converter import to_ox

    from ontocast.onto.constants import RDF_REIFIES
    from ontocast.onto.rdfgraph import _oxigraph_inner_store, is_rdflib_triple

    graph = RDFGraph(store="oxigraph")
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("230")))
    graph.add((node, UNIT, URIRef(Q + "DAY")))
    # Reified provenance, as tool/agg/rewriter.py emits it.
    inner = cast(ox.Store, _oxigraph_inner_store(graph.store))
    graph_ctx = to_ox(graph.identifier)
    inner.add(
        ox.Quad(
            ox.BlankNode(),
            ox.NamedNode(str(RDF_REIFIES)),
            ox.Triple(
                ox.NamedNode(str(node)),
                ox.NamedNode(str(NUMERIC)),
                ox.Literal("230"),
            ),
            graph_ctx,
        )
    )
    assert any(not is_rdflib_triple(triple) for triple in graph)

    result = _repair(graph)

    assert [record.kind for record in result.records] == ["shacl_retype"]
    assert (node, NUMERIC, Literal("230", datatype=XSD.decimal)) in result.graph
    # The provenance triple term is still there.
    assert any(not is_rdflib_triple(triple) for triple in result.graph)


def test_catalog_only_focus_never_produces_a_phantom_prune() -> None:
    """A focus node that lives only in the mixed-in catalog is not the gate's.

    Blank catalog nodes bypass the namespace scope check (they have no
    namespace), and a node absent from the facts graph used to read as
    "asserts nothing" — yielding an empty repair record whose no-op pass
    tripped the strict-decrease revert and discarded genuine repairs.
    """
    from rdflib import BNode

    shapes = _value_shapes()
    ontology = _unit_catalog()
    # A blank node typed as the target class, declared only in the catalog.
    ontology.add((BNode(), RDF.type, VALUE_CLASS))

    graph = RDFGraph()
    placeholder = URIRef(CD + "v_empty")
    observation = URIRef(CD + "obs")
    graph.add((placeholder, RDF.type, VALUE_CLASS))
    graph.add((placeholder, RDFS.label, Literal("efficiency")))
    graph.add((observation, URIRef(Q + "hasValue"), placeholder))

    result = apply_shacl_repairs(
        graph,
        shapes,
        ontology,
        mode="prune",
        passes=2,
        fact_namespaces=[CD],
        code_predicates=[str(UCUM)],
    )

    assert result.reverted is False
    assert [record.kind for record in result.records] == ["shacl_prune"]
    assert all(record.triple_count > 0 for record in result.records)
    assert len(result.graph) == 0


def test_rewrite_mode_does_not_prune() -> None:
    """``rewrite`` is the never-remove-a-triple tier."""
    graph = RDFGraph()
    placeholder = URIRef(CD + "v_empty")
    graph.add((placeholder, RDF.type, VALUE_CLASS))
    graph.add((placeholder, RDFS.label, Literal("efficiency")))

    result = _repair(graph, mode="rewrite")

    assert result.records == []
    assert len(result.graph) == 2


def test_autofix_off_reports_without_repairing() -> None:
    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("230")))

    result = _repair(graph, mode="off")

    assert not result.ran
    assert result.records == []
    assert list(result.graph.objects(node, NUMERIC)) == [Literal("230")]


def test_ontology_is_mixed_into_the_data_graph() -> None:
    """A catalog individual's type lives in the ontology, not in the facts.

    Validating the facts alone failed every ``sh:class`` constraint pointing at
    a catalog term -- violations describing the absent schema, not the data.
    """
    shapes = RDFGraph()
    shape = URIRef(Q + "UnitShape")
    unit_prop = URIRef(Q + "p_unit_class")
    shapes.add((shape, RDF.type, SH.NodeShape))
    shapes.add((shape, SH.targetClass, VALUE_CLASS))
    shapes.add((shape, SH.property, unit_prop))
    shapes.add((unit_prop, SH.path, UNIT))
    shapes.add((unit_prop, SH["class"], UNIT_CLASS))

    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, UNIT, URIRef(Q + "DAY")))

    assert len(run_shacl(graph, shapes, ontology_graph=None) or []) == 1
    assert run_shacl(graph, shapes, ontology_graph=_unit_catalog()) == []


def test_rdfs_inference_sees_the_specialised_predicate() -> None:
    """SHACL paths carry no subPropertyOf entailment.

    The renderer emits the most specific predicate it can. A shape naming the
    superproperty then reports a statement that *is* there as missing —
    unless RDFS inference runs first. (Class targets need no help: SHACL
    resolves those through rdfs:subClassOf itself.)
    """
    ontology = _unit_catalog()
    precise = URIRef(Q + "preciseNumericValue")
    ontology.add((precise, RDFS.subPropertyOf, NUMERIC))

    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, precise, Literal("4", datatype=XSD.decimal)))
    graph.add((node, UNIT, URIRef(Q + "DAY")))

    without = (
        run_shacl(graph, _value_shapes(), ontology_graph=ontology, inference="none")
        or []
    )
    assert [str(violation.path) for violation in without] == [str(NUMERIC)]
    assert (
        run_shacl(graph, _value_shapes(), ontology_graph=ontology, inference="rdfs")
        == []
    )


# --- code resolution ---------------------------------------------------------


def test_code_literal_links_to_the_catalog_individual() -> None:
    """``ucumCode "d"`` with no unit link gains ``unit <DAY>``."""
    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, UCUM, Literal("d")))

    added, records = resolve_code_literals(graph, _unit_catalog(), [str(UCUM)])

    assert added == 1
    assert list(graph.objects(node, UNIT)) == [URIRef(Q + "DAY")]
    assert [record.kind for record in records] == ["code_resolved"]


def test_code_resolution_leaves_an_existing_link_alone() -> None:
    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, UCUM, Literal("d")))
    graph.add((node, UNIT, URIRef(Q + "OTHER")))

    added, _ = resolve_code_literals(graph, _unit_catalog(), [str(UCUM)])

    assert added == 0
    assert list(graph.objects(node, UNIT)) == [URIRef(Q + "OTHER")]


def test_code_resolution_needs_an_unambiguous_code() -> None:
    ontology = _unit_catalog()
    twin = URIRef(Q + "DAY_TWIN")
    ontology.add((twin, RDF.type, UNIT_CLASS))
    ontology.add((twin, UCUM, Literal("d")))

    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, UCUM, Literal("d")))

    added, _ = resolve_code_literals(graph, ontology, [str(UCUM)])

    assert added == 0


def test_code_resolution_falls_back_to_observed_usage() -> None:
    """A vendored projection may declare individuals but no rdfs:range."""
    ontology = _unit_catalog()
    ontology.remove((UNIT, RDFS.range, UNIT_CLASS))

    graph = RDFGraph()
    linked = URIRef(CD + "v_linked")
    graph.add((linked, RDF.type, VALUE_CLASS))
    graph.add((linked, UNIT, URIRef(Q + "DAY")))
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, UCUM, Literal("d")))

    added, _ = resolve_code_literals(graph, ontology, [str(UCUM)])

    assert added == 1
    assert list(graph.objects(node, UNIT)) == [URIRef(Q + "DAY")]


# --- conformance summary -----------------------------------------------------


def test_summary_separates_conforms_from_never_checked() -> None:
    """ "No SHACL findings" is meaningless without knowing whether it ran."""
    assert summarize_conformance([], shacl_evaluated=True)["conforms"] is True
    assert summarize_conformance([], shacl_evaluated=False)["conforms"] is None
    assert summarize_conformance([], shacl_evaluated=None)["conforms"] is None


def test_summary_groups_violations_by_constraint() -> None:
    """One systematic gap must not read as N independent defects."""
    findings = [
        FactsValidationFinding(
            kind=FactsValidationFindingKind.SHACL,
            message="missing qualifier",
            subject=f"{CD}v{index}",
            component=str(SH.MinCountConstraintComponent),
            source_shape=Q + "ValueShape",
        )
        for index in range(3)
    ]
    summary = summarize_conformance(findings, shacl_evaluated=True)

    assert summary["conforms"] is False
    assert summary["shacl_by_constraint"] == {"MinCountConstraintComponent": 3}
    assert summary["shacl_by_shape"] == {Q + "ValueShape": 3}


def test_gate_repairs_and_reports_through_the_node(tmp_path) -> None:
    """End-to-end through VALIDATE_FACTS: repair applied, result reported.

    Pins the wiring, not the repair logic: shapes reach the gate, the LLM-free
    pass mutates the served graph, and the conformance summary plus the repair
    records land on the state the API and the batch dump read.
    """
    from ontocast.onto.content_unit import ContentUnit, OutputType
    from ontocast.onto.state import AgentState
    from ontocast.stategraph.node_factories import make_validate_facts_node

    (tmp_path / "shapes.ttl").write_text(
        _value_shapes().serialize(format="turtle"), encoding="utf-8"
    )

    node = URIRef(CD + "v1")
    facts = RDFGraph()
    facts.add((node, RDF.type, VALUE_CLASS))
    facts.add((node, NUMERIC, Literal("230")))
    facts.add((node, UNIT, URIRef(Q + "DAY")))

    state = AgentState()
    state.current_domain = "https://x.org"
    state.doc_hid = "1"
    state.aggregated_facts = facts
    state.facts_ontology_context = _unit_catalog()
    state.facts_units = [
        ContentUnit(
            text="t",
            index=0,
            doc_iri=URIRef("https://x.org/doc/1"),
            graph=facts,
            type=OutputType.FACTS,
        )
    ]

    tools = _fake_tools(None, shapes_dir=str(tmp_path))
    make_validate_facts_node(tools)(state)

    assert list(state.aggregated_facts.objects(node, NUMERIC)) == [
        Literal("230", datatype=XSD.decimal)
    ]
    assert [record.kind for record in state.facts_gate_repairs] == ["shacl_retype"]
    assert state.facts_conformance["conforms"] is True
    assert state.facts_conformance["repairs_applied"] == {"shacl_retype": 1}
    assert state.retrieval_metrics["facts_shacl_violations_before"] == 1
    assert state.retrieval_metrics["facts_shacl_violations_after"] == 0


def test_blank_property_shape_still_drives_the_retype_repair() -> None:
    """``sh:property [ sh:path … ; sh:datatype … ]`` is the common style.

    The reported ``sh:sourceShape`` is then a blank node. Narrowing it to
    URIRef discarded it, so the datatype lookup had nothing to resolve and
    every inline-shaped violation fell through to "reported, not repaired".
    """
    shapes = RDFGraph()
    shape = URIRef(Q + "InlineValueShape")
    blank_prop = BNode()
    shapes.add((shape, RDF.type, SH.NodeShape))
    shapes.add((shape, SH.targetClass, VALUE_CLASS))
    shapes.add((shape, SH.property, blank_prop))
    shapes.add((blank_prop, SH.path, NUMERIC))
    shapes.add((blank_prop, SH.datatype, XSD.decimal))

    graph = RDFGraph()
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("230")))

    result = apply_shacl_repairs(
        graph,
        shapes,
        _unit_catalog(),
        mode="prune",
        passes=2,
        fact_namespaces=[CD],
        code_predicates=[str(UCUM)],
    )

    assert [record.kind for record in result.records] == ["shacl_retype"]
    assert list(result.graph.objects(node, NUMERIC)) == [
        Literal("230", datatype=XSD.decimal)
    ]
    assert result.violations_after == 0


def test_pruning_sweeps_the_provenance_reifier_of_the_removed_triple() -> None:
    """A pruned node is also named inside ``rdf:reifies <<( s p o )>>``.

    Neither an incoming nor an outgoing pattern matches a node sitting in a
    triple term, so the reifier and its ``prov:wasDerivedFrom`` used to survive
    the prune, describing a statement that no longer exists.
    """
    ox = pytest.importorskip("pyoxigraph")
    from oxrdflib._converter import to_ox

    from ontocast.onto.constants import PROV, RDF_REIFIES
    from ontocast.onto.rdfgraph import _oxigraph_inner_store

    graph = RDFGraph(store="oxigraph")
    placeholder = URIRef(CD + "v_empty")
    observation = URIRef(CD + "obs")
    has_value = URIRef(Q + "hasValue")
    graph.add((placeholder, RDF.type, VALUE_CLASS))
    graph.add((placeholder, RDFS.label, Literal("efficiency")))
    graph.add((observation, has_value, placeholder))

    inner = cast(ox.Store, _oxigraph_inner_store(graph.store))
    graph_ctx = to_ox(graph.identifier)
    reifier = ox.BlankNode()
    inner.add(
        ox.Quad(
            reifier,
            ox.NamedNode(str(RDF_REIFIES)),
            ox.Triple(
                ox.NamedNode(str(observation)),
                ox.NamedNode(str(has_value)),
                ox.NamedNode(str(placeholder)),
            ),
            graph_ctx,
        )
    )
    inner.add(
        ox.Quad(
            reifier,
            ox.NamedNode(str(PROV.wasDerivedFrom)),
            ox.NamedNode("https://example.org/doc/chunk0"),
            graph_ctx,
        )
    )

    result = _repair(graph)

    assert [record.kind for record in result.records] == ["shacl_prune"]
    assert len(result.graph) == 0
    assert (
        list(
            inner.quads_for_pattern(
                None, ox.NamedNode(str(RDF_REIFIES)), None, graph_ctx
            )
        )
        == []
    )
    assert (
        list(
            inner.quads_for_pattern(
                None, ox.NamedNode(str(PROV.wasDerivedFrom)), None, graph_ctx
            )
        )
        == []
    )


def _reified(graph: RDFGraph, triple: tuple, derived_from: str):
    """Attach an RDF 1.2 reifier with one ``prov:wasDerivedFrom`` arc.

    Returns ``(pyoxigraph module, inner store, graph context, reifier node)``.
    """
    ox = pytest.importorskip("pyoxigraph")
    from oxrdflib._converter import to_ox

    from ontocast.onto.constants import PROV, RDF_REIFIES
    from ontocast.onto.rdfgraph import _oxigraph_inner_store

    inner = cast(ox.Store, _oxigraph_inner_store(graph.store))
    graph_ctx = to_ox(graph.identifier)
    reifier = ox.BlankNode()
    subject, predicate, obj = (to_ox(position) for position in triple)
    inner.add(
        ox.Quad(
            reifier,
            ox.NamedNode(str(RDF_REIFIES)),
            ox.Triple(subject, predicate, obj),
            graph_ctx,
        )
    )
    inner.add(
        ox.Quad(
            reifier,
            ox.NamedNode(str(PROV.wasDerivedFrom)),
            ox.NamedNode(derived_from),
            graph_ctx,
        )
    )
    return ox, inner, graph_ctx, reifier


def _reifies_quads(ox, inner, graph_ctx) -> list:
    from ontocast.onto.constants import RDF_REIFIES

    return list(
        inner.quads_for_pattern(None, ox.NamedNode(str(RDF_REIFIES)), None, graph_ctx)
    )


def test_retype_moves_the_provenance_reifier_to_the_retyped_statement() -> None:
    """A retype rewrites a statement; its provenance must follow, not dangle.

    The prune sweep deletes a reifier because the statement is gone. Here the
    statement survives with a different object, so dropping the reifier would
    lose the derivation of a triple that is still served.
    """
    graph = RDFGraph(store="oxigraph")
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("230")))
    graph.add((node, UNIT, URIRef(Q + "DAY")))

    ox, inner, graph_ctx, reifier = _reified(
        graph, (node, NUMERIC, Literal("230")), "https://example.org/doc/chunk0"
    )

    result = _repair(graph)

    assert [record.kind for record in result.records] == ["shacl_retype"]
    quads = _reifies_quads(ox, inner, graph_ctx)
    assert len(quads) == 1
    term = quads[0].object
    assert isinstance(term, ox.Triple)
    assert term.object == ox.Literal("230", datatype=ox.NamedNode(str(XSD.decimal)))
    # The reifier node is untouched, so its prov arcs travel with it.
    assert quads[0].subject == reifier
    from ontocast.onto.constants import PROV

    assert (
        len(
            list(
                inner.quads_for_pattern(
                    reifier, ox.NamedNode(str(PROV.wasDerivedFrom)), None, graph_ctx
                )
            )
        )
        == 1
    )


def test_code_resolution_moves_the_provenance_reifier() -> None:
    """Resolving ``unit "d"`` to a catalog IRI carries the reifier across too."""
    graph = RDFGraph(store="oxigraph")
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("230", datatype=XSD.decimal)))
    graph.add((node, UNIT, Literal("d")))

    ox, inner, graph_ctx, _ = _reified(
        graph, (node, UNIT, Literal("d")), "https://example.org/doc/chunk0"
    )

    result = _repair(graph)

    assert [record.kind for record in result.records] == ["shacl_code_resolved"]
    quads = _reifies_quads(ox, inner, graph_ctx)
    assert len(quads) == 1
    term = quads[0].object
    assert isinstance(term, ox.Triple)
    assert term.object == ox.NamedNode(Q + "DAY")


def test_reverted_pass_leaves_the_retyped_reifier_alone() -> None:
    """Retargeting runs after the accept test, so a reverted pass is a no-op."""
    graph = RDFGraph(store="oxigraph")
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("not-a-number")))

    ox, inner, graph_ctx, _ = _reified(
        graph,
        (node, NUMERIC, Literal("not-a-number")),
        "https://example.org/doc/chunk0",
    )

    _repair(graph)

    quads = _reifies_quads(ox, inner, graph_ctx)
    assert len(quads) == 1
    term = quads[0].object
    assert isinstance(term, ox.Triple)
    assert term.object == ox.Literal("not-a-number")


def test_reverted_pass_leaves_the_provenance_reifier_alone() -> None:
    """The sweep runs only after a pass is accepted, so a revert keeps it."""
    ox = pytest.importorskip("pyoxigraph")
    from oxrdflib._converter import to_ox

    from ontocast.onto.constants import RDF_REIFIES
    from ontocast.onto.rdfgraph import _oxigraph_inner_store

    graph = RDFGraph(store="oxigraph")
    node = URIRef(CD + "v1")
    graph.add((node, RDF.type, VALUE_CLASS))
    graph.add((node, NUMERIC, Literal("not-a-number")))

    inner = cast(ox.Store, _oxigraph_inner_store(graph.store))
    graph_ctx = to_ox(graph.identifier)
    inner.add(
        ox.Quad(
            ox.BlankNode(),
            ox.NamedNode(str(RDF_REIFIES)),
            ox.Triple(
                ox.NamedNode(str(node)),
                ox.NamedNode(str(NUMERIC)),
                ox.Literal("not-a-number"),
            ),
            graph_ctx,
        )
    )

    _repair(graph)

    assert (
        len(
            list(
                inner.quads_for_pattern(
                    None, ox.NamedNode(str(RDF_REIFIES)), None, graph_ctx
                )
            )
        )
        == 1
    )


def test_blank_node_shacl_violation_reaches_the_report() -> None:
    """A blank-node focus is repairable, so it must also be reportable.

    Scope was decided on the *projected* finding, whose subject is a
    stringified blank node matching no namespace prefix. Every blank-node
    violation was therefore filtered out, and ``facts_validation_findings``
    under-counted exactly the nodes the repair pass had acted on.
    """
    shapes = RDFGraph()
    shape = URIRef(Q + "BlankFocusShape")
    numeric_prop = URIRef(Q + "bp_numeric")
    shapes.add((shape, RDF.type, SH.NodeShape))
    shapes.add((shape, SH.targetClass, VALUE_CLASS))
    shapes.add((shape, SH.property, numeric_prop))
    shapes.add((numeric_prop, SH.path, NUMERIC))
    shapes.add((numeric_prop, SH.minCount, Literal(1)))

    graph = RDFGraph()
    anonymous = BNode()
    graph.add((URIRef(CD + "obs"), URIRef(Q + "hasValue"), anonymous))
    graph.add((anonymous, RDF.type, VALUE_CLASS))
    graph.add((anonymous, RDFS.label, Literal("efficiency")))

    report = validate_aggregated_facts(
        graph, _unit_catalog(), shapes_graph=shapes, fact_namespaces=[CD]
    )

    shacl_findings = [
        finding
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.SHACL
    ]
    assert shacl_findings, "blank-node violation was dropped from the report"
    assert any(finding.subject == str(anonymous) for finding in shacl_findings)


def test_catalog_blank_node_violation_stays_out_of_the_report() -> None:
    """Widening the filter must not admit blank nodes from the mixed-in catalog.

    Presence in the facts graph is the boundary, the same test the repair pass
    applies — not "blank nodes are always in scope".
    """
    shapes = RDFGraph()
    shape = URIRef(Q + "CatalogShape")
    numeric_prop = URIRef(Q + "cp_numeric")
    shapes.add((shape, RDF.type, SH.NodeShape))
    shapes.add((shape, SH.targetClass, VALUE_CLASS))
    shapes.add((shape, SH.property, numeric_prop))
    shapes.add((numeric_prop, SH.path, NUMERIC))
    shapes.add((numeric_prop, SH.minCount, Literal(1)))

    ontology = _unit_catalog()
    catalog_blank = BNode()
    ontology.add((catalog_blank, RDF.type, VALUE_CLASS))

    graph = RDFGraph()
    graph.add((URIRef(CD + "obs"), NUMERIC, Literal("1", datatype=XSD.decimal)))

    report = validate_aggregated_facts(
        graph, ontology, shapes_graph=shapes, fact_namespaces=[CD]
    )

    assert not [
        finding
        for finding in report.findings
        if finding.kind == FactsValidationFindingKind.SHACL
        and finding.subject == str(catalog_blank)
    ]
