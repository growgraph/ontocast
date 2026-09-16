"""Guards against a finished run that never had a vocabulary.

The failure these cover shipped once: every content unit rendered against an
empty ontology chapter, the extractor fell back on generic vocabulary, and the
SHACL gate reported ``conforms: true`` because no node in the output matched
any shape target. The run looked clean at every checkpoint.
"""

import pytest

from ontocast.onto.model import FactsValidationFinding, FactsValidationFindingKind
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.facts_validation import (
    count_shacl_focus_nodes,
    summarize_conformance,
)
from ontocast.tool.facts_validation.terms import collect_catalog_terms
from ontocast.tool.facts_validation.unit_findings import (
    _domain_adherence_findings,
    domain_vocabulary_share,
)

MATSCI = "https://growgraph.dev/ontologies/matsci#"
CD = "https://growgraph.dev/facts/"

_SHAPES = f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix matsci: <{MATSCI}> .
matsci:SampleShape a sh:NodeShape ; sh:targetClass matsci:NanocrystalSample .
"""

_ONTOLOGY = f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix matsci: <{MATSCI}> .
matsci:NanocrystalSample a owl:Class ; rdfs:label "Nanocrystal sample" .
matsci:SpecialSample a owl:Class ; rdfs:subClassOf matsci:NanocrystalSample .
matsci:hasAmount a owl:ObjectProperty ; rdfs:label "has amount" .
"""

_GROUNDED = f"""
@prefix cd: <{CD}> .
@prefix matsci: <{MATSCI}> .
cd:s1 a matsci:NanocrystalSample ; matsci:hasAmount cd:a1 .
"""

_UNGROUNDED = f"""
@prefix cd: <{CD}> .
@prefix schema: <https://schema.org/> .
cd:s1 a schema:Product ; schema:additionalProperty cd:pv1 .
cd:pv1 a schema:PropertyValue ; schema:value "30" ; schema:unitText "meV" .
"""


def _g(ttl: str) -> RDFGraph:
    graph = RDFGraph()
    graph.parse(data=ttl, format="turtle")
    return graph


def test_conformance_over_zero_focus_nodes_is_not_a_pass() -> None:
    summary = summarize_conformance([], shacl_evaluated=True, focus_nodes=0)

    assert summary["shacl_vacuous"] is True
    assert summary["conforms"] is None, "no violations over nothing is not conformance"


def test_conformance_over_real_focus_nodes_still_passes() -> None:
    summary = summarize_conformance([], shacl_evaluated=True, focus_nodes=7)

    assert summary["shacl_vacuous"] is False
    assert summary["conforms"] is True
    assert summary["shacl_focus_nodes"] == 7


def test_violations_still_fail_a_non_vacuous_run() -> None:
    finding = FactsValidationFinding(
        kind=FactsValidationFindingKind.SHACL, message="missing unit", severity="error"
    )
    summary = summarize_conformance([finding], shacl_evaluated=True, focus_nodes=3)

    assert summary["conforms"] is False


def test_focus_nodes_counts_targeted_instances_and_subclasses() -> None:
    shapes = _g(_SHAPES)
    data = _g(_ONTOLOGY + _GROUNDED)

    assert count_shacl_focus_nodes(data, shapes) == 1

    subclassed = _g(
        _ONTOLOGY + f"@prefix cd: <{CD}> .\n@prefix matsci: <{MATSCI}> .\n"
        "cd:s2 a matsci:SpecialSample .\n"
    )
    assert count_shacl_focus_nodes(subclassed, shapes) == 1, (
        "sh:targetClass is subclass-aware, so a subclass instance is in scope"
    )


def test_a_generic_graph_matches_no_shape_and_is_reported_vacuous() -> None:
    """The exact shape of the shipped failure."""
    focus = count_shacl_focus_nodes(_g(_UNGROUNDED), _g(_SHAPES))
    summary = summarize_conformance([], shacl_evaluated=True, focus_nodes=focus)

    assert focus == 0
    assert summary["conforms"] is None
    assert summary["shacl_vacuous"] is True


def test_focus_nodes_is_none_without_shapes() -> None:
    assert count_shacl_focus_nodes(_g(_GROUNDED), None) is None


@pytest.mark.parametrize(
    ("ttl", "expect_finding"),
    [(_GROUNDED, False), (_UNGROUNDED, True)],
)
def test_domain_adherence_flags_a_wholly_generic_render(
    ttl: str, expect_finding: bool
) -> None:
    catalog = collect_catalog_terms(_g(_ONTOLOGY))
    findings = _domain_adherence_findings(_g(ttl), catalog, [CD], 0.15)

    assert bool(findings) is expect_finding
    if expect_finding:
        assert findings[0].mandatory, "a vocabulary-less render must drive a repair"


def test_domain_adherence_is_silent_without_a_catalog() -> None:
    """A deployment that deliberately extracts without one is not spammed."""
    assert _domain_adherence_findings(_g(_UNGROUNDED), set(), [CD], 0.15) == []


def test_domain_adherence_can_be_disabled() -> None:
    catalog = collect_catalog_terms(_g(_ONTOLOGY))
    assert _domain_adherence_findings(_g(_UNGROUNDED), catalog, [CD], 0.0) == []


_FRONT_MATTER = f"""
@prefix cd: <{CD}> .
@prefix schema: <https://schema.org/> .
cd:doi_1 a schema:PropertyValue ; schema:value "10.1000/xyz" .
cd:author_1 a schema:Person .
"""


def test_domain_adherence_needs_a_minimum_denominator() -> None:
    """Two generic terms are not a render that abandoned the catalog.

    A front-matter unit (an identifier, an author) types a couple of nodes
    with generic vocabulary and has nothing to say in the catalog's terms.
    Judging a share over that denominator raised a mandatory finding the
    critic answered by retyping the identifier as a quantity value.
    """
    catalog = collect_catalog_terms(_g(_ONTOLOGY))
    graph = _g(_FRONT_MATTER)
    _, total = domain_vocabulary_share(graph, catalog, [CD])
    assert total < 4

    assert _domain_adherence_findings(graph, catalog, [CD], 0.15) == []
    assert _domain_adherence_findings(graph, catalog, [CD], 0.15, min_terms=1), (
        "the floor, not the share, is what silenced it"
    )


def test_domain_adherence_is_not_judged_on_citation_metadata() -> None:
    """A reference list is rendered with the bibliographic vocabulary by
    instruction, so its catalog share is zero by design."""
    from ontocast.tool.facts_validation import collect_unit_findings

    def findings(is_citation_metadata: bool):
        return [
            finding
            for finding in collect_unit_findings(
                graph=_g(_UNGROUNDED),
                ontology_graph=_g(_ONTOLOGY),
                quarantined=[],
                extraction_text="",
                fact_namespaces=[CD],
                coverage_limit=0,
                is_citation_metadata=is_citation_metadata,
            )
            if finding.kind == "domain_adherence"
        ]

    assert findings(False), "the same graph is flagged as a content unit"
    assert findings(True) == []


def test_share_excludes_plumbing_and_minted_instances() -> None:
    """rdf:type and rdfs:label are in every graph and in most catalogs.

    Counting them would let any graph clear the floor for free.
    """
    catalog = collect_catalog_terms(_g(_ONTOLOGY))
    from_catalog, total = domain_vocabulary_share(_g(_GROUNDED), catalog, [CD])

    assert (from_catalog, total) == (2, 2), (
        "matsci:NanocrystalSample + matsci:hasAmount"
    )
