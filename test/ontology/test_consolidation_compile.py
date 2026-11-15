"""Term actions compiled to a graph patch: validation, defaults, each action."""

import pytest
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef

from ontocast.onto.model import OntologyConsolidationReport, OntologyTermAction
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_validation.consolidate import (
    compile_actions,
    resolve_actions,
)
from ontocast.tool.ontology_validation.plan import minted_terms
from ontocast.tool.ontology_validation.shape import build_catalog_surface_index

pytestmark = pytest.mark.unit

CAT = "https://example.com/cat#"
H = "https://example.com/home#"
PREFIXES = {"cat": CAT, "home": H}


def u(local: str, ns: str = H) -> URIRef:
    return URIRef(f"{ns}{local}")


def _catalog():
    graph = RDFGraph()
    graph.add((u("Sample", CAT), RDF.type, OWL.Class))
    graph.add((u("Sample", CAT), RDFS.label, Literal("Sample")))
    graph.add((u("hasPart", CAT), RDF.type, OWL.ObjectProperty))
    return build_catalog_surface_index([graph])


def _document() -> RDFGraph:
    graph = RDFGraph()
    for local in ("Film", "ThinFilm", "CsFilm", "Device", "Superradiance"):
        graph.add((u(local), RDF.type, OWL.Class))
        graph.add((u(local), RDFS.label, Literal(local)))
    graph.add((u("ThinFilm"), RDFS.subClassOf, u("Film")))
    graph.add((u("CsFilm"), RDFS.subClassOf, u("ThinFilm")))
    graph.add((u("hasFilm"), RDF.type, OWL.ObjectProperty))
    graph.add((u("hasFilm"), RDFS.range, u("CsFilm")))
    graph.add((u("hasFilm"), RDFS.domain, u("Device")))
    return graph


def _decide(actions, graph, candidates=frozenset()):
    catalog = _catalog()
    minted = minted_terms([graph], catalog)
    decided, ignored = resolve_actions(
        actions,
        minted=minted,
        candidates=set(candidates),
        catalog_index=catalog,
        prefixes=PREFIXES,
    )
    return compile_actions(decided, graph, minted=minted), ignored


def act(term, action, target=None):
    return OntologyTermAction(term=term, action=action, target=target)


def test_the_wire_accepts_action_synonyms():
    report = OntologyConsolidationReport.model_validate(
        {"actions": [{"term": "home:Film", "action": "Merge", "target": ""}]}
    )
    assert report.actions[0].action == "merge_into"
    assert report.actions[0].target is None


def test_defaults_keep_planned_terms_and_drop_candidates():
    graph = _document()
    compiled, ignored = _decide([], graph, candidates={str(u("Superradiance"))})
    assert ignored == []
    assert compiled.decided[str(u("Film"))].action == "keep"
    assert compiled.decided[str(u("Superradiance"))].action == "drop"
    compiled.apply_to(graph)
    assert (u("Superradiance"), None, None) not in graph
    assert (u("Film"), RDF.type, OWL.Class) in graph


def test_merge_into_a_catalog_term_rewrites_uses_and_removes_the_definition():
    graph = _document()
    compiled, _ = _decide([act("home:Device", "merge_into", "cat:Sample")], graph)
    compiled.apply_to(graph)
    assert (u("Device"), None, None) not in graph
    assert (u("hasFilm"), RDFS.domain, u("Sample", CAT)) in graph
    # The catalog term's own axioms are not touched.
    assert (u("Sample", CAT), None, None) not in graph


def test_merge_chains_follow_to_a_surviving_term_and_refuse_a_dropped_one():
    graph = _document()
    compiled, ignored = _decide(
        [
            act("CsFilm", "merge_into", "ThinFilm"),
            act("ThinFilm", "merge_into", "home:Film"),
            act("Device", "merge_into", "Superradiance"),
            act("Superradiance", "drop"),
        ],
        graph,
    )
    assert compiled.decided[str(u("CsFilm"))].target == str(u("Film"))
    assert compiled.decided[str(u("Device"))].action == "keep"
    assert ("Device", "merge target does not survive") in [
        (term.rsplit("#", 1)[-1], reason) for term, reason in ignored
    ]
    compiled.apply_to(graph)
    assert (u("hasFilm"), RDFS.range, u("Film")) in graph


def test_generalise_replaces_named_parents():
    graph = _document()
    compiled, _ = _decide([act("home:CsFilm", "generalise_to", "home:Film")], graph)
    compiled.apply_to(graph)
    assert set(graph.objects(u("CsFilm"), RDFS.subClassOf)) == {u("Film")}


def test_demote_turns_a_class_into_a_concept_and_moves_its_uses_up():
    graph = _document()
    compiled, _ = _decide([act("home:CsFilm", "demote_to_property_value")], graph)
    compiled.apply_to(graph)
    assert (u("CsFilm"), RDF.type, OWL.Class) not in graph
    assert (u("CsFilm"), RDF.type, SKOS.Concept) in graph
    assert (u("CsFilm"), SKOS.broader, u("ThinFilm")) in graph
    assert (u("hasFilm"), RDFS.range, u("ThinFilm")) in graph
    assert (u("hasFilm"), RDFS.range, u("CsFilm")) not in graph


def test_invalid_actions_are_ignored_and_fall_back_to_the_default():
    graph = _document()
    compiled, ignored = _decide(
        [
            act("cat:Sample", "drop"),
            act("home:Film", "merge_into", "cat:hasPart"),
            act("home:hasFilm", "demote_to_property_value"),
            act("home:Device", "drop"),
            act("home:Device", "keep"),
        ],
        graph,
    )
    reasons = sorted(reason for _, reason in ignored)
    assert reasons == [
        "invalid target for merge_into",
        "not a listed minted term",
        "only a class can be demoted",
        "second action for the term",
    ]
    assert compiled.decided[str(u("Film"))].action == "keep"
    assert compiled.decided[str(u("Device"))].action == "drop"


def test_the_patch_is_a_delete_then_insert_graph_update():
    graph = _document()
    compiled, _ = _decide([act("home:Device", "merge_into", "cat:Sample")], graph)
    update = compiled.to_graph_update()
    assert [op.type for op in update.triple_operations] == ["delete", "insert"]


def test_an_action_on_a_term_not_under_review_is_ignored():
    graph = _document()
    catalog = _catalog()
    minted = minted_terms([graph], catalog)
    decided, ignored = resolve_actions(
        [act("home:Device", "drop"), act("home:Film", "drop")],
        minted=minted,
        candidates=set(),
        catalog_index=catalog,
        prefixes=PREFIXES,
        reviewed={str(u("Film"))},
    )
    assert decided[str(u("Device"))].action == "keep"
    assert decided[str(u("Film"))].action == "drop"
    assert ignored == [("home:Device", "not up for review")]


def test_structural_flags_name_variants_value_sets_and_catalog_duplicates():
    from ontocast.tool.ontology_validation.consolidate import structural_flags

    graph = RDFGraph()
    for local, label in (
        ("CesiumFilm", "Cesium film"),
        ("LeadFilm", "Lead film"),
        ("TinFilm", "Tin film"),
        ("SampleCopy", "Sample"),
        ("Widget", "Widget"),
    ):
        graph.add((u(local), RDF.type, OWL.Class))
        graph.add((u(local), RDFS.label, Literal(label)))
    for local in ("CesiumFilm", "LeadFilm", "TinFilm"):
        graph.add((u(local), RDFS.subClassOf, u("Sample", CAT)))
    graph.add((u("hasPart"), RDF.type, OWL.ObjectProperty))
    graph.add((u("hasInnerPart"), RDF.type, OWL.ObjectProperty))
    catalog = _catalog()
    minted = minted_terms([graph], catalog)

    flags = structural_flags(minted, [graph], catalog)

    assert flags[str(u("CesiumFilm"))] == {"value_set": ()}
    assert flags[str(u("SampleCopy"))]["catalog_duplicate"] == (str(u("Sample", CAT)),)
    assert set(flags[str(u("hasInnerPart"))]["variant"]) == {
        str(u("hasPart")),
        str(u("hasPart", CAT)),
    }
    assert str(u("Widget")) not in flags


def test_a_minted_copy_of_a_well_known_term_is_flagged_with_it():
    from ontocast.tool.ontology_validation.consolidate import structural_flags

    graph = RDFGraph()
    graph.add((u("observation"), RDF.type, OWL.Class))
    graph.add((u("Volcano"), RDF.type, OWL.Class))
    graph.add((u("hasResult"), RDF.type, OWL.ObjectProperty))
    # Same name, other role: a class named like a well-known property.
    graph.add((u("HasResultSet"), RDF.type, OWL.Class))
    graph.add((u("Result"), RDF.type, OWL.Class))
    catalog = _catalog()

    flags = structural_flags(minted_terms([graph], catalog), [graph], catalog)

    assert flags[str(u("observation"))]["well_known_duplicate"] == (
        "http://www.w3.org/ns/sosa/Observation",
        "https://schema.org/Observation",
    )
    assert flags[str(u("Volcano"))]["well_known_duplicate"] == (
        "https://schema.org/Volcano",
    )
    assert flags[str(u("hasResult"))]["well_known_duplicate"] == (
        "http://www.w3.org/ns/sosa/hasResult",
    )
    assert str(u("HasResultSet")) not in flags


def test_a_well_known_target_must_exist_and_share_the_role():
    graph = _document()
    actions = [
        act("home:Film", "merge_into", "https://schema.org/NoSuchClass"),
        act("home:Device", "merge_into", "https://schema.org/name"),
        act("home:Superradiance", "merge_into", "https://schema.org/Event"),
    ]

    compiled, ignored = _decide(actions, graph)

    assert sorted(term for term, _ in ignored) == ["home:Device", "home:Film"]
    assert compiled.decided[str(u("Superradiance"))].target == (
        "https://schema.org/Event"
    )


def test_the_consolidation_prompt_shows_what_each_flag_points_at():
    from ontocast.agent.consolidate_ontology import flag_targets

    assert flag_targets(
        {
            "variant": ("https://example.com/home#hasPart",),
            "well_known_duplicate": (
                "http://www.w3.org/ns/sosa/Observation",
                "https://schema.org/Observation",
            ),
            "candidate": (),
        }
    ) == [
        ("https://example.com/home#hasPart", "name variant"),
        (
            "http://www.w3.org/ns/sosa/Observation",
            "same name in a well-known vocabulary",
        ),
        ("https://schema.org/Observation", "same name in a well-known vocabulary"),
    ]
