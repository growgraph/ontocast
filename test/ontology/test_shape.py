"""Shape measures and the whole-catalog name index."""

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef
from rdflib.namespace import SOSA

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_validation.shape import (
    build_catalog_surface_index,
    has_own_structure,
    label_tokens,
    minted_class_iris,
    named_class_depth,
    resolve_unique,
    specific_label_reason,
    value_set_families,
)

pytestmark = pytest.mark.unit

EX = "https://example.com/onto#"


def _chain(*names: str) -> RDFGraph:
    """``names[0] ⊂ names[1] ⊂ …``, every class declared."""
    graph = RDFGraph()
    for name in names:
        graph.add((URIRef(EX + name), RDF.type, OWL.Class))
    for child, parent in zip(names, names[1:]):
        graph.add((URIRef(EX + child), RDFS.subClassOf, URIRef(EX + parent)))
    return graph


def test_depth_counts_steps_to_a_class_with_no_named_parent() -> None:
    graph = _chain("C", "B", "A")
    depth, chain = named_class_depth(URIRef(EX + "C"), graph)
    assert depth == 2
    assert chain == [URIRef(EX + n) for n in ("C", "B", "A")]


def test_depth_stops_at_a_well_known_term() -> None:
    graph = _chain("C", "B")
    graph.add((URIRef(EX + "B"), RDFS.subClassOf, SOSA.Sample))
    # A well-known term ends the chain even if the graph places it deeper.
    graph.add((SOSA.Sample, RDFS.subClassOf, URIRef(EX + "Z")))
    depth, chain = named_class_depth(URIRef(EX + "C"), graph)
    assert depth == 2
    assert chain[-1] == SOSA.Sample


def test_a_restriction_parent_is_not_a_step() -> None:
    graph = _chain("C", "B")
    restriction = BNode()
    graph.add((URIRef(EX + "C"), RDFS.subClassOf, restriction))
    graph.add((restriction, RDF.type, OWL.Restriction))
    depth, _ = named_class_depth(URIRef(EX + "C"), graph)
    assert depth == 1


def test_depth_takes_the_longest_of_several_parents() -> None:
    graph = _chain("C", "B", "A")
    graph.add((URIRef(EX + "C"), RDFS.subClassOf, URIRef(EX + "Root")))
    depth, _ = named_class_depth(URIRef(EX + "C"), graph)
    assert depth == 2


def test_a_cycle_terminates() -> None:
    graph = _chain("A", "B", "C")
    graph.add((URIRef(EX + "C"), RDFS.subClassOf, URIRef(EX + "A")))
    depth, _ = named_class_depth(URIRef(EX + "A"), graph)
    assert depth == 2


def test_label_tokens_keep_hyphenated_words() -> None:
    assert label_tokens("Zero-dimensional  Perovskite") == [
        "zero-dimensional",
        "perovskite",
    ]


def test_three_siblings_one_word_apart_form_a_family() -> None:
    families = value_set_families(
        {
            "p": [
                ("a", "Cesium perovskite"),
                ("b", "Formamidinium perovskite"),
                ("c", "Methylammonium perovskite"),
            ]
        }
    )
    assert len(families) == 1
    assert families[0].template == "* perovskite"
    assert [iri for iri, _ in families[0].members] == ["a", "b", "c"]


def test_two_siblings_are_not_a_family() -> None:
    assert value_set_families({"p": [("a", "Red car"), ("b", "Blue car")]}) == []


def test_siblings_two_words_apart_are_not_a_family() -> None:
    families = value_set_families(
        {"p": [("a", "Red fast car"), ("b", "Blue slow car"), ("c", "Green old car")]}
    )
    assert families == []


def test_single_word_labels_are_not_a_family() -> None:
    families = value_set_families({"p": [("a", "Car"), ("b", "Bus"), ("c", "Van")]})
    assert families == []


def test_a_child_is_counted_in_one_family_only() -> None:
    families = value_set_families(
        {
            "p": [
                ("a", "red car"),
                ("b", "blue car"),
                ("c", "green car"),
                ("d", "red van"),
                ("e", "red bus"),
            ]
        }
    )
    members = [iri for family in families for iri, _ in family.members]
    assert sorted(members) == sorted(set(members))


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("Detector", None),
        ("Perovskite sample", None),
        ("Hybrid organic-inorganic perovskite", None),
        ("XRD", None),
        ("Sample 3", "digit"),
        ("CsPbBr3 film", "digit"),
        ("NaCl crystal", "formula"),
        ("thermally evaporated gold back contact", "long"),
    ],
)
def test_specific_label_reason(label: str, reason: str | None) -> None:
    assert specific_label_reason(label) == reason


def test_own_structure_excludes_defined_and_used_classes() -> None:
    graph = _chain("C", "B")
    c = URIRef(EX + "C")
    assert not has_own_structure(c, graph)
    assert has_own_structure(URIRef(EX + "B"), graph)  # has a subclass
    graph.add((URIRef(EX + "p"), RDFS.domain, c))
    assert has_own_structure(c, graph)


def test_minted_class_iris_reads_declarations_and_hierarchy() -> None:
    graph = RDFGraph()
    graph.add((URIRef(EX + "A"), RDF.type, OWL.Class))
    graph.add((URIRef(EX + "B"), RDFS.subClassOf, URIRef(EX + "A")))
    graph.add((URIRef(EX + "p"), RDF.type, OWL.ObjectProperty))
    assert minted_class_iris(graph) == {EX + "A", EX + "B"}


def test_index_matches_labels_case_insensitively_and_local_names() -> None:
    graph = RDFGraph()
    term = URIRef(EX + "plRedShift")
    graph.add((term, RDF.type, OWL.DatatypeProperty))
    graph.add((term, RDFS.label, Literal("PL red  shift", lang="en")))
    index = build_catalog_surface_index([graph])
    assert resolve_unique(index, label="pl red shift") == (str(term), "label")
    assert resolve_unique(index, local="PLRedShift") == (str(term), "local_name")
    assert str(term) in index.properties


def test_an_ambiguous_name_resolves_to_nothing() -> None:
    graph = RDFGraph()
    for namespace in ("https://a.example/#", "https://b.example/#"):
        graph.add((URIRef(namespace + "Sample"), RDFS.label, Literal("Sample")))
    index = build_catalog_surface_index([graph])
    assert resolve_unique(index, label="sample", local="Sample") is None


def test_variant_pairs_need_the_same_head_and_role():
    from ontocast.tool.ontology_validation.shape import variant_term_pairs

    ns = "https://example.com/x#"
    minted = {
        f"{ns}hasOpticalMicrocavity": "property",
        f"{ns}hasInitialPhase": "property",
        f"{ns}hasOpticalModeEnergy": "property",
        f"{ns}OpticalCavity": "class",
    }
    pool = {
        **minted,
        f"{ns}hasOpticalCavity": "property",
        f"{ns}hasPhase": "property",
        f"{ns}hasOpticalMode": "property",
    }
    assert variant_term_pairs(minted, pool) == [
        (f"{ns}hasOpticalCavity", f"{ns}hasOpticalMicrocavity"),
        (f"{ns}hasPhase", f"{ns}hasInitialPhase"),
    ]
