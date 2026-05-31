"""Tests for induced-subgraph and schema-centric snapshot assembly."""

from rdflib import BNode, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.representation_text import ROLE_PREDICATE, ROLE_RESOURCE
from ontocast.tool.sparql import (
    SPARQLTool,
    _build_concept_relevance,
    _classify_and_promote_seeds,
    _crosslink_property_seeds,
    _filter_overbroad_namespace_map,
    _find_schema_uri_connected_components,
    _prune_degenerate_restriction_bnodes,
    _prune_disconnected_uri_entities,
    _prune_orphaned_bnode_subjects,
    _strip_redundant_generic_types,
)
from ontocast.tool.vector_store.core import GraphAtom
from ontocast.tool.vector_store.patch_retriever import _ranked_entity_weights

BASE = "https://growgraph.dev/ontologies/"
QQVAL = Namespace(f"{BASE}qqval#")
MATSCI = Namespace(f"{BASE}matsci-ontology#")
PEROV = Namespace(f"{BASE}perovskitemat#")


def _ontology(iri: str, graph: RDFGraph) -> Ontology:
    return Ontology(
        iri=iri,
        graph=graph,
        title=iri.rsplit("/", 1)[-1],
    )


def test_bind_implicit_namespaces_skips_parent_directory_stem() -> None:
    graph = RDFGraph()
    graph.bind("qqval", QQVAL)
    graph.add((URIRef(f"{BASE}qqval"), RDF.type, OWL.Ontology))
    graph.add((URIRef(f"{BASE}perovskitemat"), RDF.type, OWL.Ontology))
    graph.add((QQVAL["Approximate"], RDF.type, OWL.NamedIndividual))

    graph.bind_implicit_namespaces(prefix_base="qqval")

    bound = {prefix: str(ns) for prefix, ns in graph.namespaces() if prefix}
    assert "qqval_ontologies" not in bound
    assert bound["qqval"] == str(QQVAL)


def test_filter_overbroad_namespace_map_drops_parent_directory_uri() -> None:
    ns_map = {
        "qqval_ontologies": f"{BASE}",
        "qqval": str(QQVAL),
        "matsci-ontology": str(MATSCI),
        "perovskitemat": str(PEROV),
    }
    filtered = _filter_overbroad_namespace_map(ns_map)
    assert "qqval_ontologies" not in filtered
    assert filtered["qqval"] == str(QQVAL)
    assert filtered["matsci-ontology"] == str(MATSCI)


def test_prune_orphaned_bnode_subjects_removes_unreferenced_restrictions() -> None:
    graph = RDFGraph()
    orphan = BNode("orphan")
    parent = BNode("parent")
    prop = PEROV["hasASiteComponent"]
    graph.add((orphan, OWL.onProperty, prop))
    graph.add((parent, OWL.onProperty, prop))
    graph.add((PEROV["Perovskite"], RDFS.subClassOf, parent))

    _prune_orphaned_bnode_subjects(graph)

    assert (orphan, OWL.onProperty, prop) not in graph
    assert (parent, OWL.onProperty, prop) in graph


def test_ranked_entity_weights_preserves_entity_role() -> None:
    atoms = [
        GraphAtom(
            atom_id="a1",
            ontology_iri=f"{BASE}matsci-ontology",
            iri=str(MATSCI["usesMethod"]),
            entity_role=ROLE_PREDICATE,
            core_representation="uses method",
            neighborhood_representation="",
            score=0.9,
        ),
        GraphAtom(
            atom_id="a2",
            ontology_iri=f"{BASE}matsci-ontology",
            iri=str(MATSCI["Material"]),
            entity_role=ROLE_RESOURCE,
            core_representation="Material",
            neighborhood_representation="",
            score=0.8,
        ),
    ]
    ranked, scores, roles = _ranked_entity_weights(atoms)
    assert ranked[0] == str(MATSCI["usesMethod"])
    assert roles[str(MATSCI["usesMethod"])] == ROLE_PREDICATE
    assert roles[str(MATSCI["Material"])] == ROLE_RESOURCE
    assert scores[str(MATSCI["usesMethod"])] == 0.9


def test_classify_promotes_named_individual_to_domain_class() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    individual = MATSCI["Photoluminescence"]
    method_class = MATSCI["OpticalCharacterizationMethod"]
    graph.add((individual, RDF.type, OWL.NamedIndividual))
    graph.add((individual, RDF.type, method_class))
    graph.add((method_class, RDF.type, OWL.Class))
    graph.add((method_class, RDFS.label, Literal("Optical characterization method")))

    ontology_subjects: frozenset[str] = frozenset()
    concept, props = _classify_and_promote_seeds(
        [str(individual)],
        graph,
        {str(individual): ROLE_RESOURCE},
        ontology_subjects,
    )
    assert str(method_class) in concept
    assert str(individual) not in concept
    assert not props


def test_crosslink_adds_property_by_domain() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    method_class = MATSCI["OpticalCharacterizationMethod"]
    uses_method = MATSCI["usesMethod"]
    graph.add((method_class, RDF.type, OWL.Class))
    graph.add((uses_method, RDF.type, OWL.ObjectProperty))
    graph.add((uses_method, RDFS.domain, method_class))
    graph.add((uses_method, RDFS.label, Literal("uses method")))

    linked = _crosslink_property_seeds(
        graph,
        [str(method_class)],
        [],
        frozenset(),
    )
    assert str(uses_method) in linked


def test_strip_redundant_named_individual_type() -> None:
    graph = RDFGraph()
    entity = MATSCI["PLE"]
    method_class = MATSCI["OpticalCharacterizationMethod"]
    graph.add((entity, RDF.type, OWL.NamedIndividual))
    graph.add((entity, RDF.type, method_class))

    _strip_redundant_generic_types(graph)

    assert (entity, RDF.type, OWL.NamedIndividual) not in graph
    assert (entity, RDF.type, method_class) in graph


def test_build_induced_subgraph_schema_centric_connected_patch() -> None:
    matsci_graph = RDFGraph()
    matsci_graph.bind("matsci-ontology", MATSCI)
    matsci_graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))
    matsci_graph.add(
        (URIRef(f"{BASE}matsci-ontology"), DCTERMS.creator, Literal("growgraph.dev"))
    )

    method_class = MATSCI["OpticalCharacterizationMethod"]
    char_class = MATSCI["CharacterizationMethod"]
    individual = MATSCI["Photoluminescence"]
    uses_method = MATSCI["usesMethod"]

    matsci_graph.add((method_class, RDF.type, OWL.Class))
    matsci_graph.add(
        (method_class, RDFS.label, Literal("Optical characterization method"))
    )
    matsci_graph.add((method_class, RDFS.subClassOf, char_class))
    matsci_graph.add((char_class, RDF.type, OWL.Class))
    matsci_graph.add((char_class, RDFS.label, Literal("Characterization method")))

    matsci_graph.add((individual, RDF.type, OWL.NamedIndividual))
    matsci_graph.add((individual, RDF.type, method_class))

    matsci_graph.add((uses_method, RDF.type, OWL.ObjectProperty))
    matsci_graph.add((uses_method, RDFS.label, Literal("uses method")))
    matsci_graph.add((uses_method, RDFS.domain, char_class))
    matsci_graph.add((uses_method, RDFS.range, method_class))

    ontologies = [_ontology(f"{BASE}matsci-ontology", matsci_graph)]
    entity_uris = [str(individual)]
    entity_roles = {str(individual): ROLE_RESOURCE}

    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=entity_uris,
        entity_relevance={str(individual): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles=entity_roles,
    )

    assert (uses_method, RDFS.domain, char_class) in result
    assert (uses_method, RDFS.range, method_class) in result
    assert (
        method_class,
        RDFS.label,
        Literal("Optical characterization method"),
    ) in result
    assert (individual, RDF.type, OWL.NamedIndividual) not in result
    assert not any(
        str(s).endswith("matsci-ontology") and p == RDF.type
        for s, p, o in result
        if o == OWL.Ontology
    )
    assert not any(p == DCTERMS.creator for _, p, _ in result)
    assert len(_find_schema_uri_connected_components(result)) == 1


def test_build_concept_relevance_inherits_individual_score_to_class() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    individual = MATSCI["Photoluminescence"]
    method_class = MATSCI["OpticalCharacterizationMethod"]
    graph.add((individual, RDF.type, OWL.NamedIndividual))
    graph.add((individual, RDF.type, method_class))

    relevance = _build_concept_relevance(
        [str(individual)],
        graph,
        {str(individual): 0.75},
        frozenset(),
    )
    assert relevance[str(method_class)] == 0.75


def test_build_induced_subgraph_class_seed_includes_label_and_subclass() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))
    bare_class = MATSCI["ActuatingEquipment"]
    parent = MATSCI["ProcessingEquipment"]
    graph.add((bare_class, RDF.type, OWL.Class))
    graph.add((bare_class, RDFS.label, Literal("Actuating equipment")))
    graph.add((bare_class, RDFS.subClassOf, parent))
    graph.add((parent, RDF.type, OWL.Class))
    graph.add((parent, RDFS.label, Literal("Processing equipment")))

    ontologies = [_ontology(f"{BASE}matsci-ontology", graph)]
    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(bare_class)],
        entity_relevance={str(bare_class): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(bare_class): ROLE_RESOURCE},
        hub_seed_count=1,
        ancestor_closure_depth=2,
    )
    assert (bare_class, RDFS.label, Literal("Actuating equipment")) in result
    assert (bare_class, RDFS.subClassOf, parent) in result
    assert (parent, RDFS.label, Literal("Processing equipment")) in result


def test_build_induced_subgraph_shared_ancestor_connects_two_seeds() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))
    root = MATSCI["CharacterizationMethod"]
    child_a = MATSCI["OpticalCharacterizationMethod"]
    child_b = MATSCI["StructuralCharacterizationMethod"]
    for cls, label in (
        (root, "Characterization method"),
        (child_a, "Optical characterization method"),
        (child_b, "Structural characterization method"),
    ):
        graph.add((cls, RDF.type, OWL.Class))
        graph.add((cls, RDFS.label, Literal(label)))
    graph.add((child_a, RDFS.subClassOf, root))
    graph.add((child_b, RDFS.subClassOf, root))

    ontologies = [_ontology(f"{BASE}matsci-ontology", graph)]
    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(child_a), str(child_b)],
        entity_relevance={str(child_a): 1.0, str(child_b): 0.9},
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={
            str(child_a): ROLE_RESOURCE,
            str(child_b): ROLE_RESOURCE,
        },
        hub_seed_count=2,
        ancestor_closure_depth=2,
    )
    assert (child_a, RDFS.subClassOf, root) in result
    assert (child_b, RDFS.subClassOf, root) in result
    assert (root, RDFS.label, Literal("Characterization method")) in result


def test_build_induced_subgraph_late_seed_gets_subclass_under_tight_budget() -> None:
    """AssemblyProcess-like: label+comment must not appear without subClassOf."""
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))

    process = MATSCI["Process"]
    assembly = MATSCI["AssemblyProcess"]
    graph.add((process, RDF.type, OWL.Class))
    graph.add((process, RDFS.label, Literal("Process")))
    graph.add((assembly, RDF.type, OWL.Class))
    graph.add((assembly, RDFS.label, Literal("Assembly process")))
    graph.add(
        (assembly, RDFS.comment, Literal("A concrete execution of an assembly method."))
    )
    graph.add((assembly, RDFS.subClassOf, process))

    filler_seeds: list[str] = []
    for idx in range(12):
        cls = MATSCI[f"FillerClass{idx}"]
        parent = MATSCI[f"FillerParent{idx}"]
        graph.add((cls, RDF.type, OWL.Class))
        graph.add((cls, RDFS.label, Literal(f"Filler {idx}")))
        graph.add((cls, RDFS.comment, Literal(f"Comment for filler {idx}.")))
        graph.add((cls, RDFS.subClassOf, parent))
        graph.add((parent, RDF.type, OWL.Class))
        graph.add((parent, RDFS.label, Literal(f"Parent {idx}")))
        filler_seeds.append(str(cls))

    ontologies = [_ontology(f"{BASE}matsci-ontology", graph)]
    entity_uris = filler_seeds + [str(assembly)]
    entity_relevance = {uri: 1.0 - (idx * 0.05) for idx, uri in enumerate(entity_uris)}
    entity_relevance[str(assembly)] = 0.05

    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=entity_uris,
        entity_relevance=entity_relevance,
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=80,
        estimated_triples_per_query=12,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={uri: ROLE_RESOURCE for uri in entity_uris},
        hub_seed_count=8,
        ancestor_closure_depth=2,
    )

    assert (assembly, RDFS.label, Literal("Assembly process")) in result
    assert (
        assembly,
        RDFS.comment,
        Literal("A concrete execution of an assembly method."),
    ) in result
    assert (assembly, RDFS.subClassOf, process) in result


def test_build_induced_subgraph_bfs_class_node_includes_subclass() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))

    root = MATSCI["Process"]
    assembly = MATSCI["AssemblyProcess"]
    graph.add((root, RDF.type, OWL.Class))
    graph.add((root, RDFS.label, Literal("Process")))
    graph.add((assembly, RDF.type, OWL.Class))
    graph.add((assembly, RDFS.label, Literal("Assembly process")))
    graph.add(
        (assembly, RDFS.comment, Literal("A concrete execution of an assembly method."))
    )
    graph.add((assembly, RDFS.subClassOf, root))

    ontologies = [_ontology(f"{BASE}matsci-ontology", graph)]
    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(root)],
        entity_relevance={str(root): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(root): ROLE_RESOURCE},
        hub_seed_count=1,
        ancestor_closure_depth=1,
    )

    assert (assembly, RDFS.label, Literal("Assembly process")) in result
    assert (assembly, RDFS.subClassOf, root) in result


def test_strip_redundant_owl_class_when_subclass_present() -> None:
    graph = RDFGraph()
    parent = MATSCI["Process"]
    cls = MATSCI["AssemblyProcess"]
    graph.add((cls, RDF.type, OWL.Class))
    graph.add((cls, RDFS.subClassOf, parent))

    _strip_redundant_generic_types(graph)

    assert (cls, RDF.type, OWL.Class) not in graph
    assert (cls, RDFS.subClassOf, parent) in graph


def test_prune_disconnected_uri_literal_island() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    island = MATSCI["OrphanClass"]
    root = MATSCI["Process"]
    other = MATSCI["OtherClass"]
    uses = MATSCI["usesMethod"]
    graph.add((island, RDFS.label, Literal("Orphan")))
    graph.add((root, RDFS.label, Literal("Process")))
    graph.add((uses, RDFS.domain, root))
    graph.add((uses, RDFS.range, other))

    pruned = _prune_disconnected_uri_entities(graph, {str(root)})

    assert pruned == 1
    assert (island, RDFS.label, Literal("Orphan")) not in graph
    assert (root, RDFS.label, Literal("Process")) in graph


def test_quantity_range_no_empty_restriction_bnodes() -> None:
    graph = RDFGraph()
    graph.bind("qqval", QQVAL)
    graph.add((URIRef(f"{BASE}qqval"), RDF.type, OWL.Ontology))

    quantity_value = URIRef(f"{BASE}qudt#QuantityValue")
    parent = QQVAL["Quantity"]
    cls = QQVAL["QuantityRange"]
    prop_lower = QQVAL["hasLowerBound"]
    prop_upper = QQVAL["hasUpperBound"]

    graph.add((parent, RDF.type, OWL.Class))
    graph.add((parent, RDFS.label, Literal("Quantity")))
    graph.add((cls, RDF.type, OWL.Class))
    graph.add((cls, RDFS.label, Literal("Quantity range")))
    graph.add((cls, RDFS.comment, Literal("A quantitative interval.")))
    graph.add((cls, RDFS.subClassOf, parent))

    restriction_lower = BNode("restriction_lower")
    graph.add((cls, RDFS.subClassOf, restriction_lower))
    graph.add((restriction_lower, RDF.type, OWL.Restriction))
    graph.add((restriction_lower, OWL.onProperty, prop_lower))
    graph.add((restriction_lower, OWL.someValuesFrom, quantity_value))
    graph.add((prop_lower, RDF.type, OWL.ObjectProperty))
    graph.add((prop_lower, RDFS.label, Literal("has lower bound")))

    restriction_upper = BNode("restriction_upper")
    graph.add((cls, RDFS.subClassOf, restriction_upper))
    graph.add((restriction_upper, RDF.type, OWL.Restriction))
    graph.add((restriction_upper, OWL.onProperty, prop_upper))
    graph.add((restriction_upper, OWL.someValuesFrom, quantity_value))

    stub_restriction = BNode("stub")
    graph.add((cls, RDFS.subClassOf, stub_restriction))
    graph.add((stub_restriction, RDF.type, OWL.Class))

    ontologies = [_ontology(f"{BASE}qqval", graph)]
    result, metrics = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(cls)],
        entity_relevance={str(cls): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(cls): ROLE_RESOURCE},
        hub_seed_count=1,
        ancestor_closure_depth=2,
    )

    assert (cls, RDFS.subClassOf, parent) in result
    assert (cls, RDFS.subClassOf, stub_restriction) not in result
    for bnode in (restriction_lower, restriction_upper):
        if (cls, RDFS.subClassOf, bnode) in result:
            assert any(
                pred == OWL.onProperty
                for _, pred, _ in result.triples((bnode, None, None))
            )
    turtle = result.serialize(format="turtle")
    if isinstance(turtle, bytes):
        turtle = turtle.decode("utf-8")
    assert "subClassOf [ ]" not in turtle
    assert "subClassOf [ a owl:Class ]" not in turtle


def test_prune_degenerate_restriction_bnodes_removes_stub() -> None:
    graph = RDFGraph()
    cls = QQVAL["QuantityRange"]
    stub = BNode("stub")
    graph.add((cls, RDFS.subClassOf, stub))
    graph.add((stub, RDF.type, OWL.Class))

    dropped = _prune_degenerate_restriction_bnodes(graph)

    assert dropped == 1
    assert (cls, RDFS.subClassOf, stub) not in graph


def test_domain_range_linker_connects_bare_class() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))

    method_class = MATSCI["OpticalCharacterizationMethod"]
    uses_method = MATSCI["usesMethod"]
    graph.add((method_class, RDF.type, OWL.Class))
    graph.add((method_class, RDFS.label, Literal("Optical characterization method")))
    graph.add((uses_method, RDF.type, OWL.ObjectProperty))
    graph.add((uses_method, RDFS.label, Literal("uses method")))
    graph.add((uses_method, RDFS.domain, method_class))
    graph.add((uses_method, RDFS.range, MATSCI["CharacterizationMethod"]))

    ontologies = [_ontology(f"{BASE}matsci-ontology", graph)]
    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(method_class)],
        entity_relevance={str(method_class): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=0,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(method_class): ROLE_RESOURCE},
        hub_seed_count=1,
        ancestor_closure_depth=1,
    )

    assert (uses_method, RDFS.domain, method_class) in result
    assert len(_find_schema_uri_connected_components(result)) == 1


def test_property_only_path_runs_finalization() -> None:
    graph = RDFGraph()
    graph.bind("matsci-ontology", MATSCI)
    graph.add((URIRef(f"{BASE}matsci-ontology"), RDF.type, OWL.Ontology))

    method_class = MATSCI["OpticalCharacterizationMethod"]
    uses_method = MATSCI["usesMethod"]
    graph.add((method_class, RDF.type, OWL.Class))
    graph.add((method_class, RDFS.label, Literal("Optical characterization method")))
    graph.add((uses_method, RDF.type, OWL.ObjectProperty))
    graph.add((uses_method, RDFS.label, Literal("uses method")))
    graph.add((uses_method, RDFS.domain, method_class))

    ontologies = [_ontology(f"{BASE}matsci-ontology", graph)]
    result, metrics = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(uses_method)],
        entity_relevance={str(uses_method): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=1,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(uses_method): ROLE_PREDICATE},
        hub_seed_count=1,
        ancestor_closure_depth=1,
    )

    assert (uses_method, RDFS.domain, method_class) in result
    assert (
        method_class,
        RDFS.label,
        Literal("Optical characterization method"),
    ) in result
    assert "snapshot_uri_components" in metrics
    turtle = result.serialize(format="turtle")
    if isinstance(turtle, bytes):
        turtle = turtle.decode("utf-8")
    assert "subClassOf [ ]" not in turtle
