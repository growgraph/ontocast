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
    _find_schema_uri_connected_components,
    _interleave_by_group,
    _prune_degenerate_restriction_bnodes,
    _prune_disconnected_uri_entities,
    _prune_orphaned_bnode_subjects,
    _strip_redundant_generic_types,
    filter_overbroad_namespace_map,
)
from ontocast.tool.vector_store.core import GraphAtom
from ontocast.tool.vector_store.patch_retriever import _ranked_entity_weights

BASE = "https://growgraph.dev/ontologies/"
QQVAL = Namespace(f"{BASE}qqval#")
MATSCI = Namespace(f"{BASE}matsci#")
PEROV = Namespace(f"{BASE}perovskitemat#")


def _ontology(iri: str, graph: RDFGraph) -> Ontology:
    return Ontology(
        iri=iri,
        graph=graph,
        title=iri.rsplit("/", 1)[-1],
    )


def _assert_no_degenerate_restriction_bnodes(graph) -> None:
    """No `rdfs:subClassOf` bnode may be empty or a bare `owl:Class`.

    Asserted against the graph rather than against serialized Turtle: the old
    form matched the strings "subClassOf [ ]" / "subClassOf [ a owl:Class ]",
    which is rdflib's formatting rather than our contract, and would pass
    silently the day rdflib lays those out differently.
    """
    for _, _, bnode in graph.triples((None, RDFS.subClassOf, None)):
        if not isinstance(bnode, BNode):
            continue
        statements = {
            (pred, obj) for _, pred, obj in graph.triples((bnode, None, None))
        }
        assert statements, f"empty restriction bnode {bnode} survived"
        assert statements != {(RDF.type, OWL.Class)}, (
            f"bnode {bnode} carries only `a owl:Class`"
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


def test_bind_implicit_namespaces_own_stem_gets_plain_prefix() -> None:
    """An ontology's own namespace binds as ``matsci``, not ``matsci_matsci``."""
    graph = RDFGraph()
    graph.add((MATSCI["Material"], RDF.type, OWL.Class))
    graph.add((MATSCI["Method"], RDF.type, OWL.Class))
    graph.add((PEROV["Perovskite"], RDF.type, OWL.Class))
    graph.add((PEROV["Halide"], RDF.type, OWL.Class))

    graph.bind_implicit_namespaces(prefix_base="matsci")

    bound = {prefix: str(ns) for prefix, ns in graph.namespaces() if prefix}
    assert bound.get("matsci") == str(MATSCI)
    assert "matsci_matsci" not in bound
    # Foreign stems still get the disambiguating base.
    assert bound.get("matsci_perovskitemat") == str(PEROV)


def _lineage(iri: str, version: str | None, hash_: str | None):
    from ontocast.onto.ontology_header import OntologyHeader

    return OntologyHeader(iri=iri, graph_uri=iri, version=version, hash=hash_)


def test_select_relevant_ontologies_hash_mismatch_falls_back_same_version() -> None:
    """A hash filter selects among entries; it never empties an IRI wholesale.

    Graph hashes are unstable under serialization round-trips (literal lexical
    forms sit outside URDNA2015), so atom-payload hashes routinely disagree with
    catalog hashes computed in another process. The old exact-hash requirement
    silently dropped whole ontologies from the prompt context.
    """
    from ontocast.tool.sparql import select_relevant_ontologies

    entry = _lineage(f"{BASE}units", "1.0.0", "catalog-hash")
    selected = select_relevant_ontologies(
        [entry],
        [f"{BASE}units"],
        {f"{BASE}units": {"1.0.0"}},
        {f"{BASE}units": {"atoms-hash"}},
    )
    assert selected == [entry]


def test_select_relevant_ontologies_exact_hash_still_disambiguates_versions() -> None:
    from ontocast.tool.sparql import select_relevant_ontologies

    old = _lineage(f"{BASE}units", "1.0.0", "old-hash")
    new = _lineage(f"{BASE}units", "2.0.0", "new-hash")
    selected = select_relevant_ontologies(
        [old, new],
        [f"{BASE}units"],
        None,
        {f"{BASE}units": {"new-hash"}},
    )
    assert selected == [new]


def test_select_relevant_ontologies_version_mismatch_falls_back_to_all() -> None:
    from ontocast.tool.sparql import select_relevant_ontologies

    entry = _lineage(f"{BASE}units", "1.0.0", "h1")
    selected = select_relevant_ontologies(
        [entry],
        [f"{BASE}units"],
        {f"{BASE}units": {"9.9.9"}},
        None,
    )
    assert selected == [entry]


def test_select_relevant_ontologies_iri_filter_still_excludes() -> None:
    from ontocast.tool.sparql import select_relevant_ontologies

    wanted = _lineage(f"{BASE}units", "1.0.0", "h1")
    other = _lineage(f"{BASE}matsci", "1.0.0", "h2")
    selected = select_relevant_ontologies([wanted, other], [f"{BASE}units"], None, None)
    assert selected == [wanted]


def test_filter_overbroad_namespace_map_drops_parent_directory_uri() -> None:
    ns_map = {
        "qqval_ontologies": f"{BASE}",
        "qqval": str(QQVAL),
        "matsci": str(MATSCI),
        "perovskitemat": str(PEROV),
    }
    filtered = filter_overbroad_namespace_map(ns_map)
    assert "qqval_ontologies" not in filtered
    assert filtered["qqval"] == str(QQVAL)
    assert filtered["matsci"] == str(MATSCI)


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
            ontology_iri=f"{BASE}matsci",
            iri=str(MATSCI["usesMethod"]),
            entity_role=ROLE_PREDICATE,
            core_representation="uses method",
            neighborhood_representation="",
            score=0.9,
        ),
        GraphAtom(
            atom_id="a2",
            ontology_iri=f"{BASE}matsci",
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


def test_classify_keeps_named_individual_alongside_its_domain_class() -> None:
    graph = RDFGraph()
    graph.bind("matsci", MATSCI)
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
    # The class is added, but the individual retrieval actually matched is retained:
    # the facts contract expects pre-declared reference individuals to stay reusable.
    assert str(method_class) in concept
    assert str(individual) in concept
    assert not props


def test_crosslink_adds_property_by_domain() -> None:
    graph = RDFGraph()
    graph.bind("matsci", MATSCI)
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
    matsci_graph.bind("matsci", MATSCI)
    matsci_graph.add((URIRef(f"{BASE}matsci"), RDF.type, OWL.Ontology))
    matsci_graph.add(
        (URIRef(f"{BASE}matsci"), DCTERMS.creator, Literal("growgraph.dev"))
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

    ontologies = [_ontology(f"{BASE}matsci", matsci_graph)]
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
        str(s).endswith("matsci") and p == RDF.type
        for s, p, o in result
        if o == OWL.Ontology
    )
    assert not any(p == DCTERMS.creator for _, p, _ in result)
    assert len(_find_schema_uri_connected_components(result)) == 1


def test_build_concept_relevance_inherits_individual_score_to_class() -> None:
    graph = RDFGraph()
    graph.bind("matsci", MATSCI)
    individual = MATSCI["Photoluminescence"]
    method_class = MATSCI["OpticalCharacterizationMethod"]
    graph.add((individual, RDF.type, OWL.NamedIndividual))
    graph.add((individual, RDF.type, method_class))

    relevance, first_rank = _build_concept_relevance(
        [str(individual)],
        graph,
        {str(individual): 0.75},
        frozenset(),
    )
    # The retrieved individual keeps its own score; the promoted class inherits it.
    assert relevance[str(individual)] == 0.75
    assert relevance[str(method_class)] == 0.75
    assert first_rank[str(individual)] == 0
    assert first_rank[str(method_class)] == 0


def test_build_concept_relevance_type_promotion_factor_scales_class_only() -> None:
    graph = RDFGraph()
    individual = MATSCI["Photoluminescence"]
    method_class = MATSCI["OpticalCharacterizationMethod"]
    graph.add((individual, RDF.type, method_class))

    relevance, _ = _build_concept_relevance(
        [str(individual)],
        graph,
        {str(individual): 0.8},
        frozenset(),
        type_promotion_score_factor=0.5,
    )
    assert relevance[str(individual)] == 0.8
    assert relevance[str(method_class)] == 0.4


def test_snapshot_binds_only_used_prefixes() -> None:
    """Prefixes of merged ontologies that contribute no triple stay unbound."""
    munits = Namespace(f"{BASE}matsci-units#")
    graph = RDFGraph()
    graph.bind("matsci", MATSCI)
    graph.bind("matsciunits", munits)
    graph.add((URIRef(f"{BASE}matsci"), RDF.type, OWL.Ontology))
    material = MATSCI["Material"]
    graph.add((material, RDF.type, OWL.Class))
    graph.add((material, RDFS.label, Literal("Material")))
    # matsciunits namespace declared but no triple from it is retrieved.
    graph.add((munits["millielectronvolt"], RDFS.label, Literal("meV")))

    ontologies = [_ontology(f"{BASE}matsci", graph)]
    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(material)],
        entity_relevance={str(material): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=0,
        max_total_triples=3,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(material): ROLE_RESOURCE},
        hub_seed_count=1,
        ancestor_closure_depth=0,
    )
    bound = {prefix for prefix, _ in result.namespaces() if prefix}
    assert "matsci" in bound
    assert "matsciunits" not in bound


def test_ontology_round_robin_seed_order_interleaves_groups() -> None:
    ordered = _interleave_by_group(
        ["a1", "b1", "a2", "a3", "b2"],
        {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B"},
    )
    assert ordered == ["a1", "b1", "a2", "b2", "a3"]


def test_build_induced_subgraph_shared_ancestor_connects_two_seeds() -> None:
    graph = RDFGraph()
    graph.bind("matsci", MATSCI)
    graph.add((URIRef(f"{BASE}matsci"), RDF.type, OWL.Ontology))
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

    ontologies = [_ontology(f"{BASE}matsci", graph)]
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
    graph.bind("matsci", MATSCI)
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
    _assert_no_degenerate_restriction_bnodes(result)


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
    graph.bind("matsci", MATSCI)
    graph.add((URIRef(f"{BASE}matsci"), RDF.type, OWL.Ontology))

    method_class = MATSCI["OpticalCharacterizationMethod"]
    uses_method = MATSCI["usesMethod"]
    graph.add((method_class, RDF.type, OWL.Class))
    graph.add((method_class, RDFS.label, Literal("Optical characterization method")))
    graph.add((uses_method, RDF.type, OWL.ObjectProperty))
    graph.add((uses_method, RDFS.label, Literal("uses method")))
    graph.add((uses_method, RDFS.domain, method_class))
    graph.add((uses_method, RDFS.range, MATSCI["CharacterizationMethod"]))

    ontologies = [_ontology(f"{BASE}matsci", graph)]
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
    graph.bind("matsci", MATSCI)
    graph.add((URIRef(f"{BASE}matsci"), RDF.type, OWL.Ontology))

    method_class = MATSCI["OpticalCharacterizationMethod"]
    uses_method = MATSCI["usesMethod"]
    graph.add((method_class, RDF.type, OWL.Class))
    graph.add((method_class, RDFS.label, Literal("Optical characterization method")))
    graph.add((uses_method, RDF.type, OWL.ObjectProperty))
    graph.add((uses_method, RDFS.label, Literal("uses method")))
    graph.add((uses_method, RDFS.domain, method_class))

    ontologies = [_ontology(f"{BASE}matsci", graph)]
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
    _assert_no_degenerate_restriction_bnodes(result)


def test_referenced_domain_class_gets_symbol_predicates_and_types() -> None:
    """A property seed's domain class must carry its notation/symbol triples.

    Observed shape: a property was admitted but its domain class entered the
    snapshot without notation or types, so the renderer fell back to the
    parent class.
    """
    from rdflib.namespace import SKOS

    graph = RDFGraph()
    graph.bind("matsci", MATSCI)
    graph.add((URIRef(f"{BASE}matsci"), RDF.type, OWL.Ontology))

    domain_class = MATSCI["NanocrystalSuperlatticeSample"]
    parent = MATSCI["SuperlatticeSample"]
    prop = MATSCI["hasPhotonPropagationEffect"]
    graph.add((domain_class, RDF.type, OWL.Class))
    graph.add((domain_class, RDFS.label, Literal("Nanocrystal superlattice sample")))
    graph.add((domain_class, SKOS.notation, Literal("NC SL")))
    graph.add((domain_class, RDFS.subClassOf, parent))
    graph.add((parent, RDF.type, OWL.Class))
    graph.add((parent, RDFS.label, Literal("Superlattice sample")))
    graph.add((prop, RDF.type, OWL.ObjectProperty))
    graph.add((prop, RDFS.label, Literal("has photon propagation effect")))
    graph.add((prop, RDFS.domain, domain_class))
    graph.add((prop, RDFS.range, MATSCI["PhotonPropagationEffect"]))

    ontologies = [_ontology(f"{BASE}matsci", graph)]
    result, _ = SPARQLTool._build_induced_subgraph(
        ontologies=ontologies,
        entity_uris=[str(prop)],
        entity_relevance={str(prop): 1.0},
        ontology_iris=[ontologies[0].iri],
        depth=0,
        max_total_triples=300,
        estimated_triples_per_query=24,
        ontology_version_filters=None,
        ontology_hash_filters=None,
        entity_roles={str(prop): ROLE_PREDICATE},
        hub_seed_count=1,
        ancestor_closure_depth=1,
        extra_description_predicates=(SKOS.notation,),
    )

    assert (prop, RDFS.domain, domain_class) in result
    # The referenced class is materialized with its symbol predicate...
    assert (domain_class, SKOS.notation, Literal("NC SL")) in result
    # ...and its label.
    assert (
        domain_class,
        RDFS.label,
        Literal("Nanocrystal superlattice sample"),
    ) in result
