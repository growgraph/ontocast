"""Tests for ontology identity, aliases, working-context, and prefix merge."""

import pytest
from rdflib import OWL, RDF, URIRef

from ontocast.onto.namespace_merge import choose_best_prefix, merge_namespace_bindings
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import derive_ontology_id, normalize_ontology_iri
from ontocast.tool.ontology_manager import OntologyManager
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit


def _with_author_prefix(ontology: Ontology, prefix: str) -> Ontology:
    """Rebind the ontology's own namespace to an author prefix after stamping.

    A stamp binds the id as the prefix; a caller may still hand the catalog a
    graph bound under another name, which registers as an alias.
    """
    ontology.graph.bind(prefix, ontology.namespace, override=True, replace=True)
    assert ontology.prefix == prefix
    return ontology


def test_derive_ontology_id_skos_conventional() -> None:
    assert derive_ontology_id("http://www.w3.org/2004/02/skos/core#") == "skos"
    assert derive_ontology_id("http://www.w3.org/2004/02/skos/core") == "skos"


def test_derive_ontology_id_foaf_conventional() -> None:
    """rdflib conventional map wins over the opaque ``0.1`` path tail."""
    assert derive_ontology_id("http://xmlns.com/foaf/0.1/") == "foaf"


def test_derive_ontology_id_rejects_pure_numeric_tail() -> None:
    assert derive_ontology_id("https://example.org/ns/0.1/") is None


def test_derive_ontology_id_brick_conventional() -> None:
    assert derive_ontology_id("https://brickschema.org/schema/Brick#") == "brick"


def test_normalize_ontology_iri_strips_delimiters() -> None:
    assert (
        normalize_ontology_iri("https://example.org/ont#")
        == normalize_ontology_iri("https://example.org/ont/")
        == "https://example.org/ont"
    )
    assert (
        normalize_ontology_iri("<https://example.org/ont>") == "https://example.org/ont"
    )


def test_merge_namespace_bindings_renames_conflicts() -> None:
    existing = {"ex": "https://example.org/a#"}
    incoming = {"ex": "https://example.org/b#"}
    merged = merge_namespace_bindings(existing, incoming)
    assert merged["ex"] == "https://example.org/a#"
    renamed = [v for k, v in merged.items() if k != "ex"]
    assert "https://example.org/b#" in renamed


def test_choose_best_prefix_prefers_author() -> None:
    chosen = choose_best_prefix(
        "https://growgraph.dev/ontologies/observation#",
        ["observation", "obs"],
        preferred_namespace_prefixes={
            "https://growgraph.dev/ontologies/observation#": "obs"
        },
    )
    assert chosen == "obs"


def test_rdfgraph_add_preserves_conflicting_prefixes() -> None:
    left = RDFGraph()
    left.bind("ex", URIRef("https://example.org/a#"))
    left.add(
        (
            URIRef("https://example.org/a#A"),
            RDF.type,
            OWL.Class,
        )
    )
    right = RDFGraph()
    right.bind("ex", URIRef("https://example.org/b#"))
    right.add(
        (
            URIRef("https://example.org/b#B"),
            RDF.type,
            OWL.Class,
        )
    )
    merged = left + right
    bindings = {p: str(u) for p, u in merged.namespaces() if p and p.startswith("ex")}
    assert "https://example.org/a#" in bindings.values()
    assert "https://example.org/b#" in bindings.values()


def test_adopting_binds_the_id_as_the_prefix_of_the_own_namespace() -> None:
    """The author's ``obs:`` is replaced: the id is the preferred prefix."""
    ontology = stamped_ontology(
        """
        @prefix obs: <https://growgraph.dev/ontologies/observation#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://growgraph.dev/ontologies/observation> a owl:Ontology .
        obs:Reading a owl:Class .
        """,
        ontology_id="observation",
    )
    assert ontology.prefix == "observation"
    bound = {p for p, u in ontology.graph.namespaces() if str(u) == ontology.namespace}
    assert bound == {"observation"}


def test_ontology_manager_aliases_prefix_and_id() -> None:
    ontology = _with_author_prefix(
        stamped_ontology(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <https://growgraph.dev/ontologies/observation> a owl:Ontology .
            <https://growgraph.dev/ontologies/observation#Reading> a owl:Class .
            """,
            ontology_id="observation",
        ),
        "obs",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology)

    assert mgr.resolve_ontology_ref("observation") == ontology.iri
    assert mgr.resolve_ontology_ref("obs") == ontology.iri
    assert mgr.resolve_ontology_ref(ontology.iri) == ontology.iri
    assert "obs" in mgr
    head = mgr.get_head(ontology_id="obs")
    assert head is not None
    assert head.iri == ontology.iri


def test_ontology_manager_allows_prefix_differing_from_ontology_id() -> None:
    ontology = _with_author_prefix(
        stamped_ontology(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <https://growgraph.dev/ontologies/lifecycle> a owl:Ontology .
            <https://growgraph.dev/ontologies/lifecycle#Stage> a owl:Class .
            """,
            ontology_id="lifecycle",
        ),
        "life",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology)  # must not raise
    assert mgr.resolve_ontology_ref("life") == ontology.iri
    assert mgr.resolve_ontology_ref("lifecycle") == ontology.iri


def test_ontology_manager_rejects_same_iri_different_ontology_id() -> None:
    ontology_a = stamped_ontology(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.org/finance> a owl:Ontology .
        <https://example.org/finance#Account> a owl:Class .
        """,
        ontology_id="finance",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology_a)

    ontology_b = stamped_ontology(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.org/finance> a owl:Ontology .
        <https://example.org/finance#extra> a owl:Class .
        """,
        ontology_id="accounting",
    )
    with pytest.raises(ValueError, match="already bound to identity"):
        mgr.add_ontology(ontology_b)


def test_ontology_manager_rejects_alias_across_iris() -> None:
    ontology_a = stamped_ontology(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.org/finance> a owl:Ontology .
        <https://example.org/finance#Account> a owl:Class .
        """,
        ontology_id="finance",
    )
    mgr = OntologyManager()
    mgr.add_ontology(ontology_a)

    ontology_b = stamped_ontology(
        """
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        <https://example.com/finance> a owl:Ontology .
        <https://example.com/finance#Ledger> a owl:Class .
        """,
        ontology_id="finance",
    )
    with pytest.raises(ValueError, match="already bound to IRI"):
        mgr.add_ontology(ontology_b)


def test_ontology_manager_prefix_collision_degrades_to_iri_addressing() -> None:
    """Two ontologies bound under the same author prefix coexist in one catalog.

    The colliding prefix alias is skipped (first binding wins); the second
    ontology stays addressable by IRI and ontology_id.
    """
    ontology_a = _with_author_prefix(
        stamped_ontology(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <https://example.org/music> a owl:Ontology .
            <https://example.org/music#Song> a owl:Class .
            """,
            ontology_id="music",
        ),
        "onto",
    )
    ontology_b = _with_author_prefix(
        stamped_ontology(
            """
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            <https://example.org/film> a owl:Ontology .
            <https://example.org/film#Movie> a owl:Class .
            """,
            ontology_id="film",
        ),
        "onto",
    )

    mgr = OntologyManager()
    mgr.add_ontology(ontology_a)
    mgr.add_ontology(ontology_b)  # must not raise

    # First binding wins the prefix alias; both stay fully addressable.
    assert mgr.resolve_ontology_ref("onto") == ontology_a.iri
    assert mgr.resolve_ontology_ref("music") == ontology_a.iri
    assert mgr.resolve_ontology_ref("film") == ontology_b.iri
    assert mgr.resolve_ontology_ref(ontology_b.iri) == ontology_b.iri


def test_no_owl_ontology_does_not_promote_prefix() -> None:
    graph = RDFGraph()
    graph.bind("ex", URIRef("https://example.org/mystery#"))
    graph.add(
        (
            URIRef("https://example.org/mystery#Thing"),
            RDF.type,
            OWL.Class,
        )
    )
    ontology = Ontology(graph=graph)
    assert ontology.ontology_id is None
    assert ontology.is_null()
    assert list(ontology.graph.triples((None, RDF.type, OWL.Ontology))) == []
