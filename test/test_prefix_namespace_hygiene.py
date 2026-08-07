"""Regression tests for domain-prefix / namespace hygiene in prompts and graphs."""

import pytest
from rdflib import OWL, RDF, URIRef

from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.onto.util import RDFLIB_DEFAULT_NAMESPACE_URIS
from ontocast.prompt.facts_guidelines import format_facts_operational_guidelines
from ontocast.prompt.ontology_context import extract_domain_prefix_pairs


def _matsci_turtle(*, prefix: str = "matsci") -> str:
    return f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix {prefix}: <https://growgraph.dev/ontologies/matsci#> .

<https://growgraph.dev/ontologies/matsci> a owl:Ontology ;
    rdfs:label "Material Science Ontology"@en .

{prefix}:Material a owl:Class ;
    rdfs:label "Material"@en .
"""


def test_extract_domain_prefix_pairs_excludes_rdflib_defaults() -> None:
    graph = RDFGraph()
    graph.parse(data=_matsci_turtle(), format="turtle")
    # Force-bind a few rdflib defaults that previously leaked into the prompt.
    graph.bind("brick", URIRef("https://brickschema.org/schema/Brick#"))
    graph.bind("csvw", URIRef("http://www.w3.org/ns/csvw#"))
    graph.bind("geo", URIRef("http://www.opengis.net/ont/geosparql#"))
    ontology = Ontology(graph=graph)

    pairs = extract_domain_prefix_pairs(ontology)
    prefixes = {p for p, _ in pairs}
    namespaces = {ns for _, ns in pairs}

    assert "matsci" in prefixes
    assert "brick" not in prefixes
    assert "csvw" not in prefixes
    assert "geo" not in prefixes
    assert "xml" not in prefixes
    assert namespaces.isdisjoint(RDFLIB_DEFAULT_NAMESPACE_URIS)


def test_author_short_prefix_kept_not_rebound_to_ontology_id() -> None:
    graph = RDFGraph()
    graph.parse(data=_matsci_turtle(prefix="matsci"), format="turtle")
    ontology = Ontology(graph=graph)

    assert ontology.ontology_id == "matsci"
    domain_bindings = [
        (prefix, str(uri))
        for prefix, uri in ontology.graph.namespaces()
        if str(uri) == "https://growgraph.dev/ontologies/matsci#"
    ]
    assert domain_bindings == [("matsci", "https://growgraph.dev/ontologies/matsci#")]
    assert ontology.prefix == "matsci"


def test_degenerate_ns_prefix_is_rebound_and_old_binding_removed() -> None:
    graph = RDFGraph()
    ns = "https://growgraph.dev/ontologies/matsci#"
    graph.bind("ns11", URIRef(ns))
    graph.add(
        (
            URIRef("https://growgraph.dev/ontologies/matsci"),
            RDF.type,
            OWL.Ontology,
        )
    )
    ontology = Ontology(graph=graph)

    assert ontology.ontology_id == "matsci"
    domain_prefixes = [
        prefix for prefix, uri in ontology.graph.namespaces() if str(uri) == ns
    ]
    assert domain_prefixes == ["matsci"]
    assert "ns11" not in domain_prefixes


def test_sanitize_prefixes_namespaces_preserves_xml_binding() -> None:
    graph = RDFGraph()
    graph.parse(data=_matsci_turtle(), format="turtle")
    xml_uri = "http://www.w3.org/XML/1998/namespace"
    before = {prefix: str(uri) for prefix, uri in graph.namespaces()}
    assert before.get("xml") == xml_uri

    graph.sanitize_prefixes_namespaces()

    after = {prefix: str(uri) for prefix, uri in graph.namespaces()}
    assert after.get("xml") == xml_uri
    assert "xml1" not in after
    assert not any(uri == f"{xml_uri}/" for uri in after.values())


def test_facts_guidelines_domain_clause_appears_once() -> None:
    clause = (
        "domain ontologies `matsci:` (<https://growgraph.dev/ontologies/matsci#>), "
        "`qqval:` (<https://growgraph.dev/ontologies/qqval#>)"
    )
    guidelines = format_facts_operational_guidelines(
        facts_namespace="https://example.com/facts/",
        domain_ontologies_clause=clause,
        jsonld=False,
    )
    assert guidelines.count(clause) == 1
    assert "domain ontology namespace(s) above" in guidelines
    # `ex:` is a MANDATORY repair finding, so the prompt must never advertise it:
    # every occurrence used to cost a repair render. This is the one behavioural
    # guard worth keeping from the old prompt-wording smoke tests; the rest
    # matched section numbers ("1d. SPECIFICITY RULE") and broke on renumbering.
    assert "ex:" not in guidelines


# --- Author-prefix persistence through the triple-store boundary (sh:declare) ---

_AUTHOR_PREFIX_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix myunits: <https://example.org/ontologies/my-units#> .

<https://example.org/ontologies/my-units> a owl:Ontology ;
    owl:versionInfo "1.0.0" .

myunits:someUnit a owl:NamedIndividual, qudt:Unit ;
    rdfs:label "some unit"@en .

myunits:otherUnit a owl:NamedIndividual, qudt:Unit ;
    rdfs:label "other unit"@en .
"""

_MY_UNITS_NS = "https://example.org/ontologies/my-units#"


def _author_ontology() -> Ontology:
    graph = RDFGraph()
    graph.parse(data=_AUTHOR_PREFIX_TTL, format="turtle")
    return Ontology(graph=graph)


def test_materialize_prefix_declarations_targets_used_author_bindings() -> None:
    ontology = _author_ontology()
    ontology.graph.bind("unused", URIRef("https://example.org/never-used#"))

    added = ontology.graph.materialize_prefix_declarations(URIRef(ontology.iri))

    declared = ontology.graph.declared_prefix_map()
    assert added == 1
    assert declared == {_MY_UNITS_NS: "myunits"}
    # Well-known namespaces (qudt) are recoverable from the canonical tables and
    # unused bindings advertise nothing — neither is persisted.
    assert "http://qudt.org/schema/qudt/" not in declared
    assert "https://example.org/never-used#" not in declared


def test_materialize_prefix_declarations_idempotent_and_hash_neutral() -> None:
    ontology = _author_ontology()
    hash_before = ontology.hash

    first = ontology.graph.materialize_prefix_declarations(URIRef(ontology.iri))
    second = ontology.graph.materialize_prefix_declarations(URIRef(ontology.iri))

    assert first == 1
    assert second == 0
    rehashed = Ontology(graph=ontology.graph, iri=ontology.iri)
    assert rehashed.hash == hash_before


@pytest.mark.anyio
async def test_author_prefix_survives_store_round_trip() -> None:
    from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager

    ontology = _author_ontology()
    hash_before = ontology.hash

    manager = InMemoryTripleStoreManager()
    await manager.aserialize(ontology)
    fetched = (await manager.afetch_ontologies())[0]

    bindings = [p for p, u in fetched.graph.namespaces() if str(u) == _MY_UNITS_NS]
    assert bindings == ["myunits"]
    assert fetched.prefix == "myunits"
    # Declarations are hash-neutral, so identity survives the round trip too
    # (content is float-free, so no literal-lexical-form drift interferes).
    assert fetched.hash == hash_before


def test_bind_used_prefixes_prefers_declared_over_heuristic() -> None:
    from ontocast.tool.sparql import _bind_used_prefixes

    namespace = "https://example.org/ontologies/my-units#"
    graph = RDFGraph()
    graph.parse(data=_AUTHOR_PREFIX_TTL, format="turtle")
    snapshot = RDFGraph()
    for triple in graph.triples((URIRef(f"{namespace}someUnit"), None, None)):
        snapshot.add(triple)

    # The plainest-name heuristic would pick "mu"; the author declared "my_units".
    prefix_map = {"mu": namespace, "my_units": namespace}
    _bind_used_prefixes(snapshot, prefix_map, {namespace: "my_units"})

    bound = {p: str(u) for p, u in snapshot.namespaces()}
    assert bound.get("my_units") == namespace
    assert "mu" not in bound


def test_jsonld_prompt_context_ignores_plain_literal_text() -> None:
    """A plain literal like "time: 10 minutes" must not retain the `time:` binding."""
    import json

    from rdflib import RDFS, Literal

    graph = RDFGraph()
    facts_ns = "https://example.org/facts#"
    graph.bind("cd", facts_ns)
    graph.bind("time", "http://www.w3.org/2006/time#")
    subject = URIRef(f"{facts_ns}s")
    graph.add((subject, RDFS.comment, Literal("time: 10 minutes elapsed")))

    payload = json.loads(graph.serialize_compact_jsonld_for_prompt())
    context = payload["@context"]

    assert "cd" in context
    assert "rdfs" in context
    assert "time" not in context


def test_jsonld_prompt_context_keeps_datatype_and_reference_prefixes() -> None:
    """Prefixes used in @id references and compact datatypes survive filtering."""
    import json

    from rdflib import Literal
    from rdflib.namespace import XSD

    graph = RDFGraph()
    facts_ns = "https://example.org/facts#"
    qudt_ns = "http://qudt.org/schema/qudt/"
    graph.bind("cd", facts_ns)
    graph.bind("qudt", qudt_ns)
    subject = URIRef(f"{facts_ns}s")
    graph.add(
        (
            subject,
            URIRef(f"{qudt_ns}numericValue"),
            Literal("230", datatype=XSD.decimal),
        )
    )
    graph.add((subject, RDF.type, URIRef(f"{facts_ns}Measurement")))

    payload = json.loads(graph.serialize_compact_jsonld_for_prompt())
    context = payload["@context"]

    assert "cd" in context
    assert "qudt" in context
    assert "xsd" in context  # via the compact datatype "@type": "xsd:decimal"


def test_normalize_namespace_iri_canonicalizes_schema_org_aliases() -> None:
    from ontocast.onto.iri_policy import normalize_namespace_iri

    assert normalize_namespace_iri("http://schema.org/") == "https://schema.org/"
    assert normalize_namespace_iri("<http://schema.org/>") == "https://schema.org/"
    assert normalize_namespace_iri("http://schema.org") == "https://schema.org/"
    assert normalize_namespace_iri("https://schema.org/") == "https://schema.org/"


def test_sanitize_remaps_http_schema_org_terms_to_https() -> None:
    graph = RDFGraph()
    graph.parse(
        data="""
@prefix schema: <http://schema.org/> .
@prefix schema1: <https://schema.org/> .

<https://example.com/a> a schema:Person ;
    schema:name "A" .
<https://example.com/b> a schema1:Person ;
    schema1:name "B" .
""",
        format="turtle",
    )
    graph.sanitize_prefixes_namespaces()

    subjects_of_https_person = set(
        graph.subjects(RDF.type, URIRef("https://schema.org/Person"))
    )
    assert subjects_of_https_person == {
        URIRef("https://example.com/a"),
        URIRef("https://example.com/b"),
    }
    assert not list(graph.subjects(RDF.type, URIRef("http://schema.org/Person")))

    schema_bindings = {
        prefix: str(uri)
        for prefix, uri in graph.namespaces()
        if str(uri) in ("http://schema.org/", "https://schema.org/")
    }
    assert schema_bindings == {"schema": "https://schema.org/"}


def test_postprocess_facts_units_merged_graph_has_single_schema_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ontocast.onto.content_unit import ContentUnit, OutputType
    from ontocast.tool.agg.aggregate import EmbeddingBasedAggregator

    def _unit(index: int, turtle: str) -> ContentUnit:
        graph = RDFGraph()
        graph.parse(data=turtle, format="turtle")
        return ContentUnit(
            text=f"unit {index}",
            index=index,
            doc_iri="https://growgraph.dev/doc/testdoc",
            type=OutputType.FACTS,
            graph=graph,
        )

    units = [
        _unit(
            0,
            """
@prefix schema: <http://schema.org/> .
<https://growgraph.dev/doc/testdoc/a> a schema:Person ; schema:name "A" .
""",
        ),
        _unit(
            1,
            """
@prefix schema: <https://schema.org/> .
<https://growgraph.dev/doc/testdoc/b> a schema:Person ; schema:name "B" .
""",
        ),
    ]
    aggregator = EmbeddingBasedAggregator()

    def cluster_by_scheme_insensitive_iri(representations):
        """Group `http://schema.org/X` with `https://schema.org/X`.

        That pairing is what a real embedding model produces here, and it is the
        premise of the test rather than its subject: what is under test is that
        the merged graph ends up with a single `schema:` binding. Stating the
        grouping directly keeps the assertion honest and off the model.
        """
        groups: dict[str, list[URIRef]] = {}
        for entity in representations:
            key = str(entity).replace("http://", "https://", 1)
            groups.setdefault(key, []).append(entity)
        return list(groups.values()), {}

    monkeypatch.setattr(
        aggregator.clusterer, "cluster_entities", cluster_by_scheme_insensitive_iri
    )
    result = aggregator.postprocess_facts_units(units, RDFGraph())

    serialized = result.graph.serialize(format="turtle")
    assert "schema1" not in serialized
    assert "http://schema.org/" not in serialized
    assert set(
        result.graph.subjects(RDF.type, URIRef("https://schema.org/Person"))
    ) == {
        URIRef("https://growgraph.dev/doc/testdoc/a"),
        URIRef("https://growgraph.dev/doc/testdoc/b"),
    }
