"""The version header: written once at the stamp, read back strictly."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.ontology import REMOVED_VERSION_TYPE, Ontology, content_hash
from ontocast.onto.ontology_header import (
    CatalogContractError,
    adopt_ontology,
    diff_versions,
    read_header,
    stamp_version,
    tombstone_graph,
    without_version_header,
)
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from test.ontology_helpers import next_version, stamped_ontology

pytestmark = pytest.mark.unit

T0 = datetime(2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc)
T1 = datetime(2024, 5, 7, 7, 8, 9, tzinfo=timezone.utc)

IRI = "https://example.org/tmi"

ROOT_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix tmi: <https://example.org/tmi#> .

<https://example.org/tmi> a owl:Ontology ;
    dcterms:title "The TMI Ontology" ;
    dcterms:description "Terms of the TMI domain." ;
    owl:versionInfo "2.1.0" .

tmi:Thing a owl:Class ; rdfs:label "Thing" .
tmi:Part a owl:Class ; rdfs:subClassOf tmi:Thing .
"""


def _root() -> Ontology:
    return stamped_ontology(ROOT_TTL, now=T0)


def test_read_header_round_trips_every_field_of_a_stamped_root() -> None:
    root = _root()
    assert root.version_iri is not None

    header = read_header(root.graph, root.version_iri)

    assert header.graph_uri == root.version_iri
    assert header.iri == IRI
    assert header.ontology_id == "tmi"
    assert header.title == "The TMI Ontology"
    assert header.description == "Terms of the TMI domain."
    assert header.version == "2.1.0"
    assert header.version_iri == root.version_iri
    assert header.version_iri == f"{IRI}#{root.hash}"
    assert header.prior_version_iris == []
    assert header.created_at == T0
    assert header.modified_at == T0
    assert header.removed is False
    assert [b.version_iri for b in root.bases] == [root.version_iri]


def test_header_predicates_are_written_on_the_ontology_subject() -> None:
    root = _root()
    subject = URIRef(IRI)
    objects = {p: set(root.graph.objects(subject, p)) for p in (OWL.versionIRI,)}
    assert objects[OWL.versionIRI] == {URIRef(root.version_iri or "")}
    assert set(root.graph.objects(subject, RDFS.label)) == {Literal("The TMI Ontology")}
    assert (subject, RDF.type, OWL.Ontology) in root.graph


@pytest.mark.anyio
async def test_header_survives_the_store_and_the_catalog_read() -> None:
    root = _root()
    assert root.version_iri is not None
    manager = InMemoryTripleStoreManager()
    await manager.async_init()
    manager.serialize(root)

    catalog = await manager.afetch_ontology_catalog()
    assert len(catalog) == 1
    header = catalog[0]
    expected = read_header(root.graph, root.version_iri)
    assert header.model_dump(exclude={"hash"}) == expected.model_dump(exclude={"hash"})

    fetched = (await manager.afetch_ontologies_by_iri([IRI]))[0]
    for field in (
        "iri",
        "ontology_id",
        "title",
        "description",
        "version",
        "version_iri",
        "prior_version_iris",
        "created_at",
        "modified_at",
        "removed",
        "hash",
    ):
        assert getattr(fetched, field) == getattr(root, field), field
    assert [b.version_iri for b in fetched.bases] == [root.version_iri]


def _stored_graph() -> tuple[str, RDFGraph]:
    root = _root()
    assert root.version_iri is not None
    return root.version_iri, root.graph.copy()


def _drop_version_iri(graph: RDFGraph) -> None:
    graph.remove((URIRef(IRI), OWL.versionIRI, None))


def _move_version_iri(graph: RDFGraph) -> None:
    graph.remove((URIRef(IRI), OWL.versionIRI, None))
    graph.add((URIRef(IRI), OWL.versionIRI, URIRef(f"{IRI}#elsewhere")))


def _second_ontology_subject(graph: RDFGraph) -> None:
    graph.add((URIRef("https://example.org/intruder"), RDF.type, OWL.Ontology))


@pytest.mark.parametrize(
    "corrupt",
    [_drop_version_iri, _move_version_iri, _second_ontology_subject],
    ids=["missing-versionIRI", "versionIRI-not-the-graph", "two-ontology-subjects"],
)
def test_read_header_raises_naming_the_graph_when_the_header_is_broken(
    corrupt,
) -> None:
    graph_uri, graph = _stored_graph()
    corrupt(graph)

    with pytest.raises(CatalogContractError) as info:
        read_header(graph, graph_uri)

    assert info.value.graph_uri == graph_uri
    assert graph_uri in str(info.value)


def _data_ontology_files() -> list[Path]:
    data = Path(__file__).resolve().parent.parent / "data"
    declared: list[Path] = []
    for path in sorted(data.rglob("*.ttl")):
        graph = RDFGraph()
        graph.parse(path, format="turtle")
        if any(True for _ in graph.triples((None, RDF.type, OWL.Ontology))):
            declared.append(path)
    return declared


DATA_ONTOLOGY_FILES = _data_ontology_files()


def test_data_ontology_files_are_discoverable() -> None:
    assert DATA_ONTOLOGY_FILES, "no ontology file under test/data declares one"


@pytest.mark.parametrize("path", DATA_ONTOLOGY_FILES, ids=lambda p: p.stem)
def test_adopt_ontology_stamps_each_data_file_as_a_root(path: Path) -> None:
    graph = RDFGraph()
    graph.parse(path, format="turtle")
    declared_iri = next(
        str(s) for s, _, _ in graph.triples((None, RDF.type, OWL.Ontology))
    )

    adopted = adopt_ontology(graph, now=T0)

    assert adopted.iri == declared_iri
    assert adopted.version_iri == f"{declared_iri}#{adopted.hash}"
    assert adopted.prior_version_iris == []
    assert adopted.created_at == adopted.modified_at == T0
    assert adopted.ontology_id and adopted.title
    assert read_header(adopted.graph, adopted.version_iri).iri == declared_iri


def test_adopt_ontology_refuses_a_graph_without_an_ontology_subject() -> None:
    graph = RDFGraph()
    graph.add((URIRef(f"{IRI}#Thing"), RDF.type, OWL.Class))
    with pytest.raises(ValueError, match="exactly one owl:Ontology"):
        adopt_ontology(graph)


def test_an_author_extra_survives_stamping_and_is_content() -> None:
    plain = _root()
    graph = RDFGraph._from_turtle_str(ROOT_TTL)
    imported = URIRef("https://example.org/other")
    graph.add((URIRef(IRI), OWL.imports, imported))

    with_import = stamped_ontology(graph, now=T0)

    assert (URIRef(IRI), OWL.imports, imported) in with_import.graph
    assert with_import.hash != plain.hash
    assert with_import.version_iri != plain.version_iri
    assert content_hash(with_import.graph, IRI) == with_import.hash


def test_stamp_version_of_an_unchanged_artifact_is_none() -> None:
    root = _root()
    unchanged = root.derive_updated_version(root.graph.copy())

    assert stamp_version(unchanged, now=T1) is None


def test_restamping_identical_content_yields_the_same_version_iri() -> None:
    first = stamped_ontology(ROOT_TTL, now=T0)
    second = stamped_ontology(ROOT_TTL, now=T1)

    assert first.hash == second.hash
    assert first.version_iri == second.version_iri
    assert first.modified_at != second.modified_at, "dates are not content"


def test_stamp_version_refuses_an_empty_artifact() -> None:
    with pytest.raises(ValueError, match="empty or null"):
        stamp_version(Ontology(ontology_id="tmi", iri=IRI, title="TMI"))


def test_tombstone_graph_keeps_the_header_and_drops_content() -> None:
    root = _root()
    assert root.version_iri is not None
    content_before = set(without_version_header(root.graph))
    assert content_before

    tombstone = tombstone_graph(root.graph, root.version_iri)

    header = read_header(tombstone, root.version_iri)
    assert header.removed is True
    assert header.version_iri == root.version_iri
    assert header.iri == root.iri
    assert (URIRef(root.version_iri), RDF.type, REMOVED_VERSION_TYPE) in tombstone
    content = {
        triple
        for triple in without_version_header(tombstone)
        if triple != (URIRef(root.version_iri), RDF.type, REMOVED_VERSION_TYPE)
    }
    assert content == set()
    assert (URIRef(f"{IRI}#Thing"), RDF.type, OWL.Class) not in tombstone
    assert (URIRef(f"{IRI}#Part"), RDF.type, OWL.Class) not in tombstone


@pytest.mark.anyio
async def test_aremove_version_writes_a_tombstone() -> None:
    root = _root()
    assert root.version_iri is not None
    manager = InMemoryTripleStoreManager()
    await manager.async_init()
    await manager.aserialize(root)

    await manager.aremove_version(root.version_iri)

    stored = await manager.afetch_named_graph(root.version_iri)
    assert stored is not None
    header = read_header(stored, root.version_iri)
    assert header.removed is True
    content = {
        triple
        for triple in without_version_header(stored)
        if triple != (URIRef(root.version_iri), RDF.type, REMOVED_VERSION_TYPE)
    }
    assert content == set()


def test_diff_versions_reports_added_and_removed_ntriples() -> None:
    root = _root()
    tip_graph = root.graph.copy()
    tip_graph.add((URIRef(f"{IRI}#Extra"), RDF.type, OWL.Class))
    tip_graph.remove((URIRef(f"{IRI}#Part"), RDF.type, OWL.Class))
    tip = next_version(root, tip_graph)
    assert root.version_iri and tip.version_iri

    diff = diff_versions(
        root.graph, tip.graph, from_iri=root.version_iri, to_iri=tip.version_iri
    )

    assert diff.from_iri == root.version_iri
    assert diff.to_iri == tip.version_iri
    assert diff.added_count == 1
    assert diff.removed_count == 1
    assert any("Extra" in line for line in diff.added)
    assert any("Part" in line for line in diff.removed)
