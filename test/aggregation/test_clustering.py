import importlib
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from rdflib import URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.tool.agg import clustering
from ontocast.tool.agg.clustering import (
    ClusterRepresentativeSelector,
    EntityClusterer,
    get_shared_sentence_transformer,
)
from ontocast.tool.agg.normalizer import EntityRepresentation


def test_shared_sentence_transformer_cache_reuses_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clustering._EMBEDDER_CACHE.clear()
    created: list[str] = []

    class _FakeModel:
        def __init__(self, model_name: str) -> None:
            created.append(model_name)

    def _fake_import_module(name: str) -> SimpleNamespace | ModuleType:
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=_FakeModel)
        return importlib.import_module(name)

    monkeypatch.setattr(clustering.importlib, "import_module", _fake_import_module)

    shared = get_shared_sentence_transformer("test-model")
    clusterer_embedder = EntityClusterer(embedding_model="test-model").embedder

    assert shared is clusterer_embedder
    assert created == ["test-model"]
    clustering._EMBEDDER_CACHE.clear()


def test_simplicity_score_prefers_simple_uris(
    cluster_representative_selector: ClusterRepresentativeSelector,
) -> None:
    simple = URIRef("http://ex.org/Thing")
    complex_uri = URIRef("http://example.org/deeply/nested/path/ComplexEntity_123")

    simple_score = cluster_representative_selector.compute_simplicity_score(simple)
    complex_score = cluster_representative_selector.compute_simplicity_score(
        complex_uri
    )

    assert simple_score < complex_score


def test_select_representative_prefers_ontology_entity(
    cluster_representative_selector: ClusterRepresentativeSelector,
) -> None:
    ont_entity = URIRef("http://ontology.org/Thing")
    chunk_entity = URIRef(f"{DEFAULT_IRI}/entity_long_name")

    ont_rep = Mock(is_ontology_entity=True)
    chunk_rep = Mock(is_ontology_entity=False)
    reps = cast(
        dict[URIRef, EntityRepresentation],
        {ont_entity: ont_rep, chunk_entity: chunk_rep},
    )

    selected = cluster_representative_selector.select_representative(
        [ont_entity, chunk_entity], reps
    )
    assert selected == ont_entity


def test_select_representative_prefers_simple_non_ontology_uri(
    cluster_representative_selector: ClusterRepresentativeSelector,
) -> None:
    simple = URIRef("http://chunk1.org/Thing")
    complex_uri = URIRef("http://chunk2.org/very_long_complex_entity_name_123")

    simple_rep = Mock(is_ontology_entity=False)
    complex_rep = Mock(is_ontology_entity=False)
    reps = cast(
        dict[URIRef, EntityRepresentation],
        {simple: simple_rep, complex_uri: complex_rep},
    )

    selected = cluster_representative_selector.select_representative(
        [simple, complex_uri], reps
    )
    assert selected == simple


def test_select_representative_returns_singleton(
    cluster_representative_selector: ClusterRepresentativeSelector,
) -> None:
    entity = URIRef("http://chunk1.org/Only")
    rep = Mock(is_ontology_entity=False)
    reps = cast(dict[URIRef, EntityRepresentation], {entity: rep})

    selected = cluster_representative_selector.select_representative([entity], reps)
    assert selected == entity


def test_create_mapping_maps_all_cluster_members(
    cluster_representative_selector: ClusterRepresentativeSelector,
) -> None:
    e1 = URIRef("http://chunk1.org/A")
    e2 = URIRef("http://chunk1.org/B")
    e3 = URIRef("http://chunk2.org/C")

    rep1 = Mock(is_ontology_entity=False)
    rep2 = Mock(is_ontology_entity=False)
    rep3 = Mock(is_ontology_entity=False)

    reps = cast(dict[URIRef, EntityRepresentation], {e1: rep1, e2: rep2, e3: rep3})
    mapping = cluster_representative_selector.create_mapping([[e1, e2], [e3]], reps)

    assert mapping[e1] == mapping[e2]
    assert mapping[e3] == e3


def test_select_representative_prefers_explicit_known_ontology_map(
    cluster_representative_selector: ClusterRepresentativeSelector,
) -> None:
    known_ontology = URIRef("http://example.org/onto#Conviction")
    tentative_ontology = URIRef("http://example.org/onto#Conviction1")
    reps = cast(
        dict[URIRef, EntityRepresentation],
        {
            known_ontology: Mock(is_ontology_entity=False),
            tentative_ontology: Mock(is_ontology_entity=True),
        },
    )

    selected = cluster_representative_selector.select_representative(
        [known_ontology, tentative_ontology],
        reps,
        entity_is_known_ontology={
            known_ontology: True,
            tentative_ontology: False,
        },
    )
    assert selected == known_ontology
