"""Tests for tenant/project dataset and collection naming."""

import pytest

from ontocast.config import FusekiConfig, QdrantConfig
from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
    tenant_project_store_name,
)


def test_default_names_use_double_dash() -> None:
    assert tenant_project_facts_name(DEFAULT_TENANT, DEFAULT_PROJECT) == (
        "ontocast--test--facts"
    )
    assert tenant_project_ontologies_name(DEFAULT_TENANT, DEFAULT_PROJECT) == (
        "ontocast--test--ontologies"
    )


def test_custom_sep() -> None:
    assert tenant_project_facts_name("a", "b", sep="__") == "a__b__facts"


def test_store_kind_literal() -> None:
    assert tenant_project_store_name("x", "y", "facts") == "x--y--facts"


def test_empty_tenant_rejected() -> None:
    with pytest.raises(ValueError):
        tenant_project_facts_name("", "p")


def test_fuseki_config_default_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUSEKI_DATASET", raising=False)
    monkeypatch.delenv("FUSEKI_ONTOLOGIES_DATASET", raising=False)
    c = FusekiConfig(uri="http://localhost:3030", auth="u/p")
    assert c.dataset == tenant_project_facts_name(DEFAULT_TENANT, DEFAULT_PROJECT)
    assert c.ontologies_dataset == tenant_project_ontologies_name(
        DEFAULT_TENANT, DEFAULT_PROJECT
    )


def test_fuseki_explicit_dataset_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUSEKI_DATASET", raising=False)
    monkeypatch.delenv("FUSEKI_ONTOLOGIES_DATASET", raising=False)
    c = FusekiConfig(
        uri="http://localhost:3030",
        auth="u/p",
        dataset="legacy_facts",
        ontologies_dataset="legacy_onto",
    )
    assert c.dataset == "legacy_facts"
    assert c.ontologies_dataset == "legacy_onto"


def test_qdrant_config_default_collections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_ONTOLOGY_COLLECTION", raising=False)
    monkeypatch.delenv("QDRANT_FACTS_COLLECTION", raising=False)
    c = QdrantConfig(uri="http://localhost:6333")
    assert c.ontology_collection == tenant_project_ontologies_name(
        DEFAULT_TENANT, DEFAULT_PROJECT
    )
    assert c.facts_collection == tenant_project_facts_name(
        DEFAULT_TENANT, DEFAULT_PROJECT
    )


def test_qdrant_explicit_ontology_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_ONTOLOGY_COLLECTION", raising=False)
    monkeypatch.delenv("QDRANT_FACTS_COLLECTION", raising=False)
    c = QdrantConfig(
        uri="http://localhost:6333",
        ontology_collection="my_atoms",
    )
    assert c.ontology_collection == "my_atoms"
    assert c.facts_collection == tenant_project_facts_name(
        DEFAULT_TENANT, DEFAULT_PROJECT
    )
