"""Tests for the shapes partition of the triple store.

The load-bearing property is the separation: a shapes document declares its own
``owl:Ontology`` header, so if it were stored beside the ontologies it would be
discovered as a catalog entry and offered to the renderer as schema.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import pytest

from ontocast.config import Config, FactsValidationConfig, ToolConfig
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.shapes_catalog import ShapesCatalog
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.toolbox import ToolBox

SHAPES_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix q:   <https://example.org/q#> .

<https://example.org/q-shapes> a owl:Ontology ;
    owl:versionInfo "1.0.0" .

q:ValueShape a sh:NodeShape ;
    sh:targetClass q:QuantityValue ;
    sh:property [ sh:path q:numericValue ; sh:datatype xsd:decimal ] .
"""

HEADERLESS_SHAPES_TTL = """
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix q:   <https://example.org/q#> .

q:BareShape a sh:NodeShape ;
    sh:targetClass q:Thing .
"""


def _catalog() -> tuple[ShapesCatalog, InMemoryTripleStoreManager]:
    store = InMemoryTripleStoreManager()
    catalog = ShapesCatalog()
    catalog.register_triple_store(store)
    return catalog, store


def _seed_dir(tmp: Path, *files: tuple[str, str]) -> Path:
    directory = tmp / "shapes"
    directory.mkdir(parents=True, exist_ok=True)
    for name, ttl in files:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ttl, encoding="utf-8")
    return directory


# --- seeding -----------------------------------------------------------------


def test_seed_directory_lands_in_the_partition() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp), ("q-shapes.ttl", SHAPES_TTL))
        catalog, _ = _catalog()
        asyncio.run(catalog.sync(str(directory)))

        graph = catalog.graph()
        assert graph is not None and len(graph)
        assert asyncio.run(catalog.list_graph_uris()) == [
            "https://example.org/q-shapes"
        ]


def test_seed_search_is_recursive() -> None:
    """``FACTS_SHAPES_DIR`` accepted nested layouts before shapes were stored."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp), ("nested/deeper/q-shapes.ttl", SHAPES_TTL))
        catalog, _ = _catalog()
        asyncio.run(catalog.sync(str(directory)))
        assert asyncio.run(catalog.list_graph_uris()) == [
            "https://example.org/q-shapes"
        ]


def test_headerless_shapes_file_is_named_by_its_path() -> None:
    """A bare SHACL file has no IRI; its name must still be stable across edits."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp), ("bare.ttl", HEADERLESS_SHAPES_TTL))
        catalog, _ = _catalog()
        asyncio.run(catalog.sync(str(directory)))
        assert asyncio.run(catalog.list_graph_uris()) == ["urn:shapes:bare.ttl"]

        # Editing the file replaces the document rather than accumulating a
        # second copy beside it.
        (directory / "bare.ttl").write_text(
            HEADERLESS_SHAPES_TTL.replace("q:Thing", "q:OtherThing"), encoding="utf-8"
        )
        asyncio.run(catalog.sync(str(directory)))
        assert asyncio.run(catalog.list_graph_uris()) == ["urn:shapes:bare.ttl"]


def test_reseeding_replaces_rather_than_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp), ("q-shapes.ttl", SHAPES_TTL))
        catalog, _ = _catalog()
        asyncio.run(catalog.sync(str(directory)))
        first = len(catalog.graph() or RDFGraph())
        asyncio.run(catalog.sync(str(directory)))
        assert len(catalog.graph() or RDFGraph()) == first


# --- degraded configuration must never read as "clean" -----------------------


def test_missing_shapes_dir_warns(caplog) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        catalog, _ = _catalog()
        with caplog.at_level(logging.WARNING):
            asyncio.run(catalog.sync(str(Path(tmp) / "nope")))
        assert catalog.graph() is None
        assert "not a directory" in caplog.text


def test_empty_shapes_dir_warns(caplog) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp))
        catalog, _ = _catalog()
        with caplog.at_level(logging.WARNING):
            asyncio.run(catalog.sync(str(directory)))
        assert catalog.graph() is None
        assert "no .ttl shape files" in caplog.text


def test_unparseable_shapes_file_warns_and_is_skipped(caplog) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(
            Path(tmp), ("broken.ttl", "@prefix bad <"), ("q-shapes.ttl", SHAPES_TTL)
        )
        catalog, _ = _catalog()
        with caplog.at_level(logging.WARNING):
            asyncio.run(catalog.sync(str(directory)))
        assert "Failed to parse shapes file" in caplog.text
        assert asyncio.run(catalog.list_graph_uris()) == [
            "https://example.org/q-shapes"
        ]


def test_no_shapes_anywhere_is_none_not_an_empty_graph() -> None:
    """``None`` keeps ``shacl_evaluated`` at "never checked"."""
    catalog, _ = _catalog()
    asyncio.run(catalog.sync(None))
    assert catalog.graph() is None


# --- the separation invariant ------------------------------------------------


def test_stored_shapes_never_enter_the_ontology_catalog() -> None:
    """A shapes document declares ``owl:Ontology``; it must still not be a catalog entry.

    Catalog discovery claims every named graph carrying an ``owl:Ontology``
    subject. Kept in the ontologies dataset, the shapes above would register as
    an ontology, be indexed as ontology atoms, and be offered to the renderer as
    schema.
    """
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp), ("q-shapes.ttl", SHAPES_TTL))
        toolbox = ToolBox(
            Config(
                tool_config=ToolConfig(
                    facts_validation=FactsValidationConfig.model_construct(
                        shapes_dir=str(directory)
                    )
                )
            )
        )

        async def main() -> None:
            await toolbox.shapes_catalog.sync(str(directory))
            assert toolbox.shapes_catalog.graph() is not None

            store = toolbox.triple_store_manager
            assert store is not None
            assert await store.afetch_ontology_catalog() == []
            assert await store.afetch_ontologies() == []

        asyncio.run(main())


def test_shapes_are_invisible_from_another_tenant() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = _seed_dir(Path(tmp), ("q-shapes.ttl", SHAPES_TTL))
        toolbox = ToolBox(Config(tool_config=ToolConfig()))

        async def main() -> None:
            await toolbox.update_tenancy_with_vector_mode(
                "acme",
                "x",
                initialize_vector_store=False,
                fail_on_vector_store_error=False,
            )
            await toolbox.shapes_catalog.sync(str(directory))
            assert toolbox.shapes_catalog.graph() is not None

            await toolbox.update_tenancy_with_vector_mode(
                "acme",
                "y",
                initialize_vector_store=False,
                fail_on_vector_store_error=False,
            )
            assert toolbox.shapes_catalog.graph() is None

        asyncio.run(main())


# --- mutation ----------------------------------------------------------------


def test_upload_replaces_a_document_at_the_same_iri() -> None:
    toolbox = ToolBox(Config(tool_config=ToolConfig()))

    async def main() -> None:
        first = await toolbox.ingest_shapes_ttl(
            SHAPES_TTL.encode(), filename="q-shapes.ttl"
        )
        assert first == "https://example.org/q-shapes"
        before = len(toolbox.shapes_catalog.graph() or RDFGraph())

        again = await toolbox.ingest_shapes_ttl(
            SHAPES_TTL.encode(), filename="q-shapes.ttl"
        )
        assert again == first
        assert len(toolbox.shapes_catalog.graph() or RDFGraph()) == before

        await toolbox.delete_shapes_by_uri(first)
        assert toolbox.shapes_catalog.graph() is None

    asyncio.run(main())


def test_uploading_an_empty_document_is_rejected() -> None:
    toolbox = ToolBox(Config(tool_config=ToolConfig()))

    async def main() -> None:
        with pytest.raises(ValueError, match="no triples"):
            await toolbox.ingest_shapes_ttl(b"", filename="empty.ttl")

    asyncio.run(main())


def test_uploading_invalid_turtle_is_rejected() -> None:
    toolbox = ToolBox(Config(tool_config=ToolConfig()))

    async def main() -> None:
        with pytest.raises(ValueError, match="Invalid Turtle"):
            await toolbox.ingest_shapes_ttl(b"@prefix bad <", filename="bad.ttl")

    asyncio.run(main())


# --- flush policy ------------------------------------------------------------


def test_flush_retains_shapes_by_default() -> None:
    """Dropping shapes on a flush disarms the gate without an error."""
    store = InMemoryTripleStoreManager()
    catalog = ShapesCatalog()
    catalog.register_triple_store(store)

    async def main() -> None:
        graph = RDFGraph()
        graph.parse(data=SHAPES_TTL, format="turtle")
        await catalog.ingest(graph, graph_uri="https://example.org/q-shapes")
        assert catalog.graph() is not None

        await store.clean()
        await catalog.sync()
        assert catalog.graph() is not None

        await store.clean(include_shapes=True)
        await catalog.sync()
        assert catalog.graph() is None

    asyncio.run(main())


def test_tenancy_flush_retains_shapes_by_default() -> None:
    store = InMemoryTripleStoreManager()
    catalog = ShapesCatalog()
    catalog.register_triple_store(store)

    async def main() -> None:
        await store.update_tenancy("acme", "x")
        graph = RDFGraph()
        graph.parse(data=SHAPES_TTL, format="turtle")
        await catalog.ingest(graph, graph_uri="https://example.org/q-shapes")

        await store.clean_tenancy("acme", "x")
        await catalog.sync()
        assert catalog.graph() is not None

        await store.clean_tenancy("acme", "x", include_shapes=True)
        await catalog.sync()
        assert catalog.graph() is None

    asyncio.run(main())
