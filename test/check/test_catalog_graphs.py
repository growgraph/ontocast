"""``ToolBox.catalog_graphs``: the merged catalog and shapes, no pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rdflib import OWL, RDF, SH, URIRef

from ontocast.config import Config, PathConfig, ToolConfig
from ontocast.onto.ontology import Ontology
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

_ONTO = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://example.org/{name}> a owl:Ontology ; rdfs:label "{name}" .
<https://example.org/{name}#Thing> a owl:Class .
"""

_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
<https://example.org/shapes#S> a sh:NodeShape ;
    sh:targetClass <https://example.org/alpha#Thing> .
"""


def _toolbox(tmp_path: Path) -> ToolBox:
    tool_config = ToolConfig(path_config=PathConfig())
    toolbox = ToolBox(Config(tool_config=tool_config))
    store = InMemoryTripleStoreManager()
    toolbox.triple_store_manager = store
    toolbox.ontology_manager.register_triple_store(store)
    toolbox.shapes_catalog.register_triple_store(store)
    return toolbox


def test_catalog_graphs_in_memory(tmp_path: Path) -> None:
    toolbox = _toolbox(tmp_path)
    for name in ("alpha", "beta"):
        path = tmp_path / f"{name}.ttl"
        path.write_text(_ONTO.format(name=name), encoding="utf-8")
        toolbox.ontology_manager.add_ontology(
            Ontology.from_file(path), skip_vector_index=True
        )
    shapes_dir = tmp_path / "shapes"
    shapes_dir.mkdir()
    (shapes_dir / "s.ttl").write_text(_SHAPES, encoding="utf-8")
    asyncio.run(toolbox.shapes_catalog.sync(str(shapes_dir)))

    catalog, shapes = asyncio.run(toolbox.catalog_graphs())

    for name in ("alpha", "beta"):
        assert (URIRef(f"https://example.org/{name}"), RDF.type, OWL.Ontology) in (
            catalog
        )
    assert shapes is not None
    assert (URIRef("https://example.org/shapes#S"), RDF.type, SH.NodeShape) in shapes


def test_catalog_graphs_empty_catalog(tmp_path: Path) -> None:
    toolbox = _toolbox(tmp_path)
    catalog, shapes = asyncio.run(toolbox.catalog_graphs())
    assert len(catalog) == 0
    assert shapes is None


def test_catalog_graphs_returns_a_private_copy(tmp_path: Path) -> None:
    toolbox = _toolbox(tmp_path)
    path = tmp_path / "alpha.ttl"
    path.write_text(_ONTO.format(name="alpha"), encoding="utf-8")
    toolbox.ontology_manager.add_ontology(
        Ontology.from_file(path), skip_vector_index=True
    )
    first, _ = asyncio.run(toolbox.catalog_graphs())
    size = len(first)
    first.add((URIRef("https://x/s"), RDF.type, OWL.Class))
    second, _ = asyncio.run(toolbox.catalog_graphs())
    assert len(second) == size


_UNDESCRIBED = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<https://example.org/gamma> a owl:Ontology .
<https://example.org/gamma#Thing> a owl:Class .
"""


@pytest.mark.parametrize("summarize", [False, True])
def test_initialize_without_summaries_makes_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summarize: bool
) -> None:
    async def no_model(*_args, **_kwargs):
        raise AssertionError("a model call was made")

    monkeypatch.setattr("ontocast.toolbox.render_ontology_summary", no_model)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "gamma.ttl").write_text(_UNDESCRIBED, encoding="utf-8")
    toolbox = _toolbox(tmp_path)
    toolbox.config.tool_config.path_config.ontology_directory = seeds

    if summarize:
        # The default is unchanged: an undescribed ontology is summarized.
        with pytest.raises(AssertionError, match="model call"):
            asyncio.run(toolbox.initialize())
        return
    asyncio.run(toolbox.initialize(summarize_ontologies=False))
    catalog, _ = asyncio.run(toolbox.catalog_graphs())
    assert (URIRef("https://example.org/gamma#Thing"), RDF.type, OWL.Class) in catalog
