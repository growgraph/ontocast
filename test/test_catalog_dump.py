"""The catalog dump: the whole catalog after each document, as the store holds it.

A document's ``.ontology.ttl`` holds only the ontologies that document touched,
so following a catalog across a batch meant stitching dumps together. The
catalog dump reads every current version back from the triple store -- what the
next document is extracted against.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.api.process_helpers import dump_catalog_nquads
from ontocast.config import Config, PathConfig, ToolConfig
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.triple_manager.in_memory import InMemoryTripleStoreManager
from ontocast.toolbox import ToolBox

pytestmark = pytest.mark.unit

IRI = "https://example.org/seed"
THING = URIRef(f"{IRI}#Thing")
GADGET = URIRef(f"{IRI}#Gadget")


def _seed() -> Ontology:
    graph = RDFGraph()
    graph.add((URIRef(IRI), RDF.type, OWL.Ontology))
    graph.add((THING, RDF.type, OWL.Class))
    graph.add((THING, RDFS.label, Literal("Thing")))
    return Ontology(graph=graph, iri=IRI, ontology_id="seed", version="1.0.0")


def _toolbox(ontology_directory: Path | None = None) -> ToolBox:
    toolbox = ToolBox(
        Config(
            tool_config=ToolConfig(
                path_config=PathConfig(ontology_directory=ontology_directory)
            )
        )
    )
    toolbox.triple_store_manager = InMemoryTripleStoreManager()
    toolbox.ontology_manager.register_triple_store(toolbox.triple_store_manager)
    return toolbox


def _dump(toolbox: ToolBox, out: Path, line: int) -> list[str]:
    path = asyncio.run(
        dump_catalog_nquads(
            toolbox, Path("stream.txt"), line_number=line, output_dir=out
        )
    )
    assert path.name == f"stream.L{line}.catalog.nq"
    return path.read_text(encoding="utf-8").splitlines()


def test_each_dump_is_the_current_catalog(tmp_path) -> None:
    toolbox = _toolbox()
    seed = _seed()
    asyncio.run(toolbox.triple_store_manager.aserialize(seed))

    first = _dump(toolbox, tmp_path, 1)

    evolved_graph = seed.graph.copy()
    evolved_graph.add((GADGET, RDF.type, OWL.Class))
    asyncio.run(
        toolbox.triple_store_manager.aserialize(
            seed.derive_updated_version(evolved_graph)
        )
    )
    second = _dump(toolbox, tmp_path, 2)

    gadget = f"<{GADGET}> <{RDF.type}> <{OWL.Class}> <{IRI}> ."
    assert gadget not in first
    assert gadget in second
    thing_lines = {line for line in first if line.startswith(f"<{THING}>")}
    assert thing_lines and thing_lines <= set(second)
    hashes = [line for line in second if "dc/terms/identifier" in line]
    assert len(hashes) == 1, "one version per ontology, not every version"
    assert hashes[0] not in first
    assert second == sorted(second)
    assert all(line.endswith(f"<{IRI}> .") for line in second)


def test_a_fresh_in_memory_run_sees_only_its_seed() -> None:
    """Re-seeding is how a frozen run starts: nothing may carry over."""
    with tempfile.TemporaryDirectory() as tmp:
        seeds = Path(tmp) / "ontologies"
        seeds.mkdir()
        seed = _seed()
        (seeds / "seed.ttl").write_text(
            seed.graph.serialize(format="turtle"), encoding="utf-8"
        )

        earlier = _toolbox(seeds)
        evolved = seed.graph.copy()
        evolved.add((GADGET, RDF.type, OWL.Class))
        asyncio.run(
            earlier.triple_store_manager.aserialize(
                seed.derive_updated_version(evolved)
            )
        )

        fresh = _toolbox(seeds)
        synced = asyncio.run(fresh._synchronize_ontologies())

        assert [o.iri for o in synced] == [IRI]
        assert (GADGET, RDF.type, OWL.Class) not in synced[0].graph
