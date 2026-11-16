"""The resolution graph and the union view against a live Fuseki.

Runs only with ``-m integration`` and ``ONTOCAST_TEST_FUSEKI_URI`` set (with
``ONTOCAST_TEST_FUSEKI_AUTH`` as ``user:password`` when the server needs it).
It creates throwaway datasets and deletes them afterwards.
"""

import asyncio
import os
import uuid
from collections.abc import Iterator

import httpx
import pytest

from ontocast.link import IdentityDeclarations, build, cluster_of
from ontocast.tool.triple_manager.fuseki import FusekiTripleStoreManager
from test.link.helpers import (
    DOC1,
    DOC2,
    DOC3,
    DOMAIN,
    VALUE_CLASSES,
    catalog_graph,
    document,
    e,
)

pytestmark = pytest.mark.integration

URI = os.environ.get("ONTOCAST_TEST_FUSEKI_URI")
AUTH = os.environ.get("ONTOCAST_TEST_FUSEKI_AUTH")


@pytest.fixture
def fuseki() -> Iterator[FusekiTripleStoreManager]:
    if not URI:
        pytest.skip("ONTOCAST_TEST_FUSEKI_URI is not set")
    suffix = uuid.uuid4().hex[:8]
    names = [
        f"pytest-link-{suffix}-{kind}" for kind in ("facts", "ontologies", "shapes")
    ]
    manager = FusekiTripleStoreManager(
        uri=URI,
        auth=AUTH,
        dataset=names[0],
        ontologies_dataset=names[1],
        shapes_dataset=names[2],
    )
    asyncio.run(manager.async_init())
    yield manager
    auth = manager._prepare_auth()
    root = URI.rstrip("/")

    async def drop() -> None:
        async with httpx.AsyncClient(auth=auth, timeout=30.0) as client:
            for name in names:
                await client.delete(f"{root}/$/datasets/{name}")

    asyncio.run(drop())


def test_union_view_and_clusters_on_fuseki(fuseki: FusekiTripleStoreManager) -> None:
    declarations = IdentityDeclarations.of([catalog_graph()])

    async def run() -> None:
        for doc in (DOC1, DOC2, DOC3):
            await fuseki.aserialize(document(doc), graph_uri=doc)
        report = await build(
            fuseki, declarations, domain=DOMAIN, value_classes=VALUE_CLASSES
        )
        assert report.documents == 3
        query = (
            "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> "
            "SELECT DISTINCT ?b WHERE { ?a skos:closeMatch ?m . ?b skos:closeMatch ?m . "
            f"FILTER(?a = <{e(DOC1, 'paper')}> && ?a != ?b) }}"
        )
        rows = await fuseki.aselect(query, store="facts", union=True)
        assert rows == [{"b": str(e(DOC2, "study"))}]
        cluster = await cluster_of(fuseki, declarations, str(e(DOC1, "echr")))
        assert cluster is not None
        assert {m.iri for m in cluster.members} == {
            str(e(DOC1, "echr")),
            str(e(DOC2, "court")),
        }
        assert [k.iri for k in cluster.kept_apart] == [str(e(DOC3, "echr"))]

    asyncio.run(run())
