"""A derived artifact keeps its bases; stamping it rewrites the header.

Between a read from the store and the next stamp an artifact's graph may
carry the header of the version it came from. Nothing in that header is
content: it never moves the hash, and the stamp replaces it wholesale.
"""

from datetime import datetime, timezone
from typing import cast

import pytest
from rdflib import DCTERMS, OWL, RDF, XSD, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.ontology import Ontology, content_hash
from ontocast.onto.ontology_header import stamp_version
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import ToolBox
from test.ontology_helpers import stamped_ontology

pytestmark = pytest.mark.unit

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)

IRI = "https://example.org/onto"


def _make_base_ontology() -> Ontology:
    graph = RDFGraph()
    graph.add((URIRef(IRI), RDF.type, OWL.Ontology))
    graph.add((URIRef(f"{IRI}#Person"), RDF.type, OWL.Class))
    return stamped_ontology(graph, now=T0)


def test_derive_updated_version_keeps_bases_and_drops_the_stamp() -> None:
    base = _make_base_ontology()
    assert base.hash is not None and base.version_iri is not None
    onto_iri = URIRef(IRI)

    updated_graph = base.graph.copy()
    updated_graph.add((URIRef(f"{IRI}#Organization"), RDF.type, OWL.Class))
    hash_without_stale = content_hash(updated_graph, IRI)
    # Stale header triples, as a graph read back from the store carries them.
    updated_graph.add((onto_iri, OWL.priorVersion, URIRef(f"{IRI}#stale-parent")))
    updated_graph.add((onto_iri, OWL.versionIRI, URIRef(f"{IRI}#stale-hash")))
    updated_graph.add(
        (
            onto_iri,
            DCTERMS.created,
            Literal("2001-01-01T00:00:00+00:00", datatype=XSD.dateTime),
        )
    )

    updated = base.derive_updated_version(updated_graph)

    assert updated.version_iri is None
    assert updated.prior_version_iris == []
    assert updated.modified_at is None
    assert [b.version_iri for b in updated.bases] == [base.version_iri]
    assert updated.hash is not None and updated.hash != base.hash
    assert updated.hash == hash_without_stale, "the header is not content"


def test_stamping_a_derived_artifact_rewrites_the_header() -> None:
    base = _make_base_ontology()
    assert base.version_iri is not None
    onto_iri = URIRef(IRI)

    updated_graph = base.graph.copy()
    updated_graph.add((URIRef(f"{IRI}#Organization"), RDF.type, OWL.Class))
    updated_graph.add((onto_iri, OWL.priorVersion, URIRef(f"{IRI}#stale-parent")))
    updated_graph.add(
        (
            onto_iri,
            DCTERMS.created,
            Literal("2001-01-01T00:00:00+00:00", datatype=XSD.dateTime),
        )
    )

    stamped = stamp_version(base.derive_updated_version(updated_graph), now=T1)

    assert stamped is not None
    assert stamped.prior_version_iris == [base.version_iri]
    assert stamped.created_at == base.created_at
    assert stamped.modified_at == T1
    assert set(stamped.graph.objects(onto_iri, OWL.versionIRI)) == {
        URIRef(stamped.version_iri or "")
    }
    assert set(stamped.graph.objects(onto_iri, OWL.priorVersion)) == {
        URIRef(base.version_iri)
    }
    created_values = list(stamped.graph.objects(onto_iri, DCTERMS.created))
    assert len(created_values) == 1
    assert datetime.fromisoformat(str(created_values[0])) == base.created_at


class _DummyTools:
    pass


def test_normalize_ontology_units_derives_a_child_of_the_base() -> None:
    base = _make_base_ontology()
    assert base.hash is not None and base.version_iri is not None

    doc_iri = URIRef("https://example.org/doc/alpha")
    delta_graph = RDFGraph()
    delta_graph.add((URIRef(f"{IRI}#Case"), RDF.type, OWL.Class))
    unit = ContentUnit(
        text="delta",
        index=0,
        doc_iri=doc_iri,
        graph=delta_graph,
        type=OutputType.ONTOLOGIES,
    )

    normalized, applied, provenance = normalize_ontology_units(
        units=[unit],
        tools=cast(ToolBox, _DummyTools()),
        base_ontology=base,
        require_base=True,
    )

    assert len(applied) == 1
    assert normalized.hash is not None and normalized.hash != base.hash
    assert normalized.version_iri is None
    assert [b.version_iri for b in normalized.bases] == [base.version_iri]
    assert len(provenance) == 0
    assert (URIRef(f"{IRI}#Case"), RDF.type, OWL.Class) in normalized.graph

    stamped = stamp_version(normalized, updates=applied, now=T1)
    assert stamped is not None
    assert stamped.prior_version_iris == [base.version_iri]
    assert stamped.version == "1.0.1"
    assert (URIRef(IRI), OWL.priorVersion, URIRef(base.version_iri)) in stamped.graph
