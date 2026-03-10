from datetime import datetime

from rdflib import DCTERMS, OWL, RDF, XSD, Literal, URIRef

from ontocast.agent.normalize_ontology import normalize_ontology_units
from ontocast.onto.constants import PROV
from ontocast.onto.content_unit import ContentUnit, OutputType
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph


def _make_base_ontology() -> Ontology:
    base_iri = URIRef("https://example.org/onto")
    graph = RDFGraph()
    graph.add((base_iri, RDF.type, OWL.Ontology))
    graph.add((URIRef(f"{base_iri}#Person"), RDF.type, OWL.Class))
    return Ontology(graph=graph, iri=str(base_iri))


def test_derive_updated_version_refreshes_lineage_metadata() -> None:
    base = _make_base_ontology()
    assert base.hash is not None
    base_hash = base.hash

    onto_iri = URIRef(base.iri)
    updated_graph = base.graph.copy()
    updated_graph.add((URIRef(f"{base.iri}#Organization"), RDF.type, OWL.Class))
    updated_graph.add((onto_iri, PROV.wasDerivedFrom, URIRef("urn:hash:stale-parent")))
    updated_graph.add((onto_iri, DCTERMS.identifier, Literal("hash:stale-hash")))
    updated_graph.add(
        (
            onto_iri,
            DCTERMS.created,
            Literal("2001-01-01T00:00:00+00:00", datatype=XSD.dateTime),
        )
    )

    updated = base.derive_updated_version(updated_graph)

    assert updated.hash is not None
    assert updated.hash != base_hash
    assert updated.parent_hashes == [base_hash]
    assert updated.created_at is not None

    hash_identifiers = {
        str(obj)
        for _, _, obj in updated.graph.triples((onto_iri, DCTERMS.identifier, None))
        if str(obj).startswith("hash:")
    }
    parent_uris = {
        str(obj)
        for _, _, obj in updated.graph.triples((onto_iri, PROV.wasDerivedFrom, None))
    }
    created_values = [
        str(obj)
        for _, _, obj in updated.graph.triples((onto_iri, DCTERMS.created, None))
    ]

    assert hash_identifiers == {f"hash:{updated.hash}"}
    assert "hash:stale-hash" not in hash_identifiers
    assert parent_uris == {f"urn:hash:{base_hash}"}
    assert "urn:hash:stale-parent" not in parent_uris
    assert len(created_values) == 1
    assert datetime.fromisoformat(created_values[0]) == updated.created_at


class _DummyAggregator:
    def __init__(self, aggregated_graph: RDFGraph):
        self._aggregated_graph = aggregated_graph

    def aggregate_graphs(self, units: list[ContentUnit]) -> RDFGraph:
        return self._aggregated_graph


class _DummyTools:
    def __init__(self, aggregated_graph: RDFGraph):
        self.aggregator = _DummyAggregator(aggregated_graph)


def test_normalize_ontology_units_refreshes_lineage_for_updated_base() -> None:
    base = _make_base_ontology()
    assert base.hash is not None
    base_hash = base.hash

    doc_iri = URIRef("https://example.org/doc/alpha")
    delta_graph = RDFGraph()
    delta_graph.add((URIRef(f"{base.iri}#Case"), RDF.type, OWL.Class))
    unit = ContentUnit(
        text="delta",
        index=0,
        doc_iri=doc_iri,
        graph=delta_graph,
        type=OutputType.ONTOLOGIES,
    )

    normalized, applied = normalize_ontology_units(
        units=[unit],
        tools=_DummyTools(delta_graph),
        base_ontology=base,
        require_base=True,
    )

    onto_iri = URIRef(base.iri)
    assert len(applied) == 1
    assert normalized.hash is not None
    assert normalized.hash != base_hash
    assert normalized.parent_hashes == [base_hash]
    assert normalized.created_at is not None
    assert (URIRef(f"{base.iri}#Case"), RDF.type, OWL.Class) in normalized.graph
    assert (
        onto_iri,
        PROV.wasDerivedFrom,
        URIRef(f"urn:hash:{base_hash}"),
    ) in normalized.graph
