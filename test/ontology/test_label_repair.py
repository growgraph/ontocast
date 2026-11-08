"""Labels derived from local names for unlabelled minted terms."""

import pytest
from rdflib import OWL, RDF, RDFS, Literal, URIRef

from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.ontology_validation.label_repair import (
    derive_missing_labels,
    label_from_local_name,
)

pytestmark = pytest.mark.unit

EX = "https://example.com/onto#"


@pytest.mark.parametrize(
    ("local", "label"),
    [
        ("PerovskiteSample", "Perovskite sample"),
        ("hasPLRedShift", "has PL red shift"),
        ("ASiteComponent", "A site component"),
        ("XRDPattern", "XRD pattern"),
        ("sample_holder", "sample holder"),
        ("Sample3", "Sample 3"),
        ("Q12345", None),
        ("3f9a2b7c1d", None),
        ("12", None),
        ("", None),
    ],
)
def test_label_from_local_name(local: str, label: str | None) -> None:
    assert label_from_local_name(local) == label


def test_unlabelled_declared_terms_get_a_label() -> None:
    cls = URIRef(EX + "PerovskiteSample")
    prop = URIRef(EX + "hasPLRedShift")
    inserts = RDFGraph()
    inserts.add((cls, RDF.type, OWL.Class))
    inserts.add((prop, RDFS.domain, cls))
    labels = derive_missing_labels(inserts, None)
    assert set(labels) == {
        (cls, RDFS.label, Literal("Perovskite sample", lang="en")),
        (prop, RDFS.label, Literal("has PL red shift", lang="en")),
    }


def test_labelled_and_opaque_terms_are_left_alone() -> None:
    labelled = URIRef(EX + "Detector")
    snapshot_labelled = URIRef(EX + "Device")
    opaque = URIRef(EX + "Q42")
    inserts = RDFGraph()
    inserts.add((labelled, RDF.type, OWL.Class))
    inserts.add((labelled, RDFS.label, Literal("detector")))
    inserts.add((snapshot_labelled, RDFS.subClassOf, labelled))
    inserts.add((opaque, RDF.type, OWL.Class))
    snapshot = RDFGraph()
    snapshot.add((snapshot_labelled, RDFS.label, Literal("Device")))
    assert len(derive_missing_labels(inserts, snapshot)) == 0
