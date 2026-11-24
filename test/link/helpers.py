"""A small catalog and three documents that exercise every kind of link.

Catalog (``ex:``):

- ``Court`` is globally named; ``AppealCourt`` inherits it; ``LocalCourt``
  states ``false`` and stops it. ``Sample`` is not annotated.
- ``Article`` has key ``(doi)``; ``ecli`` is inverse-functional; ``seat`` is
  functional; ``Person`` is disjoint with ``Court``.
- ``Quantity`` is a value class, annotated globally named to show values are
  never linked anyway.

Documents:

- d1: the ECHR seated in Strasbourg, "Sample 1", an article with DOI
  ``10.1/X``, a metadata author with identifier ``0000-1``, a quantity.
- d2: the same court written in lower case, typed ``AppealCourt``, seated in
  Strasbourg; "Sample 1"; the same DOI with different spacing and case and a
  different label; the same author identifier under another name.
- d3: a court with the same name seated in Luxembourg (a functional
  conflict), and a ``LocalCourt`` of that name (not linked).
"""

from __future__ import annotations

from rdflib import OWL, RDF, RDFS, BNode, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, FOAF, PROV, XSD

from ontocast.config import (
    Config,
    FusekiConfig,
    LinkConfig,
    LLMConfig,
    PathConfig,
    ToolConfig,
)
from ontocast.onto.constants import GLOBALLY_NAMED, SCHEMA
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.toolbox import ToolBox
from test.ontology_helpers import DOMAIN, stamped_ontology

EX = Namespace(f"{DOMAIN}/lk#")
DOC1 = f"{DOMAIN}/doc/d1/"
DOC2 = f"{DOMAIN}/doc/d2/"
DOC3 = f"{DOMAIN}/doc/d3/"
VALUE_CLASSES = frozenset({str(EX.Quantity)})


def e(doc: str, local: str) -> URIRef:
    return URIRef(f"{doc}{local}")


def catalog_graph(*, court_globally_named: bool = True) -> RDFGraph:
    g = RDFGraph()
    for cls in (
        EX.Court,
        EX.AppealCourt,
        EX.LocalCourt,
        EX.Sample,
        EX.Article,
        EX.Person,
        EX.Quantity,
    ):
        g.add((cls, RDF.type, OWL.Class))
    g.add((EX.Court, GLOBALLY_NAMED, Literal(court_globally_named)))
    g.add((EX.AppealCourt, RDFS.subClassOf, EX.Court))
    g.add((EX.LocalCourt, RDFS.subClassOf, EX.Court))
    g.add((EX.LocalCourt, GLOBALLY_NAMED, Literal(False)))
    g.add((EX.Quantity, GLOBALLY_NAMED, Literal(True)))
    key_list = BNode()
    Collection(g, key_list, [EX.doi])
    g.add((EX.Article, OWL.hasKey, key_list))
    g.add((EX.ecli, RDF.type, OWL.InverseFunctionalProperty))
    g.add((EX.seat, RDF.type, OWL.FunctionalProperty))
    g.add((EX.Person, OWL.disjointWith, EX.Court))
    return g


def _doc_node(g: RDFGraph, doc: str) -> URIRef:
    """The document node as the pipeline mints it."""
    node = URIRef(doc.rstrip("/"))
    g.add((node, RDF.type, PROV.Entity))
    g.add((node, RDF.type, FOAF.Document))
    return node


def document(doc: str) -> RDFGraph:
    g = RDFGraph()
    node = _doc_node(g, doc)
    if doc == DOC1:
        court = e(doc, "echr")
        g.add((court, RDF.type, EX.Court))
        g.add((court, RDFS.label, Literal("European Court of Human Rights")))
        g.add((court, EX.seat, Literal("Strasbourg")))
        g.add((e(doc, "sample1"), RDF.type, EX.Sample))
        g.add((e(doc, "sample1"), RDFS.label, Literal("Sample 1")))
        g.add((e(doc, "paper"), RDF.type, EX.Article))
        g.add((e(doc, "paper"), RDFS.label, Literal("Paper one")))
        g.add((e(doc, "paper"), EX.doi, Literal("10.1/X")))
        g.add((e(doc, "janeDoe"), RDF.type, SCHEMA.Person))
        g.add((e(doc, "janeDoe"), RDFS.label, Literal("Jane Doe")))
        g.add((e(doc, "janeDoe"), DCTERMS.identifier, Literal("0000-1")))
        g.add((node, DCTERMS.creator, e(doc, "janeDoe")))
        quantity = URIRef(f"{doc}echr.budget")
        g.add((quantity, RDF.type, EX.Quantity))
        g.add((quantity, RDFS.label, Literal("Budget")))
        g.add((quantity, EX.value, Literal("10", datatype=XSD.integer)))
        g.add((court, EX.budget, quantity))
    elif doc == DOC2:
        court = e(doc, "court")
        g.add((court, RDF.type, EX.AppealCourt))
        g.add((court, RDFS.label, Literal("european court of human rights")))
        g.add((court, EX.seat, Literal("Strasbourg")))
        g.add((e(doc, "sampleOne"), RDF.type, EX.Sample))
        g.add((e(doc, "sampleOne"), RDFS.label, Literal("Sample 1")))
        g.add((e(doc, "study"), RDF.type, EX.Article))
        g.add((e(doc, "study"), RDFS.label, Literal("A different label")))
        g.add((e(doc, "study"), EX.doi, Literal(" 10.1/x ")))
        g.add((e(doc, "jDoe"), RDF.type, SCHEMA.Person))
        g.add((e(doc, "jDoe"), RDFS.label, Literal("J. Doe")))
        g.add((e(doc, "jDoe"), DCTERMS.identifier, Literal("0000-1")))
        g.add((node, DCTERMS.creator, e(doc, "jDoe")))
        quantity = URIRef(f"{doc}court.budget")
        g.add((quantity, RDF.type, EX.Quantity))
        g.add((quantity, RDFS.label, Literal("Budget")))
        g.add((court, EX.budget, quantity))
    elif doc == DOC3:
        court = e(doc, "echr")
        g.add((court, RDF.type, EX.Court))
        g.add((court, RDFS.label, Literal("European Court of Human Rights")))
        g.add((court, EX.seat, Literal("Luxembourg")))
        local = e(doc, "localCourt")
        g.add((local, RDF.type, EX.LocalCourt))
        g.add((local, RDFS.label, Literal("European Court of Human Rights")))
    return g


def make_tools(
    *, link_enabled: bool = False, court_globally_named: bool = True
) -> ToolBox:
    """A ToolBox over the in-memory store holding the catalog (no documents yet)."""
    tools = ToolBox(
        Config(
            tool_config=ToolConfig(
                path_config=PathConfig(),
                llm_config=LLMConfig(),
                link=LinkConfig(enabled=link_enabled),
                # The in-memory store, whatever the shell exports.
                fuseki=FusekiConfig(uri=None, auth=None),
            )
        )
    )
    tools.ontology_manager.add_ontology(
        stamped_ontology(
            catalog_graph(court_globally_named=court_globally_named), ontology_id="lk"
        ),
        skip_vector_index=True,
    )
    return tools


async def store_documents(tools: ToolBox, docs: list[str]) -> None:
    store = tools.triple_store_manager
    assert store is not None
    for doc in docs:
        await store.aserialize(document(doc), graph_uri=doc)


def canonical(graph: RDFGraph) -> str:
    """A graph's triples and reifications as sorted N-Triples lines.

    Turtle writers order statements by hash, so equal graphs can serialise to
    different bytes; this form is equal exactly when the content is.
    """
    lines = {f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in graph}
    if graph.has_reifications:
        lines |= {
            f"{reifier.n3()} reifies {s.n3()} {p.n3()} {o.n3()} ."
            for reifier, (s, p, o) in graph.reifications
        }
    return "\n".join(sorted(lines))
