"""The link pass: what one document's facts graph links, and what it never links."""

import pytest
from rdflib import RDF, RDFS, SKOS, Literal, URIRef

from ontocast.link import IdentityDeclarations, links_for, match_node_iri
from ontocast.link.links import MATCH_KIND, key_value
from ontocast.onto.rdfgraph import RDFGraph
from ontocast.tool.agg.uri_builder import normalize_local_name, to_lower_camel_case
from test.link.helpers import (
    DOC1,
    DOC2,
    DOC3,
    DOMAIN,
    EX,
    VALUE_CLASSES,
    canonical,
    catalog_graph,
    document,
    e,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def declarations() -> IdentityDeclarations:
    return IdentityDeclarations.of([catalog_graph()])


def _links(doc: str, declarations: IdentityDeclarations) -> RDFGraph:
    return links_for(
        document(doc), declarations, domain=DOMAIN, value_classes=VALUE_CLASSES
    )


def _targets(links: RDFGraph, entity: URIRef) -> set[URIRef]:
    return {o for o in links.objects(entity, SKOS.closeMatch) if isinstance(o, URIRef)}


def _kind(links: RDFGraph, node: URIRef) -> str:
    return str(next(links.objects(node, MATCH_KIND)))


def test_declarations_are_read_from_the_catalog(
    declarations: IdentityDeclarations,
) -> None:
    assert declarations.declaring_classes([str(EX.AppealCourt)]) == {str(EX.Court)}
    assert declarations.declaring_classes([str(EX.LocalCourt)]) == frozenset()
    assert declarations.declaring_classes([str(EX.Sample)]) == frozenset()
    assert declarations.keys_for([str(EX.Article)]) == [
        (str(EX.Article), (str(EX.doi),))
    ]
    assert str(EX.ecli) in declarations.inverse_functional
    assert str(EX.seat) in declarations.functional
    assert declarations.are_disjoint([str(EX.Person)], [str(EX.AppealCourt)])


def test_conflicting_annotations_resolve_to_false() -> None:
    declarations = IdentityDeclarations.of(
        [catalog_graph(), catalog_graph(court_globally_named=False)]
    )
    assert declarations.declaring_classes([str(EX.Court)]) == frozenset()


def test_a_globally_named_entity_links_by_name_across_documents(
    declarations: IdentityDeclarations,
) -> None:
    first = _targets(_links(DOC1, declarations), e(DOC1, "echr"))
    second = _targets(_links(DOC2, declarations), e(DOC2, "court"))
    shared = first & second
    assert len(shared) == 1
    assert _kind(_links(DOC1, declarations), next(iter(shared))) == "label"


def test_a_key_links_under_different_labels(declarations: IdentityDeclarations) -> None:
    first = _targets(_links(DOC1, declarations), e(DOC1, "paper"))
    second = _targets(_links(DOC2, declarations), e(DOC2, "study"))
    assert first == second
    assert _kind(_links(DOC1, declarations), next(iter(first))) == "key"


def test_a_caller_identifier_links_metadata_entities(
    declarations: IdentityDeclarations,
) -> None:
    first = _targets(_links(DOC1, declarations), e(DOC1, "janeDoe"))
    second = _targets(_links(DOC2, declarations), e(DOC2, "jDoe"))
    assert first == second
    assert _kind(_links(DOC1, declarations), next(iter(first))) == "identifier"


def test_document_local_names_are_never_linked(
    declarations: IdentityDeclarations,
) -> None:
    assert not _targets(_links(DOC1, declarations), e(DOC1, "sample1"))
    assert not _targets(_links(DOC3, declarations), e(DOC3, "localCourt"))


def test_values_and_literals_are_never_linked(
    declarations: IdentityDeclarations,
) -> None:
    links = _links(DOC1, declarations)
    linked = set(links.subjects(SKOS.closeMatch, None))
    assert URIRef(f"{DOC1}echr.budget") not in linked
    assert all(isinstance(s, URIRef) and str(s).startswith(DOC1) for s in linked)


def test_a_document_local_key_value_forms_no_key() -> None:
    declarations = IdentityDeclarations.of([catalog_graph()])
    assert key_value(URIRef(f"{DOC1}somewhere"), declarations.catalog_terms) is None
    assert key_value(EX.Court, declarations.catalog_terms) == str(EX.Court)
    assert key_value(Literal(" 10.1/X "), declarations.catalog_terms) == "10.1/x"


def test_an_inverse_functional_value_links() -> None:
    declarations = IdentityDeclarations.of([catalog_graph()])
    graphs = []
    for doc in (DOC1, DOC2):
        g = RDFGraph()
        case = e(doc, "case")
        g.add((case, RDF.type, EX.Case))
        g.add((case, RDFS.label, Literal(f"case in {doc}")))
        g.add((case, EX.ecli, Literal("ECLI:CE:ECHR:2020:1")))
        graphs.append(links_for(g, declarations, domain=DOMAIN))
    assert _targets(graphs[0], e(DOC1, "case")) == _targets(graphs[1], e(DOC2, "case"))


def test_nothing_declared_gives_an_empty_graph() -> None:
    links = links_for(document(DOC3), IdentityDeclarations(), domain=DOMAIN)
    assert len(links) == 0


def test_the_link_pass_is_pure(declarations: IdentityDeclarations) -> None:
    once = canonical(_links(DOC1, declarations))
    assert all(canonical(_links(DOC1, declarations)) == once for _ in range(5))


def test_match_nodes_are_minted_under_the_domain() -> None:
    node = match_node_iri(DOMAIN, {"kind": "label", "class": "x", "value": "y"})
    assert str(node).startswith(f"{DOMAIN}/match/")
    assert node == match_node_iri(
        DOMAIN + "/", {"value": "y", "class": "x", "kind": "label"}
    )


@pytest.mark.parametrize(
    "text", ["identity links", "identity-links", "ontology_provenance"]
)
def test_minted_local_names_never_contain_a_hyphen(text: str) -> None:
    """The auxiliary graph names rely on this: no entity can be named like them."""
    from ontocast.tool.agg.normalizer import EntityRepresentation
    from ontocast.tool.agg.uri_builder import EntityRole
    from ontocast.tool.representation_text import normalize_text

    assert "-" not in to_lower_camel_case(normalize_text(text))
    rep = EntityRepresentation(
        iri=URIRef(f"https://growgraph.dev/facts/{text.replace(' ', '_')}_2"),
        normal_form=normalize_text(text),
        types=[],
        properties=[],
        labels=[],
        is_ontology_entity=False,
    )
    assert "-" not in normalize_local_name(rep, EntityRole.INSTANCE)
