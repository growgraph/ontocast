"""Helpers for deriving prompt-ready domain ontology namespace context."""

from rdflib import OWL, RDF, RDFS, XSD, Graph, URIRef

from ontocast.onto.constants import DEFAULT_IRI
from ontocast.onto.iri_policy import is_opaque_local_name
from ontocast.onto.ontology import Ontology

#: The RDF languages and the facts namespace: never a domain vocabulary, even
#: where a graph declares terms in them (``owl:Thing``).
_NON_DOMAIN_NAMESPACES: frozenset[str] = frozenset(
    {str(RDF), str(RDFS), str(OWL), str(XSD), DEFAULT_IRI}
)


def extract_domain_prefix_pairs_from_graph(
    graph: Graph,
    *,
    fallback_prefix: str | None = None,
    fallback_namespace: str | None = None,
) -> list[tuple[str, str]]:
    """The prefix and namespace of each vocabulary *graph* defines terms in.

    A namespace is listed when a subject of *graph* lives in it, under the
    prefix the graph shows that subject with, so the prompt's clause and its
    term sheet agree. A binding no term lives in is not listed -- a document IRI
    that terms only point to (``rdfs:isDefinedBy <…/prov-o#>``), an rdflib
    default -- and neither are the RDF languages and the facts namespace. A
    standard vocabulary in the context as an ontology (PROV-O, SOSA) is listed:
    its terms are the domain vocabulary. The vocabulary with the most terms
    comes first, so an ontology's own namespace leads the ones it borrows from.
    """
    found: dict[str, str] = {}
    terms: dict[str, int] = {}
    subjects = sorted(
        (s for s in set(graph.subjects()) if isinstance(s, URIRef)), key=str
    )
    for subject in subjects:
        try:
            prefix, namespace, _ = graph.namespace_manager.compute_qname(
                str(subject), generate=False
            )
        except (KeyError, ValueError):
            continue
        namespace = str(namespace)
        if not prefix or namespace in _NON_DOMAIN_NAMESPACES:
            continue
        found.setdefault(namespace, prefix)
        terms[namespace] = terms.get(namespace, 0) + 1
    pairs = [
        (found[namespace], namespace)
        for namespace in sorted(found, key=lambda ns: (-terms[ns], found[ns]))
    ]
    if pairs:
        return pairs

    if fallback_prefix and fallback_namespace:
        return [(fallback_prefix, fallback_namespace)]
    return []


def extract_domain_prefix_pairs(ontology: Ontology) -> list[tuple[str, str]]:
    """Return domain prefix/namespace pairs present in ontology graph."""
    return extract_domain_prefix_pairs_from_graph(
        ontology.graph,
        fallback_prefix=ontology.prefix,
        fallback_namespace=ontology.namespace,
    )


def format_ontologies_clause(pairs: list[tuple[str, str]]) -> str:
    """Format a human-readable ontology clause including prefix and namespace for prompts.

    Produces e.g. "domain ontology `fcaont:` (<https://example.org/fcaont/>)" so a
    single variable conveys both which prefix to use and which namespace it maps to.
    """
    if not pairs:
        return "domain ontology namespaces declared in the provided ontology graph"
    items = [f"`{prefix}:` (<{namespace}>)" for prefix, namespace in pairs]
    if len(items) == 1:
        return f"domain ontology {items[0]}"
    return f"domain ontologies {', '.join(items)}"


def _qname(graph: Graph, uri: URIRef) -> str:
    """Return the prefixed name for *uri*, falling back to the full IRI."""
    try:
        qn = graph.namespace_manager.qname(uri)
        return qn
    except Exception:
        return str(uri)


def build_label_index(graph: Graph) -> str:
    """Build a label→IRI index for ontologies with opaque local names.

    Only emits a section when the graph contains opaque IRIs (Wikidata-style
    Q/P codes, hash IDs, etc.) so the LLM can map text mentions to canonical IRIs.
    Returns an empty string when all local names are already human-readable.
    """
    entries: list[tuple[str, str]] = []
    for subj, _, label_node in graph.triples((None, RDFS.label, None)):
        if not isinstance(subj, URIRef):
            continue
        local = str(subj).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        if not is_opaque_local_name(local):
            continue
        qn = _qname(graph, subj)
        entries.append((str(label_node), qn))

    if not entries:
        return ""

    entries.sort(key=lambda t: t[0].lower())
    lines = [
        "\n\n# TERM INDEX",
        "The ontology uses opaque IRIs. Use this index to map text mentions to their canonical IRI.",
        "label → IRI",
    ]
    for label, qn in entries:
        lines.append(f'  "{label}" → {qn}')
    return "\n".join(lines)


def build_property_summary(graph: Graph) -> str:
    """Build a resolved property summary for ontologies with opaque local names.

    Resolves domain/range opaque IRIs to their rdfs:label so the LLM does not
    need to mentally join triples. Returns an empty string when no properties
    with opaque IRIs are found.
    """
    label_map: dict[str, str] = {}
    for subj, _, label_node in graph.triples((None, RDFS.label, None)):
        if isinstance(subj, URIRef):
            label_map[str(subj)] = str(label_node)

    prop_types = (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property)
    props: list[URIRef] = []
    for pt in prop_types:
        for subj, _, _ in graph.triples((None, RDF.type, pt)):
            if isinstance(subj, URIRef):
                props.append(subj)

    opaque_props = [
        p
        for p in props
        if is_opaque_local_name(str(p).rsplit("/", 1)[-1].rsplit("#", 1)[-1])
    ]
    if not opaque_props:
        return ""

    lines = [
        "\n\n# PROPERTY SUMMARY (with resolved labels)",
        "Format: IRI  label  [domain: label (IRI)]  [range: label (IRI)]",
    ]
    for prop in sorted(opaque_props, key=lambda p: label_map.get(str(p), str(p))):
        prop_label = label_map.get(str(prop), "?")
        qn = _qname(graph, prop)
        domain_parts: list[str] = []
        for _, _, d in graph.triples((prop, RDFS.domain, None)):
            if isinstance(d, URIRef):
                dl = label_map.get(str(d), "")
                dqn = _qname(graph, d)
                domain_parts.append(f"{dl} ({dqn})" if dl else dqn)
        range_parts: list[str] = []
        for _, _, r in graph.triples((prop, RDFS.range, None)):
            if isinstance(r, URIRef):
                rl = label_map.get(str(r), "")
                rqn = _qname(graph, r)
                range_parts.append(f"{rl} ({rqn})" if rl else rqn)
        row = f'  {qn}  "{prop_label}"'
        if domain_parts:
            row += f"  domain: {', '.join(domain_parts)}"
        if range_parts:
            row += f"  range: {', '.join(range_parts)}"
        lines.append(row)
    return "\n".join(lines)


def build_ontology_index(graph: Graph) -> str:
    """Combine label index and property summary into a single prompt appendix.

    Returns an empty string when the ontology does not use opaque IRIs.
    """
    return build_label_index(graph) + build_property_summary(graph)
