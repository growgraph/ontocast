"""Helpers for deriving prompt-ready domain ontology namespace context."""

from rdflib import OWL, RDF, RDFS, Graph, URIRef

from ontocast.onto.constants import COMMON_PREFIXES, DEFAULT_IRI
from ontocast.onto.ontology import Ontology

_STANDARD_NAMESPACES: frozenset[str] = frozenset(
    uri.strip("<>") for uri in COMMON_PREFIXES.values()
) | {"https://schema.org/", DEFAULT_IRI}


def extract_domain_prefix_pairs(ontology: Ontology) -> list[tuple[str, str]]:
    """Return domain prefix/namespace pairs present in ontology graph."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for prefix, namespace_uri in ontology.graph.namespaces():
        if not prefix:
            continue
        namespace = str(namespace_uri)
        if namespace in _STANDARD_NAMESPACES:
            continue
        pair = (prefix, namespace)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    if pairs:
        return pairs

    if ontology.prefix and ontology.namespace:
        return [(ontology.prefix, ontology.namespace)]
    return []


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


def _is_opaque_local_name(local: str) -> bool:
    """Return True when a local IRI name carries no human-readable semantics.

    Heuristic: opaque if the local name is a Q/P-number (Wikidata style),
    a pure numeric string, or a hex/hash string of length >= 8.
    """
    if not local:
        return False
    if local[0] in ("Q", "P") and local[1:].isdigit():
        return True
    if local.isdigit():
        return True
    hex_chars = set("0123456789abcdefABCDEF")
    if len(local) >= 8 and all(c in hex_chars for c in local):
        return True
    return False


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
        if not _is_opaque_local_name(local):
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
        if _is_opaque_local_name(str(p).rsplit("/", 1)[-1].rsplit("#", 1)[-1])
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
