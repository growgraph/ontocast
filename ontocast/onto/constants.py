from rdflib import Namespace, URIRef

from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)

DEFAULT_DOMAIN = "https://growgraph.dev"
ONTOLOGY_NULL_ID = "__null__"
ONTOLOGY_NULL_IRI = f"{DEFAULT_DOMAIN}/{ONTOLOGY_NULL_ID}/"
DEFAULT_IRI = f"{DEFAULT_DOMAIN}/facts/"
CHUNK_NULL_IRI = f"{DEFAULT_DOMAIN}/__null__/"
DEFAULT_DATASET = tenant_project_facts_name(DEFAULT_TENANT, DEFAULT_PROJECT)
DEFAULT_ONTOLOGIES_DATASET = tenant_project_ontologies_name(
    DEFAULT_TENANT, DEFAULT_PROJECT
)
COMMON_PREFIXES = {
    "xsd": "<http://www.w3.org/2001/XMLSchema#>",
    "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
    "rdfs": "<http://www.w3.org/2000/01/rdf-schema#>",
    "owl": "<http://www.w3.org/2002/07/owl#>",
    "dc": "<http://purl.org/dc/elements/1.1/>",
    "dcterms": "<http://purl.org/dc/terms/>",
    "skos": "<http://www.w3.org/2004/02/skos/core#>",
    "foaf": "<http://xmlns.com/foaf/0.1/>",
    "schema": "<http://schema.org/>",
    "prov": "<http://www.w3.org/ns/prov#>",
    "ex": "<http://example.org/>",
}

# Cross-domain vocabularies merged only at LLM ingest repair (not default serialization).
WELL_KNOWN_PREFIXES: dict[str, str] = {
    "qudt": "http://qudt.org/schema/qudt/",
    "unit": "http://qudt.org/vocab/unit/",
    "quantitykind": "http://qudt.org/vocab/quantitykind/",
    "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "time": "http://www.w3.org/2006/time#",
    "sh": "http://www.w3.org/ns/shacl#",
    "dcat": "http://www.w3.org/ns/dcat#",
    "void": "http://rdfs.org/ns/void#",
}


def prefix_lookup_for_ingest() -> dict[str, str]:
    """Prefix map for Turtle/JSON-LD ingest repair (COMMON + WELL_KNOWN, bare URIs)."""
    lookup: dict[str, str] = {}
    for prefix, uri in COMMON_PREFIXES.items():
        lookup[prefix] = uri.strip("<>")
    lookup.update(WELL_KNOWN_PREFIXES)
    return lookup


PROV = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("https://schema.org/")

# RDF 1.2 term for linking a reification node to its quoted triple.
# Not yet in rdflib's RDF namespace, so we define it manually.
RDF_REIFIES = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
