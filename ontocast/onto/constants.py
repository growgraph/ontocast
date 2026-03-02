from rdflib import Namespace, URIRef

DEFAULT_DOMAIN = "https://growgraph.dev"
ONTOLOGY_NULL_ID = "__null__"
ONTOLOGY_NULL_IRI = f"{DEFAULT_DOMAIN}/{ONTOLOGY_NULL_ID}"
DEFAULT_IRI = f"{DEFAULT_DOMAIN}/facts"
CHUNK_NULL_IRI = f"{DEFAULT_DOMAIN}/__null__"
DEFAULT_DATASET = "dataset0"
DEFAULT_ONTOLOGIES_DATASET = "ontologies"
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
PROV = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("https://schema.org/")

# RDF 1.2 term for linking a reification node to its quoted triple.
# Not yet in rdflib's RDF namespace, so we define it manually.
RDF_REIFIES = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
