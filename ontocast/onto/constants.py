from rdflib import Namespace

DEFAULT_DOMAIN = "https://example.com"
ONTOLOGY_NULL_ID = "__null__"
ONTOLOGY_NULL_IRI = f"{DEFAULT_DOMAIN}/{ONTOLOGY_NULL_ID}"
DEFAULT_CHUNK_IRI = f"{DEFAULT_DOMAIN}/ch#"
CHUNK_NULL_ID = "__null__"
CHUNK_NULL_IRI = f"{DEFAULT_CHUNK_IRI}{CHUNK_NULL_ID}"
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
