from rdflib import Namespace, URIRef

from ontocast.onto.tenancy import (
    DEFAULT_PROJECT,
    DEFAULT_TENANT,
    tenant_project_facts_name,
    tenant_project_ontologies_name,
)

DEFAULT_DOMAIN = "https://growgraph.dev"

# Cache entries are regenerable, so the on-disk cache is bounded automatically
# rather than growing without limit. Defined here rather than in tool.cache so
# that config.settings can carry the defaults without importing the tool layer.
DEFAULT_CACHE_MAX_BYTES = 1024**3  # 1 GB
DEFAULT_CACHE_PRUNE_EVERY = 256

ONTOLOGY_NULL_ID = "__null__"
ONTOLOGY_NULL_IRI = f"{DEFAULT_DOMAIN}/{ONTOLOGY_NULL_ID}/"
DEFAULT_IRI = f"{DEFAULT_DOMAIN}/facts/"
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
    "schema": "<https://schema.org/>",
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

#: Terms OntoCast mints for pipeline telemetry that no standard vocabulary
#: covers. Deliberately outside ``DEFAULT_IRI``: fact namespaces drive the
#: SHACL gate's repair scope, and pipeline metadata must never be a repair
#: target.
ONTOCAST = Namespace(f"{DEFAULT_DOMAIN}/ontocast#")

#: Classes and predicates the pipeline itself mints on provenance/chunk nodes.
#: Scaffolding, not domain vocabulary: the non-catalog-vocabulary check must not
#: report them as terms the renderer improvised, because no catalog will ever
#: supply them. Classes are here as well as predicates because a type is
#: recorded against its object IRI, so a namespace guard on the predicate
#: (``rdf:type``) never sees them.
PROVENANCE_METADATA_TERMS: frozenset[URIRef] = frozenset(
    {
        PROV.Entity,
        PROV.generatedAtTime,
        PROV.wasDerivedFrom,
        SCHEMA.Text,
        SCHEMA.position,
        SCHEMA.identifier,
        SCHEMA.articleSection,
        ONTOCAST.sectionLabelSource,
        ONTOCAST.sectionLabelConfidence,
    }
)

# RDF 1.2 term for linking a reification node to its quoted triple.
# Not yet in rdflib's RDF namespace, so we define it manually.
RDF_REIFIES = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
