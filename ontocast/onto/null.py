from rdflib import URIRef
from rdflib.namespace import OWL, RDF

from ontocast.onto.constants import ONTOLOGY_NULL_IRI
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph

NULL_ONTOLOGY = Ontology(
    ontology_id=None,
    title=None,
    description=None,
    graph=RDFGraph(),
    iri=ONTOLOGY_NULL_IRI,
)
null_iri_ref = URIRef(ONTOLOGY_NULL_IRI)
# Add a marker in the graph to denote this is a null ontology
NULL_ONTOLOGY.graph.add((null_iri_ref, RDF.type, OWL.Ontology))
