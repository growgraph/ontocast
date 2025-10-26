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
