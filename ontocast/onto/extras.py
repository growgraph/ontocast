from ontocast.onto.constants import ONTOLOGY_NULL_ID, ONTOLOGY_NULL_IRI
from ontocast.onto.ontology import Ontology
from ontocast.onto.rdfgraph import RDFGraph

NULL_ONTOLOGY = Ontology(
    ontology_id=ONTOLOGY_NULL_ID,
    title="null title",
    description="null description",
    graph=RDFGraph(),
    iri=ONTOLOGY_NULL_IRI,
)
