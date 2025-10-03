ontology_instruction_fresh = """
Propose/develop a new domain ontology based on the provided document. When deciding on the name and scope, remember that the document you are given is just an example, so the ontology name, ontology identifier and scope should be at least one level of abstraction above the scope of the document."""


specific_ontology_instruction_fresh = """
1. all new abstract entities/classes/types or properties added to the new ontology must be linked to entities from basic ontologies (RDFS, OWL, schema etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc
2. propose a domain specific and succinct specifier  if for the new ontology, which should be an abbreviation, consistent with the Ontology property `ontology_id`, for example it could be `abc` for a hypothetical A... B... of C... Ontology.
3. derive from a proposed `ontology_id` an IRI (URI) using domain {current_domain}, for example `{current_domain}/abc`
4. explicitly use namespace `co:` for entities/properties placed in the proposed ontology.
"""


ontology_instruction_update = """
Update/complement the domain ontology {ontology_iri} provided below with abstract entities and relations that can be inferred from the document.

{ontology_desc}

Feel free to modify the description of the ontology to make it more accurate and complete, but to change neither the ontology IRI nor name.

```ttl
{ontology_str}
```
"""

specific_ontology_instruction_update = """
- all new abstract entities/classes/types or properties added to <{ontology_namespace}> ontology must be linked to entities from either domain ontology <{ontology_namespace}> or basic ontologies (RDFS, OWL, schema etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc
- add new constraints and axioms if needed."""


instructions = """
Follow the instructions:


{specific_ontology_instruction}
1. ontology must be provided in turtle (ttl) format as a single string.
2. (IMPORTANT) define all prefixes for all namespaces used in the ontology, etc rdf, rdfs, owl, schema, etc.
3. in case you are familiar with domain specific ontologies, feel free to use them. For example (Financial Industry Business Ontology (FIBO) in finance, or XBRL-to-RDF transformations.
4. do not add facts, or concrete entities from the document.
5. make sure newly introduced entities are well linked / described by their properties.
6. assign where possible correct units to numeric literals.
7. make sure that the semantic representation is faithful to the document, feel to use your knowledge and common sense to make the ontology more complete and accurate.
8. feel free to update/assign the version of the ontology using semantic versioning convention.
"""

failure_instruction = """
IMPORTANT: The previous attempt to generate ontology triples failed/was unsatisfactory.


It failed at the stage: {failure_stage}


{failure_reason}


Please address ALL the issues outlined in the critique. We will be penalized :( for each unaddressed issue."""

template_prompt = """

{ontology_instruction}


{instructions}


Here is the document:


```
{text}
```


{failure_instruction}


{format_instructions}

"""
