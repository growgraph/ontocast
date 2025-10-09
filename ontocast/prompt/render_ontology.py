template_prompt = """
{preamble}

{intro_instruction}

{ontology_instruction}

{user_instruction}

{ontology_ttl}

{text}

### OUTPUT instructions
{output_instruction}
"""

system_preamble = """
# INSTRUCTION
You are an expert in SPARQL and ontology engineering.

"""

# no_seed_first_visit
intro_instruction_first_visit_no_seed = """
0. Develop a new domain ontology based on the provided document. When deciding on the name and scope, remember that the document you are given is an example, so the ontology name, ontology identifier and scope should be at least one level of abstraction above the scope of the document.
1. propose a domain specific and succinct specifier if for the new ontology, which should be an abbreviation, consistent with the Ontology property `ontology_id`, for example it could be `abc` for a hypothetical A... B... of C... Ontology.
2. derive from a proposed `ontology_id` an IRI (URI) using domain {current_domain}, for example `{current_domain}/abc`


"""


# no_seed_ontology_first_visit
instruction_first_visit_no_seed = """
"""

# with_seed_ontology_first_visit
intro_instruction_first_visit_seed = """
Update/modify the domain ontology {ontology_iri} provided below with abstract entities and relations that can be inferred from the document or known to hold in the domain the document pertains to.

{ontology_desc}

Feel free to modify the description of the ontology to make it more accurate and complete, but to change neither the ontology IRI nor name.
"""

# univeral
general_ontology_instruction = """
### GENERAL
0. use prefix `co:` for entities/properties placed in the current domain ontology.
1. all abstract entities/classes/types or properties added to ontology must be linked to entities from basic ontologies (RDFS, OWL, schema etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc or connected to newly introduced entities in the current ontology
3. do not introduce entities known to you from other domain ontologies, rather connect new entities to known ontologies
4. do not add facts, or concrete entities from the document
5. make sure newly introduced entities are well linked / described by their properties
6. define units associated with measurable quantities.
7. make sure that the semantic representation is faithful to the document, use your knowledge and common sense to make the ontology more complete and accurate.
8. update/assign the version of the ontology using semantic versioning convention.
"""

# no_seed_ontology_first_visit
output_instruction_ttl = """
1. ontology must be provided in turtle format as a single string
2. define all prefixes for all namespaces used in the ontology, etc rdf, rdfs, owl, schema, etc.
"""


output_instruction_sparql = """
1. generate SPARQL operations that modify the existing ontology, not replace it entirely
Generate operations in the following structured format:

## ADD Section
- INSERT new ontology classes, properties, and relationships
- Add new domain-specific concepts
- Extend existing hierarchies

## UPDATE Section
- MODIFY existing ontology elements
- Update descriptions, labels, or relationships
- Refine existing concepts

## REMOVE Section
- DELETE obsolete ontology elements
- Remove outdated concepts or relationships
- Clean up redundant elements
"""
