template_prompt = """
{preamble}

{intro_instruction}

{ontology_instruction}

{user_instruction}

{improvement_instruction}

{ontology_ttl}

{text}

{output_instruction}

{format_instructions}
"""

intro_instruction_fresh = """
1. Develop a new domain ontology based on the provided document. When deciding on the name and scope, remember that the document you are given is an example, so the ontology name, ontology identifier and scope should be at least one level of abstraction above the scope of the document.
2. Propose a domain specific and succinct specifier if for the new ontology, which should be an abbreviation, consistent with the Ontology property `ontology_id`, for example it could be `abc` for a hypothetical A... B... of C... Ontology.
3. From the proposed `ontology_id` derive an IRI (URI) using domain {current_domain}, for example `{current_domain}/abc`
"""


intro_instruction_update = """
Update/modify the domain ontology {ontology_iri} provided below with abstract entities and relations that can be inferred from the document or known to hold in the domain the document pertains to.

{ontology_desc}

Feel free to modify the description of the ontology to make it more accurate and complete, but to change neither the ontology IRI nor name.
"""

prefix_instruction_fresh = """
Use prefix based on the proposed `ontology_id`
"""

prefix_instruction_update = """
Use prefix `{ontology_prefix}` for entities/properties placed in the current domain ontology.
"""


general_ontology_instruction = """
### GENERAL
1. {prefix_instruction}
2. all abstract entities/classes/types or properties added to ontology must be linked to entities from basic ontologies (RDFS, OWL, schema etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc or connected to newly introduced entities in the current ontology
3. do not introduce entities known to you from other domain ontologies, rather connect new entities to known ontologies
4. do not add facts or concrete entities from the document
5. make sure newly introduced entities are well linked / described by their properties
6. define units associated with measurable quantities.
7. make sure that the semantic representation is faithful to the document, use your knowledge and common sense to make the ontology more complete and accurate.
8. update/assign the version of the ontology using semantic versioning convention.
"""


improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the ontology was not deemed accurate by Critic, who left the following suggestions for improvement:

{suggestions_instruction}
"""
