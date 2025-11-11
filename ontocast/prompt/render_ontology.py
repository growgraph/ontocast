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

Feel free to update the description of the ontology to make it more accurate and complete, do not change neither ontology IRI nor predix nor id.
"""

prefix_instruction = """Use prefix `{ontology_prefix}` for entities/properties placed in the current domain ontology. DECLARE the prefix in preamble!"""
prefix_instruction_fresh = """Define a new prefix for the current domain ontology. DECLARE the prefix in preamble!"""


general_ontology_instruction = """
### GENERAL

1. **Only model abstract concepts — no instances or facts from the document** (e.g., no specific case names, dates, or people).

2. **All abstract entities (classes/properties) must connect to:**
   - **Standard vocabularies (RDFS, OWL, schema.org, SKOS) via rdfs:subClassOf, rdf:type, rdfs:subPropertyOf, etc.**
   - **OR other entities within this ontology**
   - **Example: `legal:CourtDecision rdfs:subClassOf schema:Event .`**

3. **Every new entity must have:**
   - **rdfs:label (required)**
   - **rdfs:comment describing its purpose (required)**
   - **At least one relationship to existing classes/properties**

4. **Ensure ontology faithfully represents domain semantics from the document.** Use **domain knowledge to add implicit relationships** not explicitly stated but clearly implied.

5. **Maintain consistency with existing conventions:**
   - **Language: Use same language for labels/comments as existing ontology**
   - **Naming: Follow existing PascalCase/camelCase patterns**
   - **Structure: Respect existing hierarchy depth and property usage patterns**

6. {prefix_instruction}

7. **Define property characteristics when applicable:**
   - **owl:FunctionalProperty** — property has at most one value (e.g., `foaf:homepage`, `dcterms:identifier`)
   - **owl:InverseFunctionalProperty** — value uniquely identifies the subject (e.g., `foaf:mbox`, `schema:email`)
   - **owl:TransitiveProperty** — if A→B and B→C, then A→C (e.g., `skos:broader`, `org:subOrganizationOf`)
   - **owl:SymmetricProperty** — if A→B, then B→A (e.g., `foaf:knows`, `schema:relatedTo`)
   - **owl:AsymmetricProperty** — if A→B, then NOT B→A (e.g., `org:hasSubOrganization`, `prov:wasDerivedFrom`)
   - **owl:ReflexiveProperty** — every entity relates to itself (e.g., `owl:sameAs`)
   - **owl:IrreflexiveProperty** — no entity relates to itself (e.g., `owl:differentFrom`)

8. **For measurable properties, specify units using schema:unitCode, rdfs:comment, or explicit unit classes** (e.g., `schema:duration schema:unitCode "DAY"` or `time:numericDuration rdfs:comment "Duration measured in days"`).

9. **When introducing entities from other domain ontologies, declare their namespace prefixes** (e.g., `@prefix foaf: <http://xmlns.com/foaf/0.1/> .` or `@prefix dcterms: <http://purl.org/dc/terms/> .`).
"""


improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the ontology was not deemed accurate by Critic, who left the following suggestions for improvement:

{suggestions_instruction}
"""
