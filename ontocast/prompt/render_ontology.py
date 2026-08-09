from ontocast.onto.constants import DEFAULT_IRI

template_prompt = """
{preamble}

{intro_instruction}

{ontology_instruction}

{user_instruction}

{improvement_instruction}

{ontology_ttl}

{text}

{external_evidence}

{output_instruction}

{format_instructions}
"""

intro_instruction_fresh = """
1. Develop a new domain ontology based on the provided document. When deciding on the name and scope, remember that the document you are given is an example, so the ontology name, ontology identifier and scope should be at least one level of abstraction above the scope of the document.
2. Propose a domain specific and succinct specifier if for the new ontology, which should be an abbreviation, consistent with the Ontology property `ontology_id`, for example it could be `abc` for a hypothetical A... B... of C... Ontology.
3. From the proposed `ontology_id` derive an IRI (URI) using domain {current_domain}, for example `{current_domain}/abc`
"""


intro_instruction_update_single = """
Complement the domain ontology {ontology_iri} provided below with abstract entities and relations that can be inferred from the document or known to hold in the domain the document pertains to.

{ontology_desc}

Emit ONLY new abstract entities/relations as GraphUpdate inserts. Do NOT restate triples already present in the provided ontology graph. Do not change the ontology IRI, prefix, or ontology_id.
"""


intro_instruction_update_multi = """
Complement the provided multi-source ontology context with abstract entities and relations inferred from the document.

{ontology_desc}

The context may combine patches from multiple catalog ontologies:
{source_list}

Rules:
- Propose ONLY *new* schema (classes/properties/axioms). Do NOT restate triples already in the provided graph.
- Place each new entity under the matching existing domain prefix from the domain-ontologies clause below.
- Never invent a new domain ontology id/IRI; never collapse distinct source namespaces.
- Cross-links to other listed ontologies via existing IRIs are fine; do not mint terms under foreign namespaces you do not own.
"""

prefix_instruction = """Use {domain_ontologies_clause} for entities/properties placed in the domain ontologies. DECLARE all prefixes in preamble! New terms MUST use only those listed domain prefixes."""


general_ontology_instruction = f"""
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

8. **For measurable properties, model units as IRIs first:** reference or declare a unit individual (e.g. a `qudt:Unit` instance) and point to it via an object property with a class range. Only fall back to a literal code on a declared string-code property (e.g. `schema:unitCode "DAY"`) or an `rdfs:comment` when no unit individual is available — never attach a string literal to a property whose range is a class.

9. **Never model ontology classes/properties under `cd:`.** The `cd:` namespace (`{DEFAULT_IRI}`) is reserved for factual instances only.

10. **When introducing entities from other domain ontologies, declare their namespace prefixes** (e.g., `@prefix foaf: <http://xmlns.com/foaf/0.1/> .` or `@prefix dcterms: <http://purl.org/dc/terms/> .`).

11. **Complement only:** do not re-emit schema triples that already appear in the provided ontology context; GraphUpdate inserts must be new.

{{search_guidelines}}
"""


improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the ontology was not deemed accurate by Critic, who left the following suggestions for improvement:

{suggestions_instruction}
"""
