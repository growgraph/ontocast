from ontocast.onto.constants import DEFAULT_IRI

template_prompt = """
{preamble}

{intro_instruction}

{ontology_instruction}

{plan_instruction}

{user_instruction}

{improvement_instruction}

{ontology_ttl}

{text}

{external_evidence}

{output_instruction}

{format_instructions}
"""

#: Fills the home-ontology step: one proposal per document, before the
#: per-unit fan-out. The catalog clause lists the ids a document may extend.
home_ontology_instruction = """
Decide which ontology this document's new vocabulary belongs to.

The document is an example of its domain: name and scope the ontology at least one level of abstraction above the document itself.

{catalog_clause}

Return:
- `ontology_id`: the short lowercase handle (2-32 characters: letters, digits, `_` or `-`, starting with a letter), for example `abc` for a hypothetical A... B... of C... Ontology. Return an EXISTING id from the catalog above when one covers the document's domain; otherwise a new one.
- `title`: a short noun phrase.
- `description`: 3-4 sentences on domain, purpose, scope and applicability.

Do not return an IRI: it is derived from the id.
"""

home_ontology_catalog_clause_empty = "The catalog holds no ontology yet."

home_ontology_catalog_clause = """The catalog holds these ontologies (id: title):
{catalog_list}
"""

#: The fresh render: terms only, under the home ontology minted by code.
intro_instruction_fresh = """
Develop the domain ontology `{ontology_id}` ({title}) from the provided document.

{description}

Every new class and property lives under the namespace <{namespace}>, bound to the prefix `{ontology_id}:`. Do NOT write an `owl:Ontology` header, version, title or description triples: identity and versioning are added by the system.
"""


intro_instruction_update_single = """
Complement the domain ontology {ontology_iri} provided below with reusable domain classes and relations that can be inferred from the document or known to hold in the domain the document pertains to.

{ontology_desc}

Emit ONLY new classes/relations as GraphUpdate inserts. Do NOT restate triples already present in the provided ontology graph. Do not change the ontology IRI, prefix, or ontology_id.
"""


home_namespace_clause = """
Every new class and property of this document lives under the namespace <{namespace}>, bound to the prefix `{ontology_id}:`. Terms under any other namespace are used by their IRIs only: never declare a new term in another namespace, and never add an axiom whose subject is a term of another namespace.
"""


intro_instruction_update_multi = """
Complement the provided multi-source ontology context with reusable domain classes and relations inferred from the document.

{ontology_desc}

The context may combine patches from multiple catalog ontologies:
{source_list}

Rules:
- Propose ONLY *new* schema (classes/properties/axioms). Do NOT restate triples already in the provided graph.
- Place each new entity under the document's home namespace when one is named below; otherwise under the matching existing domain prefix from the domain-ontologies clause below.
- Never invent a new domain ontology id/IRI; never collapse distinct source namespaces.
- Cross-links to other listed ontologies via existing IRIs are fine; do not mint terms under foreign namespaces you do not own.
"""

#: Prepended to either update intro when the context was assembled by vector
#: retrieval. The chapter is then a *retrieved subset*, and a prompt that
#: presents it as the whole ontology invites two errors this block heads off:
#: re-minting terms the catalog already defines (they simply were not
#: retrieved), and deleting statements whose justification lives in the
#: unretrieved remainder.
partial_context_notice = """
IMPORTANT — PARTIAL CONTEXT: the ontology chapter below is a RETRIEVED SUBSET
of the catalog ontologies, selected for relevance to this text. It is NOT the
complete ontology. Therefore:
- The catalog may already define a concept that is not shown here. Prefer
  reusing and specializing the terms you can see; introduce a new term only
  when nothing shown fits, name it by its standard domain terminology, and
  place it under the matching existing domain prefix.
- Never delete or rewrite a shown statement unless the document contradicts it
  AND you redeclare the corrected statement in the same update. Absence of
  supporting context is not evidence of error — the support may simply not
  have been retrieved.
"""

prefix_instruction = """Use {domain_ontologies_clause} for entities/properties placed in the domain ontologies. DECLARE all prefixes in preamble! New terms MUST use only those listed domain prefixes."""


general_ontology_instruction = f"""
### GENERAL

1. **Model the domain, not the document.** Mint a class only at the level a second document in the same domain would reuse. A particular case the document reports (a named case, a person, a date, one measured value) is an individual or a property value in the facts graph, not a class.

2. **All new classes and properties must connect to the rest of the vocabulary:**
   - **A class: rdfs:subClassOf a class of this ontology or of a standard vocabulary (RDFS, OWL, schema.org, SKOS).**
   - **A property: through its rdfs:domain and rdfs:range; rdfs:subPropertyOf only a more general property.**
   - **A parent is of the same kind as its child. `schema:Property` and `rdf:Property` are classes: never a property's parent.**
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

5a. **Subclass or property value?** A distinction that varies from one instance to the next (a kind, a method, a status, a category) is a property with values: an object property whose values are SKOS concepts, or a datatype property. It is never a family of sibling subclasses. Example: `ex:Vehicle` with `ex:hasFuelType` and `ex:Diesel a skos:Concept`, `ex:Petrol a skos:Concept`, not `ex:DieselVehicle` and `ex:PetrolVehicle`. A SKOS concept is the only individual an ontology update may declare: a named person, place, organisation or event belongs in the facts graph. A subclass is justified only when it carries properties or restrictions of its own.

5b. **Reuse before minting, and stay shallow.** When a term in the ontology context or a well-known vocabulary (schema.org, SKOS, SOSA/SSN, QUDT, PROV-O) names the concept, use it. Attach a new class at most two levels below an existing class, and do not build chains of single-child intermediate classes.

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


#: The document plan, shown to every unit: the terms it may mint and the
#: existing terms it must use instead of minting.
plan_instruction_template = """
### DOCUMENT PLAN

The terms of this document were planned with the whole document in view. Mint new classes and properties ONLY from this list, with the local names, labels and parents given:

{minted_lines}
{reuse_block}
Any other class or property you mint is held back and is written only if a later review keeps it. When a concept fits a planned or existing term, use that term.
"""

plan_reuse_block = """
Use these existing terms instead of minting:

{reuse_lines}
"""

improvement_instruction_template = """\n\n
# IMPROVEMENT INSTRUCTION

The current iteration of the ontology was not deemed accurate by Critic, who left the following suggestions for improvement:

{suggestions_instruction}
"""
