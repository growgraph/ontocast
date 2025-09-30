ontology_instruction = """
```ttl
{ontology_str}
```
"""


template_prompt = """
Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.

# Instructions

- The facts (entities that are more concrete than the ones defined in ontologies) should be defined in custom namespace <{current_doc_namespace}> using the prefix `cd:` ( e.g. `@prefix cd: {current_doc_namespace} .` )
- Use the provided domain ontology <{ontology_namespace}> (provided below) together with standard ontologies (RDFS, OWL, schema.org, etc.) to identify or infer entities, classes, types, and relationships
- When referring to the domain ontology, use the namespace <{ontology_namespace}> with the prefix `{ontology_prefix}:`
- All entities in the <{current_doc_namespace}> namespace (facts) must be linked to entities from either domain ontology <{ontology_namespace}> or basic ontologies (RDFS, OWL etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person, schema:Organization, etc
- Define all prefixes for all namespaces used in the ontology, etc rdf, rdfs, owl, schema, etc
- Prefer ontology IRIs: If a term (class/property/individual) appears in the provided domain ontology or any standard ontology, use that ontology IRI — do not create a `cd:` IRI with the same local name
- Enforce typing: Every `cd:` instance must have an rdf:type triple that points to an ontology class (e.g. cd:case-12345 rdf:type fca:LegalCase .)
- Maximize atomicity: decompose complex facts into simple subject-predicate-object statements
- Literals Handling:
    - Keep literals atomic - break down complex values into separate triples
    - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float, xsd:date, xsd:dateTime
    - Dates: Use ISO 8601 format (e.g., "2024-01-15"^^xsd:date)
    - Numbers: Always use typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal)
    - Currencies: Include currency codes (e.g., "1000"^^xsd:decimal with schema:priceCurrency "USD")
- To extract data from tables, use CSV on the Web (CSVW) to describe tables
- No comments in Turtle: Output must contain only @prefix declarations and triples. Do not include comments (lines starting with #)

# Domain Ontology

{ontology_instruction}

# Text for processing:

```
{text}
```

{failure_instruction}

{format_instructions}
"""
