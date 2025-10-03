ontology_instruction = """
```ttl
{ontology_str}
```
"""


template_prompt = """
Generate semantic triples representing facts (not abstract entities) based on provided domain ontology.

# Instructions

1. The facts (entities that are more concrete than the ones defined in ontologies) should be defined in custom namespace <{current_doc_namespace}> using the prefix `cd:` ( e.g. `@prefix cd: {current_doc_namespace} .` )
2. Use the provided domain ontology <{ontology_namespace}> (below) and standard ontologies (RDFS, OWL, schema.org, etc.) to identify/infer entities, classes, types, and relationships
3. Thoroughly Extract and Link: Extract all possible text mentions that correspond to entities, classes, types, or relationships defined in the domain ontology <{ontology_namespace}>. When referring to the domain ontology, use the prefix `{ontology_prefix}:`
4. Enforce typing: all `cd:` entities (facts) must be linked (e.g. using rdf:type) to entities from either the DOMAIN ONTOLOGY <{ontology_namespace}> or basic ontologies (RDFS, OWL, etc), e.g. rdfs:Class, rdf:Property, schema:Person, schema:Organization, etc.
5. Define all prefixes for all namespaces used rdf, rdfs, owl, schema, etc
6. Prefer Ontology IRIs: If a term (class/property/individual) exists in the domain or any standard ontology, use its IRI, **do not** create a `cd:` IRI with the same local name.
7. Maximize atomicity: decompose complex facts and complex literals into simple subject-predicate-object statements
8. Literals Handling:
    - Use appropriate XSD datatypes: xsd:integer, xsd:decimal, xsd:float, xsd:date, xsd:dateTime
    - Dates: Use ISO 8601 format (e.g., "2024-01-15"^^xsd:date)
    - Numbers: Always use typed literals (e.g., "42"^^xsd:integer, "99.95"^^xsd:decimal)
    - Currencies: Include currency codes (e.g., "1000"^^xsd:decimal with schema:priceCurrency "USD")
9. To extract data from tables, use CSV on the Web (CSVW) to describe tables
10. No comments in Turtle: Output must contain only @prefix declarations and triples. Do not include comments (lines starting with #)

# Domain Ontology

{ontology_instruction}

# Text for processing:

```
{text}
```

{failure_instruction}

{format_instructions}
"""
