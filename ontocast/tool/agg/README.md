## Naming & Normalization Conventions

We follow standard RDF / Semantic Web naming conventions:

- Classes (entities / types) use PascalCase

```ttl
ex:Case
ex:JudicialDecision
```


- Properties (predicates) use lower camelCase

```ttl 
ex:hasDecision
ex:datePublished
```


- Individuals with natural names use PascalCase

```ttl
ex:FrenchCourtOfCassation
```

Individuals with structured or external identifiers preserve their structure
(underscores and digits are allowed and encouraged)

```shell
ex:Case_2023_456
ex:Decision_2021_09_15
```


## Notes

- Underscores are avoided in ontology terms (classes, properties).
- Underscores are acceptable for instances derived from external IDs.
- Prefer stable, readable IRIs; store human-facing identifiers explicitly when needed:

```ttl
ex:Case_2023_456 ex:caseNumber "2023/456" .
```
