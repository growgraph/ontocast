# Checking facts

`check_facts` inspects a facts graph you already have against a catalog
ontology, without a model call and without changing the graph. It reports
what would not survive as clean facts: terms the catalog does not define,
values of the wrong kind, instances nothing says anything about.

```python
from ontocast import check_facts
from ontocast.onto.rdfgraph import RDFGraph

facts = RDFGraph.from_file("facts.ttl")
catalog = RDFGraph.from_file("ontology.ttl")
report = check_facts(facts, catalog, vocabulary="strict")
```

From the command line:

```bash
ontocast check facts.ttl --ontology-dir ontologies/ --shapes-dir shapes/
ontocast check part1.ttl part2.ttl --ontology-dir ontologies/ --vocabulary open --json
ontocast check facts.ttl --tenant T --project P     # catalog held by the configured store
```

The command needs the `server` extra (`pip install "ontocast[server]"`);
SHACL shapes are evaluated only with the `shacl` extra installed. Several
files are merged into one graph first. The text output opens with a summary
line (triples, instances, errors, warnings). Exit status is 0 without error
findings, 1 with any, and 2 when the check cannot run: a file cannot be
parsed or holds no triples, or the catalog cannot be loaded.

`--tenant/--project` read the catalog and shapes from the triple store, so
they need the same store settings as `ontocast serve`, and the LLM settings
it validates at start-up. The check itself makes no LLM call; under
`ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology`, start-up may still
index the catalog with the configured embedding model. Start-up neither wipes
the vector partition nor loads seed directories, but, as `ontocast serve`
does, it writes the catalog it loaded back to the store unchanged and may
reindex it or prune vector entries for ontologies no longer in it. The
tenant options cannot be combined with `--ontology-dir` or `--shapes-dir`.

## What is checked

| Finding | Meaning |
|---|---|
| `non_catalog_vocabulary` | a term the catalog does not define, whether in another namespace or missing from a namespace the catalog declares terms in |
| `unknown_term` | an IRI missing from a well-known vocabulary in any position (`unit:Million` gets `unit:MILLION` suggested, never applied), an individual of a value whose property's range is a closed class that neither the catalog nor a shipped list holds, or a term the catalog does not define in a placeholder namespace such as `example.org`; an error in both modes |
| `domain_violation` | a property used on a subject outside its declared domain |
| `role_outside_statement` | a statement's result, subject, property or context link held by a node that is not a statement |
| `empty_value` | a value node with no number, bound, unit or qualifier (warning) |
| `functional_violation` | a functional property with several values |
| `untyped_value` | a plain value that would be valid typed (warning) |
| `invalid_literal` | a value that cannot be read as its datatype, or a literal that is no value of its property's declared range: the catalog's, or one a well-known vocabulary publishes (`qudt:numericValue` takes any XSD number). A numeric range takes any XSD numeric form, so `"1e3"` fits `xsd:decimal`; language-tagged text fits no numeric, temporal or boolean range |
| `temporal_range_mismatch` | a valid typed literal of the wrong type on a temporal range; the calendar types (`xsd:date`, `dateTime`, `gYear`, `gYearMonth`) are one family, so a valid value of any of them fits a range declaring another |
| `shacl` | a SHACL violation, when shapes are given |
| `untyped_instance`, `isolated_instance`, `silent_instance`, `dangling_reference`, `twin_labels` | instance hygiene |
| `catalog_subject`, `schema_in_facts` | schema statements in the facts graph |

`vocabulary="strict"` (the default) makes `non_catalog_vocabulary` an error;
`"open"` makes it a warning.

## Instances

Only facts are checked. OntoCast's provenance (content-unit nodes and
statement reifiers) is left out, and the document node is not an instance.

Instances are decided per IRI: an IRI subject is an instance unless it is a
class or property of the catalog, a term declared in the checked graph, or
under a well-known namespace. The catalog's own individuals (`ex:Paris a
ex:City`) are instances, and keep the types the catalog gives them. Pass `instance_namespaces` to restrict instances to given
namespaces.

## Empty catalog

With no catalog the checks that need one are skipped and the report sets
`catalog_empty`; well-known-term checks and literal syntax still run.

## The report

`FactsCheckReport` holds `findings` (errors first, capped by `max_findings`),
`findings_total`, complete `counts` by kind and severity, `metrics`
(triples, instances, isolated, untyped, twin labels), `shacl_evaluated` and
`catalog_empty`. `ontocast check --json` prints it as JSON.
