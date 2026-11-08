# Curating the catalog

Every document that renders an ontology can add classes and properties to a
shared catalog ontology. Some of those terms only ever serve the document that
added them. Curation finds them across all the documents processed so far and
decides what to do with each one.

## What curation reviews

Curation looks only at terms a document added. A term from a seed file or an
upload is *authored* and never changes. OntoCast tells the two apart from the
ontology's version history: each version records the document it came from,
and a term belongs to the version that first declared it.

A term added by a document is flagged for review when it:

- is used by the facts of one document only, or of none (counted when the
  facts store answers SPARQL, as Fuseki and the in-memory store do);
- has a name that adds one word to another term's (`hasOpticalCavity` and
  `hasOpticalMicrocavity`), or the same words;
- is one of three or more sibling classes whose labels differ by one word, a
  value set modelled as subclasses;
- names an authored term.

Only flagged terms go to the model, in batches. For each one it answers
`keep`, `merge_into` another term, `generalise_to` a broader parent,
`demote_to_property_value` (the class becomes a SKOS concept) or `drop`.
OntoCast turns the answers into one change of the ontology without a model,
and ignores an answer about an authored term or a term that was not flagged.

## Run it

```bash
ontocast curate --ontology my_ontology          # report only
ontocast curate --ontology my_ontology --apply  # write a new version
```

Without `--ontology`, every catalog ontology is curated. `--json report.json`
writes the full report: each added term, the document that added it, the
documents that use it, its flags and the decided action. `--no-facts-usage`
skips the usage counts. A dry run still makes the model calls; only the write
is skipped.

From Python:

```python
from ontocast import curate_catalog

reports = await curate_catalog(tools, ontology="my_ontology", apply=False)
```

`--apply` writes the change as the ontology's next version, derived from a
`urn:ontocast:curation:<time>` source rather than a document, so later
curation runs still attribute each term to the document that added it.

The calls are charged to the `ontology_curation` phase in the budget; give it
its own model with `LLM_MODEL_BY_PHASE`.
