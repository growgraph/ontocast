# Validation and SHACL

A language model makes mistakes a machine can catch: a number stored as text, a
property that does not exist, two people merged into one node. This page shows
what OntoCast checks, how to add your own rules as SHACL shapes, and how to read
the result of a run.

## What OntoCast checks, and when

| Stage | When | What happens | LLM calls |
|---|---|---|---|
| Repair | As each part of the document (content unit) is extracted | Mechanical slips are fixed in place: a number typed as text, a property name one token off a real one, a unit code without its unit | None |
| Review | After each extraction | Checks list what is still wrong (unknown terms, values hidden in labels, numbers in the text missing from the graph); the critic reviews the facts against them and its fixes are applied as a patch | One per critic pass |
| Gate | Once per document, after the parts are merged | Merged facts are checked for nodes that hold two values where one is expected, and against your SHACL shapes; what can be fixed without a model is fixed | None |

Ontology changes get the same per-unit review, with checks suited to new terms
(missing labels, subclass cycles, terms minted outside the ontology's
namespace), plus a check for newly minted terms that duplicate one already in
the catalog. The ontology critic is off by default
([`ONTOLOGY_CRITIC_PASSES`](../reference/configuration/ontology-validation.md#ontology_critic_passes)).

Validation never withholds output. The gate repairs what it can and reports the
rest, so a run with findings still returns its facts. [Validation layers and the
critic loop](../internals/validation_layers.md) describes every check, the
critic budgets, and the rules that keep a review from destroying data.

## Using SHACL shapes

Shapes are your rules for the facts: which properties a node of a class must
have, which datatype a value takes, which class an object belongs to. OntoCast
validates the merged facts against them at the gate and shows them to the model
while it extracts. SHACL needs the `shacl` extra:

```bash
pip install "ontocast[shacl]"
```

### Where shapes come from

OntoCast keeps shapes in the triple store, in a shapes partition per tenant and
project, next to its facts and ontologies. The gate reads that partition. You
fill it in either of two ways:

- **At startup**, from a directory of Turtle files, searched recursively: set
  [`FACTS_SHAPES_DIR`](../reference/configuration/facts-validation.md#facts_shapes_dir)
  or pass `--shapes-dir` to `ontocast serve` or `ontocast process`. Each file is
  written to the partition on every startup, so an edited file takes effect on
  restart. OntoCast never writes to the directory.
- **On a running server**, through `/shapes`:

```bash
curl -X POST http://127.0.0.1:8999/shapes -F "file=@my-shapes.ttl"
curl http://127.0.0.1:8999/shapes
```

A shapes document that declares `<iri> a owl:Ontology` is stored under that IRI,
so uploading it again replaces it; one without a header is named after its file.
`DELETE /shapes/{graph_uri}` removes a document from the partition and leaves
your seed files alone, so a seeded document returns on the next restart. See the
[HTTP API](../reference/http_api.md#shapes).

An ontology that carries `sh:NodeShape` declarations inline is also used as a
source of shapes, with no setup.

If shapes are configured but cannot be used (the extra is missing, the
directory does not exist or holds no `.ttl` files), OntoCast logs a warning and
the run reports SHACL as not evaluated, never as passed.

### Why shapes are not stored with the ontologies

A shapes document usually declares itself an `owl:Ontology`, and OntoCast treats
every graph with that declaration in the ontologies store as part of the
catalog. Stored there, your shapes would be indexed as ontology terms and
offered to the model as schema to extract with. A partition of their own keeps
them out of the catalog.

`POST /flush` keeps the shapes partition: facts and ontologies come back when
you rerun, while dropping the shapes would silently switch the SHACL gate off.
Add `?include_shapes=true` to drop them too.

## How the validation is set up

The gate validates the merged facts together with the ontology context they were
extracted against. Facts say that a value has the unit `unit:DAY`; only the
ontology says that `unit:DAY` is a `qudt:Unit`. Without the ontology, every
`sh:class` constraint that points at a catalog term would fail.

[`FACTS_SHACL_INFERENCE`](../reference/configuration/facts-validation.md#facts_shacl_inference)
is `rdfs` by default. SHACL follows `rdfs:subClassOf` when it picks the nodes a
shape targets, but not `rdfs:subPropertyOf` along a property path: a shape that
requires `obs:hasResult` does not see a more specific subproperty the model
used, and reports the statement as missing. RDFS inference adds the general
statement, so turning it off raises the number of violations. Use `none` only
when your shapes name exactly the properties the facts use. `owlrl` applies OWL
RL reasoning as well.

Two more settings: [`FACTS_SHACL_ADVANCED`](../reference/configuration/facts-validation.md#facts_shacl_advanced)
enables SHACL-AF features such as `sh:sparql` constraints, and
[`FACTS_SHACL_MAX_TRIPLES`](../reference/configuration/facts-validation.md#facts_shacl_max_triples)
skips SHACL, with a warning, on a merged graph larger than the limit.

!!! warning "Facts-only runs need the ontology context too"
    With `RENDER_MODE=facts` no ontology stage runs, so OntoCast builds the
    context for the gate from the ontology slices the parts were extracted
    against ([`FACTS_CONTEXT_FROM_UNITS`](../reference/configuration/facts-validation.md#facts_context_from_units),
    on by default). If that context is empty, `sh:class` constraints fail for
    every node typed with a subclass of the class a shape names, and the gate
    skips the checks that need a vocabulary. The run then sets
    `validated_without_ontology_context` in `retrieval_metrics`; read its
    violation counts as an upper bound.

## Showing the shapes to the model

The model is held to rules it can read. With
[`FACTS_SHAPES_PROMPT_CONTRACT`](../reference/configuration/facts-validation.md#facts_shapes_prompt_contract)
at `auto` (the default), OntoCast turns the shapes into a conformance chapter
that the facts render and the critic both see. Each constraint contributes its
`sh:message`, or a generated line such as "path: at least 1, of type C" when it
has none; a SPARQL constraint without a message is left out with a warning, so
write `sh:message` for every constraint.

| Value | Which shapes each part of the document sees |
|---|---|
| `auto` (default) | All of them while they fit [`FACTS_SHAPES_PROMPT_MAX_LINES`](../reference/configuration/facts-validation.md#facts_shapes_prompt_max_lines); beyond that, as `context` |
| `full` | All of them, cut at the line limit |
| `context` | Only the shapes whose classes and properties appear in the part's ontology context |
| `off` | None |

Terms the shapes require are never reported as unknown terms, whichever shapes
a part sees.

## LLM-free autofix

[`FACTS_SHACL_AUTOFIX`](../reference/configuration/facts-validation.md#facts_shacl_autofix)
repairs violations at the gate in code. A repair either rewrites a value into a
term the catalog already declares, or removes a node that states nothing:

| Mode | Violation | Repair |
|---|---|---|
| `rewrite`, `prune` | `sh:datatype` | Retype the literal to the declared datatype, if it parses as that type |
| `rewrite`, `prune` | `sh:class`, `sh:nodeKind` | Replace a string with the catalog IRI whose label or code matches it exactly, if exactly one does |
| `prune` (default) | `sh:minCount` | Remove a node that states nothing beyond its type and label, with the one link to it |
| `off` | any | Report only |

A node that carries a real value but lacks a required property is reported,
never filled in or deleted. `sh:maxCount`, `sh:not`, qualified value shapes and
SPARQL constraints are reported only. A repair pass is kept only if it lowers
the number of violations, up to
[`FACTS_SHACL_AUTOFIX_PASSES`](../reference/configuration/facts-validation.md#facts_shacl_autofix_passes)
passes. Provenance follows a rewritten statement and is removed with a pruned
one.

## Reading the result

`POST /process` and `POST /process_unit` report validation under `metadata`:

| Field | What it holds |
|---|---|
| `facts_conformance` | The summary: `conforms`, counts of errors and warnings, `by_kind`, SHACL violations grouped by constraint (`shacl_by_constraint`) and by shape (`shacl_by_shape`), and `repairs_applied` by kind |
| `facts_validation_findings` | One entry per finding still open after the repairs |
| `facts_gate_repairs` | What the gate changed, and why |
| `facts_repairs` | The repairs made while each part was extracted, keyed by part |

```json
"facts_conformance": {
  "shacl_evaluated": true,
  "conforms": false,
  "shacl_focus_nodes": 120,
  "errors": 9,
  "shacl_by_constraint": {"MinCountConstraintComponent": 7, "DatatypeConstraintComponent": 2},
  "repairs_applied": {"shacl_prune": 3}
}
```

`conforms` is `null` when SHACL did not run, and also when it ran but no node in
the facts matched any shape (`shacl_vacuous: true`): in both cases nothing was
checked. Group by constraint before you triage: many violations of one
constraint on one shape are usually one modeling gap.

`ontocast process` writes the same report next to the facts as
`<name>.facts.validation.json`, with the per-part repairs and the parts that
failed. [Reading run telemetry](telemetry.md) covers the counters in
`retrieval_metrics`.
