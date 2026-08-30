# Validation: Facts, Ontology Deltas and SHACL

OntoCast treats the renderer LLM as a transcriber, not a guarantor. Everything
it emits passes through deterministic checks, and everything a machine can fix
is fixed by a machine — without asking the model again.

This page describes the three facts validation layers, which of them cost a
provider call, how SHACL fits in, and how to read the result. The ontology
loop's own deterministic lane — which validates a unit's *delta*, not its
graph, and does not yet gate — is
[at the end](#ontology-delta-validation-shadow-mode).

## The three layers

| # | Layer | Where | LLM calls |
|---|-------|-------|-----------|
| 1 | **Machine repair, at parse time** | `agent/render_facts.py::_normalize_and_repair_graph`, per unit, on every rendered graph | **none** |
| 2 | **Finding-driven repair render** | `stategraph/atomic.py::_run_finding_driven_repair`, per unit — fed by deterministic findings *and* the critic's blocking fixes | **up to `FACTS_LLM_REPAIR_VISITS`** |
| 3 | **Post-merge gate** | `VALIDATE_FACTS` node, once per document | **none** |

### How many LLM calls a facts unit really costs

At the default `MAX_VISITS=1` extraction is **not** one call per unit:

```
render_facts                      1 provider call
criticise_facts                   1 more, unless FACTS_LLM_REPAIR_VISITS=0
  ↓  mechanical fixes compiled and applied here, no LLM call
finding-driven repair render      1 more, if mandatory findings or
                                  unresolved critic fixes remain
                                  (up to FACTS_LLM_REPAIR_VISITS, default 1)
```

`MAX_VISITS` bounds *renders*, not the critic. A verdict no longer needs a
spare render slot to land in: fixes that quote triples the graph actually holds
are compiled into a patch and applied directly, and only the rest are handed to
a repair render. The critic is skipped only at `FACTS_LLM_REPAIR_VISITS=0`,
where there is genuinely nowhere to put its output.

The *trigger* is deterministic — quarantined literals, unknown terms, alias
leftovers — but the *fix* is bought from the model. Set
`FACTS_LLM_REPAIR_VISITS=0` to pin extraction at exactly one call per unit and
leave the residue to layers 1 and 3.

The ontology loop has no repair stage, so at `MAX_VISITS=1` it is genuinely one
call per unit. It does run its own deterministic validator — see
[Ontology delta validation](#ontology-delta-validation-shadow-mode) — but in
shadow mode: findings are recorded and shown to the critic, and cost nothing.

Above `MAX_VISITS=1` the facts critic runs, and a **rejection costs one repair
render, not another full render**. So the ceiling is:

```
render_facts                      1 provider call
  ↓
criticise_facts                   1 more (MAX_VISITS > 1 only)
  ↓  rejected -> its blocking fixes become findings
finding-driven repair render      1 more, up to FACTS_LLM_REPAIR_VISITS
```

That total does not grow with `MAX_VISITS`; the bound now governs render
*failure* retries only.

### What makes a render acceptable

Acceptance is `material_defects()` over evidence that can be pointed at:

| Signal | Blocks? |
|---|---|
| Deterministic finding with `mandatory=True` | always |
| Critic `TripleFix` at or above `FACTS_ACCEPT_BLOCKING_SEVERITY` (default `critical`) | yes |
| Critic `TripleFix` with `action=REMOVE` | **never**, whatever its severity — the repair contract forbids resolving a finding by deleting the statement |
| Advisory findings (`numeric_coverage`) | no |
| The critic's `score` / `success` | **no** — recorded as telemetry only |

The critic's score gated this until it was measured. A model asked to propose
improvements proposes them, so a `> 90` threshold it is never shown rejected
nearly every render, while the deterministic findings the loop had already
computed played no part at all. Set
`FACTS_ACCEPT_BLOCKING_SEVERITY=never` to let deterministic findings gate alone.
Per-document critic telemetry — call count, accept count, score histogram, fix
severity histogram — is written to the run manifest under `critic`.

## Layer 1 — machine repair at parse time

Applied to every rendered graph before it is accepted. Each repair is recorded
as a `GraphRepairRecord` and reported per unit in `facts_repairs`, so a consumer
can always tell machine-altered triples from what the model asserted.

| Repair | What it does |
|--------|--------------|
| Numeric literal retyping | `qudt:numericValue 230` → `"230"^^xsd:decimal` when the schema declares a numeric range |
| `rdf:type` literal coercion | A type emitted as a string becomes an IRI when it resolves unambiguously |
| Near-miss predicate rewrite | `qudt:value` → `qudt:numericValue` when exactly one catalog term is a near match (`FACTS_PROPERTY_ALIAS_MIN_RATIO`) |
| **Code resolution** | A node carrying `qudt:ucumCode "d"` but no unit link gains `qudt:unit unit:DAY` when exactly one catalog individual declares that code (`FACTS_CODE_PREDICATES`) |
| Degenerate-bound promotion | Equal lower/upper bounds collapse to a single scalar on the configured numeric-value property — active only when the quantity fallback vocabulary names `numeric_value`, `lower_bound` **and** `upper_bound` roles |

### Which terms count as unknown

`UNKNOWN_TERM` findings drive mandatory repair renders, so the closure rule
behind them matters. A namespace is *closed* — members the catalog does not
list get flagged — only when the catalog **declares** terms in it
(subject-position statements). A namespace the catalog merely *references*
(`qudt:QuantityValue` in a `rdfs:subClassOf`, `qudt:unit` in an
`owl:onProperty`) is an external vocabulary the catalog borrows from and stays
open: the catalog is not an authority on its membership.

Three exemptions apply even inside closed namespaces:
`FACTS_ADDITIONAL_STANDARD_NAMESPACES`, the quantity fallback vocabulary
(`FACTS_QUANTITY_FALLBACK_VOCABULARY` — the validator must never order the
renderer to remove the vocabulary the prompt itself recommends), and
`FACTS_CODE_PREDICATES`.

Two guardrails on the repair prompt: alias candidates are **role-filtered**
(a predicate never gets a known class suggested, and vice versa), and every
mandatory finding must be resolved by *rewriting in place* — a repair response
that only deletes statements is flagged as data destruction, not a fix. The
gate additionally cross-checks the SHACL shapes against these rules at load
time and logs an error for any property the shapes require but the term
validator would flag — data cannot satisfy both sides.

A companion mandatory finding, `LABEL_ONLY_NUMBER`, fires when a node carries
the fallback vocabulary's unit property but no numeric literal on any
property while its label holds a number as prose — a measurement that is
invisible to every query.

Code resolution is schema-driven, not vocabulary-specific: the connecting
property is whichever object property the ontology context declares with a range
the resolved individual is typed as and a domain the subject satisfies. Where
the schema declares no range — common in vendored vocabulary projections — it
falls back to how the graph already links that kind of subject to that kind of
individual, and only when the evidence is unambiguous.

Ambiguity is never resolved by guessing: two catalog terms claiming one code
means no repair and a reported finding.

## Layer 3 — the post-merge gate

After aggregation, `VALIDATE_FACTS` checks invariants over the merged graph.
Before validating, duplicate literals that differ only in language tag or
datatype on one (subject, predicate) — `"X"@en` alongside `"X"^^xsd:string`
alongside `"X"` — are collapsed to one surviving form
(`FACTS_LITERAL_VARIANT_DEDUPE`, default on; the language-tagged form wins,
then the plain form). Each removal is a `literal_variant_pruned` repair
record, and reified provenance moves to the surviving triple.

| Finding kind | Severity | What acts on it |
|--------------|----------|-----------------|
| `FUNCTIONAL_VIOLATION` | error | un-merge repair |
| `SUSPECT_MULTI_VALUE` | error (configurable) | un-merge repair |
| `DEGENERATE_COREFERENCE` | error | un-merge repair |
| `SHACL` | error / warning | **SHACL autofix**; reported, never un-merged |
| `NON_CATALOG_VOCABULARY` | warning | reported (marks a retrieval miss) |
| `DANGLING_REFERENCE` | warning | reported |
| `MIXED_OBJECT_KINDS` | warning | reported (predicate used with both IRI and literal objects) |

`SUSPECT_MULTI_VALUE` has three branches: ≥ 2 distinct canonical **numeric**
values on one (subject, predicate); ≥ 2 mutually irreconcilable short
**string** values (name variants like "Mr Beer" / "Mr Karlheinz Beer" are
alias-compatible and pass; "Mrs E. Palm" / "Mrs W. Thomassen" on one node is
the signature of distinct people collapsed together) on a predicate that is
string-single-valued for a dominant majority of subjects; and ≥ 2 **IRI**
objects on a dominantly single-valued predicate.

The first three kinds are *merge signatures*: their shape is "two things that
are not the same got one IRI", which un-merging can repair
(`FACTS_MERGE_REPAIR_PASSES` passes of cluster vetoes plus re-aggregation,
kept only if the merge-signature error count strictly drops).

SHACL findings are excluded from that loop by design. A missing required
property or a datatype mismatch says a node is under-specified, not that two
entities were confused — un-merging cannot fix it, and letting it into the veto
set dissolved legitimate clusters.

## SHACL

### Where shapes come from

Shapes are a **deployment artifact**, stored in the triple store in their own
tenancy partition (`{tenant}--{project}--shapes`) alongside facts and
ontologies. The gate reads that partition, so a containerised worker needs no
shapes directory, and a per-tenant catalog carries its own shapes.

Two sources, merged:

- **The shapes partition** — seeded once at startup from `FACTS_SHAPES_DIR`
  (every `.ttl` under the directory, recursively) and mutable thereafter over
  [`/shapes`](api.md#shapes). The directory is a read-only bootstrap fixture,
  the same contract `ONTOCAST_ONTOLOGY_DIRECTORY` has: nothing is written back
  to it, and a `DELETE /shapes/...` never touches your files.
- **The ontology context itself**, when it already carries `sh:NodeShape`
  declarations inline. This is the zero-config path for catalogs that ship
  shapes next to their schema.

Requires the extra: `uv sync --extra shacl`. Configuring shapes without it, or
pointing at a missing or empty directory, logs a **warning** — a skipped run is
never reported as a clean one.

#### Why shapes are not stored with the ontologies

Catalog discovery claims every named graph holding an `owl:Ontology` subject —
and a shapes document declares one (`<…/qqval-shapes> a owl:Ontology`). Stored
in the ontologies dataset, each shapes file would register as a catalog
ontology, be indexed as ontology atoms, and be offered to the renderer as
first-class schema. The separate partition makes that impossible structurally
rather than by a filter every read path has to remember.

A shapes document is addressed by the ontology IRI it declares, so re-uploading
it replaces it. One with no header is named after its seed path
(`urn:shapes:<relative/path.ttl>`) or, when uploaded, its filename — stable
across edits, so re-seeding replaces the document instead of accumulating
stale copies beside it.

#### Flushing

`POST /flush` **retains** the shapes partition by default. Facts and ontologies
come back from a rerun; shapes are the validation contract, and dropping them
disarms the gate without an error — subsequent runs report
`shacl_evaluated: null` rather than failing. Pass `?include_shapes=true` to drop
them too.

### How the validation is set up

```bash
FACTS_SHAPES_DIR=/path/to/shapes
FACTS_SHACL_INFERENCE=rdfs      # none | rdfs | owlrl
FACTS_SHACL_ADVANCED=true       # SHACL Advanced Features (sh:sparql, …)
FACTS_SHACL_MAX_TRIPLES=200000  # skip with a warning above this size
```

Two details matter more than they look:

**The ontology context is mixed into the data graph.** A facts graph states that
a value uses `unit:DAY`; that this individual *is* a `qudt:Unit` is stated only
in the catalog. Validating the facts alone fails every `sh:class` constraint
pointing at a catalog term — violations that describe the absent schema, not the
data. On a catalog-heavy document this is routinely a large fraction of all
reported violations.

**RDFS inference is on by default.** SHACL resolves class targets through
`rdfs:subClassOf` itself, but property paths carry no entailment: a shape naming
`obs:hasResult` does not see the `life:hasStorageResult` the renderer emitted,
and reports a statement that is present as missing. Turning inference off
therefore raises the violation count rather than lowering it.

### LLM-free autofix

`FACTS_SHACL_AUTOFIX` repairs violations in code, in a bounded
validate → repair → revalidate loop (`FACTS_SHACL_AUTOFIX_PASSES`). A pass is
kept only if the violation count strictly drops; otherwise it is reverted.

| Mode | Constraint | Machine action | Guard |
|------|-----------|----------------|-------|
| `rewrite`, `prune` | `sh:datatype` | Retype the literal to the declared datatype | Must parse as that datatype |
| `rewrite`, `prune` | `sh:class`, `sh:nodeKind` | Replace a string literal with the catalog IRI declaring it | Exact, case-sensitive, **unique** surface form |
| `prune` (default) | `sh:minCount` | Drop the focus node and its incoming edge | Node asserts nothing beyond `rdf:type`/`rdfs:label`, and at most one subject references it |

`off` reports without repairing.

**Reported, never repaired:** `sh:maxCount` (owned by the functional-violation
and un-merge machinery), `sh:not`, `sh:qualifiedValueShape`, and SPARQL
constraints.

The contract, in one line: **a repair either rewrites a term the catalog already
declares, or removes a node that asserts nothing.** Nothing is invented. A value
node carrying a real number but missing a required qualifier is neither filled
in nor deleted — it stays a reported finding, because filling it in would be
fabrication and deleting it would be data loss.

**Provenance survives a repair.** Every asserted triple is described by an
RDF 1.2 reifier (`_:r rdf:reifies <<( s p o )>>`), and no subject/object pattern
matches a term sitting *inside* a triple term — so a repair that rewrote a
statement left its provenance describing the pre-repair version. The two
rewriting repairs now **retarget** the reifier onto the replacement, keeping its
`prov:wasDerivedFrom` arcs intact; the prune repair **deletes** it, because the
statement it described is gone. A node both retyped and pruned in one pass is
swept: the prune wins. Both happen only after a pass is accepted, so a reverted
pass leaves provenance untouched.

Pruning is scoped to fact namespaces: ontology entities are never rewritten by
the gate.

### Reading the result

`POST /process` and `POST /process_unit` return, under `metadata` (the
single-unit route runs the same gate minus the un-merge repair, which has no
meaning for one unit):

```jsonc
{
  "facts_conformance": {
    "shacl_evaluated": true,
    "conforms": false,
    "findings": 74,
    "errors": 74,
    "warnings": 0,
    "by_kind": {"shacl": 74},
    "shacl_violations": 74,
    "shacl_by_constraint": {
      "MinCountConstraintComponent": 36,
      "SPARQLConstraintComponent": 35,
      "MaxCountConstraintComponent": 3
    },
    "shacl_by_shape": {"…#QualifiedQuantityValueShape": 36},
    "repairs_applied": {"shacl_prune": 2, "code_resolved": 2}
  },
  "facts_validation_findings": [ /* one entry per residual finding */ ],
  "facts_gate_repairs":        [ /* what the gate changed, and why */ ]
}
```

`conforms` is `null` when SHACL did not run — "no SHACL findings" reads
identically for *conforms* and *never checked*, so the two are kept apart
explicitly.

Grouping by constraint component is what makes a residue diagnosable: 36
`MinCountConstraintComponent` violations on one shape are one modelling gap, not
36 problems to triage.

Batch runs (`ontocast process --output-dir …`) write the same payload beside the
facts Turtle as `<name>.facts.validation.json`.

## Ontology delta validation (shadow mode)

The ontology loop has the same deterministic lane the facts loop gates on,
with one structural difference: it validates the unit's net **insert/delete
delta** against the prompt snapshot, never the whole working graph. The
working graph is `snapshot + delta`, so validating it would test the shared
catalog context against itself and attribute every pre-existing third-party
defect to this unit. Two facts rules are deliberately absent: `UNKNOWN_TERM`
is inverted here (minting new terms in a writable namespace is the ontology
renderer's job), and connectivity is left to the document-level
`STRUCTURAL_CHECK` node (a per-unit delta connects to the snapshot, not to
itself).

Checks (`tool/ontology_validation/unit_findings.py`), all mandatory unless
noted:

| Finding | What it catches |
|---|---|
| `foreign_namespace` | Term minted under a namespace no context ontology declares terms under — the catalog apply step silently drops these as unattributable, so this predicts triple loss. Also fires on `example.org` placeholders and ontology terms minted in facts namespaces. Skipped (except placeholders) on the fresh-create path, where there is no namespace authority |
| `degenerate_restriction` | `owl:Restriction` blank node with fewer than 2 meaningful predicates — it constrains nothing |
| `missing_label` | Newly declared class/property without `rdfs:label`/`skos:prefLabel` |
| `subclass_cycle` | An inserted `rdfs:subClassOf` edge closing a cycle through snapshot + delta |
| `role_confusion` | A catalog class used as a predicate, or a catalog property used in class position |
| `cardinality_contradiction` | Functional / max-cardinality-1 declaration contradicted by a min-cardinality ≥ 2, where this unit participates in the conflict |
| `foreign_delete` | Deleting catalog content whose subject the unit does not redeclare — ontology deletes propagate onto shared, versioned catalog terminals cross-document |
| `label_collision` (advisory) | New term whose label duplicates an existing catalog surface form — likely a re-mint of a concept that should be reused |

**Shadow mode means the gate is unchanged.** The ontology critic still accepts
on `success or score > 90` — unlike the facts score gate, that threshold is
backed by a scoring rubric in the critic's own prompt (`> 90` is its top band,
"Excellent — minor refinements only"), so what it demands is perfection.
Whether that is the right operating point is an empirical question that is not
yet answered: the ontology critic does not run under `render_mode: facts`, so
no recorded data covers it. The findings are
collected before every critic call (and at loop exit, so the residual exists
even at `MAX_VISITS=1` where the critic is skipped), injected into the critic
prompt as MANDATORY items, and recorded per attempt: score, severity mix,
findings counts, the delta size, and how many proposed fixes target
snapshot-owned terms the delta never touched. Once a sampling run yields the
distribution, the gate gets the same `material_defects()` treatment the facts
loop received.

Per-document telemetry lands in the run manifest under `ontology_critic`
(sibling of `critic`) and in `retrieval_metrics` as
`ontology_findings_residual` / `ontology_mandatory_residual` /
`ontology_critic_calls` / `ontology_critic_accepted`.

### Reduce-time policies: the terminal is the authority

Under `ONTOLOGY_CONTEXT_MODE=selected_vector_search_ontology` the per-unit
snapshot is a **retrieved subset** of the catalog, so judgments that require
knowing what the catalog contains cannot be made inside the unit. Three
policies therefore run at the reduce step, where the full catalog terminals
are already in hand:

| Policy | Behavior | Metric |
|---|---|---|
| **Minted-duplicate reconciliation** (`ONTOLOGY_RECONCILE_MINTED_TERMS`, default `detect`) | A newly minted term whose label/prefLabel/notation exactly matches one full-terminal term of compatible role is a re-mint of a concept retrieval failed to surface — the per-unit label-collision check indexes the snapshot, which is exactly where the duplicate is not. `detect` records the pairs; `rewrite` substitutes the minted IRI (flip only after `detect` has shown the matches are true duplicates); `off` disables | `minted_duplicates`, `minted_duplicate_pairs`, `minted_duplicates_rewritten` |
| **Delete policy** (vector mode only, no knob) | A merged delete whose subject the merged inserts do not redeclare was judged on partial evidence and would propagate onto shared catalog terminals; it is dropped. Single-ontology modes are untouched — there the model saw the whole graph | `deletes_dropped_unredeclared` |
| **Fresh-path reconciliation** | N units minting fresh ontologies under the same IRI union-merge into one root version (previously: silent last-wins, other units' content dropped); overlap across distinct fresh IRIs is counted, not merged | `fresh_ontologies_merged`, `fresh_minted_duplicates` |

Divergence between what a unit saw and what the catalog holds is also counted:
`apply_deletes_no_match` is the number of delete triples absent from the
terminal at apply time — a stale vector index is the usual cause. Alongside it,
`unattributed_insert_triples` / `unattributed_delete_triples` count delta
triples the namespace partition could not attribute to any catalog terminal,
which is the drop the `foreign_namespace` unit finding predicts.

!!! warning "These counters do not leave the process yet"
    All of the above accumulate in `AgentState.ontology_reduce_metrics`, which
    — unlike `retrieval_metrics` — is carried into neither the `/process`
    response metadata nor the run manifest. Today they are reachable only from
    the returned state object (an embedded graph, or `run_unit_pipeline`).
    Individual events do log — a minted duplicate raises a `WARNING` naming
    both IRIs — but the counts themselves do not. Read the state object until
    the plumbing lands.

The render and critic prompts declare the partial view explicitly in vector
mode (a PARTIAL CONTEXT notice), so the model is told that an absent term may
simply be unretrieved rather than missing.

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTS_SHAPES_DIR` | — | **Seed** directory of SHACL shape files (recursive), materialized into the shapes partition at startup; inline `sh:NodeShape` in the ontology context is picked up automatically |
| `FUSEKI_SHAPES_DATASET` | derived from tenant/project | Fuseki dataset backing the shapes partition |
| `FACTS_SHACL_INFERENCE` | `rdfs` | `none`, `rdfs` or `owlrl` pre-inference |
| `FACTS_SHACL_ADVANCED` | `true` | Enable SHACL Advanced Features |
| `FACTS_SHACL_MAX_TRIPLES` | `200000` | Skip validation above this graph size (`0` disables the guard) |
| `FACTS_SHACL_AUTOFIX` | `prune` | `off`, `rewrite`, or `prune` |
| `FACTS_SHACL_AUTOFIX_PASSES` | `1` | Bounded validate → repair → revalidate rounds |
| `FACTS_CODE_PREDICATES` | `qudt:ucumCode`, `qudt:symbol`, `skos:notation` | Predicates whose literals are machine-resolvable codes |
| `FACTS_LLM_REPAIR_VISITS` | `1` | Finding-driven repair renders per unit — **each one is a provider call** |
| `FACTS_MERGE_REPAIR_PASSES` | `1` | Un-merge passes at the gate |
| `FACTS_SUSPECT_MULTI_VALUE_SEVERITY` | `error` | Severity of `SUSPECT_MULTI_VALUE` findings |
| `FACTS_SUSPECT_MULTI_VALUE_REQUIRE_CROSS_UNIT` | `false` | Report an IRI-branch `SUSPECT_MULTI_VALUE` as an error only when its objects came from different units |
| `FACTS_NUMERIC_IDENTIFIER_GUARD` | `false` | Keep identifier digit groups out of the numeric-coverage inventory |
| `FACTS_CONTEXT_FROM_UNITS` | `false` | In facts-only runs, seed the merge/validate ontology context from the snapshots the units resolved |
| `FACTS_FUNCTIONAL_MIN_SINGLE_SUPPORT` | `3` | Subjects needed before a predicate counts as empirically functional |
| `FACTS_ACCEPT_BLOCKING_SEVERITY` | `critical` | Which critic-proposed fix severities block a unit from leaving the loop; `never` lets deterministic findings gate alone. A `REMOVE` fix never blocks at any setting |
| `FACTS_LITERAL_VARIANT_DEDUPE` | `true` | Collapse duplicate literals differing only in language tag or datatype before the gate validates |
| `ONTOLOGY_RECONCILE_MINTED_TERMS` | `detect` | Reduce-time minted-duplicate scan: `off`, `detect` (count only), `rewrite` (substitute the catalog IRI) |

The last three default to off because each changes what the pipeline extracts or
reports, and the right setting depends on the corpus:

- **`FACTS_SUSPECT_MULTI_VALUE_REQUIRE_CROSS_UNIT`.** The IRI branch of
  `SUSPECT_MULTI_VALUE` flags any subject carrying two objects on a predicate
  that is single-valued elsewhere in the graph, and error findings drive the
  un-merge repair. Frequency alone is weak evidence: a genuinely multi-valued
  statement is rare by construction, so a correct one — two agents named in one
  sentence — looks exactly like a bad merge and gets repaired away. Provenance
  separates them, because only objects arriving from *different* units could
  have been brought together by an identity decision. With this set, a pair
  asserted within a single unit is reported as a warning and never vetoes a
  cluster. The numeric and string branches are unaffected: two distinct
  quantities on one node are a defect whatever their provenance.
- **`FACTS_NUMERIC_IDENTIFIER_GUARD`.** The numeric-coverage finding lists
  numbers present in the source text but absent from the graph, and the repair
  render acts on it. Digit groups sitting against an identifier separator
  (`600/92`, `10.1234/example`) are parts of one identifier, not quantities;
  offering them invites the repair to structure a file number into numeric
  properties, which the post-merge multi-value check then flags. A digit group
  standing alone as its own token is deliberately *not* covered — nothing around
  it distinguishes a file-number component from a small quantity — and a value
  with its unit (`8.5 nm`) or a range (`10-15 meV`) is untouched.
- **`FACTS_CONTEXT_FROM_UNITS`.** With `RENDER_MODE=facts` no ontology stage
  runs, so there are no reduced artifacts to merge and the document-level
  ontology context is empty — even though every unit resolved and rendered
  against a real one. Both consumers see that: the aggregator loses the type and
  functionality declarations its guards read, and the gate skips every check
  that needs a vocabulary. Watch `validated_without_ontology_context` and
  `ontology_snapshot_triples` in the retrieval metrics to see which side you are
  on.

See [Configuration System](configuration.md) for the full surface and
[Entity Disambiguation](aggregation.md) for the merge stage this gate sits behind.
