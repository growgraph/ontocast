# Facts Validation and SHACL

OntoCast treats the renderer LLM as a transcriber, not a guarantor. Everything
it emits passes through deterministic checks, and everything a machine can fix
is fixed by a machine — without asking the model again.

This page describes the three validation layers, which of them cost a provider
call, how SHACL fits in, and how to read the result.

## The three layers

| # | Layer | Where | LLM calls |
|---|-------|-------|-----------|
| 1 | **Machine repair, at parse time** | `agent/render_facts.py::_normalize_and_repair_graph`, per unit, on every rendered graph | **none** |
| 2 | **Finding-driven repair render** | `stategraph/atomic.py::_run_finding_driven_repair`, per unit | **up to `FACTS_LLM_REPAIR_VISITS`** |
| 3 | **Post-merge gate** | `VALIDATE_FACTS` node, once per document | **none** |

### How many LLM calls a facts unit really costs

At the default `MAX_VISITS=1` the critic never runs, but extraction is **not**
one call per unit:

```
render_facts                      1 provider call
  ↓  critic skipped (MAX_VISITS reached)
finding-driven repair render      1 more, if mandatory findings remain
                                  (up to FACTS_LLM_REPAIR_VISITS, default 1)
```

The *trigger* is deterministic — quarantined literals, unknown terms, alias
leftovers — but the *fix* is bought from the model. Set
`FACTS_LLM_REPAIR_VISITS=0` to pin extraction at exactly one call per unit and
leave the residue to layers 1 and 3.

The ontology loop has no repair stage, so at `MAX_VISITS=1` it is genuinely one
call per unit.

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

| Finding kind | Severity | What acts on it |
|--------------|----------|-----------------|
| `FUNCTIONAL_VIOLATION` | error | un-merge repair |
| `SUSPECT_MULTI_VALUE` | error (configurable) | un-merge repair |
| `DEGENERATE_COREFERENCE` | error | un-merge repair |
| `SHACL` | error / warning | **SHACL autofix**; reported, never un-merged |
| `NON_CATALOG_VOCABULARY` | warning | reported (marks a retrieval miss) |
| `DANGLING_REFERENCE` | warning | reported |

The first three are *merge signatures*: their shape is "two things that are not
the same got one IRI", which un-merging can repair (`FACTS_MERGE_REPAIR_PASSES`
passes of cluster vetoes plus re-aggregation, kept only if the merge-signature
error count strictly drops).

SHACL findings are excluded from that loop by design. A missing required
property or a datatype mismatch says a node is under-specified, not that two
entities were confused — un-merging cannot fix it, and letting it into the veto
set dissolved legitimate clusters.

## SHACL

### Where shapes come from

Two sources, merged:

- **`FACTS_SHAPES_DIR`** — every `.ttl` under the directory (recursive).
- **The ontology context itself**, when it already carries `sh:NodeShape`
  declarations inline. This is the zero-config path for catalogs that ship
  shapes next to their schema.

Requires the extra: `uv sync --extra shacl`. Configuring shapes without it, or
pointing at a missing or empty directory, logs a **warning** — a skipped run is
never reported as a clean one.

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
data. On the three-document matsci pilot this accounted for **128 of 360**
reported violations.

**RDFS inference is on by default.** SHACL resolves class targets through
`rdfs:subClassOf` itself, but property paths carry no entailment: a shape naming
`obs:hasResult` does not see the `life:hasStorageResult` the renderer emitted,
and reports a statement that is present as missing. Same pilot: 268 violations
at `inference=none` against 232 with RDFS.

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

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTS_SHAPES_DIR` | — | Directory of SHACL shape files; inline `sh:NodeShape` in the ontology context is picked up automatically |
| `FACTS_SHACL_INFERENCE` | `rdfs` | `none`, `rdfs` or `owlrl` pre-inference |
| `FACTS_SHACL_ADVANCED` | `true` | Enable SHACL Advanced Features |
| `FACTS_SHACL_MAX_TRIPLES` | `200000` | Skip validation above this graph size (`0` disables the guard) |
| `FACTS_SHACL_AUTOFIX` | `prune` | `off`, `rewrite`, or `prune` |
| `FACTS_SHACL_AUTOFIX_PASSES` | `1` | Bounded validate → repair → revalidate rounds |
| `FACTS_CODE_PREDICATES` | `qudt:ucumCode`, `qudt:symbol`, `skos:notation` | Predicates whose literals are machine-resolvable codes |
| `FACTS_LLM_REPAIR_VISITS` | `1` | Finding-driven repair renders per unit — **each one is a provider call** |
| `FACTS_MERGE_REPAIR_PASSES` | `1` | Un-merge passes at the gate |
| `FACTS_SUSPECT_MULTI_VALUE_SEVERITY` | `error` | Severity of `SUSPECT_MULTI_VALUE` findings |
| `FACTS_FUNCTIONAL_MIN_SINGLE_SUPPORT` | `3` | Subjects needed before a predicate counts as empirically functional |

See [Configuration System](configuration.md) for the full surface and
[Entity Disambiguation](aggregation.md) for the merge stage this gate sits behind.
