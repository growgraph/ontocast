# How prompts are built

This page describes how OntoCast assembles the prompts of the facts and
ontology loops, and in particular the ontology chapter, which is the part your
settings change. Read it to predict what a setting does to prompt size, to the
provider's prefix cache, and to the LLM response cache. [Performance
tuning](../guides/performance.md) covers when to change each setting.

## Chapter order

The facts render prompt (`ontocast/prompt/render_facts.py`) is built from
chapters in this order:

1. the fixed preamble;
2. the conformance chapter: your SHACL shapes as rules
   (`ontocast/prompt/shapes_contract.py`), empty without shapes;
3. the ontology chapter;
4. the task, the facts guidelines, the user instruction and the text of the
   content unit;
5. the output instruction and format instructions.

The facts critic prompt (`ontocast/prompt/criticise_facts.py`) opens with the
same first three chapters, byte for byte, and a test pins the two openings to
each other. Everything specific to the call comes after the ontology chapter,
so the critic call of a unit can be served from the prefix its render call
wrote to the provider's cache.

The conformance chapter follows
[`FACTS_SHAPES_PROMPT_CONTRACT`](../reference/configuration/facts-validation.md#facts_shapes_prompt_contract):
`full` lists every shape up to
[`FACTS_SHAPES_PROMPT_MAX_LINES`](../reference/configuration/facts-validation.md#facts_shapes_prompt_max_lines),
`context` only the shapes whose targets appear in the unit's ontology context,
`auto` switches from `full` to `context` once the shapes exceed the line cap,
and `off` drops the chapter. Under `context` the chapter differs per unit;
[Validation](../guides/validation.md) explains how shapes are used.

## The ontology chapter

Every ontology chapter passes through
`GraphFormatProfile.format_ontology_chapter` (`ontocast/prompt/graph_format.py`),
which does three things in order:

1. bounds the text literals, when a [text cap](#text-caps) is set (facts
   render and critic chapters only);
2. condenses the graph toward
   [`ONTOLOGY_CONTEXT_MAX_TRIPLES`](#condensing-to-ontology_context_max_triples);
3. renders the result in the chapter format.

The facts loop builds its chapter through `OntologySnapshot.prompt_chapter`
(`ontocast/onto/ontology_snapshot.py`), which memoizes it on the snapshot. The
key is the graph's identity and size, the chapter syntax, the triple budget and
the text caps. A snapshot shared across the fan-out therefore builds its
chapter once, and the render and critic calls of a unit read identical bytes.

### Formats

[`ONTOLOGY_CHAPTER_FORMAT`](../reference/configuration/pipeline.md#ontology_chapter_format)
applies to the facts render and critic prompts only. The completion pass
builds its own, narrower term sheet (`ontocast/prompt/complete_facts.py`)
whatever this setting says.

| Format | What the chapter holds |
|---|---|
| `inherit` | The graph serialized in `LLM_GRAPH_FORMAT`: compact JSON-LD in a `json` block, or canonical Turtle in a `ttl` block, followed by a term index for ontologies with opaque IRIs (labels, and property domains and ranges) |
| `turtle` | The same, always in canonical Turtle. The model's output keeps `LLM_GRAPH_FORMAT` |
| `term_sheet` | A listing built by `ontocast/prompt/term_sheet.py`, with no term index |

The term sheet groups terms under `Classes`, `Properties` and `Individuals`,
one line each, sorted by IRI so the same snapshot always renders the same
bytes:

```
## Classes
  ex:PowderSample  "Powder sample"  < ex:Sample
## Properties  (domain -> range)
  ex:hasThickness  "has thickness"  ex:Sample -> ex:QuantityValue
## Individuals  (units, vocabulary values)
  ex:Approximate  "approximately"  : ex:EpistemicQualifier  ~ ~; ≈; ca.; about
      note: Use when the text qualifies a value as approximate.
```

A line keeps:

- the prefixed name, and the shortest `rdfs:label` or `skos:prefLabel` as the
  display name;
- every other label and `skos:altLabel` after `~`, without a count limit;
- for a class, its named parents (`rdfs:subClassOf`, `owl:equivalentClass`);
  for a property, `domain -> range` and its super-properties; for an
  individual, its types other than `owl:NamedIndividual`, `rdfs:Resource` and
  `owl:Thing`;
- one `note:`, the first of `skos:scopeNote`, `skos:definition` and
  `rdfs:comment` the term has.

It drops the per-statement syntax of RDF, anonymous class expressions such as
`owl:Restriction` blank nodes, and every other predicate. A term without a
label or note keeps its name and structure. Terms are classified by declared
type, falling back on structure: a subject with a domain, range or
super-property is a property, and one with a superclass is a class.

### How `auto` resolves

`auto` is resolved once, when `ServerConfig` is validated
(`ontocast/config/settings.py`): `term_sheet` when `RENDER_MODE=facts`,
`inherit` otherwise. The prompt profile, the run manifest and the LLM cache key
see only the resolved value. An explicit `term_sheet` with any other render
mode is a configuration error, because the ontology loop answers with a patch
against the statements in its chapter, and a listing has no statements to
patch.

A per-request `render_mode` does not re-resolve it. A request that switches an
`ontology_and_facts` deployment to `facts` gets a graph chapter rather than a
term sheet: more tokens, same result. The converse cannot fail, because the
chapter format reaches only the facts loop. The ontology loop always renders
its chapter in `LLM_GRAPH_FORMAT`, and the ontology critic reads an indexed
version of it in which each statement carries an id it can cite.

## Condensing to ONTOLOGY_CONTEXT_MAX_TRIPLES

`ontocast/onto/ontology_condense.py` trims a graph over
[`ONTOLOGY_CONTEXT_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_context_max_triples)
in three passes, stopping as soon as it fits:

1. **Header and list noise**: `rdf:first`, `rdf:rest`, `owl:imports`, the
   `owl:` version and compatibility annotations, and `dcterms:creator`,
   `license`, `created`, `modified`, `identifier`, `publisher`, `contributor`.
2. **Redundant structure**: `rdf:type owl:Class`, `rdfs:Class` or
   `owl:NamedIndividual` where the subject has a more informative type or a
   named superclass; restriction blank nodes with fewer than two constraining
   predicates; blank nodes nothing refers to.
3. **Glosses**: `rdfs:comment`, `skos:definition`, `skos:scopeNote`,
   `skos:altLabel`.

`rdfs:label`, `skos:prefLabel`, `rdf:type`, `rdfs:subClassOf`,
`owl:equivalentClass`, `rdfs:domain`, `rdfs:range` and `rdfs:subPropertyOf`
are never dropped, and neither is any predicate the passes do not name. The
budget is therefore best-effort: a graph still over it after the third pass is
used as is, with a warning that names the remedy (a smaller catalog, or
`selected_vector_search_ontology`). Leaving the setting empty disables
condensing.

The budget applies to every ontology chapter: both loops, render and critic.
The ontology critic's chapter is condensed before its statements are numbered,
so ids exist only for statements the critic is shown.
[`ONTOLOGY_MAX_TRIPLES`](../reference/configuration/pipeline.md#ontology_max_triples)
is unrelated: it bounds the ontology loop's working graph, and an update that
would exceed it is discarded and counted as
`ontology/update_rejected_over_budget`.

## Text caps

The text caps (`TextCaps` in `ontocast/onto/ontology_condense.py`) group text
predicates into three roles:

| Role | Predicates | Setting |
|---|---|---|
| Naming | `rdfs:label`, `skos:prefLabel`, `skos:altLabel` | [`ONTOLOGY_TEXT_MAX_CHARS_NAMING`](../reference/configuration/pipeline.md#ontology_text_max_chars_naming) |
| Contract | `skos:scopeNote`, `skos:definition` | [`ONTOLOGY_TEXT_MAX_CHARS_CONTRACT`](../reference/configuration/pipeline.md#ontology_text_max_chars_contract) |
| Prose | `rdfs:comment`, `skos:example`, `skos:note`, `skos:editorialNote`, `skos:historyNote` | [`ONTOLOGY_TEXT_MAX_CHARS_PROSE`](../reference/configuration/pipeline.md#ontology_text_max_chars_prose) |

The per-role caps apply first. Then, if
[`ONTOLOGY_TEXT_TOTAL_BUDGET`](../reference/configuration/pipeline.md#ontology_text_total_budget)
is set and the summed length is over it, the roles are tightened in the order
prose, contract, naming. Each role is clipped to the largest cap that brings
the total within budget, found by bisection, but never below a fixed minimum
per role; tightening stops as soon as the total fits. Naming has the highest
minimum, so short names are left alone. A chapter still over budget is used as
is, with a warning and the `chapter/text_over_budget` counter: what remains is
the vocabulary itself, and the remedy is fewer terms.

Clipping one literal works by content first: when the text has more than one
sentence and the first fits the cap, OntoCast keeps the first sentence, plus
the second if it fits and states when the term applies (words such as *use*,
*only*, *when*, *must*). Otherwise it cuts at the last word boundary within the
cap. Either way it appends `…` and keeps the literal's language tag or
datatype. No statement is removed.

Caps apply to the facts render and critic chapters. The ontology
loop's chapters are never capped: its critic cites statements by id and may
delete one, and a clipped literal would not match the statement it names.
With every cap unset, literals are left exactly as authored and prompts do not
change.

The `chapter/*` counters are recorded once per chapter built, not per call
that reads it, and only when at least one cap is set; with caps set they are
recorded even when nothing was clipped. See [Telemetry
counters](telemetry_counters.md#budget-counters).

## Document-scope chapters

How many distinct ontology chapters a document's facts fan-out builds depends
on the render mode and
[`ONTOLOGY_CONTEXT_SCOPE`](../reference/configuration/pipeline.md#ontology_context_scope)
(`ontocast/stategraph/node_factories.py`, `ontocast/stategraph/context_resolver.py`):

- **After an ontology stage** (`ontology_and_facts`), the reduced ontology
  artifacts are merged once per document and every unit reads that merge. The
  counter `ctx/merge_document_ontology.calls` is `1` per document.
- **Facts only, `unit` scope** (default): each unit resolves its own context,
  so chapters match only where two units resolve the same context.
- **Facts only, `document` scope**: every unit's context is resolved first
  (bounded by `PARALLEL_WORKERS`), and the union, with source IRIs sorted and
  ontology headers stripped, becomes the one chapter every unit reads. A unit
  whose own context is empty does not empty the union.
  `ctx/union_document_ontology.calls` counts this, and
  `ontology_snapshot_triples` reports the union's size.

[`FANOUT_WARMUP_UNITS`](../reference/configuration/pipeline.md#fanout_warmup_units)
then runs that many units to completion before starting the rest, so the rest
can read the prefix the first ones wrote. Both settings act only in the
document fan-out. The single-unit path (`/process_unit`, and `ontocast
process --use-unit-pipeline`) ignores them, and a run manifest from that path
leaves them out.

### The closure budget

In `selected_vector_search_ontology` mode, retrieval includes a small source
ontology whole once any of its terms is retrieved, up to
[`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES`](../reference/configuration/ontology-retrieval.md#ontology_patch_small_module_closure_max_triples)
triples per ontology. A vocabulary shown in part invites the model to invent
the missing property names.
[`ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TOTAL_TRIPLES`](../reference/configuration/ontology-retrieval.md#ontology_patch_small_module_closure_max_total_triples)
caps what these inclusions add together. Candidates are admitted best first,
by the highest retrieval score any of their terms reached, with ties broken by
IRI; one too large for the remaining budget is skipped and the next is tried.
`patch_retrieval.module_closure_iris`, `module_closure_triples` and
`module_closure_declined_iris` record the outcome. [Retrieval](retrieval.md)
covers the rest of retrieval sizing.

## What joins the LLM cache key

The [LLM response cache](../guides/llm_caching.md) key is built from the full
prompt text and `llm_cache_config` in `ontocast/tool/llm.py`:

- the cache format version, provider, model name, temperature and base URL;
- `LLM_THINK`, `LLM_NUM_PREDICT` and `LLM_NUM_CTX`;
- `LLM_REASONING_EFFORT` and `LLM_THINKING_BUDGET`, only when set;
- the output schema name, for structured calls, and any call arguments.

Everything on this page reaches the key through the prompt text: the chapter
format, the triple budget, the text caps, the scope and the shapes contract
change the prompt, and so the key, whenever they change what the model is
shown. A setting that changes nothing in a given prompt, such as a cap no
literal reaches, leaves that key alone. `LLM_PROMPT_CACHE_KEY` only routes
requests at the provider and is not part of the key.
