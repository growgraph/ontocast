# Entity disambiguation

Each content unit is extracted on its own, so the same sample, company or
person mentioned in three parts of a document arrives as three entities. This
page explains how OntoCast decides which entities are the same, so you can
predict when two of them will be merged and find out why two were not.

## When two entities merge

Merging happens once per document, after every unit's facts are extracted
(see [How OntoCast works](index.md#the-facts-block)). For each pair of
entities:

1. **Names alone prove nothing.** Two units that both write
   `cd:temperature_value` may mean two different measurements, so OntoCast
   keeps each unit's names apart and treats the pair like any other candidate
   ([`AGG_UNIT_SCOPED_FACT_IRIS`](../reference/configuration/aggregation.md#agg_unit_scoped_fact_iris)).
   Terms of your ontology, such as an individual it declares, are the same
   wherever they appear.
2. **A pair becomes a candidate** when the embeddings of their labels are at
   least
   [`AGG_CANDIDATE_SIMILARITY_THRESHOLD`](../reference/configuration/aggregation.md#agg_candidate_similarity_threshold)
   similar, computed with
   [`AGG_EMBEDDING_MODEL`](../reference/configuration/aggregation.md#agg_embedding_model).
   It is also a candidate, whatever its labels, when both entities state the
   same short value for a property that identifies things, such as a case
   number or a catalog code
   ([`AGG_NATURAL_KEY_MERGE`](../reference/configuration/aggregation.md#agg_natural_key_merge)).
3. **Guards then check the candidate** against what the two entities state, and
   any one of them can refuse it (see the table below). The threshold is set
   low on purpose: the guards, not the similarity score, decide.
4. **Refusals hold for the whole group.** If A and C were refused, they stay
   apart even when A matches B and B matches C.
5. **The accepted groups are merged** under one IRI. Entities that were not
   merged but share a name keep it with a suffix: `cd:sample_a` and
   `cd:sample_a_1`.
6. **Validation can undo a merge.** If the merged graph shows a sign of a wrong
   merge, such as a single-valued property with two values, the offending
   group is split and the document's facts are merged again, up to
   [`FACTS_MERGE_REPAIR_PASSES`](../reference/configuration/facts-validation.md#facts_merge_repair_passes)
   times. See [Validation and SHACL](../guides/validation.md).

## Why a merge was refused

`retrieval_metrics.facts_rejected_merges` counts the refused pairs. With
[`LOGGING_LEVEL=DEBUG`](../reference/configuration/pipeline.md#logging_level),
the log names each pair and the reasons, which can be several:

| Reason | The pair was refused because | Setting |
|---|---|---|
| `sibling` | Both are values of one subject, such as the lower and upper bound of a range, or two samples in a series | [`AGG_SIBLING_GUARD_SCOPE`](../reference/configuration/aggregation.md#agg_sibling_guard_scope) |
| `literal_conflict` | They state different values for the same property, such as two different temperatures | [`AGG_LITERAL_CONFLICT_GUARD`](../reference/configuration/aggregation.md#agg_literal_conflict_guard) |
| `functional_iri_conflict` | They point to different things through a property that allows one value, such as two different units | [`AGG_FUNCTIONAL_MIN_EMPIRICAL_SUPPORT`](../reference/configuration/aggregation.md#agg_functional_min_empirical_support) |
| `initials_conflict` | Their labels differ only in an initial or a single letter: "Sample A" and "Sample B" | [`AGG_INITIALS_DISTINCT_GUARD`](../reference/configuration/aggregation.md#agg_initials_distinct_guard) |
| `role` | One is used as a property, the other as a thing | |
| `type` | Their types are incompatible | [`AGG_TYPE_GUARD_UNTYPED`](../reference/configuration/aggregation.md#agg_type_guard_untyped) decides whether a typed and an untyped entity may merge |
| `lexical` | Their labels are not close enough. Entities that carry values must match exactly; others may match loosely | [`AGG_LEXICAL_LABEL_JACCARD`](../reference/configuration/aggregation.md#agg_lexical_label_jaccard), [`AGG_LEXICAL_SEQUENCE_RATIO`](../reference/configuration/aggregation.md#agg_lexical_sequence_ratio), [`AGG_LEXICAL_TOKEN_JACCARD`](../reference/configuration/aggregation.md#agg_lexical_token_jaccard) |
| `cluster_veto` | The pair itself was fine, but merging it would have joined two entities another guard had refused | |

A property counts as allowing one value when the ontology says so, or when
every subject in the document uses it once.

## Tuning

- **Too many duplicates** (one thing, several entities): lower
  `AGG_CANDIDATE_SIMILARITY_THRESHOLD`. More pairs reach the guards; the guards
  still refuse the ones whose facts disagree.
- **Wrong merges** (two things, one entity): find the pair in the log. If no
  guard fired, raise the threshold or the lexical settings; if a guard is off,
  turn it on.
- A guard that is switched off shows its effect as a change in
  `facts_rejected_merges`, which is how to measure what it does on your
  documents.

## Comparing two graphs

The same matching compares an extracted graph with a reference one, for
example to evaluate a run against a hand-made graph. Three routes do this:
`POST /match/entities` aligns entities across graphs, `POST
/match/derive-matches` turns that into pairs, and `POST /match/evaluate`
scores precision, recall and F1 for triples, facts and entities. Entities are
the subjects and objects that are not vocabulary (classes, properties,
predicates). A score with nothing to divide by, such as precision for an empty
prediction, is `null`, so it does not count as zero in an average. The
`match-graphs` command runs the same comparison over two directories of Turtle
files. See the [HTTP API](../reference/http_api.md).

The route, the command and the `ontocast_align_entities` agent tool use
[`AGG_SIMILARITY_THRESHOLD`](../reference/configuration/aggregation.md#agg_similarity_threshold)
and [`AGG_EMBEDDING_MODEL`](../reference/configuration/aggregation.md#agg_embedding_model)
unless the caller names a `similarity_threshold` or `embedding_model`. Neither
affects merging during a run, which uses only the candidate threshold above.

**Facts** scores count only relations between instances, such as a sample to
its substrate; **triple** scores also count types and the class hierarchy. An
ontology term that both graphs use is matched once and is not counted as an
extra entity on either side.

## What to read next

- [Aggregation settings](../reference/configuration/aggregation.md): every setting with its default.
- [Validation and SHACL](../guides/validation.md): the checks that run after merging.
- [Performance tuning](../guides/performance.md): sharing one embedding model between chunking, retrieval and merging.
