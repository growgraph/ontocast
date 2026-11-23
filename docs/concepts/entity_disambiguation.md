# Entity disambiguation

Each content unit is extracted on its own, so the same sample, company or
person mentioned in three parts of a document arrives as three entities. This
page explains how OntoCast decides which entities are the same, so you can
predict when two of them will be merged and find out why two were not.

## When two entities merge

Merging happens once per document, after every unit's facts are extracted
(see [How OntoCast works](index.md#the-facts-block)). Entities of different
documents are never merged; [Cross-document identity](cross_document_identity.md)
links them instead. For each pair of entities:

1. **Names alone prove nothing.** Two units that both write
   `cd:temperature_value` may mean two different measurements, so OntoCast
   keeps each unit's names apart and treats the pair like any other candidate
   ([`AGG_UNIT_SCOPED_FACT_IRIS`](../reference/configuration/aggregation.md#agg_unit_scoped_fact_iris)).
   Terms of your ontology, such as an individual it declares, are the same
   wherever they appear. A term of a well-known vocabulary written with
   another scheme or trailing separator (`http://schema.org/Person` for
   `https://schema.org/Person`, `sosa#` for `sosa/`), and a literal datatype
   so written, is rewritten to the vocabulary's own namespace as soon as the
   model's output is parsed, so it is the same term too.
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
4. **A candidate merges only on evidence of identity:** a shared key, a
   coreference "same" verdict, an equal label, or a label that matches with
   initials ("D. Baranov" and "Dmitry Baranov"). A label that only overlaps
   another ("sample batch" and "large sample batch", words in common, similar
   spelling, or IRIs with the same local name) makes identity a question, not
   an answer: such a pair waits for a coreference verdict and stays apart
   without one. Only entities the document minted are put to coreference, so
   a vocabulary term is never merged by its name: a misspelled ontology term
   stays as written and validation reports it with the term it likely meant
   (see [Checking facts](checking_facts.md)).
5. **Stronger evidence is applied first:** keys and verdicts, then equal labels
   and initials, the most similar pair first within each. So a generic mention
   cannot join two specific things before their own twins are merged.
6. **Refusals hold for the whole group.** If A and C were refused, they stay
   apart even when A matches B and B matches C. Guards read the merges made so
   far: two places in one region minted under two names are one place once the
   regions merge, so refused pairs are tried again until nothing changes. A
   merge can also make two things distinct: two places reporting one measure
   with different results are distinct once their measures are one. After
   each batch of merges the merged groups are checked again, and a merge that
   brought two now-distinct members together is undone for good. The outcome
   does not depend on how the entities are named.
7. **The accepted groups are merged** under one IRI. Entities that were not
   merged but share a name keep it with a suffix: `cd:sample_a` and
   `cd:sample_a_1`.
8. **Values that say one thing are one value.** Two values (blank nodes of a
   value class, such as `qudt:QuantityValue` or `schema:MonetaryAmount`) of
   one statement and property that share a number, and differ in nothing both
   state, become one value with everything either stated and one lexical form
   of each number: the same number written by two parts is served once. An
   estimate and its interval share no number and stay two values; so does one
   number in two measurement units, or with two different texts on one
   property (two currencies, two street addresses). A blank node of no value
   class is never merged.
9. **Validation can undo a merge.** If the merged graph shows a sign of a wrong
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
| `same_unit` | One unit minted both: a unit names each thing once, so its two IRIs are two things | |
| `relation` | One points to the other, or validation or a coreference "different" verdict split them | |
| `literal_conflict` | They state different values for the same property, such as two different sizes | |
| `functional_iri_conflict` | They point to different things through a property that allows one value, such as two different units; things merged in the same pass count as one | [`AGG_FUNCTIONAL_MIN_EMPIRICAL_SUPPORT`](../reference/configuration/aggregation.md#agg_functional_min_empirical_support) |
| `statement_conflict` | Statements about them report one property with conflicting results and no differing time, condition or procedure: one sample cannot have two sizes at once. Statements with different times (a time series) do not conflict | |
| `dimension_conflict` | They are properties, and the results reported for them are in measurement units of different dimensions: a length and a duration are two quantities. Inert for results without a unit, or with a unit whose dimension OntoCast does not ship | |
| `initials_conflict` | Their labels differ only in an initial or a single letter: "Sample A" and "Sample B" | |
| `role` | One is used as a property, the other as a thing | |
| `type` | Two entities found in the text: your ontologies declare their types, or the types' superclasses, disjoint (`owl:disjointWith`, `owl:AllDisjointClasses`). Parts of a document type one thing differently, so types that merely differ do not refuse a merge. An entity and an individual of your ontology: they share no type | [`AGG_TYPE_GUARD_UNTYPED`](../reference/configuration/aggregation.md#agg_type_guard_untyped) decides whether a typed and an untyped entity may merge |
| `overlap` | Their labels overlap but are not equal: one inside the other, words in common, similar spelling, or (when one has no label, or against a term of your ontology) the same IRI local name. The pair is a coreference candidate and merges only on a "same" verdict. Entities that carry values must match exactly. Labels that carry different numbers ("June 2021", "29 June 2021") never overlap | [`AGG_LEXICAL_LABEL_JACCARD`](../reference/configuration/aggregation.md#agg_lexical_label_jaccard), [`AGG_LEXICAL_SEQUENCE_RATIO`](../reference/configuration/aggregation.md#agg_lexical_sequence_ratio), [`AGG_LEXICAL_TOKEN_JACCARD`](../reference/configuration/aggregation.md#agg_lexical_token_jaccard) |
| `lexical` | Their labels are not alike at all. Two labelled entities are compared by their labels alone. Case changes inside a word are kept: "ABCs" is one word, not "A" and "BCs" | |
| `cluster_veto` | The pair itself was fine, but merging it would have joined two entities another guard had refused | |

A property counts as allowing one value when the ontology says so, or when
every subject in the document uses it once and at least
`AGG_FUNCTIONAL_MIN_EMPIRICAL_SUPPORT` subjects use it.

Entities that share a label are always compared, even when their embeddings
place them apart; a label shared by more than eight entities is generic and
brings nothing together.

## Coreference the guards cannot settle

The parts of a document are extracted in parallel, so no part sees the
entities another part named. After merging, one LLM call per document (of two
parts or more) puts the pairs the rules above cannot settle to the model:

- first, labels that overlap without being equal ("sample batch" and "large
  sample batch"), with no guard saying they are distinct: the rules above
  hand exactly these pairs to the model, and they join only on a "same"
  verdict;
- then, taking turns: candidates whose labels are not alike at all ("the
  first applicant" and "Jane Doe"), and two entities of one type from
  different parts that are the only two pointing to one node by one
  property (both "applicant in" one case).

Within each group the most similar pair comes first.

The model sees each entity as its part stated it (labels, types, values,
relations, and the sentence that names it) and answers same, different or
unsure for each pair. "Same" merges the pair like a shared key, "different"
keeps it apart, and every guard above still applies: the model cannot merge
two things one part told apart. Its verdicts also hold when validation
re-merges the parts.
The step is on by default;
[`AGG_LLM_COREFERENCE=false`](../reference/configuration/aggregation.md#agg_llm_coreference)
turns it off, and
[`AGG_LLM_COREFERENCE_MAX_PAIRS`](../reference/configuration/aggregation.md#agg_llm_coreference_max_pairs)
bounds the pairs per call (80 by default), all asked in that one call.
`facts_coreference_pairs`,
`facts_coreference_same` and `facts_coreference_different` in
`retrieval_metrics` count them.

## Tuning

- **Too many duplicates** (one thing, several entities): lower
  `AGG_CANDIDATE_SIMILARITY_THRESHOLD`. More pairs reach the guards; the guards
  still refuse the ones whose facts disagree.
- **Wrong merges** (two things, one entity): find the pair in the log. A merge
  needs an equal label, initials, a shared key or a "same" verdict, so look
  for which one joined them.
- **Pairs left for coreference:** with `AGG_LLM_COREFERENCE=false`, or beyond
  `AGG_LLM_COREFERENCE_MAX_PAIRS`, labels that only overlap stay apart. They
  show in `facts_twin_labels` only when the labels are equal, so compare
  `facts_merge_rejections_by_reason["overlap"]` with the pairs asked.
- To see what merging does at all, run once with
  [`AGG_ENABLED=false`](../reference/configuration/aggregation.md#agg_enabled):
  every entity stays its own node, with its provenance, and no candidate is
  formed. It needs `AGG_UNIT_SCOPED_FACT_IRIS=true`, or same-named entities
  from different units fuse by name anyway. Validation can still rewrite the
  merged graph; set `FACTS_MERGE_REPAIR_PASSES=0` for a graph that only
  extraction shaped.

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
