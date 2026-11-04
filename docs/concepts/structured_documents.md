# Structured documents

Often only part of a document is worth extracting: the methods and results of
a paper, the risk factors of an annual report. This page explains how OntoCast
labels each part of a document with its section, how it decides which kind of
document it is reading, how it decides which parts to drop, and how you choose
the sections to extract.

## Section labels

While chunking, OntoCast gives every content unit the label of the section it
came from, such as `methods`, `results` or `risk_factors`. Units never span two
labeled sections, so each has at most one label. A unit OntoCast cannot place
confidently gets no label at all: an unlabeled unit is merely never selected by
`target_sections`, while a wrongly labeled one would be selected by mistake.
For the same reason, size never joins unlabeled text to a labeled unit;
neighbouring unlabeled sections are joined with each other instead. A label
alone never drops a unit: see [how a unit is routed](#how-a-unit-is-routed).

Labels are recorded in the facts' [provenance](ontologies_and_facts.md#provenance),
together with how each was decided and how confident that decision was.

## Section schemas

Labels only mean something for a kind of document. A **section schema** is the
set of labels for one kind, with the headings that identify each label. Every
document is read against one schema:

| Schema | Documents | Boilerplate | Reference lists |
|---|---|---|---|
| `academic` (default) | Research articles, conference papers, theses, reviews | `acknowledgements` | `references` |
| `financial` | Annual and quarterly reports, filings | | `references` |
| `clinical` | Trial protocols and study reports | | `references` |
| `legal` | Contracts and other agreements between parties | | |
| `patent` | Patents and patent applications | `citations` | |
| `standard` | Specifications and standards for implementers | `acknowledgements` | `references` |
| `manual` | User guides, installation guides, runbooks | | |
| `fiction` | Novels and short fiction | `acknowledgements` | |
| `news` | News articles and press releases | `boilerplate`, `contact` | |
| `general` | Anything else | | `references` |

Each label has a route class in its schema file: `content` (every label not
listed above), `boilerplate` or `citations`. The class is evidence for
routing, never a decision on its own.

Use `ontocast sections` (below) to see a schema's labels on a real document.

### Which schema a document gets

OntoCast never overrides what you say; it only fills the gap when you say
nothing. In order:

1. `section_schema_id`, when you pass it.
2. `document_type_hint`, when it names a known kind of document as a whole
   word: `10-K` and `annual report` mean `financial`, `thesis` means
   `academic`, `RFC` means `standard`.
3. Detection from the document's own headings.
4. `academic`, as a fallback: nothing chose it, so its boilerplate sections are
   not dropped.

Detection is set by
[`CHUNK_SECTION_SCHEMA_DETECT`](../reference/configuration/chunking.md#chunk_section_schema_detect):

| Value | How the schema is detected | Cost |
|---|---|---|
| `off` | Not at all; the default schema is used | None |
| `lexical` | From headings that only one schema recognizes | None |
| `headings` (default) | The above, then each heading votes for the schema whose labels it is closest to | The chunker's local embedding model (`semantic-chunking` extra), used only when the first step cannot decide. Without the extra, this is `lexical` |
| `auto` | The above, then the body text, for documents with almost no headings | Same model, more text |

Detection gives up rather than guess, and the document is then read as
`academic`: a wrong schema would mislabel every section. Raising
[`CHUNK_SECTION_SCHEMA_DETECT_MIN_MARGIN`](../reference/configuration/chunking.md#chunk_section_schema_detect_min_margin)
makes it detect less often and more surely. `general` is never detected; pass
it explicitly.

## How sections are recognized

[`CHUNK_SECTION_CLASSIFIER`](../reference/configuration/chunking.md#chunk_section_classifier)
sets how far OntoCast goes to label a section. Each step only sees what the
steps before it left unlabeled:

| Value | Steps | LLM calls |
|---|---|---|
| `off` | None. Units are not labeled, so section filters and boilerplate exclusions are off; reference lists and front matter are still recognized from their content | None |
| `heading` | The document's outline; heading patterns and keywords, so `Results and Discussion` and `Experimental Section` are recognized; filling a gap from the schema's usual section order, never backwards | None |
| `heuristic` (default) | The above, plus labeling regions without headings from their content | None |
| `llm` | The above, plus one batched call over whatever is still unlabeled | About one per document |

A heading OntoCast does not recognize still ends the section before it, so a
label never runs on into the next section.

Two settings refine this:

- [`CHUNK_SECTION_DENSITY`](../reference/configuration/chunking.md#chunk_section_density)
  decides what content alone may label. `conservative` (default) recognizes only
  reference lists and acknowledgements (at least two thanks or funding
  statements), whose form is unmistakable. Such a label applies to that part
  only; it is not carried to the parts after it.
  `aggressive` also guesses methods, results and introduction, which content
  does not separate cleanly, so a filter may then act on a wrong label.
- [`CHUNK_SECTION_TEXT_HEADINGS`](../reference/configuration/chunking.md#chunk_section_text_headings)
  finds headings in plain text, for documents whose conversion produced no
  heading structure.

## Choosing the sections to extract

Pass these with the request, on `/process` (query, form or JSON body) or as
flags of `ontocast process` (`--target-sections` and so on). They are not
environment settings.

| Parameter | What it does |
|---|---|
| `target_sections` | Keep only these sections |
| `exclude_sections` | Drop these sections, where the label is strong (see below). Unset, the schema's boilerplate sections are dropped; an empty value keeps everything |
| `section_schema_id` | The schema to read the document against |
| `document_type_hint` | Free text describing the document, from which the schema is chosen |
| `summarize_sections` | Summarize these units before extraction; `*` or empty for all. It never changes what is extracted |
| `summary_max_sentences` | The length of each summary |

Lists are comma-separated (`results,methods`) or a JSON array. For example:

```bash
curl -X POST http://127.0.0.1:8999/process \
  -F "file=@paper.pdf" -F "target_sections=methods,results"
```

```bash
ontocast process --input-path ./reports --output-dir ./out \
  --section-schema-id financial --target-sections risk_factors,md_and_a
```

`target_sections` applies first, to labels of any strength, then routing. A label no schema
knows is dropped with a warning; a list in which no label is known is rejected
(HTTP `422`, or a usage error from the CLI), so a typo cannot silently switch
filtering off.

When a selection removes every unit, the run by default finishes with an empty
graph, which looks the same as a document with nothing to extract. Set
[`CHUNK_SECTION_FILTER_ON_EMPTY=error`](../reference/configuration/chunking.md#chunk_section_filter_on_empty)
to fail the run instead (HTTP `422`, or a failed file in a batch run). This
also covers a schema's boilerplate exclusions emptying a document on their own.

`target_sections` keeps only labeled units, so text whose label is missing is
lost to it. The run manifest records the characters it removed and how many of
them were unlabeled, and a warning is logged when it removes most of a
document.

### Summaries

With `summarize_sections`, each selected unit is summarized by the model just
before it is extracted, and the render and critic read the summary instead of
the full text. That costs one LLM call per unit and loses detail, so use it for
long, discursive sections rather than for tables of measurements. A unit is
summarized once, even when both the ontology and the facts block run.

## How a unit is routed

Before extraction every unit takes one route: extracted, extracted as
bibliographic records, kept but flagged as front or back matter, or dropped.
A drop needs a **strong** label and content that agrees, or content that is
conclusive on its own. A label is strong when it comes from an exact heading
(`References`, `Acknowledgements`, `Author Contributions`) or from the content
itself; a keyword inside a longer heading (`Funding Mechanisms`), a label
carried from a neighbouring part or one set by the model is weak, and never
drops a unit.

| Unit | Route |
|---|---|
| A reference list, by its content (nearly every paragraph is one cited work), or under a strong reference-list label its content agrees with | [`CHUNK_BIBLIOGRAPHY_MODE`](../reference/configuration/chunking.md#chunk_bibliography_mode): `skip` (default) drops it; `citations_only` extracts bibliographic records (works, authors, venues), never domain facts; `domain_facts` treats it like any other unit |
| A strong boilerplate label on text that states no measurement, when `exclude_sections` is unset and the schema was not a fallback | Dropped |
| A strong label `exclude_sections` names | Dropped |
| Front or back matter recognized from its content: author information, notes, licence, data availability, competing interests | [`CHUNK_NON_CONTENT_MODE`](../reference/configuration/chunking.md#chunk_non_content_mode): `extract` (default) keeps it, marked as non-content; `skip` drops it |
| Shorter than [`CHUNK_MIN_UNIT_CHARS`](../reference/configuration/chunking.md#chunk_min_unit_chars) | Dropped |
| Anything else | Extracted |

A reference list at the start or end of a unit, such as an unheaded
bibliography after a conclusion, is first cut into a unit of its own, so the
prose around it is kept. Table rows never count as cited works: a table that
cites its sources holds data. Nor does an entry that states a measurement,
which is how a discussion of prior studies differs from a reference list.

Every unit records the reason for its route (`route_reason`), every route
other than plain extraction is logged with the unit's size, label and the step
that labeled it, and the run manifest's `route_histogram` counts units and
characters per route and reason. In `citations_only` mode the record
vocabulary defaults to schema.org; set
[`CHUNK_CITATION_VOCABULARY`](../reference/configuration/chunking.md#chunk_citation_vocabulary)
to use another, such as BIBO. The detectors work on English, Latin-script
documents; elsewhere they find less, never more.

## Preview before you extract

A wrong label changes what is extracted without showing up anywhere in the
result. `ontocast sections` shows the decisions before you spend anything:

```bash
ontocast sections --input-path ./paper.pdf
ontocast sections --input-path ./paper.pdf --target-sections results --as-json
```

It prints the schema (with the step that chose it and the evidence for the
runners-up), the document's outline, and every unit's label with the step that
decided it and its confidence, and its route with the reason, so you see what
would be dropped and why. It makes no LLM calls and needs no credentials,
unless you pass `--section-classifier llm`.

## What to read next

- [How OntoCast works](index.md): where chunking sits in the pipeline.
- [Chunking settings](../reference/configuration/chunking.md): every chunking setting with its default.
- [HTTP API](../reference/http_api.md): the request parameters on `/process`.
