# Structured documents

Often only part of a document is worth extracting: the methods and results of
a paper, the risk factors of an annual report. This page explains how OntoCast
labels each part of a document with its section, how it decides which kind of
document it is reading, and how you choose the sections to extract.

## Section labels

While chunking, OntoCast gives every content unit the label of the section it
came from, such as `methods`, `results` or `risk_factors`. Units never span two
sections, so each has at most one label. A unit OntoCast cannot place confidently
gets no label at all: an unlabeled unit is merely never selected by a filter,
while a wrongly labeled one would be dropped or kept by mistake.

Labels are recorded in the facts' [provenance](ontologies_and_facts.md#provenance),
together with how each was decided and how confident that decision was.

## Section schemas

Labels only mean something for a kind of document. A **section schema** is the
set of labels for one kind, with the headings that identify each label. Every
document is read against one schema:

| Schema | Documents | Excluded by default |
|---|---|---|
| `academic` (default) | Research articles, conference papers, theses, reviews | `acknowledgements`, `appendix` |
| `financial` | Annual and quarterly reports, filings | |
| `clinical` | Trial protocols and study reports | |
| `legal` | Contracts and other agreements between parties | |
| `patent` | Patents and patent applications | `citations` |
| `standard` | Specifications and standards for implementers | `acknowledgements` |
| `manual` | User guides, installation guides, runbooks | |
| `fiction` | Novels and short fiction | |
| `news` | News articles and press releases | `boilerplate`, `contact` |
| `general` | Anything else | |

Use `ontocast sections` (below) to see a schema's labels on a real document.

### Which schema a document gets

OntoCast never overrides what you say; it only fills the gap when you say
nothing. In order:

1. `section_schema_id`, when you pass it.
2. `document_type_hint`, when it names a known kind of document as a whole
   word: `10-K` and `annual report` mean `financial`, `thesis` means
   `academic`, `RFC` means `standard`.
3. Detection from the document's own headings.
4. `academic`.

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
| `off` | None. Units are not labeled, and section filters and default exclusions are off | None |
| `heading` | The document's outline; heading patterns and keywords, so `Results and Discussion` and `Experimental Section` are recognized; filling a gap from the schema's usual section order, never backwards | None |
| `heuristic` (default) | The above, plus labeling regions without headings from their content | None |
| `llm` | The above, plus one batched call over whatever is still unlabeled | About one per document |

A heading OntoCast does not recognize still ends the section before it, so a
label never runs on into the next section.

Two settings refine this:

- [`CHUNK_SECTION_DENSITY`](../reference/configuration/chunking.md#chunk_section_density)
  decides what content alone may label. `conservative` (default) recognizes only
  reference lists and acknowledgements (at least two thanks or funding
  statements), whose form is unmistakable. An excluded label found this way
  applies to that part only; it is not carried to the parts after it.
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
| `exclude_sections` | Drop these sections. Unset, the schema's default exclusions apply; an empty value keeps everything |
| `section_schema_id` | The schema to read the document against |
| `document_type_hint` | Free text describing the document, from which the schema is chosen |
| `summarize_sections` | Summarize these units before extraction; `*` or empty for all |
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

`target_sections` applies first, then `exclude_sections`. A label no schema
knows is dropped with a warning; a list in which no label is known is rejected
(HTTP `422`, or a usage error from the CLI), so a typo cannot silently switch
filtering off.

When a selection removes every unit, the run by default finishes with an empty
graph, which looks the same as a document with nothing to extract. Set
[`CHUNK_SECTION_FILTER_ON_EMPTY=error`](../reference/configuration/chunking.md#chunk_section_filter_on_empty)
to fail the run instead (HTTP `422`, or a failed file in a batch run). This
also covers a schema's default exclusions emptying a document on their own.

### Summaries

With `summarize_sections`, each selected unit is summarized by the model just
before it is extracted, and the render and critic read the summary instead of
the full text. That costs one LLM call per unit and loses detail, so use it for
long, discursive sections rather than for tables of measurements. A unit is
summarized once, even when both the ontology and the facts block run. Without
`target_sections`, a `summarize_sections` list also acts as the selection.

## Reference lists and front matter

Some units state no domain facts at all. OntoCast recognizes two kinds, from
their section label or their form, and routes them before extraction:

| Units | Setting | Values |
|---|---|---|
| Reference lists and bibliographies | [`CHUNK_BIBLIOGRAPHY_MODE`](../reference/configuration/chunking.md#chunk_bibliography_mode) | `skip` (default): dropped. `citations_only`: extracted as bibliographic records (works, authors, venues), never as domain facts. `domain_facts`: treated like any other unit |
| Front and back matter: author information, notes, licence, data availability, competing interests, funding, acknowledgements | [`CHUNK_NON_CONTENT_MODE`](../reference/configuration/chunking.md#chunk_non_content_mode) | `extract` (default): kept and marked as non-content. `skip`: dropped |

A front- or back-matter unit that states a measurement is always kept. In
`citations_only` mode the record vocabulary defaults to schema.org; set
[`CHUNK_CITATION_VOCABULARY`](../reference/configuration/chunking.md#chunk_citation_vocabulary)
to use another, such as BIBO. To drop fragments, such as a stray heading or a
caption, below a minimum length, set
[`CHUNK_MIN_UNIT_CHARS`](../reference/configuration/chunking.md#chunk_min_unit_chars).

Every dropped unit is logged, and the run manifest counts them by rule. Both
detectors work on English, Latin-script documents; elsewhere they find less,
never more.

## Preview before you extract

A wrong label changes what is extracted without showing up anywhere in the
result. `ontocast sections` shows the decisions before you spend anything:

```bash
ontocast sections --input-path ./paper.pdf
ontocast sections --input-path ./paper.pdf --target-sections results --as-json
```

It prints the schema (with the step that chose it and the evidence for the
runners-up), the document's outline, and every unit's label with the step that
decided it and its confidence. It makes no LLM calls and needs no credentials,
unless you pass `--section-classifier llm`.

## What to read next

- [How OntoCast works](index.md): where chunking sits in the pipeline.
- [Chunking settings](../reference/configuration/chunking.md): every chunking setting with its default.
- [HTTP API](../reference/http_api.md): the request parameters on `/process`.
