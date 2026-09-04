---
search:
  boost: 3
---

# Configuration System

OntoCast configuration is powered by Pydantic `BaseSettings` and is loaded from environment variables (typically via `.env`).

!!! tip "This page is the complete reference — around 200 variables"

    If you are configuring OntoCast for the first time, start from
    [Configuration Playbooks](playbooks.md) and `.env.example.minimal` instead:
    47 variables, grouped by decision, with a playbook per task. Come back here
    for the full surface once you know which knob you need.

## Overview

- Typed config sections with defaults
- Environment variable parsing (including lists and booleans)
- Validation for provider/model compatibility
- Unified `Config` object shared across tools and server

## Configuration Shape

```python
Config
├── tool_config: ToolConfig
│   ├── llm_config: LLMConfig
│   ├── chunk_config: ChunkConfig
│   ├── converter_config: ConverterConfig
│   ├── path_config: PathConfig
│   ├── fuseki: FusekiConfig
│   ├── domain: DomainConfig
│   ├── web_search: WebSearchConfig
│   ├── aggregation: AggregationConfig
│   ├── embedding: EmbeddingConfig
│   ├── patch_retrieval: PatchRetrievalConfig
│   ├── vector_store: VectorStoreConfig
│   ├── qdrant: QdrantConfig
│   ├── lancedb: LanceDBConfig
│   └── facts_validation: FactsValidationConfig
├── server: ServerConfig
├── logging_level: str | None
└── clean: bool
```

## Environment Variables

### LLM

```bash
LLM_PROVIDER=openai                     # openai | ollama | anthropic | google
LLM_MODEL_NAME=gpt-5.4
LLM_TEMPERATURE=0.0
LLM_API_KEY=your_api_key_here           # required for openai, anthropic, google
LLM_BASE_URL=http://localhost:11434     # optional (ollama; anthropic proxy URL)
```

| Provider | Example `LLM_MODEL_NAME` | `LLM_API_KEY` |
|----------|--------------------------|---------------|
| `openai` | `gpt-5.4` | Required |
| `ollama` | `llama3.1` | Not used (`LLM_BASE_URL` required) |
| `anthropic` | `claude-sonnet-4-20250514` | Required |
| `google` | `gemini-2.0-flash` | Required |

OntoCast uses `LLM_API_KEY` for all cloud providers (not `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`).

`LLM_JSON_MODE=true` constrains OpenAI decoding to syntactically valid JSON
(`response_format: json_object`). Every response is parsed as a JSON envelope
whatever `LLM_GRAPH_FORMAT` is set to -- Turtle only makes the graph fields flat
strings *inside* that envelope -- so this makes a class of syntax error
unreachable rather than repaired after the fact. It is off by default because
OpenAI rejects a request in this mode unless the word "JSON" appears in the
prompt, which is a property of the prompt set in use. Strict `json_schema` mode
is deliberately not offered: it requires closed schemas and the graph fields are
open. OpenAI only; other providers ignore it.

#### `LLM_MODEL_NAME` accepts any string

The model enums in `ontocast.config` (`OpenAIModel`, `OllamaModel`, `ClaudeModel`,
`GeminiModel`) are **presets, not a whitelist** — they exist so common choices are
discoverable and type-checkable. Any other string is passed through to the provider,
which is the authority on whether it exists; OntoCast logs a warning and continues.

That matters for two cases a fixed list cannot serve: a model released after your
OntoCast version, and an **OpenAI-compatible endpoint** hosting another vendor's
models. For the second, combine `LLM_PROVIDER=openai` with the vendor's `LLM_BASE_URL`:

```bash
# Moonshot (Kimi)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.moonshot.ai/v1
LLM_MODEL_NAME=kimi-k3
LLM_API_KEY=your_moonshot_key

# Alibaba Model Studio (Qwen)
LLM_PROVIDER=openai
LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen3-max
LLM_API_KEY=your_dashscope_key
```

The same shape works for OpenRouter, Together, and a self-hosted vLLM server. Open-weight
Qwen, Kimi and DeepSeek builds run locally through `LLM_PROVIDER=ollama` instead — see
the Ollama controls below, and set `LLM_THINK` for the reasoning variants.

**Disk cache and provider concurrency** (see [LLM Caching](llm_caching.md)):

```bash
LLM_CACHE_ENABLED=true          # read/write disk cache (default true)
LLM_CACHE_READ_ONLY=false       # use cache without writing new entries
LLM_MAX_INFLIGHT=16             # max concurrent provider requests (all documents)
LLM_REQUEST_TIMEOUT_SECONDS=180 # abandon a call after this; empty to wait forever
```

A hung provider call holds both a unit-worker slot and an `LLM_MAX_INFLIGHT`
slot, so without `LLM_REQUEST_TIMEOUT_SECONDS` a couple of them permanently
shrink the pipeline's effective width. A timed-out call fails only its own unit.

**Cloud reasoning controls:**

```bash
LLM_REASONING_EFFORT=           # none|minimal|low|medium|high|xhigh: OpenAI
                                # reasoning_effort, Gemini 3+ thinking_level;
                                # unset = provider default
LLM_THINKING_BUDGET=            # Gemini 2.5 thinking_budget: 0 off, -1 model-chosen,
                                # N caps it; unset uses the model default
```

**Ollama-specific generation controls** (ignored by other providers):

```bash
LLM_THINK=                      # true/false: thinking mode for qwen3, deepseek-r1, ... (Ollama)
LLM_NUM_PREDICT=                # max tokens to generate; unset = Ollama default
LLM_NUM_CTX=                    # context window (prompt + output). Ollama defaults to
                                # 2048-4096; raise to 16384+ for large prompts
```

`LLM_REASONING_EFFORT` and `LLM_THINKING_BUDGET` are two spellings of one
lever: reasoning tokens are counted inside the output total and can dominate
it (`reasoning_share_of_output` in the run manifest says by how much).
`LLM_REASONING_EFFORT` is the discrete one, read by OpenAI reasoning models
and by Gemini 3+; `LLM_THINKING_BUDGET` is the Gemini 2.5 integer spelling,
superseded from Gemini 3 on. Its vocabulary is the union across providers and
across model generations of one provider: the floor of the scale is spelled
`minimal` by some models and `none` by others, and `xhigh` is newer than both.
Which levels a given model accepts is the provider's business — nothing here
maps model names to levels, so a level the model does not take comes back as a
rejected request and aborts the run rather than being silently downgraded. On Google the two are mutually exclusive and
setting both is rejected at startup; setting the budget on a Gemini 3+ model
logs a warning and is ignored. Each knob joins the LLM cache key only when it
is set — leaving it unset keeps existing cache entries valid, setting it
re-issues every prompt.

`LLM_THINK=false` guarantees a non-empty `content` response; `true` captures
reasoning separately. Leaving it unset can yield an empty response when a
thinking model spends its whole budget reasoning — raise `LLM_NUM_PREDICT`
alongside it.

```bash
# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=claude-sonnet-4-20250514
LLM_API_KEY=your_anthropic_api_key_here

# Google Gemini
LLM_PROVIDER=google
LLM_MODEL_NAME=gemini-2.0-flash
LLM_API_KEY=your_google_api_key_here
```

### What happens to a response that will not parse

None of this is configurable — it is the contract every `PydanticOutputParser`
call runs under, and it decides how much a badly-behaved model costs. Read it
alongside the `llm/*` counters in
[Observability](observability.md#2-the-run-manifest), which is where each stage
below is counted.

A response is sanitised, then parsed **strictly** (`agent/common.py`):

1. **Sanitise.** `unescape_json_delimiters` repairs two malformations that are
   the model's, not the payload's: escaping the quotes that *delimit* JSON
   strings (`"text_fragment": \"…\",`) and escaping the whitespace between
   tokens. The scan is string-aware, so a legitimate in-string `\"` is left
   alone. JSON comments and trailing commas are stripped next.
2. **Parse strictly.** `parse_json_object` runs a real `json.loads`, keeping
   only the `strict=False` leniency for raw control characters inside strings —
   which models do emit. The lenient parser this replaced degraded a broken
   document to `None`, or to a silently *truncated* prefix, so the retry prompt
   carried a pydantic `input_value=None` error naming nothing.
3. **Fall back, twice.** A fenced ```` ```json ```` block is extracted and
   strict-parsed; failing that, `repair_bracket_kinds` rewrites closing brackets
   to the kind their opener demands. It only substitutes characters — never
   inserts, deletes or reorders — and gives up entirely on an unmatched closer
   or an unclosed frame at EOF, because that is genuine truncation and closing
   it would fabricate a payload the model never sent. A successful repair is
   counted as `llm/json_bracket_repair`.
4. **Fail loudly.** Anything still unparsed raises `LLMJsonParseError` carrying
   the decoder's line, column and a ±150-character window. The retry prompt
   shows the model that window rather than the whole ~11 KB response.

Retries (`llm/parse_retry`) back off exponentially with jitter, so N units
failing together do not re-issue in lockstep. Two rules bound them:

- **The same JSON *syntax* error twice ends the call** (`llm/parse_abandoned`).
  A model that emits one structural malformation twice emits it a third time,
  so the remaining attempts are spend with no expected return. The comparison
  is on the error *class*, not its position, which drifts between
  regenerations. Schema `ValidationError`s are excluded — those retries do
  converge.
- **Only parsing failures retry.** A rate limit or connection error propagates
  on first occurrence: showing the model its own malformed output is
  meaningless when no output arrived, and retrying would triple the request
  rate exactly when the provider is asking for less. The single exception is a
  **timeout**, which is not a "send less" signal — it gets one identical
  re-issue per call before propagating, because at `MAX_VISITS=2` a lost render
  silently costs a unit its entire critique.

### Server

```bash
HOST=127.0.0.1                           # loopback by default; see note below
PORT=8999
BASE_RECURSION_LIMIT=1000                # LangGraph step ceiling, scaled by ESTIMATED_CHUNKS
ESTIMATED_CHUNKS=30                      # expected units per document; only sizes the limit above
MAX_VISITS_PER_NODE=1                    # canonical name; MAX_VISITS is an accepted alias
FACTS_CRITIC_PASSES=1                    # review-and-patch passes per facts unit
RENDER_MODE=ontology_and_facts           # which pipeline blocks run — see below
LLM_GRAPH_FORMAT=jsonld                  # jsonld | turtle (legacy) — see below
ONTOLOGY_CHAPTER_FORMAT=inherit          # inherit | turtle | term_sheet: the facts prompts' ontology chapter only — see below
# ONTOLOGY_TEXT_MAX_CHARS_NAMING=80      # cap on labels / alt labels in that chapter (unset = as authored)
# ONTOLOGY_TEXT_MAX_CHARS_CONTRACT=240   # cap on scope notes / definitions
# ONTOLOGY_TEXT_MAX_CHARS_PROSE=160      # cap on rdfs:comment and other notes
# ONTOLOGY_TEXT_TOTAL_BUDGET=40000       # ceiling on all chapter text together
ONTOLOGY_CONTEXT_MODE=selected_single_ontology   # where per-unit schema comes from — see below
#ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID=catalog_iri_or_id_or_prefix
ONTOLOGY_CONTEXT_MAX_TRIPLES=4000        # prompt budget for the ontology chapter — see below
#ONTOLOGY_MAX_TRIPLES=                   # write-path growth backstop, NOT a context cap; unset = unlimited
PARALLEL_WORKERS=16                      # concurrent content-unit workers; startup warns when above LLM_MAX_INFLIGHT
ENABLE_ONTOLOGY_CONSOLIDATION=false      # optional post-normalization merge pass; inert for multi-ontology documents
# MAX_CONCURRENT_PROCESSES=4      # optional cap on simultaneous /process handlers
# MAX_TENANCY_SCOPES=16           # resident per-tenant/project ToolBoxes (LRU)
```

!!! warning "The server has no authentication"

    `HOST` defaults to `127.0.0.1`. There is no authentication, authorization,
    or request-size limit on any route, and `POST /flush` is destructive and
    can target any tenancy partition. Set `HOST=0.0.0.0` only behind a proxy
    that authenticates; the server logs a warning when you do.

!!! note "The two per-unit budgets are independent"

    `MAX_VISITS` retries a render that **failed**. A render that succeeded is
    never repeated: re-extracting a whole unit to fix a local defect is the
    expensive answer to a cheap question, and reliably introduces new defects.
    Raise it only if renders are failing outright.

    `FACTS_CRITIC_PASSES` (default `1`) buys review-and-patch passes. Each pass
    re-runs the deterministic checks for free, sends the graph and its findings
    to the critic, and applies what comes back as a compiled patch — no second
    call to carry out the fixes. So a facts unit costs **two** provider calls at
    the defaults, whatever `MAX_VISITS` is. Set `FACTS_CRITIC_PASSES=0` for
    exactly one call per unit.

    Neither budget multiplies the other. A pass also ends the loop early when it
    changed nothing or was rolled back, so raising the pass count buys passes
    only while they are still doing something. See
    [Validation](validation.md#how-many-llm-calls-a-facts-unit-really-costs).

    The ontology loop is the same loop, with `ONTOLOGY_CRITIC_PASSES` defaulting
    to `0`.

`MAX_CONCURRENT_PROCESSES` **queues** requests beyond the limit; they are not
rejected.

### Chunking

```bash
CHUNK_MIN_SIZE=3000                 # chars; floor before segments coalesce
CHUNK_MAX_SIZE=12000                # chars; ceiling before a section block is split
CHUNK_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2  # a 2nd resident model at this default
CHUNK_SEGMENTER=semantic            # semantic (sections-first, default) | docling
CHUNK_SECTION_CLASSIFIER=heuristic  # off | heading | heuristic (default) | llm
CHUNK_SECTION_DENSITY=conservative  # off | conservative (default) | aggressive
CHUNK_SECTION_TEXT_HEADINGS=true    # detect headings in documents with no markdown structure
CHUNK_SECTION_LLM_BATCH_SIZE=40     # excerpts per LLM call when classifier=llm; 0 = one call each
CHUNK_SECTION_TAG_MIN_CHARS=80      # min size for section tagging; smaller segments coalesce first
CHUNK_SECTION_SCHEMA_DETECT=headings  # off | lexical | headings (default) | auto
CHUNK_SECTION_SCHEMA_DETECT_MIN_SCORE=2.0           # evidence the winner must clear
CHUNK_SECTION_SCHEMA_DETECT_MIN_MARGIN=1.8          # factor over the runner-up
CHUNK_SECTION_SCHEMA_DETECT_CONTENT_MIN_MARGIN=4.0  # stricter margin for the content tier
CHUNK_SECTION_FILTER_ON_EMPTY=warn  # warn (default) | error
CHUNK_BIBLIOGRAPHY_MODE=skip        # skip | citations_only | domain_facts
CHUNK_MIN_UNIT_CHARS=0              # chars; drop units below this before the fan-out (0 = off)
CHUNK_NON_CONTENT_MODE=extract      # extract | skip — front/back-matter units (authors, notes, licence, …)
CHUNK_MAX_MEASUREMENTS_PER_UNIT=0   # split units stating more unit-adjacent numbers than this (0 = off)
```

`CHUNK_SEGMENTER=semantic` (default) detects section spans on the markdown
export, splits at section boundaries, and semantic-chunks within each
oversized section block, so chunks never straddle sections and inherit their
section label deterministically. `docling` uses Docling `HybridChunker`
structural segments instead.

`CHUNK_EMBEDDING_MODEL` is the sentence-transformers checkpoint used for
semantic chunking and embedding-based schema detection. It shares one
process-wide model with `EMBEDDING_MODEL_NAME` (retrieval) and
`AGG_EMBEDDING_MODEL` (entity disambiguation) whenever the names match — but it
defaults to a *different* checkpoint from the other two (mpnet-base, ~1.1 GB, vs
MiniLM ~458 MB), so **a default run holds two resident models**. Aligning all
three is the single-model, low-memory configuration:

```bash
# One resident local model instead of two.
CHUNK_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
AGG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Changing `CHUNK_EMBEDDING_MODEL` invalidates the on-disk chunk cache and shifts
chunk boundaries, which in turn shifts what each unit extracts. It also affects
the `CHUNK_SECTION_SCHEMA_DETECT_*` thresholds below, which are calibrated
against the default model's score distribution — re-derive them if you change
it. Retrieval and chunking dimensions are independent: the vector store's
dimension is fixed in its collection schema, the chunker's is not.

`CHUNK_MIN_UNIT_CHARS` drops content units shorter than the given number of
characters before the extraction fan-out; `0` (the default) disables the floor.
It is distinct from `CHUNK_MIN_SIZE`, which is a target the chunker aims at
while merging segments -- a heading stub or a caption fragment can still leave
chunking well under it. Every such unit costs a patch retrieval, a full render
and a critic call to extract from a fragment; the per-node durations and the
budget summary are what to size it against. Each drop is logged individually,
as with bibliography routing.

`CHUNK_NON_CONTENT_MODE` routes front and back matter: units headed author
information, contributions, notes, ORCID, data or code availability, competing
or conflicting interests, licence, open access, supporting information,
associated content, additional information, publisher's note, funding or
acknowledgements (also any unit whose section label is `acknowledgements`)
**that state no unit-adjacent number**; units whose tokens are mostly emails,
URLs, ORCIDs and initials; and short licence notices. A measurement anywhere in
the unit keeps it -- the heading list is evidence, not a verdict. `extract`
(the default) keeps such units and flags them `is_non_content`, which
downstream checks that assume domain prose read; `skip` drops them before the
fan-out. Routing order is `CHUNK_MIN_UNIT_CHARS` → bibliography → non-content,
every decision is logged, and the run manifest counts what each rule dropped
(`undersized_units_skipped`, `bibliography_units_skipped`,
`non_content_units_skipped`). Like bibliography detection it is
English/Latin-script bound -- elsewhere it only under-detects.

`CHUNK_MAX_MEASUREMENTS_PER_UNIT` splits a sized unit at the sentence or
paragraph boundary nearest its midpoint while it states more unit-adjacent
numbers than the cap, recursing on the pieces; `0` (the default) disables it.
Pieces inherit the unit's headings, references and section label and never
fall below `CHUNK_MIN_SIZE` (a unit shorter than twice the floor stays whole).
Unlike lowering `CHUNK_MIN_SIZE` / `CHUNK_MAX_SIZE`, it reaches only the units
where extraction has the most to lose -- a dense methods paragraph -- and
leaves sparse prose in large units where the ontology chapter is paid for
once. Each split is logged. A number counts when it stands next to a unit
token (`96 meV`, `77 K`, `0.5 %`), matched against the built-in unit lexicon
(`ontocast.util.measurement_lexicon`); a range yields one mention per number.

#### Section classification

`CHUNK_SECTION_CLASSIFIER` selects how far the classification cascade runs.
Each tier only sees what the previous one left unlabeled, so cost rises only
where cheaper evidence ran out:

| Value | Tiers | LLM calls |
|---|---|---|
| `off` | none — no tagging, and section filters and schema default exclusions are disabled | 0 |
| `heading` | document outline, heading patterns and keywords, order-guarded fill | 0 |
| `heuristic` **(default)** | the above plus content-density classification | 0 |
| `llm` | the above plus one batched call over whatever remains | ~1 per document |

Only `llm` costs anything during chunking. The default changed from `llm` in
0.5.x: the deterministic tiers now resolve the headings that previously needed
a model, so `--target-sections results` is free.

`CHUNK_SECTION_DENSITY` controls the content tier, which labels regions that
carry no usable heading. `conservative` (default) recognises only reference
lists and acknowledgements, whose surface form is near-unique. `aggressive`
also guesses methods/results/introduction from figure-reference, quantity and
citation densities — these signals do **not** separate those sections cleanly,
and a wrong label is acted on silently by the section filters, so it is opt-in.

`CHUNK_SECTION_TEXT_HEADINGS` enables heading detection from plain-text layout
(short, blank-line-delimited, upper-case or numbered lines) for documents whose
conversion produced no markdown heading structure at all.

#### Schema detection

The cascade above labels sections *within* a schema; `CHUNK_SECTION_SCHEMA_DETECT`
decides **which** schema, when the request supplies neither `section_schema_id`
nor a matching `document_type_hint`. Without it, a 10-Q submitted with no hint is
scored against the academic default and comes back entirely unlabeled.

| Value | Tiers | Cost |
|---|---|---|
| `off` | none — always the manifest default (`academic`) | 0 |
| `lexical` | headings only one schema recognises | 0, no model |
| `headings` **(default)** | the above plus embedding-based heading voting | reuses the chunker's model |
| `auto` | the above plus content classification on heading-poor documents | same model, more text |

`headings` is the default rather than `auto` because the content tier is
measurably unsafe. On the nine-document corpus it ranks 7/9 correctly, but its
one confident error is severe: chemistry body prose scores `standard` over
`academic` by more than the acceptance margin, so a heading-free paper would be
relabeled wholesale. It is therefore gated to documents with essentially no
headings, excludes `news` (a measured semantic attractor), and demands
`CONTENT_MIN_MARGIN` (4.0) rather than the heading tiers' 1.8. Enable `auto`
only for corpora of heading-free documents you have checked.

In practice the free tier does the work: all nine corpus cells and all five
in-repo documents resolve on `lexical` alone, so `headings` loads no model for
them. Lowering `MIN_SCORE` or `MIN_MARGIN` trades abstentions for confident
errors — the tightest correct margin in the corpus is a trial protocol at 2.0×
against `academic`, which is what the 1.8 default leaves room for.

Use `ontocast sections --input-path <file>` to see the resolved schema, the tier
that chose it, and the ranked candidate evidence, along with what the classifier
decided — see [Structured documents](concepts.md#structured-documents).

`CHUNK_BIBLIOGRAPHY_MODE` routes chunks detected as bibliography/reference
lists (via section label or citation-density heuristics): `skip` (default)
drops the chunks before extraction, `citations_only` extracts bibliographic
metadata only (`schema:ScholarlyArticle` + `schema:citation`, no domain facts
from citation titles), `domain_facts` restores the legacy behavior.

Request-level section filtering (`target_sections` allowlist,
`exclude_sections` denylist with per-schema defaults, `summarize_sections`)
is documented in [Structured documents](concepts.md#structured-documents).

#### Section filtering

An unrecognised section label is dropped with a warning, but a list where
**every** label is unrecognised is rejected outright (HTTP `422`, or a usage
error from the CLI). The two differ because their consequences do: a partly
recognised list still expresses an intent, while an empty result reads
downstream as an explicit "no sections" and therefore *replaces* the resolved
schema's `default_exclude` instead of adding to it. Failing there keeps a typo
from silently switching off section handling the caller never touched.

`CHUNK_SECTION_FILTER_ON_EMPTY` decides what happens when a section selection
removes **every** segment:

| Value | Behaviour |
|-------|-----------|
| `warn` (default) | Log a warning and continue. The run extracts zero chunks and reports success. |
| `error` | Fail the run: HTTP `422` with `error_code=empty_section_selection:<param>`, or a non-zero exit for `ontocast process` (the file is counted as failed; other files still run). |

The default is opt-in-safe but genuinely ambiguous: an empty result reads
exactly like a document that had nothing to extract. Use `error` when a
selection is expected to match — a typo in `target_sections`, or a document
whose headings did not classify as expected, is then a loud failure rather than
an empty graph.

It covers **both** directions: the `target_sections` / `summarize_sections`
allowlist and the `exclude_sections` denylist — including a schema's
`default_exclude`, which can empty a document with no caller involvement at
all. `ontocast sections` always behaves as `warn`, since a diagnostic has to
survive the condition it is diagnosing.

### Docling converter

Use these settings to tune Docling's standard document-conversion pipeline, especially for born-digital publisher PDFs where embedded ligatures can be split into patterns like `di ff usion`.

```bash
CONVERTER_PROFILE=default               # default | born_digital
# CONVERTER_PDF_BACKEND=docling_parse   # docling_parse | pypdfium2
# CONVERTER_DO_OCR=true
# CONVERTER_DO_TABLE_STRUCTURE=true
# CONVERTER_FORCE_BACKEND_TEXT=false
# CONVERTER_TABLE_CELL_MATCHING=true
# CONVERTER_LAYOUT_MODEL=heron          # heron | heron_101 | egret_medium | egret_large | egret_xlarge | v2
# CONVERTER_OCR_ENGINE=auto             # auto | easyocr | rapidocr | tesseract_cli | tesseract
# CONVERTER_OCR_LANG=
# CONVERTER_FORCE_FULL_PAGE_OCR=false
# CONVERTER_OCR_BITMAP_AREA_THRESHOLD=0.05
# CONVERTER_REPAIR_LIGATURE_GAPS=false  # on under profile=born_digital
# CONVERTER_REPAIR_NUMERIC_ARTIFACTS=false  # entities, CR wraps, flattened exponents; not part of born_digital
```

Recommended preset for publisher PDFs with selectable text:

```bash
CONVERTER_PROFILE=born_digital
```

That preset currently implies:

| Setting | Value |
|---------|-------|
| `CONVERTER_PDF_BACKEND` | `pypdfium2` |
| `CONVERTER_DO_OCR` | `false` |
| `CONVERTER_FORCE_BACKEND_TEXT` | `true` |
| `CONVERTER_REPAIR_LIGATURE_GAPS` | `true` |

Notes:

- `CONVERTER_REPAIR_LIGATURE_GAPS` repairs ASCII `fi` / `fl` / `ff` gap patterns that Docling passes through on some publisher PDFs. It is off by default but **on** under `CONVERTER_PROFILE=born_digital`, and it participates in the converter cache key, so flipping it re-converts. It becomes removable — as a breaking change — once Docling normalises these patterns upstream.
- `CONVERTER_REPAIR_NUMERIC_ARTIFACTS` repairs pattern-local artifacts that the export leaves inside values: the HTML entities `&lt;` `&gt;` `&amp;` `&quot;` `&apos;` (an explicit map, not a general unescape), carriage-return column wraps (`\r` plus inline whitespace before a newline; a blank line after it stays a paragraph break), flattened exponents (`2 × 10 6` → `2 × 10^6`, and a bare `10 6` only behind `~`, `≈`, `∼`, `≃` or "order of"), and single-sided ligature gaps with only one reading (a gap before `ff…`, a gap after a letter-preceded `fi`/`fl`; `the field`, `fl oz`, `cliff face` are left alone). It does **not** touch superscript/subscript duplication, citation markers fused into a value, or a gap before `fi`/`fl` -- those need layout evidence the exported text no longer carries. Off by default and not part of `born_digital`; it composes with `CONVERTER_REPAIR_LIGATURE_GAPS` (the two-sided rule runs first) and joins the converter cache key only when on, so every earlier conversion stays cached.
- Prefer `CONVERTER_PROFILE=born_digital` for text-selectable PDFs before trying heavier OCR settings.
- If OCR remains enabled and you pick `rapidocr`, set `CONVERTER_OCR_LANG=english` for English scans; RapidOCR's upstream default language is Chinese.

### Structured documents (per request)

No environment variables. Pass on `POST /process`, multipart form, JSON body, or CLI batch mode:

| Parameter | CLI flag | Description |
|-----------|----------|-------------|
| `target_sections` | `--target-sections` | Comma-separated or JSON list; enables tagging and keeps only these sections |
| `exclude_sections` | `--exclude-sections` | Comma-separated or JSON list; enables tagging and drops these sections |
| `summarize_sections` | `--summarize-sections` | Enables tagging + summarization; `*` or empty = all chunks |
| `summary_max_sentences` | `--summary-max-sentences` | Max sentences per summary (default `5`) |
| `max_visits` | `--max-visits` | Render/critic retry budget per loop (default from `MAX_VISITS`) |
| `section_schema_id` | `--section-schema-id` | Section label schema (`academic`, `financial`, `legal`, …) |
| `document_type_hint` | `--document-type-hint` | Free-text hint to resolve schema when `section_schema_id` is omitted |
| `document_metadata` | `--document-metadata` | JSON object of caller-asserted document identity (DOI/ISBN, ids, title, typed entities) — see [Concepts](concepts.md#document-level-identity-metadata) |

```bash
ontocast process --input-path ./papers/ \
  --output-dir ./out \
  --target-sections results,methods \
  --summarize-sections results \
  --summary-max-sentences 5
```

Details: [API Endpoints](api.md#post-process), [Workflow](workflow.md#2-chunking-and-optional-structured-preprocessing).

### Triple Stores

```bash
# Fuseki — dataset names default to ontocast--test--{facts,ontologies,shapes}
FUSEKI_URI=http://localhost:3030
FUSEKI_AUTH=admin/admin
#FUSEKI_DATASET=custom--project--facts
#FUSEKI_ONTOLOGIES_DATASET=custom--project--ontologies
#FUSEKI_SHAPES_DATASET=custom--project--shapes
```

SHACL shapes get a dataset of their own: catalog discovery claims every named
graph carrying an `owl:Ontology` subject, and a shapes document declares one.
See [Validation](validation.md#why-shapes-are-not-stored-with-the-ontologies).

See [Tenancy](tenancy.md) for how tenant/project names relate to dataset, collection, and table names.

### Embeddings

```bash
EMBEDDING_PROVIDER=huggingface          # huggingface | openai | ollama
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# EMBEDDING_API_KEY=
# EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_DIMENSION=384
# EMBEDDING_QUERY_PREFIX=
# EMBEDDING_DOCUMENT_PREFIX=
```

**Query/document prefixes.** Asymmetric retrieval models are trained with a distinct
instruction on each side and lose accuracy when query and document are encoded
identically. The default paraphrase model is symmetric and wants neither prefix.

| Model family | `EMBEDDING_QUERY_PREFIX` | `EMBEDDING_DOCUMENT_PREFIX` |
|---|---|---|
| `paraphrase-*` (default) | *(empty)* | *(empty)* |
| `BAAI/bge-*` | `Represent this sentence for searching relevant passages: ` | *(empty)* |
| `intfloat/e5-*` | `query: ` | `passage: ` |

Both prefixes are part of the stored embedding contract, so changing either invalidates
an existing index: rerun with `--wipe-vector-store` (or `VECTOR_STORE_WIPE_ON_INIT=true`).
A mismatch fails loudly with `EmbeddingContractMismatchError` rather than quietly
degrading retrieval.

### Qdrant

```bash
QDRANT_URI=http://localhost:6333
QDRANT_API_KEY=abc123-qwe
QDRANT_GRPC_PORT=6334
QDRANT_USE_GRPC=false
QDRANT_TIMEOUT_SECONDS=30                # whole seconds; the client accepts nothing finer
# QDRANT_ONTOLOGY_COLLECTION=ontocast--test--ontologies
# QDRANT_FACTS_COLLECTION=ontocast--test--facts
```

### Vector store (backend-agnostic)

Applies to both Qdrant and LanceDB:

```bash
# auto (default): Qdrant if QDRANT_URI is set, LanceDB if enabled, otherwise
# vector retrieval is disabled. Explicit: qdrant | lancedb | none.
# VECTOR_STORE_BACKEND=auto
VECTOR_STORE_TOP_K=20
VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH=2
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=1200
VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY=24
# VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT=16
# VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH=3
# VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN=false
# VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR=1.0
# VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER=score
# VECTOR_STORE_PROPOSITION_WINDOW_SENTENCES=2
# VECTOR_STORE_PROPOSITION_MAX_WINDOWS=16
# VECTOR_STORE_PROPOSITION_RETRIEVAL_ENABLED=true
# VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE=0.5
# VECTOR_STORE_FUSION_CORE_WEIGHT=0.7
# VECTOR_STORE_FUSION_NEIGHBORHOOD_WEIGHT=0.15
# VECTOR_STORE_FUSION_BM25_WEIGHT=0.8
# VECTOR_STORE_INDEX_UNDESCRIBED_IRIS=false
# VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS=false
# VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES=
# VECTOR_STORE_DEDUP_MODE=iri               # identity used when de-duplicating indexed atoms
# VECTOR_STORE_EMBEDDING_BATCH_SIZE=64
# VECTOR_STORE_REINDEX_CONCURRENCY=2
# VECTOR_STORE_WIPE_ON_INIT=false
# VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT=true
```

| Variable | Default | Role |
|----------|---------|------|
| `VECTOR_STORE_TOP_K` | `20` | Vector hits per channel per proposition window |
| `VECTOR_STORE_INDUCED_SUBGRAPH_DEPTH` | `2` | BFS depth for hub seed expansion |
| `VECTOR_STORE_INDUCED_SUBGRAPH_HUB_SEED_COUNT` | `16` | Top seeds that receive full BFS budget (`0` = all seeds) |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ANCESTOR_CLOSURE_DEPTH` | `3` | `rdfs:subClassOf` hops in the schema shell |
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `1200` | Global triple cap returned to the LLM. The binding constraint in practice — every seed-side knob is flat while this is low |
| `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` | `24` | Per-entity BFS quota hint during retrieval |
| `VECTOR_STORE_INDUCED_SUBGRAPH_CANDIDATE_PUSHDOWN` | `false` | Opt-in SPARQL `CONSTRUCT` neighborhood instead of merging whole ontology graphs (see [Ontology Context](ontology_context.md#candidate-pushdown-opt-in)) |
| `VECTOR_STORE_INDUCED_SUBGRAPH_TYPE_PROMOTION_SCORE_FACTOR` | `1.0` | Fraction of a retrieved seed's score inherited by its promoted `rdf:type` IRIs; the seed always keeps its own score |
| `VECTOR_STORE_INDUCED_SUBGRAPH_SEED_ORDER` | `score` | Seed expansion order under the triple budget: `score` (global relevance) or `ontology_round_robin` (interleave source ontologies) |
| `VECTOR_STORE_INDUCED_SUBGRAPH_SYMBOL_PREDICATES` | trigger predicates | Symbol/notation predicates admitted as seed descriptions between names and glosses (empty list disables). This is the *retrieval* half; keep it in agreement with `VECTOR_STORE_SYMBOL_PREDICATES` below |
| `VECTOR_STORE_LABEL_PREDICATES` | `rdfs:label`, `skos:prefLabel`, `dcterms:title`, `skos:altLabel`, `dcterms:alternative` | Predicates whose literals are **indexed** as declared labels, in descending priority. Changing this requires a reindex |
| `VECTOR_STORE_SYMBOL_PREDICATES` | `skos:notation`, `qudt:symbol`, `qudt:ucumCode` | Predicates whose literals are **indexed** as symbols, collected against their own budget so multilingual labels cannot crowd them out. The indexing half of the pair above; configuring only the retrieval half changes what surfaces without changing what is stored. Changing this requires a reindex |
| `VECTOR_STORE_PROPOSITION_WINDOW_SENTENCES` | `2` | Sentences per proposition window for multi-query retrieval |
| `VECTOR_STORE_PROPOSITION_MAX_WINDOWS` | `16` | Cap on windows per excerpt; when a chunk has more, windows are sampled at an even stride spanning both endpoints (not “first N only”) |
| `VECTOR_STORE_PROPOSITION_RETRIEVAL_ENABLED` | `true` | Multi-query proposition retrieval for induced-graph mode |
| `VECTOR_STORE_CONSISTENCY_CRITIC_MIN_FUSED_SCORE` | `0.5` | Min weighted reciprocal-rank score for the consistency critic to flag a cross-ontology conflict (not cosine). Renamed from `VECTOR_STORE_CONSISTENCY_CRITIC_SIMILARITY_THRESHOLD` (old default `0.7`) |
| `VECTOR_STORE_FUSION_CORE_WEIGHT` | `0.7` | Dense core-vector weight in rank fusion (weights are normalized, so only ratios matter) |
| `VECTOR_STORE_FUSION_NEIGHBORHOOD_WEIGHT` | `0.15` | Dense neighborhood-vector weight; the neighborhood text describes a term's edges, so it corroborates the core lane more than it adds to it |
| `VECTOR_STORE_FUSION_BM25_WEIGHT` | `0.8` | Sparse BM25 weight. A term whose surface form is a symbol (`meV`, a chemical formula) is often invisible to the dense lanes, so the sparse lane is its only evidence — see [Ontology Context](ontology_context.md#bm25-index-recreate) |
| `VECTOR_STORE_INDEX_UNDESCRIBED_IRIS` | `false` | Atomize IRIs an ontology only *references* (object/predicate position) in addition to ones it describes. Reindex on change |
| `VECTOR_STORE_EMBED_STANDARD_VOCAB_IRIS` | `false` | Atomize RDF/OWL/SKOS/DC/SHACL/schema.org IRIs instead of skipping them. Reindex on change |
| `VECTOR_STORE_EXTRA_EXCLUDED_NAMESPACE_PREFIXES` | *(empty)* | Extra IRI prefixes never atomized from ontology sources, on top of the standard-vocabulary set. Reindex on change |
| `VECTOR_STORE_EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding request during ontology indexing (raise for remote APIs; lower if VRAM-bound) |
| `VECTOR_STORE_REINDEX_CONCURRENCY` | `2` | Max ontologies materialized/reindexed in parallel at `ToolBox.initialize` |
| `VECTOR_STORE_WIPE_ON_INIT` | `false` | Drop the current tenant/project vector partition before recreate+reindex (clean slate; also CLI `--wipe-vector-store`) |
| `VECTOR_STORE_PRUNE_ORPHAN_IRIS_ON_INIT` | `true` | Delete indexed ontology IRIs absent from the synchronized catalog (covers IRI renames without a full wipe) |
| `VECTOR_STORE_LEXICAL_TRIGGER_ENABLED` | `true` | Exact-match lane for notation/symbol tokens in raw chunk text |
| `VECTOR_STORE_LEXICAL_TRIGGER_PREDICATES` | `skos:notation`, `qudt:symbol`, `qudt:ucumCode` | Predicate IRIs whose literals become case-preserved triggers |
| `VECTOR_STORE_LEXICAL_TRIGGER_HEURISTIC_ENABLED` | `true` | Promote code-shaped labels/altLabels when no notation is declared |
| `VECTOR_STORE_LEXICAL_TRIGGER_MIN_LEN` | `2` | Minimum length for heuristic promotion |
| `VECTOR_STORE_LEXICAL_TRIGGER_MAX_LEN` | `24` | Maximum length for heuristic promotion |
| `VECTOR_STORE_LEXICAL_TRIGGER_HEURISTIC_MAX_PER_ENTITY` | `2` | Cap on heuristic triggers per entity |
| `VECTOR_STORE_LEXICAL_TRIGGER_MAX_ATOMS` | `16` | Additive cap on trigger seeds per retrieval call (outside semantic budget) |
| `VECTOR_STORE_LEXICAL_TRIGGER_SCORE` | `0.35` | Score assigned to trigger hits (calibrated against fused rank scores: rank-1 core = 0.583, merged floor = 0.18) |
| `VECTOR_STORE_LEXICAL_TRIGGER_FUSION` | `max_merge` | `max_merge` promotes an already-retrieved atom to `max(semantic, trigger)` score; `append` (legacy) only adds unseen atoms |
| `VECTOR_STORE_QUERY_UNIT_SIGNALS_ENABLED` | `false` | Match number-adjacent tokens ("4-15 days", "200 kV", "0.5 %") case-insensitively and plural-tolerantly against catalog labels/symbols/UCUM codes; matched entities join the snapshot seeds at trigger score, outside the semantic budget |
| `VECTOR_STORE_SYMBOL_CASE_MISMATCH_POLICY` | `demote` | Merge-time treatment of atoms whose declared symbol surfaces (`skos:notation`, `qudt:symbol`, `qudt:ucumCode`) match a query token only case-insensitively with no exact-case match anywhere — the BM25/dense text is case-folded, so prose "meV" also retrieves `unit:MegaEV` (symbol "MeV"). `demote` multiplies the atom score, `drop` removes it, `off` keeps legacy behavior; exact-case and label-only matches are never touched |
| `VECTOR_STORE_SYMBOL_CASE_MISMATCH_DEMOTE_FACTOR` | `0.5` | Score multiplier applied under the `demote` policy |
| `FACTS_OBJECT_PROPERTY_LITERAL_CHECK` | `true` | Quarantine string literals on predicates whose schema range is a class (e.g. `qudt:unit`); surfaced to the facts critic and the deterministic repair loop |
| `FACTS_CRITIC_PASSES` | `1` | Review-and-patch passes per unit, **in provider calls**. Each pass re-runs the deterministic checks for free, sends the graph and its findings to the critic, and applies what comes back as a compiled patch. `0` leaves the residue to the LLM-free repairs and the gate |
| `FACTS_LLM_REPAIR_VISITS` | unset | Deprecated alias for `FACTS_CRITIC_PASSES`, honoured for one release |
| `FACTS_CRITIC_MAX_DELETE_SHARE` | `0.25` | Largest share of a unit graph one pass may remove; past it the pass keeps its inserts and drops its deletes |
| `FACTS_CRITIC_MIN_DELETES` | `5` | Deletions always permitted regardless of share, so short units stay correctable |
| `FACTS_CRITIC_ALLOW_SUBJECT_RENAME` | `false` | Whether a `REPLACE` may delete about one subject while writing about another |
| `FACTS_COMPLETION_PASSES` | `0` | Insert-only completion passes per unit, **in provider calls**, run after the critic loop above and only while the numeric-coverage inventory still lists a measurement absent from the graph. Shown a compact term sheet and the unit's existing catalog-typed subjects instead of the full ontology chapter; every proposed fix is `action=ADD` and goes through the same per-subject regression check a critic fix does. `0` (default) disables it — see [Validation](validation.md#completion-pass-insert-only-recovery) |
| `ONTOLOGY_CRITIC_PASSES` | `0` | The same budget for ontology units, opt-in |
| `ONTOLOGY_CRITIC_MAX_DELETE_SHARE` | `0.10` | Stricter than the facts side: an ontology delete propagates onto shared catalog terminals |
| `ONTOLOGY_CRITIC_MIN_DELETES` | `3` | Deletions always permitted regardless of share |
| `ONTOLOGY_ACCEPT_BLOCKING_FINDING_KINDS` | destructive subset | Deterministic ontology findings that block acceptance |
| `FACTS_PROPERTY_ALIAS_MIN_RATIO` | `0.95` | Tie-break floor for deterministic near-miss property rewrites in catalog namespaces. Only token containment or folded equality qualifies a candidate (`qudt:value` → `qudt:numericValue`); the ratio decides between several qualifying candidates and never licenses a rewrite alone. Predicates present anywhere in the full catalog are left untouched |
| `FACTS_MERGE_REPAIR_PASSES` | `1` | Un-merge budget at the post-aggregation `VALIDATE_FACTS` gate: *merge-signature* error findings (functional violation, suspect multi-value, degenerate coreference) on merged subjects become full-cluster pair vetoes and the facts units are re-aggregated. `0` records findings without repairing. SHACL findings never drive it |
| `FACTS_CODE_PREDICATES` | `qudt:ucumCode`, `qudt:symbol`, `skos:notation` | Predicates whose literal objects are machine-resolvable codes. A node carrying `qudt:ucumCode "d"` but no unit link gains the object property pointing at the catalog individual declaring that code, when exactly one does. Exact and case-sensitive — these are codes, not labels |
| `FACTS_SUSPECT_MULTI_VALUE_SEVERITY` | `error` | Severity of SUSPECT_MULTI_VALUE gate findings (multiple distinct numeric values on one predicate; mutually irreconcilable short string values on a dominantly string-single-valued predicate; or multiple objects on a dominantly single-valued predicate); only `error` findings drive the un-merge repair |
| `FACTS_LITERAL_VARIANT_DEDUPE` | `true` | Collapse duplicate literals differing only in language tag or datatype on one (subject, predicate) before validation — `"X"@en` alongside `"X"^^xsd:string` alongside `"X"`. The language-tagged form wins, then the plain form; reified provenance follows the survivor. Each removal is a `literal_variant_pruned` repair record |
| `FACTS_SHAPES_DIR` | — | **Seed** directory of SHACL shape files (searched recursively), materialized at startup into the tenant's `{tenant}--{project}--shapes` partition — the same read-only bootstrap contract `ONTOCAST_ONTOLOGY_DIRECTORY` has. The gate validates against the partition, so shapes uploaded over `POST /shapes` are equally in force and a container needs no shapes directory. `sh:NodeShape` triples inlined in the ontology context are picked up automatically. SHACL runs only when `pyshacl` is installed (`uv sync --extra shacl`). Setting this without the extra, or pointing it at an empty directory, logs a **warning** — it never passes silently; a directory that does not exist stops `ontocast process` outright. `--shapes-dir` overrides it per run |
| `FACTS_SHACL_INFERENCE` | `rdfs` | Pre-inference for the SHACL run: `none`, `rdfs`, `owlrl`. RDFS by default because SHACL property paths carry no `rdfs:subPropertyOf` entailment, so a shape naming a superproperty reports the specialised predicate the renderer emitted as missing |
| `FACTS_SHACL_ADVANCED` | `true` | Enable SHACL Advanced Features (`sh:sparql` constraints, node expressions) |
| `FACTS_SHACL_MAX_TRIPLES` | `200000` | Skip SHACL with a warning above this graph size; `0` disables the guard |
| `FACTS_SHACL_AUTOFIX` | `prune` | LLM-free repair of SHACL violations at the gate. `rewrite` retypes literals against `sh:datatype` and resolves a literal to the unique catalog IRI declaring it; `prune` additionally drops `sh:minCount` violators that assert nothing beyond `rdf:type`/`rdfs:label`; `off` reports only. Nothing is ever invented — see [Validation](validation.md#llm-free-autofix) |
| `FACTS_SHACL_AUTOFIX_PASSES` | `1` | Bounded validate → repair → revalidate rounds; a pass that does not strictly reduce violations is reverted |
| `FACTS_FUNCTIONAL_MIN_SINGLE_SUPPORT` | `3` | Distinct single-valued subjects a predicate needs before the gate treats it as empirically functional. Below this the evidence is too thin to call a second value a violation |
| `FACTS_QUANTITY_FALLBACK_VOCABULARY` | QUDT | Role → term mapping the facts prompt names as the fallback for bounded/approximate quantities when retrieval supplied no suitable class. Roles: `value_class`, `numeric_value`, `unit`, plus optional `lower_bound`/`upper_bound` (and roles containing `inclusive`) naming the catalog's range properties. Override for catalogs modelling quantities with another vocabulary; set to `{}` to forbid the fallback entirely and keep the renderer inside the provided context. Terms in a configured fallback namespace are reported by `NON_CATALOG_VOCABULARY` as a *deliberate* fallback, and the configured terms are exempt from `UNKNOWN_TERM`. When all of `numeric_value`/`lower_bound`/`upper_bound` are set, equal-bound pairs are promoted to a single scalar at parse time; the `unit` role drives the `LABEL_ONLY_NUMBER` finding — see [Validation](validation.md#which-terms-count-as-unknown) |
| `FACTS_DOMAIN_ADHERENCE_MIN_SHARE` | `0.15` | Floor on the fraction of a render's distinct schema terms (predicates and `rdf:type` objects, excluding minted instances and RDF/RDFS/OWL/XSD/SKOS/DC/PROV plumbing) that must come from the unit's ontology context. Below it a mandatory `DOMAIN_ADHERENCE` finding asks for a rewrite. Catches what no per-triple check can see: a render that says everything in a generic vocabulary is well-formed term by term, exempt from `UNKNOWN_TERM`, and matched by no shape — it reads as extracted while answering nothing. `0` disables; leave it disabled if you extract without a catalog |
| `FACTS_ADDITIONAL_STANDARD_NAMESPACES` | schema.org | Namespaces exempt from `UNKNOWN_TERM` beyond the RDF/OWL substrate and annotation/provenance terms. Only meta-vocabularies are built in; a domain vocabulary shared across catalogs (SOSA/SSN, CSVW, FOAF, Dublin Core profiles) is exempted here. schema.org is the default because the shipped citation vocabulary uses it |
| `CHUNK_CITATION_VOCABULARY` | schema.org | Role → term mapping used by the citation-metadata prompt in `citations_only` mode. Bibliographic entries are not domain content, so unlike the rest of the pipeline these terms are configuration rather than retrieval. Roles: `work_class`, `fallback_class`, `title`, `author`, `author_name`, `date_published`, `venue`, `identifier`, `cites` |

**Migration note:** retrieval knobs formerly named `QDRANT_TOP_K`, `QDRANT_INDUCED_SUBGRAPH_*`, etc. are **ignored**. Use `VECTOR_STORE_*`. `QDRANT_*` covers connection/transport only.

**BM25 / sparse schema:** Qdrant collections must declare IDF sparse vectors and index label-enriched text. A stale collection fails with `EmbeddingContractMismatchError` — recreate via `VECTOR_STORE_WIPE_ON_INIT=true` or `--wipe-vector-store`. The surface-form contract is now `sf4` (ontology sources atomize only IRIs they describe; `sf3` added `lexical_triggers` for the exact-match lane). Every collection built under `sf3` or earlier needs one reindex.

Catalog graphs are served through `OntologyManager` (see [Ontology Catalog](../architecture/ontology_catalog.md)).

### LanceDB (embedded alternative)

Enable when `QDRANT_URI` is unset. Requires the optional extra: `uv sync --extra lancedb`.

```bash
LANCEDB_ENABLED=true
# LANCEDB_DATA_DIR=~/.lancedb_data          # on-disk location of the embedded tables
# Table names derive from tenant/project when unset (ontocast--test--ontologies / --facts)
# LANCEDB_ONTOLOGY_TABLE=
# LANCEDB_FACTS_TABLE=
```

`QDRANT_URI` and `LANCEDB_ENABLED=true` cannot both be set.

### Backend storage details

Rarely changed; defaults suit both backends.

```bash
# Qdrant collection geometry — must match the embedding model
# QDRANT_VECTOR_SIZE=384                     # must equal EMBEDDING_DIMENSION
# QDRANT_DISTANCE=Cosine                     # Cosine | Dot | Euclid
# QDRANT_UPSERT_BATCH_SIZE=256               # points per upsert call during indexing

# Partition (table/collection) names within the configured backend
# (derived from tenant/project when unset: ontocast--test--ontologies / --facts)
# VECTOR_STORE_ONTOLOGY_TABLE=
# VECTOR_STORE_FACTS_TABLE=

# Sparse (BM25) model for the lexical lane
# EMBEDDING_BM25_MODEL_NAME=Qdrant/bm25

# Atom de-duplication identity
# VECTOR_STORE_DEDUP_INCLUDE_VERSION=true    # treat ontology versions as distinct
# VECTOR_STORE_DEDUP_INCLUDE_HASH=true       # treat content hashes as distinct
# VECTOR_STORE_DEDUP_QUERY_HITS_BY_IRI=true  # collapse repeat hits on one IRI

# Prompt/diagnostic shaping
# VECTOR_STORE_MINIMAL_LABEL_LIMIT=...       # cap on label-only atom text
# ONTOLOGY_PATCH_DUMP_ONTOLOGY_RANKS=false   # dump per-ontology retrieval ranks
```

Changing `QDRANT_VECTOR_SIZE` or `EMBEDDING_BM25_MODEL_NAME` invalidates an
existing collection — reindex with `VECTOR_STORE_WIPE_ON_INIT=true`.

Budget behavior:

- `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` is the global upper bound returned to the LLM.
- `VECTOR_STORE_INDUCED_SUBGRAPH_ESTIMATED_TRIPLES_PER_QUERY` shapes per-entity allocation during retrieval.

!!! warning "Retrieval defaults are single-corpus fits"

    The shipped retrieval defaults — notably
    `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` (raised 550 → 1200),
    `PER_ONTOLOGY_ATOM_FLOOR`, `PER_ROLE_ATOM_FLOOR`, and
    `SCHEMA_CLOSURE_MAX_ENTITIES` — were tuned one axis at a time against a
    single catalog, and the triple budget more than doubles prompt-context cost
    per unit. Treat them as a starting point and re-sweep for your own catalog;
    each field's description in `ontocast/config/settings.py` records what it
    controls.

See [Ontology Context](ontology_context.md) for vector-search mode requirements.

### Ontology Patch Retrieval

Post-vector scoring and capping (backend-agnostic; prefix `ONTOLOGY_PATCH_`). Applied after hybrid dense + BM25 retrieval, before induced-subgraph expansion.

**Default path** (simple): max-score IRI dedupe → global score order → window-scaled hard cap. A non-zero `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` inserts per-ontology round-robin (best-scoring ontologies first) instead of plain score order. Relative floors, hybrid tier merge, merged-score ratio, and MMR are advanced opt-in.

```bash
ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=max_score
ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA=0
ONTOLOGY_PATCH_SEEDS_PER_WINDOW=4
ONTOLOGY_PATCH_MAX_ATOMS_BASE=96
ONTOLOGY_PATCH_MAX_ATOMS=96
ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE=0.18
ONTOLOGY_PATCH_MMR_LAMBDA=1.0
# Advanced (off by default):
# ONTOLOGY_PATCH_PER_QUERY_CORE_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_PER_QUERY_NEIGHBORHOOD_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_PER_QUERY_BM25_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.0
# ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE=hybrid
# ONTOLOGY_PATCH_MAX_ATOMS_TIER1=12
# ONTOLOGY_PATCH_MIN_ENTITY_SCORE=0.3
```

| Variable | Default | Role |
|----------|---------|------|
| `ONTOLOGY_PATCH_CROSS_QUERY_MERGE_MODE` | `max_score` | Default merge; `sum_score` sums per-window scores; `hybrid` is tier-1 + tier-2 |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_SEED_QUOTA` | `0` | Max seeds per ontology in round-robin fill; `0` (default) means global score order, which measured better on both recall and precision |
| `ONTOLOGY_PATCH_SEEDS_PER_WINDOW` | `4` | Scales effective cap with proposition window count |
| `ONTOLOGY_PATCH_MAX_ATOMS_BASE` | `96` | Floor for effective atom cap; below the candidate pool it silently clips seeds on multi-ontology catalogs |
| `ONTOLOGY_PATCH_MAX_ATOMS` | `96` | Hard cap: `min(max_atoms, max(base, seeds_per_window × n_queries))` (`0` = unlimited) |
| `ONTOLOGY_PATCH_MIN_MERGED_MAX_SCORE` | `0.18` | Empty patch when the best **per-window** fused score is below this (`0` disables); evaluated before cross-window merge |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `1.0` | `1.0` skips MMR; lower values enable diversity rerank |
| `ONTOLOGY_PATCH_PER_QUERY_CORE_SCORE_RATIO` | `0.0` | Advanced: per-window core relative floor (`0` disables) |
| `ONTOLOGY_PATCH_PER_QUERY_NEIGHBORHOOD_SCORE_RATIO` | `0.0` | Advanced: per-window neighborhood relative floor |
| `ONTOLOGY_PATCH_PER_QUERY_BM25_SCORE_RATIO` | `0.0` | Advanced: per-window BM25 relative floor |
| `ONTOLOGY_PATCH_MIN_CORE_QUERY_BEST_SCORE` | `0.0` | If `> 0`, windows whose top core score is below this contribute no core hits |
| `ONTOLOGY_PATCH_MIN_NEIGHBORHOOD_QUERY_BEST_SCORE` | `0.0` | If `> 0`, windows whose top neighborhood score is below this contribute no neighborhood hits |
| `ONTOLOGY_PATCH_MIN_BM25_QUERY_BEST_SCORE` | `0.0` | If `> 0`, windows whose top BM25 score is below this contribute no BM25 hits |
| `ONTOLOGY_PATCH_MERGED_SCORE_RATIO` | `0.0` | Advanced: drop seeds below `top_score × ratio` (`0` disables) |
| `ONTOLOGY_PATCH_MAX_ATOMS_TIER1` | `12` | Hybrid only: global tier-1 cap (`0` = no cap) |
| `ONTOLOGY_PATCH_MIN_ENTITY_SCORE` | `0.3` | Hybrid only: tier-2 minimum fused score |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR` | `2` | Reserve pass before the global fill: every contributing ontology is guaranteed `min(floor, its candidates)` seed slots (round-robin). Unlike the quota (a ceiling), the floor protects small modules from starvation at the atom cap. `0` disables |
| `ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` | `300` | Include a source ontology's whole header-stripped graph in the snapshot when it has ≥ 1 admitted atom and at most this many triples (prevents near-miss property improvisation on tiny vocabularies). `0` disables |
| `ONTOLOGY_PATCH_PER_ROLE_ATOM_FLOOR` | `12` | Reserve pass for predicate-role atoms before the global fill. Prose reads as noun phrases, so classes out-score the properties that link them in a shared ranking. `0` disables |
| `ONTOLOGY_PATCH_SCHEMA_CLOSURE_MAX_ENTITIES` | `32` | Cap on terms admitted by `rdfs:domain`/`rdfs:range` closure over the seeds: properties whose domain/range names an admitted class (or an ancestor), plus the domain/range classes of admitted properties. `0` disables |
| `ONTOLOGY_PATCH_SCHEMA_CLOSURE_ANCESTOR_DEPTH` | `2` | How far to walk `rdfs:subClassOf` upward when matching a property's declared domain/range against an admitted class |

#### Retrieval tuning: what each knob does

The ranges below are starting points, not results. Tune against
`catalog_context_triples` and the snapshot your own renderer receives; the
retrieval metrics in [Observability](observability.md#retrieval-metrics) are
what to read while doing so.

| Variable | Default | Useful range | Effect |
|---|---|---|---|
| `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` | `1200` | 1000–1600 | **Gates everything.** Set too low, every knob below is flat, because the snapshot is already pinned at the cap. Raise this first; it saturates |
| `ONTOLOGY_PATCH_SMALL_MODULE_CLOSURE_MAX_TRIPLES` | `300` | 250–400 | The largest single lever: admits a small module whole once it has won a seed. Set it above the largest module you need entire; going past that adds triples for little gain |
| `ONTOLOGY_PATCH_PER_ONTOLOGY_ATOM_FLOOR` | `2` | 2–4 | Saturates quickly. **The closure above is inert without this** — a module must win at least one seed before its graph is considered |
| `ONTOLOGY_PATCH_SCHEMA_CLOSURE_MAX_ENTITIES` | `32` | 16–48 | Admits properties whose domain/range names an admitted class. Saturates well before the top of the range |
| `ONTOLOGY_PATCH_PER_ROLE_ATOM_FLOOR` | `12` | 8–16 | Weak on its own; contributes once the schema closure is on |
| `ONTOLOGY_PATCH_MAX_ATOMS` / `_BASE` | `96` | 96–192 | Trades directly against context size once the triple budget is not binding |
| `VECTOR_STORE_TOP_K` | `20` | — | **Insensitive** (10–40 all within 1 point). Effectively capped by `MAX_ATOMS_BASE`; leave alone |
| `ONTOLOGY_PATCH_MMR_LAMBDA` | `1.0` | — | **Insensitive** (0.5–1.0 identical on this corpus). Leave alone unless you see near-duplicate terms crowding the snapshot |

Combined at the defaults: needed-term recall 11/11, declared-property coverage
82%, at roughly 2.7× the snapshot size of the old settings. That size increase is
the cost of the fix — if prompt budget matters more than recall for your corpus,
lower `MAX_TOTAL_TRIPLES` first and accept the coverage loss.

**Tighter preset** (optional precision knobs for noisy catalogs — see [Ontology Context](ontology_context.md)):

```bash
ONTOLOGY_PATCH_MAX_ATOMS=32
ONTOLOGY_PATCH_MERGED_SCORE_RATIO=0.5
ONTOLOGY_PATCH_MMR_LAMBDA=0.85
VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES=600
```

### Paths and Domain

```bash
CURRENT_DOMAIN=https://example.com               # base for minted IRIs; also the default facts namespace
ONTOCAST_ONTOLOGY_DIRECTORY=/path/to/ontology/files   # seed .ttl files, synced to the catalog on startup (CLI: --ontology-dir)
ONTOCAST_CACHE_DIR=/path/to/cache/directory      # LLM + converter disk cache root

# Cache eviction. The cache bounds itself: once it exceeds the ceiling,
# least-recently-used entries are deleted. Set to 0 to disable.
# Accepts a byte count or a human size ("1GB", "500MB").
ONTOCAST_CACHE_MAX_BYTES=1GB
ONTOCAST_CACHE_TTL_DAYS=30
ONTOCAST_CACHE_PRUNE_EVERY=256
```

See [LLM Caching](llm_caching.md#cache-size-and-eviction) for how eviction is
scheduled and the `ontocast cache` commands that drive it manually.

### Aggregation

```bash
AGG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
AGG_SIMILARITY_THRESHOLD=0.80          # EntityAligner only (align_entities, match-graphs)
AGG_CANDIDATE_SIMILARITY_THRESHOLD=0.70  # the in-pipeline aggregator's threshold
AGG_LEXICAL_LABEL_JACCARD=0.5
AGG_LEXICAL_SEQUENCE_RATIO=0.90
AGG_LEXICAL_TOKEN_JACCARD=0.75
AGG_FUNCTIONAL_MIN_EMPIRICAL_SUPPORT=2
AGG_SIBLING_GUARD_SCOPE=subject
AGG_LITERAL_CONFLICT_GUARD=true
AGG_INITIALS_DISTINCT_GUARD=true
AGG_NATURAL_KEY_MERGE=true
AGG_TYPE_GUARD_UNTYPED=permissive
AGG_UNIT_SCOPED_FACT_IRIS=true           # identity across units is decided by the aggregator, not by name
```

See [Aggregation](aggregation.md) for what each threshold and guard does.

### Web Search

Search is "search-later": nodes run without search first, and only request
external evidence when needed. **The whole block is inert while
`WEB_SEARCH_ENABLED=false`** (the default) — the remaining twenty variables
describe a lane that does not run until you turn it on.

| Variable | Default | Role |
|----------|---------|------|
| `WEB_SEARCH_ENABLED` | `false` | Master switch. Node execution still starts without search and only searches when node output requests it |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | Search provider |
| `WEB_SEARCH_TOP_K` | `3` | Results fetched per query (1–10) |
| `WEB_SEARCH_TIMEOUT_SECONDS` | `8.0` | Per-request search timeout (1.0–60.0) |
| `WEB_SEARCH_MAX_SNIPPET_CHARS` | `400` | Snippet truncation limit per hit (80–2000) |
| `WEB_SEARCH_MIN_SNIPPET_CHARS` | `40` | Minimum snippet length to keep a hit at all |
| `WEB_SEARCH_MAX_TOTAL_CHARS` | `1800` | Total evidence text budget across hits (200–10000) |

**Which nodes may search.** Each gate allows *search-eligible retries* for that prompt; the first pass is always no-search. Ontology nodes are on, facts nodes are off — facts are meant to come from the document, not the web.

| Variable | Default | Role |
|----------|---------|------|
| `WEB_SEARCH_ONTOLOGY_RENDER_ENABLED` | `true` | Ontology render retries may search |
| `WEB_SEARCH_ONTOLOGY_CRITIC_ENABLED` | `true` | Ontology critic retries may search |
| `WEB_SEARCH_FACTS_RENDER_ENABLED` | `false` | Facts render retries may search |
| `WEB_SEARCH_FACTS_CRITIC_ENABLED` | `false` | Facts critic retries may search |

**Query planner and guardrails.**

| Variable | Default | Role |
|----------|---------|------|
| `WEB_SEARCH_PLANNER_ENABLED` | `true` | Use an LLM planner to decide what to search for |
| `WEB_SEARCH_PLANNER_MAX_QUERIES` | `3` | Focused queries per node (1–8) |
| `WEB_SEARCH_PLANNER_MIN_QUERY_CHARS` | `12` | Minimum query length accepted by guardrails |
| `WEB_SEARCH_PLANNER_MIN_CONFIDENCE` | `0.35` | Planner confidence below which no search runs |
| `WEB_SEARCH_REUSE_EVIDENCE_ACROSS_ATTEMPT` | `true` | Reuse node-scoped evidence between retries of the same unit |
| `WEB_SEARCH_ALLOWED_DOMAINS` | *(empty)* | Comma-separated allowlist of source domains |
| `WEB_SEARCH_BLOCKED_DOMAINS` | *(empty)* | Comma-separated blocklist of source domains |
| `WEB_SEARCH_REGION` | `wt-wt` | DuckDuckGo region code |
| `WEB_SEARCH_SAFESEARCH` | `moderate` | DuckDuckGo safesearch mode |

!!! note "Enabling search adds a second critic call per pass"

    The critic breaks out of its loop immediately when it fails *without*
    requesting external evidence, so with search off it runs at most once per
    render whatever `MAX_VISITS` says. Turning search on is what opens the
    quadratic path — bound it with `MAX_CRITIC_VISITS_PER_NODE`.

### Other

```bash
CLEAN=false                              # flush triple store before `ontocast process` batch
LOGGING_LEVEL=info                       # debug | info | warning | error
```

## LLM Graph Format (`LLM_GRAPH_FORMAT`)

- `jsonld` (default): the LLM emits compact JSON-LD objects (`@context` + `@graph`); prompt context uses `` ```json `` blocks.
- `turtle` (legacy): the LLM emits RDF graph fields as Turtle strings; prompt context chapters use `` ```ttl `` blocks. Kept for providers whose structured output handles strings more reliably than nested objects.
- Domain models (`GraphUpdate`, critique reports, etc.) are **single canonical classes** at runtime. The format affects only LLM wire encoding, not duplicate Pydantic types.

Overridable per request as `llm_graph_format`, with the same precedence and the
same 400-on-typo contract as [`RENDER_MODE`](#render-mode-render_mode).

`ONTOLOGY_CHAPTER_FORMAT` (`inherit` default, `turtle`, or `term_sheet`)
narrows the choice to one chapter. With `turtle` the `# ONTOLOGY` chapter of the
facts render and critic prompts is serialized as canonical Turtle while
everything else keeps the wire format above; that chapter is the bulk of a facts
prompt and JSON-LD spends about twice the characters per triple, so this is the
context lever that leaves parsing untouched. With `term_sheet` the chapter stops
being a serialized graph and becomes a line-per-term listing — name, surface
forms, type, hierarchy, domain/range and scope note, without the per-statement
RDF scaffolding or `rdfs:comment` — which is by a wide margin the cheapest of
the three. See [Performance](performance.md) for what each keeps and drops.

All three apply to the facts loop only, are not overridable per request, and
change the LLM cache key for facts calls. `term_sheet` additionally requires
`RENDER_MODE=facts`: the ontology loop emits a patch against the statements in
its chapter, so that chapter has to be a graph, and a configuration that asks
for both is rejected at startup rather than silently falling back.

### Ontology chapter text caps

`ONTOLOGY_TEXT_MAX_CHARS_NAMING`, `ONTOLOGY_TEXT_MAX_CHARS_CONTRACT`,
`ONTOLOGY_TEXT_MAX_CHARS_PROSE` and `ONTOLOGY_TEXT_TOTAL_BUDGET` bound the text
literals in the ontology chapter, in whichever of the three formats it is built.
`ONTOLOGY_CONTEXT_MAX_TRIPLES` below is a *count* and says nothing about how
long any one literal may be, so a chapter well inside it can still be unbounded.

All four are unset by default and are inert when unset — a catalog whose labels
and comments are already short sees byte-identical prompts and cache keys.
Clipping is on a word boundary with a visible marker. Over the total budget,
prose is tightened then dropped, then contracts, and only then are names clipped
to a floor; names are never dropped. The run manifest's `budget.counters`
reports `chapter/text_chars_before`, `chapter/text_chars_after`,
`chapter/literals_clipped`, `chapter/literals_dropped` and
`chapter/text_over_budget`.

## Ontology Context Size (`ONTOLOGY_CONTEXT_MAX_TRIPLES`)

How much ontology is serialized into each prompt. This is the knob for context
blow-up — **not** `ONTOLOGY_MAX_TRIPLES`, which despite the name bounds the
per-unit *working graph* on the write path and is unset by default.

Only vector mode ever bounded the context. In `selected_single_ontology` (the
default) and `fixed_single_ontology` the whole selected ontology was serialized
into every prompt, and the facts fan-out serialized the union of every ontology
artifact — with no cap at all.

| Mode | Bound on what reaches the LLM |
|---|---|
| `selected_single_ontology` | `ONTOLOGY_CONTEXT_MAX_TRIPLES` |
| `fixed_single_ontology` | `ONTOLOGY_CONTEXT_MAX_TRIPLES` |
| `selected_vector_search_ontology` | `VECTOR_STORE_INDUCED_SUBGRAPH_MAX_TOTAL_TRIPLES` (`1200`) binds first, then the budget above as a backstop |
| Facts prompts (merged document context) | `ONTOLOGY_CONTEXT_MAX_TRIPLES` |

**What it costs.** JSON-LD spends roughly twice the characters per triple that
Turtle does, so the same budget is about twice the prompt under the default wire
format. See [Performance](performance.md#how-much-a-triple-costs).

**How the budget is met.** Over budget, triples are dropped in increasing order
of harm, stopping as soon as the graph fits:

1. Header and RDF-list noise — `owl:versionInfo`, `owl:imports`, `dcterms:creator`/`license`/`created`, `rdf:first`/`rest`.
2. Redundant structure — generic `rdf:type owl:Class`/`owl:NamedIndividual` where an informative type exists, stub restriction blank nodes, orphaned blank nodes.
3. Glosses — `rdfs:comment`, `skos:definition`, `skos:scopeNote`, `skos:altLabel`.

`rdfs:label`/`skos:prefLabel`, `rdf:type`, `rdfs:subClassOf`/`owl:equivalentClass`
and `rdfs:domain`/`range`/`subPropertyOf` are **never** dropped.

!!! warning "This is best-effort, not a hard ceiling"

    A graph that still exceeds the budget after step 3 is passed through
    oversized with a `WARNING`, because dropping labels or domain/range to hit a
    number produces an extraction failure that looks like a bad model. Treat
    that warning as "split the catalog, or switch to
    `selected_vector_search_ontology`", not as "raise the number". Set to empty
    to disable condensing entirely.

`ONTOLOGY_SNAPSHOT_TRIPLES` in the run manifest reports the resolved snapshot
size for every mode.

## Render Mode (`RENDER_MODE`)

Selects **which blocks of the pipeline run**. This is the coarsest control in the
system: two of the three values skip an entire half of the graph.

- `ontology_and_facts` (default): the full graph — ontology block, then facts block.
- `ontology`: stop after the ontology block. `RENDER_FACTS`, `MERGE_FACTS` and the
  `VALIDATE_FACTS` SHACL gate never run, and **no fact graph is written to the
  triple store**. Use it to build or extend a schema from a corpus.
- `facts`: skip the ontology block entirely and go straight from chunking to
  `RENDER_FACTS`. Extraction relies **wholly on the existing catalog** — whatever
  the per-unit context resolver supplies is treated as read-only schema, and no
  new terms are added to it. Use it to populate instances against a schema you
  have already settled.

!!! warning "`RENDER_MODE=facts` against an empty catalog produces almost nothing"

    Nothing in this mode creates ontology terms, so an empty or badly matched
    catalog leaves the renderer with no schema to instantiate against. Seed the
    catalog first (see [Ontology Context](ontology_context.md#seeding-the-catalog)),
    or run `ontology_and_facts` once.

    This is the one mode `ontocast process` **refuses to start** in with an
    empty catalog, and the only mode `ONTOLOGY_CONTEXT_REQUIRED` applies to.
    The other two answer an empty context by creating vocabulary, so for them
    it is a starting point rather than a fault — see
    [Ontology Context](ontology_context.md#what-if-the-context-is-empty).

!!! tip "Recommended companions for `RENDER_MODE=facts`"

    Three settings earn their keep in facts-only runs and belong together in a
    facts-extraction environment (the first is on by default; it is listed so a
    preset that pins it off stands out):

    ```bash
    FACTS_CONTEXT_FROM_UNITS=true      # gate + aggregator see the units' resolved
                                       # ontology context instead of an empty graph
    LLM_JSON_MODE=true                 # constrained decoding (OpenAI): removes the
                                       # malformed-JSON retry failure mode
    FACTS_SHAPES_PROMPT_CONTRACT=auto  # default; shows the renderer the shapes
                                       # rulebook it is validated against
    ```

    With `FACTS_CONTEXT_FROM_UNITS=false` the SHACL gate validates against an
    empty ontology and overstates class violations (recorded as
    `validated_without_ontology_context`) — see
    [Validation](validation.md#how-the-validation-is-set-up).

Which stages that corresponds to is drawn out in [Workflow](workflow.md).

**Per request.** `render_mode` is overridable on both `/process` and
`/process_unit`, and is read identically from the query string, a JSON body, or
a multipart form field. Precedence:

```text
query parameter  >  JSON / form body  >  RENDER_MODE  >  ontology_and_facts
```

An unrecognised value is rejected with **400** rather than silently falling back
to the default — a typo used to run the wrong pipeline and return `200`. There is
no CLI flag: `ontocast process` batch runs take the environment value.

## Ontology Context Mode (`ONTOLOGY_CONTEXT_MODE`)

Selects **where the schema shown to the LLM comes from**, per content unit.
Orthogonal to `RENDER_MODE`: it applies to whichever blocks that runs.

- `selected_single_ontology` (default): LLM picks one catalog ontology per content unit; no vector store required. Costs **one extra LLM call per content unit** for the selection itself.
- `selected_vector_search_ontology`: hybrid vector retrieval + induced subgraph; requires `QDRANT_URI` **or** `LANCEDB_ENABLED=true` plus embedding settings. This is also the only mode in which the **consistency critic** stage runs.
- `fixed_single_ontology`: pin one catalog ontology via `ONTOLOGY_CONTEXT_FIXED_ONTOLOGY_ID` — ontology **IRI**, short `ontology_id`, or author **prefix**.

!!! warning "Setting a fixed ontology id silently overrides the mode"

    A non-empty `ontology_context_fixed_ontology_id` **forces**
    `fixed_single_ontology`, whatever `ontology_context_mode` says and whatever
    the server default is. Passing both a fixed id and
    `ontology_context_mode=selected_vector_search_ontology` gets you fixed mode,
    with no error. Clear the id to use any other mode.

If vector mode is requested while no vector backend is available, the API returns `409` with `error_code: VECTOR_STORE_UNAVAILABLE`. The CLI is stricter: `ontocast serve` and `ontocast process` fail at **startup** rather than per request, both for vector mode with no backend and for fixed mode with no configured id.

Details: [Ontology Context](ontology_context.md). Catalog read path: [Ontology Catalog](../architecture/ontology_catalog.md).

## Usage

```python
from ontocast.config import Config

config = Config()
tool_config = config.get_tool_config()

print(config.server.port)
print(config.server.max_visits_per_node)
print(tool_config.llm_config.provider)
print(tool_config.path_config.cache_dir)
```

## Graph Matching API

Entity alignment and evaluation endpoints are documented in [API Endpoints](api.md#graph-matching).

## Validation Notes

- `LLM_PROVIDER=openai`, `anthropic`, or `google` requires `LLM_API_KEY`.
- `LLM_MODEL_NAME` outside the provider's preset enum logs a warning and is passed
  through — the provider validates it, not OntoCast.
- `MAX_VISITS_PER_NODE` is the canonical name; `MAX_VISITS` is an accepted alias for it. Set one, not both.
- `RENDER_MODE`, `ONTOLOGY_CONTEXT_MODE` and `LLM_GRAPH_FORMAT` reject an unrecognised per-request value with `400` rather than falling back to the environment default.
- `RECURSION_LIMIT` was renamed to `BASE_RECURSION_LIMIT`.
- `WEB_SEARCH_ALLOWED_DOMAINS` and `WEB_SEARCH_BLOCKED_DOMAINS` accept comma-separated values.
- `LLM_CACHE_ENABLED` and `LLM_CACHE_READ_ONLY` control disk cache read/write behavior.
- `LLM_MAX_INFLIGHT` must be ≥ 1; `MAX_CONCURRENT_PROCESSES` must be ≥ 1 when set.

## Recommended Workflow

0. Or skip straight to [Configuration Playbooks](playbooks.md), which does the
   first three steps for you per task.
1. Copy `.env.example` to `.env`.
2. Fill in LLM credentials and backend settings.
3. Start with defaults for chunking, search, and aggregation.
4. Tune only after inspecting extraction quality and runtime.
