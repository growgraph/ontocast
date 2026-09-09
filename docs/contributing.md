# Contributing to OntoCast

We welcome contributions! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install development dependencies:
   ```bash
   uv sync --all-extras
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Run tests:
   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -m "not slow"
   ```
   That is the bar the default run has to clear: offline, model-free, no
   provider credentials. Markers carve out the rest — `slow` loads an ML model
   or takes more than a few seconds, `integration` needs a live service. Both
   are deselected unless you pass `-m`, and service-gated tests skip themselves
   when the service is unreachable.

   **Do not source `.env` into the test run.** The suite is only meaningful
   against declared defaults, and a developer's live configuration silently
   invalidates it — a local `RENDER_MODE=facts` leaves the entire ontology
   block untested while the suite still reports green. `pytest-dotenv` is
   blocked in `addopts` (`-pno:dotenv`) and `test/conftest.py` fails the run
   outright if a pipeline mode selector leaked in from the shell. Set what an
   individual test needs with `monkeypatch`. For an integration run, export
   only the service URLs it needs:
   ```bash
   QDRANT_URI=http://localhost:6333 uv run pytest -m integration
   ```

3. Build docs locally after doc or API changes:
   ```bash
   uv run mkdocs build
   ```

4. Commit, push, and open a Pull Request

## Testing

### Test layout

Tests are grouped into packages mirroring the source tree:

| Package | Covers |
|---|---|
| `test/facts/` | `tool/facts_validation/` — term policy, per-unit findings, the gate, acceptance, the repair loop |
| `test/ontology/` | the ontology lane — catalog identity, per-unit context, delta validation, reconcile, loop telemetry |
| `test/chunking/` | conversion to content units — segmentation, section labels, schema detection |
| `test/aggregation/` | `tool/agg/` — entity disambiguation, merge guards, provenance |
| `test/manual/` | opt-in, needs `ONTOCAST_RUN_MANUAL_TESTS=1`; not collected otherwise |

Everything else stays at the top level. Put a new test beside the ones covering
the same subsystem rather than adding a top-level module.

When consolidating modules, check for top-level names defined differently in
each — private fixture factories especially. Concatenating two files that both
define `_tools()` shadows one with the other, and the suite still passes while
half of it stops testing what it was written for.

### Test fixtures live under `test/`

The sdist ships `/test` and a short allowlist of root files; it does not ship
`docs/`, `demo/`, or any corpus directory. A test that resolves a path outside
`test/` therefore cannot run from a published sdist, and three such tests once
skipped silently on every machine that lacked the corpus rather than failing.
`test/test_repo_isolation.py` enforces this: put fixtures under `test/data/`,
and if a test genuinely must read a declaration file at the repo root, add it
to that module's `ALLOWED_ESCAPES` with the reason.

### Measurement lives elsewhere

Performance and extraction-quality numbers do not belong in this repository's
documentation, changelog, or docstrings. `ontocast` is a technical package: its
docs state mechanisms, contracts and defaults, which stay true across corpora
and model versions. A measured figure does not — it is true of one corpus, one
model and one day, and once written down it is quietly wrong from then on, with
nothing to detect that it drifted.

So:

- **Do** describe what a knob controls, which direction it moves things, and
  what saturates. **Do not** attach the number a sweep produced.
- **Do** name the telemetry a reader should use to measure their own setup —
  `retrieval_metrics`, `budget`, the run manifest. **Do not** substitute your
  numbers for theirs.
- **Do not** name a corpus, a benchmark, or an individual evaluation run —
  in prose, in a changelog entry, in a test name, or in a docstring. A reader
  outside this repository cannot resolve those names, and a reader inside the
  project should be reading the measurement system instead.
- Justify a default by its **mechanism** ("saturates quickly", "gates
  everything below it"), not by the run that chose it.

Benchmark and evaluation results are tracked systematically in
`ontocast-validation`. Link there when a number is genuinely needed.

This is about *claims*, not vocabulary. Fixture data is exempt: an example namespace, a sample document that happens to say "we used a benchmark", or a test graph named after a domain are arbitrary test inputs, not assertions about a measured run.

### Retrieval quality

`test/test_retrieval_recall.py` measures whether a relevant catalog term
actually survives to the prompt snapshot — the thing the plumbing tests, which
assert ordering and parameter pass-through against fake vectors, cannot see. It
reports two numbers per run:

- **seed recall** — the expected term reached the final seed set, so it survived
  vector search, the cross-window merge and the atom cap;
- **snapshot recall** — the term is also *defined* in the returned graph, so it
  survived induced-subgraph expansion, budget caps and component pruning.

The gap between them attributes a loss to the graph stage rather than the vector
stage, which is what makes a regression localisable without bisecting. Wiring
and scoring live in `test/retrieval_runner.py`; `test/retrieval_sweep.py` reuses
them to sweep retrieval parameters in one process.

It makes **no LLM call** — retrieval is embeddings and graph work — so it needs
no provider credentials, and the toolbox's eagerly constructed client is pinned
to a provider that needs no API key. The backend may be an embedded LanceDB, so
no service has to be running:

```bash
ONTOCAST_RECALL_CORPUS=<corpus dir> LANCEDB_ENABLED=true \
  uv run pytest test/test_retrieval_recall.py -m "integration and slow" -s
```

A corpus is a directory holding `cases.jsonl` (`{"id", "text",
"expected_iris", "ontology_iri"}`) and `ontologies/*.ttl`; the tests skip
themselves when `ONTOCAST_RECALL_CORPUS` is unset, and the corpora themselves
live in `ontocast-validation` rather than here. `ONTOCAST_RECALL_JSON` writes
the funnel as JSON. One index serves a whole sweep — every retrieval knob is
applied at merge or expansion time, so none of them joins the embedding
fingerprint — which is what `ONTOCAST_RECALL_COLLECTION_SUFFIX` and
`ONTOCAST_RECALL_SKIP_INDEX` are for.

Two cautions. Real embeddings are not bit-reproducible run to run, so compare
arms by rank and by which ontologies contributed no seed term at all, not by
exact percentages; and the on-topic precision figure is a noise proxy that
penalises legitimately retrieved parent terms from another module. This is an
ablation instrument, not an equality assertion.

Also in-repo: `test/test_retrieval_predicate_recall.py` for predicate-surface
coverage, and the per-run `retrieval_metrics` reported by the API and batch
dumps. See [Ontology Context —
Diagnostics](user_guide/ontology_context.md#diagnostics).

## Documentation

- User-facing guides live in `docs/` (MkDocs). Update `mkdocs.yml` nav when adding pages.
- **API reference** under `docs/reference/` is **generated at build time** by `docs/gen_pages.py` from Python modules. Do not commit hand-written reference stubs; add docstrings in code instead.
- **Workflow diagrams** in `docs/assets/` (`graph*`, `ontology_loop*`, `facts_loop*`) are generated by `uv run plot-graph` (requires optional `pygraphviz` for PNG/SVG). Loop diagrams default to the core path; `*_evidence.*` includes optional web-search branches.
- Keep `README.md` concise; put detailed explanations in `docs/`.
- Update `CHANGELOG.md` for user-visible changes.

## Code Style

- Python 3.12+ with type hints everywhere
- Follow PEP 8; use `pydantic.BaseModel` for structured data in the library
- Google-style docstrings on public APIs
- Match existing naming and patterns in the module you edit

## Pull Request Checklist

1. Tests pass (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -m "not slow"`)
2. Docs build (`uv run mkdocs build`) when docs or public API changed
3. `CHANGELOG.md` updated for notable changes
4. Clear PR description with problem and solution

## Reporting Issues

Include Python version, OntoCast version, steps to reproduce, expected vs actual behavior, and relevant logs.
