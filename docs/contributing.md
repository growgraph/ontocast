# Contributing to OntoCast

This page covers what you need to change OntoCast and get the change merged:
setting up, running the tests, writing documentation, and the checklist a pull
request has to pass.

## Set up

1. Fork the repository on GitHub and clone your fork.
2. Install every extra, including the development and documentation tools:

    ```bash
    uv sync --all-extras
    ```

3. Install the pre-commit hooks:

    ```bash
    uv run pre-commit install
    ```

Run every Python command through `uv run`, so it uses the project's locked
environment.

## Run the tests

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -m "not slow"
```

This is the run every change has to pass: offline, without model weights or
provider credentials. Two markers carve out the rest. `slow` tests load an ML
model or take more than a few seconds; `integration` tests need a live service
and skip themselves when it is unreachable. Select them with `-m`, exporting
only the service URLs they need:

```bash
QDRANT_URI=http://localhost:6333 uv run pytest -m integration
```

!!! warning "Do not load your `.env` into a test run"
    The suite checks the declared defaults, and your live configuration
    silently changes what it tests: a local `RENDER_MODE=facts` leaves the whole
    ontology half untested while the suite stays green. `pytest-dotenv` is
    disabled in `addopts` (`-pno:dotenv`), and `test/conftest.py` stops the run
    if a pipeline mode setting is present in the environment. Set what a single
    test needs with `monkeypatch`.

### Test layout

Tests are grouped by the part of the source they cover:

| Package | Covers |
|---|---|
| `test/facts/` | `tool/facts_validation/`: term policy, per-unit findings, the gate, acceptance, the repair loop |
| `test/ontology/` | The ontology half: catalog identity, per-unit context, delta validation, reconciliation, loop telemetry |
| `test/chunking/` | Conversion to content units: segmentation, section labels, schema detection |
| `test/aggregation/` | `tool/agg/`: entity disambiguation, merge guards, provenance |
| `test/manual/` | Opt-in; collected only with `ONTOCAST_RUN_MANUAL_TESTS=1` |

Everything else stays at the top level of `test/`. Put a new test beside the
tests for the same subsystem. When you merge two test modules, check for
top-level names both define, private fixture factories especially: a
duplicate silently shadows the other, and half the tests stop testing what
they were written for while the suite still passes.

### Fixtures live under `test/`

The source distribution ships `test/` but not `docs/`, `demo/` or any data
directory, so a test that reads a file outside `test/` cannot run from it. Put
fixtures under `test/data/`. `test/test_repo_isolation.py` enforces this; if a
test must read a file at the repository root, add it to `ALLOWED_ESCAPES` in
that module with the reason.

### Retrieval quality

`test/test_retrieval_recall.py` checks whether a relevant catalog term reaches
the ontology context a unit is shown, reporting how many expected terms survive
vector search and how many survive graph expansion. It makes no LLM call and
can run on an embedded LanceDB. Point `ONTOCAST_RECALL_CORPUS` at a corpus
directory (`cases.jsonl` plus `ontologies/*.ttl`); without it the tests skip.
The corpora live in `ontocast-validation`.

```bash
ONTOCAST_RECALL_CORPUS=<corpus dir> LANCEDB_ENABLED=true \
  uv run pytest test/test_retrieval_recall.py -m "integration and slow" -s
```

Set `ONTOCAST_RECALL_JSON` to write the results to a file; the module
docstring lists the other controls.

## Measurement lives elsewhere

Documentation, the changelog, docstrings, comments and test names in this
repository describe mechanisms, contracts and defaults. They do not carry
measured results:

- Describe what a setting controls, which direction it moves things, and what
  saturates. Do not quote the number a sweep produced.
- Name the telemetry a reader uses to measure their own deployment:
  `budget`, `retrieval_metrics`, the run manifest.
- Do not name a corpus, a benchmark or an evaluation run, and do not name a
  test, fixture or flag after one; name it after what it guards.
- Justify a default by its mechanism ("saturates quickly", "gates everything
  below it"), not by the run that chose it.

Benchmark and evaluation results are tracked in `ontocast-validation`; link
there when a number is needed. The rule covers claims, not vocabulary: fixture
data such as an example namespace or a sample document is exempt.

## Documentation

The documentation is built with ProperDocs. To work on it, install the
documentation tools and serve the site locally:

```bash
uv sync --extra dev --extra docs
uv run properdocs serve
```

Name every extra you need in one `uv sync` command: `uv sync` removes the
extras you leave out. Before you open a pull request, check that the strict
build passes:

```bash
uv run properdocs build --strict
```

- Add a new page to the `nav` in `properdocs.yml`.
- The [Python API reference](reference/python/index.md) and the
  [configuration reference](reference/configuration/index.md) are generated at
  build time (`docs/_build/gen_pages.py`, `docs/_build/gen_config.py`). Do not
  write pages for them: add docstrings to a module, and a `description` to a
  settings `Field`.
- A `Field` description states what the setting does, in the present tense.
  `test/test_settings_reference.py` fails on history wording such as
  "previously", "no longer" or "deprecated".
- The workflow diagrams in `docs/assets/` are generated: regenerate them with
  `uv run plot-graph`, which needs the `plot` extra.
- Keep `README.md` short and put detail in `docs/`.

## Code style

- Python 3.12 or later, with type hints everywhere.
- `pydantic.BaseModel` for structured data.
- Google-style docstrings on public APIs.
- Follow the naming and patterns of the module you edit.

## Pull request checklist

1. The tests pass: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -m "not slow"`.
2. The docs build with `uv run properdocs build --strict` when you changed
   docs, settings or public API.
3. `CHANGELOG.md` has an entry for every user-visible change.
4. The description states the problem and the solution.

## Reporting issues

Include your Python and OntoCast versions, the steps to reproduce, what you
expected and what happened, and the relevant logs.
