# Internals

These pages explain how OntoCast works inside: the mechanisms behind the
settings, the counters in the run telemetry, and the boundaries between
components. Read them when you contribute to OntoCast, or when you tune a
deployment beyond what the [guides](../guides/index.md) cover. They name
modules and classes, which may change between releases.

| Page | What it explains |
|---|---|
| [How ontology retrieval works](retrieval.md) | How each content unit's ontology context is retrieved, scored and sized |
| [Validation layers and the critic loop](validation_layers.md) | The checks extracted ontology and facts pass through, and how critic and repair budgets are spent |
| [How prompts are built](prompt_construction.md) | What goes into each render and critic prompt, and in which order |
| [Telemetry counters](telemetry_counters.md) | Every counter in `budget` and `retrieval_metrics` |
| [How LLM responses are parsed](llm_responses.md) | What happens to a model response before it becomes a result |
| [Ontology catalog](ontology_catalog.md) | Which component owns stored ontologies, how tenancy scopes them, and what a custom triple-store backend implements |
