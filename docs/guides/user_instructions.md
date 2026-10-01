# Writing user instructions

User instructions let you tell OntoCast what matters in your documents, such as
which entities, relations or measurements to extract, without changing its
prompts. This page shows the three instructions, how to pass them, and how to
write ones that help.

## The three instructions

| Instruction | Shown to | Use it to |
|---|---|---|
| `ontology_user_instruction` | The ontology render and its critic | Say which concepts and relations the ontology should model |
| `ontology_selection_user_instruction` | The call that picks an ontology for each part, in `selected_single_ontology` mode | Say which ontology to prefer when several could fit |
| `facts_user_instruction` | The facts render, its critic and the completion pass | Say which facts to extract and how to report them |

When web search is enabled, the ontology or facts instruction also seeds its
search queries.

Each instruction is added to the prompt under a `USER INSTRUCTION` heading. It
adds to OntoCast's own rules and never replaces them. In particular, the facts
rules always apply: every new entity goes in the `cd:` namespace, the domain
ontology is read-only, and a class is never used as an instance; see
[Ontologies and facts](../concepts/ontologies_and_facts.md). The `cd:`
namespace is fixed and cannot be configured. Use the facts instruction to steer
*what* is extracted, not namespaces or IRIs.

An instruction is part of the prompt, so changing it is a cache miss for every
call it reaches; see [LLM caching](llm_caching.md).

## Passing instructions

**HTTP.** `/process` and `/process_unit` accept the three fields in the query
string, as multipart form fields, or in a JSON body:

```bash
curl -X POST http://127.0.0.1:8999/process \
  -F "file=@report.pdf" \
  -F "ontology_user_instruction=Model companies, products and the deals between them." \
  -F "facts_user_instruction=Extract every monetary amount with its currency and period."
```

```bash
curl -X POST http://127.0.0.1:8999/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Acme Corp acquired Widget Ltd for $40M in Q3 2024.",
    "ontology_selection_user_instruction": "Prefer the finance ontology when several fit.",
    "facts_user_instruction": "Extract every monetary amount with its currency and period."
  }'
```

In a JSON body, put the document in `text`. Without it, OntoCast takes the
longest string field as the document, which could be an instruction.

**Batch.** `ontocast process` takes the facts instruction as a flag:

```bash
ontocast process --input-path ./docs --output-dir ./out \
  --facts-user-instruction "Extract every monetary amount with its currency and period."
```

For the other two, process `.json` input files that carry the instruction
fields next to `text`, as in the JSON body above. A field in the file wins over
the flag. The run manifest records the length of the facts instruction
(`validation_config.facts_user_instruction_chars`), not its text, so you can
share a manifest without sharing your instructions.

**Python.** Set the fields on the `AgentState` you pass to the pipeline:

```python
from ontocast import AgentState, run_unit_pipeline

state = AgentState(
    raw_input={"report.txt": text.encode()},
    ontology_user_instruction="Model companies, products and the deals between them.",
    facts_user_instruction="Extract every monetary amount with its currency and period.",
)
ontology_result, facts_result = await run_unit_pipeline(state, tools)
```

The same state works as input to the compiled graph; see [Embedding OntoCast
in your agent](embedding.md).

## Writing instructions that help

Name the things you want, in the terms your documents use, and say what form
the values should take. A vague instruction changes nothing; a specific one
changes what the model looks for.

| Instead of | Write |
|---|---|
| Extract everything important. | Extract company names, products, and acquisition and partnership deals between companies. |
| Extract numbers and dates. | Extract every measurement with its unit, and every date as an ISO 8601 date. |
| Extract medical information. | Extract diagnoses, treatments and dosages, and link each dosage to its treatment. |
| Use good ontologies. | Prefer the clinical ontology over the general one when both fit. |

Some example sets:

- **Financial reports.** Ontology: "Model companies, revenue streams and financial metrics." Facts: "Extract every monetary amount with its currency and reporting period, and every growth rate as a percentage."
- **Scientific papers.** Ontology: "Model materials, methods and measured properties." Facts: "Extract each measured value with its unit, the sample it was measured on, and the conditions."
- **Legal documents.** Ontology: "Model parties, obligations and the provisions that create them." Facts: "Extract each party, each obligation, and the clause that states it."

Start with one instruction, run a few parts with `--head-chunks`, and compare
the output with and without it before adding more.

## Checking that an instruction was applied

The instructions are not echoed in the `/process` response. Two ways to
confirm one reached the pipeline:

- With `LOGGING_LEVEL=debug`, the server logs each instruction it read from a JSON body or `.json` file (`Set facts user instruction: ...`).
- For batch runs, `validation_config.facts_user_instruction_chars` in the run manifest is non-zero.

If an instruction seems ignored, check that its field name is spelled exactly
as above: an unknown field is dropped without an error. A selection
instruction has no effect outside `selected_single_ontology` mode.
