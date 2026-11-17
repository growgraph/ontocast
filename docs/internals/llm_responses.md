# How LLM responses are parsed

Every structured LLM call in OntoCast expects a JSON object. This page describes
what happens to a response before it becomes a model, and what a malformed
response costs. None of it is configurable. The repairs and retries are
counted under `llm/*` in the run's `budget.counters`; see [Reading run
telemetry](../guides/telemetry.md).

## Steps

1. **Clean up.** Two malformations that come from the model, not from the
   payload, are repaired: escaped quotes around JSON strings
   (`"text_fragment": \"…\",`) and escaped whitespace between tokens. The scan
   knows where strings are, so an escaped quote inside a string is kept. JSON
   comments and trailing commas are then removed.
2. **Parse strictly.** The text is parsed with a real JSON parser. The only
   leniency is raw control characters inside strings, which models do emit.
3. **Fall back.** If that fails, a fenced ```` ```json ```` block is
   extracted and parsed. A complete object followed only by closing brackets
   and whitespace is taken without them (`llm/json_surplus_closer`). An object
   closed after its first member and continued with more members is taken
   without that early brace when the rest closes one object with no key named
   twice (`llm/json_premature_closer`); any other trailing content still fails. Otherwise closing brackets are rewritten to
   the kind their opening bracket needs. This repair only substitutes
   characters; it never inserts, deletes or reorders them, and it gives up on an
   unmatched closing bracket or an unclosed one at the end, because that is a
   truncated response and closing it would invent content. A successful repair
   counts as `llm/json_bracket_repair`.
4. **Fail with a location.** A response that still does not parse raises
   `LLMJsonParseError` with the line, the column and about 150 characters on
   either side. The retry prompt shows the model this error and an excerpt of
   its response around the error position, not the whole response.

## Retries

Parse failures are retried with exponential backoff and jitter
(`llm/parse_retry`), so parts that fail together do not retry in step. Two rules
bound the retries:

- **The same JSON syntax error twice ends the call** (`llm/parse_abandoned`). A
  model that makes one structural mistake twice makes it again. Errors are
  compared by kind, not position. Schema validation errors are not counted
  here, because those retries do converge.
- **Only parse failures are retried.** A rate limit or connection error is
  raised at once: no output arrived, and retrying would raise the request rate
  exactly when the provider asks for less. A timeout is the exception: it is
  re-issued once before it is raised.
