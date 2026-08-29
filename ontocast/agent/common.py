import asyncio
import json
import logging
import random
import re
from typing import Any, TypeVar, cast

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel

from ontocast.onto.enum import LLMGraphFormat, WorkflowNode
from ontocast.onto.model import Suggestions
from ontocast.prompt.common import (
    suggestion_concrete_template,
    suggestion_general_template,
)
from ontocast.prompt.render_facts import (
    improvement_instruction_template as facts_template,
)
from ontocast.prompt.render_ontology import (
    improvement_instruction_template as ontology_template,
)
from ontocast.tool import LLMTool
from ontocast.tool.llm import (
    LLMRequestTimeoutError,
    _content_to_str,
    record_active_count,
)

logger = logging.getLogger(__name__)

#: Base delay for parse-retry backoff, doubled per attempt.
RETRY_BACKOFF_BASE_SECONDS = 0.5
#: Upper bound, so a retry never stalls a unit worker for long.
RETRY_BACKOFF_MAX_SECONDS = 8.0


def _retry_backoff_seconds(attempt: int) -> float:
    """Delay before retry number ``attempt`` (1-based), with jitter.

    Jitter matters more than the delay itself here: without it, the units that
    failed to parse in the same fan-out wave all re-issue at the same instant,
    recreating the burst that may have caused the failure.
    """
    capped = min(
        RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX_SECONDS
    )
    return capped * random.uniform(0.5, 1.0)


T = TypeVar("T", bound=BaseModel)

_JSON_COMMENT_RE = re.compile(r'"(?:[^"\\]|\\.)*"|//[^\n]*')
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def strip_json_comments(text: str) -> str:
    """Remove single-line // comments from JSON-like text while preserving string literals.

    The LLM occasionally emits JavaScript-style // comments inside JSON output,
    which are not valid JSON.  This function strips them by scanning the text
    token by token: JSON string literals (which may contain '//') are kept
    intact, while bare // … sequences are dropped.
    """

    def _replace(m: re.Match) -> str:
        matched = m.group()
        return matched if matched.startswith('"') else ""

    return _JSON_COMMENT_RE.sub(_replace, text)


def strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]`` (invalid in strict JSON)."""
    prev = None
    while prev != text:
        prev = text
        text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def unescape_json_delimiters(text: str) -> str:
    """Repair strings whose *delimiting* quotes the LLM escaped.

    Observed malformation: ``"text_fragment": \\"quoted text\\",`` — the model
    escapes the quotes that should open and close the JSON string, which is
    invalid JSON and defeats the lenient parser too. This scans the text with
    string-awareness: a ``\\"`` where a key/value must start (after ``:``,
    ``,``, ``{`` or ``[``) is rewritten to ``"``, and inside such a repaired
    string a ``\\"`` followed by a delimiter (``,``, ``}``, ``]``, ``:``) is
    rewritten as its closer. Legitimate ``\\"`` escapes inside normally
    delimited strings are left untouched. The same responses escape token
    whitespace too (``\\",\\n      "action"``) — outside a string a backslash
    is never valid JSON, so ``\\n``/``\\t``/``\\r`` there are rewritten to the
    whitespace they denote.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    repaired_open = False  # current string was opened by a repaired \"
    last_significant = ""  # last non-space char seen outside strings

    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == '"' and repaired_open:
                    # Candidate closer: only if a delimiter (or EOF) follows.
                    j = i + 2
                    while j < n and text[j] in " \t\r\n":
                        j += 1
                    if j >= n or text[j] in ",}]:":
                        out.append('"')
                        i += 2
                        in_string = False
                        repaired_open = False
                        last_significant = '"'
                        continue
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            out.append(ch)
            if ch == '"':
                in_string = False
                repaired_open = False
                last_significant = '"'
            i += 1
            continue

        if (
            ch == "\\"
            and i + 1 < n
            and text[i + 1] == '"'
            and last_significant in (":", ",", "{", "[", "")
        ):
            out.append('"')
            i += 2
            in_string = True
            repaired_open = True
            continue

        if ch == "\\" and i + 1 < n and text[i + 1] in "nrt":
            out.append({"n": "\n", "r": "\r", "t": "\t"}[text[i + 1]])
            i += 2
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            repaired_open = False
        if not ch.isspace():
            last_significant = ch
        i += 1

    return "".join(out)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*(?:```|$)", re.DOTALL)

#: Context window shown around a decode error, both directions.
JSON_ERROR_WINDOW = 150


class LLMJsonParseError(ValueError):
    """A JSON *syntax* error in an LLM response, carrying the decoder's position.

    Distinguished from the schema ``ValidationError`` that follows a successful
    parse: only syntax errors get the position-window retry feedback, and only
    they are abandoned on a repeated error class -- a model that emits the same
    structural malformation twice emits it a third time, whereas a schema
    mismatch does converge.
    """

    def __init__(self, message: str, *, msg: str, pos: int, doc: str) -> None:
        super().__init__(message)
        self.msg = msg
        self.pos = pos
        self.doc = doc


def _format_json_error(text: str, error: json.JSONDecodeError) -> str:
    """Render a JSON decode error with a context window around the position."""
    start = max(0, error.pos - JSON_ERROR_WINDOW)
    end = min(len(text), error.pos + JSON_ERROR_WINDOW)
    return (
        f"Response is not valid JSON: {error.msg} at line {error.lineno} "
        f"column {error.colno} (char {error.pos}). Text around the error:\n"
        f"...{text[start:end]}..."
    )


def repair_bracket_kinds(text: str) -> tuple[str, int]:
    """Rewrite each closing bracket to the kind its opener demands.

    Models lose track of *which* frame they are closing while emitting a long
    payload, and produce the right number of closers in the wrong kinds -- the
    measured failure was ``] }`` where ``} ]`` was due, at the tail of a
    JSON-LD document nested inside a singleton list. The bracket counts balance,
    so nothing upstream notices; only the decoder does.

    The transform never inserts, deletes, or reorders: it only substitutes one
    character for another, and only where the document is already invalid.
    Repair is abandoned entirely (0 fixes, text unchanged) when a closer appears
    with no open frame, or when frames remain open at EOF -- that is genuine
    truncation, and closing it here would fabricate a payload the model never
    emitted.

    Args:
        text: Candidate JSON that failed strict parsing.

    Returns:
        The repaired text and the number of characters substituted.
    """
    out = list(text)
    stack: list[str] = []
    fixes = 0
    in_string = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if in_string:
            # Raw control characters are legal inside strings here because the
            # strict parse runs with strict=False; only escapes need skipping.
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return text, 0
            wanted = "}" if stack.pop() == "{" else "]"
            if wanted != ch:
                out[i] = wanted
                fixes += 1
        i += 1

    if stack:
        return text, 0
    return "".join(out), fixes


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse LLM JSON output strictly, failing loudly with position context.

    Langchain's ``parse_json_markdown`` degrades to a partial parser that
    silently returns ``None`` — or a truncated prefix of the object — for
    malformed input; validating that produced the informationless
    ``input_value=None`` retry feedback, and retries repeated the same
    malformation. Here strict parsing runs first, a fenced ``` block is
    extracted and strict-parsed as the next fallback, mismatched bracket
    *kinds* are repaired as the last one (see :func:`repair_bracket_kinds`),
    and any remaining failure raises :class:`LLMJsonParseError` with the
    strict error's line/column and a ±150-char context window, so retry
    feedback names the exact broken spot.

    ``strict=False`` mirrors the old path's one legitimate leniency — raw
    control characters inside string literals, which the models do emit —
    while every structural error still raises.
    """

    def _loads_object(candidate: str) -> dict:
        parsed = json.loads(candidate, strict=False)
        if not isinstance(parsed, dict):
            raise ValueError(
                "Response parsed as JSON but is not an object; got "
                f"{type(parsed).__name__}: {str(parsed)[:200]}"
            )
        return parsed

    try:
        return _loads_object(text)
    except json.JSONDecodeError as strict_error:
        reported = strict_error
        match = _JSON_FENCE_RE.search(text)
        if match and match.group(1):
            fenced = match.group(1)
            try:
                return _loads_object(fenced)
            except json.JSONDecodeError as fence_error:
                text, reported = fenced, fence_error

        repaired, fixes = repair_bracket_kinds(text)
        if fixes:
            try:
                parsed = _loads_object(repaired)
            except json.JSONDecodeError:
                pass
            else:
                logger.warning(
                    "Repaired %d mismatched JSON bracket(s) in LLM response "
                    "(%s at char %d)",
                    fixes,
                    reported.msg,
                    reported.pos,
                )
                record_active_count("llm/json_bracket_repair")
                return parsed

        raise LLMJsonParseError(
            _format_json_error(text, reported),
            msg=reported.msg,
            pos=reported.pos,
            doc=text,
        ) from strict_error


#: Feedback excerpt bounds. A malformed response is up to ~11 KB and pasting all
#: of it into every retry inflated the prompt without adding signal: the break is
#: at a single position, and for a schema error the head and tail carry the shape.
FEEDBACK_BEFORE_ERROR = 600
FEEDBACK_AFTER_ERROR = 200
FEEDBACK_HEAD = 800
FEEDBACK_TAIL = 1200


def _feedback_excerpt(content: str, error: Exception) -> str:
    """Bounded slice of a failed response to show the model on retry."""
    if isinstance(error, LLMJsonParseError):
        doc = error.doc or content
        start = max(0, error.pos - FEEDBACK_BEFORE_ERROR)
        end = min(len(doc), error.pos + FEEDBACK_AFTER_ERROR)
        prefix = "..." if start else ""
        suffix = "..." if end < len(doc) else ""
        return f"{prefix}{doc[start:end]}{suffix}"

    if len(content) <= FEEDBACK_HEAD + FEEDBACK_TAIL:
        return content
    elided = len(content) - FEEDBACK_HEAD - FEEDBACK_TAIL
    return (
        f"{content[:FEEDBACK_HEAD]}\n... [{elided} characters elided] ...\n"
        f"{content[-FEEDBACK_TAIL:]}"
    )


def render_suggestions_prompt(suggestions: Suggestions, stage: WorkflowNode) -> str:
    """Generate prompt templates from the suggestions.

    Returns:
        Combined string with general and concrete templates.
        Returns empty string if both fields are empty.
    """

    # Generate general template if systemic_critique_summary is not empty
    general_template = ""
    if suggestions.systemic_critique_summary.strip():
        general_template = suggestion_general_template.format(
            general_suggestion=suggestions.systemic_critique_summary
        )

    concrete_template = ""
    if suggestions.actionable_fixes:
        # Generate concrete template if actionable_fixes is not empty
        concrete_template = suggestion_concrete_template.format(
            suggestion_str=suggestions.to_markdown()
        )

    if stage == WorkflowNode.TEXT_TO_FACTS:
        template = facts_template
    elif stage == WorkflowNode.TEXT_TO_ONTOLOGY:
        template = ontology_template
    else:
        raise ValueError(f"Stage {stage} not supported")
    if general_template or concrete_template:
        final_prompt = template.format(
            suggestions_instruction=f"\n\n{general_template}\n\n{concrete_template}"
        )
    else:
        final_prompt = ""
    return final_prompt


async def call_llm_with_retry(
    llm_tool: LLMTool,
    prompt: BasePromptTemplate,
    parser: BaseOutputParser[T],
    prompt_kwargs: dict[str, Any],
    max_retries: int = 3,
    retry_error_feedback: bool = True,
    llm_graph_format: LLMGraphFormat | None = None,
) -> T:
    """Call LLM and parse response with automatic retry on parsing failures.

    This utility function implements a common pattern across agent functions:
    1. Call LLM with a prompt
    2. Parse the response
    3. Retry if parsing fails (up to max_retries times)

    On retry, if retry_error_feedback is True, the error message from the previous
    attempt is included in the prompt to help the LLM correct its output format.

    Only *parsing* failures are retried with feedback. A transport-level
    failure (rate limit, connection error) propagates on the first occurrence:
    the retry exists to show the model its own malformed output, which is
    meaningless when no output arrived, and retrying would triple the request
    rate exactly when the provider is asking for less of it. The one exception
    is a request *timeout*: unlike a 429 it is not a provider "send less"
    signal, and losing the call silently costs a unit one of its few loop
    visits — so a single identical re-issue (per outer call) is allowed before
    the timeout propagates.

    Retries back off exponentially with jitter, so N units failing to parse
    simultaneously do not re-issue in lockstep.

    Args:
        llm_tool: The LLM tool instance to use for generation.
        prompt: The prompt template to format and send to the LLM.
        parser: The output parser to parse the LLM response.
        prompt_kwargs: Keyword arguments to pass to prompt.format_prompt().
        max_retries: Maximum number of retry attempts (default: 3).
        retry_error_feedback: Whether to include error feedback in retry prompts (default: True).
        llm_graph_format: When set, passed explicitly to ``model_validate`` as
            ``context={"llm_graph_format": ...}`` so graph wire fields coerce correctly.

    Returns:
        The parsed output of type T.

    Raises:
        Exception: The provider's error on a transport failure, or the last
            parsing error once retries are exhausted.
    """
    last_error: Exception | None = None
    last_sanitized_content: str | None = None
    last_json_error_msg: str | None = None
    original_format_instructions = prompt_kwargs.get("format_instructions", "")
    timeout_retry_available = True

    for attempt in range(max_retries):
        # Create a copy of prompt_kwargs for this attempt
        attempt_kwargs = prompt_kwargs.copy()

        # On retry, add error feedback to help LLM correct format
        if attempt > 0 and retry_error_feedback and last_error is not None:
            # Use sanitized content in error feedback for consistency
            feedback_content = _feedback_excerpt(
                last_sanitized_content or "", last_error
            )
            error_feedback = (
                f"\n\nIMPORTANT: The previous attempt failed to parse the response. "
                f"Error: {str(last_error)}\n"
                f"Previous response (for reference):\n{feedback_content}\n\n"
                f"Please ensure your response strictly follows the format instructions "
                f"and does not contain any control characters or invalid syntax."
            )
            # Add error feedback to format_instructions if present
            if "format_instructions" in attempt_kwargs:
                attempt_kwargs["format_instructions"] = (
                    original_format_instructions + error_feedback
                )
            else:
                # If no format_instructions, add as a new field
                attempt_kwargs["parsing_error_feedback"] = error_feedback

        if attempt > 0:
            record_active_count("llm/parse_retry")
            await asyncio.sleep(_retry_backoff_seconds(attempt))

        # Outside the try below on purpose: a transport failure must propagate
        # rather than be re-sent with parse-error feedback attached. A timeout
        # alone gets one identical re-issue (shared across parse attempts).
        formatted_prompt = prompt.format_prompt(**attempt_kwargs)
        try:
            response = await llm_tool(formatted_prompt)
        except LLMRequestTimeoutError:
            if not timeout_retry_available:
                raise
            timeout_retry_available = False
            logger.warning("LLM request timed out; re-issuing once before propagating")
            await asyncio.sleep(_retry_backoff_seconds(1))
            response = await llm_tool(formatted_prompt)

        try:
            content_to_parse = strip_trailing_commas(
                strip_json_comments(
                    unescape_json_delimiters(_content_to_str(response.content))
                )
            )
            last_sanitized_content = content_to_parse

            if isinstance(parser, PydanticOutputParser):
                json_object = parse_json_object(content_to_parse)
                model_cls = cast(type[BaseModel], parser.pydantic_object)
                context = (
                    {"llm_graph_format": llm_graph_format}
                    if llm_graph_format is not None
                    else None
                )
                parsed = cast(
                    T,
                    model_cls.model_validate(json_object, context=context),
                )
            else:
                parsed = parser.parse(content_to_parse)
            logger.debug(
                f"Successfully parsed LLM response on attempt {attempt + 1}/{max_retries}"
            )
            return parsed

        except Exception as e:
            last_error = e
            # One line per attempt: the same ±150-char window was previously
            # printed by every attempt, by the exhaustion branch, and again by
            # the calling agent -- five dumps for one failed render.
            if isinstance(e, LLMJsonParseError):
                logger.warning(
                    "Failed to parse LLM response on attempt %d/%d: %s at char %d",
                    attempt + 1,
                    max_retries,
                    e.msg,
                    e.pos,
                )
                logger.debug("Unparsable LLM response: %s", str(e))
            else:
                logger.warning(
                    f"Failed to parse LLM response on attempt "
                    f"{attempt + 1}/{max_retries}: {str(e)}"
                )

            # A repeated syntax-error *class* is a fixed point: the model has now
            # emitted the same structural malformation twice and will emit it
            # again, so the remaining attempts are pure spend. Position is not
            # part of the comparison -- it drifts between regenerations. Schema
            # ValidationErrors are excluded: those retries do converge.
            if isinstance(e, LLMJsonParseError):
                repeated = e.msg == last_json_error_msg
                last_json_error_msg = e.msg
                if repeated and attempt < max_retries - 1:
                    logger.error(
                        "Abandoning LLM response after %d/%d attempts: the same "
                        "JSON syntax error (%s) recurred. Last error: %s",
                        attempt + 1,
                        max_retries,
                        e.msg,
                        str(e),
                    )
                    record_active_count("llm/parse_abandoned")
                    raise
            else:
                last_json_error_msg = None

            # If this was the last attempt, raise the error
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to parse LLM response after {max_retries} attempts. "
                    f"Last error: {str(e)}"
                )
                record_active_count("llm/parse_abandoned")
                raise

    raise RuntimeError("Unexpected error in call_llm_with_retry")
