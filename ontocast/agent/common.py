import logging
import re
from typing import Any, TypeVar

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

from ontocast.onto.enum import LLMGraphFormat, WorkflowNode
from ontocast.onto.llm_graph_payload import llm_graph_format_ctx
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

logger = logging.getLogger(__name__)

T = TypeVar("T")

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

    Args:
        llm_tool: The LLM tool instance to use for generation.
        prompt: The prompt template to format and send to the LLM.
        parser: The output parser to parse the LLM response.
        prompt_kwargs: Keyword arguments to pass to prompt.format_prompt().
        max_retries: Maximum number of retry attempts (default: 3).
        retry_error_feedback: Whether to include error feedback in retry prompts (default: True).
        llm_graph_format: When set, ``llm_graph_format_ctx`` is active for the whole retry loop
            so canonical parsers coerce graph wire payloads to ``RDFGraph``.

    Returns:
        The parsed output of type T.

    Raises:
        Exception: If parsing fails after all retry attempts, raises the last parsing error.
    """
    last_error: Exception | None = None
    last_sanitized_content: str | None = None
    original_format_instructions = prompt_kwargs.get("format_instructions", "")
    fmt_token = (
        llm_graph_format_ctx.set(llm_graph_format)
        if llm_graph_format is not None
        else None
    )

    try:
        for attempt in range(max_retries):
            try:
                # Create a copy of prompt_kwargs for this attempt
                attempt_kwargs = prompt_kwargs.copy()

                # On retry, add error feedback to help LLM correct format
                if attempt > 0 and retry_error_feedback and last_error is not None:
                    # Use sanitized content in error feedback for consistency
                    feedback_content = (
                        last_sanitized_content if last_sanitized_content else ""
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

                # Call LLM
                response = await llm_tool(prompt.format_prompt(**attempt_kwargs))
                content_to_parse = strip_trailing_commas(
                    strip_json_comments(response.content)
                )
                last_sanitized_content = content_to_parse

                parsed = parser.parse(content_to_parse)
                logger.debug(
                    f"Successfully parsed LLM response on attempt {attempt + 1}/{max_retries}"
                )
                return parsed

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to parse LLM response on attempt {attempt + 1}/{max_retries}: {str(e)}"
                )

                # If this was the last attempt, raise the error
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to parse LLM response after {max_retries} attempts. "
                        f"Last error: {str(e)}"
                    )
                    raise
    finally:
        if fmt_token is not None:
            llm_graph_format_ctx.reset(fmt_token)

    raise RuntimeError("Unexpected error in call_llm_with_retry")
