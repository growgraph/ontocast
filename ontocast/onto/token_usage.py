"""Provider-reported token usage for a single LLM call.

Deliberately a leaf module with no OntoCast imports: it is shared between
:class:`ontocast.onto.state.BudgetTracker` and :mod:`ontocast.tool.llm`, and
``ontocast.tool.__init__`` imports ``llm``, so anything ``llm`` imports must not
reach back into :mod:`ontocast.tool`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token counts for one LLM call, as far as the provider reports them.

    Every field is optional because reporting is provider-dependent: the totals
    arrive from most providers, the detail fields only from those that bill
    reasoning or prompt-cache reads separately. ``None`` means *not reported*,
    which is not the same as zero -- a run whose provider stays silent should
    not look like a run that used no tokens.
    """

    input_tokens: int | None = Field(default=None, description="Prompt tokens.")
    output_tokens: int | None = Field(default=None, description="Completion tokens.")
    reasoning_tokens: int | None = Field(
        default=None,
        description=(
            "Thinking tokens, counted inside output_tokens. Dominates output "
            "cost for reasoning models (qwen3, deepseek-r1, kimi), which this "
            "package drives through LLMConfig.think."
        ),
    )
    cache_read_input_tokens: int | None = Field(
        default=None,
        description=(
            "Prompt tokens served from the *provider's* cache, counted inside "
            "input_tokens and billed at a fraction of the fresh rate. Unrelated "
            "to OntoCast's own on-disk response cache."
        ),
    )
    cache_creation_input_tokens: int | None = Field(
        default=None,
        description="Prompt tokens written to the provider's cache.",
    )

    def is_empty(self) -> bool:
        """True when the provider reported nothing at all."""
        return all(
            value is None for value in self.model_dump(exclude_none=False).values()
        )
