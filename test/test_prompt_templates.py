"""Smoke coverage for every ``ontocast.prompt`` module.

Seven of these modules had no test contact at all, so a template edit that
broke a ``.format()`` slot -- an unbalanced brace, a malformed conversion, a
stray single brace where the double-escaped ``{{search_guidelines}}`` form was
meant -- surfaced only on a live run, after paying for the call.

The checks are structural: they assert a template can be parsed and filled,
never what it says.

Two kinds of module-level string live here and only one is a template. A
template is rendered, by ``str.format`` or by ``PromptTemplate``. A *literal
block* (``facts_literal_rules_jsonld``, ``_OUTPUT_INSTRUCTION_JSONLD``) is
substituted **into** a template's slot and legitimately carries raw JSON braces
that ``format`` rejects -- ``{"@value": "2024-01-15"}`` is a JSON-LD example,
not a slot. Checking those would fail on correct code, so the two rendering
paths are discovered from the source instead of guessed from names.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import pkgutil
import re
from string import Formatter

import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import ontocast
import ontocast.prompt

pytestmark = pytest.mark.unit

_FORMATTER = Formatter()

#: ``some_template.format(`` / ``prompt.some_template.format(`` call sites.
_FORMAT_CALL = re.compile(r"(?:^|[^\w.])(?:\w+\.)*(\w+)\.format\(")

#: Constants handed to ``PromptTemplate(template=...)`` rather than formatted
#: directly. LangChain validates the f-string on construction.
_PROMPT_TEMPLATE_CONSTANT = "template_prompt"


def _prompt_modules() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.iter_modules(ontocast.prompt.__path__)
        if not module.ispkg
    )


def _formatted_names() -> frozenset[str]:
    """Names the package calls ``.format()`` on, anywhere in its source."""
    root = pathlib.Path(ontocast.__file__).parent
    names: set[str] = set()
    for path in root.rglob("*.py"):
        names.update(_FORMAT_CALL.findall(path.read_text(encoding="utf-8")))
    return frozenset(names)


def _string_constants(module) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in vars(module).items()
        if isinstance(value, str) and not name.startswith("__")
    ]


def _assert_renders(label: str, template: str) -> None:
    """Parse the template, reject positional slots, and fill every named one."""
    try:
        parsed = list(_FORMATTER.parse(template))
    except ValueError as exc:  # pragma: no cover - the failure being guarded
        pytest.fail(f"{label} is not parseable: {exc}")

    assert "" not in {field for _, field, _, _ in parsed if field is not None}, (
        f"{label} uses a positional slot, which makes it order-dependent"
    )

    fields = {field for _, field, _, _ in parsed if field}
    try:
        template.format(**dict.fromkeys(fields, "x"))
    except (KeyError, IndexError, ValueError) as exc:  # pragma: no cover
        pytest.fail(f"{label} does not fill: {exc!r}")


MODULES = _prompt_modules()
FORMATTED_NAMES = _formatted_names()


def test_prompt_package_has_modules() -> None:
    """Guard the guard: empty discovery would make everything below vacuous."""
    assert len(MODULES) >= 10


def test_format_call_discovery_finds_a_known_template() -> None:
    """Guard the guard: a broken regex would silently skip every fill check."""
    assert "facts_instruction_shared" in FORMATTED_NAMES


@pytest.mark.parametrize("module_name", MODULES)
def test_formatted_templates_render(module_name: str) -> None:
    """Every template the package formats must fill from its declared slots."""
    module = importlib.import_module(f"ontocast.prompt.{module_name}")
    for name, template in _string_constants(module):
        if name in FORMATTED_NAMES:
            _assert_renders(f"{module_name}.{name}", template)


@pytest.mark.parametrize("module_name", MODULES)
def test_prompt_template_constants_build(module_name: str) -> None:
    """``template_prompt`` must survive PromptTemplate's f-string validation.

    This is the assembly path for the render and critic prompts, and the one an
    edit is most likely to break: the agents declare ``input_variables``
    explicitly, so a slot added to the text and not to that list raises only
    when the prompt is built for a real call.
    """
    module = importlib.import_module(f"ontocast.prompt.{module_name}")
    template = getattr(module, _PROMPT_TEMPLATE_CONSTANT, None)
    if template is None:
        pytest.skip(f"{module_name} declares no {_PROMPT_TEMPLATE_CONSTANT}")

    built = PromptTemplate.from_template(template)
    assert built.input_variables, "a prompt with no slots is a constant, not a template"
    rendered = built.format(**dict.fromkeys(built.input_variables, "x"))
    assert rendered.strip()


@pytest.mark.parametrize("module_name", MODULES)
def test_chat_templates_render(module_name: str) -> None:
    """A ChatPromptTemplate must render from its own declared input variables."""
    module = importlib.import_module(f"ontocast.prompt.{module_name}")
    for name, template in vars(module).items():
        if not isinstance(template, ChatPromptTemplate):
            continue
        messages = template.format_messages(
            **dict.fromkeys(template.input_variables, "x")
        )
        assert messages, f"{module_name}.{name} rendered no messages"


def test_facts_prompts_carry_the_conformance_placeholder() -> None:
    """Render and critic share one conformance contract slot."""
    from ontocast.prompt import criticise_facts, render_facts

    assert "{conformance_chapter}" in render_facts.template_prompt
    assert "{conformance_chapter}" in criticise_facts.template_prompt


def _slot_order(template: str) -> list[str]:
    """Named slots of a template, in the order they are filled."""
    return [field for _, field, _, _ in _FORMATTER.parse(template) if field]


def shared_prefix_length(left: str, right: str) -> int:
    """Characters from the start that two prompts have in common.

    This is what a provider's prefix cache can serve the second call from the
    first, so it is the number the chapter order is arranged to maximise.
    """
    return len(os.path.commonprefix([left, right]))


def test_facts_prompt_chapter_order_puts_the_shared_prefix_first() -> None:
    """Render and critic open identically: preamble, conformance, ontology.

    The ontology chapter is most of a facts prompt and identical between the
    render and the critic call on a unit. Everything phase-specific -- the
    operational or evaluation guidelines, the user instruction, the text --
    comes after it, so the two prompts are byte-identical up to the end of
    the chapter and a provider's prefix cache can serve the second call what
    the first one paid for.
    """
    from ontocast.prompt import criticise_facts, render_facts

    shared_head = ["preamble", "conformance_chapter", "ontology_chapter"]
    assert _slot_order(render_facts.template_prompt) == [
        *shared_head,
        "facts_instruction",
        "user_instruction",
        "text_chapter",
        "fact_chapter",
        "improvement_instruction",
        "output_instruction",
        "format_instructions",
    ]
    assert _slot_order(criticise_facts.template_prompt) == [
        *shared_head,
        "evaluation_instruction",
        "user_instruction",
        "text_chapter",
        "facts_chapter",
        "graph_format_instruction",
        "format_instructions",
    ]


def test_facts_prompt_heads_are_shared_verbatim() -> None:
    """The prefix starts at byte zero: preamble and separators must not differ."""
    from ontocast.prompt import criticise_facts, render_facts

    assert render_facts.preamble == criticise_facts.preamble
    render_head = render_facts.template_prompt.split("{ontology_chapter}")[0]
    critic_head = criticise_facts.template_prompt.split("{ontology_chapter}")[0]
    assert render_head == critic_head


@pytest.mark.anyio
async def test_render_and_critic_prompts_share_the_prefix_through_the_ontology(
    monkeypatch,
) -> None:
    """The two calls on a unit agree byte-for-byte up to the ontology chapter.

    Drives the real agents rather than the templates, so a chapter builder,
    profile lookup or separator that drifts between the two is caught here
    and not on a bill.
    """
    from ontocast.onto.model import FactsCritiqueReport, FactsRenderReport
    from ontocast.onto.rdfgraph import RDFGraph
    from ontocast.onto.unit_states import UnitFactsState
    from test.snapshot_helpers import snapshot_from_ontology
    from test.test_agent_facts import (
        _build_content_unit,
        _build_ontology,
        _build_tools,
    )

    # The package re-exports the agent *functions* under these names, so the
    # modules have to be reached by path.
    render_agent = importlib.import_module("ontocast.agent.render_facts")
    critic_agent = importlib.import_module("ontocast.agent.criticise_facts")

    prompts: dict[str, str] = {}
    chapters: dict[str, str] = {}

    def capturing(name: str, report):
        async def fake(**kwargs):
            prompts[name] = kwargs["prompt"].format(**kwargs["prompt_kwargs"])
            chapters[name] = kwargs["prompt_kwargs"]["ontology_chapter"]
            return report

        return fake

    rendered = RDFGraph()
    rendered.parse(
        data="@prefix ex: <https://example.com/ns#> .\nex:alice ex:worksFor ex:acme .",
        format="turtle",
    )
    monkeypatch.setattr(
        render_agent,
        "call_llm_with_retry",
        capturing(
            "render",
            FactsRenderReport(
                semantic_graph=rendered,
                ontology_relevance_score=90,
                triples_generation_score=90,
            ),
        ),
    )
    monkeypatch.setattr(
        critic_agent,
        "call_llm_with_retry",
        capturing(
            "critic",
            FactsCritiqueReport(
                success=True,
                score=95,
                actionable_triple_fixes=[],
                systemic_critique_summary="",
            ),
        ),
    )

    state = UnitFactsState(
        content_unit=_build_content_unit(with_graph=False),
        ontology_snapshot=snapshot_from_ontology(_build_ontology()),
        conformance_chapter="# CONFORMANCE REQUIREMENTS\n- ex:Person:\n  - needs ex:name",
        facts_user_instruction="Prefer lowercase local names.",
    )
    tools = _build_tools()
    await render_agent.render_facts_fresh(state, tools=tools)
    await critic_agent.criticise_facts(state, tools=tools)

    render_prompt, critic_prompt = prompts["render"], prompts["critic"]
    assert chapters["render"] == chapters["critic"]
    end_of_ontology = render_prompt.index(chapters["render"]) + len(chapters["render"])
    assert shared_prefix_length(render_prompt, critic_prompt) >= end_of_ontology

    shared = render_prompt[:end_of_ontology]
    assert "# CONFORMANCE REQUIREMENTS" in shared
    assert "# ONTOLOGY" in shared
    # Everything phase-specific is after the shared prefix, in both prompts.
    for prompt in (render_prompt, critic_prompt):
        assert prompt.index("# TASK") >= end_of_ontology
        assert prompt.index("# USER INSTRUCTION") >= end_of_ontology
        assert prompt.index("# TEXT") >= end_of_ontology


def test_quantitative_completeness_rule_is_stated() -> None:
    """Rule 3a counters the ontology-scoped selectivity of rule 3 without
    weakening the anti-junk guard of rule 4."""
    from ontocast.prompt import facts_guidelines

    text = facts_guidelines.facts_instruction_shared
    assert "QUANTITATIVE COMPLETENESS" in text
    assert "EVERY quantitative statement" in text
    # The anti-junk guard stays verbatim.
    assert "Do NOT mint an entity for a bare number" in text


def test_critic_completeness_guideline_routes_misses_to_the_completion_pass() -> None:
    """The critic lists what it judges missed; it does not mint nodes for it.

    Asking for an ADD per listed number produced placeholder subjects named
    after ignored tokens, which the compiler now refuses; the guideline has to
    say so in the same breath as it asks for completeness.
    """
    from ontocast.prompt import criticise_facts

    text = criticise_facts.evaluation_instruction
    assert "NUMERIC COVERAGE" in text
    assert "propose an ADD fix per listed number" in text
    assert "systemic_critique_summary" in text
    assert "completion pass" in text
    assert "NEVER mint an entity or placeholder node" in text
