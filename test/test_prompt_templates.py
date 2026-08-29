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
