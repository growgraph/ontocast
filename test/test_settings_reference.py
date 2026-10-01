"""The settings models are the configuration reference; keep them readable.

The docs build renders every ``Field`` description into the configuration
reference (``docs/_build/gen_config.py``), and the pipeline diagrams name
settings in their labels. These tests guard both against drift.
"""

import re
from pathlib import Path

import pytest

from ontocast.cli.plot_graph import facts_loop_flow, ontology_loop_flow
from ontocast.config.env_names import env_names, iter_settings_fields
from ontocast.config.settings import Config, LLMConfig, ServerConfig

pytestmark = pytest.mark.unit

_ENV_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

#: Wording that describes how a setting changed rather than what it does.
#: That belongs in the changelog, not in the reference.
_HISTORY = re.compile(
    r"\b(previously|no longer|formerly|historical(?:ly)?|legacy|"
    r"deprecated|superseded|was renamed|used to be|in \d+\.\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def _known_env_names() -> set[str]:
    return {name for field in iter_settings_fields(Config) for name in field.env_names}


def test_env_names_follow_prefix_and_aliases() -> None:
    assert env_names(LLMConfig, "provider") == ("LLM_PROVIDER",)
    # A validation_alias bypasses the prefix.
    assert env_names(LLMConfig, "llm_max_inflight") == (
        "LLM_MAX_INFLIGHT",
        "MAX_INFLIGHT",
    )
    assert env_names(ServerConfig, "max_visits_per_node") == (
        "MAX_VISITS_PER_NODE",
        "MAX_VISITS",
    )


def test_every_setting_has_a_description() -> None:
    undescribed = [
        f.env_name
        for f in iter_settings_fields(Config)
        if not (f.info.description or "").strip()
    ]
    assert undescribed == []


def test_descriptions_state_behaviour_not_history() -> None:
    offenders = {
        f.env_name: match.group(0)
        for f in iter_settings_fields(Config)
        if (match := _HISTORY.search(f.info.description or ""))
    }
    assert offenders == {}


@pytest.mark.parametrize("include_evidence", [False, True])
@pytest.mark.parametrize("flow", [facts_loop_flow, ontology_loop_flow])
def test_diagram_labels_name_real_settings(flow, include_evidence: bool) -> None:
    graph = flow(include_evidence=include_evidence)
    labels = [n.label for n in graph.nodes] + [e.label for e in graph.edges]
    named = {token for label in labels for token in _ENV_TOKEN.findall(label)}
    assert named, "the loop diagrams are expected to name their budget settings"
    assert named <= _known_env_names()


#: Names in the docs that share a settings prefix but are not settings: test
#: harness variables, enum members, an error code and a query constant.
_DOCS_NON_SETTINGS = {
    "FACTS_PHASE",
    "ONTOLOGY_PHASE",
    "ONTOLOGY_AND_FACTS",
    "ONTOLOGY_HEADER_QUERY",
    "ONTOCAST_RECALL_CORPUS",
    "ONTOCAST_RECALL_JSON",
    "ONTOCAST_RUN_MANUAL_TESTS",
    "VECTOR_STORE_UNAVAILABLE",
}


def test_docs_name_only_real_settings() -> None:
    """A renamed or removed setting must not live on in the docs."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    if not docs.exists():
        pytest.skip("docs/ is not shipped with this checkout")
    fields = list(iter_settings_fields(Config))
    known = _known_env_names() | _DOCS_NON_SETTINGS
    prefixes = tuple(
        {str(f.group.model_config.get("env_prefix", "")).upper() for f in fields} - {""}
    )
    unknown = {
        f"{page.relative_to(docs)}: {token}"
        for page in docs.rglob("*.md")
        for token in _ENV_TOKEN.findall(page.read_text())
        if token.startswith(prefixes) and token not in known
    }
    assert sorted(unknown) == []
