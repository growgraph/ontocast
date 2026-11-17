"""``ontocast check``: exit codes, output and directory inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ontocast.check import FactsCheckReport
from ontocast.cli.check import check as check_command

pytestmark = pytest.mark.unit

_ONTOLOGY = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix c: <https://catalog.test/o#> .

<https://catalog.test/o> a owl:Ontology .
c:Thing a owl:Class ; rdfs:label "Thing" .
c:knows a owl:ObjectProperty ; rdfs:domain c:Thing ; rdfs:range c:Thing .
"""

_CLEAN = """\
@prefix c: <https://catalog.test/o#> .
@prefix d: <https://data.test/d/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

d:a a c:Thing ; rdfs:label "A" ; c:knows d:b .
d:b a c:Thing ; rdfs:label "B" .
"""

_BAD = """\
@prefix c: <https://catalog.test/o#> .
@prefix d: <https://data.test/d/> .
@prefix x: <https://other.test/x#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

d:a a c:Thing ; rdfs:label "A" ; x:invented d:b .
d:b a c:Thing ; rdfs:label "B" .
"""


def _setup(tmp_path: Path, facts: str) -> tuple[Path, Path]:
    onto = tmp_path / "onto"
    onto.mkdir()
    (onto / "o.ttl").write_text(_ONTOLOGY, encoding="utf-8")
    path = tmp_path / "facts.ttl"
    path.write_text(facts, encoding="utf-8")
    return onto, path


def _run(onto: Path, *args: str):
    return CliRunner().invoke(check_command, ["--ontology-dir", str(onto), *args])


def test_clean_graph_exit_0(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _CLEAN)
    result = _run(onto, str(facts))
    assert result.exit_code == 0, result.output


def test_error_findings_exit_1(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _BAD)
    result = _run(onto, str(facts))
    assert result.exit_code == 1, result.output
    assert "non_catalog_vocabulary" in result.output


def test_parse_error_exit_2(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, "this is not turtle {{{")
    result = _run(onto, str(facts))
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_json_output_parses_as_report(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _BAD)
    result = _run(onto, str(facts), "--json")
    assert result.exit_code == 1
    report = FactsCheckReport.model_validate_json(result.output)
    assert report.counts


def test_open_vocabulary_downgrades_to_warning(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _BAD)
    result = _run(onto, str(facts), "--vocabulary", "open")
    assert result.exit_code == 0, result.output


def test_files_are_merged(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _CLEAN)
    extra = tmp_path / "more.ttl"
    extra.write_text(_BAD.replace("d:a a", "d:z a"), encoding="utf-8")
    result = _run(onto, str(facts), str(extra), "--json")
    assert result.exit_code == 1
    assert FactsCheckReport.model_validate_json(result.output).metrics.instances >= 3


_PLACEHOLDER_ONTOLOGY = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/onto#> .

ex:Thing a owl:Class .
ex:knows a owl:ObjectProperty ; rdfs:domain ex:Thing ; rdfs:range ex:Thing .
"""

_PLACEHOLDER_CLEAN = """\
@prefix ex: <http://example.org/onto#> .
@prefix d: <http://example.org/data/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

d:a a ex:Thing ; rdfs:label "A" ; ex:knows d:b .
d:b a ex:Thing ; rdfs:label "B" .
"""


def test_placeholder_namespace_catalog_clean_graph_exit_0(tmp_path: Path) -> None:
    onto = tmp_path / "onto"
    onto.mkdir()
    (onto / "o.ttl").write_text(_PLACEHOLDER_ONTOLOGY, encoding="utf-8")
    facts = tmp_path / "facts.ttl"
    facts.write_text(_PLACEHOLDER_CLEAN, encoding="utf-8")
    result = _run(onto, str(facts))
    assert result.exit_code == 0, result.output
    assert "unknown_term" not in result.output


def test_zero_triples_exit_2(tmp_path: Path) -> None:
    onto, _ = _setup(tmp_path, _CLEAN)
    path = tmp_path / "data.json"
    path.write_text('{"name": "not JSON-LD"}', encoding="utf-8")
    result = _run(onto, str(path))
    assert result.exit_code == 2, result.output
    assert "no triples" in result.output


def test_text_output_starts_with_a_summary(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _BAD)
    result = _run(onto, str(facts))
    first = result.output.splitlines()[0]
    assert "triples" in first
    assert "instances" in first
    assert "1 error" in first


def test_unknown_suffix_message_is_for_the_cli(tmp_path: Path) -> None:
    onto, _ = _setup(tmp_path, _CLEAN)
    path = tmp_path / "facts.dat"
    path.write_text(_CLEAN, encoding="utf-8")
    result = _run(onto, str(path))
    assert result.exit_code == 2
    assert "format=" not in result.output
    assert ".ttl" in result.output


def test_tenant_with_directories_is_a_usage_error(tmp_path: Path) -> None:
    onto, facts = _setup(tmp_path, _CLEAN)
    result = _run(onto, str(facts), "--tenant", "t", "--project", "p")
    assert result.exit_code == 2
    assert "--tenant" in result.output


class _FakeTools:
    def __init__(self, catalog) -> None:
        self.catalog = catalog
        self.closed = False

    async def catalog_graphs(self):
        return self.catalog, None

    async def aclose(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self, tools: _FakeTools) -> None:
        self.tools = tools


def test_tenant_path_asks_for_no_summaries_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ontocast.cli import server as server_module
    from ontocast.onto.rdfgraph import RDFGraph

    catalog = RDFGraph()
    catalog.parse(data=_ONTOLOGY, format="turtle")
    tools = _FakeTools(catalog)
    seen: dict = {}

    async def bootstrap(**kwargs):
        seen.update(kwargs)
        return _FakeRuntime(tools)

    monkeypatch.setattr(server_module, "_abootstrap_tools", bootstrap)
    _, facts = _setup(tmp_path, _CLEAN)
    result = CliRunner().invoke(
        check_command, [str(facts), "--tenant", "t", "--project", "p"]
    )
    assert result.exit_code == 0, result.output
    assert seen["tenant"] == "t" and seen["project"] == "p"
    assert seen["summarize_ontologies"] is False
    assert tools.closed


@pytest.mark.parametrize("error", [RuntimeError("store unreachable"), None])
def test_tenant_bootstrap_failure_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception | None
) -> None:
    import click

    from ontocast.cli import server as server_module

    failure = error or click.ClickException("LLM_API_KEY is required")

    async def bootstrap(**_kwargs):
        raise failure

    monkeypatch.setattr(server_module, "_abootstrap_tools", bootstrap)
    _, facts = _setup(tmp_path, _CLEAN)
    result = CliRunner().invoke(
        check_command, [str(facts), "--tenant", "t", "--project", "p"]
    )
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output


def test_tenant_path_neither_wipes_nor_seeds_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ontocast.cli import server as server_module
    from ontocast.onto.rdfgraph import RDFGraph

    catalog = RDFGraph()
    catalog.parse(data=_ONTOLOGY, format="turtle")
    seen: dict = {}

    async def bootstrap(**kwargs):
        seen.update(kwargs)
        return _FakeRuntime(_FakeTools(catalog))

    monkeypatch.setattr(server_module, "_abootstrap_tools", bootstrap)
    _, facts = _setup(tmp_path, _CLEAN)
    result = CliRunner().invoke(
        check_command, [str(facts), "--tenant", "t", "--project", "p"]
    )
    assert result.exit_code == 0, result.output
    assert seen["wipe_vector_store"] is False
    # "" clears both seed directories, so initialize has nothing to sync.
    assert seen["ontology_dir"] == "" and seen["shapes_dir"] == ""


def test_cleared_seed_directories_leave_nothing_to_sync(tmp_path: Path) -> None:
    from ontocast.cli.server import _prepare_path_config
    from ontocast.config import Config

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "o.ttl").write_text(_ONTOLOGY, encoding="utf-8")
    config = Config()
    config.tool_config.path_config.ontology_directory = seeds
    config.tool_config.facts_validation.shapes_dir = str(seeds)
    _prepare_path_config(config, ontology_dir="", shapes_dir="")
    assert config.tool_config.path_config.ontology_directory is None
    assert config.tool_config.facts_validation.shapes_dir is None
