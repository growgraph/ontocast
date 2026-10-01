"""Generate the configuration reference from the settings models.

Every leaf field of ``ontocast.config.settings.Config`` becomes one entry under
its environment-variable name, with its type, default and ``Field``
description. The pages are virtual (``mkdocs-gen-files``); edit the
descriptions in ``ontocast/config/settings.py``, not here.

Each settings group must be assigned to exactly one page in ``PAGES``; an
unassigned group fails the build, so a new group is never left undocumented.
The pages are listed in the ``properdocs.yml`` nav; a page missing there fails
the strict build.
"""

import json
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

import mkdocs_gen_files
from pydantic import SecretStr
from pydantic.fields import FieldInfo

from ontocast.config.env_names import SettingsField, iter_settings_fields
from ontocast.config.settings import Config

OUT = Path("reference", "configuration")
SETTINGS_SOURCE = Path("..", "ontocast", "config", "settings.py")

# slug, title, one-line summary, settings groups in display order
PAGES: list[tuple[str, str, str, list[str]]] = [
    (
        "pipeline",
        "Pipeline and server",
        "What a run does, how wide it fans out, and where the server listens.",
        ["ServerConfig", "Config", "DomainConfig"],
    ),
    (
        "llm",
        "Language model",
        "Provider, model, credentials, request limits and the LLM response cache.",
        ["LLMConfig"],
    ),
    (
        "conversion",
        "Document conversion",
        "How PDF and PowerPoint files become text.",
        ["ConverterConfig"],
    ),
    (
        "chunking",
        "Chunking",
        "How text is cut into content units and which sections are kept.",
        ["ChunkConfig"],
    ),
    (
        "ontology-retrieval",
        "Ontology retrieval",
        "How each unit's slice of the ontology catalog is retrieved and sized.",
        ["VectorStoreConfig", "PatchRetrievalConfig"],
    ),
    (
        "embeddings",
        "Embeddings and vector stores",
        "The embedding model and the LanceDB or Qdrant backend behind retrieval.",
        ["EmbeddingConfig", "LanceDBConfig", "QdrantConfig"],
    ),
    (
        "facts-validation",
        "Facts validation",
        "The facts critic, repairs, acceptance rules and SHACL validation.",
        ["FactsValidationConfig"],
    ),
    (
        "ontology-validation",
        "Ontology validation",
        "The ontology critic and its acceptance rules.",
        ["OntologyValidationConfig"],
    ),
    (
        "aggregation",
        "Entity disambiguation",
        "When two extracted entities are merged into one.",
        ["AggregationConfig"],
    ),
    (
        "web-search",
        "Web search",
        "Optional web evidence for ontology rendering and critique.",
        ["WebSearchConfig"],
    ),
    (
        "storage",
        "Storage and paths",
        "The Fuseki triple store, seed ontologies and on-disk caches.",
        ["FusekiConfig", "PathConfig"],
    ),
]

GROUP_TITLES = {
    "ServerConfig": "Run",
    "Config": "Process",
    "DomainConfig": "Domain",
    "VectorStoreConfig": "Retrieval",
    "PatchRetrievalConfig": "Patch sizing",
    "EmbeddingConfig": "Embeddings",
    "LanceDBConfig": "LanceDB",
    "QdrantConfig": "Qdrant",
    "FusekiConfig": "Fuseki",
    "PathConfig": "Paths and caches",
}

_RST_LITERAL = re.compile(r"``(.+?)``")
_RST_ROLE = re.compile(
    r":(?:class|func|meth|attr|mod|data|obj):`~?(?:[\w.]+\.)?([\w]+)`"
)


def _type_text(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        return "one of " + ", ".join(f"`{a}`" for a in get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return "one of " + ", ".join(f"`{m.value}`" for m in annotation)
    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not NoneType]
        if str in args:
            # Any string is accepted, so listed known values are not a choice.
            args = [a for a in args if not _is_choice(a)]
        return " or ".join(_type_text(a) for a in args)
    if origin is not None:
        return f"`{_plain(annotation)}`"
    if isinstance(annotation, type):
        return f"`{annotation.__name__}`"
    return f"`{_plain(annotation)}`"


def _is_choice(annotation: Any) -> bool:
    return get_origin(annotation) is Literal or (
        isinstance(annotation, type) and issubclass(annotation, Enum)
    )


def _plain(annotation: Any) -> str:
    return re.sub(r"\b(?:typing|pathlib|pydantic(?:\.types)?)\.", "", str(annotation))


def _default_text(info: FieldInfo) -> str:
    if info.is_required():
        return "required"
    value = info.get_default(call_default_factory=True)
    if value is None or isinstance(value, SecretStr):
        return "unset"
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (set, frozenset)):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple, dict)):
        if not value:
            return "empty"
        text = json.dumps(value, default=str, ensure_ascii=False)
    else:
        text = str(value)
    if text == "":
        return "empty"
    if len(text) > 90:
        text = text[:87] + "..."
    return f"`{text}`"


def _description(info: FieldInfo) -> str:
    text = (info.description or "").strip()
    text = _RST_LITERAL.sub(r"`\1`", text)
    return _RST_ROLE.sub(r"`\1`", text)


def _write_field(f: Any, field: SettingsField) -> None:
    f.write(f"### `{field.env_name}`\n\n")
    meta = [
        f"Default: {_default_text(field.info)}",
        f"Type: {_type_text(field.info.annotation)}",
    ]
    if len(field.env_names) > 1:
        meta.append(
            "Also read from " + ", ".join(f"`{n}`" for n in field.env_names[1:])
        )
    f.write(" · ".join(meta) + "\n{ .oc-setting-meta }\n\n")
    f.write(_description(field.info) + "\n\n")


def main() -> None:
    by_group: dict[str, list[SettingsField]] = defaultdict(list)
    for field in iter_settings_fields(Config):
        by_group[field.group.__name__].append(field)

    assigned = [g for *_, groups in PAGES for g in groups]
    unassigned = sorted(set(by_group) - set(assigned))
    if unassigned:
        raise RuntimeError(f"settings groups missing from PAGES: {unassigned}")

    for slug, title, summary, groups in PAGES:
        path = OUT / f"{slug}.md"
        with mkdocs_gen_files.open(path, "w") as f:
            f.write(f"# {title}\n\n{summary}\n\n")
            for group in groups:
                if len(groups) > 1:
                    f.write(f"## {GROUP_TITLES[group]}\n\n")
                for field in by_group[group]:
                    _write_field(f, field)
        mkdocs_gen_files.set_edit_path(path, SETTINGS_SOURCE)

    total = sum(len(v) for v in by_group.values())
    with mkdocs_gen_files.open(OUT / "index.md", "w") as f:
        f.write(
            "# Configuration reference\n\n"
            "OntoCast reads its settings from environment variables, or from a "
            "`.env` file in the working directory. This reference lists all "
            f"{total} of them, generated from the settings models. To choose "
            "settings for a task, start with "
            "[Configuring OntoCast](../../guides/configuration.md).\n\n"
            "| Page | What it controls |\n|---|---|\n"
        )
        for slug, title, summary, _ in PAGES:
            f.write(f"| [{title}]({slug}.md) | {summary} |\n")
    mkdocs_gen_files.set_edit_path(OUT / "index.md", SETTINGS_SOURCE)


main()
