"""``.env.example`` is the advertised configuration surface; keep it true.

Every env-settable field in ``config/settings.py`` is either documented in
``.env.example`` or listed here as a deliberate omission. Without this the two
drift silently: a knob added to settings is invisible to anyone reading the
example file, and a variable removed from settings keeps being advertised as
live long after it became a no-op. Both had happened by 0.6.0 -- nine
``CHUNK_*`` settings were undocumented, including the one the performance guide
tells you to align.
"""

import re
from pathlib import Path

from pydantic import AliasChoices
from pydantic_settings import BaseSettings

import ontocast.config.settings as settings_module

#: Declared but deliberately absent from .env.example, with the reason.
DELIBERATELY_UNDOCUMENTED = {
    # Unprefixed alias of LLM_MAX_INFLIGHT, which is documented. Advertising
    # both spellings would invite setting them to different values.
    "MAX_INFLIGHT",
}


def _declared_env_vars() -> dict[str, str]:
    """Map every env-settable variable to the field that declares it."""
    declared: dict[str, str] = {}
    for name in dir(settings_module):
        obj = getattr(settings_module, name)
        if not (isinstance(obj, type) and issubclass(obj, BaseSettings)):
            continue
        prefix = (obj.model_config.get("env_prefix") or "").upper()
        for field_name, field in obj.model_fields.items():
            annotation = field.annotation
            # A nested BaseSettings field is a sub-model, not a scalar var.
            if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
                continue
            alias = field.validation_alias or field.alias
            if isinstance(alias, AliasChoices):
                # AliasChoices bypasses env_prefix: each choice is a literal
                # variable name.
                names = [str(choice).upper() for choice in alias.choices]
            elif isinstance(alias, str):
                names = [alias.upper()]
            else:
                names = [f"{prefix}{field_name}".upper()]
            for env_name in names:
                declared.setdefault(env_name, f"{name}.{field_name}")
    return declared


def _documented_env_vars(filename: str = ".env.example") -> set[str]:
    """Variables assigned in an example env file, commented-out lines included."""
    example = Path(__file__).resolve().parents[1] / filename
    return set(
        re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", example.read_text(), re.MULTILINE)
    )


def test_every_setting_is_documented_or_deliberately_omitted() -> None:
    declared = _declared_env_vars()
    documented = _documented_env_vars()

    missing = {
        name: field
        for name, field in declared.items()
        if name not in documented and name not in DELIBERATELY_UNDOCUMENTED
    }

    assert missing == {}, (
        "Undocumented settings -- add them to .env.example, or to "
        f"DELIBERATELY_UNDOCUMENTED with a reason: {sorted(missing)}"
    )


def test_env_example_advertises_no_removed_settings() -> None:
    """A documented variable that no field reads is a silent no-op.

    This is the direction that bites hardest on an upgrade: 0.6.0 removed
    CHUNK_BREAKPOINT_THRESHOLD_TYPE/_AMOUNT and renamed FACTS_REPAIR_VISITS,
    and an unknown variable is ignored rather than rejected.
    """
    declared = _declared_env_vars()
    documented = _documented_env_vars()

    stale = sorted(name for name in documented if name not in declared)

    assert stale == [], f".env.example advertises variables no setting reads: {stale}"


#: The three settings that each name a local sentence-transformer checkpoint.
#: `SharedEncoder` caches by the literal `(model name, device)` string, so these
#: only share one resident model when the spellings match character for
#: character -- a bare name and a prefixed one resolve to the same files on the
#: hub and still load twice.
_LOCAL_ENCODER_VARS = (
    "CHUNK_EMBEDDING_MODEL",
    "EMBEDDING_MODEL_NAME",
    "AGG_EMBEDDING_MODEL",
)


def test_env_example_spells_encoder_models_with_the_org_prefix() -> None:
    """A dropped `sentence-transformers/` prefix costs ~650 MB, silently.

    `.env.example` shipped `AGG_EMBEDDING_MODEL` and `EMBEDDING_MODEL_NAME`
    unprefixed while the declared defaults carried the prefix, so copying the
    file and then following the performance guide's advice to align all three
    produced *two copies of the same checkpoint* -- the exact outcome the
    alignment is meant to avoid. Nothing else detects this: both spellings are
    valid and both work.
    """
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()

    unprefixed = []
    for var in _LOCAL_ENCODER_VARS:
        for value in re.findall(rf"^#?\s*{var}=(\S+)", text, re.MULTILINE):
            if not value.startswith("sentence-transformers/"):
                unprefixed.append(f"{var}={value}")

    assert unprefixed == [], (
        "These name a local encoder without the `sentence-transformers/` prefix, "
        "so they will not share a resident model with the prefixed defaults: "
        f"{unprefixed}"
    )


#: The point of `.env.example.minimal` is that it is short enough to read in one
#: sitting. Without a ceiling it accretes back toward the full surface one
#: "obviously this matters too" at a time, and the playbooks stop being usable.
#:
#: Raised from 35 once, deliberately, to cover chunking, conversion, the local
#: encoder alignment, SHACL shapes and the web-search toggle -- all of which are
#: decisions a first-time user has to make. Roughly a quarter of the full
#: surface is the intended shape; past that, reconsider rather than raise again.
MINIMAL_ENV_CEILING = 55


def test_minimal_env_example_names_only_real_settings() -> None:
    """A curated file rots more quietly than a generated one.

    Nothing regenerates `.env.example.minimal`, so a renamed or removed setting
    leaves a variable in it that silently does nothing -- worse than an
    undocumented knob, because the user believes they configured something.
    """
    declared = _declared_env_vars()
    minimal = _documented_env_vars(".env.example.minimal")

    unknown = sorted(name for name in minimal if name not in declared)

    assert unknown == [], (
        f".env.example.minimal names variables no setting reads: {unknown}"
    )


def test_minimal_env_example_is_a_subset_of_the_full_one() -> None:
    """The minimal file is a curated view, not a second source of truth."""
    full = _documented_env_vars()
    minimal = _documented_env_vars(".env.example.minimal")

    only_in_minimal = sorted(minimal - full)

    assert only_in_minimal == [], (
        "These are in .env.example.minimal but not .env.example, so the full "
        f"reference is missing them: {only_in_minimal}"
    )


#: Files that quote a variable count at the reader. Prose numbers rot the moment
#: either env file changes -- five of these went stale in a single edit.
_FILES_QUOTING_COUNTS = (
    "README.md",
    ".env.example",
    "docs/index.md",
    "docs/user_guide/configuration.md",
)


def test_quoted_variable_counts_are_accurate() -> None:
    """ "46 variables instead of 201" has to stay true, or stop being written.

    Only *exact* claims are checked. Hedged prose ("around 200 variables") is
    deliberately approximate and should not have to churn every time a knob is
    added -- that is the point of hedging it.
    """
    root = Path(__file__).resolve().parents[1]
    real = {
        len(_documented_env_vars()),
        len(_documented_env_vars(".env.example.minimal")),
    }
    hedged = re.compile(r"(?:around|roughly|about|~|under|over)\s*$", re.IGNORECASE)

    wrong: list[str] = []
    for name in _FILES_QUOTING_COUNTS:
        text = (root / name).read_text()
        for match in re.finditer(r"(\d+)\s+variables", text):
            if hedged.search(text[: match.start()]):
                continue
            if int(match.group(1)) not in real:
                wrong.append(f"{name}: claims {match.group(1)}")

    assert wrong == [], (
        f"Stale variable counts (actual: {sorted(real)}): {wrong}. Update the "
        "prose, or rephrase it so it does not quote a number."
    )


def test_minimal_env_example_stays_minimal() -> None:
    minimal = _documented_env_vars(".env.example.minimal")

    assert len(minimal) <= MINIMAL_ENV_CEILING, (
        f".env.example.minimal lists {len(minimal)} variables, over the "
        f"{MINIMAL_ENV_CEILING} ceiling. Move one out before adding another, or "
        "raise the ceiling deliberately."
    )
