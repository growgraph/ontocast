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


def _documented_env_vars() -> set[str]:
    """Variables assigned in .env.example, commented-out lines included."""
    example = Path(__file__).resolve().parents[1] / ".env.example"
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
