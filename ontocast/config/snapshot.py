"""The settings a run is configured with, as a JSON-safe, secret-free mapping.

Keyed by environment-variable name, so a run's record reads in the same
vocabulary as the ``.env`` that configured it, and two records diff key by key.
"""

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic_core import to_jsonable_python
from pydantic_settings import BaseSettings

from ontocast.config.env_names import iter_settings_values

#: Field names whose value is a credential. A new credential field must be
#: added here; ``test_config_snapshot.py`` fails on an unlisted name that looks
#: like one.
SECRET_FIELD_NAMES = frozenset({"api_key", "auth"})

REDACTED = "<redacted>"


def _strip_url_credentials(value: str) -> str:
    """Drop ``user:password@`` from a URL; any other string is returned as is."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or "@" not in parts.netloc:
        return value
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"{REDACTED}@{host}"))


def config_snapshot(settings: BaseSettings) -> dict[str, Any]:
    """Every leaf setting under ``settings`` with the value in force.

    Credentials are replaced by a marker (``None`` stays ``None``, so the record
    still says whether one was set), and credentials embedded in URLs are
    stripped.

    Args:
        settings: Usually the run's :class:`~ontocast.config.settings.Config`.

    Returns:
        ``{ENV_NAME: value}`` with JSON-safe values, in declaration order.
    """
    snapshot: dict[str, Any] = {}
    for field, value in iter_settings_values(settings):
        if field.name in SECRET_FIELD_NAMES:
            value = None if value is None else REDACTED
        elif isinstance(value, str):
            value = _strip_url_credentials(value)
        snapshot[field.env_name] = to_jsonable_python(value)
    return snapshot
