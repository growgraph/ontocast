"""Classify environment assignments against the declared settings.

Pydantic-settings ignores an environment variable that no field reads, so a
renamed or removed setting in a configuration file silently does nothing, and a
value copied from an earlier default keeps overriding the current one. This
module tells the cases apart. Findings name variables, never their values.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum

from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings

from ontocast.config.env_names import SettingsField, iter_settings_fields
from ontocast.config.settings import Config


class Verdict(StrEnum):
    """What one assignment does relative to the current defaults."""

    UNKNOWN = "unknown"
    """No setting reads the variable: it is renamed, removed, or not OntoCast's."""
    INVALID = "invalid"
    """The setting rejects the value, alone or combined with the file's others."""
    REDUNDANT = "redundant"
    """The value equals the current default; deleting the line changes nothing."""
    PINNED = "pinned"
    """The value differs from the current default, which it overrides."""


class Finding(BaseModel):
    """The verdict on one assignment."""

    name: str
    verdict: Verdict
    detail: str = ""


class AuditReport(BaseModel):
    """Per-variable findings plus errors only the full configuration raises."""

    findings: list[Finding]
    config_errors: list[str]

    @property
    def ok(self) -> bool:
        """True when nothing is unknown or invalid and the whole config builds."""
        return not self.config_errors and all(
            f.verdict in (Verdict.REDUNDANT, Verdict.PINNED) for f in self.findings
        )


def declared_env_names() -> dict[str, SettingsField]:
    """Map every environment name a setting reads to the field reading it."""
    declared: dict[str, SettingsField] = {}
    for field in iter_settings_fields(Config):
        for name in field.env_names:
            declared.setdefault(name, field)
    return declared


@contextmanager
def _environment(
    assignments: Mapping[str, str], declared: Collection[str]
) -> Iterator[None]:
    """Run with only ``assignments`` among the declared names; restore after.

    Settings read names case-insensitively, so any spelling of a declared name
    is removed. Log output is suppressed: settings validators warn about
    combinations, and probing builds many of them.
    """
    saved = dict(os.environ)
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for key in list(os.environ):
            if key.upper() in declared:
                del os.environ[key]
        os.environ.update(assignments)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        logging.disable(previous_disable)


def _messages(exc: Exception) -> str:
    """Error text without the rejected input, so secrets are never echoed."""
    if isinstance(exc, ValidationError):
        return "; ".join(e["msg"] for e in exc.errors(include_input=False))
    return str(exc)


def _build(
    group: type[BaseSettings],
    assignments: Mapping[str, str],
    declared: Collection[str],
) -> tuple[BaseSettings | None, str]:
    with _environment(assignments, declared):
        try:
            return group(), ""
        except (ValidationError, ValueError) as exc:
            return None, _messages(exc)


def audit_assignments(assignments: Mapping[str, str | None]) -> AuditReport:
    """Classify each assignment as unknown, invalid, redundant or pinned.

    Each settings group is built from the file's assignments to it, so values
    are judged as the running program would parse them. A group that rejects
    the combination is retried one variable at a time to say which value is at
    fault; values that are each valid but jointly rejected are all reported
    invalid. The ambient environment never takes part.

    Args:
        assignments: Variable name to value, as read from one file. ``None``
            stands for a line with no ``=``.

    Returns:
        Findings in input order, and any error raised only when the complete
        configuration is built from the valid assignments.
    """
    declared = declared_env_names()
    findings: dict[str, Finding] = {}
    known: dict[str, str] = {}
    by_group: dict[type[BaseSettings], list[str]] = {}

    for name, value in assignments.items():
        field = declared.get(name.upper())
        if field is None:
            findings[name] = Finding(name=name, verdict=Verdict.UNKNOWN)
        elif value is None:
            findings[name] = Finding(
                name=name, verdict=Verdict.INVALID, detail="no value assigned"
            )
        else:
            known[name] = value
            by_group.setdefault(field.group, []).append(name)

    def classify(name: str, instance: BaseSettings, defaults: BaseSettings) -> None:
        attr = declared[name.upper()].name
        same = getattr(instance, attr) == getattr(defaults, attr)
        findings[name] = Finding(
            name=name, verdict=Verdict.REDUNDANT if same else Verdict.PINNED
        )

    for group, names in by_group.items():
        defaults, error = _build(group, {}, declared)
        if defaults is None:
            raise RuntimeError(f"{group.__name__} rejects its own defaults: {error}")
        instance, error = _build(group, {n: known[n] for n in names}, declared)
        if instance is not None:
            for name in names:
                classify(name, instance, defaults)
            continue
        alone_invalid = False
        for name in names:
            single, single_error = _build(group, {name: known[name]}, declared)
            if single is None:
                alone_invalid = True
                findings[name] = Finding(
                    name=name, verdict=Verdict.INVALID, detail=single_error
                )
            else:
                classify(name, single, defaults)
        if not alone_invalid:
            for name in names:
                findings[name] = Finding(
                    name=name,
                    verdict=Verdict.INVALID,
                    detail=f"rejected in combination: {error}",
                )

    valid = {
        n: v
        for n, v in known.items()
        if findings[n].verdict in (Verdict.REDUNDANT, Verdict.PINNED)
    }
    config_errors: list[str] = []
    _, error = _build(Config, valid, declared)
    if error:
        config_errors.append(error)

    return AuditReport(
        findings=[findings[name] for name in assignments],
        config_errors=config_errors,
    )
