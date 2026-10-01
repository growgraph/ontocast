"""Environment-variable names of the settings fields.

Every leaf field of :class:`~ontocast.config.settings.Config` is read from the
environment by the ``BaseSettings`` group that declares it. The name is the
group's ``env_prefix`` plus the field name, upper-cased, unless the field
declares a ``validation_alias``: those names bypass the prefix.
"""

from collections.abc import Iterator

from pydantic import AliasChoices, BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings


class SettingsField(BaseModel):
    """One leaf settings field and the environment names it is read from."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    group: type[BaseSettings]
    name: str
    info: FieldInfo
    env_names: tuple[str, ...]

    @property
    def env_name(self) -> str:
        """The primary environment name: the first alias, or prefix + name."""
        return self.env_names[0]


def env_names(group: type[BaseSettings], field_name: str) -> tuple[str, ...]:
    """Return every environment name a settings field is read from.

    Args:
        group: The ``BaseSettings`` class that declares the field.
        field_name: The Python name of the field.

    Returns:
        The names in precedence order, upper-cased.
    """
    alias = group.model_fields[field_name].validation_alias
    if isinstance(alias, AliasChoices):
        names = tuple(c.upper() for c in alias.choices if isinstance(c, str))
        if names:
            return names
    if isinstance(alias, str):
        return (alias.upper(),)
    prefix = str(group.model_config.get("env_prefix", ""))
    return ((prefix + field_name).upper(),)


def iter_settings_fields(root: type[BaseSettings]) -> Iterator[SettingsField]:
    """Yield every leaf field under ``root``, depth first in declaration order.

    A field whose type is itself a ``BaseSettings`` group is descended into
    rather than yielded.
    """
    for name, info in root.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseSettings):
            yield from iter_settings_fields(annotation)
            continue
        yield SettingsField(
            group=root, name=name, info=info, env_names=env_names(root, name)
        )
