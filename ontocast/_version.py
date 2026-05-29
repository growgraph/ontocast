"""Installed package version (from ``pyproject.toml`` project metadata)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ontocast")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
