"""Generic caching functionality for OntoCast tools.

This module provides a generic caching mechanism that can be used by various
tools to cache their results based on input content and configuration parameters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ontocast.onto.constants import DEFAULT_CACHE_MAX_BYTES, DEFAULT_CACHE_PRUNE_EVERY
from ontocast.util.hash import render_bytes_hash, render_text_hash

logger = logging.getLogger(__name__)

__all__ = [
    "CacheConfig",
    "Cacher",
    "PruneReport",
    "ToolCacher",
    "CHUNKER_CACHE_SUBDIR",
    "CONVERTER_CACHE_SUBDIR",
    "DEFAULT_CACHE_MAX_BYTES",
    "DEFAULT_CACHE_PRUNE_EVERY",
    "KNOWN_CACHE_SUBDIRS",
    "LLM_CACHE_SUBDIR",
]

# Cache-key discriminators. None is a meaningful value here, not an absent one:
# an unset base_url or num_ctx still has to take part in the key, and dropping
# it produces entries that hash differently from the ones the reader looks up.
type CacheConfig = dict[str, str | int | float | bool | None]

# Subdirectory names the current code writes to. Anything else under the cache
# directory is a leftover from an older layout; `ontocast cache prune
# --orphaned` clears those.
LLM_CACHE_SUBDIR = "llm"
CHUNKER_CACHE_SUBDIR = "chunker"
# Kept at the historical name deliberately. Renaming it to "converter" would
# collide with the v1 directory still present on older installs, whose entries
# have a different result shape; versioning now happens inside the key instead.
CONVERTER_CACHE_SUBDIR = "converter_v3"
KNOWN_CACHE_SUBDIRS = frozenset(
    {LLM_CACHE_SUBDIR, CHUNKER_CACHE_SUBDIR, CONVERTER_CACHE_SUBDIR}
)


class _Unset:
    """Sentinel for "argument not supplied", distinct from an explicit None."""


UNSET = _Unset()


class SubdirStats(BaseModel):
    """File count and size for one tool's cache subdirectory."""

    files: int = Field(default=0, description="Number of cache entries.")
    size_bytes: int = Field(default=0, description="Disk used, in bytes.")


class CacheStats(BaseModel):
    """Cache size, in total and per tool subdirectory."""

    total_files: int = Field(default=0, description="Number of cache entries.")
    total_size_bytes: int = Field(default=0, description="Disk used, in bytes.")
    subdirectories: dict[str, SubdirStats] = Field(
        default_factory=dict, description="Per-tool breakdown."
    )


class PruneReport(BaseModel):
    """Outcome of a :meth:`Cacher.prune` pass."""

    files_removed: int = Field(default=0, description="Cache entries deleted.")
    bytes_reclaimed: int = Field(default=0, description="Disk freed, in bytes.")
    bytes_remaining: int = Field(
        default=0, description="Total cache size after pruning, in bytes."
    )

    @property
    def changed(self) -> bool:
        """True when the pass actually deleted something."""
        return self.files_removed > 0


def _running_under_pytest() -> bool:
    """Whether this process is a test run.

    Checked via ``PYTEST_CURRENT_TEST``/``sys.modules`` rather than the shell's
    ``$_`` variable: under ``uv run pytest`` (the documented way to run this
    suite) ``$_`` holds the path to ``uv``, so the old probe never fired and
    tests silently shared the developer's real cache.
    """
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _unlink(path: Path) -> bool:
    """Delete a cache entry, tolerating a concurrent deletion.

    Returns:
        bool: True when this call removed the file.
    """
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning("Could not remove cache file %s: %s", path, e)
        return False


def _get_default_cache_dir() -> Path:
    """Get the default cache directory based on the environment.

    Returns:
        Path: The appropriate cache directory path.
    """
    # Check if we're in a test environment
    if _running_under_pytest():
        # In tests, use a test-specific cache directory
        return Path.cwd() / ".test_cache"

    # Check for common cache environment variables
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "ontocast"

    # Use platform-appropriate cache directory
    if os.name == "nt":  # Windows
        cache_dir = Path.home() / "AppData" / "Local" / "ontocast"
    else:  # Unix-like systems
        cache_dir = Path.home() / ".cache" / "ontocast"

    return cache_dir


class Cacher:
    """Shared caching class for OntoCast tools.

    This class provides a unified interface for caching results from various
    tools based on input content and configuration parameters. It manages
    multiple subdirectories for different tools from a single instance.

    Attributes:
        cache_dir: Base directory for caching.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        config: object | None = None,
        max_bytes: int | None | _Unset = UNSET,
        ttl_days: int | None | _Unset = UNSET,
        prune_every: int | _Unset = UNSET,
    ):
        """Initialize the shared cacher.

        Args:
            cache_dir: Base directory for caching. If None, uses config or platform-appropriate default.
            config: Optional config object to get cache_dir from.
            max_bytes: Size ceiling for the whole cache directory. ``None`` or
                ``0`` disables automatic pruning. Omit to take the configured
                value, or the 1 GB default when there is no config.
            ttl_days: Drop entries not used for this many days. ``None``
                disables the age cut.
            prune_every: Re-check the size ceiling after this many writes.
        """
        path_cfg = None
        if config is not None:
            tool_cfg = getattr(config, "tool_config", None)
            path_cfg = (
                getattr(tool_cfg, "path_config", None) if tool_cfg is not None else None
            )

        if cache_dir is None and path_cfg is not None:
            cache_dir = path_cfg.cache_dir

        if cache_dir is None:
            cache_dir = _get_default_cache_dir()

        # Explicit arguments win over config, so callers (and the CLI) can
        # override a configured policy for a single run. A sentinel rather than
        # None distinguishes "not specified" from "explicitly disabled".
        if isinstance(max_bytes, _Unset):
            max_bytes = (
                path_cfg.cache_max_bytes
                if path_cfg is not None
                else DEFAULT_CACHE_MAX_BYTES
            )
        if isinstance(ttl_days, _Unset):
            ttl_days = path_cfg.cache_ttl_days if path_cfg is not None else None
        if isinstance(prune_every, _Unset):
            prune_every = (
                path_cfg.cache_prune_every
                if path_cfg is not None
                else DEFAULT_CACHE_PRUNE_EVERY
            )

        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.ttl_days = ttl_days
        self.prune_every = max(1, prune_every)
        self._writes_since_prune = 0
        # Subdirectories already created, so the hot path skips the mkdir
        # syscall it used to pay on every single get and set.
        self._known_subdirs: set[str] = set()
        logger.debug(f"Shared cache directory set to: {self.cache_dir}")

    def _get_tool_cache_dir(self, subdirectory: str) -> Path:
        """Get the cache directory for a specific tool subdirectory.

        Args:
            subdirectory: The tool subdirectory name.

        Returns:
            Path: The full path to the tool's cache directory.
        """
        tool_cache_dir = self.cache_dir / subdirectory
        if subdirectory not in self._known_subdirs:
            tool_cache_dir.mkdir(parents=True, exist_ok=True)
            self._known_subdirs.add(subdirectory)
        return tool_cache_dir

    def _generate_cache_key(
        self,
        content: str | bytes,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> str:
        """Generate a cache key based on content and configuration.

        Args:
            content: The input content (text, bytes, etc.).
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.

        Returns:
            str: A hash string to use as cache key.
        """
        # Binary content is hashed directly. Decoding it with errors="ignore"
        # first -- as this did once -- throws away most of a PDF before hashing,
        # leaving the key resting on whichever bytes happen to be valid UTF-8.
        if isinstance(content, bytes):
            content_repr = f"sha256:{render_bytes_hash(content, digits=None)}"
        else:
            content_repr = str(content)

        # Create a dictionary with all relevant parameters
        cache_data = {
            "content": content_repr,
            "config": config or {},
            "kwargs": kwargs,
        }

        # Convert to JSON string and hash it
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return render_text_hash(cache_string, digits=None)

    def _get_cache_file_path(self, cache_key: str, subdirectory: str) -> Path:
        """Get the cache file path for a given cache key and subdirectory.

        Args:
            cache_key: The cache key.
            subdirectory: The tool subdirectory name.

        Returns:
            Path: The path to the cache file.
        """
        tool_cache_dir = self._get_tool_cache_dir(subdirectory)
        return tool_cache_dir / f"{cache_key}.json"

    def get(
        self,
        content: str | bytes,
        subdirectory: str,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> str | dict | list | None:
        """Get cached result for given content and configuration.

        Args:
            content: The input content.
            subdirectory: The tool subdirectory name.
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.

        Returns:
            Optional[Any]: The cached result or None if not found.
        """
        cache_key = self._generate_cache_key(content, config, **kwargs)
        cache_file = self._get_cache_file_path(cache_key, subdirectory)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                return cached_data.get("result")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None

    def set(
        self,
        content: str | bytes,
        result: str | dict | list,
        subdirectory: str,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> None:
        """Cache a result for given content and configuration.

        Args:
            content: The input content.
            result: The result to cache.
            subdirectory: The tool subdirectory name.
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.
        """
        cache_key = self._generate_cache_key(content, config, **kwargs)
        cache_file = self._get_cache_file_path(cache_key, subdirectory)

        # Prepare data for caching
        cache_data = {
            "result": result,
            "content": str(content)[:100] + "..."
            if len(str(content)) > 100
            else str(content),
            "config": config or {},
            "kwargs": kwargs,
        }

        # Write to a private temp file and rename into place. A plain open(...,
        # "w") truncates first, so with PARALLEL_WORKERS units in flight -- or a
        # Ctrl-C mid-write -- a concurrent reader sees half a JSON document.
        # os.replace is atomic on POSIX and Windows.
        #
        # mkstemp rather than a pid-derived name: two threads writing the *same*
        # key concurrently (two units with identical prompts) would otherwise
        # share one temp path and interleave into it, so the file renamed into
        # place would itself be corrupt.
        handle, tmp_name = tempfile.mkstemp(
            dir=cache_file.parent, prefix=f"{cache_file.stem}.", suffix=".tmp"
        )
        tmp_file = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, default=str)
            os.replace(tmp_file, cache_file)
            logger.debug(f"Cached result to {cache_file}")
        except (IOError, OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")
            tmp_file.unlink(missing_ok=True)
            return

        self._note_write()

    def _note_write(self) -> None:
        """Count a write and prune once the threshold is crossed.

        Pruning is advisory, so the counter is deliberately lock-free: under
        concurrency a check may be missed or run twice, and neither matters.
        """
        if not self.max_bytes and self.ttl_days is None:
            return
        self._writes_since_prune += 1
        if self._writes_since_prune < self.prune_every:
            return
        self._writes_since_prune = 0
        self.prune()

    async def aget(
        self,
        content: str | bytes,
        subdirectory: str,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> str | dict | list | None:
        """Async :meth:`get`, running the disk read off the event loop."""
        return await asyncio.to_thread(
            self.get, content, subdirectory, config, **kwargs
        )

    async def aset(
        self,
        content: str | bytes,
        result: str | dict | list,
        subdirectory: str,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> None:
        """Async :meth:`set`, running the disk write off the event loop.

        The write can trigger a prune, which walks the whole cache tree -- a
        good reason on its own not to do this inline on the loop.
        """
        await asyncio.to_thread(
            self.set, content, result, subdirectory, config, **kwargs
        )

    @staticmethod
    def _entries_under(root: Path, pattern: str) -> list[tuple[Path, int, float]]:
        """Cache entries under ``root`` as ``(path, size_bytes, last_used_epoch)``.

        ``atime`` is the recency signal: an entry earns its disk by being
        *read*, and ``mtime`` would evict entries that are written once and hit
        constantly. On ``noatime`` mounts ``atime`` stops advancing and can fall
        behind ``mtime``, so the newer of the two is used.
        """
        entries: list[tuple[Path, int, float]] = []
        for path in root.glob(pattern):
            try:
                stat = path.stat()
            except OSError:
                # A concurrent worker may have replaced or removed it.
                continue
            entries.append((path, stat.st_size, max(stat.st_atime, stat.st_mtime)))
        return entries

    def _entries(self) -> list[tuple[Path, int, float]]:
        """Every cache entry as ``(path, size_bytes, last_used_epoch)``."""
        return self._entries_under(self.cache_dir, "**/*.json")

    def prune(
        self,
        max_bytes: int | None = None,
        ttl_days: int | None = None,
    ) -> PruneReport:
        """Bound the cache by age and total size.

        Expired entries are dropped first, then least-recently-used entries
        until the total fits under the ceiling. Only regenerable cache entries
        under :attr:`cache_dir` are ever touched.

        Args:
            max_bytes: Size ceiling; defaults to the configured one. ``None``
                or ``0`` skips the size pass.
            ttl_days: Age cut in days; defaults to the configured one.

        Returns:
            PruneReport: What was removed and what remains.
        """
        limit = self.max_bytes if max_bytes is None else max_bytes
        ttl = self.ttl_days if ttl_days is None else ttl_days

        entries = self._entries()
        removed = 0
        reclaimed = 0

        if ttl is not None and ttl > 0:
            cutoff = time.time() - ttl * 86400
            kept: list[tuple[Path, int, float]] = []
            for path, size, used in entries:
                if used < cutoff:
                    if _unlink(path):
                        removed += 1
                        reclaimed += size
                else:
                    kept.append((path, size, used))
            entries = kept

        total = sum(size for _, size, _ in entries)

        if limit:
            # Least-recently-used first.
            for path, size, _ in sorted(entries, key=lambda item: item[2]):
                if total <= limit:
                    break
                if _unlink(path):
                    removed += 1
                    reclaimed += size
                    total -= size

        report = PruneReport(
            files_removed=removed, bytes_reclaimed=reclaimed, bytes_remaining=total
        )
        if report.changed:
            logger.info(
                "Pruned cache %s: removed %s entries, reclaimed %.1f MB, %.1f MB remain",
                self.cache_dir,
                report.files_removed,
                report.bytes_reclaimed / 1e6,
                report.bytes_remaining / 1e6,
            )
        return report

    async def aprune(
        self,
        max_bytes: int | None = None,
        ttl_days: int | None = None,
    ) -> PruneReport:
        """Async :meth:`prune`, running the tree walk off the event loop."""
        return await asyncio.to_thread(self.prune, max_bytes, ttl_days)

    def prune_orphaned_subdirs(self, live_subdirs: Collection[str]) -> PruneReport:
        """Remove cache subdirectories no current tool writes to.

        Kept manual (see the ``ontocast cache prune --orphaned`` command):
        "no live tool claims this" is an inference that a downgrade or a
        third-party tool can invalidate, unlike the size pass, which only ever
        discards entries the current code would regenerate.

        Args:
            live_subdirs: Subdirectory names still in use.

        Returns:
            PruneReport: What was removed.
        """
        removed = 0
        reclaimed = 0
        for child in sorted(self.cache_dir.iterdir()):
            if not child.is_dir() or child.name in live_subdirs:
                continue
            for path in child.glob("**/*"):
                if path.is_file():
                    size = path.stat().st_size
                    if _unlink(path):
                        removed += 1
                        reclaimed += size
            try:
                child.rmdir()
            except OSError as e:
                logger.warning("Could not remove cache subdirectory %s: %s", child, e)

        total = sum(size for _, size, _ in self._entries())
        return PruneReport(
            files_removed=removed, bytes_reclaimed=reclaimed, bytes_remaining=total
        )

    def clear(self, subdirectory: str | None = None) -> None:
        """Clear cached results.

        Args:
            subdirectory: If provided, clear only this subdirectory. If None, clear all.
        """
        if subdirectory is None:
            # Clear all subdirectories
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("**/*.json"):
                    _unlink(cache_file)
                logger.info(f"Cleared all cache directories: {self.cache_dir}")
        else:
            # Clear specific subdirectory
            tool_cache_dir = self._get_tool_cache_dir(subdirectory)
            if tool_cache_dir.exists():
                for cache_file in tool_cache_dir.glob("*.json"):
                    _unlink(cache_file)
                logger.info(f"Cleared cache directory: {tool_cache_dir}")

    def cache_stats(self, subdirectory: str | None = None) -> CacheStats:
        """Cache size, in total and per tool subdirectory.

        Walks the cache tree, so this is not a hot-path call; from async code
        use :meth:`~ontocast.tool.llm.LLMTool.aget_cache_stats` or wrap it in
        ``asyncio.to_thread``.

        Args:
            subdirectory: Restrict to one tool's subdirectory. None covers all,
                and is the only form that fills in ``subdirectories``.

        Returns:
            CacheStats: File counts and byte totals.
        """
        if subdirectory is not None:
            tool_cache_dir = self.cache_dir / subdirectory
            if not tool_cache_dir.exists():
                return CacheStats()
            sizes = [
                size for _, size, _ in self._entries_under(tool_cache_dir, "*.json")
            ]
            return CacheStats(total_files=len(sizes), total_size_bytes=sum(sizes))

        if not self.cache_dir.exists():
            return CacheStats()

        stats = CacheStats()
        for path, size, _ in self._entries():
            stats.total_files += 1
            stats.total_size_bytes += size
            subdir = stats.subdirectories.setdefault(path.parent.name, SubdirStats())
            subdir.files += 1
            subdir.size_bytes += size
        return stats

    def get_cache_stats(
        self,
        subdirectory: str | None = None,
    ) -> dict[str, Any]:
        """Cache statistics as a plain dict, for JSON responses.

        See :meth:`cache_stats` for a typed result.

        Args:
            subdirectory: If provided, get stats for this subdirectory only. If None, get stats for all.

        Returns:
            dict: Dictionary with cache statistics.
        """
        stats = self.cache_stats(subdirectory)
        if subdirectory is not None:
            return stats.model_dump(exclude={"subdirectories"})
        return stats.model_dump()


class ToolCacher:
    """Tool-specific wrapper for the shared Cacher.

    This class provides a tool-specific interface to the shared Cacher,
    automatically handling the subdirectory parameter.
    """

    def __init__(self, shared_cacher: Cacher, subdirectory: str):
        """Initialize the tool cacher.

        Args:
            shared_cacher: The shared Cacher instance.
            subdirectory: The subdirectory name for this tool.
        """
        self.shared_cacher = shared_cacher
        self.subdirectory = subdirectory

    def get(
        self,
        content: str | bytes,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> str | dict | list | None:
        """Get cached result for given content and configuration.

        Args:
            content: The input content.
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.

        Returns:
            Optional[Any]: The cached result or None if not found.
        """
        return self.shared_cacher.get(
            content=content, subdirectory=self.subdirectory, config=config, **kwargs
        )

    def set(
        self,
        content: str | bytes,
        result: str | dict | list,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> None:
        """Cache a result for given content and configuration.

        Args:
            content: The input content.
            result: The result to cache.
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.
        """
        self.shared_cacher.set(
            content=content,
            result=result,
            subdirectory=self.subdirectory,
            config=config,
            **kwargs,
        )

    async def aget(
        self,
        content: str | bytes,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> str | dict | list | None:
        """Async :meth:`get`, running the disk read off the event loop."""
        return await self.shared_cacher.aget(
            content=content, subdirectory=self.subdirectory, config=config, **kwargs
        )

    async def aset(
        self,
        content: str | bytes,
        result: str | dict | list,
        config: CacheConfig | None = None,
        **kwargs: str | int | float | bool,
    ) -> None:
        """Async :meth:`set`, running the disk write off the event loop."""
        await self.shared_cacher.aset(
            content=content,
            result=result,
            subdirectory=self.subdirectory,
            config=config,
            **kwargs,
        )

    def clear(self) -> None:
        """Clear cached results for this tool."""
        self.shared_cacher.clear(subdirectory=self.subdirectory)

    def get_cache_stats(
        self,
    ) -> dict[str, int | dict[str, int] | dict[str, dict[str, int]]]:
        """Get cache statistics for this tool.

        Returns:
            Dict[str, int]: Dictionary with cache statistics.
        """
        return self.shared_cacher.get_cache_stats(subdirectory=self.subdirectory)
