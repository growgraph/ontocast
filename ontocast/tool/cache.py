"""Generic caching functionality for OntoCast tools.

This module provides a generic caching mechanism that can be used by various
tools to cache their results based on input content and configuration parameters.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ontocast.config import Config

logger = logging.getLogger(__name__)


def _get_default_cache_dir() -> Path:
    """Get the default cache directory based on the environment.

    Returns:
        Path: The appropriate cache directory path.
    """
    # Check if we're in a test environment
    if "pytest" in os.environ.get("_", ""):
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
        config: "Config | None" = None,
    ):
        """Initialize the shared cacher.

        Args:
            cache_dir: Base directory for caching. If None, uses config or platform-appropriate default.
            config: Optional config object to get cache_dir from.
        """
        if cache_dir is None and config is not None:
            # Try to get cache_dir from config
            if hasattr(config, "tool_config") and hasattr(
                config.tool_config, "path_config"
            ):
                cache_dir = config.tool_config.path_config.cache_dir

        if cache_dir is None:
            cache_dir = _get_default_cache_dir()

        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Shared cache directory set to: {self.cache_dir}")

    def _get_tool_cache_dir(self, subdirectory: str) -> Path:
        """Get the cache directory for a specific tool subdirectory.

        Args:
            subdirectory: The tool subdirectory name.

        Returns:
            Path: The full path to the tool's cache directory.
        """
        tool_cache_dir = self.cache_dir / subdirectory
        tool_cache_dir.mkdir(parents=True, exist_ok=True)
        return tool_cache_dir

    def _generate_cache_key(
        self,
        content: str | bytes,
        config: dict[str, str | int | float | bool] | None = None,
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
        # Convert content to string for hashing
        if isinstance(content, bytes):
            content_str = content.decode("utf-8", errors="ignore")
        else:
            content_str = str(content)

        # Create a dictionary with all relevant parameters
        cache_data = {
            "content": content_str,
            "config": config or {},
            "kwargs": kwargs,
        }

        # Convert to JSON string and hash it
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_string.encode()).hexdigest()

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
        config: dict[str, str | int | float | bool] | None = None,
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
        config: dict[str, str | int | float | bool] | None = None,
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

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, default=str)
            logger.debug(f"Cached result to {cache_file}")
        except IOError as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

    def clear(self, subdirectory: str | None = None) -> None:
        """Clear cached results.

        Args:
            subdirectory: If provided, clear only this subdirectory. If None, clear all.
        """
        if subdirectory is None:
            # Clear all subdirectories
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("**/*.json"):
                    cache_file.unlink()
                logger.info(f"Cleared all cache directories: {self.cache_dir}")
        else:
            # Clear specific subdirectory
            tool_cache_dir = self._get_tool_cache_dir(subdirectory)
            if tool_cache_dir.exists():
                for cache_file in tool_cache_dir.glob("*.json"):
                    cache_file.unlink()
                logger.info(f"Cleared cache directory: {tool_cache_dir}")

    def get_cache_stats(
        self, subdirectory: str | None = None
    ) -> dict[str, int | dict[str, int]]:
        """Get cache statistics.

        Args:
            subdirectory: If provided, get stats for this subdirectory only. If None, get stats for all.

        Returns:
            Dict[str, Any]: Dictionary with cache statistics.
        """
        if subdirectory is None:
            # Get stats for all subdirectories
            if not self.cache_dir.exists():
                return {"total_files": 0, "total_size_bytes": 0, "subdirectories": {}}

            cache_files = list(self.cache_dir.glob("**/*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)

            # Group by subdirectory
            subdir_stats = {}
            for cache_file in cache_files:
                subdir = cache_file.parent.name
                if subdir not in subdir_stats:
                    subdir_stats[subdir] = {"files": 0, "size_bytes": 0}
                subdir_stats[subdir]["files"] += 1
                subdir_stats[subdir]["size_bytes"] += cache_file.stat().st_size

            return {
                "total_files": len(cache_files),
                "total_size_bytes": total_size,
                "subdirectories": subdir_stats,
            }
        else:
            # Get stats for specific subdirectory
            tool_cache_dir = self._get_tool_cache_dir(subdirectory)
            if not tool_cache_dir.exists():
                return {"total_files": 0, "total_size_bytes": 0}

            cache_files = list(tool_cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "total_files": len(cache_files),
                "total_size_bytes": total_size,
            }


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
        config: dict[str, str | int | float | bool] | None = None,
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
        config: dict[str, str | int | float | bool] | None = None,
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

    def clear(self) -> None:
        """Clear cached results for this tool."""
        self.shared_cacher.clear(subdirectory=self.subdirectory)

    def get_cache_stats(self) -> dict[str, int | dict[str, int]]:
        """Get cache statistics for this tool.

        Returns:
            Dict[str, int]: Dictionary with cache statistics.
        """
        return self.shared_cacher.get_cache_stats(subdirectory=self.subdirectory)
