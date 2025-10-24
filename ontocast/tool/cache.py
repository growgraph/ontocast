"""Generic caching functionality for OntoCast tools.

This module provides a generic caching mechanism that can be used by various
tools to cache their results based on input content and configuration parameters.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
    """Generic caching class for OntoCast tools.

    This class provides a unified interface for caching results from various
    tools based on input content and configuration parameters.

    Attributes:
        cache_dir: Base directory for caching.
        subdirectory: Subdirectory within cache_dir for this specific tool.
    """

    def __init__(
        self,
        subdirectory: str = "cache",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the cacher.

        Args:
            subdirectory: Subdirectory within the base cache directory for this specific tool.
            cache_dir: Base directory for caching. If None, uses platform-appropriate default.
        """
        if cache_dir is None:
            cache_dir = _get_default_cache_dir()

        self.cache_dir = Path(cache_dir).expanduser()
        self.subdirectory = subdirectory
        self.tool_cache_dir = self.cache_dir / subdirectory

        # Create cache directory if it doesn't exist
        self.tool_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory set to: {self.tool_cache_dir}")

    def _generate_cache_key(
        self,
        content: Union[str, bytes],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
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

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given cache key.

        Args:
            cache_key: The cache key.

        Returns:
            Path: The path to the cache file.
        """
        return self.tool_cache_dir / f"{cache_key}.json"

    def get(
        self,
        content: Union[str, bytes],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Get cached result for given content and configuration.

        Args:
            content: The input content.
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.

        Returns:
            Optional[Any]: The cached result or None if not found.
        """
        cache_key = self._generate_cache_key(content, config, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)

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
        content: Union[str, bytes],
        result: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Cache a result for given content and configuration.

        Args:
            content: The input content.
            result: The result to cache.
            config: Optional configuration dictionary.
            **kwargs: Additional parameters that affect the result.
        """
        cache_key = self._generate_cache_key(content, config, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)

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

    def clear(self) -> None:
        """Clear all cached results for this tool."""
        if self.tool_cache_dir.exists():
            for cache_file in self.tool_cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared cache directory: {self.tool_cache_dir}")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict[str, int]: Dictionary with cache statistics.
        """
        if not self.tool_cache_dir.exists():
            return {"total_files": 0, "total_size_bytes": 0}

        cache_files = list(self.tool_cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
        }
