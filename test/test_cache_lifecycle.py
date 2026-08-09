"""Tests for Cacher key construction, durability, and automatic pruning."""

from __future__ import annotations

import asyncio
import json
import os
import threading

import pytest

from ontocast.config import PathConfig
from ontocast.tool.cache import Cacher, _get_default_cache_dir, _running_under_pytest


@pytest.fixture
def cacher(tmp_path) -> Cacher:
    """A Cacher with automatic pruning off, so tests drive it explicitly."""
    return Cacher(cache_dir=tmp_path / "cache", max_bytes=0, ttl_days=None)


def _write(
    cacher: Cacher, key: str, payload: str, used_at: float | None = None
) -> None:
    cacher.set(key, payload, subdirectory="llm")
    if used_at is not None:
        path = cacher._get_cache_file_path(cacher._generate_cache_key(key, None), "llm")
        os.utime(path, (used_at, used_at))


def test_binary_content_is_hashed_not_lossily_decoded(cacher) -> None:
    """Two distinct binary payloads must not collide.

    Both of these decode to the same string under ``errors="ignore"``, which is
    how keys used to be derived -- so a PDF's key rested only on whichever
    bytes happened to be valid UTF-8.
    """
    left = b"\xff\xfeHELLO\xff"
    right = b"\xfeHELLO\xff\xff"
    assert left.decode("utf-8", errors="ignore") == right.decode(
        "utf-8", errors="ignore"
    )

    cacher.set(left, "left result", subdirectory="converter")
    cacher.set(right, "right result", subdirectory="converter")

    assert cacher.get(left, subdirectory="converter") == "left result"
    assert cacher.get(right, subdirectory="converter") == "right result"


def test_truncated_entry_is_a_miss_and_is_overwritten(cacher) -> None:
    """A half-written file must not raise, and must be repairable."""
    cacher.set("prompt", {"content": "ok"}, subdirectory="llm")
    path = cacher._get_cache_file_path(
        cacher._generate_cache_key("prompt", None), "llm"
    )
    path.write_text('{"result": {"conte', encoding="utf-8")

    assert cacher.get("prompt", subdirectory="llm") is None

    cacher.set("prompt", {"content": "ok"}, subdirectory="llm")
    assert cacher.get("prompt", subdirectory="llm") == {"content": "ok"}


def test_set_leaves_no_temp_files_behind(cacher) -> None:
    cacher.set("prompt", {"content": "ok"}, subdirectory="llm")
    assert list((cacher.cache_dir / "llm").glob("*.tmp")) == []


def test_concurrent_writes_of_the_same_key_stay_readable(cacher) -> None:
    """Same-key writers must not share a temp path and interleave into it."""
    payload = {"content": "x" * 20_000}
    errors: list[BaseException] = []

    def write() -> None:
        try:
            for _ in range(20):
                cacher.set("hot key", payload, subdirectory="llm")
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert cacher.get("hot key", subdirectory="llm") == payload
    assert list((cacher.cache_dir / "llm").glob("*.tmp")) == []


def test_prune_evicts_least_recently_used_until_under_ceiling(cacher) -> None:
    payload = "x" * 2000
    _write(cacher, "oldest", payload, used_at=1_000_000)
    _write(cacher, "middle", payload, used_at=2_000_000)
    _write(cacher, "newest", payload, used_at=3_000_000)

    total = cacher.cache_stats().total_size_bytes
    entry_size = total // 3

    report = cacher.prune(max_bytes=int(entry_size * 1.5))

    assert report.files_removed == 2
    assert report.bytes_remaining <= entry_size * 1.5
    # The most recently used survives, and is still readable.
    assert cacher.get("newest", subdirectory="llm") == payload
    assert cacher.get("oldest", subdirectory="llm") is None


def test_prune_ttl_applies_before_the_size_pass(cacher) -> None:
    _write(cacher, "stale", "x" * 100, used_at=1_000_000)
    _write(cacher, "fresh", "x" * 100)

    report = cacher.prune(max_bytes=0, ttl_days=1)

    assert report.files_removed == 1
    assert cacher.get("fresh", subdirectory="llm") == "x" * 100
    assert cacher.get("stale", subdirectory="llm") is None


def test_prune_is_a_noop_when_disabled(cacher) -> None:
    _write(cacher, "keep", "x" * 5000, used_at=1_000_000)

    report = cacher.prune(max_bytes=0, ttl_days=None)

    assert report.files_removed == 0
    assert cacher.get("keep", subdirectory="llm") == "x" * 5000


def test_writes_trigger_a_prune_once_the_threshold_is_crossed(tmp_path) -> None:
    """Steady-state trimming rides on the write counter, not just startup."""
    cacher = Cacher(
        cache_dir=tmp_path / "cache", max_bytes=1, ttl_days=None, prune_every=3
    )

    for index in range(2):
        cacher.set(f"key-{index}", "x" * 100, subdirectory="llm")
    assert cacher.cache_stats().total_files == 2, "must not prune before the threshold"

    cacher.set("key-2", "x" * 100, subdirectory="llm")
    # A 1-byte ceiling evicts everything the pass can see; the entry written
    # last survives only if it postdates the walk.
    assert cacher.cache_stats().total_files <= 1


def test_async_get_and_set_round_trip(cacher) -> None:
    async def run() -> None:
        await cacher.aset("prompt", {"content": "hi"}, subdirectory="llm")
        assert await cacher.aget("prompt", subdirectory="llm") == {"content": "hi"}

    asyncio.run(run())


def test_prune_orphaned_subdirs_keeps_live_ones(cacher) -> None:
    cacher.set("live", "kept", subdirectory="llm")
    orphan_dir = cacher.cache_dir / "converter_v2"
    orphan_dir.mkdir()
    (orphan_dir / "stale.json").write_text(json.dumps({"result": "x"}))

    report = cacher.prune_orphaned_subdirs({"llm"})

    assert report.files_removed == 1
    assert not orphan_dir.exists()
    assert cacher.get("live", subdirectory="llm") == "kept"


def test_pytest_is_detected_so_tests_do_not_share_the_real_cache() -> None:
    """The probe used to read ``$_``, which is ``uv`` under ``uv run pytest``."""
    assert _running_under_pytest()
    assert _get_default_cache_dir().name == ".test_cache"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1GB", 1024**3),
        ("100MB", 100 * 1024**2),
        ("512 mb", 512 * 1024**2),
        ("2048", 2048),
        (1024, 1024),
    ],
)
def test_cache_max_bytes_accepts_human_sizes(raw, expected) -> None:
    assert PathConfig(cache_max_bytes=raw).cache_max_bytes == expected
