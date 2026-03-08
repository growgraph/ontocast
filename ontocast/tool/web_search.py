"""Web-search providers used for optional ontology grounding."""

import asyncio
from typing import Any

from ontocast.tool.atomic import SearchHit


class DuckDuckGoSearchProvider:
    """DuckDuckGo-backed search provider."""

    def __init__(
        self,
        timeout_seconds: int | float = 8,
        region: str = "wt-wt",
        safesearch: str = "moderate",
    ):
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.region = region
        self.safesearch = safesearch

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Search DuckDuckGo and return normalized hits."""
        return await asyncio.to_thread(
            self._search_sync, query=query, max_results=max_results
        )

    def _search_sync(self, query: str, max_results: int) -> list[SearchHit]:
        # Import lazily so environments without this optional dependency can still
        # run with web search disabled.
        from duckduckgo_search import DDGS

        hits: list[SearchHit] = []
        with DDGS(timeout=self.timeout_seconds) as ddgs:
            results = ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results,
            )
            for item in results:
                normalized = self._normalize_item(item)
                if normalized is not None:
                    hits.append(normalized)
        return hits

    def _normalize_item(self, item: Any) -> SearchHit | None:
        if not isinstance(item, dict):
            return None

        title = str(item.get("title") or item.get("heading") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("body") or item.get("snippet") or "").strip()
        if not url or not snippet:
            return None

        if not title:
            title = url

        return SearchHit(title=title, url=url, snippet=snippet)
