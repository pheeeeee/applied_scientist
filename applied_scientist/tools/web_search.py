from __future__ import annotations

import json
import os

from applied_scientist.tools.base import Tool


class SearchWeb(Tool):
    name = "search_web"
    description = "Search the web. Returns a list of results with title, URL, and snippet."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum results to return",
                            "default": 10},
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 10) -> str:
        try:
            import requests  # noqa: F811
        except ImportError:
            return "ERROR: requests not installed. Run: pip install applied-scientist[search]"
        self._requests = requests
        backend = os.environ.get("WEB_SEARCH_BACKEND", "serper")
        api_key = os.environ.get("WEB_SEARCH_API_KEY", "")
        if not api_key:
            return "ERROR: WEB_SEARCH_API_KEY not set."

        try:
            if backend == "serper":
                results = self._serper(query, max_results, api_key)
            elif backend == "serpapi":
                results = self._serpapi(query, max_results, api_key)
            else:
                return f"ERROR: Unknown search backend: {backend}. Use 'serper' or 'serpapi'."
        except Exception as e:
            return f"ERROR: Search failed: {e}"

        return json.dumps(results[:max_results], indent=2)

    def _serper(self, query: str, max_results: int, api_key: str) -> list[dict]:
        resp = self._requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title", ""), "url": r.get("link", ""),
             "snippet": r.get("snippet", "")}
            for r in resp.json().get("organic", [])
        ]

    def _serpapi(self, query: str, max_results: int, api_key: str) -> list[dict]:
        resp = self._requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": api_key, "num": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title", ""), "url": r.get("link", ""),
             "snippet": r.get("snippet", "")}
            for r in resp.json().get("organic_results", [])
        ]
