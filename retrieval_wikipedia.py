from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WikiPassage:
    title: str
    url: str
    snippet: str


def _get_json(url: str, params: dict[str, Any], timeout_s: float = 20) -> dict[str, Any]:
    qs = urllib.parse.urlencode(params)
    full = f"{url}?{qs}"
    req = urllib.request.Request(
        full,
        headers={
            "User-Agent": "SocialRAG/0.1 (edu project; MediaWiki API)",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def wikipedia_search_passages(query: str, *, k: int = 6, lang: str = "en") -> list[WikiPassage]:
    """
    Lightweight passage retrieval using Wikipedia's MediaWiki API.
    Returns title + short snippet + page URL (sufficient for a baseline RAG).
    """
    api = f"https://{lang}.wikipedia.org/w/api.php"
    data = _get_json(
        api,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": int(k),
            "format": "json",
            "utf8": 1,
        },
    )

    out: list[WikiPassage] = []
    for item in data.get("query", {}).get("search", []):
        title = str(item.get("title", ""))
        snippet = str(item.get("snippet", "")).replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        if not title:
            continue
        url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        out.append(WikiPassage(title=title, url=url, snippet=snippet))
    return out

