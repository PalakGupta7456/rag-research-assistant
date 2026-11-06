"""Fetch the newest ML papers from the arXiv API and keep a rolling corpus."""
from __future__ import annotations

import logging
import re
import time

import feedparser
import pandas as pd
import requests

from .config import ARXIV_API, CATEGORIES, FETCH_PER_RUN, MAX_PAPERS

log = logging.getLogger(__name__)
COLUMNS = ["id", "title", "abstract", "authors", "category", "published", "url"]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_atom(content: bytes) -> pd.DataFrame:
    feed = feedparser.parse(content)
    rows = []
    for e in feed.entries:
        arxiv_id = e.get("id", "").rsplit("/abs/", 1)[-1]
        base_id = re.sub(r"v\d+$", "", arxiv_id)           # one row per paper, not per version
        rows.append({
            "id": base_id, "title": _clean(e.get("title")), "abstract": _clean(e.get("summary")),
            "authors": ", ".join(a.get("name", "") for a in e.get("authors", [])[:6]),
            "category": e.get("arxiv_primary_category", {}).get("term") or (e.get("tags") or [{}])[0].get("term", ""),
            "published": pd.Timestamp(e.get("published")).tz_convert("UTC") if e.get("published") else pd.NaT,
            "url": f"https://arxiv.org/abs/{base_id}",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_latest(n: int = FETCH_PER_RUN, page: int = 200) -> pd.DataFrame:
    query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    frames = []
    for start in range(0, n, page):
        for attempt in range(4):
            try:
                r = requests.get(ARXIV_API, timeout=60, params={
                    "search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
                    "start": start, "max_results": page})
                r.raise_for_status()
                frames.append(parse_atom(r.content))
                break
            except requests.RequestException as exc:  # pragma: no cover - network
                log.warning("arXiv request failed (%s); retrying", exc)
                time.sleep(5 * (attempt + 1))
        time.sleep(3.1)  # arXiv API etiquette: one request every 3 seconds
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def merge(corpus: pd.DataFrame | None, new: pd.DataFrame, cap: int = MAX_PAPERS) -> tuple[pd.DataFrame, int]:
    new = new[(new["abstract"].str.len() > 200) & new["title"].ne("")].drop_duplicates("id")
    if corpus is None or corpus.empty:
        merged, added = new, len(new)
    else:
        fresh = new[~new["id"].isin(corpus["id"])]
        merged, added = pd.concat([corpus, fresh], ignore_index=True), len(fresh)
    merged = merged.sort_values("published", ascending=False).head(cap).reset_index(drop=True)
    return merged, added
