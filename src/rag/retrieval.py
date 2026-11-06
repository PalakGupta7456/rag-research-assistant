"""Hybrid retrieval: BM25 (lexical) + dense embeddings (semantic), fused with
Reciprocal Rank Fusion, optionally re-ranked by a cross-encoder."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import RRF_K

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were with we our
these those their they which using use based via can into than also such not more new paper propose proposed show""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class BM25:
    """Okapi BM25 (k1=1.5, b=0.75). Same formula is implemented in the browser (docs/assets/app.js)."""
    k1: float = 1.5
    b: float = 0.75
    docs: list[Counter] = field(default_factory=list)
    idf: dict = field(default_factory=dict)
    lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    avgdl: float = 0.0

    def fit(self, texts: list[str]) -> BM25:
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avgdl, 1e-9))
        for i, d in enumerate(self.docs):
            total = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    total += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[i])
            s[i] = total
        return s


def rrf(rankings: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank)."""
    n = len(rankings[0])
    fused = np.zeros(n)
    for order in rankings:
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        fused += 1.0 / (k + ranks)
    return fused


class Retriever:
    def __init__(self, texts: list[str], embeddings: np.ndarray, encoder, reranker=None):
        self.texts = texts
        self.emb = embeddings
        self.encoder = encoder
        self.bm25 = BM25().fit(texts)
        self.reranker = reranker

    def search(self, query: str, k: int = 10, mode: str = "hybrid", rerank_top: int = 30) -> list[tuple[int, float]]:
        if mode == "bm25":
            s = self.bm25.scores(query)
        elif mode == "dense":
            s = self.emb @ self.encoder.encode([query])[0]
        else:
            lex = self.bm25.scores(query)
            dense = self.emb @ self.encoder.encode([query])[0]
            s = rrf([np.argsort(-lex), np.argsort(-dense)])
        order = np.argsort(-s)
        if mode == "hybrid+rerank" and self.reranker is not None:
            cand = order[:rerank_top]
            rs = self.reranker.predict([(query, self.texts[i]) for i in cand])
            order = cand[np.argsort(-rs)]
            return [(int(i), float(v)) for i, v in zip(order[:k], sorted(rs, reverse=True)[:k], strict=False)]
        return [(int(i), float(s[i])) for i in order[:k]]
