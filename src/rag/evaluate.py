"""Retrieval evaluation without hand labels: title -> abstract retrieval.

Each paper's title is used as the query, and its own *abstract only* (title
removed) as the single relevant document. Titles are short, rephrased
summaries of the abstract, so this measures whether a retriever finds the
right paper from a paraphrase. It is a standard proxy for known-item search.
"""
from __future__ import annotations

import time

import numpy as np

from .retrieval import Retriever

MODES = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def evaluate(papers, encoder, reranker=None, n_queries: int = 300, seed: int = 0, ks=(1, 5, 10)) -> dict:
    abstracts = papers["abstract"].tolist()
    emb = encoder.encode(abstracts)
    retr = Retriever(abstracts, emb, encoder, reranker)
    rng = np.random.default_rng(seed)
    qidx = rng.choice(len(papers), min(n_queries, len(papers)), replace=False)
    out = {}
    for mode in MODES:
        if mode == "hybrid+rerank" and reranker is None:
            continue
        ranks, t0 = [], time.time()
        for i in qidx:
            hits = [d for d, _ in retr.search(papers["title"].iloc[i], k=max(ks), mode=mode)]
            ranks.append(hits.index(i) + 1 if i in hits else None)
        dt = (time.time() - t0) / len(qidx) * 1000
        out[mode] = {**{f"hit@{k}": round(float(np.mean([r is not None and r <= k for r in ranks])), 4) for k in ks},
                     "mrr@10": round(float(np.mean([1 / r if r else 0 for r in ranks])), 4),
                     "ms_per_query": round(dt, 1)}
    return {"protocol": "title→abstract known-item retrieval", "n_queries": int(len(qidx)), "corpus_size": int(len(papers)), "results": out}
