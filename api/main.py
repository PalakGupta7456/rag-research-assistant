"""RAG API.

    GET  /search?q=...&mode=hybrid&k=5     ranked papers
    POST /ask {"question": "..."}          retrieved sources + grounded, cited answer
Generative answers need ANTHROPIC_API_KEY; without it the API answers extractively.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import __version__
from rag.answer import citation_precision, extractive, generative
from rag.build import doc_text
from rag.config import CORPUS
from rag.models import load_encoder
from rag.retrieval import Retriever

STATE: dict = {}
Mode = Literal["bm25", "dense", "hybrid", "hybrid+rerank"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    lightweight = os.getenv("LIGHTWEIGHT_MODELS", "0") == "1"
    corpus = STATE.get("corpus_path", CORPUS)
    if corpus.exists():
        papers = pd.read_parquet(corpus)
        enc = load_encoder(lightweight)
        npy = corpus.with_suffix(".npy")
        emb = np.load(npy) if npy.exists() and not lightweight else enc.encode(doc_text(papers))
        reranker = None
        if not lightweight and os.getenv("RERANK", "1") == "1":
            from rag.models import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        STATE.update(papers=papers, encoder=enc, retriever=Retriever(doc_text(papers), emb, enc, reranker))
    yield


app = FastAPI(title="Research RAG API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _require():
    if "retriever" not in STATE:
        raise HTTPException(503, "Corpus not loaded. Run `python -m rag.build` first.")


def _paper(i: int, score: float) -> dict:
    p = STATE["papers"].iloc[i]
    return {"id": p["id"], "title": p["title"], "abstract": p["abstract"], "authors": p["authors"],
            "category": p["category"], "url": p["url"], "score": round(score, 4)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "papers": int(len(STATE["papers"])) if "papers" in STATE else 0,
            "generative": bool(os.getenv("ANTHROPIC_API_KEY")), "version": __version__}


@app.get("/search")
def search(q: str, mode: Mode = "hybrid", k: int = 5) -> dict:
    _require()
    t0 = time.time()
    hits = STATE["retriever"].search(q, k=max(1, min(k, 20)), mode=mode)
    return {"query": q, "mode": mode, "ms": round((time.time() - t0) * 1000, 1), "results": [_paper(i, s) for i, s in hits]}


class AskIn(BaseModel):
    question: str = Field(min_length=5, max_length=500, examples=["How do recent papers reduce hallucination in RAG?"])
    k: int = Field(5, ge=1, le=10)
    mode: Mode = "hybrid"


@app.post("/ask")
def ask(body: AskIn) -> dict:
    _require()
    t0 = time.time()
    hits = STATE["retriever"].search(body.question, k=body.k, mode=body.mode)
    sources = [_paper(i, s) for i, s in hits]
    t_retrieve = time.time() - t0
    answer = generative(body.question, sources) or extractive(body.question, sources, STATE["encoder"])
    return {
        "question": body.question, "answer": answer, "sources": sources,
        "checks": {"citation_precision": citation_precision(answer["answer"], len(sources))},
        "timing_ms": {"retrieve": round(t_retrieve * 1000, 1), "total": round((time.time() - t0) * 1000, 1)},
    }
