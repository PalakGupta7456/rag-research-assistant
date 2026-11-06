"""Answer composition: grounded, cited, and honest when the corpus has no answer.

* Extractive (always available, no LLM): the sentences most similar to the
  question from the top retrieved papers, each with a [n] citation.
* Generative (when ANTHROPIC_API_KEY is set): Claude writes an answer using
  only the numbered sources, citing them, and must say so if they don't cover it.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .config import LLM_MODEL

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

SYSTEM = (
    "You answer questions about recent machine-learning research using ONLY the numbered sources provided. "
    "Cite every claim with its source number in square brackets, e.g. [2]. "
    "If the sources do not contain the answer, say so plainly and suggest what to search for instead. "
    "Be concise: at most 150 words."
)


def build_context(papers: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {p['title']} ({p['id']})\n{p['abstract']}" for i, p in enumerate(papers))


def extractive(question: str, papers: list[dict], encoder, max_sentences: int = 4, min_sim: float = 0.25) -> dict:
    cands = [(i, s) for i, p in enumerate(papers) for s in SENT_SPLIT.split(p["abstract"]) if len(s) > 40]
    if not cands:
        return {"mode": "extractive", "answer": "", "sentences": []}
    sims = encoder.encode([s for _, s in cands]) @ encoder.encode([question])[0]
    picked, seen_papers = [], set()
    for j in np.argsort(-sims):
        i, s = cands[j]
        if sims[j] < min_sim or len(picked) >= max_sentences:
            break
        if i in seen_papers and len(seen_papers) < min(3, len(papers)):
            continue  # prefer breadth across papers first
        picked.append({"text": s, "source": i + 1, "score": round(float(sims[j]), 3)})
        seen_papers.add(i)
    if not picked:
        return {"mode": "extractive", "answer": "None of the retrieved papers directly address this question.", "sentences": []}
    return {"mode": "extractive", "answer": " ".join(f"{p['text']} [{p['source']}]" for p in picked), "sentences": picked}


def generative(question: str, papers: list[dict], model: str = LLM_MODEL) -> dict | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model, max_tokens=400, system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n\n{build_context(papers)}\n\nQuestion: {question}"}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(papers)})
    return {"mode": "generative", "model": model, "answer": text, "cited": cited,
            "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}}


def citation_precision(answer: str, n_sources: int) -> float:
    """Share of [n] markers that point at a real source: a cheap hallucinated-citation check."""
    refs = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    return 1.0 if not refs else sum(1 <= r <= n_sources for r in refs) / len(refs)
