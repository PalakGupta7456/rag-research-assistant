"""Daily build: refresh the corpus, embed it, evaluate retrieval, publish the browser index.

Published files (docs/data/):
  papers.json      metadata + text for each paper (BM25 is computed in the browser)
  embeddings.bin   int8 matrix [n_papers x 384], row-normalised then scaled by 127
  stats.json       corpus stats, category mix and the retrieval benchmark
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import ingest
from .config import CORPUS, EMBED_MODEL, SITE_DATA_DIR
from .evaluate import evaluate
from .models import load_encoder

log = logging.getLogger("rag")


def doc_text(df: pd.DataFrame) -> list[str]:
    return (df["title"] + ". " + df["abstract"]).tolist()


def quantize(emb: np.ndarray) -> np.ndarray:
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return np.clip(np.round(emb * 127), -127, 127).astype(np.int8)


def publish(papers: pd.DataFrame, emb: np.ndarray, bench: dict | None, out_dir=SITE_DATA_DIR, added: int = 0, model: str = EMBED_MODEL) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    q = quantize(emb)
    (out_dir / "embeddings.bin").write_bytes(q.tobytes())
    recs = [{"id": r.id, "t": r.title, "a": r.abstract, "au": r.authors, "c": r.category,
             "d": pd.Timestamp(r.published).strftime("%Y-%m-%d")} for r in papers.itertuples()]
    (out_dir / "papers.json").write_text(json.dumps(recs, separators=(",", ":"), ensure_ascii=False))
    # quantisation check: cosine agreement between float and int8 vectors
    sample = slice(0, min(200, len(emb)))
    approx = (q[sample].astype(np.float32) / 127) @ (q[sample].astype(np.float32) / 127).T
    exact = emb[sample] @ emb[sample].T
    stats = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_papers": int(len(papers)), "added_this_run": int(added), "dim": int(emb.shape[1]), "model": model,
        "date_range": [str(pd.Timestamp(papers["published"].min()).date()), str(pd.Timestamp(papers["published"].max()).date())],
        "categories": papers["category"].value_counts().head(8).to_dict(),
        "index_bytes": int(q.nbytes), "quantization_max_abs_error": round(float(np.abs(approx - exact).max()), 4),
        "benchmark": bench,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def run(lightweight: bool = False, new: pd.DataFrame | None = None, corpus_path=CORPUS, out_dir=SITE_DATA_DIR, eval_queries: int = 300) -> dict:
    if new is None:
        new = ingest.fetch_latest()
    old = pd.read_parquet(corpus_path) if corpus_path.exists() else None
    papers, added = ingest.merge(old, new)
    log.info("corpus %d papers (+%d)", len(papers), added)
    encoder = load_encoder(lightweight)
    emb = encoder.encode(doc_text(papers))
    reranker = None
    if not lightweight:
        from .models import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    bench = evaluate(papers, encoder, reranker, n_queries=eval_queries)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    papers.to_parquet(corpus_path, index=False)
    np.save(corpus_path.with_suffix(".npy"), emb)  # float32 embeddings for the API
    return publish(papers, emb, bench, out_dir, added, model=type(encoder).__name__ if lightweight else EMBED_MODEL)


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    new = None
    if a.synthetic:
        from .synthetic import generate
        new = generate()
    s = run(lightweight=a.lightweight, new=new)
    print(json.dumps({k: v for k, v in s.items() if k != "categories"}, indent=2))


if __name__ == "__main__":
    main()
