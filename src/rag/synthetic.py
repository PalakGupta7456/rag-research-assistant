"""Small generated corpus with the arXiv schema (tests and offline runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import COLUMNS

TOPICS = {
    "cs.CL": ("retrieval-augmented generation", ["retrieval", "generation", "hallucination", "grounding", "citations", "documents"]),
    "cs.LG": ("diffusion models", ["diffusion", "denoising", "sampling", "score", "noise schedule", "image synthesis"]),
    "cs.AI": ("reinforcement learning from human feedback", ["reward model", "preference", "policy", "alignment", "PPO", "human feedback"]),
    "cs.CV": ("vision transformers", ["patches", "attention", "image classification", "ViT", "self-supervised", "segmentation"]),
    "stat.ML": ("conformal prediction", ["coverage", "uncertainty", "calibration", "prediction sets", "distribution-free", "quantile"]),
    "cs.IR": ("dense retrieval", ["embeddings", "bi-encoder", "re-ranking", "BM25", "contrastive", "passage ranking"]),
}
VERBS = ["improves", "reduces", "analyses", "scales", "stabilises", "accelerates"]
OBJECTS = ["sample efficiency", "inference latency", "robustness to distribution shift", "memory use", "faithfulness", "training stability"]


def generate(n: int = 240, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(TOPICS)
    rows = []
    end = pd.Timestamp.now(tz="UTC")
    for i in range(n):
        c = cats[i % len(cats)]
        name, kws = TOPICS[c]
        k = rng.choice(kws, 3, replace=False)
        v, o = rng.choice(VERBS), rng.choice(OBJECTS)
        title = f"{str(k[0]).title()}-aware {name} that {v} {o}"
        abstract = (f"We study {name} with a focus on {k[0]} and {k[1]}. Existing approaches struggle with {o}. "
                    f"We introduce a method based on {k[2]} that {v} {o} across {rng.integers(3, 12)} benchmarks. "
                    f"Experiments show gains of {rng.integers(2, 30)} percent over strong baselines while keeping compute modest. "
                    f"Code and models are released to support future work on {name}.")
        rows.append({"id": f"2611.{i:05d}", "title": title, "abstract": abstract, "authors": "A. Author, B. Author", "category": c,
                     "published": end - pd.Timedelta(hours=int(rng.integers(0, 24 * 20))), "url": f"https://arxiv.org/abs/2611.{i:05d}"})
    return pd.DataFrame(rows, columns=COLUMNS)
