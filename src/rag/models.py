"""Embedding / re-ranking models behind tiny interfaces (real ones in production, stand-ins in tests)."""
from __future__ import annotations

import numpy as np

from .config import EMBED_MODEL, RERANK_MODEL


class STEncoder:
    def __init__(self, name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


class CrossEncoderReranker:
    def __init__(self, name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name, device="cpu")

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(self.model.predict(pairs, show_progress_bar=False))


class HashingEncoder:
    """Offline stand-in: character n-gram hashing vectors (no downloads)."""

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


def load_encoder(lightweight: bool = False):
    return HashingEncoder() if lightweight else STEncoder()
