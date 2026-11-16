"""Text embedders.

The default HashingEmbedder needs no model download or API key, which keeps the
prototype runnable anywhere. It hashes words, bigrams and taxonomy topics into a
fixed-size vector. Topic features carry most of the cross-vocabulary signal.

For production I'd swap in a real sentence-embedding model (OllamaEmbedder below,
or a hosted embedding API). Nothing else changes: the router only sees vectors.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from functools import lru_cache
from typing import Protocol

import httpx
import numpy as np

from .taxonomy import TOPICS, detect_topics
from .text import tokens
from .util import normalize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


@lru_cache(maxsize=200_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    # Stable across processes (unlike hash()), so stored vectors stay valid.
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 2048, topic_boost: float = 3.0):
        self.dim = dim
        self.topic_boost = topic_boost

    def features(self, text: str) -> Counter:
        toks = tokens(text)
        feats: Counter = Counter()
        for tok in toks:
            feats["w:" + tok] += 1.0
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}_{b}"] += 0.5
        for slug, conf in detect_topics(text).items():
            feats["t:" + slug] += self.topic_boost * conf * TOPICS[slug].weight
        return feats

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, count in self.features(text).items():
            idx, sign = _bucket(feat, self.dim)
            # sublinear tf, so one word repeated ten times doesn't dominate
            weight = 1.0 + math.log(count) if count >= 1 else count
            vec[idx] += sign * weight
        return normalize(vec)


class OllamaEmbedder:
    """Dense embeddings from a local Ollama server (e.g. nomic-embed-text)."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)
        self.dim = len(self.embed("dimension probe"))

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": text})
        resp.raise_for_status()
        vec = np.asarray(resp.json()["embeddings"][0], dtype=np.float32)
        return normalize(vec)


def make_embedder(kind: str = "hashing", **kwargs) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder(**kwargs)
    if kind.startswith("ollama"):
        model = kind.split(":", 1)[1] if ":" in kind else "nomic-embed-text"
        return OllamaEmbedder(model=model, **kwargs)
    raise ValueError(f"unknown embedder {kind!r}")
