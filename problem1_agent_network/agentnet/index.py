"""In-memory vector index over agent interest vectors.

Brute force on purpose: for 10k agents x 2048 dims a single matrix-vector
product is ~20M multiply-adds, i.e. about a millisecond with numpy. Past ~100k
agents, or with several API replicas, this moves into Postgres (pgvector HNSW,
see docs/schema.sql) or a dedicated vector store, behind the same two methods.
"""
from __future__ import annotations

import threading

import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._dirty = False
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._vectors)

    def upsert(self, agent_id: str, vec: np.ndarray) -> None:
        if vec.shape != (self.dim,):
            raise ValueError(f"expected vector of dim {self.dim}, got {vec.shape}")
        with self._lock:
            self._vectors[agent_id] = vec.astype(np.float32)
            self._dirty = True

    def remove(self, agent_id: str) -> None:
        with self._lock:
            if self._vectors.pop(agent_id, None) is not None:
                self._dirty = True

    def _snapshot(self) -> tuple[list[str], np.ndarray]:
        with self._lock:
            if self._dirty:
                self._ids = list(self._vectors)
                self._matrix = (
                    np.stack([self._vectors[i] for i in self._ids])
                    if self._ids
                    else np.zeros((0, self.dim), dtype=np.float32)
                )
                self._dirty = False
            return self._ids, self._matrix

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        ids, matrix = self._snapshot()
        if not ids:
            return []
        sims = matrix @ query.astype(np.float32)
        k = min(k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(ids[i], float(sims[i])) for i in top]

    def similarity(self, query: np.ndarray, agent_ids) -> dict[str, float]:
        with self._lock:
            return {
                aid: float(self._vectors[aid] @ query)
                for aid in agent_ids
                if aid in self._vectors
            }
