from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone

import numpy as np


def utcnow() -> datetime:
    # Naive UTC everywhere. SQLite drops tzinfo on the way back out, and mixing
    # aware/naive datetimes in comparisons is a classic source of bugs.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    return str(uuid.uuid4())


def new_token(prefix: str = "") -> str:
    return prefix + secrets.token_urlsafe(32)


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def vec_to_blob(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec
