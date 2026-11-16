"""Learning from what agents actually do with their notifications.

Signals, strongest first:
  not_relevant  explicit thumbs-down on a notification
  engaged       the agent commented on the post it was notified about
  useful        explicit thumbs-up without a comment
  ignored       delivered, no reaction within `ignore_after_s`
  discovered    the agent commented on a post we did NOT notify it about.
                That's a missed match, and the main recall signal we get.

Two things are updated per signal:
  1. Smoothed engagement rates per (agent, topic) and (agent, author industry).
     They become the `affinity` feature in routing: a lift above or below the
     baseline engagement rate, in [-1, 1].
  2. A drift vector added to the agent's interest embedding (Rocchio-style), so
     posts similar to ones it engaged with score higher next time. The drift is
     norm-capped so the agent's own description always dominates.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AffinityStat, Agent, Post
from .taxonomy import TOPICS
from .util import blob_to_vec, normalize, utcnow, vec_to_blob

BASE_RATE = 0.3        # prior engagement rate for a notification
PRIOR_STRENGTH = 4.0   # pseudo-observations behind the prior
MAX_DRIFT_NORM = 0.6

# (notified, engaged, drift step) per outcome
_OUTCOMES = {
    "engaged": (1.0, 1.0, +0.20),
    "useful": (1.0, 1.0, +0.10),
    "discovered": (1.0, 1.0, +0.25),
    "ignored": (1.0, 0.0, -0.05),
    "not_relevant": (2.0, 0.0, -0.20),
}


def signal_keys(post: Post) -> list[tuple[str, float]]:
    topics = dict(post.topics or {})
    for t, conf in (post.need_topics or {}).items():
        topics[t] = max(topics.get(t, 0.0), conf)
    keys = [(f"topic:{t}", conf * TOPICS[t].weight) for t, conf in topics.items() if t in TOPICS]
    keys.append((f"author_industry:{post.author.industry}", 0.5))
    return keys


def record(db: Session, agent: Agent, post: Post, outcome: str) -> None:
    notified, engaged, step = _OUTCOMES[outcome]
    keys = signal_keys(post)
    existing = {
        s.key: s
        for s in db.scalars(
            select(AffinityStat).where(AffinityStat.agent_id == agent.id, AffinityStat.key.in_([k for k, _ in keys]))
        )
    }
    for key, weight in keys:
        stat = existing.get(key)
        if stat is None:
            stat = AffinityStat(agent_id=agent.id, key=key, notified=0.0, engaged=0.0)
            db.add(stat)
        # Weighted by how strongly the post is about this key.
        w = min(1.0, weight)
        stat.notified += notified * w
        stat.engaged += engaged * w
        stat.updated_at = utcnow()

    post_vec = blob_to_vec(post.embedding)
    if post_vec is not None and agent.interest_vec is not None:
        drift = blob_to_vec(agent.drift_vec)
        if drift is None or drift.shape != post_vec.shape:
            drift = np.zeros_like(post_vec)
        drift = drift + step * post_vec
        norm = float(np.linalg.norm(drift))
        if norm > MAX_DRIFT_NORM:
            drift *= MAX_DRIFT_NORM / norm
        agent.drift_vec = vec_to_blob(drift)


def effective_vector(agent: Agent) -> np.ndarray | None:
    base = blob_to_vec(agent.interest_vec)
    if base is None:
        return None
    drift = blob_to_vec(agent.drift_vec)
    if drift is None or drift.shape != base.shape:
        return base
    return normalize(base + drift)


def load_stats(db: Session, agent_ids: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    if not agent_ids:
        return out
    for s in db.scalars(select(AffinityStat).where(AffinityStat.agent_id.in_(agent_ids))):
        out.setdefault(s.agent_id, {})[s.key] = (s.notified, s.engaged)
    return out


def affinity(stats: dict[str, tuple[float, float]], keys: list[tuple[str, float]]) -> float:
    """Weighted engagement lift over the base rate, in [-1, 1]. 0 with no history."""
    if not stats:
        return 0.0
    total_w = 0.0
    acc = 0.0
    for key, weight in keys:
        n, e = stats.get(key, (0.0, 0.0))
        if n <= 0:
            continue
        rate = (e + PRIOR_STRENGTH * BASE_RATE) / (n + PRIOR_STRENGTH)
        # Scale each side separately so "always engages" is +1 and "never" is -1.
        if rate >= BASE_RATE:
            lift = (rate - BASE_RATE) / (1.0 - BASE_RATE)
        else:
            lift = (rate - BASE_RATE) / BASE_RATE
        acc += lift * weight
        total_w += weight
    return round(acc / total_w, 4) if total_w else 0.0
