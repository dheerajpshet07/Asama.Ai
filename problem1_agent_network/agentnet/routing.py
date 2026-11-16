"""Relevance routing: who should hear about a post.

Pipeline per post (all of it is in Router.route):

  1. Candidates   union of  (a) agents named in the post, (b) followers of the
                  author or of the post's topics, (c) agents whose declared topics
                  overlap the post, (d) top-K nearest interest vectors.
                  Cheap and recall-oriented.
  2. Hard filters suspended agents, the author, mutes, excluded topics.
  3. Scoring      a weighted blend of semantic similarity, topic overlap,
                  learned affinity and audience hints, scaled by author trust.
  4. Decision     score >= threshold -> notify. Borderline scores go to an LLM
                  judge (batched, one call per post) when one is configured.
                  Mentions always notify. Followers always receive the post
                  (pushed above a lower bar, inbox-only below it).
  5. Budgets      per-post recipient cap and per-agent hourly push budget. Overflow
                  isn't dropped, it lands in the agent's inbox without a push.

Every scored candidate comes back with its features and a readable reason, so
the decision can be logged and explained.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import feedback
from .config import Settings
from .embeddings import Embedder
from .index import VectorIndex
from .judge import RelevanceJudge
from .models import Agent, Delivery, Post, Subscription
from .safety import reach_factor
from .taxonomy import INDUSTRY_TOPIC, TOPICS, detect_topics, split_needs, topic_mass
from .text import phrase, phrase_text
from .util import blob_to_vec, utcnow, vec_to_blob

# Score blend. Tuned on eval/labeled_posts.json (scripts/eval_routing.py).
#
# The topical signal is the stronger of two matches:
#   interest:   what the post is about  vs  what the agent wants to hear about
#   complement: what the post asks for  vs  what the agent can offer
# Either one alone can carry a post over the threshold; both together add a bonus.
W_PRIMARY = 0.60
BOTH_BONUS = 0.25      # share of the weaker match added on top of the stronger one
W_SEMANTIC = 0.30      # embedding similarity, catches phrasing the taxonomy misses
SEMANTIC_FLOOR = 0.05  # cosine below this is noise for the hashing embedder
SEMANTIC_FULL = 0.40   # ...and at this it saturates
W_AFFINITY = 0.20      # learned from feedback, in [-1, 1]
W_AUDIENCE = 0.10      # author explicitly targeted the agent's industry
INFERRED_TOPIC_WEIGHT = 0.7  # topics read from free text count less than declared ones
NEED_MIN_WEIGHT = 0.6  # generic topics (offers, partnerships, festive) are never "needs"


@dataclass
class Decision:
    agent: Agent
    score: float
    action: str        # notify | inbox | drop | judge_no | filtered
    decided_by: str    # score | judge | mention | follow | filter | cap | budget
    reason: str
    features: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- enrichment

def profile_topics(agent_text: str, declared: list[str]) -> dict[str, float]:
    merged = {t: 1.0 for t in declared}
    for t, conf in detect_topics(agent_text).items():
        merged[t] = max(merged.get(t, 0.0), round(INFERRED_TOPIC_WEIGHT * conf, 3))
    return merged


def post_topics(post: Post, author: Agent) -> tuple[dict[str, float], dict[str, float]]:
    """(what the post is about, what it asks for)."""
    about_text, need_text = split_needs(post.body)
    about = detect_topics(f"{post.title}\n{about_text}")
    for t in post.declared_topics or []:
        about[t] = 1.0
    home = INDUSTRY_TOPIC.get(author.industry)
    if home:
        # A post from a food delivery company is at least a little about food delivery.
        about[home] = max(about.get(home, 0.0), 0.5)
    # Only concrete domains count as needs, and the author's own domain is
    # context for the ask rather than part of it.
    needs = {
        t: c for t, c in detect_topics(need_text).items()
        if t != home and TOPICS[t].weight >= NEED_MIN_WEIGHT and c >= 0.5
    }
    return about, needs


def find_mentions(text: str, agents: list[Agent], author_id: str) -> list[str]:
    """Agents the post names explicitly, by @handle or company name."""
    padded = phrase_text(text)
    handles = set(re.findall(r"@([a-z0-9][a-z0-9_-]{2,39})", text.lower()))
    found = []
    for a in agents:
        if a.id == author_id:
            continue
        name = phrase(a.company_name)
        if a.handle in handles or (name and f" {name} " in padded):
            found.append(a.id)
    return found


def enrich_post(post: Post, author: Agent, embedder: Embedder, agents: list[Agent]) -> None:
    text = f"{post.title}\n{post.body}"
    post.topics, post.need_topics = post_topics(post, author)
    post.mentions = find_mentions(text, agents, author.id)
    post.embedding = vec_to_blob(embedder.embed(text))


# --------------------------------------------------------------------------- scoring

def topic_overlap(post_t: dict[str, float], agent_t: dict[str, float]) -> tuple[float, list[str]]:
    """Share of the post's (importance-weighted) topic mass that the agent covers."""
    mass = topic_mass(post_t)
    if mass <= 0:
        return 0.0, []
    shared = []
    acc = 0.0
    for t, conf in post_t.items():
        if t in agent_t and t in TOPICS:
            strength = min(conf, agent_t[t])
            acc += strength * TOPICS[t].weight
            if strength >= 0.5:  # passing mentions count, but aren't worth explaining
                shared.append((strength * TOPICS[t].weight, t))
    shared.sort(reverse=True)
    return acc / mass, [t for _, t in shared]


def interest_view(about: dict[str, float], needs: dict[str, float], capabilities: dict[str, float]) -> dict:
    """The part of a post an agent can be *interested* in.

    Topics the post is asking for are left out unless the agent offers them.
    Otherwise a company that wants the same thing as the author (another
    travel site also looking for bank partners) would look like a match.
    """
    return {t: c for t, c in about.items() if t not in needs or t in capabilities}


def combine(f: dict, trust: float) -> float:
    sem = (f["semantic"] - SEMANTIC_FLOOR) / (SEMANTIC_FULL - SEMANTIC_FLOOR)
    sem = min(1.0, max(0.0, sem))
    strong, weak = max(f["topic"], f["complement"]), min(f["topic"], f["complement"])
    primary = min(1.0, strong + BOTH_BONUS * weak)
    score = (W_PRIMARY * primary + W_SEMANTIC * sem + W_AFFINITY * f["affinity"]
             + (W_AUDIENCE if f["audience"] else 0.0))
    return round(max(0.0, min(1.0, score * reach_factor(trust))), 4)


def explain(features: dict) -> str:
    parts = []
    if features.get("mention"):
        parts.append("named in the post")
    if features.get("follow"):
        parts.append("follows " + features["follow"])
    if features.get("matched_topics"):
        parts.append("interested in " + ", ".join(features["matched_topics"][:3]))
    if features.get("offers_topics"):
        parts.append("can offer " + ", ".join(features["offers_topics"][:3]))
    parts.append(f"similarity {features['semantic']:.2f}")
    aff = features.get("affinity", 0.0)
    if aff >= 0.15:
        parts.append("usually engages with posts like this")
    elif aff <= -0.15:
        parts.append("usually ignores posts like this")
    if features.get("audience"):
        parts.append("in the post's target audience")
    return "; ".join(parts)


class Router:
    def __init__(self, settings: Settings, index: VectorIndex, judge: RelevanceJudge | None = None):
        self.settings = settings
        self.index = index
        self.judge = judge

    # -- helpers -------------------------------------------------------------

    def _subscriptions(self, db: Session, post: Post) -> tuple[dict, set, set]:
        """Returns (follow reason by agent id, agents muting the author, agents muting a post topic)."""
        topic_slugs = list(post.topics or {})
        rows = db.scalars(
            select(Subscription).where(Subscription.target.in_([post.author_id, *topic_slugs]))
        ).all()
        follows: dict[str, str] = {}
        muted_author, muted_topic = set(), set()
        for s in rows:
            if s.kind == "follow_agent" and s.target == post.author_id:
                follows[s.agent_id] = "the author"
            elif s.kind == "follow_topic" and s.target in topic_slugs:
                # only counts if the topic is a real part of the post
                if post.topics.get(s.target, 0) >= 0.5:
                    follows.setdefault(s.agent_id, f"topic '{s.target}'")
            elif s.kind == "mute_agent" and s.target == post.author_id:
                muted_author.add(s.agent_id)
            elif s.kind == "mute_topic" and s.target in topic_slugs:
                muted_topic.add(s.agent_id)
        return follows, muted_author, muted_topic

    def _recent_push_counts(self, db: Session, agent_ids: list[str]) -> dict[str, int]:
        if not agent_ids:
            return {}
        since = utcnow() - timedelta(hours=1)
        rows = db.execute(
            select(Delivery.agent_id, func.count())
            .where(Delivery.agent_id.in_(agent_ids), Delivery.channel == "push", Delivery.created_at >= since)
            .group_by(Delivery.agent_id)
        ).all()
        return {aid: n for aid, n in rows}

    @staticmethod
    def _excluded(post_t: dict[str, float], excluded: list[str]) -> bool:
        if not excluded:
            return False
        mass = topic_mass(post_t)
        ex = topic_mass({t: c for t, c in post_t.items() if t in excluded})
        return mass > 0 and ex / mass >= 0.5

    # -- main entry point ----------------------------------------------------

    def route(self, db: Session, post: Post) -> list[Decision]:
        s = self.settings
        author = post.author
        post_vec = blob_to_vec(post.embedding)
        post_t = post.topics or {}
        need_t = post.need_topics or {}

        agents = {
            a.id: a
            for a in db.scalars(select(Agent).where(Agent.status == "active", Agent.id != author.id))
        }
        if not agents or post_vec is None:
            return []

        follows, muted_author, muted_topic = self._subscriptions(db, post)
        mentions = set(post.mentions or []) & agents.keys()

        # 1. candidate generation. The topic scan is a linear pass here; with a
        # large agent base it becomes an inverted index topic -> agent ids.
        candidates = set(mentions) | (set(follows) & agents.keys())
        about_domains = {t for t, c in post_t.items() if TOPICS[t].weight >= 0.6 and c >= 0.5}
        need_domains = {t for t, c in need_t.items() if TOPICS[t].weight >= 0.6 and c >= 0.5}
        for a in agents.values():
            if about_domains & set(a.profile_topics or {}) or need_domains & set(a.capability_topics or {}):
                candidates.add(a.id)
        candidates |= {aid for aid, _ in self.index.search(post_vec, s.candidate_pool) if aid in agents}

        sims = self.index.similarity(post_vec, candidates)
        stats = feedback.load_stats(db, list(candidates))
        keys = feedback.signal_keys(post)

        decisions: list[Decision] = []
        borderline: list[Decision] = []
        lo, hi = s.judge_band

        for aid in candidates:
            agent = agents[aid]
            # 2. hard filters
            if aid in muted_author:
                decisions.append(Decision(agent, 0.0, "filtered", "filter", "muted the author"))
                continue
            if aid in muted_topic or self._excluded(post_t, agent.exclude_topics or []):
                decisions.append(Decision(agent, 0.0, "filtered", "filter", "excluded this topic"))
                continue

            # 3. features and score
            caps = agent.capability_topics or {}
            topic, matched = topic_overlap(interest_view(post_t, need_t, caps), agent.profile_topics or {})
            complement, offers = topic_overlap(need_t, caps)
            features = {
                "semantic": round(sims.get(aid, 0.0), 4),
                "topic": round(topic, 4),
                "matched_topics": matched,
                "complement": round(complement, 4),
                "offers_topics": offers,
                "affinity": feedback.affinity(stats.get(aid, {}), keys),
                "audience": agent.industry in (post.audience_industries or []),
                "follow": follows.get(aid),
                "mention": aid in mentions,
                "reach": reach_factor(author.trust_score),
            }
            score = combine(features, author.trust_score)
            threshold = agent.min_score if agent.min_score is not None else s.notify_threshold
            reason = explain(features)

            # 4. decision
            if features["mention"]:
                d = Decision(agent, max(score, threshold), "notify", "mention", reason, features)
            elif features["follow"]:
                # An explicit follow is never dropped. Weak matches go to the inbox without a push.
                action = "notify" if score >= s.follow_threshold else "inbox"
                d = Decision(agent, score, action, "follow", reason, features)
            elif self.judge is not None and lo <= score < hi:
                d = Decision(agent, score, "pending", "judge", reason, features)
                borderline.append(d)
            elif score >= threshold:
                d = Decision(agent, score, "notify", "score", reason, features)
            else:
                d = Decision(agent, score, "drop", "score", reason, features)
            d.features["threshold"] = threshold
            decisions.append(d)

        if borderline:
            verdicts = self.judge.judge(post, [d.agent for d in borderline])
            for d in borderline:
                v = verdicts.get(d.agent.id)
                if v is None:  # judge failed or skipped this one: fall back to the threshold
                    d.action = "notify" if d.score >= d.features["threshold"] else "drop"
                    d.decided_by = "score"
                else:
                    d.action = "notify" if v.notify else "judge_no"
                    d.reason = f"{d.reason}; judge: {v.reason}" if v.reason else d.reason
                    d.features["judge_confidence"] = v.confidence

        # 5. budgets
        notify = sorted((d for d in decisions if d.action == "notify"), key=lambda d: d.score, reverse=True)
        pushed = self._recent_push_counts(db, [d.agent.id for d in notify])
        slots = s.max_recipients_per_post
        for d in notify:
            if d.decided_by == "mention":
                continue  # always delivered
            if slots <= 0:
                d.action, d.decided_by = "inbox", "cap"
                continue
            budget = d.agent.hourly_budget or s.default_hourly_budget
            if pushed.get(d.agent.id, 0) >= budget:
                d.action, d.decided_by = "inbox", "budget"
                continue
            slots -= 1

        decisions.sort(key=lambda d: d.score, reverse=True)
        return decisions
