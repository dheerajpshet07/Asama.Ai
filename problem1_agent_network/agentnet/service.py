"""Business logic behind the HTTP API. Kept free of FastAPI so it can be
driven directly from scripts and tests."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import feedback
from .models import Agent, Comment, Delivery, Post, Report, RouteLog, Subscription
from .routing import enrich_post, profile_topics
from .safety import (
    TRUST_UNVERIFIED,
    TRUST_VERIFIED,
    adjust_trust,
    assess_text,
    check_comment_rate,
    check_post_rate,
    check_webhook_url,
)
from .schemas import AgentRegister, AgentUpdate, CommentCreate, PostCreate, ReportIn
from .taxonomy import TOPICS, detect_topics
from .util import hash_key, new_token, utcnow, vec_to_blob

if TYPE_CHECKING:
    from .runtime import Platform

log = logging.getLogger(__name__)


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


class Conflict(Exception):
    pass


class Invalid(Exception):
    pass


# --------------------------------------------------------------------------- agents

def refresh_profile(p: "Platform", agent: Agent) -> None:
    """Recompute the routing view of an agent after its interests change."""
    labels = ", ".join(TOPICS[t].label for t in agent.topics or [])
    agent.profile_topics = profile_topics(agent.interests, agent.topics or [])
    agent.capability_topics = detect_topics(agent.capabilities or "")
    agent.interest_vec = vec_to_blob(p.embedder.embed(f"{agent.interests}\n{labels}"))
    # Old drift was learned against the old description; start fresh.
    agent.drift_vec = None


def authenticate(db: Session, api_key: str) -> Agent | None:
    if not api_key:
        return None
    return db.scalar(select(Agent).where(Agent.api_key_hash == hash_key(api_key)))


def register_agent(p: "Platform", db: Session, data: AgentRegister) -> tuple[Agent, str, str]:
    if db.scalar(select(Agent.id).where(Agent.handle == data.handle)):
        raise Conflict(f"handle '{data.handle}' is taken")
    if data.webhook_url:
        check_webhook_url(data.webhook_url, p.settings.allow_private_webhooks)

    api_key = new_token("an_")
    secret = new_token("whsec_")
    agent = Agent(
        handle=data.handle,
        company_name=data.company_name,
        industry=data.industry,
        description=data.description,
        interests=data.interests,
        capabilities=data.capabilities,
        topics=data.topics,
        exclude_topics=data.exclude_topics,
        domain=data.domain,
        webhook_url=data.webhook_url,
        webhook_secret=secret,
        api_key_hash=hash_key(api_key),
        verification_token=new_token()[:32],
        trust_score=TRUST_UNVERIFIED,
        hourly_budget=data.hourly_budget,
        min_score=data.min_score,
    )
    refresh_profile(p, agent)
    db.add(agent)
    db.commit()
    p.reindex_agent(agent)
    return agent, api_key, secret


def update_agent(p: "Platform", db: Session, agent: Agent, data: AgentUpdate) -> Agent:
    changes = data.model_dump(exclude_unset=True)
    if changes.get("webhook_url"):
        check_webhook_url(changes["webhook_url"], p.settings.allow_private_webhooks)
    for field, value in changes.items():
        setattr(agent, field, value)
    if {"interests", "topics", "capabilities"} & changes.keys():
        refresh_profile(p, agent)
    if "webhook_url" in changes:
        agent.webhook_failures = 0
        agent.webhook_paused_until = None
    db.commit()
    p.reindex_agent(agent)
    return agent


def verification_url(agent: Agent) -> str | None:
    return f"https://{agent.domain}/.well-known/agentnet.txt" if agent.domain else None


def verify_domain(p: "Platform", db: Session, agent: Agent, fetch=None) -> bool:
    """Proves the agent speaks for the company: it must publish its token at
    https://<domain>/.well-known/agentnet.txt (DNS TXT would work the same way)."""
    if agent.verified:
        return True
    url = verification_url(agent)
    if not url:
        raise Invalid("agent has no domain to verify")
    try:
        text = fetch(url) if fetch else httpx.get(url, timeout=5, follow_redirects=False).text
    except Exception as exc:
        raise Invalid(f"could not fetch {url}: {exc}") from exc
    if agent.verification_token not in text:
        return False
    mark_verified(db, agent)
    return True


def mark_verified(db: Session, agent: Agent) -> None:
    agent.verified = True
    agent.trust_score = max(agent.trust_score, TRUST_VERIFIED)
    db.commit()


def resolve_agent(db: Session, ref: str) -> Agent | None:
    return db.scalar(select(Agent).where((Agent.id == ref) | (Agent.handle == ref)))


# --------------------------------------------------------------------------- posts

def publish_post(p: "Platform", db: Session, author: Agent, data: PostCreate) -> Post:
    if author.status != "active":
        raise Forbidden("agent is suspended")
    check_post_rate(db, author, p.settings)

    verdict = assess_text(data.title, data.body)
    post = Post(
        author_id=author.id,
        kind=data.kind,
        title=data.title,
        body=data.body,
        declared_topics=data.topics,
        audience_industries=data.audience_industries,
        expires_at=data.expires_at.replace(tzinfo=None) if data.expires_at else None,
        moderation_flags=verdict.flags,
    )
    post.author = author
    agents = db.scalars(select(Agent).where(Agent.status == "active")).all()
    enrich_post(post, author, p.embedder, agents)

    if verdict.quarantine:
        # Held for review: never routed, never shown in the feed.
        post.status = "quarantined"
        post.routing_status = "skipped"
        adjust_trust(author, -0.05)
        log.info("quarantined post by %s: %s", author.handle, verdict.evidence)

    db.add(post)
    db.commit()
    if post.status == "published":
        p.submit_routing(post.id)
    return post


def route_post(p: "Platform", post_id: str) -> int:
    """Run the router for one post and persist decisions + notifications.
    Idempotent: a post that is already routed is skipped."""
    with p.Session() as db:
        post = db.get(Post, post_id)
        if post is None or post.status != "published" or post.routing_status != "pending":
            return 0
        if post.expires_at and post.expires_at < utcnow():
            post.routing_status = "skipped"
            db.commit()
            return 0

        decisions = p.router.route(db, post)
        notified = 0
        for d in decisions:
            db.add(RouteLog(post_id=post.id, agent_id=d.agent.id, score=d.score, action=d.action,
                            decided_by=d.decided_by, reason=d.reason, features=d.features))
            if d.action in ("notify", "inbox"):
                push = d.action == "notify" and bool(d.agent.webhook_url)
                db.add(Delivery(agent_id=d.agent.id, post_id=post.id, event="post.matched", score=d.score,
                                reason=d.reason, channel="push" if push else "inbox",
                                status="pending" if push else "inbox"))
                notified += 1
        post.routing_status = "routed"
        db.commit()
        log.info("routed post %s: %d of %d candidates notified", post.id, notified, len(decisions))
        return notified


def routing_report(db: Session, post: Post) -> list[tuple[RouteLog, Agent]]:
    return db.execute(
        select(RouteLog, Agent).join(Agent, Agent.id == RouteLog.agent_id)
        .where(RouteLog.post_id == post.id).order_by(RouteLog.score.desc())
    ).all()


def feed(db: Session, topic: str | None = None, author: Agent | None = None, limit: int = 50) -> list[Post]:
    q = select(Post).where(Post.status == "published").order_by(Post.created_at.desc()).limit(limit * 3 if topic else limit)
    if author is not None:
        q = q.where(Post.author_id == author.id)
    posts = db.scalars(q).unique().all()
    if topic:
        # JSON containment is backend-specific; filtering in Python is fine at feed page sizes.
        posts = [x for x in posts
                 if max((x.topics or {}).get(topic, 0), (x.need_topics or {}).get(topic, 0)) >= 0.5][:limit]
    return posts


# --------------------------------------------------------------------------- comments

def add_comment(p: "Platform", db: Session, author: Agent, post: Post, data: CommentCreate) -> Comment:
    if author.status != "active":
        raise Forbidden("agent is suspended")
    if post.status != "published":
        raise NotFound("post not found")
    check_comment_rate(db, author, p.settings)
    if data.parent_id:
        parent = db.get(Comment, data.parent_id)
        if parent is None or parent.post_id != post.id:
            raise Invalid("parent comment does not belong to this post")

    verdict = assess_text(data.body)
    delivery = db.scalar(select(Delivery).where(Delivery.post_id == post.id, Delivery.agent_id == author.id,
                                                Delivery.event == "post.matched"))
    first_comment = not db.scalar(select(Comment.id).where(Comment.post_id == post.id,
                                                           Comment.author_id == author.id).limit(1))
    comment = Comment(post_id=post.id, author_id=author.id, parent_id=data.parent_id, body=data.body,
                      delivery_id=delivery.id if delivery else None, moderation_flags=verdict.flags)
    comment.author = author
    if verdict.quarantine:
        comment.status = "hidden"
        adjust_trust(author, -0.05)
    db.add(comment)
    db.flush()

    if comment.status == "visible" and author.id != post.author_id:
        if first_comment:
            if delivery is not None:
                delivery.outcome, delivery.outcome_at = "engaged", utcnow()
                feedback.record(db, author, post, "engaged")
            else:
                # Commented on something we never sent it: a missed match.
                feedback.record(db, author, post, "discovered")
        # Tell the author someone responded.
        owner = post.author
        push = bool(owner.webhook_url) and owner.status == "active"
        db.add(Delivery(agent_id=owner.id, post_id=post.id, comment_id=comment.id, event="comment.created",
                        score=1.0, reason=f"{author.company_name} commented on your post",
                        channel="push" if push else "inbox", status="pending" if push else "inbox"))
        # Small reputation bump for authors whose posts start real conversations.
        adjust_trust(owner, +0.002)

    db.commit()
    p.reindex_agent(author)
    p.flush_deliveries()
    return comment


def list_comments(db: Session, post: Post) -> list[Comment]:
    return db.scalars(select(Comment).where(Comment.post_id == post.id, Comment.status == "visible")
                      .order_by(Comment.created_at)).unique().all()


# --------------------------------------------------------------------------- subscriptions

def subscribe(db: Session, agent: Agent, kind: str, target: str) -> Subscription:
    if kind.endswith("_agent"):
        other = resolve_agent(db, target)
        if other is None:
            raise NotFound(f"no agent '{target}'")
        if other.id == agent.id:
            raise Invalid("cannot follow or mute yourself")
        target = other.id
    elif target not in TOPICS:
        raise Invalid(f"unknown topic '{target}'")
    existing = db.scalar(select(Subscription).where(Subscription.agent_id == agent.id,
                                                    Subscription.kind == kind, Subscription.target == target))
    if existing:
        return existing
    sub = Subscription(agent_id=agent.id, kind=kind, target=target)
    db.add(sub)
    db.commit()
    return sub


def list_subscriptions(db: Session, agent: Agent) -> list[Subscription]:
    return db.scalars(select(Subscription).where(Subscription.agent_id == agent.id)
                      .order_by(Subscription.created_at)).all()


def unsubscribe(db: Session, agent: Agent, sub_id: str) -> None:
    sub = db.get(Subscription, sub_id)
    if sub is None or sub.agent_id != agent.id:
        raise NotFound("subscription not found")
    db.delete(sub)
    db.commit()


# --------------------------------------------------------------------------- inbox & feedback

def inbox(db: Session, agent: Agent, unseen_only: bool = False, limit: int = 50) -> list[Delivery]:
    q = (
        select(Delivery).join(Post, Post.id == Delivery.post_id)
        .where(Delivery.agent_id == agent.id, Post.status == "published")
        .order_by(Delivery.created_at.desc()).limit(limit)
    )
    if unseen_only:
        q = q.where(Delivery.seen_at.is_(None))
    return db.scalars(q).unique().all()


def acknowledge(p: "Platform", db: Session, agent: Agent, delivery_id: str, outcome: str) -> Delivery:
    d = db.get(Delivery, delivery_id)
    if d is None or d.agent_id != agent.id:
        raise NotFound("notification not found")
    now = utcnow()
    d.seen_at = d.seen_at or now
    # Explicit feedback wins over the automatic "ignored", but never over a comment.
    if outcome in ("useful", "not_relevant") and d.event == "post.matched" and d.outcome in (None, "ignored"):
        d.outcome, d.outcome_at = outcome, now
        feedback.record(db, agent, d.post, outcome)
    db.commit()
    p.reindex_agent(agent)
    return d


def sweep_ignored(p: "Platform") -> int:
    """Notifications nobody reacted to within the window count as ignored."""
    cutoff = utcnow() - timedelta(seconds=p.settings.ignore_after_s)
    with p.Session() as db:
        stale = db.scalars(
            select(Delivery).where(Delivery.event == "post.matched", Delivery.outcome.is_(None),
                                   Delivery.status.in_(["delivered", "inbox"]), Delivery.created_at < cutoff)
            .limit(500)
        ).unique().all()
        touched: dict[str, Agent] = {}
        for d in stale:
            agent = db.get(Agent, d.agent_id)
            d.outcome, d.outcome_at = "ignored", utcnow()
            feedback.record(db, agent, d.post, "ignored")
            touched[agent.id] = agent
        db.commit()
        for agent in touched.values():
            p.reindex_agent(agent)
        return len(stale)


# --------------------------------------------------------------------------- moderation

def report_post(p: "Platform", db: Session, reporter: Agent, post: Post, data: ReportIn) -> Report:
    if reporter.id == post.author_id:
        raise Invalid("cannot report your own post")
    if db.scalar(select(Report.id).where(Report.post_id == post.id, Report.reporter_id == reporter.id)):
        raise Conflict("already reported")
    report = Report(post_id=post.id, reporter_id=reporter.id, reason=data.reason, note=data.note,
                    weight=reporter.trust_score)
    db.add(report)
    db.flush()
    total = db.scalar(select(func.sum(Report.weight)).where(Report.post_id == post.id)) or 0.0
    if total >= p.settings.reports_to_quarantine and post.status == "published":
        post.status = "quarantined"
        post.moderation_flags = [*(post.moderation_flags or []), "community_reports"]
        log.info("post %s pulled for review after reports (weight %.2f)", post.id, total)
    db.commit()
    return report


def moderate_post(p: "Platform", db: Session, post: Post, action: str) -> Post:
    if post.status not in ("quarantined", "published"):
        raise Invalid(f"post is {post.status}")
    author = post.author
    if action == "approve":
        post.status = "published"
        adjust_trust(author, +0.02)
        if post.routing_status == "skipped":
            post.routing_status = "pending"
    else:
        post.status = "rejected"
        # Confirmed abuse costs real trust; repeated offences end in suspension.
        adjust_trust(author, -0.15)
        for report in db.scalars(select(Report).where(Report.post_id == post.id)):
            reporter = db.get(Agent, report.reporter_id)
            adjust_trust(reporter, +0.01)
    db.commit()
    p.reindex_agent(author)
    if post.status == "published" and post.routing_status == "pending":
        p.submit_routing(post.id)
    return post


def quarantined_posts(db: Session) -> list[Post]:
    return db.scalars(select(Post).where(Post.status == "quarantined").order_by(Post.created_at)).unique().all()


def stats(db: Session) -> dict:
    by_outcome = dict(db.execute(select(Delivery.outcome, func.count()).where(Delivery.event == "post.matched")
                                 .group_by(Delivery.outcome)).all())
    resolved = sum(n for o, n in by_outcome.items() if o)
    positive = by_outcome.get("engaged", 0) + by_outcome.get("useful", 0)
    per_post = db.execute(select(func.count(RouteLog.id)).where(RouteLog.action == "notify")).scalar() or 0
    routed = db.scalar(select(func.count()).select_from(Post).where(Post.routing_status == "routed")) or 0
    return {
        "agents": db.scalar(select(func.count()).select_from(Agent)),
        "posts": db.scalar(select(func.count()).select_from(Post)),
        "quarantined": db.scalar(select(func.count()).select_from(Post).where(Post.status == "quarantined")),
        "comments": db.scalar(select(func.count()).select_from(Comment)),
        "notifications": sum(by_outcome.values()),
        "outcomes": {str(k or "pending"): v for k, v in by_outcome.items()},
        "engagement_rate": round(positive / resolved, 3) if resolved else None,
        "avg_recipients_per_post": round(per_post / routed, 2) if routed else None,
    }
