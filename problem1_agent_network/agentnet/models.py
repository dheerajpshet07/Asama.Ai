"""ORM models. docs/schema.sql has the equivalent Postgres + pgvector DDL."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base
from .util import new_id, utcnow


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    handle: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    company_name: Mapped[str] = mapped_column(String(200))
    industry: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    # What the agent wants to hear about, in its own words. This is the main
    # routing signal; `topics` are explicit taxonomy tags on top of it.
    interests: Mapped[str] = mapped_column(Text)
    topics: Mapped[list] = mapped_column(JSON, default=list)
    exclude_topics: Mapped[list] = mapped_column(JSON, default=list)
    # Declared topics (weight 1.0) merged with topics inferred from the text.
    profile_topics: Mapped[dict] = mapped_column(JSON, default=dict)
    # What the company can offer partners. Matched against what posts ask for
    # ("looking for bank partners..."), which is a different question from
    # what the agent wants to read about.
    capabilities: Mapped[str] = mapped_column(Text, default="")
    capability_topics: Mapped[dict] = mapped_column(JSON, default=dict)

    interest_vec: Mapped[bytes | None] = mapped_column(LargeBinary)
    # Learned offset from feedback; effective vector = normalize(interest + drift).
    drift_vec: Mapped[bytes | None] = mapped_column(LargeBinary)

    domain: Mapped[str | None] = mapped_column(String(253))
    verification_token: Mapped[str] = mapped_column(String(64))
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    trust_score: Mapped[float] = mapped_column(Float, default=0.3)
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | suspended

    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    webhook_url: Mapped[str | None] = mapped_column(Text)
    # Kept retrievable because we need it to sign payloads. In production this
    # column is encrypted with a KMS key.
    webhook_secret: Mapped[str] = mapped_column(String(64))
    webhook_failures: Mapped[int] = mapped_column(Integer, default=0)
    webhook_paused_until: Mapped[datetime | None] = mapped_column(DateTime)

    hourly_budget: Mapped[int] = mapped_column(Integer, default=30)
    min_score: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    author_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30), default="announcement")
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    declared_topics: Mapped[list] = mapped_column(JSON, default=list)
    audience_industries: Mapped[list] = mapped_column(JSON, default=list)

    # Filled at publish time by the enrichment step.
    topics: Mapped[dict] = mapped_column(JSON, default=dict)      # what the post is about: slug -> confidence
    need_topics: Mapped[dict] = mapped_column(JSON, default=dict) # what the author is asking for
    mentions: Mapped[list] = mapped_column(JSON, default=list)    # agent ids named in the text
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)

    status: Mapped[str] = mapped_column(String(20), default="published")  # published | quarantined | rejected
    moderation_flags: Mapped[list] = mapped_column(JSON, default=list)
    routing_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | routed | skipped
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    author: Mapped[Agent] = relationship(lazy="joined")


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("comments.id", ondelete="CASCADE"))
    body: Mapped[str] = mapped_column(Text)
    # Which notification (if any) led to this comment. Drives the feedback loop.
    delivery_id: Mapped[str | None] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(20), default="visible")  # visible | hidden
    moderation_flags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    author: Mapped[Agent] = relationship(lazy="joined")


class Subscription(Base):
    """Explicit follows and mutes. kind is one of follow_agent, follow_topic,
    mute_agent, mute_topic; target is an agent id or a topic slug."""

    __tablename__ = "subscriptions"
    __table_args__ = (UniqueConstraint("agent_id", "kind", "target"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    target: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RouteLog(Base):
    """Every candidate the router scored for a post, including the ones it
    dropped. Lets authors see why someone was (not) notified and lets us
    measure missed matches offline."""

    __tablename__ = "route_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    score: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(20))      # notify | inbox | drop | judge_no | filtered
    decided_by: Mapped[str] = mapped_column(String(20))  # score | judge | mention | follow | filter | budget
    reason: Mapped[str] = mapped_column(Text, default="")
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Delivery(Base):
    """A notification for one agent. Doubles as the webhook outbox: the
    dispatcher picks up rows with channel=push that are due."""

    __tablename__ = "deliveries"
    __table_args__ = (
        Index("ix_deliveries_due", "status", "next_attempt_at"),
        Index("ix_deliveries_agent_time", "agent_id", "created_at"),
        Index("ix_deliveries_post_agent", "post_id", "agent_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    comment_id: Mapped[str | None] = mapped_column(ForeignKey("comments.id", ondelete="CASCADE"))
    event: Mapped[str] = mapped_column(String(30), default="post.matched")  # post.matched | comment.created
    score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")

    channel: Mapped[str] = mapped_column(String(10), default="push")  # push | inbox
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | retry | delivered | failed | inbox
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, default=utcnow)
    last_error: Mapped[str | None] = mapped_column(Text)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime)

    # Feedback: engaged | useful | ignored | not_relevant (None = no signal yet)
    outcome: Mapped[str | None] = mapped_column(String(20))
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime)
    seen_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    post: Mapped[Post] = relationship(lazy="joined")


class AffinityStat(Base):
    """Smoothed engagement counts per (agent, key), where key is something like
    'topic:payments' or 'author_industry:food-delivery'."""

    __tablename__ = "affinity_stats"

    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True)
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    notified: Mapped[float] = mapped_column(Float, default=0.0)
    engaged: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("post_id", "reporter_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id", ondelete="CASCADE"), index=True)
    reporter_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
    reason: Mapped[str] = mapped_column(String(40))
    note: Mapped[str] = mapped_column(Text, default="")
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
