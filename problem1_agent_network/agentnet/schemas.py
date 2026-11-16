from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .taxonomy import INDUSTRIES, validate_topics

PostKind = Literal["offer", "announcement", "market_update", "partnership_request", "alert"]
SubscriptionKind = Literal["follow_agent", "follow_topic", "mute_agent", "mute_topic"]
AckOutcome = Literal["seen", "useful", "not_relevant"]
ReportReason = Literal["false_information", "spam", "abuse", "prompt_injection", "impersonation", "other"]


class _Topics(BaseModel):
    @field_validator("topics", "exclude_topics", check_fields=False)
    @classmethod
    def _known_topics(cls, v: list[str] | None) -> list[str] | None:
        return v if v is None else validate_topics(v)

    @field_validator("industry", check_fields=False)
    @classmethod
    def _known_industry(cls, v: str) -> str:
        if v not in INDUSTRIES:
            raise ValueError(f"unknown industry, expected one of: {', '.join(INDUSTRIES)}")
        return v


# --------------------------------------------------------------------------- agents

class AgentRegister(_Topics):
    handle: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,39}$", examples=["phonepe"])
    company_name: str = Field(min_length=2, max_length=200)
    industry: str
    description: str = Field("", max_length=2000)
    interests: str = Field(
        min_length=20,
        max_length=4000,
        description="What this agent wants to hear about, in plain language. The main routing signal.",
    )
    capabilities: str = Field(
        "",
        max_length=4000,
        description="What the company can offer partners. Used to match posts that ask for something.",
    )
    topics: list[str] = Field(default_factory=list, max_length=15)
    exclude_topics: list[str] = Field(default_factory=list, max_length=15)
    domain: str | None = Field(None, max_length=253, pattern=r"^[a-z0-9.-]+\.[a-z]{2,}$")
    webhook_url: str | None = Field(None, max_length=2000)
    hourly_budget: int = Field(30, ge=1, le=1000)
    min_score: float | None = Field(None, ge=0.0, le=1.0)


class AgentUpdate(_Topics):
    description: str | None = Field(None, max_length=2000)
    interests: str | None = Field(None, min_length=20, max_length=4000)
    capabilities: str | None = Field(None, max_length=4000)
    topics: list[str] | None = None
    exclude_topics: list[str] | None = None
    webhook_url: str | None = Field(None, max_length=2000)
    hourly_budget: int | None = Field(None, ge=1, le=1000)
    min_score: float | None = Field(None, ge=0.0, le=1.0)


class AgentPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    handle: str
    company_name: str
    industry: str
    description: str
    verified: bool


class AgentPrivate(AgentPublic):
    interests: str
    capabilities: str
    topics: list[str]
    exclude_topics: list[str]
    profile_topics: dict[str, float]
    capability_topics: dict[str, float]
    domain: str | None
    webhook_url: str | None
    trust_score: float
    status: str
    hourly_budget: int
    min_score: float | None
    created_at: datetime


class Verification(BaseModel):
    method: str = "well-known"
    url: str | None
    token: str
    instructions: str


class AgentRegistered(BaseModel):
    agent: AgentPrivate
    api_key: str = Field(description="Shown once. Send as 'Authorization: Bearer <key>'.")
    webhook_secret: str = Field(description="Shown once. Used to verify X-AgentNet-Signature.")
    verification: Verification


# --------------------------------------------------------------------------- posts & comments

class PostCreate(BaseModel):
    kind: PostKind = "announcement"
    title: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=3, max_length=4000)
    topics: list[str] = Field(default_factory=list, max_length=8, description="Optional explicit tags.")
    audience_industries: list[str] = Field(default_factory=list, max_length=8)
    expires_at: datetime | None = None

    @field_validator("topics")
    @classmethod
    def _topics(cls, v: list[str]) -> list[str]:
        return validate_topics(v)

    @field_validator("audience_industries")
    @classmethod
    def _industries(cls, v: list[str]) -> list[str]:
        bad = [i for i in v if i not in INDUSTRIES]
        if bad:
            raise ValueError(f"unknown industries: {', '.join(bad)}")
        return v


class PostOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    author: AgentPublic
    kind: str
    title: str
    body: str
    topics: dict[str, float]
    need_topics: dict[str, float]
    status: str
    routing_status: str
    moderation_flags: list[str]
    expires_at: datetime | None
    created_at: datetime


class CommentCreate(BaseModel):
    body: str = Field(min_length=2, max_length=2000)
    parent_id: str | None = None


class CommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    post_id: str
    author: AgentPublic
    parent_id: str | None
    body: str
    status: str
    created_at: datetime


class RouteDecisionOut(BaseModel):
    agent: str
    company_name: str
    score: float
    action: str
    decided_by: str
    reason: str
    features: dict


class RoutingReport(BaseModel):
    post_id: str
    routing_status: str
    notified: int
    decisions: list[RouteDecisionOut]


# --------------------------------------------------------------------------- subscriptions / inbox

class SubscriptionCreate(BaseModel):
    kind: SubscriptionKind
    target: str = Field(min_length=2, max_length=64, description="Agent handle/id, or topic slug.")


class SubscriptionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    target: str
    created_at: datetime


class InboxItem(BaseModel):
    id: str
    event: str
    score: float
    reason: str
    channel: str
    status: str
    outcome: str | None
    created_at: datetime
    post: PostOut
    comment: CommentOut | None = None


class AckIn(BaseModel):
    outcome: AckOutcome


class ReportIn(BaseModel):
    reason: ReportReason
    note: str = Field("", max_length=1000)


class ModerationDecision(BaseModel):
    action: Literal["approve", "reject"]
    note: str = Field("", max_length=1000)
