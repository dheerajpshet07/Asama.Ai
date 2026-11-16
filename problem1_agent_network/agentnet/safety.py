"""Trust and safety checks.

Layers, cheapest first:
  1. Identity: every write is authenticated with the agent's API key; the author
     of a post is always the authenticated agent, never a field in the body.
  2. Rate limits that scale with trust (unverified agents get a small budget).
  3. Content rules below: prompt-injection phrasing, credential/OTP phishing,
     scammy financial claims, link shorteners. Hits quarantine the post for review
     instead of routing it; nothing quarantined reaches other agents.
  4. Community reports, weighted by the reporter's own trust.
  5. Trust score feeds back into routing reach and rate limits.

The regexes are a first line, not the whole defence. The bigger protection is that
receiving agents always get post text as clearly-marked data (see delivery.py),
and that a model-based classifier (e.g. Llama Guard) can be plugged in at
`assess_text` for production.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import Settings
from .models import Agent, Comment, Post
from .util import utcnow

_INJECTION = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"\bdisregard (all |any |your |the )?(previous |prior )?(instructions|rules|guidelines)",
    r"\b(reveal|print|show|output|send|share|leak) (me )?(your |the )?(system prompt|instructions|api[ _-]?keys?|secrets?|credentials|passwords?|tokens?|webhook secret)",
    r"\byou are now\b.{0,40}\b(mode|unrestricted|jailbroken|dan)\b",
    r"\bact as (the |an? )?(system|administrator|admin|root|developer)\b",
    r"<\s*/?\s*(system|assistant|instructions?|tool_call)\s*>",
    r"\b(begin|end) (system|admin) (prompt|message|instructions)\b",
    r"\b(call|invoke|execute|run) (the |your )?(tool|function|command)s?\b",
    r"\bnew instructions\s*:",
]
# Phrased as instructions to the reader ("enter your OTP"), so that a legitimate
# warning like "fraudsters ask users to enter their UPI PIN" is not caught.
_PHISHING = [
    r"\b(share|enter|send|confirm|verify|provide)\s+(your|ur)\s+(\w+\s+){0,2}(otp|pin|cvv|password)\b",
    r"\bkyc\b.{0,40}\b(expired|suspended|blocked)\b.{0,60}\b(click|link|call)\b",
]
_SCAM = [
    r"\bguaranteed (returns?|profits?)\b",
    r"\b(double|triple) your (money|investment)\b",
    r"\brisk[- ]free (returns?|profits?|investment)\b",
    r"\bcrypto (giveaway|airdrop)\b",
]
_SHORTENERS = re.compile(r"https?://(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|cutt\.ly|rb\.gy|shorturl\.at)/", re.I)
_RAW_IP_URL = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}", re.I)

_RULES = (
    [("prompt_injection", re.compile(p, re.I)) for p in _INJECTION]
    + [("phishing", re.compile(p, re.I)) for p in _PHISHING]
    + [("scam_claim", re.compile(p, re.I)) for p in _SCAM]
)
# Flags that hold a post for human review. Others only reduce reach.
_QUARANTINE = {"prompt_injection", "phishing", "scam_claim"}


@dataclass
class Assessment:
    flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def quarantine(self) -> bool:
        return any(f in _QUARANTINE for f in self.flags)


def assess_text(*parts: str) -> Assessment:
    text = "\n".join(p for p in parts if p)
    result = Assessment()
    for flag, rx in _RULES:
        m = rx.search(text)
        if m and flag not in result.flags:
            result.flags.append(flag)
            result.evidence.append(m.group(0)[:120])
    if _SHORTENERS.search(text):
        result.flags.append("link_shortener")
    if _RAW_IP_URL.search(text):
        result.flags.append("raw_ip_link")
    return result


# --------------------------------------------------------------------------- webhooks

class UnsafeWebhook(ValueError):
    pass


def check_webhook_url(url: str, allow_private: bool) -> str:
    """Reject webhook URLs that would let an agent make us call internal
    services (SSRF). Re-checked at delivery time because DNS can change."""
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise UnsafeWebhook("webhook must be an http(s) URL")
    if not parsed.hostname:
        raise UnsafeWebhook("webhook URL has no host")
    if allow_private:
        return url
    if parsed.scheme != "https":
        raise UnsafeWebhook("webhook must use https")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeWebhook(f"cannot resolve {parsed.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise UnsafeWebhook(f"{parsed.hostname} resolves to a non-public address")
    return url


# --------------------------------------------------------------------------- rate limits

class RateLimited(Exception):
    def __init__(self, message: str, retry_after: int = 3600):
        super().__init__(message)
        self.retry_after = retry_after


def post_limit(agent: Agent, settings: Settings) -> int:
    base = settings.posts_per_hour_verified if agent.verified else settings.posts_per_hour_unverified
    # Low-trust agents get a fraction of the budget, but never zero (they can appeal by behaving).
    return max(1, int(base * min(1.0, 0.25 + agent.trust_score)))


def check_post_rate(db: Session, agent: Agent, settings: Settings) -> None:
    since = utcnow() - timedelta(hours=1)
    n = db.scalar(select(func.count()).select_from(Post).where(Post.author_id == agent.id, Post.created_at >= since))
    limit = post_limit(agent, settings)
    if n >= limit:
        raise RateLimited(f"post limit reached ({limit}/hour)")


def check_comment_rate(db: Session, agent: Agent, settings: Settings) -> None:
    since = utcnow() - timedelta(hours=1)
    n = db.scalar(
        select(func.count()).select_from(Comment).where(Comment.author_id == agent.id, Comment.created_at >= since)
    )
    if n >= settings.comments_per_hour:
        raise RateLimited(f"comment limit reached ({settings.comments_per_hour}/hour)")


# --------------------------------------------------------------------------- trust

TRUST_UNVERIFIED = 0.3
TRUST_VERIFIED = 0.7
SUSPEND_BELOW = 0.1


def adjust_trust(agent: Agent, delta: float) -> None:
    agent.trust_score = round(max(0.0, min(1.0, agent.trust_score + delta)), 4)
    if agent.trust_score < SUSPEND_BELOW:
        agent.status = "suspended"


def reach_factor(trust: float) -> float:
    """Multiplier on relevance scores for posts by an author with this trust.
    New agents still reach people, just with a slightly higher bar."""
    return 0.75 + 0.25 * max(0.0, min(1.0, trust))
