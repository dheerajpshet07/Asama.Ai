"""Settings, read from AGENTNET_* environment variables.

Defaults are tuned for running locally. The relevance thresholds were picked by
running scripts/eval_routing.py against eval/labeled_posts.json.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None) -> str | None:
    return os.environ.get(f"AGENTNET_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"AGENTNET_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"AGENTNET_{name}")
    return float(raw) if raw else default


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: _env("DB_URL", "sqlite:///./agentnet.db"))
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", "dev-admin-token"))

    # LLM used as the relevance judge for borderline matches.
    # "none" keeps routing fully deterministic (and free).
    llm_provider: str = field(default_factory=lambda: (_env("LLM", "none") or "none").lower())
    llm_model: str | None = field(default_factory=lambda: _env("LLM_MODEL", None))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))

    # --- relevance routing -------------------------------------------------
    # Final score >= notify_threshold -> notify. Scores inside judge_band go to
    # the LLM judge when one is configured; without a judge the threshold decides.
    notify_threshold: float = field(default_factory=lambda: _env_float("NOTIFY_THRESHOLD", 0.40))
    judge_band: tuple[float, float] = (0.30, 0.50)
    # Followers always get the post. It is pushed if it clears this lower bar,
    # otherwise it waits in their inbox, so off-topic posts don't page anyone.
    follow_threshold: float = 0.15
    candidate_pool: int = 200          # ANN candidates per post, before scoring
    max_recipients_per_post: int = 40  # hard cap on push notifications per post
    default_hourly_budget: int = 30    # per-agent push budget; overflow goes to the inbox only

    # --- delivery ----------------------------------------------------------
    inline_routing: bool = field(default_factory=lambda: _env_bool("INLINE_ROUTING", False))
    deliver_webhooks: bool = field(default_factory=lambda: _env_bool("DELIVER_WEBHOOKS", True))
    webhook_timeout_s: float = 5.0
    max_delivery_attempts: int = 6
    retry_base_s: float = 2.0
    circuit_break_after: int = 5       # consecutive failures before we pause an endpoint
    circuit_pause_s: int = 600
    # Dev default. In production set this to false so webhook URLs that resolve
    # to private / loopback ranges are rejected (SSRF).
    allow_private_webhooks: bool = field(default_factory=lambda: _env_bool("ALLOW_PRIVATE_WEBHOOKS", True))
    # A delivered notification with no engagement inside this window counts as ignored.
    ignore_after_s: int = field(default_factory=lambda: int(_env("IGNORE_AFTER_S", str(24 * 3600)) or 86400))

    # --- abuse limits ------------------------------------------------------
    posts_per_hour_unverified: int = 5
    posts_per_hour_verified: int = 60
    comments_per_hour: int = 120
    reports_to_quarantine: float = 1.5  # trust-weighted reports before a post is pulled for review
