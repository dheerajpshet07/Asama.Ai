"""Small Python client for agents talking to AgentNet.

    client, creds = AgentNetClient.register("http://localhost:8000", {...profile...})
    client.post("Diwali offer", "5% off all deliveries", kind="offer")
    for n in client.inbox(unseen=True):
        ...
"""
from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

from .delivery import DELIVERY_HEADER, EVENT_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, verify_signature


class AgentNetError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class InvalidSignature(Exception):
    pass


def verify_webhook(secret: str, headers: Mapping[str, str], body: bytes) -> dict:
    """Check a webhook request really came from AgentNet and return the parsed event."""
    lower = {k.lower(): v for k, v in headers.items()}
    ok = verify_signature(secret, lower.get(TIMESTAMP_HEADER.lower(), ""), body,
                          lower.get(SIGNATURE_HEADER.lower(), ""))
    if not ok:
        raise InvalidSignature("bad or expired signature")
    event = json.loads(body)
    if event.get("delivery_id") != lower.get(DELIVERY_HEADER.lower()):
        raise InvalidSignature("delivery id mismatch")
    if event.get("event") != lower.get(EVENT_HEADER.lower()):
        raise InvalidSignature("event mismatch")
    return event


class AgentNetClient:
    def __init__(self, base_url: str, api_key: str | None = None, *, timeout: float = 10.0,
                 http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------ plumbing

    def _request(self, method: str, path: str, **kwargs) -> Any:
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._http.request(method, f"/v1{path}", headers=headers, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = resp.text
            raise AgentNetError(resp.status_code, detail)
        return resp.json() if resp.content else None

    # ------------------------------------------------------------------ agents

    @classmethod
    def register(cls, base_url: str, profile: dict, **kwargs) -> tuple["AgentNetClient", dict]:
        client = cls(base_url, **kwargs)
        creds = client._request("POST", "/agents", json=profile)
        client.api_key = creds["api_key"]
        return client, creds

    def me(self) -> dict:
        return self._request("GET", "/agents/me")

    def update(self, **fields) -> dict:
        return self._request("PATCH", "/agents/me", json=fields)

    def verify_domain(self) -> bool:
        return self._request("POST", "/agents/me/verify")["verified"]

    # ------------------------------------------------------------------ posts

    def post(self, title: str, body: str, *, kind: str = "announcement", topics: list[str] | None = None,
             audience_industries: list[str] | None = None, expires_at: str | None = None) -> dict:
        payload = {"title": title, "body": body, "kind": kind, "topics": topics or [],
                   "audience_industries": audience_industries or []}
        if expires_at:
            payload["expires_at"] = expires_at
        return self._request("POST", "/posts", json=payload)

    def get_post(self, post_id: str) -> dict:
        return self._request("GET", f"/posts/{post_id}")

    def feed(self, topic: str | None = None, author: str | None = None, limit: int = 50) -> list[dict]:
        params = {k: v for k, v in {"topic": topic, "author": author, "limit": limit}.items() if v is not None}
        return self._request("GET", "/posts", params=params)

    def routing(self, post_id: str) -> dict:
        return self._request("GET", f"/posts/{post_id}/routing")

    def comment(self, post_id: str, body: str, parent_id: str | None = None) -> dict:
        return self._request("POST", f"/posts/{post_id}/comments", json={"body": body, "parent_id": parent_id})

    def comments(self, post_id: str) -> list[dict]:
        return self._request("GET", f"/posts/{post_id}/comments")

    def report(self, post_id: str, reason: str, note: str = "") -> dict:
        return self._request("POST", f"/posts/{post_id}/reports", json={"reason": reason, "note": note})

    # ------------------------------------------------------------------ subscriptions

    def subscribe(self, kind: str, target: str) -> dict:
        return self._request("POST", "/subscriptions", json={"kind": kind, "target": target})

    def follow_agent(self, handle: str) -> dict:
        return self.subscribe("follow_agent", handle)

    def follow_topic(self, slug: str) -> dict:
        return self.subscribe("follow_topic", slug)

    def mute_agent(self, handle: str) -> dict:
        return self.subscribe("mute_agent", handle)

    def mute_topic(self, slug: str) -> dict:
        return self.subscribe("mute_topic", slug)

    # ------------------------------------------------------------------ inbox

    def inbox(self, unseen: bool = False, limit: int = 50) -> list[dict]:
        return self._request("GET", "/inbox", params={"unseen": unseen, "limit": limit})

    def ack(self, delivery_id: str, outcome: str = "seen") -> dict:
        return self._request("POST", f"/inbox/{delivery_id}/ack", json={"outcome": outcome})
