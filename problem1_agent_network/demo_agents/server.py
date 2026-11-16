"""A tiny web app that hosts the webhook endpoints of all demo agents.

Each agent gets POST /hooks/<handle>. The handler verifies the HMAC signature,
acknowledges quickly (200), and does the slow part (deciding, maybe calling an
LLM, commenting) in the background, which is how a real receiver should
behave so the platform's delivery timeout is never hit.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agentnet.client import AgentNetClient, AgentNetError, InvalidSignature, verify_webhook

from .profiles import DemoAgent
from .responder import Responder

log = logging.getLogger(__name__)


@dataclass
class AgentRuntime:
    profile: DemoAgent
    client: AgentNetClient
    webhook_secret: str
    responder: Responder
    events: list[dict] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def handle(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)
        if event["event"] == "comment.created":
            c = event["comment"]
            self.actions.append(f"saw comment from {c['author']['company_name']}")
            return
        if event["event"] != "post.matched":
            return

        post = event["post"]
        reaction = self.responder.react(event)
        try:
            if reaction.engage:
                self.client.comment(post["id"], reaction.comment)
                self.actions.append(f"commented on '{post['title']}'")
            else:
                # Telling the platform it missed is what makes routing better next time.
                self.client.ack(event["delivery_id"], "not_relevant")
                self.actions.append(f"marked '{post['title']}' not relevant ({reaction.reason})")
        except AgentNetError as exc:
            log.warning("%s could not respond: %s", self.profile.handle, exc)


def create_agents_app(runtimes: dict[str, AgentRuntime]) -> FastAPI:
    app = FastAPI(title="Demo company agents")

    @app.post("/hooks/{handle}")
    async def hook(handle: str, request: Request, background: BackgroundTasks):
        rt = runtimes.get(handle)
        if rt is None:
            raise HTTPException(404, "unknown agent")
        body = await request.body()
        try:
            event = verify_webhook(rt.webhook_secret, request.headers, body)
        except InvalidSignature as exc:
            raise HTTPException(401, str(exc)) from exc
        background.add_task(rt.handle, event)
        return {"received": event["delivery_id"]}

    return app
