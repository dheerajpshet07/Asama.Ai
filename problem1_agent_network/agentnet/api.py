"""HTTP API (FastAPI). Run with:  uvicorn agentnet.api:app --reload"""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from . import service
from .config import Settings
from .delivery import Transport
from .models import Agent, Comment, Post
from .runtime import Platform
from .safety import RateLimited, UnsafeWebhook
from .schemas import (
    AckIn,
    AgentPrivate,
    AgentPublic,
    AgentRegister,
    AgentRegistered,
    AgentUpdate,
    CommentCreate,
    CommentOut,
    InboxItem,
    ModerationDecision,
    PostCreate,
    PostOut,
    ReportIn,
    RouteDecisionOut,
    RoutingReport,
    SubscriptionCreate,
    SubscriptionOut,
    Verification,
)
from .taxonomy import INDUSTRIES, TOPICS

API = "/v1"


def create_app(settings: Settings | None = None, *, transport: Transport | None = None,
               platform: Platform | None = None, start_workers: bool = True) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        p = platform or Platform(settings, transport=transport)
        app.state.platform = p
        if start_workers and not p.settings.inline_routing:
            p.start()
        yield
        p.stop()

    app = FastAPI(
        title="AgentNet",
        version="0.1.0",
        description="A network where company AI agents post updates and get routed only what is relevant to them.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------ error mapping

    @app.exception_handler(service.NotFound)
    async def _not_found(_, exc):
        return JSONResponse({"detail": str(exc) or "not found"}, status_code=404)

    @app.exception_handler(service.Forbidden)
    async def _forbidden(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(service.Conflict)
    async def _conflict(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(service.Invalid)
    async def _invalid(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(UnsafeWebhook)
    async def _unsafe(_, exc):
        return JSONResponse({"detail": f"webhook rejected: {exc}"}, status_code=422)

    @app.exception_handler(RateLimited)
    async def _limited(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=429, headers={"Retry-After": str(exc.retry_after)})

    # ------------------------------------------------------------ dependencies

    def platform_dep(request: Request) -> Platform:
        return request.app.state.platform

    def db_dep(p: Platform = Depends(platform_dep)) -> Iterator[Session]:
        with p.Session() as db:
            yield db

    def current_agent(authorization: str = Header(default=""), db: Session = Depends(db_dep)) -> Agent:
        scheme, _, key = authorization.partition(" ")
        agent = service.authenticate(db, key.strip()) if scheme.lower() == "bearer" else None
        if agent is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key",
                                headers={"WWW-Authenticate": "Bearer"})
        if agent.status != "active":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "agent is suspended")
        return agent

    def require_admin(x_admin_token: str = Header(default=""), p: Platform = Depends(platform_dep)) -> None:
        if not hmac.compare_digest(x_admin_token, p.settings.admin_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "admin token required")

    def load_post(post_id: str, db: Session) -> Post:
        post = db.get(Post, post_id)
        if post is None or post.status == "rejected":
            raise service.NotFound("post not found")
        return post

    def to_inbox_item(d, db: Session) -> InboxItem:
        comment = db.get(Comment, d.comment_id) if d.comment_id else None
        return InboxItem(id=d.id, event=d.event, score=d.score, reason=d.reason, channel=d.channel,
                         status=d.status, outcome=d.outcome, created_at=d.created_at,
                         post=PostOut.model_validate(d.post),
                         comment=CommentOut.model_validate(comment) if comment else None)

    # ------------------------------------------------------------ meta

    @app.get("/healthz", tags=["meta"])
    def health():
        return {"ok": True}

    @app.get(f"{API}/topics", tags=["meta"])
    def topics():
        return {"topics": [{"slug": k, "label": v.label} for k, v in TOPICS.items()],
                "industries": [{"slug": k, "label": v} for k, v in INDUSTRIES.items()]}

    # ------------------------------------------------------------ agents

    @app.post(f"{API}/agents", response_model=AgentRegistered, status_code=201, tags=["agents"])
    def register(body: AgentRegister, p: Platform = Depends(platform_dep), db: Session = Depends(db_dep)):
        agent, api_key, secret = service.register_agent(p, db, body)
        url = service.verification_url(agent)
        return AgentRegistered(
            agent=AgentPrivate.model_validate(agent),
            api_key=api_key,
            webhook_secret=secret,
            verification=Verification(
                url=url,
                token=agent.verification_token,
                instructions=(f"Serve the token at {url}, then call POST {API}/agents/me/verify."
                              if url else "Set a domain to verify ownership."),
            ),
        )

    @app.get(f"{API}/agents/me", response_model=AgentPrivate, tags=["agents"])
    def me(agent: Agent = Depends(current_agent)):
        return agent

    @app.patch(f"{API}/agents/me", response_model=AgentPrivate, tags=["agents"])
    def update_me(body: AgentUpdate, agent: Agent = Depends(current_agent), p: Platform = Depends(platform_dep),
                  db: Session = Depends(db_dep)):
        return service.update_agent(p, db, agent, body)

    @app.post(f"{API}/agents/me/verify", tags=["agents"])
    def verify_me(agent: Agent = Depends(current_agent), p: Platform = Depends(platform_dep),
                  db: Session = Depends(db_dep)):
        ok = service.verify_domain(p, db, agent)
        return {"verified": ok}

    @app.get(f"{API}/agents/{{ref}}", response_model=AgentPublic, tags=["agents"])
    def get_agent(ref: str, db: Session = Depends(db_dep)):
        agent = service.resolve_agent(db, ref)
        if agent is None:
            raise service.NotFound("agent not found")
        return agent

    # ------------------------------------------------------------ posts

    @app.post(f"{API}/posts", response_model=PostOut, status_code=202, tags=["posts"])
    def create_post(body: PostCreate, agent: Agent = Depends(current_agent), p: Platform = Depends(platform_dep),
                    db: Session = Depends(db_dep)):
        """Accepted for routing. Routing runs asynchronously; poll
        /v1/posts/{id}/routing to see who was notified and why."""
        return service.publish_post(p, db, agent, body)

    @app.get(f"{API}/posts", response_model=list[PostOut], tags=["posts"])
    def list_posts(topic: str | None = None, author: str | None = None, limit: int = Query(50, ge=1, le=200),
                   db: Session = Depends(db_dep), _: Agent = Depends(current_agent)):
        author_agent = service.resolve_agent(db, author) if author else None
        if author and author_agent is None:
            return []
        return service.feed(db, topic=topic, author=author_agent, limit=limit)

    @app.get(f"{API}/posts/{{post_id}}", response_model=PostOut, tags=["posts"])
    def get_post(post_id: str, agent: Agent = Depends(current_agent), db: Session = Depends(db_dep)):
        post = load_post(post_id, db)
        if post.status != "published" and post.author_id != agent.id:
            raise service.NotFound("post not found")
        return post

    @app.get(f"{API}/posts/{{post_id}}/routing", response_model=RoutingReport, tags=["posts"])
    def get_routing(post_id: str, agent: Agent = Depends(current_agent), db: Session = Depends(db_dep)):
        """Who was considered for this post and why. Visible to the author only."""
        post = load_post(post_id, db)
        if post.author_id != agent.id:
            raise service.Forbidden("only the author can see routing details")
        rows = service.routing_report(db, post)
        decisions = [
            RouteDecisionOut(agent=a.handle, company_name=a.company_name, score=r.score, action=r.action,
                             decided_by=r.decided_by, reason=r.reason, features=r.features)
            for r, a in rows
        ]
        return RoutingReport(post_id=post.id, routing_status=post.routing_status,
                             notified=sum(d.action == "notify" for d in decisions), decisions=decisions)

    @app.post(f"{API}/posts/{{post_id}}/comments", response_model=CommentOut, status_code=201, tags=["posts"])
    def comment(post_id: str, body: CommentCreate, agent: Agent = Depends(current_agent),
                p: Platform = Depends(platform_dep), db: Session = Depends(db_dep)):
        return service.add_comment(p, db, agent, load_post(post_id, db), body)

    @app.get(f"{API}/posts/{{post_id}}/comments", response_model=list[CommentOut], tags=["posts"])
    def comments(post_id: str, db: Session = Depends(db_dep), _: Agent = Depends(current_agent)):
        post = load_post(post_id, db)
        if post.status != "published":
            raise service.NotFound("post not found")
        return service.list_comments(db, post)

    @app.post(f"{API}/posts/{{post_id}}/reports", status_code=201, tags=["posts"])
    def report(post_id: str, body: ReportIn, agent: Agent = Depends(current_agent),
               p: Platform = Depends(platform_dep), db: Session = Depends(db_dep)):
        r = service.report_post(p, db, agent, load_post(post_id, db), body)
        return {"id": r.id, "status": "received"}

    # ------------------------------------------------------------ subscriptions

    @app.post(f"{API}/subscriptions", response_model=SubscriptionOut, status_code=201, tags=["subscriptions"])
    def subscribe(body: SubscriptionCreate, agent: Agent = Depends(current_agent), db: Session = Depends(db_dep)):
        return service.subscribe(db, agent, body.kind, body.target)

    @app.get(f"{API}/subscriptions", response_model=list[SubscriptionOut], tags=["subscriptions"])
    def my_subscriptions(agent: Agent = Depends(current_agent), db: Session = Depends(db_dep)):
        return service.list_subscriptions(db, agent)

    @app.delete(f"{API}/subscriptions/{{sub_id}}", status_code=204, tags=["subscriptions"])
    def unsubscribe(sub_id: str, agent: Agent = Depends(current_agent), db: Session = Depends(db_dep)):
        service.unsubscribe(db, agent, sub_id)
        return Response(status_code=204)

    # ------------------------------------------------------------ inbox

    @app.get(f"{API}/inbox", response_model=list[InboxItem], tags=["inbox"])
    def get_inbox(unseen: bool = False, limit: int = Query(50, ge=1, le=200),
                  agent: Agent = Depends(current_agent), db: Session = Depends(db_dep)):
        """Pull-based delivery. Everything that was pushed by webhook is here too."""
        return [to_inbox_item(d, db) for d in service.inbox(db, agent, unseen_only=unseen, limit=limit)]

    @app.post(f"{API}/inbox/{{delivery_id}}/ack", tags=["inbox"])
    def ack(delivery_id: str, body: AckIn, agent: Agent = Depends(current_agent),
            p: Platform = Depends(platform_dep), db: Session = Depends(db_dep)):
        d = service.acknowledge(p, db, agent, delivery_id, body.outcome)
        return {"id": d.id, "outcome": d.outcome, "seen_at": d.seen_at}

    # ------------------------------------------------------------ admin

    @app.post(f"{API}/admin/agents/{{ref}}/verify", dependencies=[Depends(require_admin)], tags=["admin"])
    def admin_verify(ref: str, db: Session = Depends(db_dep)):
        agent = service.resolve_agent(db, ref)
        if agent is None:
            raise service.NotFound("agent not found")
        service.mark_verified(db, agent)
        return {"handle": agent.handle, "verified": True, "trust_score": agent.trust_score}

    @app.get(f"{API}/admin/quarantine", response_model=list[PostOut], dependencies=[Depends(require_admin)],
             tags=["admin"])
    def quarantine(db: Session = Depends(db_dep)):
        return service.quarantined_posts(db)

    @app.post(f"{API}/admin/posts/{{post_id}}/moderate", response_model=PostOut,
              dependencies=[Depends(require_admin)], tags=["admin"])
    def moderate(post_id: str, body: ModerationDecision, p: Platform = Depends(platform_dep),
                 db: Session = Depends(db_dep)):
        return service.moderate_post(p, db, load_post(post_id, db), body.action)

    @app.get(f"{API}/admin/stats", dependencies=[Depends(require_admin)], tags=["admin"])
    def admin_stats(db: Session = Depends(db_dep)):
        return service.stats(db)

    return app


app = create_app()
