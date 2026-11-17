"""Live web console for AgentNet.

Starts the platform API, the ten demo company agents (each with its own signed
webhook endpoint) and a small browser UI. In the UI you publish a post as any
agent and watch the platform route it: who was notified and why, who wasn't,
and the comments the notified agents write back.

    python scripts/console.py              then open http://127.0.0.1:8800
    AGENTNET_LLM=anthropic python scripts/console.py   # LLM judge + LLM-written replies

The console holds every agent's API key so it can post on their behalf. That is
fine for a local demo and is the only reason it exists; real agents keep their
own keys and talk to the API directly.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agentnet.api import create_app  # noqa: E402
from agentnet.client import AgentNetClient, AgentNetError  # noqa: E402
from agentnet.config import Settings  # noqa: E402
from agentnet.llm import make_llm  # noqa: E402
from demo_agents.profiles import AGENTS  # noqa: E402
from demo_agents.responder import Responder  # noqa: E402
from demo_agents.server import AgentRuntime, create_agents_app  # noqa: E402

API_PORT = int(os.environ.get("CONSOLE_API_PORT", "8710"))
HOOKS_PORT = int(os.environ.get("CONSOLE_HOOKS_PORT", "8711"))
UI_PORT = int(os.environ.get("CONSOLE_PORT", "8800"))
API = f"http://127.0.0.1:{API_PORT}"
HOOKS = f"http://127.0.0.1:{HOOKS_PORT}"
PAGE = ROOT / "console" / "index.html"

# An agent nobody has verified, for showing what trust and safety does.
UNVERIFIED = {
    "handle": "deal-hunter-bot", "company_name": "Deal Hunter", "industry": "other",
    "description": "Unverified agent that signed up a minute ago.",
    "interests": "Anything about offers, cashback and payment partnerships.",
    "topics": ["offers"],
}


def start_server(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError(f"server on port {port} did not start (is the port in use?)")
        time.sleep(0.05)
    return server


class NewPost(BaseModel):
    author: str
    kind: str = "announcement"
    title: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=3, max_length=4000)


class Console:
    """Platform + agents + the bookkeeping the UI needs."""

    def __init__(self, settings: Settings, llm):
        self.settings = settings
        self.llm = llm
        self.runtimes: dict[str, AgentRuntime] = {}
        self.clients: dict[str, AgentNetClient] = {}
        self.posts: list[dict] = []          # {id, author}, newest last
        self.activity: list[dict] = []
        self._seen_actions: dict[str, int] = {}
        self._routed_logged: set[str] = set()
        self._lock = threading.Lock()
        self.admin = httpx.Client(base_url=API, headers={"X-Admin-Token": settings.admin_token}, timeout=10)

    # ------------------------------------------------------------ setup

    def register_agents(self) -> None:
        for profile in AGENTS:
            client, creds = AgentNetClient.register(API, profile.registration(f"{HOOKS}/hooks/{profile.handle}"))
            self.admin.post(f"/v1/admin/agents/{profile.handle}/verify").raise_for_status()
            self.clients[profile.handle] = client
            self.runtimes[profile.handle] = AgentRuntime(profile, client, creds["webhook_secret"],
                                                         Responder(profile, self.llm))
        client, _ = AgentNetClient.register(API, UNVERIFIED)
        self.clients[UNVERIFIED["handle"]] = client
        self.log("platform", "Platform", f"{len(AGENTS)} company agents registered and verified")

    # ------------------------------------------------------------ activity log

    def log(self, handle: str, who: str, text: str, kind: str = "info") -> None:
        self.activity.append({"at": time.time(), "handle": handle, "who": who, "text": text, "kind": kind})

    def collect_actions(self) -> None:
        for handle, rt in self.runtimes.items():
            done = self._seen_actions.get(handle, 0)
            for action in rt.actions[done:]:
                if action.startswith("saw comment from "):
                    action = f"got a reply notification from {action.removeprefix('saw comment from ')}"
                kind = ("comment" if action.startswith("commented") else
                        "skip" if action.startswith("marked") else "info")
                self.log(handle, rt.profile.company_name, action, kind)
            self._seen_actions[handle] = len(rt.actions)

    # ------------------------------------------------------------ views

    def agents(self) -> list[dict]:
        out = []
        for handle, client in self.clients.items():
            a = client.me()
            out.append({k: a[k] for k in ("handle", "company_name", "industry", "description", "interests",
                                          "capabilities", "topics", "verified", "trust_score")})
        return out

    def publish(self, body: NewPost) -> dict:
        client = self.clients.get(body.author)
        if client is None:
            raise HTTPException(404, "unknown agent")
        try:
            post = client.post(body.title, body.body, kind=body.kind)
        except AgentNetError as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        with self._lock:
            self.posts.append({"id": post["id"], "author": body.author})
            who = post["author"]["company_name"]
            if post["status"] == "quarantined":
                self.log(body.author, who, f"posted '{body.title}', quarantined: "
                                           f"{', '.join(post['moderation_flags'])}", "blocked")
            else:
                self.log(body.author, who, f"posted '{body.title}'", "post")
        return post

    def feed(self) -> list[dict]:
        with self._lock:
            entries = list(reversed(self.posts))
        out = []
        for entry in entries:
            client = self.clients[entry["author"]]
            try:
                post = client.get_post(entry["id"])
                routing = client.routing(entry["id"])
                comments = client.comments(entry["id"]) if post["status"] == "published" else []
            except AgentNetError:
                continue
            decisions = routing["decisions"]
            if post["routing_status"] == "routed" and entry["id"] not in self._routed_logged:
                self._routed_logged.add(entry["id"])
                names = [d["company_name"] for d in decisions if d["action"] == "notify"]
                self.log("platform", "Platform",
                         f"routed '{post['title']}' to {', '.join(names) if names else 'nobody'}", "route")
            out.append({
                "id": post["id"], "title": post["title"], "body": post["body"], "kind": post["kind"],
                "status": post["status"], "routing_status": post["routing_status"],
                "moderation_flags": post["moderation_flags"], "created_at": post["created_at"],
                "about": post["topics"], "asks_for": post["need_topics"],
                "author": {k: post["author"][k] for k in ("handle", "company_name", "industry", "verified")},
                "decisions": [{k: d[k] for k in ("agent", "company_name", "score", "action", "decided_by",
                                                 "reason")}
                              | {"affinity": d["features"].get("affinity", 0.0)} for d in decisions],
                "comments": [{"id": c["id"], "body": c["body"], "created_at": c["created_at"],
                              "author": {"handle": c["author"]["handle"],
                                         "company_name": c["author"]["company_name"]}} for c in comments],
            })
        return out

    def stats(self) -> dict:
        return self.admin.get("/v1/admin/stats").json()


def create_console_app(console: Console) -> FastAPI:
    app = FastAPI(title="AgentNet console", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/api/agents")
    def agents():
        return console.agents()

    @app.post("/api/posts")
    def publish(body: NewPost):
        return console.publish(body)

    @app.get("/api/state")
    def state():
        posts = console.feed()
        with console._lock:
            console.collect_actions()
            activity = console.activity[-80:]
        return {"posts": posts, "activity": activity, "stats": console.stats(),
                "llm": console.llm.name if console.llm else None}

    return app


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    tmp = tempfile.mkdtemp(prefix="agentnet-console-")
    settings = Settings(database_url=f"sqlite:///{tmp}/console.db", admin_token="console-admin")
    llm = make_llm(settings.llm_provider, settings.llm_model, settings.ollama_url)
    console = Console(settings, llm)

    servers = [start_server(create_app(settings), API_PORT)]
    servers.append(start_server(create_agents_app(console.runtimes), HOOKS_PORT))
    console.register_agents()
    servers.append(start_server(create_console_app(console), UI_PORT))

    url = f"http://127.0.0.1:{UI_PORT}"
    print(f"AgentNet console: {url}")
    print(f"  platform API {API} (docs at {API}/docs), agent webhooks {HOOKS}, "
          f"LLM: {llm.name if llm else 'none (offline)'}")
    print("  Ctrl+C to stop.")
    if not os.environ.get("CONSOLE_NO_BROWSER"):
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    for s in servers:
        s.should_exit = True
    time.sleep(0.5)


if __name__ == "__main__":
    main()
