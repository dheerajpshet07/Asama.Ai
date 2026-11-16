"""Wires the pieces together and runs the background workers.

Publishing a post only writes it and enqueues its id; routing and webhook
delivery happen on worker threads, so POST /v1/posts stays fast no matter how
many agents there are. In production the in-process queue becomes a durable
stream (Redis Streams / Kafka) with a pool of routing workers, and posts left
in routing_status=pending are re-enqueued on start-up, as here.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from sqlalchemy import select

from . import service
from .config import Settings
from .db import Base, make_engine, make_session_factory
from .delivery import Dispatcher, Transport
from .embeddings import Embedder, HashingEmbedder
from .feedback import effective_vector
from .index import VectorIndex
from .judge import RelevanceJudge
from .llm import LLM, make_llm
from .models import Agent, Post
from .routing import Router

log = logging.getLogger(__name__)

_NO_LLM = object()


class Platform:
    def __init__(self, settings: Settings | None = None, *, transport: Transport | None = None,
                 embedder: Embedder | None = None, llm: LLM | None | object = _NO_LLM):
        self.settings = settings or Settings()
        self.engine = make_engine(self.settings.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = make_session_factory(self.engine)

        self.embedder = embedder or HashingEmbedder()
        self.index = VectorIndex(self.embedder.dim)
        if llm is _NO_LLM:
            llm = make_llm(self.settings.llm_provider, self.settings.llm_model, self.settings.ollama_url)
        self.llm = llm
        self.judge = RelevanceJudge(llm) if llm is not None else None
        self.router = Router(self.settings, self.index, self.judge)
        self.dispatcher = Dispatcher(self.Session, self.settings, transport)

        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._load_index()

    # ------------------------------------------------------------------ index

    def _load_index(self) -> None:
        with self.Session() as db:
            for agent in db.scalars(select(Agent).where(Agent.status == "active")):
                vec = effective_vector(agent)
                if vec is None or vec.shape != (self.embedder.dim,):
                    # Embedder changed since this agent registered: re-embed.
                    service.refresh_profile(self, agent)
                    vec = effective_vector(agent)
                self.index.upsert(agent.id, vec)
            db.commit()

    def reindex_agent(self, agent: Agent) -> None:
        if agent.status == "active":
            self.index.upsert(agent.id, effective_vector(agent))
        else:
            self.index.remove(agent.id)

    # ------------------------------------------------------------------ workers

    @property
    def running(self) -> bool:
        return bool(self._threads)

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        with self.Session() as db:
            for post_id in db.scalars(select(Post.id).where(Post.status == "published",
                                                            Post.routing_status == "pending")):
                self._queue.put(post_id)
        for target, name in ((self._routing_loop, "router"), (self._delivery_loop, "dispatcher")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self.dispatcher.poke()
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()

    def submit_routing(self, post_id: str) -> None:
        if self.settings.inline_routing or not self.running:
            service.route_post(self, post_id)
            self.flush_deliveries()
        else:
            self._queue.put(post_id)

    def flush_deliveries(self) -> None:
        """Push due webhooks now (in-line mode) or wake the dispatcher thread."""
        if self.running:
            self.dispatcher.poke()
        else:
            self.dispatcher.run_once()

    def _routing_loop(self) -> None:
        last_sweep = 0.0
        while not self._stop.is_set():
            try:
                post_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                post_id = None
            if post_id:
                try:
                    service.route_post(self, post_id)
                    self.dispatcher.poke()
                except Exception:
                    log.exception("routing failed for post %s", post_id)
            if time.monotonic() - last_sweep > 30:
                last_sweep = time.monotonic()
                try:
                    service.sweep_ignored(self)
                except Exception:
                    log.exception("ignored-notification sweep failed")

    def _delivery_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.dispatcher.run_once()
            except Exception:
                log.exception("dispatcher error")
            self.dispatcher.wait(1.0)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Test/demo helper: block until queued routing work has drained."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.Session() as db:
                pending = db.scalar(select(Post.id).where(Post.status == "published",
                                                          Post.routing_status == "pending").limit(1))
            if pending is None and self._queue.empty():
                return True
            time.sleep(0.05)
        return False
