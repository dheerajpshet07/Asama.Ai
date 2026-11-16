from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentnet import service  # noqa: E402
from agentnet.api import create_app  # noqa: E402
from agentnet.config import Settings  # noqa: E402
from agentnet.runtime import Platform  # noqa: E402
from agentnet.schemas import AgentRegister  # noqa: E402
from demo_agents.profiles import AGENTS  # noqa: E402


class RecordingTransport:
    """Stands in for HTTP: records webhook calls and answers with a scripted status."""

    def __init__(self, status: int = 200):
        self.status = status
        self.calls: list[tuple[str, bytes, dict]] = []

    def __call__(self, url, body, headers):
        self.calls.append((url, body, headers))
        if isinstance(self.status, Exception):
            raise self.status
        return self.status


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system, prompt, max_tokens=2048):
        self.prompts.append((system, prompt))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def make_settings(**overrides) -> Settings:
    base = dict(database_url="sqlite://", inline_routing=True, admin_token="test-admin")
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def transport():
    return RecordingTransport()


@pytest.fixture
def platform(transport):
    p = Platform(make_settings(), transport=transport, llm=None)
    yield p
    p.stop()


@pytest.fixture
def agents(platform):
    """All ten demo agents, registered and verified. Returns {handle: (agent, api_key, secret)}."""
    out = {}
    with platform.Session() as db:
        for a in AGENTS:
            agent, key, secret = service.register_agent(
                platform, db, AgentRegister(**a.registration(f"https://hooks.example.com/{a.handle}"))
            )
            service.mark_verified(db, agent)
            out[a.handle] = (agent, key, secret)
    return out


@pytest.fixture
def api(platform):
    app = create_app(platform=platform)
    with TestClient(app) as client:
        yield client


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}
