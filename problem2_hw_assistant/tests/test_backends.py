import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hwqa.backends import BackendError, make_backend
from hwqa.backends.openai_compat import OpenAICompatBackend
from hwqa.prompts import build_system_prompt
from hwqa.session import Session


# ---------------------------------------------------------------- OpenAI-compatible (Ollama, llama.cpp, vLLM)

class ScriptedServer:
    """A tiny chat-completions server that replays scripted assistant messages
    and records every request it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                reply = outer.replies.pop(0)
                data = json.dumps({"choices": [{"message": reply}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def tool_call(name, args, id_="call_1"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


@pytest.fixture
def server():
    servers = []

    def make(replies):
        s = ScriptedServer(replies)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()


def test_openai_compat_tool_loop(laptop, server):
    srv = server([
        tool_call("get_memory_info", {}),
        {"role": "assistant", "content": "You have 16 GB in 1 of 2 slots; you can add up to 32 GB."},
    ])
    backend = OpenAICompatBackend(build_system_prompt(laptop), base_url=srv.url, model="test")
    assert backend.local
    turn = Session(laptop, backend).ask("Can I add more RAM?")
    assert turn.answer.startswith("You have 16 GB") and turn.unverified == []
    assert turn.tool_calls == [("get_memory_info", {})]

    first, second = srv.requests
    assert first["tools"][0]["type"] == "function"
    assert {t["function"]["name"] for t in first["tools"]} >= {"get_memory_info", "get_cpu_info"}
    assert first["messages"][0]["role"] == "system" and "Never infer specs" in first["messages"][0]["content"]
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"])["data"]["upgrade"]["max_total_by_adding_bytes"] == 48 * 1024 ** 3


def test_openai_compat_rejects_bad_tool_calls_but_keeps_going(laptop, server):
    srv = server([
        tool_call("run_command", {"cmd": "rm -rf /"}),
        {"role": "assistant", "content": "I can only use the hardware tools."},
    ])
    backend = OpenAICompatBackend("sys", base_url=srv.url, model="test")
    turn = Session(laptop, backend).ask("delete everything")
    result = json.loads(srv.requests[1]["messages"][-1]["content"])
    assert result["ok"] is False and "unknown tool" in result["error"]
    assert turn.answer == "I can only use the hardware tools."


def test_openai_compat_flags_hallucinated_numbers(laptop, server):
    srv = server([
        tool_call("get_memory_info", {}),
        {"role": "assistant", "content": "You can upgrade to 128 GB."},
    ])
    turn = Session(laptop, OpenAICompatBackend("sys", base_url=srv.url, model="t")).ask("max ram?")
    assert turn.unverified == ["128 GB"]


def test_openai_compat_stops_runaway_loops(laptop, server):
    srv = server([tool_call("get_cpu_info", {}, id_=f"c{i}") for i in range(10)])
    turn = Session(laptop, OpenAICompatBackend("sys", base_url=srv.url, model="t")).ask("cpu?")
    assert "couldn't finish" in turn.answer and len(srv.requests) == 6


def test_unreachable_server_is_a_clean_error(laptop):
    backend = OpenAICompatBackend("sys", base_url="http://127.0.0.1:9/v1", model="t", timeout=1)
    with pytest.raises(BackendError):
        backend.ask("hi", Session(laptop, backend).toolbox)


def test_remote_openai_server_turns_on_redaction(laptop):
    backend = OpenAICompatBackend("sys", base_url="https://api.example.com/v1", model="t")
    assert not backend.local and Session(laptop, backend).toolbox.redact_identifiers


def test_auto_backend_never_picks_the_cloud(laptop, monkeypatch):
    import hwqa.backends as b
    monkeypatch.setattr(b, "ollama_running", lambda *a, **k: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert make_backend("auto", "sys").local


# ---------------------------------------------------------------- Anthropic (Claude)

def _block(type_, **kw):
    return types.SimpleNamespace(type=type_, **kw)


def _resp(stop, *blocks):
    return types.SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Stand-in for the SDK module so the test runs without the package installed."""
    mod = types.ModuleType("anthropic")
    for name in ("APIStatusError", "APIConnectionError", "RateLimitError", "AuthenticationError"):
        setattr(mod, name, type(name, (Exception,), {}))
    mod.Anthropic = lambda: None
    monkeypatch.setitem(sys.modules, "anthropic", mod)

    def make(responses):
        from hwqa.backends.anthropic_backend import AnthropicBackend
        messages = FakeMessages(responses)
        client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=messages))
        return AnthropicBackend("sys", client=client), messages
    return make


def test_anthropic_tool_loop(laptop, fake_anthropic):
    backend, api = fake_anthropic([
        _resp("tool_use", _block("text", text="Checking."),
              _block("tool_use", id="tu_1", name="get_memory_info", input={}),
              _block("tool_use", id="tu_2", name="get_cpu_info", input={})),
        _resp("end_turn", _block("text", text="16 GB installed; a 4-core CPU.")),
    ])
    session = Session(laptop, backend)
    assert session.toolbox.redact_identifiers  # cloud backend
    turn = session.ask("RAM and CPU?")
    assert turn.answer == "16 GB installed; a 4-core CPU."
    assert [c[0] for c in turn.tool_calls] == ["get_memory_info", "get_cpu_info"]

    first, second = api.calls
    assert first["model"] == "claude-opus-5-5"
    assert {t["name"] for t in first["tools"]} >= {"get_memory_info"} and "input_schema" in first["tools"][0]
    # both results go back in ONE user message, paired by id
    results = second["messages"][-1]
    assert results["role"] == "user" and [r["tool_use_id"] for r in results["content"]] == ["tu_1", "tu_2"]
    assert second["messages"][-2]["role"] == "assistant"


def test_anthropic_cut_off_tool_call_is_not_run(laptop, fake_anthropic):
    backend, api = fake_anthropic([
        _resp("max_tokens", _block("tool_use", id="tu_1", name="get_disk_health", input={"device": "nv"})),
    ])
    session = Session(laptop, backend)
    turn = session.ask("disk health?")
    assert turn.tool_calls == [] and "cut off" in turn.answer
    # history stays valid: the dangling tool_use is closed with an error result
    last = backend.messages[-1]
    assert last["role"] == "user" and last["content"][0]["is_error"] is True


def test_anthropic_refusal(laptop, fake_anthropic):
    backend, _ = fake_anthropic([_resp("refusal")])
    assert "declined" in Session(laptop, backend).ask("?").answer


def test_anthropic_follow_up_keeps_history(laptop, fake_anthropic):
    backend, api = fake_anthropic([
        _resp("end_turn", _block("text", text="16 GB.")),
        _resp("end_turn", _block("text", text="An Iris Xe and an MX450.")),
    ])
    s = Session(laptop, backend)
    s.ask("How much RAM?")
    s.ask("and what about the GPU?")
    roles = [m["role"] for m in api.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
