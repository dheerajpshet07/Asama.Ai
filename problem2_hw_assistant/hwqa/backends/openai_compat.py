"""Any server that speaks the OpenAI chat-completions API with function
calling: Ollama (the default, fully local), llama.cpp's server, vLLM, LM Studio,
or OpenAI itself. Uses only the standard library."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from ..toolbox import ToolBox
from .base import MAX_TOOL_ROUNDS, Backend, BackendError, Turn

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
MAX_HISTORY_TURNS = 8  # small local models have small context windows


class OpenAICompatBackend(Backend):
    def __init__(self, system_prompt: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.system_prompt = system_prompt
        host = urlparse(self.base_url).hostname or ""
        self.local = host in ("localhost", "127.0.0.1", "::1")
        self.name = f"{model} @ {self.base_url}" + ("" if self.local else " (remote)")
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages = []

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise BackendError(f"model server returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise BackendError(f"can't reach the model server at {self.base_url}: {exc}") from exc

    def _trim(self) -> None:
        user_idx = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_idx) > MAX_HISTORY_TURNS:
            self.messages = self.messages[user_idx[-MAX_HISTORY_TURNS]:]

    def ask(self, question: str, toolbox: ToolBox) -> Turn:
        self.messages.append({"role": "user", "content": question})
        self._trim()
        tools = [{"type": "function", "function": spec} for spec in toolbox.specs()]
        calls = []
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self._post({
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_prompt}, *self.messages],
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0,
            })
            try:
                msg = resp["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise BackendError(f"unexpected response from model server: {str(resp)[:200]}") from exc

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                text = (msg.get("content") or "").strip()
                self.messages.append({"role": "assistant", "content": text})
                return Turn(text, calls)

            self.messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw = fn.get("arguments") or {}
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    result = toolbox.call(name, args)
                except json.JSONDecodeError:
                    args, result = {}, {"tool": name, "ok": False, "error": "arguments were not valid JSON"}
                calls.append((name, args))
                self.messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{i}",
                                      "name": name, "content": json.dumps(result)})

        text = "I couldn't finish gathering what I needed for that question. Try asking about one part at a time."
        self.messages.append({"role": "assistant", "content": text})
        return Turn(text, calls)
