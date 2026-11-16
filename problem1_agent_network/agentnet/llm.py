"""Minimal LLM clients used by the relevance judge and by the demo agents.

Two providers, picked with AGENTNET_LLM:
  anthropic  - Claude through the official SDK (needs ANTHROPIC_API_KEY)
  ollama     - a local model through Ollama's HTTP API
Anything else (the default, "none") means no LLM: routing falls back to the
score threshold and demo agents use their template playbooks.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError("pip install anthropic to use AGENTNET_LLM=anthropic") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        anthropic = self._anthropic
        try:
            # Server-side fallback: if the request is declined by a safety
            # classifier, the API retries it on a fallback model in the same call.
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by Anthropic API") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach Anthropic API") from exc

        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request")
        return "".join(block.text for block in resp.content if block.type == "text").strip()


class OllamaLLM:
    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434"):
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.base_url = base_url.rstrip("/")
        self.name = f"ollama:{self.model}"
        self._client = httpx.Client(timeout=60)

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0.2, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return resp.json()["message"]["content"].strip()


def make_llm(provider: str, model: str | None = None, ollama_url: str = "http://localhost:11434") -> LLM | None:
    provider = (provider or "none").lower()
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "ollama":
        return OllamaLLM(model, base_url=ollama_url)
    return None


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_json(text: str):
    """Pull the first JSON object/array out of a model reply (models like to
    wrap JSON in prose or code fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise LLMError(f"no JSON in model reply: {text[:200]!r}")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"bad JSON in model reply: {exc}") from exc
