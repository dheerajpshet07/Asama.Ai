"""Backend selection.

  offline    keyword routing + templates, no model at all
  ollama     local model through Ollama's OpenAI-compatible API (default when it's running)
  openai     any OpenAI-compatible server (--base-url), e.g. llama.cpp, vLLM, LM Studio
  anthropic  Claude via the Anthropic API (data leaves the machine; identifiers are redacted)

"auto" picks Ollama if it is running locally, otherwise offline. It never
picks a cloud backend on its own: sending hardware data off the machine is the
user's decision.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Optional

from .base import Backend, BackendError, Turn
from .offline import OfflineBackend

__all__ = ["Backend", "BackendError", "Turn", "make_backend", "ollama_running"]


def ollama_running(base: str = "http://localhost:11434", timeout: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except OSError:
        return False


def make_backend(kind: str, system_prompt: str, model: Optional[str] = None,
                 base_url: Optional[str] = None) -> Backend:
    kind = (kind or "auto").lower()
    if kind == "auto":
        kind = "ollama" if ollama_running() else "offline"
    if kind == "offline":
        return OfflineBackend()
    if kind in ("ollama", "openai"):
        from .openai_compat import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatBackend
        if kind == "openai" and not base_url:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAICompatBackend(
            system_prompt,
            base_url=base_url or os.environ.get("HWQA_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.environ.get("HWQA_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("OPENAI_API_KEY") if kind == "openai" else None,
        )
    if kind == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(system_prompt, model=model or os.environ.get("HWQA_MODEL"))
    raise BackendError(f"unknown backend {kind!r} (choose offline, ollama, openai, anthropic or auto)")
