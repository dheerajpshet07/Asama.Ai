"""Claude through the official Anthropic SDK (pip install anthropic).

This sends tool results to Anthropic's API, so the session redacts serial
numbers, UUIDs and MAC addresses first (see ToolBox.redact_identifiers)."""
from __future__ import annotations

import json
from typing import Optional

from ..toolbox import ToolBox
from .base import MAX_TOOL_ROUNDS, Backend, BackendError, Turn

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicBackend(Backend):
    local = False

    def __init__(self, system_prompt: str, model: Optional[str] = None, client=None):
        try:
            import anthropic
        except ImportError as exc:
            raise BackendError("the anthropic package isn't installed: pip install anthropic") from exc
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model or DEFAULT_MODEL
        self.name = f"{self.model} (Anthropic API)"
        self.system_prompt = system_prompt
        self.messages: list = []

    def reset(self) -> None:
        self.messages = []

    def _create(self, tools: list[dict]):
        anthropic = self._anthropic
        try:
            return self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system_prompt,
                tools=tools,
                messages=self.messages,
                output_config={"effort": "medium"},
                # If a safety classifier declines, the API retries on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise BackendError("Anthropic API key missing or invalid (set ANTHROPIC_API_KEY)") from exc
        except anthropic.RateLimitError as exc:
            raise BackendError("rate limited by the Anthropic API; try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError("can't reach the Anthropic API") from exc

    def _close_tool_uses(self, tool_uses, reason: str) -> None:
        """Keep the history valid when a turn ends with tool calls we won't run."""
        if tool_uses:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": reason, "is_error": True} for b in tool_uses
            ]})

    def ask(self, question: str, toolbox: ToolBox) -> Turn:
        tools = [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
                 for s in toolbox.specs()]
        self.messages.append({"role": "user", "content": question})
        calls = []
        for _ in range(MAX_TOOL_ROUNDS):
            resp = self._create(tools)
            # Append the full content (incl. thinking blocks) unchanged, as the API expects.
            self.messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            text = "".join(b.text for b in resp.content if b.type == "text").strip()

            if resp.stop_reason == "refusal":
                self._close_tool_uses(tool_uses, "not run: the request was declined")
                return Turn("The model declined to answer that question.", calls)
            if resp.stop_reason == "max_tokens":
                # A tool call cut off mid-way may have partial arguments: never run it.
                self._close_tool_uses(tool_uses, "not run: the response was cut off")
                return Turn(text or "The answer was cut off. Please ask again more narrowly.", calls)
            if resp.stop_reason != "tool_use" or not tool_uses:
                return Turn(text, calls)

            results = []
            for b in tool_uses:
                result = toolbox.call(b.name, b.input)
                calls.append((b.name, b.input))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(result),
                                "is_error": not result.get("ok", False)})
            # All results of one turn go back in a single message.
            self.messages.append({"role": "user", "content": results})

        return Turn("I couldn't finish gathering what I needed for that question. "
                    "Try asking about one part at a time.", calls)
