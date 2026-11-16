"""The boundary between the model and the tools.

Every tool call from any backend goes through ToolBox.call, which:
  * rejects tools that don't exist or aren't available on this machine
  * validates arguments against the tool's JSON schema (types, enums, patterns,
    no extra keys), so a model can't smuggle anything in
  * serves repeat questions from a per-session cache (static facts for the whole
    session, live values like battery % for a few seconds)
  * redacts identifiers (serials, MACs, UUIDs) before results reach a model
    that runs off the machine
  * keeps a log of calls for /tools and for the grounding check
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .system import System
from .tools import TOOLS, Tool

_IDENTIFIER_KEYS = re.compile(r"serial|uuid|mac_address|asset_tag", re.I)
_MAC = re.compile(r"\b[0-9a-f]{2}(:[0-9a-f]{2}){5}\b", re.I)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if _IDENTIFIER_KEYS.search(k) and v else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _MAC.sub("[redacted MAC]", value)
    return value


class ArgumentError(ValueError):
    pass


def validate(schema: dict, args: Any) -> dict:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ArgumentError("arguments must be a JSON object")
    props = schema.get("properties", {})
    extra = set(args) - set(props)
    if extra:
        raise ArgumentError(f"unexpected argument(s): {', '.join(sorted(extra))}")
    for name in schema.get("required", []):
        if name not in args:
            raise ArgumentError(f"missing required argument: {name}")
    for name, value in args.items():
        spec = props[name]
        t = spec.get("type")
        if t == "boolean" and not isinstance(value, bool):
            raise ArgumentError(f"{name} must be true or false")
        if t == "string":
            if not isinstance(value, str):
                raise ArgumentError(f"{name} must be a string")
            if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
                raise ArgumentError(f"{name} has an invalid value: {value!r}")
        if "enum" in spec and value not in spec["enum"]:
            raise ArgumentError(f"{name} must be one of {spec['enum']}")
    return args


@dataclass
class CallRecord:
    tool: str
    args: dict
    ok: bool
    cached: bool
    at: float = field(default_factory=time.time)


class ToolBox:
    def __init__(self, system: System, redact_identifiers: bool = False):
        self.system = system
        self.redact_identifiers = redact_identifiers
        self.tools: dict[str, Tool] = {t.name: t for t in TOOLS if t.available(system)}
        self._cache: dict[tuple[str, str], tuple[float, dict]] = {}
        self.calls: list[CallRecord] = []

    def specs(self) -> list[dict]:
        """Tool definitions in a neutral shape; backends convert to their API format."""
        return [{"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self.tools.values()]

    def call(self, name: str, args: Optional[dict] = None) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            self.calls.append(CallRecord(name, args or {}, False, False))
            return {"tool": name, "ok": False, "error": f"unknown tool {name!r}",
                    "available_tools": sorted(self.tools)}
        try:
            args = validate(tool.parameters, args)
        except ArgumentError as exc:
            self.calls.append(CallRecord(name, args if isinstance(args, dict) else {}, False, False))
            return {"tool": name, "ok": False, "error": f"invalid arguments: {exc}"}

        key = (name, json.dumps(args, sort_keys=True))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < tool.ttl:
            self.calls.append(CallRecord(name, args, True, True))
            return dict(hit[1], cached=True)

        try:
            r = tool.fn(self.system, **args)
            result = {"tool": name, "ok": True, "data": r.data, "sources": r.sources,
                      "unavailable": r.unavailable, "notes": r.notes}
        except Exception as exc:  # a parser choking on unexpected output must not kill the chat
            result = {"tool": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if self.redact_identifiers:
            result = redact(result)
        self._cache[key] = (time.monotonic(), result)
        self.calls.append(CallRecord(name, args, result["ok"], False))
        return result

    def cached_results(self) -> list[dict]:
        """Everything gathered this session (used by the grounding check)."""
        return [v for _, v in self._cache.values()]

    def clear_cache(self) -> None:
        self._cache.clear()
