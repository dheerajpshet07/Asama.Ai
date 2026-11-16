from __future__ import annotations

from dataclasses import dataclass, field

from ..toolbox import ToolBox

MAX_TOOL_ROUNDS = 6  # model <-> tools round trips per question before we stop


class BackendError(RuntimeError):
    pass


@dataclass
class Turn:
    answer: str
    tool_calls: list = field(default_factory=list)   # [(tool name, args)]
    unverified: list = field(default_factory=list)   # filled in by the session's grounding check


class Backend:
    name = "backend"
    local = True  # True when no data leaves the machine

    def ask(self, question: str, toolbox: ToolBox) -> Turn:
        raise NotImplementedError

    def reset(self) -> None:
        pass
