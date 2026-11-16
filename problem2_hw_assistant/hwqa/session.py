from __future__ import annotations

from .backends import Backend, Turn
from .grounding import unverified_claims
from .system import System
from .toolbox import ToolBox


class Session:
    """One conversation: a backend, the tools, and the facts gathered so far."""

    def __init__(self, system: System, backend: Backend, redact_identifiers: bool | None = None):
        if redact_identifiers is None:
            redact_identifiers = not backend.local
        self.system = system
        self.backend = backend
        self.toolbox = ToolBox(system, redact_identifiers=redact_identifiers)

    def ask(self, question: str) -> Turn:
        turn = self.backend.ask(question, self.toolbox)
        # Numbers may legitimately come from earlier turns (follow-ups), so check
        # against everything gathered in this session, not just this turn.
        turn.unverified = unverified_claims(turn.answer, self.toolbox.cached_results())
        return turn

    def reset(self) -> None:
        self.backend.reset()
        self.toolbox.clear_cache()
