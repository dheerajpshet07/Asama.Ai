"""Command-line chat.

    hwqa                             interactive chat
    hwqa "is my SSD NVMe?"           one question, then exit
    hwqa --sudo                      allow root-only details (asks for your password once)
    hwqa --backend anthropic         use Claude (identifiers are redacted before sending)
    hwqa --fixture tests/fixtures/laptop_16gb   replay a recorded machine
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .backends import BackendError, make_backend
from .prompts import build_system_prompt
from .session import Session
from .system import COMMANDS, FixtureSystem, LiveSystem, resolve_binary
from .toolbox import ToolBox

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def dim(text: str) -> str:
    return _c("2", text)


def sudoers_rule() -> str:
    dmi = resolve_binary("dmidecode") or "/usr/sbin/dmidecode"
    smart = resolve_binary("smartctl") or "/usr/sbin/smartctl"
    return f"""# /etc/sudoers.d/hwqa
# Lets members of the "hwqa" group run these exact read-only commands as root without a password.
# Install:  sudo groupadd -f hwqa && sudo usermod -aG hwqa "$USER"
#           sudo visudo -f /etc/sudoers.d/hwqa      (paste this, save, log out and back in)
#
# Each line pins the full argument list. Avoid "*" in sudoers arguments: it also matches spaces,
# so "/dev/sd*" would allow "/dev/sda -s on" (which changes drive settings). The [..] classes below
# match exactly one character.
%hwqa ALL=(root) NOPASSWD: {dmi} -t memory, {dmi} -t system
%hwqa ALL=(root) NOPASSWD: {smart} -H -A -j /dev/nvme[0-9]n[0-9], {smart} -H -A -j /dev/sd[a-z]
"""


HELP = """Ask anything about this machine's hardware. Commands:
  /tools   tools available on this machine and what was called
  /facts   everything gathered so far, as JSON
  /audit   system commands that were run
  /clear   forget the conversation and cached facts
  /quit    exit"""


def print_turn(turn, verbose: bool) -> None:
    if verbose and turn.tool_calls:
        for name, args in turn.tool_calls:
            print(dim(f"  [tool] {name}({json.dumps(args) if args else ''})"))
    print(turn.answer)
    if turn.unverified:
        print(_c("33", f"  (couldn't verify against system data: {', '.join(turn.unverified)})"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hwqa", description="Ask questions about this Linux machine's hardware.")
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    ap.add_argument("--backend", default=os.environ.get("HWQA_BACKEND", "auto"),
                    choices=["auto", "offline", "ollama", "openai", "anthropic"])
    ap.add_argument("--model", help="model name for the backend")
    ap.add_argument("--base-url", help="server URL for the ollama/openai backends")
    ap.add_argument("--sudo", action="store_true",
                    help="ask for your sudo password once so root-only details (RAM slots, serial, SMART) work")
    ap.add_argument("--no-sudo", action="store_true", help="never try sudo")
    ap.add_argument("--fixture", metavar="DIR", help="replay a recorded machine instead of this one")
    ap.add_argument("--show-identifiers", action="store_true",
                    help="don't redact serial numbers / MAC addresses, even for a cloud backend")
    ap.add_argument("--dump", action="store_true", help="print every tool's output as JSON and exit")
    ap.add_argument("--print-sudoers", action="store_true", help="print a narrow sudoers rule and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    ap.add_argument("--version", action="version", version=f"hwqa {__version__}")
    args = ap.parse_args(argv)

    if args.print_sudoers:
        print(sudoers_rule())
        return 0

    if args.fixture:
        system = FixtureSystem(args.fixture)
    else:
        if not sys.platform.startswith("linux"):
            print("hwqa reads Linux's /proc and /sys. On this OS, try a recorded machine: --fixture DIR",
                  file=sys.stderr)
            return 2
        system = LiveSystem(allow_sudo=not args.no_sudo)
        if args.sudo:
            system.prime_sudo()

    if args.dump:
        box = ToolBox(system)
        out = {name: box.call(name) for name, t in box.tools.items() if not t.parameters.get("required")}
        print(json.dumps(out, indent=2, default=str))
        return 0

    try:
        backend = make_backend(args.backend, build_system_prompt(system), args.model, args.base_url)
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    session = Session(system, backend, redact_identifiers=False if args.show_identifiers else None)

    def ask(q: str) -> None:
        try:
            print_turn(session.ask(q), args.verbose)
        except BackendError as exc:
            print(_c("31", f"error: {exc}"))

    if args.question:
        ask(" ".join(args.question))
        return 0

    try:
        import readline  # noqa: F401  (line editing and history, where available)
    except ImportError:
        pass

    where = "on this machine" if backend.local else "remote; serials and MACs are redacted before sending"
    print(f"hwqa {__version__}  |  model: {backend.name} ({where})")
    print(dim(f"root access: {system.privilege}" + ("  (use --sudo for RAM slot details)"
                                                    if system.privilege in ("none", "unknown") else "")))
    print(dim("Type a question, or /help.\n"))
    while True:
        try:
            q = input(_c("1", "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            return 0
        if q == "/help":
            print(HELP)
        elif q == "/tools":
            for name in session.toolbox.tools:
                print(f"  {name}")
            calls = session.toolbox.calls
            if calls:
                print(dim("called: " + ", ".join(f"{c.tool}{' (cached)' if c.cached else ''}" for c in calls)))
        elif q == "/facts":
            print(json.dumps(session.toolbox.cached_results(), indent=2, default=str))
        elif q == "/audit":
            for e in system.audit:
                print(f"  {e.outcome:<34} {e.what}" + (f"  ({e.ms} ms)" if e.ms else ""))
            if not system.audit:
                print("  nothing run yet")
            print(dim("  allow-listed: " + ", ".join(sorted({s.binary for s in COMMANDS.values()}))))
        elif q == "/clear":
            session.reset()
            print("Cleared.")
        else:
            ask(q)
        print()


if __name__ == "__main__":
    sys.exit(main())
