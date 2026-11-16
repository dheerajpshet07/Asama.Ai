"""Run either project from this folder, without cd-ing into it first.

    python run.py demo            Problem 1: the Diwali demo (10 agents, routing, comments)
    python run.py eval            Problem 1: routing precision/recall on the labelled posts
    python run.py api             Problem 1: start the API, docs at http://localhost:8000/docs
    python run.py hw              Problem 2: chat about this machine's hardware (Linux)
    python run.py hw-demo         Problem 2: chat against the recorded 16 GB laptop (any OS)
    python run.py test            run both test suites

Anything after the command is passed through, e.g.
    python run.py hw-demo "Can I add more RAM, and how much?"
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P1 = ROOT / "problem1_agent_network"
P2 = ROOT / "problem2_hw_assistant"
PY = sys.executable

COMMANDS = {
    "demo": [(P1, [PY, "scripts/demo.py"])],
    "eval": [(P1, [PY, "scripts/eval_routing.py"])],
    "api": [(P1, [PY, "-m", "uvicorn", "agentnet.api:app", "--port", "8000"])],
    "hw": [(P2, [PY, "-m", "hwqa"])],
    "hw-demo": [(P2, [PY, "-m", "hwqa", "--fixture", "tests/fixtures/laptop_16gb"])],
    "test": [(P1, [PY, "-m", "pytest", "-q"]), (P2, [PY, "-m", "pytest", "-q"])],
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__.strip())
        return 0 if len(sys.argv) < 2 else 2
    name, extra = sys.argv[1], sys.argv[2:]
    if name == "api":
        print("Starting the API. Open http://localhost:8000/docs in a browser; Ctrl+C to stop.")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    status = 0
    for cwd, cmd in COMMANDS[name]:
        if name == "test":
            print(f"\n== {cwd.name} ==", flush=True)
        try:
            status = subprocess.call(cmd + extra, cwd=cwd, env=env) or status
        except KeyboardInterrupt:
            return 130
    return status


if __name__ == "__main__":
    sys.exit(main())
