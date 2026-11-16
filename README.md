# AI Engineering interview problems

Both problems from the brief, built as working, tested prototypes with design write-ups.

| | what it is | design | run it |
|---|---|---|---|
| **[Problem 1: AgentNet](problem1_agent_network/)** | A social network for company AI agents. Agents post offers and updates, the platform routes each post only to the agents that would care, and they reply as comments. | [DESIGN.md](problem1_agent_network/DESIGN.md) | `python scripts/demo.py` |
| **[Problem 2: hwqa](problem2_hw_assistant/)** | A chat assistant that answers questions about a Linux laptop's hardware using read-only, allow-listed system tools. | [DESIGN.md](problem2_hw_assistant/DESIGN.md) | `python -m hwqa` |

Neither needs an API key. Problem 1 routes deterministically, with an optional LLM judge for
borderline cases. Problem 2 has an offline mode and uses a local Ollama model when one is
running. Both can use Claude (`claude-opus-5-5` by default) through the official Anthropic SDK.

## Quick start

From this folder, `run.py` starts either project without changing directory:

```bash
pip install -r problem1_agent_network/requirements.txt pytest
python run.py demo        # Problem 1: the Diwali example end to end
python run.py eval        # Problem 1: routing precision/recall
python run.py api         # Problem 1: API with interactive docs at http://localhost:8000/docs
python run.py hw-demo     # Problem 2: chat against the recorded 16 GB laptop (any OS)
python run.py hw          # Problem 2: chat about this machine (Linux)
python run.py test        # both test suites
```

Or from inside each project:

```bash
# Problem 1
cd problem1_agent_network
pip install -r requirements.txt
python scripts/demo.py               # 10 agents, real HTTP, signed webhooks, comments, feedback loop
python scripts/eval_routing.py       # precision 0.81 / recall 0.97 on 24 labelled posts
python -m pytest -q                  # 65 tests

# Problem 2
cd problem2_hw_assistant
python -m hwqa "Can I add more RAM, and how much?"                                  # on Linux
python -m hwqa --fixture tests/fixtures/laptop_16gb "Can I add more RAM, and how much?"  # anywhere
python -m pytest -q                  # 89 tests
```

## Where the effort went

**Problem 1: relevance routing.** Agents describe what they want to hear about (`interests`)
*and* what they can offer partners (`capabilities`). Each post is split into what it's about and
what it asks for ("…looking for bank and card partners"). A cascade then runs: candidate
retrieval, hard filters, a feature score (topic overlap, complementarity, embedding similarity,
learned affinity, author trust), an LLM judge only for borderline scores, and per-agent push
budgets. Every decision is logged with a reason. Engagement, "not relevant" acks and comments
on un-routed posts feed back into the scores. Delivery uses HMAC-signed webhooks over an outbox
with retries and a circuit breaker, plus a pull inbox. Trust and safety covers domain
verification, prompt-injection and phishing quarantine, trust-weighted reports and SSRF guards.

**Problem 2: tool calling and safety.** The model only sees semantic tools (`get_memory_info`…),
never commands. Each tool runs fixed-argv, allow-listed commands from root-owned system
directories, with no shell, a scrubbed environment and timeouts. Root-only data goes through
`sudo -n` (`--sudo` once, or a pinned sudoers rule), never a prompt mid-chat. Tools return
normalised JSON plus an explicit list of what they couldn't determine and why, and they compute
derived numbers (RAM upgrade headroom) themselves. A grounding check flags any number in an
answer that no tool reported. Identifiers are redacted before data goes to a cloud model, and
the default never leaves the machine.
