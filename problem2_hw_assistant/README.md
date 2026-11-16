# Problem 2: hwqa, a hardware Q&A assistant for your Linux laptop

Ask about your machine in plain language. The assistant gathers the facts with
read-only, allow-listed system tools and answers only from what they returned.

```
$ hwqa "Can I add more RAM, and how much?"
You have 16 GB of RAM in 1 of 2 slots (1 × 16 GB DDR4 SODIMM at 3200 MT/s). The board supports
up to 64 GB, so you can add a stick of up to 32 GB in the free slot (48 GB total). To reach 64 GB
you'd have to replace the existing module as well. Buy a DDR4 SODIMM module rated 3200 MT/s or
faster; a stick the same size as the existing one keeps dual-channel symmetric. (That maximum
comes from the firmware's DMI table, which is occasionally conservative; confirm with the spec
sheet before buying.)

> and what about the GPU?
You have 2 GPUs: Intel Iris Xe Graphics (integrated, driver i915); NVIDIA GeForce MX450
(discrete, 2 GB VRAM, driver nvidia).
```

When it can't know something, it says so and tells you why:

```
> Can I upgrade my RAM?        (no root)
The OS sees 14.9 GB of RAM, and 11.4 GB is free right now. I can't determine the installed
modules, slot count and maximum supported memory: dmidecode needs root to read this. To fix
that, start hwqa with --sudo (asks for your password once), or install the read-only sudoers
rule printed by `hwqa --print-sudoers`.
```

**Design write-up: [DESIGN.md](DESIGN.md)**. It covers the tool-calling design, the safety
model, root handling, accuracy, portability and privacy.

## Install and run

Python 3.9+, standard library only.

```bash
pip install .                 # or just: python -m hwqa ...
hwqa                          # interactive chat
hwqa "is my SSD NVMe?"        # one question
hwqa --sudo                   # root-only details (RAM slots, serial, SMART); password asked once
hwqa -v                       # show which tools were called
hwqa --dump                   # every tool's JSON output, for debugging
hwqa --print-sudoers          # a narrow passwordless rule, if you prefer that to --sudo
```

In the chat: `/tools`, `/facts` (everything gathered, as JSON), `/audit` (every system
command that ran), `/clear`, `/quit`.

### Choosing the model

| `--backend` | what | data leaves the machine? |
|---|---|---|
| `auto` (default) | Ollama if it is running locally, else `offline` | no |
| `ollama` | local model via Ollama, e.g. `--model qwen2.5:7b-instruct` | no |
| `openai` | any OpenAI-compatible server, `--base-url ...` (llama.cpp, vLLM, LM Studio, OpenAI) | only if the URL is remote |
| `anthropic` | Claude (`claude-opus-5-5` by default, `--model` to change). Needs `pip install anthropic` and `ANTHROPIC_API_KEY` | yes. Serial numbers, UUIDs and MACs are redacted first |
| `offline` | no model: keyword routing to the same tools + templated answers | no |

`auto` never picks a cloud model by itself.

```bash
ollama pull qwen2.5:7b-instruct && hwqa                  # fully local, with an LLM
hwqa --backend anthropic --model claude-haiku-5-5        # cloud, cheaper/faster model
```

### Without a Linux machine

Recorded machines live in `tests/fixtures/`, and the assistant can replay them anywhere:

```bash
python -m hwqa --fixture tests/fixtures/laptop_16gb "Can I add more RAM?"
python -m hwqa --fixture tests/fixtures/thinkpad_soldered "what GPU do I have?"
```

`laptop_16gb` is the brief's example (16 GB in 1 of 2 slots, 64 GB max).
`thinkpad_soldered` has 8 GB soldered plus one slot, no root, and no lspci/lsusb/lsblk.

## Tests

```bash
pip install pytest
python -m pytest -q        # 89 tests: parsers, safety, tools, offline answers, LLM loops, grounding
```

They run on any OS against the fixtures. Two of them need a real Linux `/sys` and are skipped
elsewhere. The whole suite was also run on Ubuntu 26.04 under WSL2, together with the live CLI.
That environment is a useful worst case: virtual disks, no DMI, no lspci, and a GPU visible only
through nvidia-smi.

## Layout

```
hwqa/
  system.py      the only code that touches the OS: command allow-list, safe runner, file allow-list
  parsers.py     lscpu, dmidecode, lsblk, lspci, lsusb, meminfo, nvidia-smi, smartctl, pci.ids
  tools.py       the 12 hardware tools the model can call, with fallbacks and "unavailable" reasons
  toolbox.py     argument validation, caching, redaction: the model/tools boundary
  grounding.py   flags numbers in an answer that no tool reported
  prompts.py     system prompt
  session.py     conversation + grounding check
  backends/      offline, OpenAI-compatible (Ollama), Anthropic
  cli.py
packaging/hwqa.sudoers
tests/fixtures/  two recorded machines
```
