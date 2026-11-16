# hwqa: design notes

A chat assistant that runs on a Linux laptop and answers questions about that machine's
hardware. The interesting parts are how the model gets facts (tool calling), what it is
allowed to touch (safety), and how it avoids making things up. The LLM itself is the
replaceable part, so it comes last here.

## 1. Flow

```
 user: "Can I add more RAM?"
        │
        ▼
 ┌──────────────┐  tool specs    ┌──────────────────────────────────────────────────────┐
 │  backend     │──────────────▶ │ ToolBox (the boundary)                               │
 │ (LLM or      │ ◀──────────────│  • tool exists & is available here?                  │
 │  offline)    │  JSON results  │  • args valid against schema (types/pattern/no extras)│
 └──────┬───────┘                │  • session cache (static: forever, live: seconds)    │
        │                        │  • redact serials/MACs for remote models             │
        │ answer                 └──────────────┬───────────────────────────────────────┘
        ▼                                       │ get_memory_info()
 ┌──────────────┐                               ▼
 │ grounding    │               ┌──────────────────────────────────────────┐
 │ check        │               │ tools.py: one function per area          │
 └──────┬───────┘               │  sources in fallback order, normalised   │
        ▼                       │  JSON, "unavailable" + reasons, derived  │
   answer (+ "couldn't          │  numbers (upgrade headroom, health %)    │
   verify: …" if any)           └──────────────┬───────────────────────────┘
                                               │ run("dmidecode_memory") / read("/proc/meminfo")
                                               ▼
                                ┌──────────────────────────────────────────┐
                                │ system.py: the only code touching the OS │
                                │  command allow-list (fixed argv)         │
                                │  trusted binary dirs, shell=False,       │
                                │  minimal env, timeout, output cap        │
                                │  sudo -n for the root-only ones          │
                                │  file read allow-list (/proc, /sys …)    │
                                └──────────────────────────────────────────┘
```

## 2. Tool-calling design

**Semantic tools, not commands.** The model sees 12 tools named after what people ask about:
`get_memory_info`, `get_storage_info`, `get_gpu_info`, `get_cpu_info`, `get_battery_info`,
`get_network_info`, `get_system_overview`, `get_display_info`, `get_usb_devices`,
`get_temperatures`, `get_disk_health`, `get_capabilities`. It never sees `lscpu` or `dmidecode`,
and there is no "run command" tool. That makes the safety story simple (section 3), and it is
also easier for small local models, which pick from a dozen well-described functions far more
reliably than they write correct shell commands.

**Each tool owns its sources and fallbacks.** For example:

| tool | primary | fallbacks |
|---|---|---|
| memory | `/proc/meminfo` + `dmidecode -t memory` (root) | `free -b`; without root, usable memory only, plus an explicit "slots unavailable: needs root" |
| storage | `lsblk -J -b` | `/sys/block/*` (size, rotational, model); `/proc/mounts` for the root disk |
| GPU | `lspci -vmm -nnk` | `/sys/class/drm/card*` + `pci.ids` lookup; `nvidia-smi` adds VRAM/driver (and is the only source under WSL) |
| CPU | `lscpu` | `/proc/cpuinfo` (core ids for real core counts), cpufreq sysfs for clocks |
| system | `/sys/class/dmi/id/*` | `/proc/device-tree/model` (ARM); `dmidecode -t system` only for the serial |

**Normalised output** with stable field names and units (bytes, MHz, %). Three fields matter as
much as `data`:

* `unavailable`: `[{"what", "reason", "fix"}]`, for what couldn't be found and why
  (not installed, needs root, not exposed in a VM).
* `notes`: caveats the answer should carry (DMI max may be conservative; running in WSL).
* `sources`: where each fact came from.

**Derived numbers are computed in code.** `memory_upgrade()` works out free slots, soldered
memory, per-slot maximum, "max by adding" and "max by replacing". `battery_info()` computes
health %. The prompt tells the model to use these as given, so the brief's example
("16 GB in 1 of 2 slots, up to 64 GB, add a stick of up to 32 GB") never depends on model
arithmetic.

**Parsers are pure functions** (`parsers.py`) tested against recorded output, including
version differences: old `lsblk` JSON with strings for numbers vs new typed JSON with
`mountpoints` lists; old flat vs new indented `lscpu`; `dmidecode` sizes as `16 GB` or
`16384 MB`; `LC_ALL=C` so output isn't localised.

**Follow-ups.** LLM backends keep the conversation (with tool results), so "and what about the
GPU?" is resolved by the model, and the system prompt says earlier facts stay valid except
live values. The ToolBox cache means repeated questions don't re-run commands: static facts are
cached for the session, live ones (free memory, battery, temperatures) for 5–30 s. The offline
backend tracks the last topic for short follow-ups ("how much vram?").

## 3. Safety

The model chooses *which* tool to call, never *what runs*.

1. **Allow-listed commands with fixed argv** (`system.COMMANDS`). Ids map to a binary and a
   fixed argument tuple. Nothing from the conversation reaches argv, with one exception (the
   disk name for `smartctl`). That value is validated twice: the tool schema pattern
   `^[a-z0-9]{2,16}$`, then `^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|mmcblk\d+)$` and it must exist under
   `/sys/block`. `nvme0n1; rm -rf /` is rejected before any process starts (tested).
2. **No shell, ever.** `subprocess.run(argv, shell=False)`, so `;`, `&&`, `|` and `$()` have no
   meaning.
3. **Trusted binaries only.** Resolved from `/usr/sbin`, `/usr/bin`, `/sbin`, `/bin` (and
   `/usr/lib/wsl/lib` for WSL's nvidia-smi), never `$PATH`. They must be root-owned and not
   group/world-writable, so a fake `lscpu` in `~/bin` is ignored (tested on Linux).
4. **Contained execution.** Empty stdin, minimal fixed environment, per-command timeout, 2 MB
   output cap, `close_fds`.
5. **Read-only by construction.** Every allow-listed invocation is a query. A test fails if
   someone adds a mutating flag (`-s`, `--set`, `-w`…) to the table.
6. **File reads** are limited to `/proc/{cpuinfo,meminfo,version,mounts}`, `/proc/device-tree/`,
   `/sys/`, os-release and the PCI/USB id databases, checked **after** resolving symlinks.
   `/etc/shadow`, `~/.ssh`, `/proc/self/environ` and `/sys/../etc/shadow` are refused.
7. **Boundary validation** (`toolbox.validate`): unknown tools, unexpected keys, wrong types
   and pattern violations come back as error results the model can read, rather than as
   exceptions or silent coercion.
8. **Prompt injection** is a small surface here (the inputs are local system files), but the
   prompt still says tool output is data, not instructions.

## 4. Root

`dmidecode` (slots, max capacity, serial) and `smartctl` need root. Running the assistant, or
the LLM loop, as root would be the wrong fix. Instead, only the two root-only commands are
escalated, always via `sudo -n -- <fixed argv>`, so **sudo never prompts mid-chat**. That works
in three ways:

| option | how | trade-off |
|---|---|---|
| `hwqa --sudo` | runs `sudo -v` once at start-up (normal password prompt), then `sudo -n` rides on the cached credentials | nothing to install; credentials expire after sudo's timeout (~15 min) |
| scoped sudoers rule | `packaging/hwqa.sudoers` / `hwqa --print-sudoers`: NOPASSWD for exactly `dmidecode -t memory`, `dmidecode -t system`, `smartctl -H -A -j /dev/<disk>` | one-time setup; arguments pinned, no `*` (sudoers wildcards also match spaces) |
| neither | the tool returns `needs root` + the fix; the answer uses unprivileged data and says what's missing | always safe |

When `sudo -n` reports that a password is needed, the runner remembers it and stops trying.
For a desktop product, the next step would be a tiny root helper over D-Bus/polkit exposing
`GetMemoryTopology()` and similar. The design already allows that: it replaces `LiveSystem.run`
for those two ids.

## 5. Accuracy: not hallucinating, and saying "I don't know"

* The **system prompt** requires tool calls before answering, forbids inferring specs from
  model names, and says that an "I can't determine X because Y" answer is better than a guess.
  It includes a live summary of the machine's limits (privileges, missing tools, VM/WSL).
* **Tools say what they don't know** (`unavailable` with reason and fix), so the model has
  something concrete to say instead of a gap to fill.
* **Derived numbers come from code** (section 2).
* **Grounding check** (`grounding.py`): after each answer, numbers with units (`GB`, `GHz`, `%`,
  `°C`, `slots`…) are matched against every value the tools returned this session, after unit
  conversion (bytes → GB binary/decimal, MHz → GHz, rounding). Anything unmatched is shown as
  "couldn't verify against system data: 128 GB". It is a backstop, not a proof.
* **Environment honesty**: VMs and WSL are detected (kernel string, DMI vendor, `hypervisor`
  flag), and answers say so. Virtual disks aren't called HDDs, and the Hyper-V virtual battery's
  fake capacity isn't reported as battery health. These came out of running the tool live under
  WSL2.

## 6. Portability

* **Capability discovery** at start-up: which binaries exist, privilege level, VM/WSL. Tools that
  can't work aren't offered at all (`get_disk_health` only exists if `smartctl` does), and the
  prompt lists what's missing.
* **Fallback chains** per tool (section 2), so a minimal install without pciutils, usbutils or
  util-linux's lsblk still answers most questions. The `thinkpad_soldered` fixture is exactly
  that machine.
* **Distro and version differences** live in the parsers and are tested with recorded output.
* **ARM / non-DMI**: device-tree model, `Hardware`/`Features` fields in cpuinfo, repeated
  `lscpu` keys for big.LITTLE.
* **Hybrid Intel CPUs**: P/E thread counts from `/sys/devices/cpu_core|cpu_atom/cpus`.
* **Laptop specifics**: soldered RAM (`Row Of Chips`, LPDDR, "onboard" locators), peripheral
  batteries (`scope=Device`) excluded, Lenovo's model name in `product_version`.

## 7. Privacy

* The default is **local**. `auto` uses Ollama if it's running, otherwise the offline backend.
  It never picks a cloud model on its own.
* With a **remote** backend (Anthropic, or an OpenAI-compatible URL that isn't localhost),
  serial numbers, UUIDs, asset tags and MAC addresses are **redacted before results reach the
  model**, and the banner says the session is remote. `--show-identifiers` opts out.
* Serials and MACs are only collected when the user asks for them (`include_serial`,
  `include_mac` default to false).
* Nothing is written to disk and there's no telemetry. `/audit` shows every command that ran.

## 8. Backends

All three share one interface (`ask(question, toolbox) -> Turn`) and go through the same
ToolBox, so safety, caching, redaction and grounding don't depend on the model.

* **OpenAI-compatible** (`openai_compat.py`, stdlib `urllib`): Ollama's `/v1` by default
  (`qwen2.5:7b-instruct` handles function calling well at 7B), also llama.cpp server, vLLM,
  LM Studio. History is trimmed to the last 8 turns for small context windows.
* **Anthropic** (`anthropic_backend.py`, official SDK): manual tool loop. All tool results of a
  turn go back in one message paired by `tool_use_id`. A turn cut off at `max_tokens` never
  runs its (possibly partial) tool calls, and the history is closed with error results so it
  stays valid. Refusals are handled, and the server-side fallback option is on. Verified against
  the real SDK with a local mock API.
* **Offline** (`offline.py`): regex intent routing to the same tools plus templated answers.
  No model needed, deterministic, and a useful baseline. It reproduces the brief's example
  answer.
* Every loop is capped at 6 tool rounds per question.

## 9. Testing

89 tests (`python -m pytest -q`):

* **parsers**: recorded outputs, old and new formats
* **safety**: no shell, fixed env, `sudo -n`, injection in arguments, unknown tools, file
  allow-list, untrusted binaries, read-only guard, redaction
* **tools**: the brief's upgrade example, soldered memory, no-root paths, sysfs/drm fallbacks,
  battery/network/USB edge cases, caching, a tool crashing without killing the session
* **offline answers**: brief example text, follow-ups, "I don't know and here's why"
* **backends**: the OpenAI loop against a scripted local HTTP server (tool round trip, bad tool
  names, hallucinated numbers flagged, runaway loops stopped); the Anthropic loop with a fake
  client (pairing, cut-off tool calls, refusal, follow-up history)
* **grounding**

Also run on Ubuntu 26.04 / WSL2 with the live CLI.

## 10. Next steps

* A polkit/D-Bus helper for the root-only queries (desktop-friendly, no sudo).
* `hwqa --record DIR` to capture a machine as a fixture for bug reports.
* More tools where people actually ask: audio devices, webcam, PCIe slots / M.2 bays (DMI type 9),
  firmware updates via `fwupdmgr get-devices` (read-only).
* An eval set of question → expected-fact pairs, run against each backend/model to choose the
  default local model with numbers rather than by feel.
