"""No-LLM mode: keyword routing to tools and templated answers.

Useful when no model is available, and as a deterministic baseline. It goes
through the same ToolBox as the LLM backends, so it gets the same safety
checks and the same "unavailable, because..." information, and it can only
say what the tools returned.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional

from ..toolbox import ToolBox
from ..units import fmt_disk, fmt_ram
from .base import Backend, Turn

TOPICS: list[tuple[str, re.Pattern]] = [
    ("memory", re.compile(r"\b(ram|memory|dimms?|so-?dimms?)\b|\bslots?\b(?!.*\b(pcie|m\.2|nvme)\b)")),
    ("disk_health", re.compile(r"\b(smart|failing|wear(ing)? out|wear level)\b|"
                               r"\b(ssd|disk|drive|nvme|hdd)\b.*\bhealth(y)?\b|\bhealth(y)?\b.*\b(ssd|disk|drive|nvme|hdd)\b")),
    ("storage", re.compile(r"\b(ssd|ssds|nvme|hdd|disks?|drives?|storage|space|partitions?|sata|m\.2)\b")),
    ("gpu", re.compile(r"\b(gpus?|graphics|video card|vram|nvidia|radeon|geforce|cuda|igpu|dgpu)\b")),
    ("cpu", re.compile(r"\b(cpus?|processors?|cores?|threads?|ghz|clock speed|virtuali[sz]ation|vt-x|amd-v|avx\w*|"
                       r"hyper-?threading)\b")),
    ("battery", re.compile(r"\b(battery|batteries|charging|charger|charge|plugged|cycle count)\b")),
    ("network", re.compile(r"\b(wi-?fi|wireless|wlan|ethernet|network|lan|mac address(es)?|bluetooth)\b")),
    ("usb", re.compile(r"\busb\b")),
    ("display", re.compile(r"\b(screen|display|monitors?|resolution|panel|hdmi|displayport)\b")),
    ("temperature", re.compile(r"\b(temps?|temperatures?|hot|thermal|overheat\w*)\b")),
    ("system", re.compile(r"\b(model|manufacturer|make|brand|vendor|bios|firmware|uefi|motherboard|serial|os|"
                          r"distro|distribution|kernel|ubuntu|fedora|debian|arch linux|virtual machine|vm|wsl)\b")),
    ("capabilities", re.compile(r"\b(what can you|what can'?t you|what do you|can you see|can'?t you see|"
                                r"why can'?t you|permissions?|root access|what are you able)\b")),
]
GENERIC_MACHINE = re.compile(r"\b(laptop|computer|machine|pc|notebook|this thing)\b")
FOLLOW_UP = re.compile(r"^(and|also|what about|how about|is it|is that|does it|which|how much|how many|how fast|"
                       r"what speed|what type|what kind|any)\b")

HELP = ("I can answer questions about this machine's CPU, memory (including upgrades), disks, GPU, battery, "
        "network, USB devices, displays, temperatures, and the model/BIOS/OS. For example: "
        "\"How much RAM do I have?\", \"Is my SSD NVMe?\", \"Can I add more RAM?\"")


def _a(phrase: str) -> str:
    """'a' or 'an' for a phrase: an Acme, an 11th Gen, an 8-core, a 16 GB."""
    first = phrase.split()[0].lower() if phrase.split() else ""
    vowel = first[:1] in "aeio" or re.match(r"^(8|11|18|80+)\b|^(8|11|18)\D", first)
    return f"an {phrase}" if vowel else f"a {phrase}"


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _unavailable(res: dict, skip: tuple = ()) -> list[str]:
    out = []
    for u in res.get("unavailable", []):
        if any(s in u["what"] for s in skip):
            continue
        line = f"I can't determine the {u['what']}: {u['reason']}."
        if u.get("fix"):
            line += f" To fix that, {u['fix']}."
        out.append(line)
    return out


def _failed(res: dict) -> Optional[str]:
    if not res.get("ok"):
        return f"I couldn't read that information ({res.get('error', 'unknown error')})."
    return None


# --------------------------------------------------------------------------- composers

def memory_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    wants_upgrade = bool(re.search(r"\b(upgrade|add|more|expand|increase|max|maximum|support|slots?|extra|another)\b", q))
    wants_usage = bool(re.search(r"\b(free|available|used|usage|left|using)\b", q))
    out = []
    if d.get("installed_bytes"):
        s = f"You have {fmt_ram(d['installed_bytes'])} of RAM"
        if d.get("slots_total"):
            s += f" in {d['slots_used']} of {d['slots_total']} slot{'s' if d['slots_total'] != 1 else ''}"
        groups = Counter(
            (m.get("size_bytes"), m.get("type"), m.get("form_factor"), m.get("rated_speed"), m.get("soldered"))
            for m in d.get("slots", []) if m.get("installed"))
        parts = []
        for (size, mtype, ff, speed, soldered), n in groups.items():
            desc = " ".join(x for x in (fmt_ram(size), mtype, ff) if x)
            if speed:
                desc += f" at {speed}"
            if soldered:
                desc += ", soldered"
            parts.append(f"{n} × {desc}")
        if parts:
            s += f" ({'; '.join(parts)})"
        out.append(s + ".")
        if wants_usage or not wants_upgrade:
            if d.get("usable_bytes"):
                out.append(f"The OS can use {fmt_ram(d['usable_bytes'])} of it, and {fmt_ram(d.get('available_bytes'))} "
                           "is free right now.")
    elif d.get("usable_bytes"):
        out.append(f"The OS sees {fmt_ram(d['usable_bytes'])} of RAM, and {fmt_ram(d.get('available_bytes'))} "
                   "is free right now.")

    up = d.get("upgrade")
    if wants_upgrade and up:
        out.append(_upgrade_sentence(up))
    if wants_upgrade or not d.get("installed_bytes"):
        out.extend(_unavailable(res))
    return " ".join(out) or "I couldn't find any memory information."


def _upgrade_sentence(up: dict) -> str:
    if up.get("upgradeable_slots") == 0:
        return "All of it is soldered to the board, so it can't be upgraded."
    if up.get("possible") is None:
        return ("The firmware doesn't report the maximum the board supports, so I can't tell how far it can go. "
                "Check the manufacturer's spec sheet.")
    maxc = fmt_ram(up["max_capacity_bytes"])
    caveat = ("(That maximum comes from the firmware's DMI table, which is occasionally conservative; "
              "confirm with the spec sheet before buying.)")
    soldered = (f" {fmt_ram(up['soldered_bytes'])} of it is soldered to the board and stays."
                if up.get("soldered_bytes") else "")
    if up.get("free_slots"):
        n = up["free_slots"]
        where = "the free slot" if n == 1 else f"each of the {n} free slots"
        s = (f"The board supports up to {maxc}, so you can add a stick of up to "
             f"{fmt_ram(up['largest_module_for_free_slot_bytes'])} in {where} "
             f"({fmt_ram(up['max_total_by_adding_bytes'])} total).")
        if up.get("max_total_by_replacing_bytes", 0) > up["max_total_by_adding_bytes"]:
            s += (f" To reach {fmt_ram(up['max_total_by_replacing_bytes'])} you'd have to replace the existing "
                  "module as well.")
    elif up.get("possible"):
        s = (f"Every upgradeable slot is in use. The board supports up to {maxc}, so you'd need to swap in larger "
             f"modules to reach {fmt_ram(up['max_total_by_replacing_bytes'])}.")
    else:
        s = f"You're already at the board's maximum of {maxc}."
    buy = next((n for n in up.get("notes", []) if n.startswith("Buy ")), "")
    return " ".join(x for x in (s + soldered, buy, caveat) if x)


def cpu_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    out = []
    if d.get("model"):
        s = f"Your CPU is {_a(d['model'])}"
        counts = []
        if d.get("physical_cores"):
            counts.append(f"{d['physical_cores']} cores")
        if d.get("threads"):
            counts.append(f"{d['threads']} threads")
        if counts:
            s += f" with {' and '.join(counts)}"
        out.append(s + ".")
    hy = d.get("hybrid")
    if hy:
        out.append(f"It's a hybrid design: {hy['performance_threads']} threads on performance cores and "
                   f"{hy['efficiency_threads']} on efficiency cores.")
    if d.get("max_mhz"):
        s = f"It boosts up to {d['max_mhz'] / 1000:.1f} GHz"
        if d.get("base_ghz"):
            s += f" (base {d['base_ghz']} GHz)"
        out.append(s + ".")
    if re.search(r"virtuali[sz]|vt-x|amd-v|vm\b|kvm", q):
        v = d.get("hardware_virtualisation")
        out.append(f"Hardware virtualisation: {v}." if v else "")
    if re.search(r"avx|instruction|feature|aes|flags", q):
        out.append(f"Notable instruction set features: {_join(d.get('features', [])) or 'none of the usual ones'}.")
    if re.search(r"cache", q) and d.get("caches"):
        out.append("Caches: " + ", ".join(f"{k} {v}" for k, v in d["caches"].items()) + ".")
    out.extend(_unavailable(res))
    out.extend(n for n in res.get("notes", []) if "hypervisor" in n or "Hybrid" in n and not hy)
    return " ".join(x for x in out if x)


def storage_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    disks = res["data"].get("disks", [])
    if not disks:
        return " ".join(_unavailable(res)) or "I can't see any disks."
    main = next((d for d in disks if d.get("holds_root_filesystem")), disks[0])

    def name(d):
        return f"{d['size']} {d['model']}" if d.get("model") else f"{d['size']} disk ({d['name']})"

    out = []
    if main["media"] == "virtual disk" and re.search(r"\b(nvme|ssd|hdd|sata|hard drive|type)\b", q):
        return (f"I can't tell from here. Your system drive ({main['name']}, {main['size']}) is a virtual disk "
                "provided by the hypervisor, so the physical drive behind it isn't visible. Check on the host OS.")
    if "nvme" in q:
        if main["is_nvme"]:
            out.append(f"Yes. Your system drive, a {name(main)}, is an NVMe SSD.")
        else:
            out.append(f"No. Your system drive, a {name(main)}, connects over {main['interface']} "
                       f"({main['media']}), not NVMe.")
        others = [d for d in disks if d is not main]
        if others:
            out.append("Other drives: " + "; ".join(
                f"{name(d)} ({'removable ' if d.get('removable') else ''}{d['media']}"
                + (f" on {d['interface']}" if d["interface"] != "unknown" else "") + ")" for d in others) + ".")
    elif re.search(r"\b(space|free|full|left|usage|used)\b", q):
        parts = []
        mounted = [p for d in disks for p in d["partitions"] if p.get("usage") and p.get("mountpoint")]
        for p in sorted(mounted, key=lambda p: (p["mountpoint"] != "/", p["mountpoint"])):
            u = p["usage"]
            parts.append(f"{p['mountpoint']} has {fmt_disk(u['free_bytes'])} free of "
                         f"{fmt_disk(u['total_bytes'])} ({u['used_percent']:.0f}% used)")
        out.append(("; ".join(parts) + ".") if parts else "I can't see usage for any mounted filesystem.")
    elif re.search(r"\b(ssd|hdd|hard drive|spinning)\b", q):
        out.append(f"Your system drive, a {name(main)}, is {'an' if main['media'] in ('SSD', 'HDD') else 'a'} "
                   f"{main['media']} on {main['interface']}.")
    else:
        lines = [f"{d['name']}: {name(d)}, " + (f"{d['interface']} {d['media']}" if d["interface"] != "unknown"
                                                  else d["media"]) + (", removable" if d.get("removable") else "")
                 for d in disks]
        out.append(("You have one drive: " if len(disks) == 1 else f"You have {len(disks)} drives: ")
                   + "; ".join(lines) + ".")
    out.extend(res.get("notes", [])[:1] if any("virtual" in n.lower() for n in res.get("notes", [])) else [])
    return " ".join(out)


def disk_health_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    if d.get("error"):
        return d["error"]
    if not d.get("passed") and "passed" not in d:
        return " ".join(_unavailable(res)) or "No SMART data available."
    s = f"SMART health for {d['device']}: {'PASSED' if d['passed'] else 'FAILED'}."
    extra = []
    if d.get("percentage_used") is not None:
        extra.append(f"{d['percentage_used']}% of rated endurance used")
    if d.get("power_on_hours") is not None:
        extra.append(f"{d['power_on_hours']} power-on hours")
    if d.get("temperature_c") is not None:
        extra.append(f"{d['temperature_c']}°C now")
    if d.get("media_errors"):
        extra.append(f"{d['media_errors']} media errors")
    joined = _join(extra)
    return s + (" " + joined[:1].upper() + joined[1:] + "." if extra else "")


def gpu_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    gpus = res["data"].get("gpus", [])
    if not gpus:
        return " ".join(_unavailable(res))
    descs = []
    for g in gpus:
        bits = [g.get("kind") if g.get("kind") != "unknown" else None,
                f"{g['vram']} VRAM" if g.get("vram") else None,
                f"driver {g['driver']}" if g.get("driver") else None]
        bits = [b for b in bits if b]
        descs.append((g.get("name") or g.get("model") or "an unknown GPU") + (f" ({', '.join(bits)})" if bits else ""))
    out = [("Your GPU is " if len(descs) == 1 else f"You have {len(descs)} GPUs: ") + "; ".join(descs) + "."]
    out.extend(n for n in res.get("notes", []) if "names" in n)
    return " ".join(out)


def battery_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    if not d.get("batteries"):
        return "I don't see a battery. This looks like a desktop or a virtual machine, or the battery driver isn't loaded."
    out = []
    for b in d["batteries"]:
        s = f"The battery is at {b['charge_percent']}%" if "charge_percent" in b else "The battery"
        if b.get("status"):
            s += f" and {b['status'].lower()}"
        out.append(s + ".")
        if "health_percent" in b:
            h = f"Its health is {b['health_percent']}% of the original capacity"
            if b.get("full_capacity_wh") and b.get("design_capacity_wh"):
                h += f" ({b['full_capacity_wh']} of {b['design_capacity_wh']} Wh)"
            if b.get("cycle_count"):
                h += f" after {b['cycle_count']} charge cycles"
            out.append(h + ".")
    for a in d.get("ac_adapters", []):
        if a["type"] == "Mains":
            out.append("The charger is plugged in." if a["plugged_in"] else "The charger is not plugged in.")
            break
    out.extend(_unavailable(res))
    return " ".join(out)


def network_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    out = []
    wifi = [c for c in d.get("controllers", []) if c["type"] == "wifi"]
    eth = [c for c in d.get("controllers", []) if c["type"] == "ethernet"]
    if wifi:
        out.append("Wi-Fi: " + "; ".join(f"{c['vendor']} {c['model']}" for c in wifi) + ".")
    if eth:
        out.append("Ethernet: " + "; ".join(f"{c['vendor']} {c['model']}" for c in eth) + ".")
    for i in d.get("interfaces", []):
        if i["kind"] in ("wifi", "ethernet"):
            s = f"{i['name']} ({i['kind']}) is {i.get('state', 'unknown')}"
            if i.get("link_speed_mbps"):
                s += f" at {i['link_speed_mbps']} Mb/s"
            if i.get("mac_address"):
                s += f", MAC {i['mac_address']}"
            out.append(s + ".")
    if not out:
        virtual = [i["name"] for i in d.get("interfaces", []) if i["kind"] == "virtual"]
        out.append("I only see virtual network interfaces" + (f" ({', '.join(virtual)})" if virtual else "") + ".")
    if "bluetooth" in q or not out:
        out.append("Bluetooth is present." if d.get("bluetooth") else "I don't see a Bluetooth adapter.")
    out.extend(n for n in res.get("notes", []))
    return " ".join(out)


def usb_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    devs = res["data"].get("devices", [])
    if not devs:
        return "No USB devices are connected (apart from the built-in hubs)."
    names = [d.get("name") or f"device {d['vendor_id']}:{d.get('product_id')}" for d in devs]
    return f"{len(devs)} USB device{'s' if len(devs) != 1 else ''} connected: " + "; ".join(names) + "."


def display_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    if not d.get("outputs"):
        return " ".join(_unavailable(res))
    conn = d.get("connected", [])
    if not conn:
        return "No displays are reported as connected."
    parts = [f"{'built-in panel' if o['internal'] else o['connector']}" + (
        f" at {o['preferred_mode']}" if o.get("preferred_mode") else "") for o in conn]
    return f"Connected display{'s' if len(conn) != 1 else ''}: " + "; ".join(parts) + "."


def temperature_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    sensors = res["data"].get("sensors", [])
    if not sensors:
        return " ".join(_unavailable(res))
    hottest = max(sensors, key=lambda s: s["celsius"])
    listing = "; ".join(f"{s['sensor']} {s['celsius']:.0f}°C" for s in sensors[:8])
    return f"The hottest sensor is {hottest['sensor']} at {hottest['celsius']:.0f}°C. Readings: {listing}."


def system_answer(res: dict, q: str) -> str:
    if _failed(res):
        return _failed(res)
    d = res["data"]
    out = []
    if d.get("model"):
        maker = d.get("manufacturer")
        name = d["model"] if maker and maker.lower() in d["model"].lower() else " ".join(
            x for x in (maker, d["model"]) if x)
        s = "This is " + _a(name)
        if d.get("form_factor"):
            s += f" ({d['form_factor']})"
        out.append(s + ".")
    else:
        out.extend(_unavailable({"unavailable": [u for u in res.get("unavailable", []) if "model" in u["what"]]}))
    if re.search(r"bios|firmware|uefi", q) and d.get("bios"):
        b = d["bios"]
        s = "BIOS " + " ".join(x for x in (b.get("vendor"), f"version {b['version']}" if b.get("version") else None) if x)
        out.append(s + (f", dated {b['date']}" if b.get("date") else "") + ".")
    if re.search(r"motherboard|board", q) and d.get("motherboard"):
        out.append(f"Motherboard: {d['motherboard']}.")
    if re.search(r"\b(os|distro|distribution|kernel|ubuntu|fedora|debian|linux)\b", q) or not out:
        if d.get("os"):
            out.append(f"It's running {d['os']}" + (f" with kernel {d['kernel']}" if d.get("kernel") else "") + ".")
    if d.get("serial_number"):
        out.append(f"Serial number: {d['serial_number']}.")
    if d.get("virtualisation"):
        out.append(f"Note: this is running under {d['virtualisation']}, so you're seeing virtual hardware.")
    out.extend(_unavailable(res, skip=("model",)))
    return " ".join(out)


def capabilities_answer(res: dict, q: str) -> str:
    d = res["data"]
    missing = [b for b, s in d["commands"].items() if s.startswith("missing")]
    out = [f"Privileges: {d['privilege']}."]
    if missing:
        out.append("Not installed here: " + ", ".join(missing) + ".")
    if d.get("virtualisation"):
        out.append(f"Environment: {d['virtualisation']}.")
    return " ".join(out)


Composer = Callable[[dict, str], str]

ROUTES: dict[str, tuple[str, Callable[[str], dict], Composer]] = {
    "memory": ("get_memory_info", lambda q: {}, memory_answer),
    "storage": ("get_storage_info", lambda q: {}, storage_answer),
    "gpu": ("get_gpu_info", lambda q: {}, gpu_answer),
    "cpu": ("get_cpu_info", lambda q: {}, cpu_answer),
    "battery": ("get_battery_info", lambda q: {}, battery_answer),
    "network": ("get_network_info", lambda q: {"include_mac": "mac" in q}, network_answer),
    "usb": ("get_usb_devices", lambda q: {}, usb_answer),
    "display": ("get_display_info", lambda q: {}, display_answer),
    "temperature": ("get_temperatures", lambda q: {}, temperature_answer),
    "system": ("get_system_overview", lambda q: {"include_serial": "serial" in q}, system_answer),
    "capabilities": ("get_capabilities", lambda q: {}, capabilities_answer),
}


class OfflineBackend(Backend):
    name = "offline (rule-based, no LLM)"
    local = True

    def __init__(self):
        self.last_topics: list[str] = []

    def reset(self) -> None:
        self.last_topics = []

    @staticmethod
    def topics_for(q: str) -> list[str]:
        found = [name for name, rx in TOPICS if rx.search(q)]
        if "disk_health" in found:
            found = [t for t in found if t != "storage"]
        if not found and GENERIC_MACHINE.search(q):
            found = ["system"]
        return found

    def ask(self, question: str, toolbox: ToolBox) -> Turn:
        q = question.lower().strip()
        topics = self.topics_for(q)
        if not topics and self.last_topics and (FOLLOW_UP.search(q) or len(q.split()) <= 6):
            topics = self.last_topics
        if not topics:
            return Turn(HELP, [])

        parts, calls = [], []
        for topic in topics:
            if topic == "disk_health":
                if "get_disk_health" not in toolbox.tools:
                    parts.append("I can't check disk health: smartctl isn't installed (package smartmontools).")
                    continue
                storage = toolbox.call("get_storage_info", {})
                calls.append(("get_storage_info", {}))
                disks = storage.get("data", {}).get("disks", [])
                main = next((d for d in disks if d.get("holds_root_filesystem")), disks[0] if disks else None)
                if main is None:
                    parts.append("I can't see any disks to check.")
                    continue
                args = {"device": main["name"]}
                res = toolbox.call("get_disk_health", args)
                calls.append(("get_disk_health", args))
                parts.append(disk_health_answer(res, q))
                continue
            tool, make_args, compose = ROUTES[topic]
            args = make_args(q)
            res = toolbox.call(tool, args)
            calls.append((tool, args))
            parts.append(compose(res, q))
        self.last_topics = topics
        return Turn("\n\n".join(p for p in parts if p), calls)
