"""Hardware tools the assistant can call.

Each tool answers one area (CPU, memory, storage, ...), gathers from whatever
sources this machine has, and returns normalised JSON:

    {"tool": ..., "ok": true,
     "data": {...},                     facts, in stable field names and units
     "sources": ["lscpu", "/proc/cpuinfo"],
     "unavailable": [{"what", "reason", "fix"}],   what it could NOT find, and why
     "notes": [...]}                    caveats the answer should mention

The `unavailable` list is what lets the model say "I can't tell you X because
Y" instead of filling the gap with a guess. Derived numbers (memory upgrade
headroom, battery health %) are computed here, not left to the model's
arithmetic.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import parsers
from .system import COMMANDS, CommandResult, System
from .units import fmt_disk, fmt_ram, parse_size

PACKAGES = {
    "lscpu": "util-linux",
    "lsblk": "util-linux",
    "free": "procps",
    "lspci": "pciutils",
    "lsusb": "usbutils",
    "dmidecode": "dmidecode",
    "smartctl": "smartmontools",
    "nvidia-smi": "the NVIDIA driver",
}
ROOT_FIX = ("start hwqa with --sudo (asks for your password once), or install the read-only sudoers rule "
            "printed by `hwqa --print-sudoers`")


class Result:
    def __init__(self):
        self.data: dict = {}
        self.sources: list[str] = []
        self.unavailable: list[dict] = []
        self.notes: list[str] = []

    def missing(self, what: str, reason: str, fix: Optional[str] = None) -> None:
        item = {"what": what, "reason": reason}
        if fix:
            item["fix"] = fix
        self.unavailable.append(item)

    def missing_from(self, what: str, res: CommandResult, cmd_id: str) -> None:
        binary = COMMANDS[cmd_id].binary
        if res.error == "missing":
            self.missing(what, f"{binary} is not installed", f"install {PACKAGES.get(binary, binary)}")
        elif res.error == "needs_root":
            self.missing(what, f"{binary} needs root to read this", ROOT_FIX)
        else:
            self.missing(what, f"{binary} failed: {res.message}")


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Result]
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {},
                                                      "additionalProperties": False})
    ttl: float = math.inf   # seconds a result stays valid in the session cache
    available: Callable[[System], bool] = lambda system: True


def _read(system: System, path: str) -> Optional[str]:
    v = system.read(path)
    return v.strip() if v is not None and v.strip() else None


def _read_int(system: System, path: str) -> Optional[int]:
    v = _read(system, path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


_PLACEHOLDERS = {"to be filled by o.e.m.", "default string", "system product name", "system version",
                 "not applicable", "not specified", "none", "n/a", "invalid", "0", "0123456789", "unknown",
                 "type1productconfigid", "system manufacturer"}


def _real_value(v: Optional[str]) -> Optional[str]:
    return v if v and v.strip().lower() not in _PLACEHOLDERS else None


def _cpu_list_count(spec: Optional[str]) -> Optional[int]:
    """'0-7,16-19' -> 12"""
    if not spec:
        return None
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def _virtualisation(system: System) -> Optional[str]:
    if "microsoft" in system.kernel_release().lower():
        return "WSL2 (Windows Subsystem for Linux)"
    vendor = (_read(system, "/sys/class/dmi/id/sys_vendor") or "").lower()
    product = (_read(system, "/sys/class/dmi/id/product_name") or "").lower()
    for needle, label in (("qemu", "QEMU/KVM virtual machine"), ("vmware", "VMware virtual machine"),
                          ("innotek", "VirtualBox virtual machine"), ("xen", "Xen virtual machine")):
        if needle in vendor or needle in product:
            return label
    if "microsoft" in vendor and "virtual" in product:
        return "Hyper-V virtual machine"
    cpuinfo = system.read("/proc/cpuinfo") or ""
    if re.search(r"^flags\s*:.*\bhypervisor\b", cpuinfo, re.M):
        return "virtual machine"
    return None


# --------------------------------------------------------------------------- system

CHASSIS = {3: "desktop", 4: "low-profile desktop", 6: "mini tower", 7: "tower", 8: "portable", 9: "laptop",
           10: "notebook", 11: "handheld", 13: "all-in-one", 14: "sub-notebook", 30: "tablet",
           31: "convertible laptop", 32: "detachable laptop", 35: "mini PC", 36: "stick PC"}


def system_overview(system: System, include_serial: bool = False) -> Result:
    r = Result()
    dmi = {}
    for key in ("sys_vendor", "product_name", "product_version", "product_family", "board_vendor", "board_name",
                "bios_vendor", "bios_version", "bios_date", "chassis_type"):
        v = _real_value(_read(system, f"/sys/class/dmi/id/{key}"))
        if v:
            dmi[key] = v
    if dmi:
        r.sources.append("/sys/class/dmi/id")
        vendor = dmi.get("sys_vendor")
        # Lenovo keeps the marketing name in product_version and a type code in product_name.
        if vendor and vendor.upper() == "LENOVO" and dmi.get("product_version"):
            r.data["model"], r.data["model_code"] = dmi["product_version"], dmi.get("product_name")
        else:
            r.data["model"] = dmi.get("product_name")
        r.data["manufacturer"] = vendor
        r.data["family"] = dmi.get("product_family")
        if dmi.get("board_name"):
            r.data["motherboard"] = " ".join(x for x in (dmi.get("board_vendor"), dmi["board_name"]) if x)
        r.data["bios"] = {k: dmi.get(f"bios_{k}") for k in ("vendor", "version", "date") if dmi.get(f"bios_{k}")}
        if dmi.get("chassis_type", "").isdigit():
            r.data["form_factor"] = CHASSIS.get(int(dmi["chassis_type"]), f"chassis type {dmi['chassis_type']}")
    else:
        dt_model = _read(system, "/proc/device-tree/model")
        if dt_model:
            r.data["model"] = dt_model.strip("\x00")
            r.sources.append("/proc/device-tree/model")
        else:
            r.missing("manufacturer and model", "the firmware exposes no DMI or device-tree information "
                      "(normal inside WSL, containers and some VMs)")

    osr = parsers.parse_os_release(system.read("/etc/os-release") or system.read("/usr/lib/os-release") or "")
    if osr:
        r.data["os"] = osr.get("PRETTY_NAME") or f"{osr.get('NAME', '')} {osr.get('VERSION', '')}".strip()
        r.sources.append("/etc/os-release")
    kernel = system.kernel_release()
    if kernel:
        r.data["kernel"] = kernel

    virt = _virtualisation(system)
    if virt:
        r.data["virtualisation"] = virt
        r.notes.append(f"This is running under {virt}, so hardware details describe the virtual machine "
                       "and some physical details are hidden.")

    if include_serial:
        serial = _real_value(_read(system, "/sys/class/dmi/id/product_serial"))
        if not serial:
            res = system.run("dmidecode_system")
            if res.ok:
                for rec in parsers.parse_dmidecode(res.stdout):
                    if rec.type == 1:
                        serial = _real_value(rec.props.get("Serial Number"))
                r.sources.append("dmidecode -t system")
            else:
                r.missing_from("serial number", res, "dmidecode_system")
        if serial:
            r.data["serial_number"] = serial
        elif not r.unavailable or r.unavailable[-1]["what"] != "serial number":
            r.missing("serial number", "the firmware does not report one")
    return r


# --------------------------------------------------------------------------- CPU

NOTABLE_FLAGS = {
    "avx": "AVX", "avx2": "AVX2", "avx512f": "AVX-512", "aes": "AES-NI", "sha_ni": "SHA extensions",
    "fma": "FMA3", "vmx": "Intel VT-x", "svm": "AMD-V", "ht": "Hyper-Threading capable", "sse4_2": "SSE4.2",
}


def cpu_info(system: System) -> Result:
    r = Result()
    res = system.run("lscpu")
    info = parsers.parse_lscpu(res.stdout) if res.ok else {}
    if res.ok:
        r.sources.append("lscpu")
    else:
        r.notes.append(f"lscpu unavailable ({res.message}); using /proc/cpuinfo only")
    cpuinfo = parsers.parse_cpuinfo(system.read("/proc/cpuinfo") or "")
    if cpuinfo:
        r.sources.append("/proc/cpuinfo")
    first = cpuinfo[0] if cpuinfo else {}

    model = info.get("Model name") or first.get("model name") or first.get("Hardware") or first.get("cpu model")
    if model:
        r.data["model"] = re.sub(r"\s+", " ", re.sub(r"\((R|TM)\)", "", model, flags=re.I)).strip()
    else:
        r.missing("CPU model name", "neither lscpu nor /proc/cpuinfo report it")
    r.data["vendor"] = info.get("Vendor ID") or first.get("vendor_id")
    r.data["architecture"] = info.get("Architecture")
    if info.get("CPU op-mode(s)"):
        r.data["op_modes"] = info["CPU op-mode(s)"]

    threads = int(info["CPU(s)"]) if info.get("CPU(s)", "").isdigit() else (len(cpuinfo) or None)
    sockets = int(info["Socket(s)"]) if info.get("Socket(s)", "").isdigit() else 1
    core_ids = {(b.get("physical id"), b.get("core id")) for b in cpuinfo if "core id" in b}
    if core_ids:
        cores = len(core_ids)
    elif info.get("Core(s) per socket", "").isdigit():
        cores = int(info["Core(s) per socket"]) * sockets
    else:
        cores = None
    r.data.update({"sockets": sockets, "physical_cores": cores, "threads": threads})
    if cores and threads:
        r.data["smt_enabled"] = threads > cores

    # Intel hybrid CPUs (12th gen+) expose separate PMUs for P- and E-cores.
    p_cpus = _cpu_list_count(_read(system, "/sys/devices/cpu_core/cpus"))
    e_cpus = _cpu_list_count(_read(system, "/sys/devices/cpu_atom/cpus"))
    if p_cpus and e_cpus:
        r.data["hybrid"] = {"performance_threads": p_cpus, "efficiency_threads": e_cpus}
        r.notes.append("Hybrid CPU: performance cores run two threads each, efficiency cores one.")

    max_khz = []
    for d in system.listdir("/sys/devices/system/cpu"):
        if re.fullmatch(r"cpu\d+", d):
            v = _read_int(system, f"/sys/devices/system/cpu/{d}/cpufreq/cpuinfo_max_freq")
            if v:
                max_khz.append(v)
    if info.get("CPU max MHz"):
        r.data["max_mhz"] = round(float(info["CPU max MHz"]))
    elif max_khz:
        r.data["max_mhz"] = round(max(max_khz) / 1000)
        r.sources.append("/sys/devices/system/cpu/*/cpufreq")
    if info.get("CPU min MHz"):
        r.data["min_mhz"] = round(float(info["CPU min MHz"]))
    base = re.search(r"@\s*([\d.]+)\s*GHz", model or "")
    if base:
        r.data["base_ghz"] = float(base.group(1))
    if "max_mhz" not in r.data:
        r.missing("maximum clock speed", "the CPU frequency driver doesn't expose it (common in VMs)")

    caches = {}
    for key in ("L1d", "L1i", "L2", "L3"):
        v = info.get(f"{key} cache") or info.get(key)
        if v:
            caches[key] = v
    if caches:
        r.data["caches"] = caches

    flags = set((info.get("Flags") or first.get("flags") or first.get("Features") or "").split())
    r.data["features"] = [label for flag, label in NOTABLE_FLAGS.items() if flag in flags]
    virt = info.get("Virtualization") or ("VT-x" if "vmx" in flags else "AMD-V" if "svm" in flags else None)
    r.data["hardware_virtualisation"] = virt or ("not exposed" if "hypervisor" in flags else "not supported")
    if "hypervisor" in flags:
        r.notes.append("Running under a hypervisor: core counts and clocks are what the VM was given.")
    return r


# --------------------------------------------------------------------------- memory

def _memory_slots(records: list) -> tuple[list[dict], Optional[int]]:
    arrays = [rec for rec in records if rec.type == 16 and rec.props.get("Use", "System Memory") == "System Memory"]
    array_handles = {a.handle for a in arrays}
    max_capacity = sum(parse_size(a.props.get("Maximum Capacity")) or 0 for a in arrays) or None

    slots = []
    for rec in records:
        if rec.type != 17:
            continue
        if array_handles and rec.props.get("Array Handle") not in array_handles:
            continue  # cache/flash arrays, not system RAM
        size = parse_size(rec.props.get("Size"))
        mtype = _real_value(rec.props.get("Type"))
        ff = rec.props.get("Form Factor")
        where = f"{rec.props.get('Locator', '')} {rec.props.get('Bank Locator', '')}"
        soldered = (ff in ("Row Of Chips", "Chip") or bool(re.search(r"on-?board|soldered", where, re.I))
                    or (mtype or "").upper().startswith("LPDDR"))
        slot = {
            "locator": rec.props.get("Locator"),
            "installed": size is not None,
            "size_bytes": size,
            "type": mtype,
            "form_factor": _real_value(ff),
            "speed": _real_value(rec.props.get("Configured Memory Speed")) or _real_value(rec.props.get("Speed")),
            "rated_speed": _real_value(rec.props.get("Speed")),
            "manufacturer": _real_value(rec.props.get("Manufacturer")),
            "part_number": _real_value((rec.props.get("Part Number") or "").strip()),
            "soldered": soldered,
        }
        slots.append({k: v for k, v in slot.items() if v is not None})
    return slots, max_capacity


def memory_upgrade(slots: list[dict], max_capacity: Optional[int]) -> dict:
    """How much RAM could this machine take? Assumes the firmware's maximum is
    shared evenly across slots, which matches how boards are specified."""
    installed = sum(s.get("size_bytes", 0) for s in slots)
    soldered = sum(s.get("size_bytes", 0) for s in slots if s.get("soldered"))
    replaceable = [s for s in slots if not s.get("soldered")]
    free = [s for s in replaceable if not s.get("installed")]
    types = Counter(s.get("type") for s in slots if s.get("installed") and s.get("type"))
    forms = Counter(s.get("form_factor") for s in replaceable if s.get("installed") and s.get("form_factor"))
    speeds = Counter(s.get("rated_speed") for s in slots if s.get("installed") and s.get("rated_speed"))

    up: dict = {
        "installed_bytes": installed,
        "soldered_bytes": soldered,
        "upgradeable_slots": len(replaceable),
        "free_slots": len(free),
        "max_capacity_bytes": max_capacity,
        "notes": [],
    }
    if types:
        up["module_type"] = types.most_common(1)[0][0]
    if forms:
        up["form_factor"] = forms.most_common(1)[0][0]
    if speeds:
        up["speed"] = speeds.most_common(1)[0][0]

    if not replaceable:
        up["possible"] = False
        up["notes"].append("All memory is soldered to the board, so it cannot be upgraded.")
        return up
    if not max_capacity:
        up["possible"] = None
        up["notes"].append("The firmware doesn't report a maximum capacity; check the manufacturer's spec sheet.")
        return up

    per_slot = max_capacity // len(slots)
    if free:
        add = min(per_slot, max(0, max_capacity - installed))
        up["largest_module_for_free_slot_bytes"] = add
        up["max_total_by_adding_bytes"] = min(max_capacity, installed + add * len(free))
    up["max_total_by_replacing_bytes"] = min(max_capacity, soldered + per_slot * len(replaceable))
    up["possible"] = up["max_total_by_replacing_bytes"] > installed
    up["notes"].append("The maximum comes from the firmware's DMI table. It is usually right but can be "
                       "conservative or out of date, so confirm with the manufacturer's spec sheet before buying.")
    if free and up.get("module_type"):
        kind = " ".join(x for x in (up.get("module_type"), up.get("form_factor")) if x)
        up["notes"].append(f"Buy a {kind} module" + (f" rated {up['speed']} or faster" if up.get("speed") else "")
                           + "; a stick the same size as the existing one keeps dual-channel symmetric.")
    return up


def memory_info(system: System) -> Result:
    r = Result()
    mi = parsers.parse_meminfo(system.read("/proc/meminfo") or "")
    if mi:
        r.sources.append("/proc/meminfo")
    else:
        res = system.run("free")
        if res.ok:
            f = parsers.parse_free(res.stdout)
            mi = {"MemTotal": f.get("mem_total"), "MemAvailable": f.get("mem_available"),
                  "SwapTotal": f.get("swap_total"), "SwapFree": f.get("swap_free")}
            r.sources.append("free -b")
    if mi.get("MemTotal"):
        r.data["usable_bytes"] = mi["MemTotal"]
        r.data["available_bytes"] = mi.get("MemAvailable")
        r.data["used_percent"] = round(100 * (1 - (mi.get("MemAvailable") or 0) / mi["MemTotal"]), 1)
        r.data["swap_bytes"] = mi.get("SwapTotal")

    res = system.run("dmidecode_memory")
    if res.ok:
        slots, max_capacity = _memory_slots(parsers.parse_dmidecode(res.stdout))
        r.sources.append("dmidecode -t memory")
        if slots:
            r.data["installed_bytes"] = sum(s.get("size_bytes", 0) for s in slots)
            r.data["slots_total"] = len(slots)
            r.data["slots_used"] = sum(1 for s in slots if s.get("installed"))
            r.data["slots"] = slots
            r.data["max_capacity_bytes"] = max_capacity
            r.data["upgrade"] = memory_upgrade(slots, max_capacity)
        else:
            r.missing("memory slots", "the firmware's DMI table lists no memory devices (common in VMs)")
    else:
        r.missing_from("installed modules, slot count and maximum supported memory", res, "dmidecode_memory")

    if r.data.get("installed_bytes") and r.data.get("usable_bytes"):
        r.notes.append("Usable memory is a bit lower than installed because firmware and integrated graphics "
                       "reserve some.")
    return r


# --------------------------------------------------------------------------- storage

_SKIP_DISKS = re.compile(r"^(loop|ram|zram|sr|fd|dm-|md|nbd)")
_VIRTUAL_DISK = re.compile(r"virtual|qemu harddisk|vbox harddisk|virtio", re.I)


def _disk_kind(name: str, transport: Optional[str], rotational: Optional[bool],
               model: Optional[str] = None) -> tuple[str, str]:
    """-> (interface, media)"""
    if model and _VIRTUAL_DISK.search(model):
        # Hypervisors report virtual disks as rotational, which says nothing about the real drive.
        return "virtual", "virtual disk"
    if name.startswith("nvme") or transport == "nvme":
        return "NVMe", "SSD"
    if name.startswith("mmcblk") or transport == "mmc":
        return "eMMC/SD", "flash"
    iface = {"sata": "SATA", "ata": "SATA", "usb": "USB", "sas": "SAS", "virtio": "virtio", "scsi": "SCSI"}.get(
        transport or "", "virtio" if name.startswith("vd") else (transport.upper() if transport else "unknown"))
    media = "HDD" if rotational else "SSD" if rotational is False else "unknown"
    return iface, media


def _usage(system: System, mountpoint: Optional[str]) -> Optional[dict]:
    if not mountpoint or mountpoint.startswith("[") or mountpoint == "[SWAP]":
        return None
    u = system.disk_usage(mountpoint)
    if not u:
        return None
    total, used, free = u
    return {"total_bytes": total, "used_bytes": used, "free_bytes": free,
            "used_percent": round(100 * used / total, 1) if total else None}


def storage_info(system: System, device: Optional[str] = None) -> Result:
    r = Result()
    disks = []
    res = system.run("lsblk")
    if res.ok:
        r.sources.append("lsblk")
        for d in parsers.parse_lsblk_json(res.stdout):
            if d["type"] != "disk" or _SKIP_DISKS.match(d["name"] or ""):
                continue
            model = " ".join(x for x in (d["vendor"], d["model"]) if x) or None
            iface, media = _disk_kind(d["name"], d["transport"], d["rotational"], model)
            parts = [{"name": c["name"], "size_bytes": c["size"], "filesystem": c["fstype"],
                      "mountpoint": c["mountpoint"], "usage": _usage(system, c["mountpoint"])}
                     for c in d["children"]]
            if d["mountpoint"]:  # whole-disk filesystem, no partition table
                parts.insert(0, {"name": d["name"], "size_bytes": d["size"], "filesystem": d["fstype"],
                                 "mountpoint": d["mountpoint"], "usage": _usage(system, d["mountpoint"])})
            disks.append({"name": d["name"], "model": model,
                          "size_bytes": d["size"], "size": fmt_disk(d["size"]), "interface": iface, "media": media,
                          "removable": d["removable"], "partitions": parts})
    else:
        r.notes.append(f"lsblk unavailable ({res.message}); read disks from /sys/block instead")
        for name in system.listdir("/sys/block"):
            if _SKIP_DISKS.match(name):
                continue
            base = f"/sys/block/{name}"
            sectors = _read_int(system, f"{base}/size")
            rota = _read(system, f"{base}/queue/rotational")
            model = _read(system, f"{base}/device/model")
            iface, media = _disk_kind(name, None, None if rota is None else rota == "1", model)
            size = sectors * 512 if sectors is not None else None
            disks.append({"name": name, "model": model, "size_bytes": size,
                          "size": fmt_disk(size), "interface": iface, "media": media,
                          "removable": _read(system, f"{base}/removable") == "1", "partitions": []})
        if disks:
            r.sources.append("/sys/block")

    # lsblk's MOUNTPOINT column shows one mount per device, which may not be "/"
    # (bind mounts, WSL). /proc/mounts is authoritative for where root lives.
    root_dev = None
    for line in (system.read("/proc/mounts") or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "/" and parts[0].startswith("/dev/"):
            root_dev = parts[0][5:]
    for disk in disks:
        disk["is_nvme"] = disk["interface"] == "NVMe"
        names = {disk["name"], *(p["name"] for p in disk["partitions"])}
        if root_dev:
            # nvme0n1p3 / sda2 / mmcblk0p1 belong to nvme0n1 / sda / mmcblk0
            disk["holds_root_filesystem"] = (root_dev in names
                                             or re.fullmatch(rf"{re.escape(disk['name'])}p?\d+", root_dev) is not None)
        else:
            disk["holds_root_filesystem"] = any(p.get("mountpoint") == "/" for p in disk["partitions"])
        if disk["holds_root_filesystem"] and not any(p.get("mountpoint") == "/" for p in disk["partitions"]):
            # Make "/" visible even when lsblk showed another mount of the same device (or nothing).
            part = next((p for p in disk["partitions"] if p["name"] == root_dev), None)
            if part is None:
                part = {"name": root_dev, "size_bytes": None, "filesystem": None}
                disk["partitions"].append(part)
            part["mountpoint"], part["usage"] = "/", _usage(system, "/")
        for p in disk["partitions"]:
            if p.get("usage") is None:
                p.pop("usage", None)

    if device:
        match = [d for d in disks if d["name"] == device]
        if not match:
            r.data["error"] = f"no disk named {device!r}; disks here: {', '.join(d['name'] for d in disks) or 'none'}"
        disks = match
    r.data["disks"] = disks
    if not disks and not device:
        r.missing("disks", "no block devices are visible")
    if any(d["media"] == "virtual disk" for d in disks):
        r.notes.append("These are virtual disks provided by a hypervisor; the physical drive behind them "
                       "(NVMe, SATA, SSD or HDD) isn't visible from inside.")
    return r


def disk_health(system: System, device: str) -> Result:
    r = Result()
    res = system.run("smartctl", device)
    if not res.ok:
        if res.error == "invalid":
            r.data["error"] = res.message
        else:
            r.missing_from(f"SMART health of {device}", res, "smartctl")
        return r
    r.sources.append(f"smartctl -H -A -j /dev/{device}")
    try:
        r.data = parsers.parse_smartctl_json(res.stdout)
    except ValueError as exc:
        r.missing(f"SMART health of {device}", f"could not parse smartctl output: {exc}")
    r.data["device"] = device
    return r


# --------------------------------------------------------------------------- GPU / display

def _gpu_kind(vendor: str, name: str) -> str:
    v, n = vendor.lower(), name.lower()
    if "nvidia" in v or "nvidia" in n:
        return "discrete"
    if "intel" in v:
        return "discrete" if re.search(r"\barc\b", n) else "integrated"
    if "amd" in v or "ati" in v or "advanced micro" in v:
        if re.search(r"radeon (rx|pro)|navi \d|\brx \d", n):
            return "discrete"
        if re.search(r"renoir|cezanne|rembrandt|phoenix|barcelo|lucienne|picasso|raven|vega \d|radeon graphics|"
                     r"radeon 7\d0m|radeon 6\d0m|hawk point|strix", n):
            return "integrated"
    return "unknown"


_VENDOR_IDS = {"8086": "Intel", "10de": "NVIDIA", "1002": "AMD", "1414": "Microsoft", "15ad": "VMware",
               "1af4": "Red Hat (virtio)", "1234": "QEMU", "80ee": "VirtualBox"}
_SHORT_VENDOR = (("intel", "Intel"), ("nvidia", "NVIDIA"), ("advanced micro", "AMD"), ("amd", "AMD"), ("ati", "AMD"))


def _gpu_name(vendor: str, model: str) -> tuple[str, Optional[str]]:
    """'TigerLake-LP GT2 [Iris Xe Graphics]' from Intel -> ('Intel Iris Xe Graphics', 'TigerLake-LP GT2')."""
    short = next((s for k, s in _SHORT_VENDOR if k in vendor.lower()), vendor.split()[0] if vendor else "")
    m = re.search(r"\[([^\]]+)\]\s*$", model)
    marketing, chip = (m.group(1), model[: m.start()].strip() or None) if m else (model, None)
    name = marketing if not short or short.lower() in marketing.lower() else f"{short} {marketing}"
    return name, chip


def _pci_ids_db(system: System) -> str:
    for path in ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"):
        text = system.read(path, max_bytes=8_000_000)
        if text:
            return text
    return ""


def gpu_info(system: System) -> Result:
    r = Result()
    gpus: list[dict] = []
    res = system.run("lspci")
    if res.ok:
        r.sources.append("lspci")
        for dev in parsers.parse_lspci_vmm(res.stdout):
            if (dev["class_id"] or "").startswith("03"):
                gpus.append({"vendor": dev["vendor"], "model": dev["device"], "driver": dev["driver"],
                             "pci_slot": dev["slot"], "vendor_id": dev["vendor_id"]})
    else:
        db = None
        for card in system.listdir("/sys/class/drm"):
            if not re.fullmatch(r"card\d+", card):
                continue
            base = f"/sys/class/drm/{card}/device"
            vid = (_read(system, f"{base}/vendor") or "").replace("0x", "")
            did = (_read(system, f"{base}/device") or "").replace("0x", "")
            if not vid:
                continue
            uevent = system.read(f"{base}/uevent") or ""
            driver = re.search(r"^DRIVER=(.+)$", uevent, re.M)
            slot = re.search(r"^PCI_SLOT_NAME=(.+)$", uevent, re.M)
            if db is None:
                db = _pci_ids_db(system)
            vname, dname = parsers.lookup_pci_ids(db, vid, did) if db else (None, None)
            gpus.append({"vendor": vname or _VENDOR_IDS.get(vid, f"vendor {vid}"),
                         "model": dname or f"device {did}", "driver": driver.group(1) if driver else None,
                         "pci_slot": slot.group(1) if slot else None, "vendor_id": vid, "drm_card": card})
        if gpus:
            r.sources.append("/sys/class/drm")
            if not db:
                r.notes.append("lspci and the PCI ID database aren't installed, so GPU names are generic IDs "
                               f"(install {PACKAGES['lspci']} for proper names).")
            else:
                r.notes.append("lspci isn't installed; names come from the PCI ID database.")

    # Some sysfs-only details per card (AMD reports VRAM size there).
    for card in system.listdir("/sys/class/drm"):
        if re.fullmatch(r"card\d+", card):
            vram = _read_int(system, f"/sys/class/drm/{card}/device/mem_info_vram_total")
            slot = re.search(r"^PCI_SLOT_NAME=0000:(.+)$", system.read(f"/sys/class/drm/{card}/device/uevent") or "", re.M)
            for g in gpus:
                if vram and slot and g.get("pci_slot") in (slot.group(1), f"0000:{slot.group(1)}"):
                    g["vram_bytes"] = vram

    if system.has_command("nvidia_smi"):
        smi = system.run("nvidia_smi")
        if smi.ok:
            r.sources.append("nvidia-smi")
            nv = parsers.parse_nvidia_smi(smi.stdout)
            nv_gpus = [g for g in gpus if "nvidia" in (g.get("vendor") or "").lower()]
            for i, n in enumerate(nv):
                if i < len(nv_gpus):
                    target = nv_gpus[i]
                else:  # e.g. WSL: no PCI view, but nvidia-smi works
                    target = {"vendor": "NVIDIA", "vendor_id": "10de"}
                    gpus.append(target)
                target["model"] = n["name"]
                target["driver_version"] = n["driver_version"]
                if n["memory_mib"]:
                    target["vram_bytes"] = n["memory_mib"] * 1024 * 1024

    for g in gpus:
        g["kind"] = _gpu_kind(g.get("vendor") or "", g.get("model") or "")
        g["name"], chip = _gpu_name(g.get("vendor") or "", g.get("model") or "")
        if chip:
            g["chip"] = chip
        if g.get("vram_bytes"):
            g["vram"] = fmt_ram(g["vram_bytes"])
        g.pop("vendor_id", None)
    r.data["gpus"] = gpus
    if not gpus:
        r.missing("GPU", "no PCI display controller or DRM device is visible (normal in containers, "
                  "some VMs and WSL without GPU passthrough)")
    if any(g["kind"] == "integrated" for g in gpus):
        r.notes.append("Integrated graphics share system RAM instead of having dedicated video memory.")
    return r


def display_info(system: System) -> Result:
    r = Result()
    outputs = []
    for entry in system.listdir("/sys/class/drm"):
        m = re.fullmatch(r"card\d+-(.+)", entry)
        if not m:
            continue
        base = f"/sys/class/drm/{entry}"
        status = _read(system, f"{base}/status")
        modes = (system.read(f"{base}/modes") or "").split()
        connector = m.group(1)
        outputs.append({
            "connector": connector,
            "connected": status == "connected",
            "internal": bool(re.match(r"(eDP|LVDS|DSI)", connector)),
            "preferred_mode": modes[0] if modes else None,
        })
    if outputs:
        r.sources.append("/sys/class/drm")
    else:
        r.missing("display outputs", "no DRM connectors are visible (no GPU driver loaded, or running in a VM/WSL)")
    r.data["outputs"] = outputs
    r.data["connected"] = [o for o in outputs if o["connected"]]
    return r


# --------------------------------------------------------------------------- power, network, USB, thermals

def battery_info(system: System) -> Result:
    r = Result()
    batteries, adapters = [], []
    for name in system.listdir("/sys/class/power_supply"):
        base = f"/sys/class/power_supply/{name}"
        kind = _read(system, f"{base}/type")
        if kind == "Battery":
            if _read(system, f"{base}/scope") == "Device":
                continue  # a wireless mouse or keyboard, not the laptop battery
            full = _read_int(system, f"{base}/energy_full") or _read_int(system, f"{base}/charge_full")
            design = _read_int(system, f"{base}/energy_full_design") or _read_int(system, f"{base}/charge_full_design")
            power = _read_int(system, f"{base}/power_now")
            cycles = _read_int(system, f"{base}/cycle_count")
            model = _real_value(_read(system, f"{base}/model_name"))
            if model and "virtual" in model.lower():
                # e.g. WSL / Hyper-V: mirrors the host's charge level, but capacities are made up.
                full = design = None
                r.missing("battery health and cycle count",
                          "this is a virtual battery that only mirrors the host's charge level")
            b = {
                "name": name,
                "status": _read(system, f"{base}/status"),
                "charge_percent": _read_int(system, f"{base}/capacity"),
                "health_percent": round(100 * full / design) if full and design else None,
                "cycle_count": cycles if cycles and cycles > 0 else None,  # 0 / -1 mean "not reported"
                "technology": _real_value(_read(system, f"{base}/technology")),
                "manufacturer": _real_value(_read(system, f"{base}/manufacturer")),
                "model": model,
                "power_draw_watts": round(power / 1e6, 1) if power else None,
            }
            if design and _read_int(system, f"{base}/energy_full_design"):
                b["design_capacity_wh"] = round(design / 1e6, 1)
                b["full_capacity_wh"] = round(full / 1e6, 1) if full else None
            batteries.append({k: v for k, v in b.items() if v is not None})
        elif kind in ("Mains", "USB"):
            online = _read(system, f"{base}/online")
            if online is not None:
                adapters.append({"name": name, "type": kind, "plugged_in": online == "1"})
    r.data["batteries"] = batteries
    r.data["ac_adapters"] = adapters
    if batteries or adapters:
        r.sources.append("/sys/class/power_supply")
    if not batteries:
        r.notes.append("No battery is visible: this is a desktop, a VM, or the battery driver isn't loaded.")
    for b in batteries:
        if "health_percent" not in b and not r.unavailable:
            r.missing(f"{b['name']} health", "the battery driver doesn't report design capacity")
    return r


def network_info(system: System, include_mac: bool = False) -> Result:
    r = Result()
    ifaces = []
    for name in system.listdir("/sys/class/net"):
        base = f"/sys/class/net/{name}"
        typ = _read(system, f"{base}/type")
        has_device = system.exists(f"{base}/device")
        if name == "lo" or typ == "772":
            kind = "loopback"
        elif system.exists(f"{base}/wireless") or system.exists(f"{base}/phy80211"):
            kind = "wifi"
        elif not has_device:
            kind = "virtual"
        elif typ == "1":
            kind = "ethernet"
        else:
            kind = "other"
        if kind == "loopback":
            continue
        driver_link = system.readlink(f"{base}/device/driver") if has_device else None
        speed = _read_int(system, f"{base}/speed")
        iface = {"name": name, "kind": kind, "state": _read(system, f"{base}/operstate"),
                 "driver": os.path.basename(driver_link) if driver_link else None,
                 "link_speed_mbps": speed if speed and speed > 0 else None}
        if include_mac:
            iface["mac_address"] = _read(system, f"{base}/address")
        ifaces.append({k: v for k, v in iface.items() if v is not None})
    if ifaces:
        r.sources.append("/sys/class/net")
    r.data["interfaces"] = ifaces

    res = system.run("lspci")
    if res.ok:
        chips = [{"type": "wifi" if d["class_id"] == "0280" else "ethernet" if d["class_id"] == "0200" else d["class"],
                  "vendor": d["vendor"], "model": d["device"], "driver": d["driver"]}
                 for d in parsers.parse_lspci_vmm(res.stdout) if (d["class_id"] or "").startswith("02")]
        r.data["controllers"] = chips
        r.sources.append("lspci")
    elif res.error == "missing":
        r.notes.append("lspci isn't installed, so I can name drivers but not the network chips themselves.")
    bt = [d for d in system.listdir("/sys/class/bluetooth") if d.startswith("hci")]
    r.data["bluetooth"] = bool(bt)
    if bt:
        r.sources.append("/sys/class/bluetooth")
    return r


def usb_devices(system: System) -> Result:
    r = Result()
    devices = []
    res = system.run("lsusb")
    if res.ok:
        r.sources.append("lsusb")
        devices = [d for d in parsers.parse_lsusb(res.stdout) if d["vendor_id"] != "1d6b"]  # skip root hubs
    else:
        for d in system.listdir("/sys/bus/usb/devices"):
            if ":" in d or d.startswith("usb"):
                continue  # interfaces and root hubs
            base = f"/sys/bus/usb/devices/{d}"
            vid = _read(system, f"{base}/idVendor")
            if not vid or vid == "1d6b":
                continue
            name = " ".join(x for x in (_read(system, f"{base}/manufacturer"), _read(system, f"{base}/product")) if x)
            devices.append({"vendor_id": vid, "product_id": _read(system, f"{base}/idProduct"), "name": name or None})
        if devices:
            r.sources.append("/sys/bus/usb/devices")
        if res.error == "missing":
            r.notes.append("lsusb isn't installed; names come from the devices themselves and may be missing.")
    r.data["devices"] = devices
    return r


def temperatures(system: System) -> Result:
    r = Result()
    sensors = []
    for z in system.listdir("/sys/class/thermal"):
        if z.startswith("thermal_zone"):
            t = _read_int(system, f"/sys/class/thermal/{z}/temp")
            if t is not None and t > 0:
                sensors.append({"sensor": _read(system, f"/sys/class/thermal/{z}/type") or z,
                                "celsius": round(t / 1000, 1)})
    for h in system.listdir("/sys/class/hwmon"):
        base = f"/sys/class/hwmon/{h}"
        chip = _read(system, f"{base}/name") or h
        for f in system.listdir(base):
            m = re.fullmatch(r"temp(\d+)_input", f)
            if m:
                t = _read_int(system, f"{base}/{f}")
                if t is not None and t > 0:
                    label = _read(system, f"{base}/temp{m.group(1)}_label")
                    sensors.append({"sensor": f"{chip} {label}" if label else chip, "celsius": round(t / 1000, 1)})
    if sensors:
        r.sources.append("/sys/class/thermal, /sys/class/hwmon")
    else:
        r.missing("temperatures", "no thermal sensors are exposed (normal in VMs and WSL)")
    r.data["sensors"] = sensors
    return r


def capabilities(system: System) -> Result:
    """What this assistant can and can't see on this machine, and why."""
    r = Result()
    cmds = {}
    for cmd_id, spec in COMMANDS.items():
        if cmds.get(spec.binary) == "installed":
            continue  # several command ids can share one binary (dmidecode)
        cmds[spec.binary] = ("installed" if system.has_command(cmd_id)
                             else f"missing (package {PACKAGES.get(spec.binary, spec.binary)})")
    r.data["commands"] = cmds
    r.data["privilege"] = {
        "root": "running as root",
        "sudo": "passwordless sudo works for the read-only commands",
        "unknown": "will try `sudo -n` for root-only commands",
        "none": "no root access: memory slot details, serial numbers and SMART data are unavailable",
    }[system.privilege]
    virt = _virtualisation(system)
    if virt:
        r.data["virtualisation"] = virt
    r.sources.append("startup checks")
    return r


# --------------------------------------------------------------------------- registry

_NO_PARAMS = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS: list[Tool] = [
    Tool("get_system_overview",
         "Manufacturer, model, form factor (laptop/desktop), motherboard, BIOS version and date, OS, kernel, and "
         "whether this is a VM or WSL. Set include_serial only if the user asks for the serial number.",
         system_overview,
         {"type": "object", "properties": {"include_serial": {"type": "boolean",
                                                              "description": "Also look up the serial number."}},
          "additionalProperties": False}),
    Tool("get_cpu_info",
         "CPU model, vendor, architecture, physical cores, threads, max/base clock, caches, hybrid core layout, "
         "instruction set features (AVX2, AVX-512, AES) and hardware virtualisation support.",
         cpu_info),
    Tool("get_memory_info",
         "RAM: usable and available memory, installed modules with size/type/speed, total and free slots, the "
         "maximum the board supports, and a precomputed upgrade analysis (how much can be added). Use this for "
         "any question about RAM size, slots or upgrades.",
         memory_info, ttl=30),
    Tool("get_storage_info",
         "Disks and partitions: model, size, interface (NVMe/SATA/USB), SSD vs HDD, filesystems, mount points and "
         "free space. Optionally restrict to one disk by name (e.g. nvme0n1).",
         storage_info,
         {"type": "object", "properties": {"device": {"type": "string", "pattern": r"^[a-z0-9]{2,16}$",
                                                      "description": "Disk name such as nvme0n1 or sda."}},
          "additionalProperties": False},
         ttl=30),
    Tool("get_disk_health",
         "SMART health of one disk: pass/fail, wear (percentage used), power-on hours, temperature, media errors. "
         "Needs root.",
         disk_health,
         {"type": "object", "properties": {"device": {"type": "string", "pattern": r"^[a-z0-9]{2,16}$",
                                                      "description": "Disk name from get_storage_info, e.g. nvme0n1."}},
          "required": ["device"], "additionalProperties": False},
         ttl=300, available=lambda s: s.has_command("smartctl")),
    Tool("get_gpu_info",
         "Graphics: each GPU's vendor, model, integrated vs discrete, driver in use, and video memory where known.",
         gpu_info),
    Tool("get_display_info",
         "Display outputs: which connectors exist (HDMI, DisplayPort, internal panel), which are connected, and "
         "their preferred resolution.",
         display_info, ttl=60),
    Tool("get_battery_info",
         "Battery: charge %, charging status, health (current vs design capacity), cycle count, power draw, and "
         "whether the charger is plugged in.",
         battery_info, ttl=15),
    Tool("get_network_info",
         "Network: Wi-Fi and Ethernet interfaces, link state and speed, drivers, the network chips, and whether "
         "Bluetooth is present. Set include_mac only if the user asks for MAC addresses.",
         network_info,
         {"type": "object", "properties": {"include_mac": {"type": "boolean",
                                                           "description": "Include MAC addresses."}},
          "additionalProperties": False},
         ttl=30),
    Tool("get_usb_devices", "Devices currently connected over USB.", usb_devices, ttl=30),
    Tool("get_temperatures", "Current temperature sensor readings (CPU package, NVMe, etc.).", temperatures, ttl=5),
    Tool("get_capabilities",
         "What this assistant can and cannot inspect here: which system tools are installed, whether it has root, "
         "and whether it runs in a VM. Use it to explain why some information is unavailable.",
         capabilities),
]
