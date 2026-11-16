"""Turn raw command output into plain data. Pure functions, no I/O, so they
can be tested against recorded output from different distros and versions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .units import KiB

_ID_SUFFIX = re.compile(r"\s*\[([0-9a-fA-F]{4})\]\s*$")


# --------------------------------------------------------------------------- lscpu

def parse_lscpu(text: str) -> dict:
    """`lscpu` "Key: value" lines. Newer versions indent sub-keys and add
    section headers with empty values; both are handled. If a key repeats
    (big.LITTLE / hybrid CPUs list one block per core type) the first value
    wins and all values are kept under '<key>[]'."""
    out: dict = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        out.setdefault(key, value)
        out.setdefault(f"{key}[]", []).append(value)
    return out


def parse_cpuinfo(text: str) -> list[dict]:
    blocks, cur = [], {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = {}
            continue
        key, _, value = line.partition(":")
        cur[key.strip()] = value.strip()
    if cur:
        blocks.append(cur)
    return [b for b in blocks if "processor" in b]


# --------------------------------------------------------------------------- memory

def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo -> bytes. Values are in kB (really KiB)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+(?:\(\w+\))?):\s+(\d+)(?:\s*kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * (KiB if "kB" in line else 1)
    return out


def parse_free(text: str) -> dict[str, int]:
    """`free -b` -> {'mem_total': ..., 'mem_available': ..., 'swap_total': ...}."""
    out = {}
    lines = text.splitlines()
    if not lines:
        return out
    headers = lines[0].split()
    for line in lines[1:]:
        parts = line.split()
        if not parts or not parts[0].endswith(":"):
            continue
        row = parts[0].rstrip(":").lower()
        for h, v in zip(headers, parts[1:]):
            if v.isdigit():
                out[f"{row}_{h.replace('/', '_')}"] = int(v)
    return out


@dataclass
class DmiRecord:
    handle: str
    type: int
    title: str
    props: dict = field(default_factory=dict)


def parse_dmidecode(text: str) -> list[DmiRecord]:
    records: list[DmiRecord] = []
    cur: Optional[DmiRecord] = None
    last_key = None
    for line in text.splitlines():
        m = re.match(r"^Handle (0x[0-9A-Fa-f]+), DMI type (\d+)", line)
        if m:
            cur = DmiRecord(m.group(1), int(m.group(2)), "")
            records.append(cur)
            last_key = None
            continue
        if cur is None or not line.strip():
            continue
        if not line.startswith("\t"):
            if not cur.title:
                cur.title = line.strip()
            continue
        if line.startswith("\t\t") and last_key:
            # continuation of a list value (e.g. "Characteristics:")
            existing = cur.props.get(last_key)
            cur.props[last_key] = (existing if isinstance(existing, list) else []) + [line.strip()]
            continue
        key, _, value = line.strip().partition(":")
        cur.props[key.strip()] = value.strip()
        last_key = key.strip()
    return records


# --------------------------------------------------------------------------- storage

def _as_bool(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v in ("1", 1, "true"):
        return True
    if v in ("0", 0, "false"):
        return False
    return None


def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _clean(v) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def parse_lsblk_json(text: str) -> list[dict]:
    """`lsblk -J -b` across util-linux versions: older ones emit every value as
    a string ("rota": "0", "size": "512110190592"), newer ones use real types,
    and 2.37+ may give "mountpoints" as a list."""
    data = json.loads(text)

    def norm(dev: dict) -> dict:
        mp = dev.get("mountpoint")
        if mp is None and isinstance(dev.get("mountpoints"), list):
            mps = [m for m in dev["mountpoints"] if m]
            mp = mps[0] if mps else None
        return {
            "name": dev.get("name"),
            "type": dev.get("type"),
            "size": _as_int(dev.get("size")),
            "model": _clean(dev.get("model")),
            "vendor": _clean(dev.get("vendor")),
            "rotational": _as_bool(dev.get("rota")),
            "transport": _clean(dev.get("tran")),
            "removable": _as_bool(dev.get("rm")),
            "mountpoint": _clean(mp),
            "fstype": _clean(dev.get("fstype")),
            "children": [norm(c) for c in dev.get("children", []) or []],
        }

    return [norm(d) for d in data.get("blockdevices", [])]


# --------------------------------------------------------------------------- PCI / USB / GPU

def _split_id(value: str) -> tuple[str, Optional[str]]:
    m = _ID_SUFFIX.search(value)
    if not m:
        return value.strip(), None
    return value[: m.start()].strip(), m.group(1).lower()


def parse_lspci_vmm(text: str) -> list[dict]:
    """`lspci -vmm -nn -k`: blank-line separated records of 'Key:\\tValue'."""
    devices = []
    for block in re.split(r"\n\s*\n", text.strip()):
        rec = {}
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                rec[key.strip()] = value.strip()
        if "Slot" not in rec:
            continue
        cls, cls_id = _split_id(rec.get("Class", ""))
        vendor, vendor_id = _split_id(rec.get("Vendor", ""))
        device, device_id = _split_id(rec.get("Device", ""))
        sub_vendor, _ = _split_id(rec.get("SVendor", ""))
        devices.append({
            "slot": rec["Slot"],
            "class": cls,
            "class_id": cls_id,
            "vendor": vendor,
            "vendor_id": vendor_id,
            "device": device,
            "device_id": device_id,
            "subsystem_vendor": sub_vendor or None,
            "driver": rec.get("Driver"),
            "revision": rec.get("Rev"),
        })
    return devices


def parse_lsusb(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = re.match(r"^Bus (\d+) Device (\d+): ID ([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\s*(.*)$", line.strip())
        if m:
            out.append({"bus": m.group(1), "device": m.group(2), "vendor_id": m.group(3).lower(),
                        "product_id": m.group(4).lower(), "name": m.group(5).strip() or None})
    return out


def parse_nvidia_smi(text: str) -> list[dict]:
    """`nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader,nounits`"""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        mem = _as_int(parts[1].split()[0]) if parts[1] else None
        gpus.append({"name": parts[0], "memory_mib": mem, "driver_version": parts[2],
                     "bus_id": parts[3] if len(parts) > 3 else None})
    return gpus


def lookup_pci_ids(db_text: str, vendor_id: str, device_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Find vendor/device names in a pci.ids database (hwdata format)."""
    vendor_name = device_name = None
    in_vendor = False
    vendor_id, device_id = vendor_id.lower(), (device_id or "").lower()
    for line in db_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith("\t"):
            if in_vendor:
                break
            if line[:4].lower() == vendor_id:
                vendor_name = line[4:].strip()
                in_vendor = True
            continue
        if in_vendor and device_id and not line.startswith("\t\t") and line[1:5].lower() == device_id:
            device_name = line[5:].strip()
            break
    return vendor_name, device_name


# --------------------------------------------------------------------------- misc

def parse_os_release(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.startswith("#"):
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def parse_smartctl_json(text: str) -> dict:
    """Summary of `smartctl -H -A -j`. Works for NVMe and SATA."""
    d = json.loads(text)
    out = {
        "model": d.get("model_name"),
        "passed": (d.get("smart_status") or {}).get("passed"),
        "power_on_hours": (d.get("power_on_time") or {}).get("hours"),
        "temperature_c": (d.get("temperature") or {}).get("current"),
    }
    nvme = d.get("nvme_smart_health_information_log")
    if nvme:
        out.update({
            "percentage_used": nvme.get("percentage_used"),
            "available_spare_pct": nvme.get("available_spare"),
            "media_errors": nvme.get("media_errors"),
            "unsafe_shutdowns": nvme.get("unsafe_shutdowns"),
            "data_written_tb": round(nvme["data_units_written"] * 512_000 / 1e12, 1)
            if nvme.get("data_units_written") is not None else None,
        })
    for attr in (d.get("ata_smart_attributes") or {}).get("table", []):
        if attr.get("name") in ("Reallocated_Sector_Ct", "Current_Pending_Sector", "Wear_Leveling_Count",
                                "Percent_Lifetime_Remain", "Offline_Uncorrectable"):
            out[attr["name"]] = (attr.get("raw") or {}).get("value")
    return {k: v for k, v in out.items() if v is not None}
