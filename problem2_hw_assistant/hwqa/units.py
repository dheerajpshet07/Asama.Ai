"""Size parsing and formatting.

Two conventions on purpose, because that's what people expect to read:
  RAM   binary units shown as "GB" (a "16 GB" stick is 16 GiB, and DMI reports it that way)
  disks decimal units ("512 GB" on the box is 512 x 10^9 bytes)
"""
from __future__ import annotations

import re
from typing import Optional

KiB, MiB, GiB, TiB = 1024, 1024 ** 2, 1024 ** 3, 1024 ** 4

_SIZE = re.compile(r"^\s*([\d.]+)\s*([KMGT]i?B|bytes?|kB)\s*$", re.I)
_MULT = {"b": 1, "byte": 1, "bytes": 1, "kb": KiB, "kib": KiB, "mb": MiB, "mib": MiB, "gb": GiB, "gib": GiB,
         "tb": TiB, "tib": TiB}


def parse_size(text: Optional[str]) -> Optional[int]:
    """'16 GB' / '16384 MB' / '64 GiB' -> bytes, using binary multipliers (DMI
    and /proc conventions). Returns None for 'No Module Installed', 'Unknown', etc."""
    if not text:
        return None
    m = _SIZE.match(text)
    if not m:
        return None
    return int(float(m.group(1)) * _MULT[m.group(2).lower()])


def fmt_ram(n: Optional[int]) -> str:
    """RAM-style: binary, labelled GB. 17179869184 -> '16 GB'."""
    if n is None:
        return "unknown"
    if n >= GiB:
        v = n / GiB
        return f"{v:.0f} GB" if abs(v - round(v)) < 0.05 else f"{v:.1f} GB"
    return f"{n / MiB:.0f} MB"


def fmt_disk(n: Optional[int]) -> str:
    """Disk-style: decimal. 512110190592 -> '512 GB'."""
    if n is None:
        return "unknown"
    for unit, size in (("TB", 10 ** 12), ("GB", 10 ** 9), ("MB", 10 ** 6)):
        if n >= size:
            v = n / size
            return f"{v:.0f} {unit}" if v >= 100 or abs(v - round(v)) < 0.05 else f"{v:.1f} {unit}"
    return f"{n} bytes"
