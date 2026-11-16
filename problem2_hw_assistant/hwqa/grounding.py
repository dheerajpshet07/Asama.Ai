"""A cheap after-the-fact check that hardware numbers in an answer actually
came from a tool.

It pulls figures with units out of the answer ("16 GB", "4.2 GHz", "87%") and
looks for each one among the values the tools returned during the session,
after normalising units (bytes -> GB both binary and decimal, MHz -> GHz,
...). Anything it can't find is flagged to the user as unverified. It is a
backstop for the system prompt, not a proof: it can't see a wrong word, only a
number nobody measured.
"""
from __future__ import annotations

import re
from typing import Iterable

_CLAIM = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(TB|GB|GiB|MB|MiB|GHz|MHz|W|Wh|%|°C|C\b|MT/s|cores?|threads?|slots?)",
    re.I,
)


def _numbers(obj, out: set) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        return
    if isinstance(obj, str):
        for m in re.finditer(r"\d+(?:\.\d+)?", obj):
            out.add(float(m.group(0)))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers(v, out)


def known_values(results: Iterable[dict]) -> set[float]:
    raw: set[float] = set()
    for r in results:
        _numbers(r.get("data", {}), raw)
    derived = set(raw)
    for v in raw:
        derived.add(float(round(v)))  # "37%" for 37.2, "39°C" for 38.85
        derived.add(round(v, 1))
        if v >= 1e6:  # looks like bytes
            for base in (1024 ** 3, 1e9, 1024 ** 4, 1e12, 1024 ** 2, 1e6):
                derived.add(round(v / base, 1))
                derived.add(float(round(v / base)))
        if 100 <= v <= 10000:  # MHz / MT/s
            derived.add(round(v / 1000, 1))
            derived.add(round(v / 1000, 2))
    return derived


def unverified_claims(answer: str, results: Iterable[dict]) -> list[str]:
    known = known_values(results)
    flagged = []
    for m in _CLAIM.finditer(answer):
        value = float(m.group(1))
        if value in known or round(value, 1) in known:
            continue
        flagged.append(m.group(0).strip())
    return flagged
