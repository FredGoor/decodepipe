from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


CM_PER_IN = 2.54


def cm_to_in(x_cm: float) -> float:
    return float(x_cm) / CM_PER_IN


def norm_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def split_semicolon_list(s: str) -> List[str]:
    s = norm_str(s)
    if not s:
        return []
    parts = re.split(r"[;,]+", s)
    return [p.strip() for p in parts if p.strip()]


def safe_sheetname(name: str, used: Set[str]) -> str:
    s = re.sub(r"[:\\/\?\*\[\]]", "_", str(name)).strip()
    s = s if s else "Sheet"
    s = s[:31]
    base = s
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = (base[:31 - len(suffix)] + suffix)[:31]
        i += 1
    used.add(s)
    return s


def wrap_label_no_break(s: str, width: int) -> str:
    s = norm_str(s)
    if not s or width <= 0:
        return s
    words = s.split()
    if len(words) <= 1:
        return s
    lines: List[str] = []
    line = ""
    for w in words:
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= width:
            line = f"{line} {w}"
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


def pair_type(decoding: str) -> str:
    d = norm_str(decoding)
    if d == "WC" or "WC" in d:
        return "WC"
    return "Wobble"


def canonicalize_trna_name(tok: str) -> str:
    s = norm_str(tok)
    if not s:
        return ""
    s = re.sub(r"[\(\)\[\]\{\}]", "", s).strip()
    m = re.match(r"^([A-Za-z]+)\s*(\d+)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s


def split_trna_field_names(field: str) -> List[str]:
    s = norm_str(field)
    if not s:
        return []
    toks = re.split(r"[;,/]+|\s+", s)
    out: List[str] = []
    seen = set()
    for t in toks:
        t = canonicalize_trna_name(t.strip())
        if not t:
            continue
        if t.lower() in {"none", "nan", "na", "n/a"}:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def group_bounds_from_labels(labels: Sequence[str], key_fn) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    bounds = []
    cur = key_fn(labels[0])
    start = 0
    for i in range(1, len(labels)):
        k = key_fn(labels[i])
        if k != cur:
            bounds.append((start, i - 1, str(cur)))
            start = i
            cur = k
    bounds.append((start, len(labels) - 1, str(cur)))
    return bounds


def group_bounds_from_groupnames(groupnames: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not groupnames:
        return []
    bounds = []
    cur = norm_str(groupnames[0])
    start = 0
    for i in range(1, len(groupnames)):
        g = norm_str(groupnames[i])
        if g != cur:
            bounds.append((start, i - 1, cur))
            start = i
            cur = g
    bounds.append((start, len(groupnames) - 1, cur))
    return bounds


def canonical_mod(s: str) -> str:
    t = norm_str(s).lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t
