from __future__ import annotations

from difflib import SequenceMatcher


def filter_too_similar(rows: list[dict], max_ratio: float = 0.9) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        ratio = SequenceMatcher(None, r.get("chosen", ""), r.get("rejected", "")).ratio()
        if ratio < max_ratio:
            out.append(r)
    return out
