from __future__ import annotations


def filter_by_score(rows: list[dict], min_score: float = 3.0) -> list[dict]:
    return [
        r
        for r in rows
        if float(r.get("chosen_score", 0.0)) >= min_score and float(r.get("rejected_score", 0.0)) >= min_score
    ]
