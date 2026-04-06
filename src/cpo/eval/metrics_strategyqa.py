from __future__ import annotations


def binary_accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    ok = 0
    for p, g in zip(preds, golds):
        ok += int(p.strip().lower() == g.strip().lower())
    return ok / len(preds)
