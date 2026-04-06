from __future__ import annotations

from cpo.common.text_utils import normalize_answer


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))
