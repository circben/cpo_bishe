from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(text: str) -> str:
    return normalize_whitespace(text).lower()
