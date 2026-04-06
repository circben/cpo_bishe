from __future__ import annotations


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace(".", " ").replace(":", " ").split())


def is_terminal_state(text: str, task: str, goal_text: str | None = None) -> bool:
    lower = text.lower()
    if task == "gsm8k":
        return "the answer is" in lower
    if task == "strategyqa":
        return lower.endswith("yes") or lower.endswith("no")
    return False
