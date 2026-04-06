from __future__ import annotations


def resolve_storage_group(task: str, model_name: str, storage_tag: str) -> str:
    tag = storage_tag.strip().lower()
    if tag in {"qwen3-8b_433", "qwen3_8b_433", "qwen3-8b-433"}:
        return "Qwen3-8B_433"
    if tag in {"qwen3-8b_444", "qwen3_8b_444", "qwen3-8b-444"}:
        return "Qwen3-8B_444"
    if tag in {"qwen3-8b_555", "qwen3_8b_555", "qwen3-8b-555"}:
        return "Qwen3-8B_555"
    if tag in {"qwen2.5-3b", "qwen2_5_3b"}:
        return "Qwen2.5-3B"

    if tag in {"gsm8k", "strategyqa"}:
        return tag

    model_low = model_name.lower()
    if "qwen3-8b" in model_low:
        return "Qwen3-8B_555"
    if "qwen2.5-3b" in model_low:
        return "Qwen2.5-3B"

    if tag in {"cloud", "local"}:
        return task
    return task