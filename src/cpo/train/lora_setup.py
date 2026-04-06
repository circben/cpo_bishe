from __future__ import annotations

from peft import LoraConfig


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def build_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
