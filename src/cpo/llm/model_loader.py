from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
