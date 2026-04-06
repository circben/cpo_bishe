from __future__ import annotations

import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    prompt = "Let's think step by step: If I have 2 apples and get 3 more, how many apples?"
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=32)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
