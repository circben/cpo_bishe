from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    llm_mod = importlib.import_module("cpo.llm.openai_compatible")
    OpenAICompatibleConfig = llm_mod.OpenAICompatibleConfig
    chat_completion = llm_mod.chat_completion

    parser = argparse.ArgumentParser(description="Quick test for OpenAI-compatible Qwen3-8B API")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--enable-thinking", default=os.getenv("LLM_ENABLE_THINKING", ""))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("LLM_API_KEY is empty")

    enable_thinking = None
    if args.enable_thinking.strip():
        enable_thinking = args.enable_thinking.strip().lower() in {"1", "true", "yes"}
    cfg = OpenAICompatibleConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        enable_thinking=enable_thinking,
    )
    result = chat_completion(
        config=cfg,
        prompt=args.prompt,
        temperature=0.2,
        top_p=0.9,
        max_tokens=128,
    )
    print(result)


if __name__ == "__main__":
    main()
