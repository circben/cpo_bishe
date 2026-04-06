from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class Stage2Config:
    task: str
    split: str
    start_index: int
    total_samples: int
    batch_size: int
    max_depth: int
    width: int
    beam: int
    n_score_samples: int
    max_workers: int
    min_score: float
    max_sim_ratio: float
    strict_llm: bool
    offline: bool
    model_name: str
    storage_tag: str
    llm_provider: str
    llm_base_url: str
    llm_timeout: str
    llm_enable_thinking: str
    llm_max_retries: str
    llm_retry_backoff_sec: str


# ==========
# 日常参数区
# ==========
MINIMAL_CONFIG = {
    "task": "gsm8k",  # gsm8k | strategyqa
}


TASK_MINIMAL_PROFILES = {
    "gsm8k": {
        "split": "train",
        "start_index": 1340,
        "total_samples": 10,
        "batch_size": 10,
        "max_depth": 4,
        "width": 4,
        "beam": 4,
        "storage_tag": "qwen3-8b_444",
    },
    "strategyqa": {
        "split": "train",
        "start_index": 1320,
        "total_samples": 180,
        "batch_size": 10,
        "max_depth": 4,
        "width": 3,
        "beam": 3,
        "storage_tag": "qwen3-8b_433",
    },
}


# ==========
# 高级参数区
# ==========
ADVANCED_CONFIG = {
    "n_score_samples": 1,
    "max_workers": 1,
    "min_score": 3.0,
    "max_sim_ratio": 0.9,
    "strict_llm": True,
    "offline": False,
    "model_name": "qwen3-8b",
    "storage_tag": "qwen3-8b_444",
    "llm_provider": "openai_compatible",
    "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "llm_timeout": "180",
    "llm_enable_thinking": "false",
    "llm_max_retries": "4",
    "llm_retry_backoff_sec": "1.0",
}


def _build_config() -> Stage2Config:
    task = str(MINIMAL_CONFIG.get("task", "")).strip().lower()
    if task not in TASK_MINIMAL_PROFILES:
        allowed = ", ".join(sorted(TASK_MINIMAL_PROFILES.keys()))
        raise SystemExit(f"Unsupported task='{task}'. Allowed: {allowed}")

    merged = dict(ADVANCED_CONFIG)
    merged["task"] = task
    merged.update(TASK_MINIMAL_PROFILES[task])

    # Allow optional explicit overrides in MINIMAL_CONFIG while keeping one-key defaults.
    for key, value in MINIMAL_CONFIG.items():
        if key == "task":
            continue
        merged[key] = value
    return Stage2Config(**merged)


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("=" * 80)
    print("Running:", " ".join(cmd))
    started = time.perf_counter()
    completed = subprocess.run(cmd, check=False, env=env)
    elapsed_sec = time.perf_counter() - started
    elapsed_min = int(elapsed_sec // 60)
    elapsed_remain_sec = elapsed_sec - elapsed_min * 60
    print(
        f"Completed total_samples run in {elapsed_sec:.1f}s "
        f"({elapsed_min}m {elapsed_remain_sec:.1f}s)"
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _effective_env(cfg: Stage2Config) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["LLM_PROVIDER"] = cfg.llm_provider
    env["LLM_BASE_URL"] = cfg.llm_base_url
    env["LLM_TIMEOUT"] = cfg.llm_timeout
    env["LLM_ENABLE_THINKING"] = cfg.llm_enable_thinking
    env["LLM_MAX_RETRIES"] = cfg.llm_max_retries
    env["LLM_RETRY_BACKOFF_SEC"] = cfg.llm_retry_backoff_sec
    return env


def _validate_runtime_env(cfg: Stage2Config) -> None:
    provider = cfg.llm_provider.strip().lower()
    if provider in {"openai_compatible", "api"} and not os.getenv("LLM_API_KEY", "").strip():
        raise SystemExit("LLM_API_KEY is empty. Configure it in environment (recommended: conda activate.d).")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click stage-2 preference generation")
    parser.add_argument("--print-config", action="store_true", help="Print effective config and exit")
    args = parser.parse_args()

    cfg = _build_config()
    if args.print_config:
        print("[minimal]")
        for k, v in MINIMAL_CONFIG.items():
            print(f"{k}={v}")
        print("[advanced]")
        for k, v in ADVANCED_CONFIG.items():
            print(f"{k}={v}")
        print("[effective]")
        for k, v in asdict(cfg).items():
            print(f"{k}={v}")
        return

    _validate_runtime_env(cfg)

    py = sys.executable
    env = _effective_env(cfg)

    cmd = [
        py,
        "scripts/stage2/run_preference_batches.py",
        "--task",
        cfg.task,
        "--split",
        cfg.split,
        "--start-index",
        str(cfg.start_index),
        "--total-samples",
        str(cfg.total_samples),
        "--batch-size",
        str(cfg.batch_size),
        "--max-depth",
        str(cfg.max_depth),
        "--width",
        str(cfg.width),
        "--beam",
        str(cfg.beam),
        "--n-score-samples",
        str(cfg.n_score_samples),
        "--max-workers",
        str(cfg.max_workers),
        "--min-score",
        str(cfg.min_score),
        "--max-sim-ratio",
        str(cfg.max_sim_ratio),
        "--model-name",
        cfg.model_name,
        "--storage-tag",
        cfg.storage_tag,
    ]
    if cfg.strict_llm:
        cmd.append("--strict-llm")
    if cfg.offline:
        cmd.append("--offline")

    _run(cmd, env)


if __name__ == "__main__":
    main()
