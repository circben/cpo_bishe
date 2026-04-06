from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cpo.common.storage_group import resolve_storage_group


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_index = args.start_index + max(0, args.max_samples - 1)
    model_token = args.model_name.replace("/", "_")
    return (
        f"{model_token}_{args.max_depth}{args.width}{args.beam}_"
        f"idx{args.start_index:05d}_{end_index:05d}_{stamp}"
    )


def _resolve_storage_tag(explicit_tag: str, model_name: str, provider: str) -> str:
    tag = explicit_tag.strip().lower()
    if tag:
        return tag
    model_low = model_name.lower()
    if "qwen3-8b" in model_low:
        return "qwen3-8b_555"
    if "qwen2.5-3b" in model_low:
        return "qwen2.5-3b"
    return "cloud" if provider in {"openai_compatible", "api"} else "local"


def _summary_path(task: str, split: str, run_id: str, storage_group: str) -> Path:
    base = Path("outputs/eval/json") / task
    folder = base if storage_group.strip().lower() == task.strip().lower() else base / storage_group
    if split.strip().lower() == "test":
        folder = folder / "test"
    return folder / f"stage2_{task}_{split}_{run_id}.json"


def _record_path(task: str, split: str, storage_group: str, run_id: str) -> Path:
    base = Path("outputs/eval/json") / task
    folder = base if storage_group.strip().lower() == task.strip().lower() else base / storage_group
    if split.strip().lower() == "test":
        folder = folder / "test"
    return folder / f"run_record_{run_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one stage-2 preference job and record reproducible metadata"
    )
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--storage-tag", default="", help="Artifact tag, e.g. local/cloud")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--n-score-samples", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--max-sim-ratio", type=float, default=0.9)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--strict-llm", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Set HF offline env vars")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
    storage_tag = _resolve_storage_tag(args.storage_tag, args.model_name, provider)
    storage_group = resolve_storage_group(args.task, args.model_name, storage_tag)
    args.storage_tag = storage_tag
    args.storage_group = storage_group
    run_id = _build_run_id(args)
    py = sys.executable

    cmd = [
        py,
        "scripts/stage2/build_preferences.py",
        "--task",
        args.task,
        "--split",
        args.split,
        "--max-samples",
        str(args.max_samples),
        "--start-index",
        str(args.start_index),
        "--storage-tag",
        storage_tag,
        "--max-depth",
        str(args.max_depth),
        "--width",
        str(args.width),
        "--beam",
        str(args.beam),
        "--n-score-samples",
        str(args.n_score_samples),
        "--max-workers",
        str(args.max_workers),
        "--min-score",
        str(args.min_score),
        "--max-sim-ratio",
        str(args.max_sim_ratio),
        "--model-name",
        args.model_name,
        "--run-id",
        run_id,
    ]
    if args.strict_llm:
        cmd.append("--strict-llm")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    if args.offline:
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"

    start_ts = datetime.now().isoformat(timespec="seconds")
    start_t = time.perf_counter()

    print("=" * 80)
    print("Start:", start_ts)
    print("Run ID:", run_id)
    print("Command:", " ".join(cmd))

    completed = subprocess.run(cmd, check=False, env=env)

    end_t = time.perf_counter()
    duration_sec = round(end_t - start_t, 3)
    end_ts = datetime.now().isoformat(timespec="seconds")

    summary_path = _summary_path(args.task, args.split, run_id, storage_group)
    summary_exists = summary_path.exists()
    if summary_exists:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "available": False,
            "reason": "summary_file_not_found",
        }

    record = {
        "run_id": run_id,
        "task": args.task,
        "split": args.split,
        "storage_tag": storage_tag,
        "storage_group": storage_group,
        "start_index": args.start_index,
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_sec": duration_sec,
        "exit_code": completed.returncode,
        "python": py,
        "cwd": str(Path.cwd()),
        "offline": args.offline,
        "strict_llm": args.strict_llm,
        "command": cmd,
        "summary_path": str(summary_path),
        "summary_exists": summary_exists,
        "summary": summary,
    }

    rec_path = _record_path(args.task, args.split, storage_group, run_id)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("End:", end_ts)
    print("Duration(s):", duration_sec)
    print("Exit code:", completed.returncode)
    print("Summary JSON:", summary_path)
    print("Run record:", rec_path)

    if summary:
        print(
            "Metrics:",
            f"seen={summary.get('questions_seen')}",
            f"successful={summary.get('successful_questions')}",
            f"raw_pairs={summary.get('raw_pairs')}",
            f"filtered_pairs={summary.get('filtered_pairs')}",
        )

    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
