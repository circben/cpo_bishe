from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-2 preference jobs in indexed batches")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], default="gsm8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--total-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--n-score-samples", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--max-sim-ratio", type=float, default=0.9)
    parser.add_argument("--model-name", default="qwen3-8b")
    parser.add_argument("--storage-tag", default="")
    parser.add_argument("--strict-llm", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    remaining = args.total_samples
    current = args.start_index
    batch_no = 0

    while remaining > 0:
        batch_no += 1
        n = min(args.batch_size, remaining)
        end_index = current + n - 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_token = args.model_name.replace("/", "_")
        run_id = (
            f"{model_token}_{args.max_depth}{args.width}{args.beam}_"
            f"idx{current:05d}_{end_index:05d}_{stamp}"
        )

        cmd = [
            py,
            "scripts/stage2/run_preference_job.py",
            "--task",
            args.task,
            "--split",
            args.split,
            "--start-index",
            str(current),
            "--max-samples",
            str(n),
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
            "--storage-tag",
            args.storage_tag,
            "--run-id",
            run_id,
        ]
        if args.strict_llm:
            cmd.append("--strict-llm")
        if args.offline:
            cmd.append("--offline")

        print("=" * 80)
        print(f"[batch={batch_no}] sample_index_range={current}-{end_index} count={n}")
        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

        current += n
        remaining -= n

    print("=" * 80)
    print("All batches completed")


if __name__ == "__main__":
    main()
