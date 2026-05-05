from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("=" * 90)
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-4 evaluation runner from stage-3 manifest")
    parser.add_argument("--manifest", required=True, help="Path to stage3 manifest json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--tot-max-depth", type=int, default=3)
    parser.add_argument("--tot-candidates-per-step", type=int, default=10)
    parser.add_argument("--tot-beam-width", type=int, default=5)
    parser.add_argument("--tot-sc-votes", type=int, default=5)
    parser.add_argument("--tot-sc-temperature", type=float, default=0.7)
    parser.add_argument("--tot-max-samples", type=int, default=50)
    parser.add_argument("--tot-score-samples", type=int, default=1)
    parser.add_argument("--tot-workers", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--run-cot", type=int, choices=[0, 1], default=1)
    parser.add_argument("--run-tot", type=int, choices=[0, 1], default=1)
    parser.add_argument("--run-ts-sft", type=int, choices=[0, 1], default=1)
    parser.add_argument("--run-cpo", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-nodes", type=int, choices=[0, 1], default=0)
    parser.add_argument("--nodes-output-root", default="outputs/eval/nodes")
    parser.add_argument("--nodes-run-id", default="")

    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], default="")
    parser.add_argument("--output-prefix", default="stage4_eval")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    task_runs = manifest.get("task_runs", [])
    if not task_runs:
        raise SystemExit("Manifest contains no task_runs entries")

    selected = []
    for row in task_runs:
        task = str(row.get("task", "")).strip()
        if args.task and task != args.task:
            continue
        selected.append(row)

    if not selected:
        raise SystemExit(f"No task run matched task={args.task!r} in manifest")

    py = sys.executable
    for row in selected:
        task = row["task"]
        base_model = row["base_model"]
        ts_sft_ckpt = row["ts_sft_final_checkpoint"]
        cpo_ckpt = row["cpo_final_checkpoint"]

        for p in (ts_sft_ckpt, cpo_ckpt):
            if not Path(p).exists():
                raise SystemExit(f"Checkpoint path not found for task {task}: {p}")

        _run(
            [
                py,
                "scripts/stage4/run_all_baselines.py",
                "--task",
                task,
                "--split",
                args.split,
                "--max-samples",
                str(args.max_samples),
                "--start-index",
                str(args.start_index),
                "--base-model",
                base_model,
                "--ts-sft-model",
                ts_sft_ckpt,
                "--cpo-model",
                cpo_ckpt,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--tot-max-depth",
                str(args.tot_max_depth),
                "--tot-candidates-per-step",
                str(args.tot_candidates_per_step),
                "--tot-beam-width",
                str(args.tot_beam_width),
                "--tot-sc-votes",
                str(args.tot_sc_votes),
                "--tot-sc-temperature",
                str(args.tot_sc_temperature),
                "--tot-max-samples",
                str(args.tot_max_samples),
                "--tot-score-samples",
                str(args.tot_score_samples),
                "--tot-workers",
                str(args.tot_workers),
                "--log-interval",
                str(args.log_interval),
                "--run-cot",
                str(args.run_cot),
                "--run-tot",
                str(args.run_tot),
                "--run-ts-sft",
                str(args.run_ts_sft),
                "--run-cpo",
                str(args.run_cpo),
                "--save-nodes",
                str(args.save_nodes),
                "--nodes-output-root",
                args.nodes_output_root,
                "--nodes-run-id",
                args.nodes_run_id,
                "--output-prefix",
                f"{args.output_prefix}_{task}",
            ]
        )


if __name__ == "__main__":
    main()
