from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cpo.train.dpo_data import build_dpo_split, load_dpo_rows
from cpo.train.dpo_trainer import DPOConfig, train_dpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 minimal trainable DPO entry")
    parser.add_argument(
        "--data-path",
        default="data/processed/dpo/raw_preference/gsm8k/Qwen3-8B_444/gsm8k_train.jsonl",
        help="DPO jsonl path with prompt/chosen/rejected",
    )
    parser.add_argument("--base-model", required=True, help="Base model path or HF model id")
    parser.add_argument("--output-dir", default="outputs/checkpoints/dpo-minimal")
    parser.add_argument("--run-name", default="stage3_dpo_minimal")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--report-to", default="none", choices=["none", "wandb"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, dropped_rows = load_dpo_rows(data_path=args.data_path, max_samples=args.max_samples)
    split = build_dpo_split(rows=rows, eval_ratio=args.eval_ratio, seed=args.seed)

    config = DPOConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        report_to=args.report_to,
    )
    result = train_dpo(config=config, train_dataset=split.train, eval_dataset=split.eval)
    result["dropped_rows"] = dropped_rows
    result["total_rows"] = split.total_rows
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
