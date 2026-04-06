from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cpo.train.dpo_data import build_dpo_split, load_dpo_rows
from cpo.train.dpo_trainer import DPOConfig, train_dpo


TASK_TO_CPO_DATA = {
    "gsm8k": "data/processed/dpo/gsm8k/gsm8k_train.jsonl",  # GSM8K 的 CPO 偏好数据
    "strategyqa": "data/processed/dpo/strategyqa/strategyqa_train.jsonl",  # StrategyQA 的 CPO 偏好数据
}


# ==========
# 日常参数区
# ==========
MINIMAL_CONFIG = {
    "task": "gsm8k",  # 训练任务: gsm8k | strategyqa
}


# ==========
# 高级参数区
# ==========
ADVANCED_CONFIG = {
    "base_model": "/root/autodl-tmp/llm/Qwen3-8B",  # 基座模型路径
    "max_samples": 0,  # 样本上限，0 表示使用全部数据
    "eval_ratio": 0.1,  # 训练/验证划分比例
    "seed": 42,  # 随机种子
    "max_steps": 300,  # 训练总步数
    "eval_steps": 50,  # 每多少步做一次评估
    "save_steps": 100,  # 每多少步保存一次 checkpoint
    "logging_steps": 20,  # 每多少步打印一次训练日志
    "beta": 0.1,  # CPO(DPO) beta 参数
    "learning_rate": 2.5e-6,  # 学习率
    "train_batch_size": 1,  # 单卡训练 batch size
    "eval_batch_size": 1,  # 单卡评估 batch size
    "grad_accum": 8,  # 梯度累积步数
    "lora_r": 8,  # LoRA rank
    "lora_alpha": 16,  # LoRA alpha
    "lora_dropout": 0.05,  # LoRA dropout
    "max_prompt_length": 512,  # 输入 prompt 最大长度
    "max_length": 1024,  # 序列最大长度
    "load_best_model_at_end": True,  # 是否在训练结束时自动回载最优 checkpoint
    "metric_for_best_model": "eval_loss",  # 最优模型指标
    "greater_is_better": False,  # 指标是否越大越好（eval_loss 应为 False）
    "save_total_limit": 2,  # 最多保留多少个 checkpoint
    "report_to": "none",  # 日志上报目标: none | wandb
}


def _slug(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def _resolve_data_path(task: str, data_path: str) -> str:
    # 允许命令行显式覆盖；否则按 task 走默认偏好数据路径。
    chosen = data_path.strip() if data_path.strip() else TASK_TO_CPO_DATA[task]
    path = Path(chosen)
    if not path.exists():
        raise SystemExit(
            "CPO data not found: "
            f"{path}\n"
            "Prepare it first with scripts/stage3/prepare_dpo_data.py."
        )
    if path.stat().st_size == 0:
        raise SystemExit(f"CPO data is empty: {path}")
    return path.as_posix()


def _default_run_id(task: str, base_model: str) -> str:
    model_tag = _slug(Path(base_model).name or "model")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"stage3_cpo_{task}_{model_tag}_{ts}"


def _resolve_output_dir(task: str, base_model: str, run_id: str, output_dir: str) -> str:
    if output_dir.strip():
        return output_dir.strip()
    model_tag = _slug(Path(base_model).name or "model")
    # 目录名显式包含 cpo，方便与 tssft 产物区分。
    return (Path("outputs/checkpoints") / run_id / task / f"cpo_{task}_{model_tag}").as_posix()


def _resolve_run_name(task: str, base_model: str, run_name: str) -> str:
    if run_name.strip():
        return run_name.strip()
    model_tag = _slug(Path(base_model).name or "model")
    return f"cpo_{task}_{model_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 minimal trainable DPO entry")
    parser.add_argument("--print-config", action="store_true", help="Print effective config and exit")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], default=MINIMAL_CONFIG["task"])
    parser.add_argument(
        "--data-path",
        default="",
        help="DPO jsonl path with prompt/chosen/rejected. Empty means task default path.",
    )
    parser.add_argument("--base-model", default=ADVANCED_CONFIG["base_model"], help="Base model path or HF model id")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Empty means outputs/checkpoints/<run-id>/<task>/cpo_<task>_<model>.",
    )
    parser.add_argument("--run-name", default="", help="Trainer run_name. Empty means auto cpo_<task>_<model>.")
    parser.add_argument("--run-id", default="", help="Optional run id used in default output directory.")
    parser.add_argument("--max-samples", type=int, default=ADVANCED_CONFIG["max_samples"], help="0 means all rows")
    parser.add_argument("--eval-ratio", type=float, default=ADVANCED_CONFIG["eval_ratio"])
    parser.add_argument("--seed", type=int, default=ADVANCED_CONFIG["seed"])
    parser.add_argument("--max-steps", type=int, default=ADVANCED_CONFIG["max_steps"])
    parser.add_argument("--eval-steps", type=int, default=ADVANCED_CONFIG["eval_steps"])
    parser.add_argument("--save-steps", type=int, default=ADVANCED_CONFIG["save_steps"])
    parser.add_argument("--logging-steps", type=int, default=ADVANCED_CONFIG["logging_steps"])
    parser.add_argument("--beta", type=float, default=ADVANCED_CONFIG["beta"])
    parser.add_argument("--learning-rate", type=float, default=ADVANCED_CONFIG["learning_rate"])
    parser.add_argument("--train-batch-size", type=int, default=ADVANCED_CONFIG["train_batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=ADVANCED_CONFIG["eval_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=ADVANCED_CONFIG["grad_accum"])
    parser.add_argument("--lora-r", type=int, default=ADVANCED_CONFIG["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=ADVANCED_CONFIG["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=ADVANCED_CONFIG["lora_dropout"])
    parser.add_argument("--max-prompt-length", type=int, default=ADVANCED_CONFIG["max_prompt_length"])
    parser.add_argument("--max-length", type=int, default=ADVANCED_CONFIG["max_length"])
    parser.add_argument(
        "--load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_true",
        default=ADVANCED_CONFIG["load_best_model_at_end"],
        help="Enable loading best checkpoint at end.",
    )
    parser.add_argument(
        "--no-load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_false",
        help="Disable loading best checkpoint at end.",
    )
    parser.add_argument("--metric-for-best-model", default=ADVANCED_CONFIG["metric_for_best_model"])
    parser.add_argument(
        "--greater-is-better",
        dest="greater_is_better",
        action="store_true",
        default=ADVANCED_CONFIG["greater_is_better"],
        help="Set True if larger metric means better model.",
    )
    parser.add_argument(
        "--smaller-is-better",
        dest="greater_is_better",
        action="store_false",
        help="Set False if smaller metric means better model (e.g. eval_loss).",
    )
    parser.add_argument("--save-total-limit", type=int, default=ADVANCED_CONFIG["save_total_limit"])
    parser.add_argument("--report-to", default=ADVANCED_CONFIG["report_to"], choices=["none", "wandb"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_config:
        # 快速查看“脚本内默认值 + 命令行覆盖后的生效值”。
        print("[minimal]")
        for k, v in MINIMAL_CONFIG.items():
            print(f"{k}={v}")
        print("[advanced]")
        for k, v in ADVANCED_CONFIG.items():
            print(f"{k}={v}")
        print("[effective]")
        for k, v in vars(args).items():
            print(f"{k}={v}")
        return

    if not args.base_model.strip():
        raise SystemExit("base_model is empty. Set ADVANCED_CONFIG['base_model'] or pass --base-model.")

    run_id = args.run_id.strip() or _default_run_id(args.task, args.base_model)
    data_path = _resolve_data_path(args.task, args.data_path)
    output_dir = _resolve_output_dir(args.task, args.base_model, run_id, args.output_dir)
    run_name = _resolve_run_name(args.task, args.base_model, args.run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[cpo] task={args.task}")
    print(f"[cpo] run_id={run_id}")
    print(f"[cpo] data_path={data_path}")
    print(f"[cpo] output_dir={output_dir}")
    print(f"[cpo] run_name={run_name}")

    rows, dropped_rows = load_dpo_rows(data_path=data_path, max_samples=args.max_samples)
    # 先做 train/eval 切分，再交给 DPOTrainer 训练。
    split = build_dpo_split(rows=rows, eval_ratio=args.eval_ratio, seed=args.seed)

    config = DPOConfig(
        base_model=args.base_model,
        output_dir=output_dir,
        run_name=run_name,
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
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
    )
    result = train_dpo(config=config, train_dataset=split.train, eval_dataset=split.eval)
    # 补充数据统计与任务信息，便于记录实验可追溯性。
    result["dropped_rows"] = dropped_rows
    result["total_rows"] = split.total_rows
    result["task"] = args.task
    result["run_id"] = run_id
    result["data_path"] = data_path
    result["run_name"] = run_name
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
