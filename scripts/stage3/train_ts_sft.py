from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cpo.train.lora_setup import build_lora_config


TASK_TO_SFT_DATA = {
    "gsm8k": "data/interim/success_paths/gsm8k_success_paths_sft.jsonl",  # GSM8K 的 TS-SFT 数据
    "strategyqa": "data/interim/success_paths/strategyqa_success_paths_sft.jsonl",  # StrategyQA 的 TS-SFT 数据
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
    "max_steps": 250,  # 训练总步数
    "eval_steps": 50,  # 每多少步做一次评估
    "save_steps": 100,  # 每多少步保存一次 checkpoint
    "logging_steps": 20,  # 每多少步打印一次训练日志
    "learning_rate": 3e-6,  # 学习率
    "train_batch_size": 1,  # 单卡训练 batch size
    "eval_batch_size": 1,  # 单卡评估 batch size
    "grad_accum": 8,  # 梯度累积步数
    "lora_r": 8,  # LoRA rank
    "lora_alpha": 16,  # LoRA alpha
    "lora_dropout": 0.05,  # LoRA dropout
    "max_seq_length": 1024,  # 最大序列长度
    "warmup_steps": 20,  # 学习率预热步数
    "lr_scheduler_type": "cosine",  # 学习率调度器类型
    "weight_decay": 0.0,  # 权重衰减
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
    # 允许命令行显式覆盖；否则按 task 走默认数据路径。
    chosen = data_path.strip() if data_path.strip() else TASK_TO_SFT_DATA[task]
    path = Path(chosen)
    if not path.exists():
        raise SystemExit(f"TS-SFT data not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"TS-SFT data is empty: {path}")
    return path.as_posix()


def _default_run_id(task: str, base_model: str) -> str:
    model_tag = _slug(Path(base_model).name or "model")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"stage3_tssft_{task}_{model_tag}_{ts}"


def _resolve_output_dir(task: str, base_model: str, run_id: str, output_dir: str) -> str:
    if output_dir.strip():
        return output_dir.strip()
    model_tag = _slug(Path(base_model).name or "model")
    # 目录名显式包含 tssft，方便与 cpo 产物区分。
    return (Path("outputs/checkpoints") / run_id / task / f"tssft_{task}_{model_tag}").as_posix()


def _resolve_run_name(task: str, base_model: str, run_name: str) -> str:
    if run_name.strip():
        return run_name.strip()
    model_tag = _slug(Path(base_model).name or "model")
    return f"tssft_{task}_{model_tag}"


def _load_sft_rows(path: str, max_samples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not response:
                # Backward compatibility for existing DPO-style input.
                response = str(row.get("chosen", "")).strip()
            if not prompt or not response:
                continue
            rows.append({"prompt": prompt, "response": response})
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _split_dataset(rows: list[dict], eval_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if not rows:
        raise ValueError("No valid rows for TS-SFT.")
    ds = Dataset.from_list(rows)
    if len(rows) < 2:
        return ds, ds
    ratio = min(max(eval_ratio, 0.01), 0.5)
    split = ds.train_test_split(test_size=ratio, seed=seed)
    return split["train"], split["test"]


def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    def _tokenize(batch: dict) -> dict:
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[list[int]] = []
        labels_list: list[list[int]] = []

        prompts = batch["prompt"]
        responses = batch["response"]

        for prompt, response in zip(prompts, responses):
            prefix = f"{prompt}\n"
            full_text = f"{prefix}{response}"

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            prefix_enc = tokenizer(
                prefix,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=False,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prefix_len = min(len(prefix_enc["input_ids"]), len(input_ids))

            # completion-only loss: prompt 部分 label=-100，仅 response 参与训练损失。
            labels = [-100] * prefix_len + input_ids[prefix_len:]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda row: any(token != -100 for token in row["labels"]))
    return tokenized


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_inputs = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]

        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        # 右侧补齐 label，补齐位置统一为 -100，避免影响 loss。
        padded_labels = [label + [-100] * (max_len - len(label)) for label in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 TS-SFT training from SFT rows")
    parser.add_argument("--print-config", action="store_true", help="Print effective config and exit")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], default=MINIMAL_CONFIG["task"])
    parser.add_argument(
        "--data-path",
        default="",
        help="SFT jsonl path with prompt/response (or prompt/chosen). Empty means task default path.",
    )
    parser.add_argument("--base-model", default=ADVANCED_CONFIG["base_model"], help="Base model path or HF model id")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Empty means outputs/checkpoints/<run-id>/<task>/tssft_<task>_<model>.",
    )
    parser.add_argument("--run-name", default="", help="Trainer run_name. Empty means auto tssft_<task>_<model>.")
    parser.add_argument("--run-id", default="", help="Optional run id used in default output directory.")
    parser.add_argument("--max-samples", type=int, default=ADVANCED_CONFIG["max_samples"])
    parser.add_argument("--eval-ratio", type=float, default=ADVANCED_CONFIG["eval_ratio"])
    parser.add_argument("--seed", type=int, default=ADVANCED_CONFIG["seed"])
    parser.add_argument("--max-steps", type=int, default=ADVANCED_CONFIG["max_steps"])
    parser.add_argument("--eval-steps", type=int, default=ADVANCED_CONFIG["eval_steps"])
    parser.add_argument("--save-steps", type=int, default=ADVANCED_CONFIG["save_steps"])
    parser.add_argument("--logging-steps", type=int, default=ADVANCED_CONFIG["logging_steps"])
    parser.add_argument("--learning-rate", type=float, default=ADVANCED_CONFIG["learning_rate"])
    parser.add_argument("--train-batch-size", type=int, default=ADVANCED_CONFIG["train_batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=ADVANCED_CONFIG["eval_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=ADVANCED_CONFIG["grad_accum"])
    parser.add_argument("--lora-r", type=int, default=ADVANCED_CONFIG["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=ADVANCED_CONFIG["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=ADVANCED_CONFIG["lora_dropout"])
    parser.add_argument("--max-seq-length", type=int, default=ADVANCED_CONFIG["max_seq_length"])
    parser.add_argument("--warmup-steps", type=int, default=ADVANCED_CONFIG["warmup_steps"])
    parser.add_argument("--lr-scheduler-type", default=ADVANCED_CONFIG["lr_scheduler_type"])
    parser.add_argument("--weight-decay", type=float, default=ADVANCED_CONFIG["weight_decay"])
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

    print(f"[tssft] task={args.task}")
    print(f"[tssft] run_id={run_id}")
    print(f"[tssft] data_path={data_path}")
    print(f"[tssft] output_dir={output_dir}")
    print(f"[tssft] run_name={run_name}")

    rows = _load_sft_rows(data_path, max_samples=args.max_samples)
    train_ds, eval_ds = _split_dataset(rows, eval_ratio=args.eval_ratio, seed=args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.print_trainable_parameters()

    ta_kwargs = {
        "output_dir": output_dir,
        "run_name": run_name,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_checkpointing": True,
        "bf16": torch.cuda.is_available(),
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": args.save_total_limit,
    }
    # 兼容 transformers 不同版本参数名（evaluation_strategy / eval_strategy）。
    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"

    training_args = TrainingArguments(**ta_kwargs)

    train_tok = _tokenize_dataset(train_ds, tokenizer, args.max_seq_length)
    eval_tok = _tokenize_dataset(eval_ds, tokenizer, args.max_seq_length)
    if len(train_tok) == 0:
        raise ValueError("No train rows left after completion-only tokenization/filtering.")
    if len(eval_tok) == 0:
        raise ValueError("No eval rows left after completion-only tokenization/filtering.")

    collator = CompletionOnlyDataCollator(tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(output_dir)

    # 额外导出统一命名的最终检查点，便于下游评测脚本直接消费。
    final_ckpt = f"{output_dir}/final_checkpoint"
    trainer.model.save_pretrained(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)

    result = {
        "status": "ok",
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "train_rows_tokenized": len(train_tok),
        "eval_rows_tokenized": len(eval_tok),
        "best_model_checkpoint": str(getattr(trainer.state, "best_model_checkpoint", "") or ""),
        "task": args.task,
        "run_id": run_id,
        "data_path": data_path,
        "run_name": run_name,
        "output_dir": output_dir,
        "final_checkpoint": final_ckpt,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
