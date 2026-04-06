from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from cpo.train.lora_setup import build_lora_config


@dataclass
class DPOConfig:
    base_model: str
    output_dir: str
    run_name: str = "stage3_dpo_minimal"
    max_steps: int = 300
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 20
    beta: float = 0.1
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    warmup_steps: int = 20
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    bf16: bool = True
    report_to: str = "none"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_prompt_length: int = 512
    max_length: int = 1024


def _build_training_args(config: DPOConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        max_steps=config.max_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        remove_unused_columns=False,
        report_to=config.report_to,
    )


def train_dpo(config: DPOConfig, train_dataset: Dataset, eval_dataset: Dataset) -> dict:
    try:
        from trl import DPOTrainer
    except Exception as exc:  # pragma: no cover - import-time environment mismatch
        raise RuntimeError(
            "Failed to import TRL DPOTrainer. Check trl/torch compatibility in current env. "
            "Suggested fix: align torch and trl versions used by your project environment."
        ) from exc

    if not config.base_model.strip():
        raise ValueError("base_model is required for DPO training")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = build_lora_config(
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    training_args = _build_training_args(config)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=config.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=config.max_prompt_length,
        max_length=config.max_length,
    )
    train_result = trainer.train()
    trainer.save_model(config.output_dir)

    final_checkpoint = f"{config.output_dir}/final_checkpoint"
    trainer.model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)

    return {
        "status": "ok",
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "output_dir": config.output_dir,
        "final_checkpoint": final_checkpoint,
    }
