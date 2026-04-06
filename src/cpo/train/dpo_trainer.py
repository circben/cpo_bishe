from __future__ import annotations

from dataclasses import dataclass
import inspect
import types

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
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 2
    bf16: bool = True
    report_to: str = "none"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_prompt_length: int = 512
    max_length: int = 1024


def _build_training_args(config: DPOConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": config.output_dir,
        "run_name": config.run_name,
        "max_steps": config.max_steps,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "logging_steps": config.logging_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "save_total_limit": config.save_total_limit,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": config.bf16,
        "remove_unused_columns": False,
        "report_to": config.report_to,
    }
    # Prefer TRL's DPOConfig when available (newer TRL expects DPO-specific args object).
    try:
        from trl import DPOConfig as TRLDPOConfig

        dpo_sig = inspect.signature(TRLDPOConfig.__init__)
        if "evaluation_strategy" in dpo_sig.parameters:
            kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in dpo_sig.parameters:
            kwargs["eval_strategy"] = "steps"
        return TRLDPOConfig(
            **kwargs,
            beta=config.beta,
            max_prompt_length=config.max_prompt_length,
            max_length=config.max_length,
        )
    except Exception:
        sig = inspect.signature(TrainingArguments.__init__)
        if "evaluation_strategy" in sig.parameters:
            kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in sig.parameters:
            kwargs["eval_strategy"] = "steps"
        return TrainingArguments(**kwargs)


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

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    dpo_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in dpo_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    # Compatibility shim: some transformers versions call get_batch_samples as
    # (epoch_iterator, num_batches, device), while this TRL DPOTrainer version
    # defines it as (model, batch). Dispatch by argument shape at runtime.
    get_batch_samples_sig = inspect.signature(trainer.get_batch_samples)
    if len(get_batch_samples_sig.parameters) == 2:
        original_get_batch_samples = trainer.get_batch_samples

        def _compat_get_batch_samples(self, *args, **kwargs):
            if len(args) >= 2 and hasattr(args[0], "generate") and isinstance(args[1], dict):
                return original_get_batch_samples(args[0], args[1])
            from transformers import Trainer

            return Trainer.get_batch_samples(self, *args, **kwargs)

        trainer.get_batch_samples = types.MethodType(_compat_get_batch_samples, trainer)

    # Compatibility shim: newer transformers may pass num_items_in_batch into
    # compute_loss, but some TRL versions do not accept this kwarg.
    compute_loss_sig = inspect.signature(trainer.compute_loss)
    if "num_items_in_batch" not in compute_loss_sig.parameters:
        original_compute_loss = trainer.compute_loss

        def _compat_compute_loss(self, *args, **kwargs):
            kwargs.pop("num_items_in_batch", None)
            return original_compute_loss(*args, **kwargs)

        trainer.compute_loss = types.MethodType(_compat_compute_loss, trainer)

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
        "best_model_checkpoint": str(getattr(trainer.state, "best_model_checkpoint", "") or ""),
        "output_dir": config.output_dir,
        "final_checkpoint": final_checkpoint,
    }
