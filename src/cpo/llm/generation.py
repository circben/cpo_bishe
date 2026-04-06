from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cpo.llm.openai_compatible import OpenAICompatibleConfig, chat_completion


@dataclass
class GenerationRuntime:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    allow_fallback: bool = True
    provider: str = "local"
    api_key: str = ""
    api_base_url: str = "https://api.siliconflow.cn/v1"
    api_timeout: int = 120
    api_enable_thinking: bool | None = None
    api_max_retries: int = 3
    api_retry_backoff_sec: float = 1.0


_RUNTIME_CACHE: dict[str, tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


def _runtime_from_env(runtime: GenerationRuntime | None = None) -> GenerationRuntime:
    rt = runtime or GenerationRuntime()
    provider = os.getenv("LLM_PROVIDER", rt.provider)
    api_key = os.getenv("LLM_API_KEY", rt.api_key)
    api_base_url = os.getenv("LLM_BASE_URL", rt.api_base_url)
    api_timeout = int(os.getenv("LLM_TIMEOUT", str(rt.api_timeout)))
    api_max_retries = int(os.getenv("LLM_MAX_RETRIES", str(rt.api_max_retries)))
    api_retry_backoff_sec = float(os.getenv("LLM_RETRY_BACKOFF_SEC", str(rt.api_retry_backoff_sec)))
    enable_thinking_raw = os.getenv("LLM_ENABLE_THINKING")
    api_enable_thinking: bool | None
    if enable_thinking_raw is None:
        api_enable_thinking = rt.api_enable_thinking
    else:
        api_enable_thinking = enable_thinking_raw.strip().lower() in {"1", "true", "yes"}
    return GenerationRuntime(
        model_name=rt.model_name,
        device=rt.device,
        max_new_tokens=rt.max_new_tokens,
        temperature=rt.temperature,
        top_p=rt.top_p,
        allow_fallback=rt.allow_fallback,
        provider=provider,
        api_key=api_key,
        api_base_url=api_base_url,
        api_timeout=api_timeout,
        api_enable_thinking=api_enable_thinking,
        api_max_retries=api_max_retries,
        api_retry_backoff_sec=api_retry_backoff_sec,
    )


def _is_api_provider(runtime: GenerationRuntime) -> bool:
    return runtime.provider.lower() in {"openai_compatible", "api"}


def _get_runtime(runtime: GenerationRuntime) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    key = f"{runtime.model_name}|{runtime.device}"
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]

    # Prefer local cache first to avoid hub timeouts in unstable networks.
    try:
        tokenizer = AutoTokenizer.from_pretrained(runtime.model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(runtime.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(runtime.model_name)
        model = AutoModelForCausalLM.from_pretrained(runtime.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(runtime.device)
    model.eval()
    _RUNTIME_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def ensure_model_loaded(model_name: str, device: str | None = None) -> None:
    runtime = _runtime_from_env(
        GenerationRuntime(model_name=model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    if _is_api_provider(runtime):
        if not runtime.api_key:
            raise RuntimeError("LLM_PROVIDER is API mode but LLM_API_KEY is empty")
        return
    _get_runtime(runtime)


def _build_generation_prompt(task: str, question: str, state: str) -> str:
    if task == "gsm8k":
        return (
            "You are solving a grade-school math word problem.\n"
            f"Question: {question}\n"
            f"Reasoning so far:\n{state if state.strip() else '(none)'}\n"
            "Write exactly ONE next reasoning step. "
            "If you can finish, write: The answer is <number>."
        )
    return (
        f"Task: {task}\n"
        f"Question: {question}\n"
        f"Current reasoning state: {state}\n"
        "Generate ONE concise next reasoning thought. "
        "Do not output final answer unless it is certain."
    )


def _clean_candidate(text: str) -> str:
    line = " ".join(text.strip().split())
    if "\n" in text:
        line = text.strip().splitlines()[0].strip()
    return line[:400]


def _should_skip_inspection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "data_inspection_failed" in msg or "inappropriate content" in msg


def complete_text(
    *,
    prompt: str,
    runtime: GenerationRuntime,
    max_new_tokens: int | None = None,
    do_sample: bool = True,
) -> str:
    rt = _runtime_from_env(runtime)
    token_budget = max_new_tokens or rt.max_new_tokens
    if _is_api_provider(rt):
        cfg = OpenAICompatibleConfig(
            api_key=rt.api_key,
            base_url=rt.api_base_url,
            model=rt.model_name,
            timeout=rt.api_timeout,
            enable_thinking=rt.api_enable_thinking,
            max_retries=rt.api_max_retries,
            retry_backoff_sec=rt.api_retry_backoff_sec,
        )
        temp = rt.temperature if do_sample else 0.0
        return chat_completion(
            config=cfg,
            prompt=prompt,
            temperature=temp,
            top_p=rt.top_p,
            max_tokens=token_budget,
        )

    model, tokenizer = _get_runtime(rt)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    encoded = {k: v.to(rt.device) for k, v in encoded.items()}
    input_len = int(encoded["input_ids"].shape[1])
    kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": token_budget,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = rt.temperature
        kwargs["top_p"] = rt.top_p
    output = model.generate(**encoded, **kwargs)
    gen_tokens = output[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def generate_candidates(
    *,
    task: str,
    question: str,
    state: str,
    width: int = 5,
    runtime: GenerationRuntime | None = None,
) -> list[str]:
    rt = _runtime_from_env(runtime)
    skip_inspection_failed = os.getenv("LLM_SKIP_INSPECTION_FAILED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    inspection_blocked_count = 0
    try:
        prompt = _build_generation_prompt(task=task, question=question, state=state)
        thoughts: list[str] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = max(width * 3, width)
        while len(thoughts) < width and attempts < max_attempts:
            attempts += 1
            try:
                candidate = complete_text(prompt=prompt, runtime=rt, do_sample=True)
            except Exception as exc:
                if skip_inspection_failed and _should_skip_inspection_error(exc):
                    inspection_blocked_count += 1
                    continue
                raise
            candidate = _clean_candidate(candidate)
            if candidate and candidate not in seen:
                seen.add(candidate)
                thoughts.append(candidate)

        if thoughts:
            # Return only real unique candidates to avoid synthetic placeholder pollution.
            return thoughts
        if inspection_blocked_count > 0 and skip_inspection_failed:
            # All attempts may be blocked by content inspection; skip this node expansion gracefully.
            return []
    except Exception as exc:
        # Fall back to synthetic candidates when model/token/network is unavailable.
        if not rt.allow_fallback:
            raise RuntimeError(f"LLM generation failed for model={rt.model_name}") from exc

    return [f"{state} | candidate_step_{i + 1}" for i in range(width)]


def simple_generate_candidates(prefix: str, width: int = 5) -> list[str]:
    # Backward-compatible wrapper used by existing tests/callers.
    return generate_candidates(task="gsm8k", question=prefix, state=prefix, width=width)


def batch_generate(prefixes: Iterable[str], width: int = 5) -> dict[str, list[str]]:
    return {p: simple_generate_candidates(p, width=width) for p in prefixes}
