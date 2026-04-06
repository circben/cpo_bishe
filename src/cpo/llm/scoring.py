from __future__ import annotations

import re

from cpo.llm.generation import GenerationRuntime, complete_text


def _heuristic_score(text: str) -> float:
    # Simple bounded score in [1, 10] as robust fallback.
    score = min(10.0, max(1.0, 1.0 + (len(text) % 10)))
    return float(score)


def _parse_score(text: str) -> float | None:
    lowered = text.lower()
    if "impossible" in lowered:
        return 1.0
    if "likely" in lowered:
        return 8.0

    match = re.search(r"(10|[1-9])(\.\d+)?", lowered)
    if not match:
        return None
    value = float(match.group(0))
    return max(1.0, min(10.0, value))


def _build_score_prompt(task: str, question: str, state: str, candidate: str) -> str:
    return (
        "You are a strict reasoning evaluator.\n"
        f"Task: {task}\n"
        f"Question: {question}\n"
        f"Current state: {state}\n"
        f"Candidate next thought: {candidate}\n"
        "Score this candidate in [1, 10] by logical correctness and usefulness. "
        "Return only the score number."
    )


def _score_with_llm(task: str, question: str, state: str, candidate: str, runtime: GenerationRuntime) -> float | None:
    try:
        prompt = _build_score_prompt(task=task, question=question, state=state, candidate=candidate)
        completion = complete_text(
            prompt=prompt,
            runtime=runtime,
            max_new_tokens=8,
            do_sample=False,
        )
        return _parse_score(completion)
    except Exception:
        return None


def score_candidate(
    *,
    task: str,
    question: str,
    state: str,
    candidate: str,
    n_samples: int = 2,
    strict_llm: bool = False,
    model_name: str | None = None,
) -> float:
    runtime = GenerationRuntime(
        model_name=model_name or GenerationRuntime().model_name,
        allow_fallback=not strict_llm,
    )
    scores: list[float] = []
    for _ in range(max(1, n_samples)):
        llm_score = _score_with_llm(task=task, question=question, state=state, candidate=candidate, runtime=runtime)
        if llm_score is not None:
            scores.append(llm_score)
            continue
        if strict_llm:
            raise RuntimeError("LLM scoring failed in strict mode")
        scores.append(_heuristic_score(candidate))
    return float(sum(scores) / len(scores))


def simple_score(text: str) -> float:
    # Backward-compatible wrapper used by existing tests/callers.
    return score_candidate(task="gsm8k", question=text, state=text, candidate=text, n_samples=1)
