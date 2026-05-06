from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpo.llm.generation import GenerationRuntime, complete_text

DEFAULT_MODEL_ENTRY = "outputs/checkpoints/gsm8k/stage3_cpo_20260405_183114"
TASK_MODEL_ENTRIES = {
    "gsm8k": "outputs/checkpoints/gsm8k/stage3_cpo_20260405_183114",
    "strategyqa": "outputs/checkpoints/strategyqa/stage3_cpo_20260421_102753",
}


def _resolve_checkpoint_entry(entry_path: str) -> Path:
    path = Path(entry_path)
    if not path.exists():
        raise SystemExit(f"Model entry path not found: {path}")

    if path.is_dir() and path.name == "final_checkpoint":
        return path

    direct = path / "final_checkpoint"
    if direct.exists() and direct.is_dir():
        return direct

    candidates = sorted(path.rglob("final_checkpoint"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return path
    return candidates[0]


def _build_baseline_prompt(task: str, question: str) -> str:
    if task == "gsm8k":
        return (
            "You are solving a grade-school math problem. Think step by step and "
            "end with: The answer is <number>.\n\n"
            f"Question: {question}\nAnswer:"
        )
    return (
        "You are solving a yes/no reasoning question. Think step by step and end with yes or no.\n\n"
        f"Question: {question}\nAnswer:"
    )


def _extract_gsm8k_pred(text: str) -> str:
    lower = text.lower()
    if "the answer is" in lower:
        matches = re.findall(r"the answer is\s*[:\-]?\s*([-+]?\d+(?:\.\d+)?)", lower)
        if matches:
            return matches[0]
        tail = lower.rsplit("the answer is", 1)[1]
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
        return nums[0] if nums else tail.strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", lower)
    return nums[0] if nums else lower.strip()


def _extract_pred(task: str, text: str) -> str:
    if task == "gsm8k":
        return _extract_gsm8k_pred(text)
    s = text.strip().lower()
    if s in {"1", "true", "yes"}:
        return "yes"
    if s in {"0", "false", "no"}:
        return "no"
    if "yes" in s:
        return "yes"
    if "no" in s:
        return "no"
    return s


def _first_answer_span(task: str, text: str) -> tuple[str, int]:
    raw = str(text)
    lower = raw.lower()

    if task == "gsm8k":
        m = re.search(r"the answer is\s*[:\-]?\s*([-+]?\d+(?:\.\d+)?)", lower)
        if m:
            return m.group(1), m.end()

        for m_sent in re.finditer(r"[^.!?。！？]*[.!?。！？]", raw, flags=re.S):
            sent = m_sent.group(0)
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", sent)
            if nums:
                return nums[0], m_sent.end()
        return "", 0

    m = re.search(r"(?:the answer is|answer\s*:?)\s*(yes|no)\b", lower)
    if m:
        return m.group(1), m.end()

    for m_sent in re.finditer(r"[^.!?。！？]*[.!?。！？]", raw, flags=re.S):
        sent_lower = m_sent.group(0).lower()
        if re.search(r"\byes\b", sent_lower):
            return "yes", m_sent.end()
        if re.search(r"\bno\b", sent_lower):
            return "no", m_sent.end()
    return "", 0


def _split_reasoning_steps(text: str) -> list[str]:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines

    merged = " ".join(str(text).split())
    if not merged:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", merged) if p.strip()]
    return parts if parts else [merged]


def _build_linear_nodes_from_text(task: str, text: str) -> list[dict[str, object]]:
    _, first_end = _first_answer_span(task, text)
    chain_text = text[:first_end].strip() if first_end > 0 else text
    steps = _split_reasoning_steps(chain_text)

    nodes: list[dict[str, object]] = [
        {
            "node_id": "root",
            "parent_id": None,
            "depth": 0,
            "score": None,
            "is_terminal": False,
            "state": "",
            "thought": "",
        }
    ]

    running_steps: list[str] = []
    parent_id = "root"
    for i, step in enumerate(steps, start=1):
        node_id = f"step{i}"
        running_steps.append(step)
        nodes.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "depth": i,
                "score": 1.0,
                "is_terminal": i == len(steps),
                "state": "\n".join(running_steps),
                "thought": step,
            }
        )
        parent_id = node_id

    if len(nodes) == 1:
        nodes.append(
            {
                "node_id": "step1",
                "parent_id": "root",
                "depth": 1,
                "score": 1.0,
                "is_terminal": True,
                "state": text,
                "thought": text,
            }
        )

    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage4 text inference runner (local model)")
    parser.add_argument("--input", required=True, help="Path to input JSON with {text, meta}")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL_ENTRY)
    parser.add_argument("--device", default="")
    parser.add_argument("--do-sample", type=int, choices=[0, 1], default=0, help="Kept for compatibility; ignored.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Kept for compatibility; ignored.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Kept for compatibility; ignored.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    text = str(payload.get("text", "")).strip()
    if not text:
        raise SystemExit("Input JSON missing 'text'")

    meta = payload.get("meta", {}) or {}
    task = str(meta.get("task", "")).strip().lower()
    model_entry = args.model.strip()
    if "--model" not in sys.argv and task in TASK_MODEL_ENTRIES:
        model_entry = TASK_MODEL_ENTRIES[task]

    model_path = _resolve_checkpoint_entry(model_entry).as_posix()

    runtime_kwargs = {
        "model_name": model_path,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "provider": "local",
        "allow_fallback": False,
    }
    if args.device.strip():
        runtime_kwargs["device"] = args.device.strip()
    runtime = GenerationRuntime(**runtime_kwargs)

    prompt = text
    if task in {"gsm8k", "strategyqa"}:
        prompt = _build_baseline_prompt(task, text)

    result = complete_text(
        prompt=prompt,
        runtime=runtime,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    prediction = _extract_pred(task, result)
    nodes = _build_linear_nodes_from_text(task, result)

    output_payload = {
        "task": task,
        "method": str(meta.get("baseline", "text_infer") or "text_infer").strip().lower(),
        "question": text,
        "input": text,
        "prompt": prompt,
        "meta": meta,
        "model": model_path,
        "prediction": prediction,
        "nodes": nodes,
        "result": {
            "prediction": prediction,
            "nodes": nodes,
        },
    }

    (output_dir / "result.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
