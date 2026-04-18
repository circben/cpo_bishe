from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cpo.eval.metrics_gsm8k import exact_match
from cpo.llm.generation import GenerationRuntime, clear_runtime_cache, complete_text
from cpo.tot.sc_tot_search import ToTSearchConfig, backtrack_path, extract_final_answer, run_tot_search


_CANDIDATE_STEP_RE = re.compile(r"candidate_step_\d+", re.IGNORECASE)


def _parse_semver(v: str) -> tuple[int, int, int]:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", v)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _model_type_from_local_path(model_path: str) -> str:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return ""
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("model_type", "")).strip().lower()


def _preflight_transformers(base_model: str) -> None:
    model_type = _model_type_from_local_path(base_model)
    if model_type != "qwen3":
        return
    tf_ver = pkg_version("transformers")
    if _parse_semver(tf_ver) < (4, 51, 0):
        raise SystemExit(
            "Detected model_type=qwen3 but transformers is too old: "
            f"{tf_ver}. Please upgrade to >=4.51.0 in the running environment."
        )


def _extract_gsm8k_gold(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if match:
        return match.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    return nums[-1] if nums else answer.strip().lower()


def _extract_gsm8k_pred(text: str) -> str:
    lower = text.lower()
    if "the answer is" in lower:
        # Extract number immediately following each explicit answer phrase,
        # then prefer the final explicit answer in the response.
        matches = re.findall(r"the answer is\s*[:\-]?\s*([-+]?\d+(?:\.\d+)?)", lower)
        if matches:
            return matches[-1]
        tail = lower.rsplit("the answer is", 1)[1]
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
        return nums[0] if nums else tail.strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", lower)
    return nums[-1] if nums else lower.strip()


def _normalize_binary(text: str) -> str:
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


def _extract_pred(task: str, text: str) -> str:
    if task == "gsm8k":
        return _extract_gsm8k_pred(text)
    return _normalize_binary(text)


def _preview_text(text: str, max_len: int = 200) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def _is_placeholder_like(text: str) -> bool:
    s = str(text).strip().lower()
    if not s:
        return True
    if _CANDIDATE_STEP_RE.search(s):
        return True
    blocked = [
        "<expression>",
        "<number>",
        "candidate_step",
        "if you need to think",
        "write the next reasoning step",
        "write a reasoning step",
        "do not write anything else",
        "generate one concise next reasoning thought",
    ]
    return any(k in s for k in blocked)


def _node_pred(task: str, thought: str, state: str) -> str:
    # Stage2-style preference: terminal thought is usually more concise than full state.
    pred_thought = _extract_pred(task, thought)
    if pred_thought and not _is_placeholder_like(pred_thought):
        return pred_thought
    return _extract_pred(task, state)


def _select_tot_node(task: str, tree) -> tuple[Any, list[Any], bool]:
    all_nodes = [n for n in tree.nodes.values() if n.node_id != "root"]
    if not all_nodes:
        return tree.nodes["root"], [], False
    terminal_nodes = [n for n in all_nodes if n.is_terminal]

    def usable(node: Any) -> bool:
        if _is_placeholder_like(node.thought) and _is_placeholder_like(node.state):
            return False
        pred = _node_pred(task, node.thought, node.state)
        return bool(pred and not _is_placeholder_like(pred))

    usable_terminals = [n for n in terminal_nodes if usable(n)]
    if usable_terminals:
        return max(usable_terminals, key=lambda n: n.score), terminal_nodes, True

    if terminal_nodes:
        return max(terminal_nodes, key=lambda n: n.score), terminal_nodes, False

    usable_nodes = [n for n in all_nodes if usable(n)]
    if usable_nodes:
        return max(usable_nodes, key=lambda n: n.score), terminal_nodes, False

    return max(all_nodes, key=lambda n: n.score), terminal_nodes, False


def _is_correct(task: str, pred: str, gold: str) -> bool:
    if task == "gsm8k":
        return bool(exact_match(pred, gold))
    return _normalize_binary(pred) == _normalize_binary(gold)


def _prompt_cot(task: str, question: str) -> str:
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


def _load_eval_examples(task: str, split: str, max_samples: int, start_index: int) -> list[dict[str, Any]]:
    if task == "gsm8k":
        split_file = Path("data/raw/gsm8k") / split / f"{split}.jsonl"
        if split_file.exists():
            rows = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            ds = load_from_disk("data/raw/gsm8k")
            chosen_split = split if split in ds else "test"
            rows = [dict(ds[chosen_split][i]) for i in range(len(ds[chosen_split]))]
        end = min(start_index + max_samples, len(rows)) if max_samples > 0 else len(rows)
        return rows[start_index:end]

    split_file = Path("data/raw/strategyqa") / split / f"{split}.jsonl"
    if split_file.exists():
        rows = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        legacy_split_file = Path("data/raw/strategyqa/splits") / f"{split}.jsonl"
        if legacy_split_file.exists():
            rows = [json.loads(line) for line in legacy_split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            ds = load_from_disk("data/raw/strategyqa")
            chosen_split = split if split in ds else "train"
            rows = [dict(ds[chosen_split][i]) for i in range(len(ds[chosen_split]))]

    end = min(start_index + max_samples, len(rows)) if max_samples > 0 else len(rows)
    return rows[start_index:end]


def _row_to_qg(task: str, row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question", "")).strip()
    if task == "gsm8k":
        gold = _extract_gsm8k_gold(str(row.get("answer", "")))
    else:
        gold = _normalize_binary(str(row.get("answer", "")))
    return question, gold


def _should_log_progress(idx: int, total: int, log_interval: int) -> bool:
    # Always show first and last item to avoid silent runs on small sample sizes.
    if total <= 0:
        return False
    if idx == 0 or idx + 1 == total:
        return True
    return log_interval > 0 and (idx + 1) % log_interval == 0


def _run_cot(
    task: str,
    model: str,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    log_interval: int,
    method_tag: str,
) -> tuple[dict, list[dict[str, Any]]]:
    runtime = GenerationRuntime(model_name=model, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    details: list[dict[str, Any]] = []
    ok = 0
    t0 = time.perf_counter()
    for idx, row in enumerate(rows):
        q, gold = _row_to_qg(task, row)
        p0 = time.perf_counter()
        text = complete_text(prompt=_prompt_cot(task, q), runtime=runtime, do_sample=False)
        latency = time.perf_counter() - p0
        pred = _extract_pred(task, text)
        correct = _is_correct(task, pred, gold)
        ok += int(correct)
        details.append({"index": idx, "gold": gold, "prediction": pred, "correct": correct, "latency_sec": latency})
        if _should_log_progress(idx, len(rows), log_interval):
            cnt_avg = ok / (idx + 1)
            print(
                f"[{method_tag}] idx={idx + 1}/{len(rows)} cnt_avg={cnt_avg:.4f} "
                f"sample_latency_sec={latency:.2f}",
                flush=True,
            )
    total_latency = time.perf_counter() - t0
    summary = {
        "method": "cot",
        "model": model,
        "count": len(rows),
        "accuracy": (ok / len(rows)) if rows else 0.0,
        "avg_latency_sec": (total_latency / len(rows)) if rows else 0.0,
    }
    return summary, details


def _run_tot(
    task: str,
    model: str,
    rows: list[dict[str, Any]],
    max_depth: int,
    candidates_per_step: int,
    beam_width: int,
    sc_votes: int,
    sc_temperature: float,
    log_interval: int,
) -> tuple[dict, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    ok = 0
    t0 = time.perf_counter()
    search_cfg = ToTSearchConfig(
        max_depth=max_depth,
        candidates_per_step=candidates_per_step,
        beam_width=beam_width,
        model_name=model,
        strict_llm=True,
        sc_votes=sc_votes,
        sc_temperature=sc_temperature,
    )
    for idx, row in enumerate(rows):
        q, gold = _row_to_qg(task, row)
        p0 = time.perf_counter()
        result = run_tot_search(
            question=q,
            task=task,
            config=search_cfg,
        )
        best = result.best_node
        pred = extract_final_answer(task, best)
        best_is_usable = bool(pred)
        latency = time.perf_counter() - p0
        correct = _is_correct(task, pred, gold)
        ok += int(correct)
        path_nodes = backtrack_path(result.nodes, best.node_id)
        path_thoughts = [n.thought for n in path_nodes if n.node_id != "root"]
        details.append({
            "index": idx,
            "gold": gold,
            "prediction": pred,
            "correct": correct,
            "latency_sec": latency,
            "best_score": best.score,
            "best_is_terminal": best.is_terminal,
            "best_is_usable": best_is_usable,
            "terminal_node_count": len(result.terminal_nodes),
            "node_count": result.total_nodes,
            "best_node_id": best.node_id,
            "path_node_ids": [n.node_id for n in path_nodes],
            "path_thoughts_preview": [_preview_text(t, max_len=120) for t in path_thoughts],
            "best_thought_preview": _preview_text(best.thought),
            "best_state_preview": _preview_text(best.state),
        })
        if _should_log_progress(idx, len(rows), log_interval):
            cnt_avg = ok / (idx + 1)
            print(
                f"[tot] idx={idx + 1}/{len(rows)} cnt_avg={cnt_avg:.4f} "
                f"sample_latency_sec={latency:.2f}",
                flush=True,
            )
    total_latency = time.perf_counter() - t0
    summary = {
        "method": "tot",
        "model": model,
        "count": len(rows),
        "accuracy": (ok / len(rows)) if rows else 0.0,
        "avg_latency_sec": (total_latency / len(rows)) if rows else 0.0,
        "max_depth": max_depth,
        "width": candidates_per_step,
        "beam": beam_width,
        "sc_votes": sc_votes,
    }
    return summary, details


def _save_outputs(task: str, split: str, summaries: list[dict], detail_map: dict[str, list[dict]], output_prefix: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/eval/runs") / f"{output_prefix}_{stamp}" / task
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"{output_prefix}_{task}_{split}.json"
    csv_path = run_dir / f"{output_prefix}_{task}_{split}.csv"

    payload = {
        "task": task,
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summaries": summaries,
        "details": detail_map,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["method", "model", "count", "accuracy", "avg_latency_sec"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow({k: row.get(k, "") for k in fields})

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified baseline evaluation on one shared test set")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--base-model", required=True, help="Base model path for CoT/SC/ToT")
    parser.add_argument("--ts-sft-model", required=True, help="TS-SFT checkpoint path")
    parser.add_argument("--cpo-model", required=True, help="CPO checkpoint path")

    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--tot-max-depth", type=int, default=3)
    parser.add_argument("--tot-width", type=int, default=2)
    parser.add_argument("--tot-beam", type=int, default=2)
    parser.add_argument("--tot-max-samples", type=int, default=50, help="ToT runs on the first N rows from the shared slice; 0 means all")
    parser.add_argument("--tot-candidates-per-step", type=int, default=-1)
    parser.add_argument("--tot-beam-width", type=int, default=-1)
    parser.add_argument("--tot-sc-votes", type=int, default=5)
    parser.add_argument("--tot-sc-temperature", type=float, default=0.7)
    parser.add_argument("--tot-score-samples", type=int, default=1)
    parser.add_argument("--tot-workers", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1, help="Progress print interval in examples; 0 to disable")
    parser.add_argument("--tot-adaptive-retry", type=int, choices=[0, 1], default=1, help="deprecated")
    parser.add_argument("--tot-retry-max-depth", type=int, default=7, help="deprecated")
    parser.add_argument("--tot-retry-width", type=int, default=4, help="deprecated")
    parser.add_argument("--tot-retry-beam", type=int, default=3, help="deprecated")

    parser.add_argument("--output-prefix", default="baselines_all")
    args = parser.parse_args()

    rows = _load_eval_examples(args.task, args.split, args.max_samples, args.start_index)
    if not rows:
        raise SystemExit("No evaluation rows loaded.")
    _preflight_transformers(args.base_model)
    print(
        f"Using shared sample slice for all baselines: split={args.split}, "
        f"start_index={args.start_index}, count={len(rows)}"
    )

    summaries: list[dict] = []
    details: dict[str, list[dict]] = {}

    clear_runtime_cache()
    s_cot, d_cot = _run_cot(args.task, args.base_model, rows, args.max_new_tokens, args.log_interval, "cot")
    summaries.append(s_cot)
    details["cot"] = d_cot

    tot_rows = rows if args.tot_max_samples <= 0 else rows[: min(args.tot_max_samples, len(rows))]
    print(f"ToT subset size: {len(tot_rows)} / {len(rows)} (from the same shared slice)")
    tot_candidates_per_step = args.tot_candidates_per_step if args.tot_candidates_per_step > 0 else args.tot_width
    tot_beam_width = args.tot_beam_width if args.tot_beam_width > 0 else args.tot_beam
    s_tot, d_tot = _run_tot(
        args.task,
        args.base_model,
        tot_rows,
        args.tot_max_depth,
        tot_candidates_per_step,
        tot_beam_width,
        args.tot_sc_votes,
        args.tot_sc_temperature,
        args.log_interval,
    )
    s_tot["subset_of_count"] = len(rows)
    summaries.append(s_tot)
    details["tot"] = d_tot

    clear_runtime_cache()
    s_tssft, d_tssft = _run_cot(args.task, args.ts_sft_model, rows, args.max_new_tokens, args.log_interval, "ts_sft")
    s_tssft["method"] = "ts_sft"
    summaries.append(s_tssft)
    details["ts_sft"] = d_tssft

    clear_runtime_cache()
    s_cpo, d_cpo = _run_cot(args.task, args.cpo_model, rows, args.max_new_tokens, args.log_interval, "cpo")
    s_cpo["method"] = "cpo"
    summaries.append(s_cpo)
    details["cpo"] = d_cpo
    clear_runtime_cache()

    json_path, csv_path = _save_outputs(args.task, args.split, summaries, details, args.output_prefix)

    print(json.dumps({"task": args.task, "split": args.split, "count": len(rows), "summaries": summaries}, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
