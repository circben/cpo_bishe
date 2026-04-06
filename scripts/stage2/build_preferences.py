from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cpo.common.storage_group import resolve_storage_group
from cpo.common.jsonl_utils import read_jsonl, write_jsonl
from cpo.llm.generation import ensure_model_loaded
from cpo.preference.extractor import extract_preference_pairs
from cpo.preference.formatter import to_dpo_format
from cpo.preference.quality_filter import filter_by_score
from cpo.preference.similarity_filter import filter_too_similar
from cpo.tot.bfs_search import run_bfs_tot
from cpo.tot.serialize import save_tree_json
from cpo.tot.success_path import backtrack_path


_CANDIDATE_STEP_RE = re.compile(r"candidate_step_\d+", re.IGNORECASE)


def _normalize_binary(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"1", "true", "yes"}:
        return "yes"
    if s in {"0", "false", "no"}:
        return "no"
    return s


def _extract_gsm8k_gold(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if match:
        return match.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    return nums[-1] if nums else answer.strip().lower()


def _extract_prediction(task: str, text: str) -> str:
    lower = text.lower()
    if task == "gsm8k":
        if "the answer is" in lower:
            tail = lower.split("the answer is", 1)[1]
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
            return nums[0] if nums else tail.strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", lower)
        return nums[-1] if nums else lower.strip()
    if task == "strategyqa":
        return "yes" if "yes" in lower else ("no" if "no" in lower else lower.strip())
    return lower.strip()


def _is_correct(task: str, predicted_text: str, gold: str) -> bool:
    pred = _extract_prediction(task, predicted_text)
    if task == "gsm8k":
        return pred == gold
    if task == "strategyqa":
        return _normalize_binary(pred) == _normalize_binary(gold)
    return pred == gold


def _load_examples(task: str, split: str, max_samples: int, start_index: int = 0) -> list[dict[str, Any]]:
    split_jsonl = Path(f"data/raw/{task}") / split / f"{split}.jsonl"
    if split_jsonl.exists():
        rows = read_jsonl(split_jsonl)
        begin = max(0, min(start_index, len(rows)))
        end = min(begin + max_samples, len(rows))
        return rows[begin:end]

    if task == "strategyqa":
        legacy_split_jsonl = Path("data/raw/strategyqa/splits") / f"{split}.jsonl"
        if legacy_split_jsonl.exists():
            rows = read_jsonl(legacy_split_jsonl)
            begin = max(0, min(start_index, len(rows)))
            end = min(begin + max_samples, len(rows))
            return rows[begin:end]

    if task in {"gsm8k", "strategyqa"}:
        from datasets import load_from_disk

        ds = load_from_disk(f"data/raw/{task}")
        chosen_split = split if split in ds else ("train" if "train" in ds else next(iter(ds.keys())))
        rows = ds[chosen_split]
        begin = max(0, min(start_index, len(rows)))
        end = min(begin + max_samples, len(rows))
        out = [dict(rows[i]) for i in range(begin, end)]
        return out

    raise ValueError(f"Unsupported task: {task}")


def _resolve_storage_tag(explicit_tag: str, model_name: str) -> str:
    if explicit_tag.strip():
        return explicit_tag.strip().lower()
    model_low = model_name.lower()
    if "qwen3-8b" in model_low:
        return "qwen3-8b_555"
    if "qwen2.5-3b" in model_low:
        return "qwen2.5-3b"
    provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
    if provider in {"openai_compatible", "api"}:
        return "cloud"
    return "local"


def _task_group_dir(base: str, task: str, storage_group: str) -> Path:
    root = Path(base) / task
    if storage_group.strip().lower() == task.strip().lower():
        return root
    return root / storage_group


def _artifact_scope_dir(base: str, task: str, storage_group: str, split: str) -> Path:
    root = _task_group_dir(base, task, storage_group)
    if split.strip().lower() == "test":
        return root / "test"
    return root


def _to_question_gold(task: str, row: dict[str, Any]) -> tuple[str, str]:
    if task == "gsm8k":
        question = str(row.get("question", ""))
        gold = _extract_gsm8k_gold(str(row.get("answer", "")))
        return question, gold
    if task == "strategyqa":
        question = str(row.get("question", ""))
        gold = _normalize_binary(row.get("answer", ""))
        return question, gold
    raise ValueError(f"Unsupported task: {task}")


def _drop_placeholder_pairs(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for p in pairs:
        chosen = str(p.get("chosen", ""))
        rejected = str(p.get("rejected", ""))
        prefix = str(p.get("prefix", ""))
        text_blob = "\n".join([prefix, chosen, rejected])
        if _CANDIDATE_STEP_RE.search(text_blob):
            removed += 1
            continue
        kept.append(p)
    return kept, removed


def _select_success_node(task: str, tree, gold: str) -> str:
    terminals = [n for n in tree.nodes.values() if n.is_terminal]
    for node in terminals:
        if _is_correct(task, node.thought, gold):
            return node.node_id
    return max(tree.nodes.values(), key=lambda n: n.score).node_id


def _run_one_question(
    *,
    task: str,
    qid: str,
    question: str,
    gold: str,
    goal_text: str | None,
    max_depth: int,
    width: int,
    beam: int,
    n_score_samples: int,
    max_workers: int,
    model_name: str,
    strict_llm: bool,
    run_id: str,
    storage_group: str,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tree = run_bfs_tot(
        question=question,
        task=task,
        goal_text=goal_text,
        max_depth=max_depth,
        width=width,
        beam=beam,
        n_score_samples=n_score_samples,
        max_workers=max_workers,
        model_name=model_name,
        strict_llm=strict_llm,
    )

    tree_dir = _artifact_scope_dir("data/interim/tot_nodes", task, storage_group, split)
    tree_path = tree_dir / f"{qid}.json"
    save_tree_json(tree, tree_path)
    meta_path = tree_dir / f"{qid}.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "task": task,
        "qid": qid,
        "model_name": model_name,
        "question": question,
        "gold": gold,
        "goal_text": goal_text,
        "tree_file": str(tree_path),
        "node_count": len(tree.nodes),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    end_node_id = _select_success_node(task, tree, gold)
    success_path = backtrack_path(tree, end_node_id)
    is_correct = _is_correct(task, tree.nodes[end_node_id].state, gold)

    success_dir = _artifact_scope_dir("data/interim/success_paths", task, storage_group, split)
    success_dir.mkdir(parents=True, exist_ok=True)
    success_payload = {
        "run_id": run_id,
        "task": task,
        "qid": qid,
        "gold": gold,
        "end_node_id": end_node_id,
        "is_correct": is_correct,
        "path_node_ids": success_path,
        "path_thoughts": [tree.nodes[nid].thought for nid in success_path if nid != "root"],
        "final_state": tree.nodes[end_node_id].state,
    }
    (success_dir / f"{qid}.json").write_text(json.dumps(success_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pairs = extract_preference_pairs(tree, success_path, question)

    for p in pairs:
        p["task"] = task
        p["qid"] = qid
        p["gold"] = gold
        p["end_node_id"] = end_node_id
    stats = {
        "qid": qid,
        "node_count": len(tree.nodes),
        "is_correct": is_correct,
        "pair_count": len(pairs),
    }
    return pairs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-2 preference datasets from ToT search")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--n-score-samples", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--max-sim-ratio", type=float, default=0.9)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--strict-llm", action="store_true", help="Fail immediately if model cannot be loaded")
    parser.add_argument("--storage-tag", default="", help="Artifact storage tag, e.g. local or cloud")
    parser.add_argument("--run-id", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    storage_tag = _resolve_storage_tag(args.storage_tag, args.model_name)
    storage_group = resolve_storage_group(args.task, args.model_name, storage_tag)

    if args.strict_llm:
        ensure_model_loaded(args.model_name)

    examples = _load_examples(
        task=args.task,
        split=args.split,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    all_pairs: list[dict[str, Any]] = []
    run_stats: list[dict[str, Any]] = []

    for i, row in enumerate(examples):
        global_idx = args.start_index + i
        qid = f"{args.task}_{args.run_id}_{args.split}_{global_idx:05d}"
        question, gold = _to_question_gold(args.task, row)
        goal_text = None
        if not question.strip():
            continue
        print(f"[sample_index={global_idx}] start qid={qid}")
        pairs, stats = _run_one_question(
            task=args.task,
            qid=qid,
            question=question,
            gold=gold,
            goal_text=goal_text,
            max_depth=args.max_depth,
            width=args.width,
            beam=args.beam,
            n_score_samples=args.n_score_samples,
            max_workers=args.max_workers,
            model_name=args.model_name,
            strict_llm=args.strict_llm,
            run_id=args.run_id,
            storage_group=storage_group,
            split=args.split,
        )
        all_pairs.extend(pairs)
        run_stats.append(stats)
        print(
            f"[sample_index={global_idx}] done qid={qid} "
            f"correct={stats['is_correct']} nodes={stats['node_count']} pairs={stats['pair_count']}"
        )

    raw_pairs_before_placeholder_filter = len(all_pairs)
    all_pairs, removed_placeholder_pairs = _drop_placeholder_pairs(all_pairs)
    filtered = filter_by_score(all_pairs, min_score=args.min_score)
    filtered = filter_too_similar(filtered, max_ratio=args.max_sim_ratio)

    pref_dir = _task_group_dir("data/processed/preferences/raw_preference", args.task, storage_group)
    dpo_dir = _task_group_dir("data/processed/dpo/raw_preference", args.task, storage_group)
    pref_path = pref_dir / f"{args.task}_{args.split}.jsonl"
    dpo_path = dpo_dir / f"{args.task}_{args.split}.jsonl"
    pref_run_path = pref_dir / f"{args.task}_{args.split}_{args.run_id}.jsonl"
    dpo_run_path = dpo_dir / f"{args.task}_{args.split}_{args.run_id}.jsonl"

    wrote_outputs = bool(filtered)
    if wrote_outputs:
        dpo_rows = to_dpo_format(filtered)
        write_jsonl(pref_path, filtered)
        write_jsonl(dpo_path, dpo_rows)
        write_jsonl(pref_run_path, filtered)
        write_jsonl(dpo_run_path, dpo_rows)

    summary = {
        "run_id": args.run_id,
        "task": args.task,
        "split": args.split,
        "storage_tag": storage_tag,
        "storage_group": storage_group,
        "start_index": args.start_index,
        "questions_seen": len(run_stats),
        "successful_questions": sum(1 for s in run_stats if s["is_correct"]),
        "avg_nodes_per_question": (sum(s["node_count"] for s in run_stats) / len(run_stats)) if run_stats else 0.0,
        "raw_pairs": raw_pairs_before_placeholder_filter,
        "raw_pairs_after_placeholder_filter": len(all_pairs),
        "removed_placeholder_pairs": removed_placeholder_pairs,
        "filtered_pairs": len(filtered),
        "wrote_outputs": wrote_outputs,
        "model_name": args.model_name,
        "max_depth": args.max_depth,
        "width": args.width,
        "beam": args.beam,
    }
    summary_path = _artifact_scope_dir("outputs/eval/json", args.task, storage_group, args.split) / f"stage2_{args.task}_{args.split}_{args.run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Task: {args.task}")
    print(f"Raw pairs: {len(all_pairs)}")
    print(f"After filters: {len(filtered)}")
    print(f"Successful questions: {summary['successful_questions']} / {summary['questions_seen']}")
    if wrote_outputs:
        print(f"Preference JSONL: {pref_path}")
        print(f"DPO JSONL: {dpo_path}")
        print(f"Preference JSONL (run): {pref_run_path}")
        print(f"DPO JSONL (run): {dpo_run_path}")
    else:
        print("No filtered pairs, skip writing preference/DPO JSONL files to avoid empty artifacts.")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
