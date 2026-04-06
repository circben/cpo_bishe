from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
#可视化ToT树

def _iter_success_files(task_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(task_dir.rglob("*.json"))
        if p.is_file() and not p.name.endswith(".meta.json")
    ]


def _build_meta_index(tot_task_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in sorted(tot_task_dir.rglob("*.meta.json")):
        qid = p.name[: -len(".meta.json")]
        if qid not in index:
            index[qid] = p
    return index


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compose_response(path_thoughts: list[str], final_state: str) -> str:
    thoughts = "\n".join(t.strip() for t in path_thoughts if str(t).strip()).strip()
    final = str(final_state).strip()
    if thoughts and final:
        return f"{thoughts}\n\n{final}"
    if thoughts:
        return thoughts
    return final


def export_task(success_root: Path, tot_root: Path, task: str, output_file: Path) -> dict[str, int]:
    task_success_dir = success_root / task
    task_tot_dir = tot_root / task

    if not task_success_dir.exists():
        raise FileNotFoundError(f"Missing success directory: {task_success_dir}")
    if not task_tot_dir.exists():
        raise FileNotFoundError(f"Missing tot_nodes directory: {task_tot_dir}")

    meta_index = _build_meta_index(task_tot_dir)
    success_files = _iter_success_files(task_success_dir)

    kept = 0
    skipped_incorrect = 0
    skipped_missing_meta = 0
    skipped_missing_question = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for sf in success_files:
            payload = _load_json(sf)
            if not bool(payload.get("is_correct", False)):
                skipped_incorrect += 1
                continue

            qid = str(payload.get("qid", "")).strip()
            if not qid:
                skipped_missing_meta += 1
                continue

            meta_path = meta_index.get(qid)
            if meta_path is None:
                skipped_missing_meta += 1
                continue

            meta = _load_json(meta_path)
            question = str(meta.get("question", "")).strip()
            if not question:
                skipped_missing_question += 1
                continue

            response = _compose_response(
                path_thoughts=[str(x) for x in payload.get("path_thoughts", [])],
                final_state=str(payload.get("final_state", "")),
            )

            row = {
                "task": task,
                "qid": qid,
                "prompt": question,
                "response": response,
                "is_correct": True,
                "source_success_file": str(sf),
                "source_meta_file": str(meta_path),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    return {
        "total_success_files": len(success_files),
        "kept": kept,
        "skipped_incorrect": skipped_incorrect,
        "skipped_missing_meta": skipped_missing_meta,
        "skipped_missing_question": skipped_missing_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export correct success_paths into SFT jsonl")
    parser.add_argument("--success-root", default="data/interim/success_paths")
    parser.add_argument("--tot-root", default="data/interim/tot_nodes")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "strategyqa"])
    parser.add_argument("--output-dir", default="data/interim/success_paths")
    args = parser.parse_args()

    success_root = Path(args.success_root)
    tot_root = Path(args.tot_root)
    output_dir = Path(args.output_dir)

    for task in args.tasks:
        out_file = output_dir / f"{task}_success_paths_sft.jsonl"
        stats = export_task(success_root, tot_root, task, out_file)
        print(
            f"[{task}] output={out_file} total={stats['total_success_files']} "
            f"kept={stats['kept']} incorrect={stats['skipped_incorrect']} "
            f"missing_meta={stats['skipped_missing_meta']} missing_question={stats['skipped_missing_question']}"
        )


if __name__ == "__main__":
    main()
