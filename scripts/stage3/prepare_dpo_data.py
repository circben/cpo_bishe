from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_row(row: dict) -> dict | None:
    prompt = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not prompt or not chosen or not rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate stage-2 DPO jsonl files")
    parser.add_argument("--task", choices=["gsm8k", "strategyqa"], required=True)
    parser.add_argument("--group", required=True, help="Group folder under data/processed/dpo/raw_preference/<task>")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-root", default="data/processed/dpo/raw_preference")
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit output path (default: data/processed/dpo/raw_preference/<task>/<task>_<split>.jsonl)",
    )
    parser.add_argument("--min-rows", type=int, default=200, help="Warn when output rows below threshold")
    args = parser.parse_args()

    group_dir = Path(args.input_root) / args.task / args.group
    if not group_dir.exists():
        raise SystemExit(f"Group directory not found: {group_dir}")

    pattern = f"{args.task}_{args.split}*.jsonl"
    files = sorted(group_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched pattern '{pattern}' in {group_dir}")

    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []
    total_rows = 0
    invalid_rows = 0

    # Keep deterministic ordering while deduplicating on (prompt, chosen, rejected).
    for fp in files:
        for row in _iter_jsonl_rows(fp):
            total_rows += 1
            normalized = _normalize_row(row)
            if normalized is None:
                invalid_rows += 1
                continue
            key = (normalized["prompt"], normalized["chosen"], normalized["rejected"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)

    output = (
        Path(args.output)
        if args.output
        else Path("data/processed/dpo/raw_preference") / args.task / f"{args.task}_{args.split}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "task": args.task,
        "group": args.group,
        "split": args.split,
        "input_root": str(Path(args.input_root) / args.task / args.group).replace("\\", "/"),
        "input_files": len(files),
        "input_rows": total_rows,
        "invalid_rows": invalid_rows,
        "output_rows": len(merged),
        "output": str(output).replace("\\", "/"),
        "warning_below_min_rows": len(merged) < args.min_rows,
        "min_rows": args.min_rows,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
