from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_from_disk


def _normalize_label(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"1", "true", "yes"}:
        return "yes"
    if s in {"0", "false", "no"}:
        return "no"
    return s


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split StrategyQA train set into train/test subsets")
    parser.add_argument("--input-dir", default="data/raw/strategyqa", help="Path to load_from_disk StrategyQA dataset")
    parser.add_argument("--output-dir", default="data/splits/strategyqa", help="Output folder for split files")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio in (0,1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be between 0 and 1")

    ds = load_from_disk(args.input_dir)
    if "train" not in ds:
        raise SystemExit("StrategyQA dataset has no 'train' split")

    rows = [dict(ds["train"][i]) for i in range(len(ds["train"]))]
    for idx, row in enumerate(rows):
        row["source_index"] = idx

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = _normalize_label(row.get("answer", ""))
        grouped.setdefault(label, []).append(row)

    rng = random.Random(args.seed)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for _, bucket in grouped.items():
        rng.shuffle(bucket)
        cut = int(round(len(bucket) * args.train_ratio))
        cut = max(1, min(cut, len(bucket) - 1)) if len(bucket) > 1 else len(bucket)
        train_rows.extend(bucket[:cut])
        test_rows.extend(bucket[cut:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    out_dir = Path(args.output_dir)
    _write_json(out_dir / "train.json", train_rows)
    _write_json(out_dir / "test.json", test_rows)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "input_dir": args.input_dir,
        "output_dir": str(out_dir).replace("\\", "/"),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "total": len(rows),
        "train": len(train_rows),
        "test": len(test_rows),
        "labels": {
            label: {
                "total": len(bucket),
                "train": sum(1 for r in train_rows if _normalize_label(r.get("answer", "")) == label),
                "test": sum(1 for r in test_rows if _normalize_label(r.get("answer", "")) == label),
            }
            for label, bucket in grouped.items()
        },
    }
    _write_json(out_dir / "split_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
