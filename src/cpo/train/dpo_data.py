from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset

from cpo.common.jsonl_utils import read_jsonl


REQUIRED_DPO_COLUMNS = ("prompt", "chosen", "rejected")


@dataclass
class DPOSplit:
    train: Dataset
    eval: Dataset
    total_rows: int
    dropped_rows: int


def _normalize_row(row: dict) -> dict | None:
    normalized = {key: str(row.get(key, "")).strip() for key in REQUIRED_DPO_COLUMNS}
    if not all(normalized.values()):
        return None
    return normalized


def load_dpo_rows(data_path: str, max_samples: int = 0) -> tuple[list[dict], int]:
    rows = read_jsonl(data_path)
    normalized_rows: list[dict] = []
    dropped_rows = 0
    for row in rows:
        normalized = _normalize_row(row)
        if normalized is None:
            dropped_rows += 1
            continue
        normalized_rows.append(normalized)

    if max_samples > 0:
        normalized_rows = normalized_rows[:max_samples]
    return normalized_rows, dropped_rows


def build_dpo_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def build_dpo_split(
    rows: list[dict],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> DPOSplit:
    if not rows:
        raise ValueError("No valid DPO rows after preprocessing")
    if len(rows) < 2:
        ds = build_dpo_dataset(rows)
        return DPOSplit(train=ds, eval=ds, total_rows=len(rows), dropped_rows=0)

    ratio = min(max(eval_ratio, 0.01), 0.5)
    dataset = build_dpo_dataset(rows)
    split = dataset.train_test_split(test_size=ratio, seed=seed)
    return DPOSplit(
        train=split["train"],
        eval=split["test"],
        total_rows=len(rows),
        dropped_rows=0,
    )
