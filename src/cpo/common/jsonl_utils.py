from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result
