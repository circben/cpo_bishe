from __future__ import annotations


def to_dpo_format(rows: list[dict]) -> list[dict]:
    return [
        {
            "prompt": f"Question: {r['input']}\nHistory: {r['prefix']}",
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in rows
    ]
