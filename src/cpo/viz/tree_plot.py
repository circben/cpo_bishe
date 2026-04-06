from __future__ import annotations

from pathlib import Path


def save_tree_text(tree_dict: dict, out_file: str) -> None:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for node_id, node in tree_dict.items():
            f.write(f"{node_id}: depth={node['depth']} score={node['score']}\\n")
