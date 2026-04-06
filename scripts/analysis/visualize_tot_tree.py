from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_label(text: str, max_len: int = 64) -> str:
    s = " ".join(str(text).split())
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s.replace('"', "'")


def build_mermaid(tree: dict[str, Any], success_node_ids: set[str] | None = None) -> str:
    success_node_ids = success_node_ids or set()
    lines: list[str] = ["graph TD"]

    # Node declarations
    for node_id, node in tree.items():
        depth = node.get("depth", 0)
        score = node.get("score", 0)
        thought = _clean_label(node.get("thought", ""))
        label = f"{node_id} | d={depth} s={score} | {thought}" if thought else f"{node_id} | d={depth} s={score}"
        lines.append(f'    {node_id}["{label}"]')

    # Edges
    for node_id, node in tree.items():
        for child in node.get("children", []):
            lines.append(f"    {node_id} --> {child}")

    # Styles
    if "root" in tree:
        lines.append("    style root fill:#e8f4ff,stroke:#1976d2,stroke-width:2px")

    for node_id, node in tree.items():
        if bool(node.get("is_terminal", False)):
            lines.append(f"    style {node_id} fill:#fff8e1,stroke:#f9a825,stroke-width:1.5px")

    for nid in sorted(success_node_ids):
        if nid in tree:
            lines.append(f"    style {nid} fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a ToT tree json file to Mermaid")
    parser.add_argument("--tree-json", required=True, help="Path to a tot_nodes/*.json tree file")
    parser.add_argument("--success-json", default="", help="Optional success_paths/*.json to highlight path_node_ids")
    parser.add_argument("--out", default="", help="Output .mmd file path (default: <tree>.mmd)")
    args = parser.parse_args()

    tree_path = Path(args.tree_json)
    tree = _load_json(tree_path)

    success_nodes: set[str] = set()
    if args.success_json:
        success_payload = _load_json(Path(args.success_json))
        success_nodes = {str(x) for x in success_payload.get("path_node_ids", [])}

    mermaid = build_mermaid(tree, success_nodes)

    out_path = Path(args.out) if args.out else tree_path.with_suffix(".mmd")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mermaid, encoding="utf-8")
    print(f"Saved mermaid diagram: {out_path}")


if __name__ == "__main__":
    main()
