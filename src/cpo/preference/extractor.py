from __future__ import annotations

from cpo.tot.tree import ToTTree


def extract_preference_pairs(tree: ToTTree, success_path: list[str], question: str) -> list[dict]:
    pairs: list[dict] = []
    success_set = set(success_path)
    for node_id in success_path[1:]:
        node = tree.nodes[node_id]
        parent = tree.nodes[node.parent_id] if node.parent_id else None
        if parent is None:
            continue
        for sibling_id in parent.children:
            if sibling_id in success_set:
                continue
            sibling = tree.nodes[sibling_id]
            pairs.append(
                {
                    "input": question,
                    "prefix": parent.state,
                    "chosen": node.thought,
                    "rejected": sibling.thought,
                    "chosen_score": node.score,
                    "rejected_score": sibling.score,
                }
            )
    return pairs
