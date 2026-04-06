from __future__ import annotations

from cpo.tot.tree import ToTTree


def backtrack_path(tree: ToTTree, end_node_id: str) -> list[str]:
    path: list[str] = []
    cur = end_node_id
    while cur:
        path.append(cur)
        parent = tree.nodes[cur].parent_id
        if parent is None:
            break
        cur = parent
    return list(reversed(path))
