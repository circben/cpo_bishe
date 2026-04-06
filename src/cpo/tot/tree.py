from __future__ import annotations

from dataclasses import asdict

from .node import ToTNode


class ToTTree:
    def __init__(self) -> None:
        self.nodes: dict[str, ToTNode] = {}

    def add_node(self, node: ToTNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.node_id)

    def to_dict(self) -> dict:
        return {k: asdict(v) for k, v in self.nodes.items()}
