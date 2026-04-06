from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToTNode:
    node_id: str
    parent_id: str | None
    depth: int
    state: str
    thought: str
    score: float = 0.0
    is_terminal: bool = False
    children: list[str] = field(default_factory=list)
