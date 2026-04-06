from __future__ import annotations

from dataclasses import dataclass

from cpo.llm.generation import GenerationRuntime, generate_candidates
from cpo.llm.scoring import score_candidate
from cpo.tot.concurrency import parallel_map
from cpo.tot.node import ToTNode
from cpo.tot.termination import is_terminal_state
from cpo.tot.tree import ToTTree


@dataclass
class ToTSearchConfig:
    max_depth: int = 5
    width: int = 5
    beam: int = 5
    n_score_samples: int = 2
    max_workers: int = 4
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"


def run_bfs_tot(
    question: str,
    task: str,
    goal_text: str | None = None,
    max_depth: int = 5,
    width: int = 5,
    beam: int = 5,
    n_score_samples: int = 2,
    max_workers: int = 4,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    strict_llm: bool = False,
) -> ToTTree:
    tree = ToTTree()
    root = ToTNode(node_id="root", parent_id=None, depth=0, state="", thought="", score=0.0)
    tree.add_node(root)

    frontier = [root]
    node_counter = 0
    runtime = GenerationRuntime(model_name=model_name, allow_fallback=not strict_llm)

    def expand_parent(parent: ToTNode, depth: int) -> list[tuple[str, int, str, float, bool]]:
        generated = generate_candidates(
            task=task,
            question=question,
            state=parent.state,
            width=width,
            runtime=runtime,
        )
        rows: list[tuple[str, int, str, float, bool]] = []
        for g in generated:
            score = score_candidate(
                task=task,
                question=question,
                state=parent.state,
                candidate=g,
                n_samples=n_score_samples,
                strict_llm=False,
                model_name=model_name,
            )
            terminal = is_terminal_state(g, task, goal_text=goal_text)
            rows.append((parent.node_id, depth, g, score, terminal))
        return rows

    for depth in range(1, max_depth + 1):
        candidates: list[ToTNode] = []
        expanded_groups = parallel_map(lambda p: expand_parent(p, depth), frontier, max_workers=max_workers)
        for group in expanded_groups:
            for parent_id, cur_depth, thought, score, terminal in group:
                node_counter += 1
                parent_state = tree.nodes[parent_id].state
                merged_state = thought if not parent_state else f"{parent_state}\n{thought}"
                node = ToTNode(
                    node_id=f"n{node_counter}",
                    parent_id=parent_id,
                    depth=cur_depth,
                    state=merged_state,
                    thought=thought,
                    score=score,
                    is_terminal=terminal,
                )
                tree.add_node(node)
                candidates.append(node)

        candidates.sort(key=lambda x: x.score, reverse=True)
        frontier = candidates[:beam]
        if any(n.is_terminal for n in frontier):
            break

    return tree
