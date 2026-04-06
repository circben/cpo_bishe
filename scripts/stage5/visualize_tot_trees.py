"""
Stage 5: ToT Tree Visualization
Visualizes the Tree of Thoughts search process for each problem.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_tot_tree(json_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ToT tree from JSON file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", {}), data.get("edges", [])


def create_tree_graph(nodes: dict[str, Any], edges: list[dict]) -> nx.DiGraph:
    """Create NetworkX directed graph from nodes and edges."""
    G = nx.DiGraph()

    for node_id, node_data in nodes.items():
        G.add_node(
            node_id,
            depth=node_data.get("depth", 0),
            score=node_data.get("score", 0),
            text=node_data.get("text", "")[:50] + "..." if len(node_data.get("text", "")) > 50 else node_data.get("text", ""),
            is_terminal=node_data.get("is_terminal", False),
            is_correct=node_data.get("is_correct", False),
        )

    for edge in edges:
        G.add_edge(edge.get("parent_id", ""), edge.get("child_id", ""))

    return G


def plot_tree(
    G: nx.DiGraph,
    save_path: str,
    question: str = "",
    title: str = "ToT Tree Visualization",
    figsize: tuple[int, int] = (16, 10),
) -> None:
    """Plot and save the ToT tree visualization."""
    fig, ax = plt.subplots(figsize=figsize)

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No tree data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    node_colors = []
    node_labels = {}
    for node_id in G.nodes:
        data = G.nodes[node_id]
        score = data.get("score", 0)
        is_correct = data.get("is_correct", False)
        is_terminal = data.get("is_terminal", False)

        if is_correct:
            node_colors.append("#2ecc71")
        elif is_terminal:
            node_colors.append("#e74c3c")
        elif score >= 7:
            node_colors.append("#3498db")
        elif score >= 5:
            node_colors.append("#f39c12")
        else:
            node_colors.append("#95a5a6")

        label = f"{node_id}\nScore: {score}\n{data.get('text', '')[:30]}"
        node_labels[node_id] = label

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels=node_labels,
        node_color=node_colors,
        node_size=2000,
        font_size=7,
        font_color="white",
        edge_color="#7f8c8d",
        arrows=True,
        arrowsize=15,
    )

    if question:
        ax.set_title(f"{title}\nQ: {question[:100]}...", fontsize=12, pad=20)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=12, label="Correct (Success Path)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=12, label="Terminal (Wrong Answer)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=12, label="High Score (>=7)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f39c12", markersize=12, label="Medium Score (5-7)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=12, label="Low Score (<5)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_tot_trees(
    tot_nodes_dir: str,
    output_dir: str,
    dataset: str = "gsm8k",
    model: str = "Qwen3-8B_444",
    max_trees: int = 10,
) -> dict[str, Any]:
    """Visualize multiple ToT trees."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tree_files = sorted(Path(tot_nodes_dir).glob("*.json"))[:max_trees]

    results = {"visualized": 0, "failed": 0, "output_dir": output_dir}

    for i, tree_file in enumerate(tree_files):
        try:
            nodes, edges = load_tot_tree(str(tree_file))
            if not nodes:
                results["failed"] += 1
                continue

            question = nodes.get("root", {}).get("text", "")[:80] if "root" in nodes else ""
            save_path = os.path.join(output_dir, f"tree_{i:03d}_{tree_file.stem}.png")

            G = create_tree_graph(nodes, edges)
            plot_tree(G, save_path, question=question, title=f"Problem #{i}")

            results["visualized"] += 1
            print(f"  [{i+1}/{len(tree_files)}] Saved: {tree_file.name} -> {save_path}")

        except Exception as e:
            results["failed"] += 1
            print(f"  [!] Failed: {tree_file.name} - {e}")

    return results


def generate_html_viewer(output_dir: str, json_dir: str) -> str:
    """Generate interactive HTML viewer for ToT trees."""
    tree_files = sorted(Path(json_dir).glob("*.json"))

    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CPO ToT Tree Viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #00d9ff; margin-bottom: 30px; }
        .controls { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
        .controls input, .controls select { padding: 10px 15px; border-radius: 8px; border: 1px solid #333; background: #16213e; color: #fff; }
        .tree-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .tree-card { background: #16213e; border-radius: 12px; padding: 15px; border: 1px solid #333; transition: transform 0.2s; }
        .tree-card:hover { transform: translateY(-5px); border-color: #00d9ff; }
        .tree-card h3 { color: #00d9ff; margin-bottom: 10px; font-size: 14px; }
        .tree-preview { background: #0f3460; border-radius: 8px; padding: 10px; max-height: 200px; overflow-y: auto; font-size: 12px; line-height: 1.6; }
        .stats { display: flex; gap: 15px; margin-top: 10px; font-size: 12px; color: #aaa; }
        .stat-item { background: #0f3460; padding: 5px 10px; border-radius: 5px; }
        .node { padding: 5px 0; border-bottom: 1px solid #333; }
        .node:last-child { border-bottom: none; }
        .node-depth { color: #888; margin-right: 8px; }
        .node-score-high { color: #2ecc71; }
        .node-score-med { color: #f39c12; }
        .node-score-low { color: #e74c3c; }
        .search-box { width: 100%; margin-bottom: 20px; }
        .search-box input { width: 100%; }
        .legend { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 CPO ToT Tree Viewer</h1>
        
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div> Correct Path</div>
            <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div> Wrong Terminal</div>
            <div class="legend-item"><div class="legend-color" style="background:#3498db"></div> High Score (≥7)</div>
            <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div> Medium Score (5-7)</div>
            <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div> Low Score (&lt;5)</div>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search by filename or content..." onkeyup="filterTrees()">
        </div>
        
        <div class="tree-grid" id="treeGrid">
"""

    for tree_file in tree_files[:50]:
        try:
            with open(tree_file, encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("nodes", {})
            node_count = len(nodes)

            preview_html = ""
            for node_id, node in list(nodes.items())[:10]:
                depth = node.get("depth", 0)
                score = node.get("score", 0)
                text = node.get("text", "")[:60]
                score_class = "node-score-high" if score >= 7 else ("node-score-med" if score >= 5 else "node-score-low")

                preview_html += f'''<div class="node">
                    <span class="node-depth">[L{depth}]</span>
                    <span class="{score_class}">Score: {score}</span>
                    <div>{text}</div>
                </div>'''

            html_content += f"""
            <div class="tree-card" data-filename="{tree_file.name}">
                <h3>{tree_file.name}</h3>
                <div class="tree-preview">{preview_html}</div>
                <div class="stats">
                    <span class="stat-item">Nodes: {node_count}</span>
                </div>
            </div>
"""
        except Exception:
            continue

    html_content += """
        </div>
    </div>
    <script>
        function filterTrees() {
            const query = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.tree-card');
            cards.forEach(card => {
                const text = card.getAttribute('data-filename').toLowerCase();
                card.style.display = text.includes(query) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""

    html_path = os.path.join(output_dir, "tot_viewer.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


def main():
    parser = argparse.ArgumentParser(description="Visualize ToT Trees")
    parser.add_argument("--tot-dir", type=str, default="data/interim/tot_nodes/gsm8k/Qwen3-8B_444")
    parser.add_argument("--output-dir", type=str, default="outputs/visualization/tot_trees")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--model", type=str, default="Qwen3-8B_444")
    parser.add_argument("--max-trees", type=int, default=10)
    args = parser.parse_args()

    print(f"Visualizing ToT trees from: {args.tot_dir}")
    print(f"Output directory: {args.output_dir}")

    results = visualize_tot_trees(
        tot_nodes_dir=args.tot_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        model=args.model,
        max_trees=args.max_trees,
    )

    print(f"\n✓ Visualization complete!")
    print(f"  Visualized: {results['visualized']}")
    print(f"  Failed: {results['failed']}")

    print("\nGenerating interactive HTML viewer...")
    html_path = generate_html_viewer(args.output_dir, args.tot_dir)
    print(f"  HTML Viewer: {html_path}")


if __name__ == "__main__":
    main()
