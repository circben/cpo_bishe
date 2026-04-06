"""
Stage 5: Global Statistics Visualization
Visualizes tree structure statistics and distribution analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_tot_statistics(tot_dir: str) -> dict:
    """Load and compute statistics from ToT tree files."""
    tree_files = list(Path(tot_dir).glob("*.json"))

    stats = {
        "node_counts": [],
        "depth_counts": [],
        "score_distribution": [],
        "success_path_lengths": [],
        "branching_factors": [],
        "terminal_nodes": 0,
        "correct_nodes": 0,
        "total_trees": 0,
    }

    for tree_file in tree_files:
        try:
            with open(tree_file, encoding="utf-8") as f:
                data = json.load(f)

            nodes = data.get("nodes", {})
            if not nodes:
                continue

            stats["total_trees"] += 1
            stats["node_counts"].append(len(nodes))

            depths = [n.get("depth", 0) for n in nodes.values()]
            stats["depth_counts"].extend(depths)

            scores = [n.get("score", 0) for n in nodes.values()]
            stats["score_distribution"].extend(scores)

            terminal = sum(1 for n in nodes.values() if n.get("is_terminal", False))
            stats["terminal_nodes"] += terminal

            correct = sum(1 for n in nodes.values() if n.get("is_correct", False))
            stats["correct_nodes"] += correct

            if depths:
                max_depth = max(depths)
                stats["success_path_lengths"].append(max_depth)

        except Exception:
            continue

    return stats


def plot_node_distribution(stats: dict, save_path: str) -> None:
    """Plot distribution of nodes per tree."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    node_counts = stats.get("node_counts", [])
    if node_counts:
        axes[0].hist(node_counts, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
        axes[0].axvline(np.mean(node_counts), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {np.mean(node_counts):.1f}")
        axes[0].set_xlabel("Number of Nodes", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Nodes per Tree Distribution", fontsize=14, fontweight="bold")
        axes[0].legend()

    depth_counts = stats.get("depth_counts", [])
    if depth_counts:
        depth_freq = Counter(depth_counts)
        depths = sorted(depth_freq.keys())
        counts = [depth_freq[d] for d in depths]
        axes[1].bar(depths, counts, color="#2ecc71", edgecolor="white")
        axes[1].set_xlabel("Depth Level", fontsize=12)
        axes[1].set_ylabel("Node Count", fontsize=12)
        axes[1].set_title("Node Depth Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(stats: dict, save_path: str) -> None:
    """Plot score distribution across all nodes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scores = stats.get("score_distribution", [])
    if scores:
        axes[0].hist(scores, bins=20, color="#9b59b6", edgecolor="white", alpha=0.8)
        axes[0].axvline(np.mean(scores), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {np.mean(scores):.2f}")
        axes[0].set_xlabel("Score", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Node Score Distribution", fontsize=14, fontweight="bold")
        axes[0].legend()

    path_lengths = stats.get("success_path_lengths", [])
    if path_lengths:
        axes[1].hist(path_lengths, bins=10, color="#f39c12", edgecolor="white", alpha=0.8)
        axes[1].axvline(np.mean(path_lengths), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {np.mean(path_lengths):.1f}")
        axes[1].set_xlabel("Path Length", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Success Path Length Distribution", fontsize=14, fontweight="bold")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_statistics_boxplot(stats: dict, save_path: str) -> None:
    """Create boxplot of key statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    node_counts = stats.get("node_counts", [])
    if node_counts:
        axes[0].boxplot(node_counts, patch_artist=True, boxprops=dict(facecolor="#3498db", alpha=0.7))
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title("Nodes per Tree", fontsize=14, fontweight="bold")
        axes[0].grid(alpha=0.3)

    scores = stats.get("score_distribution", [])
    if scores:
        axes[1].boxplot(scores, patch_artist=True, boxprops=dict(facecolor="#2ecc71", alpha=0.7))
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].set_title("Node Scores", fontsize=14, fontweight="bold")
        axes[1].grid(alpha=0.3)

    path_lengths = stats.get("success_path_lengths", [])
    if path_lengths:
        axes[2].boxplot(path_lengths, patch_artist=True, boxprops=dict(facecolor="#f39c12", alpha=0.7))
        axes[2].set_ylabel("Length", fontsize=12)
        axes[2].set_title("Path Lengths", fontsize=14, fontweight="bold")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_stats_report(stats: dict, output_dir: str) -> str:
    """Generate a summary report of statistics."""
    node_counts = stats.get("node_counts", [])
    scores = stats.get("score_distribution", [])
    path_lengths = stats.get("success_path_lengths", [])

    report = f"""
CPO ToT Statistics Report
========================

Total Trees Analyzed: {stats['total_trees']}

Node Statistics:
  - Total Nodes: {sum(node_counts)}
  - Avg Nodes per Tree: {np.mean(node_counts):.1f} (±{np.std(node_counts):.1f})
  - Min/Max Nodes: {min(node_counts)}/{max(node_counts)}

Score Statistics:
  - Avg Score: {np.mean(scores):.2f} (±{np.std(scores):.2f})
  - Min/Max Score: {min(scores):.1f}/{max(scores):.1f}

Path Statistics:
  - Avg Path Length: {np.mean(path_lengths):.1f} (±{np.std(path_lengths):.1f})
  - Terminal Nodes: {stats['terminal_nodes']}
  - Correct Nodes: {stats['correct_nodes']}
"""

    report_path = os.path.join(output_dir, "statistics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Visualize ToT Statistics")
    parser.add_argument("--tot-dir", type=str, default="data/interim/tot_nodes/gsm8k/Qwen3-8B_444")
    parser.add_argument("--output-dir", type=str, default="outputs/visualization/stats")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ToT statistics from: {args.tot_dir}")
    stats = load_tot_statistics(args.tot_dir)

    print(f"Analyzed {stats['total_trees']} trees")

    print("\nGenerating visualizations...")
    plot_node_distribution(stats, os.path.join(args.output_dir, "node_distribution.png"))
    plot_score_distribution(stats, os.path.join(args.output_dir, "score_distribution.png"))
    plot_statistics_boxplot(stats, os.path.join(args.output_dir, "statistics_boxplot.png"))

    report_path = generate_stats_report(stats, args.output_dir)

    print(f"\n✓ Statistics visualization complete!")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
