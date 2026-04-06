"""
Stage 5: Run All Visualizations
Unified script to run all Stage 5 visualization tasks.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_script(script_path: str, args: list[str]) -> bool:
    """Run a Python script and return success status."""
    cmd = [sys.executable, script_path] + args
    print(f"\n{'=' * 60}")
    print(f"Running: {script_path}")
    print(f"Args: {args}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all Stage 5 visualizations")
    parser.add_argument("--tot-dir", type=str, default="data/interim/tot_nodes/gsm8k/Qwen3-8B_444")
    parser.add_argument("--eval-dir", type=str, default="outputs/eval/json/gsm8k/Qwen3-8B_444")
    parser.add_argument("--output-dir", type=str, default="outputs/visualization")
    parser.add_argument("--max-trees", type=int, default=10, help="Max trees to visualize")
    parser.add_argument("--skip-tot", action="store_true", help="Skip ToT tree visualization")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance visualization")
    parser.add_argument("--skip-stats", action="store_true", help="Skip statistics visualization")
    parser.add_argument("--demo", action="store_true", help="Generate demo charts with sample data")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base_output = args.output_dir

    results = {}

    if not args.skip_tot:
        tot_output = os.path.join(base_output, "tot_trees")
        success = run_script(str(script_dir / "visualize_tot_trees.py"), [
            "--tot-dir", args.tot_dir,
            "--output-dir", tot_output,
            "--max-trees", str(args.max_trees),
        ])
        results["ToT Trees"] = success

    if not args.skip_performance:
        perf_output = os.path.join(base_output, "performance")
        perf_args = ["--output-dir", perf_output]
        if args.demo:
            perf_args.append("--demo")
        success = run_script(str(script_dir / "visualize_performance.py"), perf_args)
        results["Performance"] = success

    if not args.skip_stats:
        stats_output = os.path.join(base_output, "stats")
        success = run_script(str(script_dir / "visualize_stats.py"), [
            "--tot-dir", args.tot_dir,
            "--output-dir", stats_output,
        ])
        results["Statistics"] = success

    print(f"\n{'=' * 60}")
    print("Stage 5 Visualization Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print(f"\nOutput directory: {base_output}")
    print("\nTo view the visualizations:")
    print(f"  1. ToT Trees: {os.path.join(base_output, 'tot_trees', 'tot_viewer.html')}")
    print(f"  2. Performance: {os.path.join(base_output, 'performance', 'performance_dashboard.html')}")


if __name__ == "__main__":
    main()
