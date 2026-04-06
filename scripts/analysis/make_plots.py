from __future__ import annotations

from cpo.eval.baselines import run_baselines_stub
from cpo.viz.performance_plot import plot_accuracy_bar


def main() -> None:
    scores = run_baselines_stub()
    plot_accuracy_bar(scores, "outputs/figures/main_results/accuracy_bar.png")
    print("Saved accuracy bar")


if __name__ == "__main__":
    main()
