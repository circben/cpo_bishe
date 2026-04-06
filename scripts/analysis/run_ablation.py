from __future__ import annotations

from cpo.viz.ablation_plot import plot_curve


def main() -> None:
    betas = [0.05, 0.1, 0.5, 1.0]
    scores = [0.42, 0.47, 0.44, 0.39]
    plot_curve(betas, scores, "outputs/figures/ablations/beta_curve.png", "beta", "accuracy")
    print("Saved beta ablation curve")


if __name__ == "__main__":
    main()
