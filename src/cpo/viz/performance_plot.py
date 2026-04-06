from __future__ import annotations

import matplotlib.pyplot as plt


def plot_accuracy_bar(results: dict[str, float], out_file: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(list(results.keys()), list(results.values()))
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
