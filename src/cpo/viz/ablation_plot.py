from __future__ import annotations

import matplotlib.pyplot as plt


def plot_curve(x: list[float], y: list[float], out_file: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
