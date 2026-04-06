from __future__ import annotations

import matplotlib.pyplot as plt


def plot_hist(values: list[float], out_file: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
