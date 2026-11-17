"""Общие настройки графиков."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.0,
    "figure.constrained_layout.use": True,
})


def fig_dir(lab_dir: str) -> str:
    d = os.path.join(lab_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, lab_dir: str, name: str):
    path = os.path.join(fig_dir(lab_dir), name)
    fig.savefig(path)
    plt.close(fig)
    print("  ->", os.path.relpath(path))
    return path


def save_results(lab_dir: str, results: dict):
    with open(os.path.join(lab_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
