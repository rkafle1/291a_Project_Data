#!/usr/bin/env python3
"""
Plot K=1 indicators for three qdrant-related result files.

Methods:
- Ours (Qdrant)
- Ablation w=1.0/0.0
- Ablation w=0.85/0.15
- Ablation w=0.55/0.45
- Ablation w=0.4/0.6

Outputs a single JPG: barplot3.jpg
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR.parent / "results"
OUTPUT_FIG = THIS_DIR / "ablation_weights.jpg"

METHOD_FILES = {
    # "Ablation w=1.00/0.00": RESULTS_DIR / "ablation_retrieval_1.0_0.0_results.json",
    "Ablation w=0.85/0.15": RESULTS_DIR / "ablation_retrieval_0.85_0.15_results.json",
    "Ours w=0.70/0.30": RESULTS_DIR / "ours_qdrant_results.json",
    "Ablation w=0.55/0.45": RESULTS_DIR / "ablation_retrieval_0.55_0.45_results.json",
    "Ablation w=0.40/0.60": RESULTS_DIR / "ablation_retrieval_0.4_0.6_results.json",
}

INDICATORS = [    # Recall first
    "recall@1",
    # "recall@3",
    "recall@5",
    "recall@10",
    # Precision next
    "precision@1",
    # "precision@3",
    "precision@5",
    "precision@10",
    # NDCG next
    "ndcg@1",
    # "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    ]


def load_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, path in METHOD_FILES.items():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = {name: {"mean": m.get("mean"), "std": m.get("std")} for name, m in payload.get("metrics", {}).items()}
        data[label] = metrics
    return data


def plot_grouped(indicators: Iterable[str] = INDICATORS, output_path: Path = OUTPUT_FIG) -> Path:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
        }
    )

    results = load_results()
    methods = list(METHOD_FILES.keys())
    indicator_names = list(indicators)

    x = np.arange(len(indicator_names))
    # Slightly slimmer bars for clearer separation when many methods are shown
    width = 0.16
    fig, ax = plt.subplots(figsize=(10, 4))

    # Colorblind-friendly palette; enough entries to cover all methods
    colors = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#E45756",  # red
        "#72B7B2",  # teal
    ]
    offsets_by_method = {}
    for idx, method in enumerate(methods):
        means = []
        stds = []
        for indicator in indicator_names:
            metric = results.get(method, {}).get(indicator)
            means.append(metric.get("mean") if metric else np.nan)
            stds.append(metric.get("std") if metric else None)

        stds_clean = [np.nan if s is None else s for s in stds]
        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds_clean, dtype=float)
        lower_err = np.minimum(stds_arr, means_arr)
        lower_err = np.clip(lower_err, 0, None)
        yerr = np.vstack([lower_err, stds_arr])

        offsets = x + (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            offsets,
            means,
            width,
            label=method,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.8,
            yerr=yerr,
            capsize=3,
        )
        offsets_by_method[method] = offsets

    ax.set_xticks(x)
    ax.set_xticklabels(indicator_names, rotation=0, ha="center")
    ax.set_ylabel("Score")
    ax.set_title("Variants of Hybrid Retrieval on Different Weights")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved bar chart to {output_path}")
    return output_path


if __name__ == "__main__":
    plot_grouped()

