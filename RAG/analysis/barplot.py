#!/usr/bin/env python3
"""
Bar plot of retrieval metrics for four methods.

- Results are loaded from JSON files under ../results/.
- By default, all common indicators are shown; edit INDICATORS_TO_PLOT to
  choose a subset.
- Saves the figure next to this script as barplot.jpg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Paths
THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR.parent / "results"
OUTPUT_FIG_JPG = THIS_DIR / "barplot.jpg"

# Map display names to result files for clarity in the legend.
# Order matters: baseline then ours for each backend.
METHOD_FILES: Dict[str, Path] = {
    "Baseline (FAISS)": RESULTS_DIR / "baseline_faiss_results.json",
    "Ours (FAISS)": RESULTS_DIR / "ours_faiss_results.json",
    "Baseline (Qdrant)": RESULTS_DIR / "baseline_qdrant_results.json",
    "Ours (Qdrant)": RESULTS_DIR / "ours_qdrant_results.json",
}

# Edit this list to control which indicators to show.
INDICATORS_TO_PLOT: List[str] = [
    # Recall first
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    # Precision next
    "precision@1",
    "precision@3",
    "precision@5",
    "precision@10",
    # NDCG next
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    # Other metrics
    "mrr",
    # "mean_latency_ms",  # latency in milliseconds
]


def load_results() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load metrics from the four result files."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, path in METHOD_FILES.items():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = {name: {"mean": m.get("mean"), "std": m.get("std")} for name, m in payload.get("metrics", {}).items()}
        # Latency is stored at the top level.
        metrics["mean_latency_ms"] = {"mean": payload.get("mean_latency_ms"), "std": None}
        results[label] = metrics
    return results


def to_table(indicators: Iterable[str], results: Dict[str, Dict[str, Dict[str, float]]]):
    """Flatten metrics into a list of rows."""
    table = []
    for indicator in indicators:
        for method, metrics in results.items():
            metric = metrics.get(indicator)
            if metric is None:
                continue
            table.append(
                {
                    "indicator": indicator,
                    "method": method,
                    "mean": metric.get("mean"),
                    "std": metric.get("std"),
                }
            )
    return table


def plot_bar(
    indicators: Optional[Iterable[str]] = None,
    output_paths: Optional[Iterable[Path]] = None,
    figsize: tuple = (9, 4),
    x_tick_rotation: float = 30,
) -> List[Path]:
    """Create grouped bar plot with one group per indicator and save to disk."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
        }
    )

    chosen = list(indicators) if indicators is not None else list(INDICATORS_TO_PLOT)
    results = load_results()
    table = to_table(chosen, results)
    if not table:
        raise ValueError("No data available for the requested indicators.")

    indicator_names = list(dict.fromkeys(row["indicator"] for row in table))  # preserve order
    methods = list(METHOD_FILES.keys())

    x = np.arange(len(indicator_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=figsize)

    # Pair light/dark colors per backend: FAISS (light/dark), Qdrant (light/dark).
    colors = ["#9ECAE1", "#08519C", "#FDD0A2", "#D94801"]
    for idx, method in enumerate(methods):
        means = []
        stds = []
        for indicator in indicator_names:
            row = next((r for r in table if r["indicator"] == indicator and r["method"] == method), None)
            means.append(row["mean"] if row else np.nan)
            stds.append(row["std"] if row else None)

        # Matplotlib error bars cannot contain None; use NaN to skip.
        stds_clean = [np.nan if s is None else s for s in stds]
        means_arr = np.array(means, dtype=float)
        stds_arr = np.array(stds_clean, dtype=float)
        # Prevent lower error from dipping below zero (metrics are non-negative).
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

    ax.set_xticks(x)
    ha = "right" if x_tick_rotation else "center"
    ax.set_xticklabels(indicator_names, rotation=x_tick_rotation, ha=ha)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Performance by Method")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    outputs = list(output_paths) if output_paths is not None else [OUTPUT_FIG_JPG]
    saved_paths: List[Path] = []
    for path in outputs:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
    print("Saved bar chart to:", ", ".join(str(p) for p in saved_paths))
    return saved_paths


if __name__ == "__main__":
    plot_bar()
