#!/usr/bin/env python3
"""
Generate three separate bar charts (one metric per figure) using barplot.py.

Each figure shows all cutoffs (@1, @3, @5, @10) for a single metric:
recall, precision, ndcg. Figures are saved as JPG next to this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from barplot import plot_bar

THIS_DIR = Path(__file__).resolve().parent

RECALL_INDICATORS: List[str] = ["recall@1", "recall@3", "recall@5", "recall@10"]
PRECISION_INDICATORS: List[str] = ["precision@1", "precision@3", "precision@5", "precision@10"]
NDCG_INDICATORS: List[str] = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]


def plot_three() -> None:
    """Render one figure per metric (recall, precision, ndcg)."""
    groups = {
        "recall": RECALL_INDICATORS,
        "precision": PRECISION_INDICATORS,
        "ndcg": NDCG_INDICATORS,
    }
    for name, indicators in groups.items():
        jpg_path = THIS_DIR / f"{name}.jpg"
        plot_bar(indicators=indicators, output_paths=[jpg_path], x_tick_rotation=0)


if __name__ == "__main__":
    plot_three()

