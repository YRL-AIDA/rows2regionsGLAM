"""p10 — Visualization: heatmaps and metrics comparison for cluster analysis.

Reads {name}_confusion.csv, {name}_metrics.json, and all_metrics.csv from
the `outputs/` subdirectory (produced by cluster_analysis.py). Generates:

  - Per-tokenizer heatmaps:     `outputs/{name}_heatmap.pdf`
  - Combined multi-panel grid:  `outputs/all_heatmaps.pdf`
  - Metrics bar chart:          `outputs/metrics_comparison.pdf`

Usage as standalone:  `python3 visualize.py`
Usage as module:      `from visualize import make_all_visualizations`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

TOKENIZER_NAMES = [
    "no_font",
    "pdf_font",
    "font_emb_16",
    "font_emb_32",
    "font_emb",
]

CLASS_NAMES = ["text", "title", "list", "table", "figure"]

# Figure dimensions (inches)
_HEATMAP_SIZE = (8, 6)
_COMBINED_SIZE = (18, 14)
_BAR_SIZE = (10, 6)

# Typography
_TITLE_SIZE = 14
_LABEL_SIZE = 12
_ANNOT_SIZE = 10
_TICK_SIZE = 11

# Output DPI
_OUTPUT_DPI = 300

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────

def _output_dir() -> Path:
    """Return the outputs/ directory (sibling of this script)."""
    return Path(__file__).resolve().parent / "outputs"


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_confusion(name: str, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load {name}_confusion.csv as a DataFrame (index = cluster names)."""
    path = (output_dir or _output_dir()) / f"{name}_confusion.csv"
    df = pd.read_csv(path, index_col=0)
    return df


def load_metrics(name: str, output_dir: Optional[Path] = None) -> dict:
    """Load {name}_metrics.json."""
    path = (output_dir or _output_dir()) / f"{name}_metrics.json"
    with open(path, "r") as f:
        return json.load(f)


def load_summary(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all_metrics.csv (index = tokenizer name)."""
    path = (output_dir or _output_dir()) / "all_metrics.csv"
    return pd.read_csv(path, index_col=0)


# ──────────────────────────────────────────────────────────────────────
# Visual style defaults
# ──────────────────────────────────────────────────────────────────────

def _set_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": _LABEL_SIZE,
            "axes.titlesize": _TITLE_SIZE,
            "axes.labelsize": _LABEL_SIZE,
            "xtick.labelsize": _TICK_SIZE,
            "ytick.labelsize": _TICK_SIZE,
            "legend.fontsize": _LABEL_SIZE,
            "figure.dpi": 100,
            "savefig.dpi": _OUTPUT_DPI,
            "savefig.bbox": "tight",
            "image.cmap": "YlOrRd",
        }
    )
    sns.set_style("whitegrid")


# ──────────────────────────────────────────────────────────────────────
# 1. Per-tokenizer heatmap
# ──────────────────────────────────────────────────────────────────────

def _make_heatmap(
    confusion: pd.DataFrame,
    metrics: dict,
    name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a single confusion-matrix heatmap on *ax*.

    Parameters
    ----------
    confusion : pd.DataFrame
        Rows = cluster names, columns = class labels.
    metrics : dict
        Must contain 'overall_purity', 'ARI', 'NMI'.
    name : str
        Tokenizer name for the title.
    ax : plt.Axes or None
        Axes to draw on; created if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=_HEATMAP_SIZE)

    # Build annotation matrix: count (int)
    annot = confusion.values.astype(int)

    sns.heatmap(
        confusion,
        annot=annot,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Number of regions"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": _ANNOT_SIZE},
    )

    ax.set_title(f"Cluster Analysis — {name}", fontsize=_TITLE_SIZE, pad=22)
    ax.set_xlabel("True class", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Cluster", fontsize=_LABEL_SIZE)

    # Rotate x-tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Subtitle line with metrics right below the title
    purity = metrics.get("overall_purity", 0)
    ari = metrics.get("ARI", 0)
    nmi = metrics.get("NMI", 0)
    subtitle = f"Purity: {purity:.3f}  |  ARI: {ari:.3f}  |  NMI: {nmi:.3f}"
    ax.text(
        0.5, 1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="bottom",
    )

    return ax


# ──────────────────────────────────────────────────────────────────────
# 2. Combined multi-panel figure
# ──────────────────────────────────────────────────────────────────────

def _make_combined_heatmaps(
    data: Dict[str, tuple],
) -> plt.Figure:
    """2×3 grid of heatmaps (last cell empty), shared colorbar.

    Parameters
    ----------
    data : dict
        tokenizer_name → (confusion_df, metrics_dict)
    """
    fig, axes = plt.subplots(2, 3, figsize=_COMBINED_SIZE, constrained_layout=True)
    axes = axes.flatten()

    # Collect all values to determine shared color limits
    all_values = np.concatenate(
        [conf.values.ravel() for conf, _ in data.values()]
    )
    vmin, vmax = 0, all_values.max()

    for i, (name, (confusion, metrics)) in enumerate(data.items()):
        ax = axes[i]
        _make_heatmap_pooled(confusion, metrics, name, ax, vmin, vmax)

    # Hide the unused 6th subplot
    axes[5].set_visible(False)

    # Remove individual cbars from all 5 panels
    for i in range(5):
        _remove_colorbar(axes[i])

    # Create a dummy mappable for the shared colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:5].tolist(),
        fraction=0.02,
        pad=0.02,
        shrink=0.85,
    )
    cbar.set_label("Number of regions", fontsize=_LABEL_SIZE)

    fig.suptitle(
        "Cluster Analysis — All Tokenizers",
        fontsize=_TITLE_SIZE + 4,
        y=1.01,
    )
    return fig


def _make_heatmap_pooled(
    confusion: pd.DataFrame,
    metrics: dict,
    name: str,
    ax: plt.Axes,
    vmin: float,
    vmax: float,
):
    """Heatmap variant for the combined figure (shared vmin/vmax, no cbar)."""
    annot = confusion.values.astype(int)

    sns.heatmap(
        confusion,
        annot=annot,
        fmt="d",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 8},
    )

    purity = metrics.get("overall_purity", 0)
    ari = metrics.get("ARI", 0)
    nmi = metrics.get("NMI", 0)
    title = f"{name}\nPur: {purity:.3f}  ARI: {ari:.3f}  NMI: {nmi:.3f}"
    ax.set_title(title, fontsize=_LABEL_SIZE, pad=6)
    ax.set_xlabel("True class", fontsize=9)
    ax.set_ylabel("Cluster", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)


def _remove_colorbar(ax: plt.Axes):
    """Remove the colorbar associated with *ax*, if any."""
    for child in ax.collections:
        if hasattr(child, "colorbar") and child.colorbar is not None:
            child.colorbar.remove()
            child.colorbar = None


# ──────────────────────────────────────────────────────────────────────
# 3. Metrics bar chart
# ──────────────────────────────────────────────────────────────────────

def _make_metrics_chart(summary: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing purity, ARI, NMI across tokenizers.

    Parameters
    ----------
    summary : pd.DataFrame
        Index = tokenizer names, columns = ['purity', 'ARI', 'NMI'].
    """
    fig, ax = plt.subplots(figsize=_BAR_SIZE, constrained_layout=True)

    metrics_list = ["purity", "ARI", "NMI"]
    tokenizers = summary.index.tolist()
    x = np.arange(len(tokenizers))
    width = 0.25

    colors = sns.color_palette("Set2", n_colors=len(metrics_list))

    for i, (metric, color) in enumerate(zip(metrics_list, colors)):
        values = [summary.loc[t, metric] for t in tokenizers]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=metric.upper() if metric != "purity" else "Purity",
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Tokenizer", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Score", fontsize=_LABEL_SIZE)
    ax.set_title("Cluster Quality Metrics by Tokenizer", fontsize=_TITLE_SIZE, pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tokenizers, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", fontsize=10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", alpha=0.4)

    return fig


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

def make_all_visualizations(
    output_dir: Optional[Path] = None,
    tokenizer_names: Optional[List[str]] = None,
) -> List[Path]:
    """Run all visualizations and return a list of saved file paths.

    Parameters
    ----------
    output_dir : Path or None
        Directory containing *_confusion.csv, *_metrics.json, all_metrics.csv.
        Defaults to `outputs/` next to this script.
    tokenizer_names : list of str or None
        Tokenizers to process. Defaults to all 5.

    Returns
    -------
    list of Path
        Paths to all output files created.
    """
    _set_style()

    out = output_dir or _output_dir()
    names = tokenizer_names or TOKENIZER_NAMES

    logger.info(f"Loading data from {out}")

    # Load all data
    all_data: Dict[str, tuple] = {}
    for name in names:
        try:
            confusion = load_confusion(name, out)
            metrics = load_metrics(name, out)
            all_data[name] = (confusion, metrics)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {name}: {e}")
            continue

    summary = load_summary(out)
    # Drop columns that aren't metrics (input_dim, n_regions)
    metrics_cols = [c for c in ["purity", "ARI", "NMI"] if c in summary.columns]
    summary_subset = summary.loc[
        [n for n in names if n in summary.index], metrics_cols
    ]

    saved: List[Path] = []

    # 1. Per-tokenizer heatmaps
    for name, (confusion, metrics) in all_data.items():
        fig, ax = plt.subplots(figsize=_HEATMAP_SIZE)
        _make_heatmap(confusion, metrics, name, ax=ax)
        path = out / f"{name}_heatmap.pdf"
        fig.savefig(path, dpi=_OUTPUT_DPI)
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved {path}")

    # 2. Combined figure
    fig_combined = _make_combined_heatmaps(all_data)
    path_combined = out / "all_heatmaps.pdf"
    fig_combined.savefig(path_combined, dpi=_OUTPUT_DPI)
    plt.close(fig_combined)
    saved.append(path_combined)
    logger.info(f"Saved {path_combined}")

    # 3. Metrics bar chart
    fig_bar = _make_metrics_chart(summary_subset)
    path_bar = out / "metrics_comparison.pdf"
    fig_bar.savefig(path_bar, dpi=_OUTPUT_DPI)
    plt.close(fig_bar)
    saved.append(path_bar)
    logger.info(f"Saved {path_bar}")

    return saved


# ──────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = make_all_visualizations()
    print(f"\nGenerated {len(paths)} files:")
    for p in paths:
        print(f"  {p}")
