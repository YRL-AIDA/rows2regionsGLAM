"""Visualize per-region confusion matrix as a heatmap."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(confusion_json_path, output_path=None):
    with open(confusion_json_path, "r") as f:
        data = json.load(f)

    classes = data["classes"]
    cm = np.array(data["confusion_matrix"])

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0)

    per_class_acc = {}
    for i, cls_name in enumerate(classes):
        if row_sums[i, 0] > 0:
            per_class_acc[cls_name] = cm[i, i] / row_sums[i, 0]
        else:
            per_class_acc[cls_name] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=axes[0],
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Number of regions"},
        annot_kws={"fontsize": 10},
    )
    axes[0].set_title("Confusion Matrix (absolute counts)", fontsize=13, pad=16)
    axes[0].set_xlabel("Predicted class", fontsize=11)
    axes[0].set_ylabel("True class", fontsize=11)

    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                annot[i, j] = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            else:
                annot[i, j] = ""

    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="YlOrRd",
        xticklabels=classes, yticklabels=classes,
        ax=axes[1],
        vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Recall (fraction of true class)"},
        annot_kws={"fontsize": 9},
    )
    axes[1].set_title("Confusion Matrix (row-normalized, recall)", fontsize=13, pad=16)
    axes[1].set_xlabel("Predicted class", fontsize=11)
    axes[1].set_ylabel("True class", fontsize=11)

    summary_lines = ["Per-class recall:"]
    for cls_name in classes:
        acc = per_class_acc[cls_name]
        summary_lines.append(f"  {cls_name:>8s}: {acc:.3f}")
    summary_text = "\n".join(summary_lines)

    fig.text(
        0.5, -0.03,
        summary_text,
        ha="center", va="top",
        fontsize=9, family="monospace",
        transform=fig.transFigure,
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {output_path}")
    else:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    if len(sys.argv) > 1:
        confusion_file = sys.argv[1]
    else:
        result_dir = Path(__file__).resolve().parent / "result"
        candidates = sorted(result_dir.glob("base_seed_*_confusion.json"))
        confusion_file = str(candidates[0]) if candidates else (
            "result/base_seed_0_confusion.json"
        )

    output_file = Path(confusion_file).with_suffix(".png")
    plot_confusion_matrix(confusion_file, output_file)