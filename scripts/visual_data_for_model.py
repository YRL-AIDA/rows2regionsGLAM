#!/usr/bin/env python3
"""
Generate an HTML report for a single PDF document with:

- Rows (thin lines) and graph edges (green=true, red=false)
- Ideal regions computed from cached true edges/nodes (thick solid, colored)
- Ground truth regions from COCO (thick dashed, colored)
- Statistics: region/row/edge/node counts, mAP (all/seg/per-class)

Usage::

    python scripts/visual_data_for_model.py PMC1247480_00001.pdf --env local
    python scripts/visual_data_for_model.py doc.pdf --env local --dir_pdf /path/to/pdfs --dir_json /path/to/cache
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import dotenv_values

# ── project root ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rows2regionsGLAM.utils.pdf_manager import PDFManager
from rows2regionsGLAM.utils.row_manager import RowManager
from rows2regionsGLAM.utils.coco_manager import COCOManager
from pagerlib.dtypes import ImageSegment, Region
from pagerlib.dtypes.relationship import Graph
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ── constants ──────────────────────────────────────────────────────────
COLORS_CLASS = [
    "gray", "tab:red", "tab:green", "tab:blue",
    "tab:orange", "tab:purple", "tab:brown",
    "tab:pink", "tab:olive", "tab:cyan", "lime", "teal",
]

# ===================================================================
# Environment helpers
# ===================================================================


def _resolve_env_path(env_arg: str) -> Path:
    """Map a short name like ``local`` to the real ``.env.local`` path."""
    templates = {
        "debug": ".env.debug",
        "local": ".env.local",
        "server": ".env.server",
        "serverML": ".env.serverML",
    }
    if env_arg in templates:
        path = _PROJECT_ROOT / templates[env_arg]
    else:
        path = Path(env_arg)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    return path


def _load_env(env_name: str) -> Dict[str, Optional[str]]:
    """Resolve and load the requested ``.env`` file into a dict."""
    return dict(dotenv_values(_resolve_env_path(env_name)))

# ===================================================================
# File locators
# ===================================================================


def _find_pdf(
    name: str, dir_pdf: Optional[str], env: Dict[str, Optional[str]]
) -> Path:
    """Locate a PDF by filename — searches TEST_PATH then DATASET_PATH."""
    if dir_pdf:
        path = Path(dir_pdf) / name
        if path.exists():
            return path
        raise FileNotFoundError(f"PDF not found in --dir_pdf: {path}")

    for key in ("TEST_PATH", "DATASET_PATH"):
        base = env.get(key)
        if base is None:
            continue
        path = Path(base) / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"PDF '{name}' not found in TEST_PATH or DATASET_PATH. "
        f"Use --dir_pdf to specify a directory."
    )


def _find_json_cache(
    name: str, dir_json: Optional[str], env: Dict[str, Optional[str]]
) -> Path:
    """Locate the cached JSON for a PDF."""
    cache_root = dir_json or env.get("CASH_PDF_PATH")
    if not cache_root:
        raise ValueError("CASH_PDF_PATH not set and --dir_json not provided")
    path = Path(cache_root)/'test' / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cached JSON not found: {path}")
    return path


def _resolve_coco(env: Dict[str, Optional[str]]) -> Tuple[str, str]:
    """Return (coco_path, dataset_name) preferring test split."""
    coco_path = env.get("TEST_COCO_PATH") or env.get("COCO_PATH")
    dataset = env.get("NAME_TEST_DATASET") or env.get("NAME_DATASET") or "publaynet"
    if not coco_path:
        raise ValueError("Neither TEST_COCO_PATH nor COCO_PATH found in env")
    return coco_path, dataset

# ===================================================================
# Region computation from cached true data (no model)
# ===================================================================


def _compute_ideal_regions(
    rows_json: List[dict],
    inds: List[List[int]],
    true_edges: List[int],
    true_nodes: List[Optional[int]],
    id2name: Dict[int, str],
) -> Tuple[List[ImageSegment], List[str], List[int]]:
    """Compute regions using ground-truth edges and node classes.

    Same connected-components logic as ``Rows2Regions.regions_from_graph``,
    but operates on cached integer labels instead of model predictions.

    Returns:
        ``(segments, category_names, category_ids)``.
    """
    graph = Graph()
    for row in rows_json:
        seg = ImageSegment(dict_p_size=row["segment"])
        graph.add_node(*seg.get_center())

    for ni, nj, is_edge in zip(inds[0], inds[1], true_edges):
        if is_edge:
            graph.add_edge(ni + 1, nj + 1)          # Graph is 1-indexed

    regions: List[Region] = []
    for component in graph.get_related_graphs():
        indexes = [n.index - 1 for n in component.get_nodes()]
        labels = [true_nodes[i] or 0 for i in indexes]       # None → other
        majority = Counter(labels).most_common(1)[0][0]
        name = id2name.get(majority, str(majority))
        regions.append(
            Region(children=[rows_json[i] for i in indexes], data={"label": name})
        )

    segments = [r.segment for r in regions]
    category_names = [r.data.get("label", "other") for r in regions]
    name_to_id = {v: k for k, v in id2name.items()}
    category_ids = [name_to_id.get(n, 0) for n in category_names]
    return segments, category_names, category_ids


# ===================================================================
# Metrics
# ===================================================================


def _compute_mAP(
    pred_boxes_xywh: List[List[float]],
    pred_cls_ids: List[int],
    true_boxes_xywh: List[List[float]],
    true_cls_ids: List[int],
    id2name: Dict[int, str],
) -> Dict[str, float]:
    """Compute mAP (all), mAP (seg), and per-class mAP."""
    if not pred_boxes_xywh:
        return {"mAP (all)": 0.0, "mAP (seg)": 0.0}

    # --- all classes ---
    metric_all = MeanAveragePrecision(box_format="xywh", class_metrics=True)
    metric_all.update(
        [
            dict(
                boxes=torch.tensor(pred_boxes_xywh, dtype=torch.float32),
                scores=torch.ones(len(pred_boxes_xywh), dtype=torch.float32),
                labels=torch.tensor(pred_cls_ids, dtype=torch.int64),
            )
        ],
        [
            dict(
                boxes=torch.tensor(true_boxes_xywh, dtype=torch.float32),
                labels=torch.tensor(true_cls_ids, dtype=torch.int64),
            )
        ],
    )
    rez = metric_all.compute()

    # --- seg only (class != 0) ---
    seg_p = [b for b, l in zip(pred_boxes_xywh, pred_cls_ids) if l != 0]
    seg_t = [b for b, l in zip(true_boxes_xywh, true_cls_ids) if l != 0]
    if seg_p and seg_t:
        metric_seg = MeanAveragePrecision(box_format="xywh")
        metric_seg.update(
            [dict(boxes=torch.tensor(seg_p, dtype=torch.float32),
                  scores=torch.ones(len(seg_p), dtype=torch.float32),
                  labels=torch.ones(len(seg_p), dtype=torch.int64))],
            [dict(boxes=torch.tensor(seg_t, dtype=torch.float32),
                  labels=torch.ones(len(seg_t), dtype=torch.int64))],
        )
        seg_map = float(metric_seg.compute()["map"])
    else:
        seg_map = 0.0

    result = {"mAP (all)": float(rez["map"]), "mAP (seg)": seg_map}
    for cls_id, val in zip(rez.get("classes", []), rez.get("map_per_class", [])):
        result[f"mAP ({id2name.get(int(cls_id), int(cls_id))})"] = float(val)
    return result


# ===================================================================
# Rendering
# ===================================================================


def _render_page(
    img: np.ndarray,
    rows_json: List[dict],
    inds: List[List[int]],
    true_edges: List[int],
    ideal_segments: List[ImageSegment],
    ideal_labels: List[str],
    true_regions: List[ImageSegment],
    true_categories: List[int],
    id2name: Dict[int, str],
    page_w: int,
    page_h: int,
    title: str = "",
) -> str:
    """Render all layers and return a base64-encoded PNG string."""
    fig, ax = plt.subplots(figsize=(page_w / 100, page_h / 100), dpi=300)
    ax.imshow(img, extent=[0, page_w, page_h, 0])

    _draw_rows(ax, rows_json)                                     # 1
    _draw_edges(ax, rows_json, inds, true_edges)                  # 2
    _draw_ideal_regions(ax, ideal_segments, ideal_labels, id2name)  # 3
    _draw_gt_regions(ax, true_regions, true_categories, id2name)   # 4
    _draw_legend(ax)                                              # 5

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _draw_rows(ax, rows_json: List[dict]) -> None:
    """Thin gray rectangles for each row."""
    for row in rows_json:
        seg = row.get("segment", {})
        if not seg:
            continue
        x0, y0 = seg.get("x_top_left", 0), seg.get("y_top_left", 0)
        w, h = seg.get("width", 0), seg.get("height", 0)
        ax.plot(
            [x0, x0 + w, x0 + w, x0, x0],
            [y0, y0, y0 + h, y0 + h, y0],
            color="lightgray", linewidth=0.3,
        )


def _draw_edges(
    ax, rows_json: List[dict], inds: List[List[int]], true_edges: List[int]
) -> None:
    """Green lines for true edges, red for false edges."""
    if not inds or len(inds[0]) == 0:
        return
    for idx, (i, j) in enumerate(zip(inds[0], inds[1])):
        is_true = bool(true_edges[idx]) if idx < len(true_edges) else False
        color = "green" if is_true else "red"
        si, sj = rows_json[i].get("segment", {}), rows_json[j].get("segment", {})
        cx_i = si.get("x_top_left", 0) + si.get("width", 0) / 2
        cy_i = si.get("y_top_left", 0) + si.get("height", 0) / 2
        cx_j = sj.get("x_top_left", 0) + sj.get("width", 0) / 2
        cy_j = sj.get("y_top_left", 0) + sj.get("height", 0) / 2
        ax.plot([cx_i, cx_j], [cy_i, cy_j], color=color, linewidth=0.4, alpha=0.7)


def _draw_ideal_regions(
    ax,
    segments: List[ImageSegment],
    labels: List[str],
    id2name: Dict[int, str],
) -> None:
    """Thick SOLID boxes for ideal regions (from true edges/nodes)."""
    name_to_id = {v: k for k, v in id2name.items()}
    for seg, label in zip(segments, labels):
        color = COLORS_CLASS[name_to_id.get(label, 0) % len(COLORS_CLASS)]
        x0, y0 = seg.x_top_left, seg.y_top_left
        x1, y1 = seg.x_bottom_right, seg.y_bottom_right
        ax.plot(
            [x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0],
            color=color, linewidth=1, linestyle="-",
        )
        ax.text(x0 + 2, y0 + 10, label, color=color, fontsize=5, fontweight="bold")


def _draw_gt_regions(
    ax,
    regions: List[ImageSegment],
    categories: List[int],
    id2name: Dict[int, str],
) -> None:
    """Thick DASHED boxes for ground truth (COCO) regions."""
    if not regions:
        return
    for seg, cat in zip(regions, categories):
        color = COLORS_CLASS[int(cat) % len(COLORS_CLASS)]
        name = id2name.get(int(cat), str(cat))
        x0, y0 = seg.x_top_left, seg.y_top_left
        x1, y1 = seg.x_bottom_right, seg.y_bottom_right
        ax.plot(
            [x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0],
            color=color, linewidth=1.5, linestyle="--",
        )
        ax.text(x1 - 30, y1 - 4, name, color=color, fontsize=5, fontweight="bold")


def _draw_legend(ax) -> None:
    """Compact legend for edge colors and line styles."""
    items = [
        mpatches.Patch(color="lightgray", label="Rows"),
        mpatches.Patch(color="green", label="True edge"),
        mpatches.Patch(color="red", label="False edge"),
        plt.Line2D([0], [0], color="black", linewidth=2.0, linestyle="-",
                   label="Ideal region (solid)"),
        plt.Line2D([0], [0], color="black", linewidth=2.0, linestyle="--",
                   label="GT region (dashed)"),
    ]
    ax.legend(handles=items, loc="lower right", fontsize=5, framealpha=0.8, ncol=1)


# ===================================================================
# HTML report
# ===================================================================


def _build_html(
    img_b64: str,
    pdf_name: str,
    stats: Dict[str, Any],
    metrics: Dict[str, float],
    id2name: Dict[int, str],
) -> str:
    """Self-contained HTML page with embedded image and metric tables."""

    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>\n" for k, v in metrics.items()
    )

    legend_cells = "".join(
        '<span style="display:inline-block;width:14px;height:14px;'
        f'background:{COLORS_CLASS[cid % len(COLORS_CLASS)]};'
        f'margin-right:4px;border:1px solid #333;"></span>'
        f'{name} (id={cid})&nbsp;&nbsp;'
        for cid, name in sorted(id2name.items())
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Visual Data — {pdf_name}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
  .container {{ max-width: 1400px; margin: 0 auto; background: #fff;
               padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.1); }}
  h1 {{ font-size: 1.4em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1em; margin-top: 24px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
  th {{ background: #f0f0f0; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                 gap: 8px; margin-top: 8px; }}
  .stat-card {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
               padding: 10px 14px; text-align: center; }}
  .stat-card .value {{ font-size: 1.6em; font-weight: bold; color: #333; }}
  .stat-card .label {{ font-size: 0.8em; color: #666; }}
  .legend {{ font-size: 0.85em; margin: 8px 0; }}
</style>
</head>
<body>
<div class="container">
<h1>Visual Data: {pdf_name}</h1>

<h2>Page Visualization</h2>
<img src="data:image/png;base64,{img_b64}" alt="Page visualization">

<h2>Legend</h2>
<div class="legend">
  Rows: thin gray &nbsp;|&nbsp;
  Edges: <span style="color:green">green</span> = true,
  <span style="color:red">red</span> = false &nbsp;|&nbsp;
  Ideal regions: thick solid &nbsp;|&nbsp;
  GT regions: thick dashed
</div>
<div class="legend">{legend_cells}</div>

<h2>Statistics</h2>
<div class="stats-grid">
  <div class="stat-card"><div class="value">{stats['num_ideal_regions']}</div><div class="label">Ideal Regions</div></div>
  <div class="stat-card"><div class="value">{stats['num_gt_regions']}</div><div class="label">GT Regions</div></div>
  <div class="stat-card"><div class="value">{stats['num_rows']}</div><div class="label">Rows</div></div>
  <div class="stat-card"><div class="value">{stats['num_edges']}</div><div class="label">Edges</div></div>
  <div class="stat-card"><div class="value">{stats['num_nodes']}</div><div class="label">Nodes</div></div>
  <div class="stat-card"><div class="value">{stats['num_true_edges']}</div><div class="label">True Edges</div></div>
  <div class="stat-card"><div class="value">{stats['num_false_edges']}</div><div class="label">False Edges</div></div>
</div>

<h2>mAP Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {metric_rows}
</table>
</div>
</body>
</html>"""


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML report: rows, edges, ideal regions, GT regions, metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdf_name", help="PDF filename (e.g. pmc1051351_04.pdf)"
    )
    parser.add_argument(
        "--env", default="local",
        help="Env shortcut (local/debug/server/serverML) or path to .env file",
    )
    parser.add_argument("--dir_pdf", default=None, help="Override PDF directory")
    parser.add_argument("--dir_json", default=None, help="Override JSON cache directory")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output HTML path (default: visual_data_{stem}.html)",
    )
    parser.add_argument("--page", type=int, default=0, help="Page number (default: 0)")
    args = parser.parse_args()

    # 1. Env
    env = _load_env(args.env)

    # 2. Locate PDF
    pdf_path = _find_pdf(args.pdf_name, args.dir_pdf, env)
    print(f"PDF:  {pdf_path}")

    # 3. Locate cached JSON
    json_path = _find_json_cache(args.pdf_name, args.dir_json, env)
    print(f"JSON: {json_path}")
    with open(json_path, "r") as f:
        cache_data = json.load(f)

    # 4. Extract rows
    pdf_manager = PDFManager()
    row_manager = RowManager()
    raw_json, img = pdf_manager.get_json_and_img_from_pdf(
        str(pdf_path), num_page=args.page
    )
    if img is None:
        sys.exit(f"Failed to extract page {args.page} from {pdf_path}")
    rows_json = row_manager.get_row_json_from_pdf_json(raw_json)
    page_w, page_h = raw_json["width"], raw_json["height"]
    print(f"Page: {page_w}×{page_h}, {len(rows_json)} rows")

    # 5. Ground truth (COCO)
    coco_path, dataset_name = _resolve_coco(env)
    coco = COCOManager(coco_path=coco_path, name_dataset=dataset_name)
    id2name: Dict[int, str] = dict(coco.coco_classes)
    try:
        true_regions, true_categories = coco(args.pdf_name, raw_json)
    except KeyError:
        true_regions, true_categories = [], []
        print("Warning: no ground truth for this PDF in COCO")

    # clean tiny annotations
    clean_r, clean_c = [], []
    for seg, cat in zip(true_regions, true_categories):
        if seg.height > 3 and seg.width > 3:
            clean_r.append(seg); clean_c.append(cat)
    true_regions, true_categories = clean_r, clean_c
    print(f"GT regions: {len(true_regions)}")

    # 6. Graph data from cache
    inds: List[List[int]] = cache_data.get("inds", [[], []])
    true_edges: List[int] = cache_data.get("true_edges", [])
    true_nodes: List[Optional[int]] = cache_data.get("true_nodes", [])

    n_nodes = cache_data.get("N", len(rows_json))
    n_edges = len(true_edges)
    n_true = sum(1 for e in true_edges if e)
    n_false = n_edges - n_true
    print(f"Graph: {n_nodes} nodes, {n_edges} edges ({n_true} true, {n_false} false)")

    # 7. Ideal regions (from true edges/nodes)
    ideal_segs, ideal_labels, ideal_cls = _compute_ideal_regions(
        rows_json, inds, true_edges, true_nodes, id2name
    )
    print(f"Ideal regions: {len(ideal_segs)}")

    # 8. Metrics
    pred_boxes = [[s.x_top_left, s.y_top_left, s.width, s.height] for s in ideal_segs]
    true_boxes = [[s.x_top_left, s.y_top_left, s.width, s.height] for s in true_regions]
    if true_regions:
        metrics = _compute_mAP(pred_boxes, ideal_cls, true_boxes, true_categories, id2name)
    else:
        metrics = {}
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 9. Render
    img_b64 = _render_page(
        img=img, rows_json=rows_json, inds=inds, true_edges=true_edges,
        ideal_segments=ideal_segs, ideal_labels=ideal_labels,
        true_regions=true_regions, true_categories=true_categories,
        id2name=id2name, page_w=page_w, page_h=page_h,
        title=Path(args.pdf_name).stem,
    )

    # 10. HTML report
    stats = {
        "num_ideal_regions": len(ideal_segs),
        "num_gt_regions": len(true_regions),
        "num_rows": len(rows_json),
        "num_edges": n_edges,
        "num_nodes": n_nodes,
        "num_true_edges": n_true,
        "num_false_edges": n_false,
    }
    html = _build_html(img_b64, args.pdf_name, stats, metrics, id2name)

    output_path = args.output or f"visual_data_{Path(args.pdf_name).stem}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()