#!/usr/bin/env python3
"""
Baseline script: compute per-class statistics for EVERY feature index of the
tokenizer's node feature vector.

For each PDF page, rows are assigned to COCO classes via geometric
intersection with ground-truth annotations. The full multi-dimensional
node feature vector produced by ``RowGLAMTokenizer.get_node_features()``
is then accumulated per class, and per-feature-index mean and stddev
tables are printed and saved to CSV.

Supported tokenizers
--------------------
  --tokenizer base           RowGLAMTokenizer (base_line_tokenizer), 21 features
  --tokenizer font           RowGLAMTokenizer (font_tokenizer), 24 features
  --tokenizer font_emb_512   RowGLAMTokenizer (font_emb_tokenizer, size=512),
                             533 features (21 base + 512 font embedding)
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values

from rows2regionsGLAM.pred_processor import PredProcessor
from rows2regionsGLAM.utils.coco_manager import COCOManager

# ── PublayNet / DocLayNet class constants ──
CLASS_NAMES: dict[int, str] = {
    0: "other",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure",
}


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════

def get_mini_seg(row_segment: dict):
    """Shrink row height by 20% (delta = height/5) to avoid boundary overlaps."""
    from pagerlib.dtypes import ImageSegment

    img_seg = ImageSegment(dict_p_size=row_segment)
    if img_seg.height < 5:
        return img_seg
    delta = int(img_seg.height / 5)
    img_seg.y_bottom_right = img_seg.y_bottom_right - delta
    img_seg.y_top_left = img_seg.y_top_left + delta
    return img_seg


def get_category(seg, region_segs, region_categories) -> Optional[int]:
    """Return class ID if *seg* intersects any region; ``None`` otherwise."""
    for r, c in zip(region_segs, region_categories):
        if seg.is_intersection(r):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Output helpers
# ═══════════════════════════════════════════════════════════════════════

def print_stat_table(
    header: list[str],
    rows: list[list[str]],
    title: Optional[str] = None,
) -> None:
    """Print a formatted table with aligned columns."""
    if title:
        print(f"  {title}")
    col_widths = [
        max(len(str(row[i])) for row in rows + [header])
        for i in range(len(header))
    ]
    separator = "─" * (sum(col_widths) + 3 * (len(col_widths) - 1))
    print(f"  {separator}")
    header_line = "  ".join(
        f"{h:<{col_widths[i]}}" for i, h in enumerate(header)
    )
    print(f"  {header_line}")
    print(f"  {separator}")
    for row in rows:
        line = "  ".join(
            f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))
        )
        print(f"  {line}")
    print()


def build_feature_legend(tokenizer) -> list[tuple[int, str, str]]:
    """Build a flat legend: list of (index, name, description) tuples.

    Expands multi-index entries from ``get_dict_vec()`` into individual
    rows (e.g. ``dot_vec`` → ``dot_vec[0]``, ``dot_vec[1]``, ...).  For
    font-embedding features beyond the named ones, generates
    ``font_emb[i]`` entries.
    """
    dict_vec = tokenizer.get_dict_vec()

    # Collect all occupied indices to detect gaps filled by embeddings.
    occupied: set[int] = set()
    named_entries: list[tuple[str, list[int]]] = []
    for name, indices in dict_vec.items():
        for idx in indices:
            occupied.add(int(idx))
        named_entries.append((name, [int(i) for i in indices]))

    # Determine total feature count from tokenizer.
    # Build a dummy (0,0) image and an empty row to inspect output shape.
    import numpy as np
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_row = {
        "segment": {"x_tl": 0, "y_tl": 0, "x_br": 5, "y_br": 5},
        "words": [],
    }
    try:
        feats = tokenizer.get_node_features([dummy_row], dummy_img)
        total_features = len(feats[0]) if feats else 0
    except Exception:
        total_features = max(occupied) + 1 if occupied else 21

    legend: list[tuple[int, str, str]] = []
    for name, indices in named_entries:
        if len(indices) == 1:
            legend.append((indices[0], name, _describe_feature(name, 0)))
        else:
            for sub_i, idx in enumerate(indices):
                legend.append((idx, f"{name}[{sub_i}]", _describe_feature(name, sub_i)))

    # Fill remaining indices (font embeddings, etc.)
    for i in range(total_features):
        if i not in occupied:
            legend.append((i, f"font_emb[{i - min(occupied) - 0}]", "Font embedding dimension"))

    legend.sort(key=lambda x: x[0])
    return legend


def _describe_feature(name: str, sub_idx: int) -> str:
    """Return a human-readable description for a named feature."""
    descriptions: dict[str, str] = {
        "x_top_left": "Left x coordinate of row bounding box",
        "x_bottom_right": "Right x coordinate of row bounding box",
        "width": "Width of row bounding box",
        "y_top_left": "Top y coordinate of row bounding box",
        "y_bottom_right": "Bottom y coordinate of row bounding box",
        "height": "Height of row bounding box",
        "norm_geom": [
            "Normalized y_top_left / page_h",
            "Normalized x_top_left / page_w",
            "Normalized (y_top_left - page_y_min) / page_h",
            "Normalized (x_top_left - page_x_min) / page_w",
        ],
        "dot_vec": [
            "Ends with '.'",
            "Ends with ','",
            "Ends with ';'",
            "Ends with ':'",
            "Ends with '?'",
            "Ends with '!'",
        ],
        "super_vec": "Fraction of uppercase characters",
        "list_ind_vec": "List indicator (regex match)",
        "heuristics_vec": [
            "text_length / (width/height)",
            "digit_count / text_length",
            "log(1 + text_length)",
        ],
        "font_width": "Max font width across words",
        "font_italic": "Max font italic flag across words",
        "font_size": "Median font size across words",
    }

    desc = descriptions.get(name, name)
    if isinstance(desc, list):
        return desc[sub_idx] if sub_idx < len(desc) else f"{name}[{sub_idx}]"
    return desc


def save_feature_csv(
    path: Path,
    matrix: dict[int, list[list[float]]],
    agg_func,
    n_features: int,
) -> None:
    """Save a per-class × per-feature table to CSV.

    Args:
        path: Output CSV file path.
        matrix: ``class_id → list of feature vectors``.
        agg_func: Aggregation function (e.g. ``statistics.mean``).
        n_features: Number of feature columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["class_id", "class_name"] + [f"f{i}" for i in range(n_features)]

    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cls_id in sorted(matrix.keys()):
            vecs = matrix[cls_id]
            if not vecs:
                continue
            row_data: dict[str, str] = {
                "class_id": str(cls_id),
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            }
            for fi in range(n_features):
                col_vals = [v[fi] for v in vecs if fi < len(v)]
                if col_vals:
                    val = agg_func(col_vals)
                    row_data[f"f{fi}"] = f"{val:.4f}"
                else:
                    row_data[f"f{fi}"] = "NaN"
            writer.writerow(row_data)

    print(f"  Results saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Tokenizer factory
# ═══════════════════════════════════════════════════════════════════════

def make_tokenizer(tokenizer_name: str):
    """Instantiate the requested tokenizer variant.

    Args:
        tokenizer_name: One of ``'base'``, ``'font'``, ``'font_emb_512'``.

    Returns:
        Tuple of ``(tokenizer_instance, label)``.
    """
    from rows2regionsGLAM.tokenizers.base_line_tokenizer.base_line_tokenizer import (
        RowGLAMTokenizer as BaseTokenizer,
    )
    from rows2regionsGLAM.tokenizers.font_tokenizer.font_tokenizer import (
        RowGLAMTokenizer as FontTokenizer,
    )
    from rows2regionsGLAM.tokenizers.font_emb_tokenizer.font_tokenizer import (
        RowGLAMTokenizer as EmbFontTokenizer,
    )

    if tokenizer_name == "base":
        return BaseTokenizer(), "base"
    elif tokenizer_name == "font":
        return FontTokenizer(), "font"
    elif tokenizer_name == "font_emb_512":
        return EmbFontTokenizer(size=512), "font_emb_512"
    else:
        raise ValueError(
            f"Unknown tokenizer '{tokenizer_name}'. "
            f"Valid: base, font, font_emb_512"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline: per-class tokenizer feature statistics"
    )
    parser.add_argument(
        '--env', default='local',
        help='Env shortcut (debug/local/server) or path to .env',
    )
    parser.add_argument(
        '--pages', type=int, default=2000,
        help='Number of pages to evaluate (default: 2000)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for page subset selection',
    )
    parser.add_argument(
        '--tokenizer', default='base',
        choices=['base', 'font', 'font_emb_512'],
        help='Tokenizer variant (default: base)',
    )
    args = parser.parse_args()

    # ── Resolve env ──
    from scripts.configs import resolve_env_file

    env_path = resolve_env_file(args.env)
    env_vars = dict(dotenv_values(env_path))

    pdf_dir = env_vars['TEST_PATH']
    coco_path = env_vars['TEST_COCO_PATH']
    name_dataset = env_vars.get('NAME_TEST_DATASET', 'publaynet')

    # Make paths absolute if relative
    for key in ('TEST_PATH', 'TEST_COCO_PATH'):
        p = Path(env_vars[key])
        if not p.is_absolute():
            env_vars[key] = str(PROJECT_ROOT / p)
    pdf_dir = Path(env_vars['TEST_PATH'])
    coco_path = Path(env_vars['TEST_COCO_PATH'])

    print(f"Env:        {env_path}")
    print(f"Tokenizer:  {args.tokenizer}")
    print(f"PDF dir:    {pdf_dir}")
    print(f"COCO:       {coco_path}")
    print(f"Max pages:  {args.pages}")
    print()

    # ── Init tokenizer ──
    tokenizer, tokenizer_label = make_tokenizer(args.tokenizer)
    print(f"Tokenizer instance: {tokenizer_label}")
    print()

    # ── Init ──
    print("Initialising COCOManager ...")
    coco_manager = COCOManager(
        loger=None,
        coco_path=str(coco_path),
        name_dataset=name_dataset,
    )
    print(f"  Classes: {coco_manager.classes}")
    print(f"  Regions for {len(coco_manager.regions)} PDFs")

    print("Initialising PredProcessor ...")
    pred = PredProcessor(loger=None)

    # ── Collect PDF names (only those with COCO annotations) ──
    all_pdfs = sorted(
        f for f in os.listdir(pdf_dir)
        if f.endswith('.pdf') and f in coco_manager.regions
    )
    print(f"PDFs with annotations: {len(all_pdfs)}")

    rng = random.Random(args.seed)
    if len(all_pdfs) > args.pages:
        subset = rng.sample(all_pdfs, args.pages)
    else:
        subset = all_pdfs
    subset = sorted(subset)
    print(f"Evaluating {len(subset)} pages (seed={args.seed})")
    print()

    # ── Accumulators ──
    # per_class_features[class_id] -> list of feature vectors (each a list of floats)
    per_class_features: dict[int, list[list[float]]] = defaultdict(list)
    per_class_invalid: dict[int, int] = defaultdict(int)
    class_counter: dict[int, int] = defaultdict(int)
    total_rows = 0
    skipped = 0

    t_start = time.time()

    # ── Main loop ──
    for i, pdf_name in enumerate(subset):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(subset)}] {rate:.1f} pages/s  "
                  f"rows={total_rows}  skipped={skipped}")

        try:
            result = pred(pdf_dir / pdf_name)
        except Exception as e:
            print(f"  SKIP {pdf_name}: {e}")
            skipped += 1
            continue

        rows_json = result['pdf_json']['rows']
        pdf_img = result['img']  # numpy image from PredProcessor

        # Get ground truth
        true_regions, true_categories = coco_manager(pdf_name, result['pdf_json'])

        # Build row→category mapping via intersection
        from pagerlib.dtypes import ImageSegment

        row_segments = [
            get_mini_seg(row['segment']) for row in rows_json
        ]

        true_nodes = [
            get_category(row_seg, true_regions, true_categories)
            for row_seg in row_segments
        ]

        # Compute tokenizer feature vectors for ALL rows in one call.
        try:
            all_features = tokenizer.get_node_features(rows_json, pdf_img)
        except Exception as e:
            print(f"  SKIP {pdf_name} (tokenizer error): {e}")
            skipped += 1
            continue

        if not all_features or not all_features[0]:
            # Empty feature output — skip page
            skipped += 1
            continue

        n_features = len(all_features[0]) if all_features else 0

        for row_json, row_seg, true_cls, feat_vec in zip(
            rows_json, row_segments, true_nodes, all_features
        ):
            # None → class 0 (other)
            cls_id = true_cls if true_cls is not None else 0

            total_rows += 1
            class_counter[cls_id] += 1

            # Check validity: feature vector is non-empty and contains no NaN/inf.
            if not feat_vec:
                per_class_invalid[cls_id] += 1
                continue
            try:
                _floats = [float(v) for v in feat_vec]
            except (ValueError, TypeError):
                per_class_invalid[cls_id] += 1
                continue

            per_class_features[cls_id].append(_floats)

    t_total = time.time() - t_start

    # ── Determine total feature count ──
    all_class_ids = sorted(class_counter.keys())
    # Find n_features from accumulated data
    if all_class_ids:
        for cls_id in all_class_ids:
            if per_class_features[cls_id]:
                n_features = len(per_class_features[cls_id][0])
                break
        else:
            n_features = 0
    else:
        n_features = 0

    # ── Print summary ──
    print()
    print("=" * 72)
    print("  BASELINE: TOKENIZER FEATURE STATISTICS BY CLASS")
    print("=" * 72)
    print(f"  Tokenizer:                 {tokenizer_label}")
    print(f"  Feature vector dimension:  {n_features}")
    print(f"  Pages processed:           {len(subset) - skipped}")
    print(f"  Pages skipped (errors):    {skipped}")
    print(f"  Total rows:                {total_rows}")
    print(f"  Processing time:           {t_total:.1f}s")
    print(f"  Rate:                      {len(subset) / t_total:.1f} pages/s")
    print()

    # ── Class distribution ──
    print(f"  Class distribution:")
    for cls_id in sorted(class_counter.keys()):
        name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        cnt = class_counter[cls_id]
        pct = cnt / total_rows * 100 if total_rows > 0 else 0.0
        print(f"    class {cls_id} ({name:7s}): {cnt:8d}  ({pct:5.1f}%)")
    print()

    if n_features == 0:
        print("  No feature data collected. Exiting.")
        return

    # ── Console display limit for large tokenizers ──
    console_feature_limit = 30 if args.tokenizer == 'font_emb_512' else n_features
    display_range_feat = min(n_features, console_feature_limit)

    # ── Compute mean per class per feature ──
    print("=" * 72)
    print("  MEAN FEATURE VALUE PER CLASS PER FEATURE INDEX")
    print("=" * 72)
    mean_header = ["Class"] + [f"f{i}" for i in range(display_range_feat)]
    mean_rows: list[list[str]] = []
    for cls_id in sorted(all_class_ids):
        vecs = per_class_features.get(cls_id, [])
        if not vecs:
            continue
        name = CLASS_NAMES.get(cls_id, f'cls_{cls_id}')
        label = f"{cls_id} ({name})"
        row_data = [label]
        for fi in range(display_range_feat):
            col_vals = [v[fi] for v in vecs if fi < len(v)]
            mean_val = statistics.mean(col_vals) if col_vals else float('nan')
            row_data.append(f"{mean_val:.2f}")
        mean_rows.append(row_data)
    if n_features > display_range_feat:
        print(f"  (Showing first {display_range_feat} of {n_features} features)")
    print_stat_table(mean_header, mean_rows, title=None)
    print()

    # ── Compute stddev per class per feature ──
    print("=" * 72)
    print("  STDDEV FEATURE VALUE PER CLASS PER FEATURE INDEX")
    print("=" * 72)
    std_header = ["Class"] + [f"f{i}" for i in range(display_range_feat)]
    std_rows: list[list[str]] = []
    for cls_id in sorted(all_class_ids):
        vecs = per_class_features.get(cls_id, [])
        if not vecs:
            continue
        name = CLASS_NAMES.get(cls_id, f'cls_{cls_id}')
        label = f"{cls_id} ({name})"
        row_data = [label]
        for fi in range(display_range_feat):
            col_vals = [v[fi] for v in vecs if fi < len(v)]
            if len(col_vals) >= 2:
                std_val = statistics.stdev(col_vals)
            else:
                std_val = float('nan')
            row_data.append(f"{std_val:.2f}")
        std_rows.append(row_data)
    if n_features > display_range_feat:
        print(f"  (Showing first {display_range_feat} of {n_features} features)")
    print_stat_table(std_header, std_rows, title=None)
    print()

    # ── Feature index legend ──
    legend = build_feature_legend(tokenizer)
    print("=" * 72)
    print("  FEATURE INDEX LEGEND")
    print("=" * 72)
    legend_header = ["Index", "Name", "Description"]
    legend_table_rows: list[list[str]] = []
    display_legend_limit = console_feature_limit
    for idx, name, desc in legend[:display_legend_limit]:
        legend_table_rows.append([str(idx), name, desc])
    if len(legend) > display_legend_limit:
        print(f"  (Showing first {display_legend_limit} of {len(legend)} features)")
    print_stat_table(legend_header, legend_table_rows, title=None)
    print()

    # ── Save CSVs (full feature set) ──
    mean_csv_path = PROJECT_ROOT / "results" / "baseline_tokenizer_features_mean.csv"
    std_csv_path = PROJECT_ROOT / "results" / "baseline_tokenizer_features_stddev.csv"

    save_feature_csv(mean_csv_path, per_class_features, statistics.mean, n_features)
    save_feature_csv(std_csv_path, per_class_features, _safe_stdev, n_features)

    print()
    print("=" * 72)
    print("  DONE.")
    print("=" * 72)


def _safe_stdev(values: list[float]) -> float:
    """Return stdev of *values*, or 0.0 if fewer than 2 data points."""
    if len(values) >= 2:
        return statistics.stdev(values)
    return 0.0


if __name__ == '__main__':
    main()
