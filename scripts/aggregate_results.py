#!/usr/bin/env python3
"""Aggregate experiment results across random seeds.

Reads a CSV file of per-seed experiment results (e.g. from
results/paper/p02_font_feature/results.csv), groups rows by base
experiment name (stripping the ``_seed_N`` suffix), and for each
numeric metric column computes the mean and 3x sample standard
deviation across seeds.

Output:
    - Formatted text table to stdout.
    - Aggregated CSV saved next to the input file (``_aggregated`` suffix).

Usage::

    python scripts/aggregate_results.py results/paper/p02_font_feature/results.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def _strip_seed_suffix(name: str) -> str:
    """Remove the ``_seed_<N>`` suffix from an experiment name.

    >>> _strip_seed_suffix("font_emb_seed_0")
    'font_emb'
    >>> _strip_seed_suffix("no_font")
    'no_font'
    """
    return re.sub(r"_seed_\d+$", "", name)


def _is_all_minus_one(series: pd.Series) -> bool:
    """Check whether every value in *series* equals -1.0 (class absent)."""
    return (series == -1.0).all()


def _aggregate_group(
    group: pd.DataFrame,
    metric_cols: list[str],
) -> dict[str, str]:
    """For a single base-experiment group, compute mean ± 3σ for each metric column.

    If every value in the column equals -1.0 (class absent from dataset),
    the result is the literal ``"-1.0"``.  Otherwise the format is
    ``"{mean:.4f} ± {3*std:.4f}"`` (ddof=1).
    """
    result: dict[str, str] = {}
    for col in metric_cols:
        values = group[col]
        if _is_all_minus_one(values):
            result[col] = "-1.0"
        else:
            mean = values.mean()
            std = values.std(ddof=1)
            result[col] = f"{mean:.4f} ± {3 * std:.4f}"
    return result


def _format_table(df: pd.DataFrame) -> str:
    """Return a pretty-printed text table with aligned columns."""
    if df.empty:
        return "(empty)"

    # Determine column widths
    col_widths: dict[str, int] = {}
    for col in df.columns:
        max_val_len = max(len(str(val)) for val in df[col])
        col_widths[col] = max(len(str(col)), max_val_len)

    lines: list[str] = []
    # Header
    header = "  ".join(str(col).ljust(col_widths[col]) for col in df.columns)
    lines.append(header)
    lines.append("-" * len(header))
    # Data rows
    for _, row in df.iterrows():
        line = "  ".join(
            str(row[col]).ljust(col_widths[col]) for col in df.columns
        )
        lines.append(line)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results across random seeds"
    )
    parser.add_argument("csv_path", help="Path to results CSV file")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path, index_col=0)

    # ------------------------------------------------------------------
    # 2. Identify metric (numeric) columns — skip 'name' and index cols
    # ------------------------------------------------------------------
    metric_cols = [
        col
        for col in df.columns
        if col != "name" and pd.api.types.is_numeric_dtype(df[col])
    ]

    # ------------------------------------------------------------------
    # 3. Derive base experiment name and group
    # ------------------------------------------------------------------
    df["_base_name"] = df["name"].apply(_strip_seed_suffix)

    rows: list[dict[str, str]] = []
    for base_name, group in df.groupby("_base_name", sort=False):
        row = _aggregate_group(group, metric_cols)
        row["experiment"] = base_name
        rows.append(row)

    # ------------------------------------------------------------------
    # 4. Build result DataFrame, experiment column first
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(rows)[["experiment", *metric_cols]]

    # ------------------------------------------------------------------
    # 5. Output
    # ------------------------------------------------------------------
    print(_format_table(result_df))
    print()

    out_path = csv_path.parent / f"{csv_path.stem}_aggregated.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved aggregated CSV → {out_path}")


if __name__ == "__main__":
    main()
