"""p10 — Cluster Analysis of Tokenizer Feature Vectors.

Evaluates a priori class separability of raw tokenizer feature vectors
(before GNN training) using KMeans clustering on region-level mean-pooled features.

Uses existing p02 test caches (tmp_feature_{name}/test/) to avoid rebuilding.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
PUBLAYNET_CLASSES = {
    0: "other",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure",
}

ANALYSIS_CLASSES = [1, 2, 3, 4, 5]  # Exclude "other" (0)
CLASS_NAMES = [PUBLAYNET_CLASSES[c] for c in ANALYSIS_CLASSES]

TOKENIZER_CONFIGS = [
    {
        "name": "no_font",
        "input_dim": 21,
        "cache_dir": "tmp_feature_no_font",
    },
    {
        "name": "pdf_font",
        "input_dim": 21+3,
        "cache_dir": "tmp_feature_pdf_font",
    },
    {
        "name": "font_emb_16",
        "input_dim": 21+16,
        "cache_dir": "tmp_feature_font_emb_16",
    },
    {
        "name": "font_emb_32",
        "input_dim": 21+32,
        "cache_dir": "tmp_feature_font_emb_32",
    },
    {
        "name": "font_emb",
        "input_dim": 21+512,
        "cache_dir": "tmp_feature_font_emb",
    },
]

# Project root: rows2regionsGLAM/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _get_cache_dir(config: dict) -> Path:
    """Resolve the test cache directory for a tokenizer config."""
    cache_path = _PROJECT_ROOT / config["cache_dir"] / "test"
    return cache_path


# ──────────────────────────────────────────────────────────────────────
# Step 1: Collect region vectors from cached JSONs
# ──────────────────────────────────────────────────────────────────────

def _load_document_data(json_path: Path) -> Optional[dict]:
    """Load a single cached document JSON.

    Returns dict with keys: X, inds, true_edges, true_nodes, N
    or None if loading fails.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None

    required = {"X", "inds", "true_edges", "true_nodes"}
    if not required.issubset(data.keys()):
        missing = required - data.keys()
        logger.warning(f"Missing keys {missing} in {json_path}")
        return None

    # Convert X to numpy array
    X = np.array(data["X"], dtype=np.float64)
    if X.ndim != 2:
        logger.warning(f"X has unexpected shape {X.shape} in {json_path}")
        return None

    return {
        "X": X,
        "inds": data["inds"],
        "true_edges": data["true_edges"],
        "true_nodes": data["true_nodes"],
        "N": data.get("N", X.shape[0]),
    }


def _build_regions_from_doc(doc: dict) -> List[Tuple[np.ndarray, int]]:
    """Build regions from a document's data.

    Each region is a connected component in the graph where nodes = rows
    and edges = adjacent rows belonging to the same ground-truth region
    (true_edges[k] == 1).

    Returns list of (mean_vector, majority_class) for each valid region.
    """
    X = doc["X"]
    inds = doc["inds"]
    true_edges = doc["true_edges"]
    true_nodes = doc["true_nodes"]
    N = len(X)

    if N == 0:
        return []

    # Handle None in true_nodes: None → 0 (other)
    node_classes = [n if n is not None else 0 for n in true_nodes]

    # Build adjacency for edges where true_edges[k] == 1
    # Use simple union-find for connected components
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    if inds:
        for k in range(len(inds[0])):
            if k < len(true_edges) and true_edges[k] == 1:
                u, v = int(inds[0][k]), int(inds[1][k])
                if 0 <= u < N and 0 <= v < N:
                    union(u, v)

    # Group nodes by component
    comp_to_nodes: Dict[int, List[int]] = {}
    for i in range(N):
        root = find(i)
        comp_to_nodes.setdefault(root, []).append(i)

    # Build regions
    regions = []
    for comp_nodes in comp_to_nodes.values():
        # Get classes in region
        classes_in_region = [node_classes[i] for i in comp_nodes]

        # Exclude "other" nodes (class 0)
        meaningful = [c for c in classes_in_region if c != 0]
        if not meaningful:
            continue

        # Majority vote
        class_counts = Counter(meaningful)
        majority_class = class_counts.most_common(1)[0][0]

        # Mean-pool features
        mu = X[comp_nodes].mean(axis=0)

        regions.append((mu, majority_class))

    return regions


def collect_region_vectors(
    cache_dir: Path, name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect region-level feature vectors for a tokenizer.

    Iterates over all cached JSON documents in cache_dir, builds regions,
    and collects mean-pooled feature vectors with majority-vote class labels.

    Args:
        cache_dir: Path to test cache directory.
        name: Tokenizer name for logging.

    Returns:
        X_all: [N_regions, D] feature vectors.
        y_all: [N_regions] class labels (1..5).
    """
    json_files = sorted(cache_dir.glob("*.json"))
    total = len(json_files)
    logger.info(f"[{name}] Found {total} cached documents")

    all_vectors = []
    all_labels = []
    skipped = 0

    for idx, json_path in enumerate(json_files):
        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(f"[{name}] Processing doc {idx + 1}/{total}")

        doc = _load_document_data(json_path)
        if doc is None:
            skipped += 1
            continue

        regions = _build_regions_from_doc(doc)
        for vec, label in regions:
            if label in ANALYSIS_CLASSES:
                all_vectors.append(vec)
                all_labels.append(label)

    if skipped:
        logger.warning(f"[{name}] Skipped {skipped} documents due to errors")

    X = np.array(all_vectors, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int64)

    logger.info(
        f"[{name}] Collected {X.shape[0]} regions "
        f"(dim={X.shape[1] if X.shape[0] > 0 else 0})"
    )
    return X, y


# ──────────────────────────────────────────────────────────────────────
# Step 2: KMeans clustering
# ──────────────────────────────────────────────────────────────────────


def run_kmeans(
    X: np.ndarray, n_clusters: int = 20, random_state: int = 42
) -> np.ndarray:
    """StandardScaler → KMeans → cluster labels."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(X_scaled)


# ──────────────────────────────────────────────────────────────────────
# Step 3: Confusion matrix & cluster naming
# ──────────────────────────────────────────────────────────────────────


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    cluster_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build confusion matrix DataFrame (rows=clusters, cols=classes)."""
    n_clusters = len(np.unique(y_pred))
    n_classes = len(class_names)

    # Map class IDs to column indices
    class_id_to_col = {c: i for i, c in enumerate(ANALYSIS_CLASSES)}

    matrix = np.zeros((n_clusters, n_classes), dtype=int)
    for true_label, cluster_label in zip(y_true, y_pred):
        col = class_id_to_col.get(int(true_label))
        if col is not None:
            matrix[cluster_label, col] += 1

    if cluster_names is None:
        cluster_names = [f"Cluster {i}" for i in range(n_clusters)]

    df = pd.DataFrame(
        matrix,
        index=cluster_names,
        columns=class_names,
    )
    return df


def assign_cluster_names(
    confusion: pd.DataFrame, class_names: List[str]
) -> List[str]:
    """Assign descriptive names to clusters based on majority class.

    Rules:
    - If majority >= 50% and second_best <= 30%: "{MajorityClass} cluster"
    - Otherwise: "Mixed {class1}+{class2} cluster"
    """
    names = []
    for idx in confusion.index:
        row = confusion.loc[idx]
        if row.sum() == 0:
            names.append(f"Empty cluster")
            continue

        proportions = row / row.sum()
        sorted_items = proportions.sort_values(ascending=False)

        majority_class = sorted_items.index[0]
        majority_pct = sorted_items.iloc[0]
        second_pct = sorted_items.iloc[1] if len(sorted_items) > 1 else 0.0

        if majority_pct >= 0.5 and second_pct <= 0.3:
            names.append(f"{majority_class} cluster")
        else:
            second_class = sorted_items.index[1]
            names.append(f"Mixed {majority_class}+{second_class} cluster")

    return names


# ──────────────────────────────────────────────────────────────────────
# Step 4: Metrics
# ──────────────────────────────────────────────────────────────────────


def compute_purity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute cluster purity, ARI, NMI.

    Returns dict with: overall_purity, per_cluster_purity, ARI, NMI.
    """
    n_clusters = len(np.unique(y_pred))

    total_samples = len(y_true)
    sum_max = 0
    per_cluster = []

    for cluster_id in range(n_clusters):
        mask = y_pred == cluster_id
        cluster_size = mask.sum()
        if cluster_size == 0:
            per_cluster.append(0.0)
            continue

        true_in_cluster = y_true[mask]
        counts = Counter(true_in_cluster)
        max_count = counts.most_common(1)[0][1]
        purity = max_count / cluster_size
        per_cluster.append(purity)
        sum_max += max_count

    overall_purity = sum_max / total_samples if total_samples > 0 else 0.0

    # Map class IDs (1-5) to 0-4 for sklearn metrics
    class_id_to_idx = {c: i for i, c in enumerate(ANALYSIS_CLASSES)}
    y_true_mapped = np.array([class_id_to_idx.get(int(y), -1) for y in y_true])
    valid = y_true_mapped >= 0
    y_true_mapped = y_true_mapped[valid]
    y_pred_valid = y_pred[valid]

    ari = adjusted_rand_score(y_true_mapped, y_pred_valid)
    nmi = normalized_mutual_info_score(y_true_mapped, y_pred_valid)

    return {
        "overall_purity": float(overall_purity),
        "per_cluster_purity": [float(p) for p in per_cluster],
        "ARI": float(ari),
        "NMI": float(nmi),
    }


# ──────────────────────────────────────────────────────────────────────
# Step 5: Output
# ──────────────────────────────────────────────────────────────────────


def _ensure_output_dir(output_dir: Path) -> Path:
    """Create output directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_confusion(
    confusion: pd.DataFrame, output_dir: Path, name: str
) -> Path:
    """Save confusion matrix CSV."""
    path = output_dir / f"{name}_confusion.csv"
    confusion.to_csv(path)
    return path


def _save_metrics(metrics: dict, cluster_names: List[str], output_dir: Path, name: str) -> Path:
    """Save metrics JSON."""
    out = dict(metrics)
    out["cluster_names"] = cluster_names
    path = output_dir / f"{name}_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def _save_summary(summary_rows: List[dict], output_dir: Path) -> Optional[Path]:
    """Save all-metrics summary CSV.

    If no data was collected (empty rows), logs a warning and returns None.
    """
    df = pd.DataFrame(summary_rows)
    if df.empty or "tokenizer" not in df.columns:
        logger.warning("No summary data to save — all tokenizers may have failed.")
        return None
    df = df.set_index("tokenizer")
    path = output_dir / "all_metrics.csv"
    df.to_csv(path)
    return path


# ──────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────


def run_all_tokenizers() -> None:
    """Run clustering analysis for all 5 tokenizers and save results."""
    output_dir = _ensure_output_dir(
        _PROJECT_ROOT / "experiments" / "paper" / "p10_cluster" / "outputs"
    )

    all_summary = []

    for config in TOKENIZER_CONFIGS:
        name = config["name"]
        cache_dir = _get_cache_dir(config)

        if not cache_dir.exists():
            logger.error(f"[{name}] Cache dir not found: {cache_dir}")
            continue

        logger.info(f"{'='*60}")
        logger.info(f"Processing tokenizer: {name}")
        logger.info(f"Cache dir: {cache_dir}")
        logger.info(f"{'='*60}")

        # Step 1: Collect region vectors
        X, y = collect_region_vectors(cache_dir, name)

        if X.shape[0] < 20:
            logger.warning(f"[{name}] Not enough regions ({X.shape[0]}), skipping")
            continue

        # Step 2: KMeans clustering
        cluster_labels = run_kmeans(X, n_clusters=20, random_state=42)
        logger.info(f"[{name}] KMeans done: {len(np.unique(cluster_labels))} clusters")

        # Step 3: Confusion matrix
        confusion = build_confusion_matrix(y, cluster_labels, CLASS_NAMES)
        cluster_names = assign_cluster_names(confusion, CLASS_NAMES)
        confusion.index = cluster_names
        logger.info(f"[{name}] Confusion matrix:\n{confusion}")

        # Step 4: Metrics
        metrics = compute_purity_metrics(y, cluster_labels)
        logger.info(
            f"[{name}] Purity={metrics['overall_purity']:.4f}, "
            f"ARI={metrics['ARI']:.4f}, NMI={metrics['NMI']:.4f}"
        )

        # Step 5: Save
        _save_confusion(confusion, output_dir, name)
        _save_metrics(metrics, cluster_names, output_dir, name)

        all_summary.append(
            {
                "tokenizer": name,
                "purity": round(metrics["overall_purity"], 4),
                "ARI": round(metrics["ARI"], 4),
                "NMI": round(metrics["NMI"], 4),
                "input_dim": config["input_dim"],
                "n_regions": X.shape[0],
            }
        )

    # Save summary
    _save_summary(all_summary, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY — Cluster Purity Metrics (before GNN training)")
    print("=" * 70)
    summary_df = pd.DataFrame(all_summary).set_index("tokenizer")
    print(summary_df.to_string())
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
