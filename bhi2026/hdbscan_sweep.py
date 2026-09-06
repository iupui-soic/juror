"""HDBSCAN parameter sweep on the journal BERTopic clustering.

Primary config: min_cluster_size=20, min_samples=10 — yielded
106 topics, 31% outliers (above the pre-registered 15-22% target).

This sweep fixes the UMAP step (deterministic seed=42) and varies the HDBSCAN
parameters. For each config we report:
  - outlier ratio
  - number of topics
  - mean/std cluster size
  - silhouette score (sklearn, on UMAP-reduced space)
  - topic diversity (mean pairwise cosine distance between topic centroids in
    UMAP space)

Output:
  - bhi2026/phase3_method_a/hdbscan_sweep_results.csv
  - bhi2026/phase3_method_a/hdbscan_sweep.png

Outputs let us pick the best config to rerun method_a_bertopic_ot.py with.
This script does NOT modify the existing Method A artifacts — it's diagnostic.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import hdbscan
import umap
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

OUT_DIR = Path("bhi2026/phase3_method_a")
EMBEDS = OUT_DIR / "journal_embeddings.npy"

UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
SEED = 42

# Sweep grid: (min_cluster_size, min_samples)
SWEEP = [
    (10, 3),
    (10, 5),
    (15, 5),
    (15, 10),
    (20, 5),
    (20, 10),    # current baseline
    (25, 10),
    (30, 10),
    (30, 15),
    (40, 20),
    (50, 25),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topic_diversity(reduced: np.ndarray, labels: np.ndarray) -> float:
    """Mean pairwise cosine distance between non-outlier cluster centroids
    in UMAP-reduced space."""
    centroids = []
    for t in sorted(set(labels)):
        if t == -1:
            continue
        members = reduced[labels == t]
        if len(members) == 0:
            continue
        centroids.append(members.mean(axis=0))
    if len(centroids) < 2:
        return 0.0
    C = np.stack(centroids)
    dist = cosine_distances(C)
    iu = np.triu_indices_from(dist, k=1)
    return float(dist[iu].mean())


def main() -> None:
    log(f"Loading journal embeddings from {EMBEDS}")
    emb = np.load(EMBEDS)  # [10374 × 1024]
    log(f"Embeddings shape: {emb.shape}")

    log("Running UMAP once (deterministic seed=42)...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=SEED,
        low_memory=True,
    )
    reduced = reducer.fit_transform(emb)
    log(f"UMAP reduced shape: {reduced.shape}")

    results = []
    for (mcs, ms) in SWEEP:
        log(f"\n=== HDBSCAN(min_cluster_size={mcs}, min_samples={ms}) ===")
        t0 = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced)
        elapsed = time.time() - t0

        n_total = len(labels)
        n_outliers = int(np.sum(labels == -1))
        outlier_ratio = n_outliers / n_total
        topic_ids = sorted(set(labels) - {-1})
        n_topics = len(topic_ids)
        cluster_sizes = np.array([np.sum(labels == t) for t in topic_ids])
        if len(cluster_sizes) > 0:
            mean_size = float(cluster_sizes.mean())
            std_size = float(cluster_sizes.std())
            min_size = int(cluster_sizes.min())
            max_size = int(cluster_sizes.max())
        else:
            mean_size = std_size = min_size = max_size = 0

        # Silhouette score (only on non-outliers, sample to speed up)
        try:
            mask = labels != -1
            if mask.sum() >= 3 and n_topics >= 2:
                # Subsample for speed (silhouette is O(n²))
                if mask.sum() > 5000:
                    idx = np.random.RandomState(SEED).choice(np.flatnonzero(mask), 5000, replace=False)
                else:
                    idx = np.flatnonzero(mask)
                sil = float(silhouette_score(reduced[idx], labels[idx], metric="euclidean"))
            else:
                sil = float("nan")
        except Exception:
            sil = float("nan")

        diversity = topic_diversity(reduced, labels)

        result = {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "n_topics": n_topics,
            "n_outliers": n_outliers,
            "outlier_ratio": round(outlier_ratio, 4),
            "mean_cluster_size": round(mean_size, 1),
            "std_cluster_size": round(std_size, 1),
            "min_cluster_size_obs": min_size,
            "max_cluster_size_obs": max_size,
            "silhouette": round(sil, 4) if not np.isnan(sil) else None,
            "centroid_diversity": round(diversity, 4),
            "elapsed_s": round(elapsed, 1),
        }
        log(f"  topics={n_topics}, outliers={n_outliers} ({outlier_ratio:.1%}), "
            f"size mean={mean_size:.1f}±{std_size:.1f} [{min_size}–{max_size}], "
            f"silhouette={sil:.4f}, diversity={diversity:.4f}, elapsed {elapsed:.1f}s")
        results.append(result)

    out_csv = OUT_DIR / "hdbscan_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log(f"\nWrote {out_csv}")

    # Plot: outlier_ratio vs n_topics, annotated with (mcs, ms)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [r["outlier_ratio"] for r in results]
    ys = [r["n_topics"] for r in results]
    for r, x, y in zip(results, xs, ys):
        marker = "*" if (r["min_cluster_size"] == 20 and r["min_samples"] == 10) else "o"
        size = 200 if marker == "*" else 60
        color = "#c62828" if r["outlier_ratio"] > 0.22 else "#2e7d32"
        ax.scatter(x, y, color=color, marker=marker, s=size, alpha=0.85)
        ax.annotate(f"({r['min_cluster_size']},{r['min_samples']})", (x, y),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.axvline(0.22, color="black", linestyle="--", alpha=0.5,
               label="Pre-registered target (≤ 22% outliers)")
    ax.axvline(0.15, color="gray", linestyle=":", alpha=0.5,
               label="Pre-registered target lower (≥ 15%)")
    ax.set_xlabel("Outlier ratio")
    ax.set_ylabel("Number of topics")
    ax.set_title("HDBSCAN sweep on journal corpus (n=10,374) — ★ = current baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hdbscan_sweep.png", dpi=120)
    plt.close(fig)
    log(f"Wrote {OUT_DIR / 'hdbscan_sweep.png'}")

    # Recommendation: smallest outlier rate that still has >= 30 topics and silhouette comparable to baseline
    baseline = next((r for r in results if r["min_cluster_size"] == 20 and r["min_samples"] == 10), None)
    if baseline is None:
        return
    log(f"\n=== Recommendation ===")
    log(f"Baseline (20, 10): topics={baseline['n_topics']}, outliers={baseline['outlier_ratio']:.1%}, silhouette={baseline['silhouette']}")
    log(f"Candidates with outlier_ratio in [0.15, 0.22] and silhouette > 0 (sorted by silhouette desc):")
    candidates = [r for r in results
                  if 0.15 <= r["outlier_ratio"] <= 0.22
                  and r["silhouette"] is not None and r["silhouette"] > 0]
    candidates.sort(key=lambda r: -r["silhouette"])
    for c in candidates[:5]:
        log(f"  ({c['min_cluster_size']}, {c['min_samples']}): topics={c['n_topics']}, "
            f"outliers={c['outlier_ratio']:.1%}, silhouette={c['silhouette']}, diversity={c['centroid_diversity']}")
    if candidates:
        best = candidates[0]
        log(f"\nRecommended: min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']}")


if __name__ == "__main__":
    main()
