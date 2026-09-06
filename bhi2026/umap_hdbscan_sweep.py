"""Combined UMAP + HDBSCAN sweep — the HDBSCAN-only sweep showed every config
lands at 29-37% outliers, so the bottleneck may be UMAP geometry, not HDBSCAN.

Vary UMAP n_neighbors, n_components, min_dist, and pick the best HDBSCAN
config per UMAP. Output:
  bhi2026/phase3_method_a/umap_hdbscan_sweep.csv
  bhi2026/phase3_method_a/umap_hdbscan_sweep.png
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

OUT_DIR = Path("bhi2026/phase3_method_a")
EMBEDS = OUT_DIR / "journal_embeddings.npy"
SEED = 42

# UMAP grid
UMAP_GRID = [
    # (n_neighbors, n_components, min_dist) — baseline first
    (15, 5, 0.0),     # baseline
    (15, 10, 0.0),    # more components
    (15, 15, 0.0),
    (30, 5, 0.0),     # more neighbors
    (30, 10, 0.0),
    (50, 5, 0.0),
    (10, 5, 0.0),     # fewer neighbors
    (10, 10, 0.0),
    (15, 5, 0.05),    # tiny separation
    (15, 5, 0.1),
]

# Per-UMAP, scan a few HDBSCAN configs to find the best outlier ratio
HDBSCAN_GRID = [
    (10, 3),
    (15, 5),
    (20, 5),
    (20, 10),
    (30, 10),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_hdbscan(reduced: np.ndarray, mcs: int, ms: int) -> tuple[np.ndarray, dict]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    n_total = len(labels)
    n_outliers = int(np.sum(labels == -1))
    topic_ids = sorted(set(labels) - {-1})
    n_topics = len(topic_ids)
    cluster_sizes = np.array([np.sum(labels == t) for t in topic_ids])
    info = {
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "outlier_ratio": n_outliers / n_total,
        "mean_size": float(cluster_sizes.mean()) if len(cluster_sizes) else 0.0,
        "std_size": float(cluster_sizes.std()) if len(cluster_sizes) else 0.0,
    }
    return labels, info


def silhouette_of(reduced: np.ndarray, labels: np.ndarray) -> float:
    try:
        mask = labels != -1
        if mask.sum() >= 3 and len(set(labels[mask])) >= 2:
            if mask.sum() > 5000:
                idx = np.random.RandomState(SEED).choice(np.flatnonzero(mask), 5000, replace=False)
            else:
                idx = np.flatnonzero(mask)
            return float(silhouette_score(reduced[idx], labels[idx], metric="euclidean"))
    except Exception:
        pass
    return float("nan")


def main() -> None:
    log(f"Loading journal embeddings from {EMBEDS}")
    emb = np.load(EMBEDS)
    log(f"Embeddings shape: {emb.shape}")

    results = []
    for (n_nbrs, n_comp, min_d) in UMAP_GRID:
        log(f"\n=== UMAP(n_neighbors={n_nbrs}, n_components={n_comp}, min_dist={min_d}) ===")
        t0 = time.time()
        reducer = umap.UMAP(n_neighbors=n_nbrs, n_components=n_comp, min_dist=min_d,
                            metric="cosine", random_state=SEED, low_memory=True)
        reduced = reducer.fit_transform(emb)
        umap_t = time.time() - t0
        log(f"  UMAP fit in {umap_t:.1f}s, shape={reduced.shape}")

        for (mcs, ms) in HDBSCAN_GRID:
            t0 = time.time()
            labels, info = fit_hdbscan(reduced, mcs, ms)
            hdb_t = time.time() - t0
            sil = silhouette_of(reduced, labels)

            results.append({
                "n_neighbors": n_nbrs,
                "n_components": n_comp,
                "min_dist": min_d,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_topics": info["n_topics"],
                "n_outliers": info["n_outliers"],
                "outlier_ratio": round(info["outlier_ratio"], 4),
                "mean_cluster_size": round(info["mean_size"], 1),
                "std_cluster_size": round(info["std_size"], 1),
                "silhouette": round(sil, 4) if not np.isnan(sil) else None,
                "umap_s": round(umap_t, 1),
                "hdbscan_s": round(hdb_t, 1),
            })
            log(f"    HDBSCAN({mcs}, {ms}): topics={info['n_topics']}, "
                f"outliers={info['outlier_ratio']:.1%}, silhouette={sil:.3f}")

    out_csv = OUT_DIR / "umap_hdbscan_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log(f"\nWrote {out_csv}")

    # Plot: outlier_ratio vs n_topics, color by silhouette
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7))
    xs = [r["outlier_ratio"] for r in results]
    ys = [r["n_topics"] for r in results]
    sils = [r["silhouette"] if r["silhouette"] is not None else 0 for r in results]
    sc = ax.scatter(xs, ys, c=sils, cmap="viridis", s=80, alpha=0.85, edgecolor="black")
    # Highlight current baseline
    for i, r in enumerate(results):
        if r["n_neighbors"] == 15 and r["n_components"] == 5 and r["min_cluster_size"] == 20 and r["min_samples"] == 10:
            ax.scatter([r["outlier_ratio"]], [r["n_topics"]], marker="*", s=400,
                       facecolors="none", edgecolors="red", linewidths=2,
                       label="Current baseline")
    plt.colorbar(sc, ax=ax, label="Silhouette score")
    ax.axvline(0.22, color="black", linestyle="--", alpha=0.5, label="Pre-registered target ≤ 22%")
    ax.axvline(0.15, color="gray", linestyle=":", alpha=0.5, label="Pre-registered target ≥ 15%")
    ax.set_xlabel("Outlier ratio")
    ax.set_ylabel("Number of topics")
    ax.set_title(f"Joint UMAP × HDBSCAN sweep on journal corpus (n=10,374) — {len(results)} configs")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "umap_hdbscan_sweep.png", dpi=120)
    plt.close(fig)
    log(f"Wrote {OUT_DIR / 'umap_hdbscan_sweep.png'}")

    # Recommendation
    in_target = [r for r in results
                 if 0.15 <= r["outlier_ratio"] <= 0.22
                 and r["silhouette"] is not None]
    log(f"\n=== Configs in 15-22% outlier target ===")
    if in_target:
        in_target.sort(key=lambda r: -r["silhouette"])
        for r in in_target[:8]:
            log(f"  UMAP({r['n_neighbors']},{r['n_components']},{r['min_dist']}) "
                f"HDBSCAN({r['min_cluster_size']},{r['min_samples']}): "
                f"topics={r['n_topics']}, outliers={r['outlier_ratio']:.1%}, sil={r['silhouette']:.3f}")
    else:
        log("  NONE — no joint config reaches 22%. The corpus may not support it.")
        # Show best 5 anyway
        results_sorted = [r for r in results if r["silhouette"] is not None]
        results_sorted.sort(key=lambda r: r["outlier_ratio"])
        log("\n  Best 5 by lowest outlier ratio (silhouette > 0):")
        for r in [x for x in results_sorted if x["silhouette"] > 0][:5]:
            log(f"    UMAP({r['n_neighbors']},{r['n_components']},{r['min_dist']}) "
                f"HDBSCAN({r['min_cluster_size']},{r['min_samples']}): "
                f"topics={r['n_topics']}, outliers={r['outlier_ratio']:.1%}, sil={r['silhouette']:.3f}")


if __name__ == "__main__":
    main()
