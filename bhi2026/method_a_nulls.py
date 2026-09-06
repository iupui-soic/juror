"""Method A null baselines.

Three baselines to contextualize the observed capstone→journal W_1 = 0.171:

  1. Inter-journal pairwise W_1 — natural ceiling. For each pair of journals,
     fit BERTopic on the union of the two and compute W_1 between their topic
     mass vectors. We approximate this more cheaply by re-using the already-
     fitted journal topic model: per-journal topic distributions are slices
     of the journal-wide topic assignment matrix, so we can compute W_1
     directly between two journals' empirical topic distributions over the
     shared 106 journal topics.

  2. Random-split within journals — natural floor for two slices of the
     same distribution. Randomly halve the journal corpus, compute W_1
     between the halves over the journal topic vocabulary. 200 splits.

  3. Random-split within capstones — same, but for capstones.

  4. Shuffle baseline — permute topic assignments randomly and recompute.
     Should give the largest W_1 (chance floor of *disagreement*).

The 'observed' baseline (capstone vs journal W_1 = 0.171) is computed in
method_a_bertopic_ot.py and replicated here for context. We use journal-
topic centroids and journal-topic-mass as the common space for the within-
journal experiments, since journal topics are richer.

Output to bhi2026/phase3_method_a/null_baselines.csv + figure.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot

PHASE3 = Path("bhi2026/phase3_method_a")
OUT_DIR = PHASE3
EPS = 1e-12
N_SPLITS = 200
RNG_SEED = 42


def load_topic_centroids(path: Path) -> tuple[list[int], np.ndarray]:
    """Load topic centroids saved as .npy. Returns (topic_ids, matrix)."""
    arr = np.load(path)
    # By construction in method_a_bertopic_ot.py, ids are sorted ascending
    # excluding the -1 outlier topic.
    return list(range(arr.shape[0])), arr  # placeholder ids; real ids unused below


def load_docs(path: Path, key: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["topic"] = int(r["topic"])
            out.append(r)
    return out


def mass_vector(docs: list[dict], topic_ids: list[int]) -> np.ndarray:
    """Convert doc list to topic-mass vector over the supplied ordered topic ids.
    Outlier (-1) docs are dropped from topic-level analysis."""
    counts = Counter(d["topic"] for d in docs if d["topic"] != -1)
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(topic_ids))
    return np.array([counts.get(t, 0) / total for t in topic_ids], dtype=float)


def w1_distance(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> float:
    return float(ot.emd2(a, b, cost))


def main() -> None:
    # Load the journal topic centroids (the richer space) and journal docs.
    jrn_centroids_path = PHASE3 / "journal_topic_centroids.npy"
    cap_centroids_path = PHASE3 / "capstone_topic_centroids.npy"
    if not (jrn_centroids_path.exists() and cap_centroids_path.exists()):
        print("ERROR: Method A centroids missing; run method_a_bertopic_ot.py first.")
        sys.exit(1)

    C_jrn = np.load(jrn_centroids_path)   # [106 × 1024]
    C_cap = np.load(cap_centroids_path)   # [8 × 1024]
    jrn_topic_ids = list(range(C_jrn.shape[0]))  # 0..105 corresponds to topics 0..105 (after dropping -1)
    cap_topic_ids = list(range(C_cap.shape[0]))  # 0..7

    # Cosine cost matrices
    cost_jrn = 1 - (C_jrn @ C_jrn.T) / (np.linalg.norm(C_jrn, axis=1, keepdims=True) * np.linalg.norm(C_jrn, axis=1)[None, :] + EPS)
    cost_cap = 1 - (C_cap @ C_cap.T) / (np.linalg.norm(C_cap, axis=1, keepdims=True) * np.linalg.norm(C_cap, axis=1)[None, :] + EPS)

    # Per-journal docs (with topics)
    jrn_docs = load_docs(PHASE3 / "journal_docs.csv", "pmid")
    cap_docs = load_docs(PHASE3 / "capstone_docs.csv", "capstone_id")
    print(f"Loaded {len(cap_docs)} capstone docs, {len(jrn_docs)} journal docs")

    # Group journal docs by journal_key
    by_journal: dict[str, list[dict]] = {}
    for d in jrn_docs:
        by_journal.setdefault(d["journal_key"], []).append(d)
    journal_keys = sorted(by_journal)
    print(f"Journals: {journal_keys}")

    rng = np.random.default_rng(RNG_SEED)
    report = {}

    # --- Baseline 1: inter-journal pairs (28 pairs) ---
    print("\n[1/4] Inter-journal pairwise W_1 in journal-topic space...")
    inter = []
    for a, b in combinations(journal_keys, 2):
        mass_a = mass_vector(by_journal[a], jrn_topic_ids)
        mass_b = mass_vector(by_journal[b], jrn_topic_ids)
        d = w1_distance(mass_a, mass_b, cost_jrn)
        inter.append({"a": a, "b": b, "w1": d})
    inter_w1 = np.array([p["w1"] for p in inter])
    report["inter_journal"] = {
        "n_pairs": len(inter), "mean": float(inter_w1.mean()),
        "median": float(np.median(inter_w1)),
        "min": float(inter_w1.min()), "max": float(inter_w1.max()),
        "pairs": inter,
    }
    print(f"  n={len(inter)} pairs  mean W_1={inter_w1.mean():.4f}  range [{inter_w1.min():.4f}, {inter_w1.max():.4f}]")

    # --- Baseline 2: random-split within journals ---
    print(f"\n[2/4] Random-split within journals (n={N_SPLITS})...")
    n_jrn = len(jrn_docs)
    half = n_jrn // 2
    splits = np.empty(N_SPLITS)
    for i in range(N_SPLITS):
        perm = rng.permutation(n_jrn)
        a = [jrn_docs[k] for k in perm[:half]]
        b = [jrn_docs[k] for k in perm[half:]]
        ma = mass_vector(a, jrn_topic_ids); mb = mass_vector(b, jrn_topic_ids)
        splits[i] = w1_distance(ma, mb, cost_jrn)
    report["random_split_journals"] = {
        "n_splits": N_SPLITS, "mean": float(splits.mean()),
        "ci_lo": float(np.percentile(splits, 2.5)), "ci_hi": float(np.percentile(splits, 97.5)),
    }
    print(f"  mean W_1={splits.mean():.4f}  CI [{np.percentile(splits, 2.5):.4f}, {np.percentile(splits, 97.5):.4f}]")

    # --- Baseline 3: random-split within capstones ---
    print(f"\n[3/4] Random-split within capstones (n={N_SPLITS})...")
    n_cap = len(cap_docs)
    half = n_cap // 2
    cap_splits = np.empty(N_SPLITS)
    for i in range(N_SPLITS):
        perm = rng.permutation(n_cap)
        a = [cap_docs[k] for k in perm[:half]]
        b = [cap_docs[k] for k in perm[half:]]
        ma = mass_vector(a, cap_topic_ids); mb = mass_vector(b, cap_topic_ids)
        cap_splits[i] = w1_distance(ma, mb, cost_cap)
    report["random_split_capstones"] = {
        "n_splits": N_SPLITS, "mean": float(cap_splits.mean()),
        "ci_lo": float(np.percentile(cap_splits, 2.5)), "ci_hi": float(np.percentile(cap_splits, 97.5)),
    }
    print(f"  mean W_1={cap_splits.mean():.4f}  CI [{np.percentile(cap_splits, 2.5):.4f}, {np.percentile(cap_splits, 97.5):.4f}]")

    # --- Baseline 4: shuffle baseline (chance floor of disagreement) ---
    print(f"\n[4/4] Shuffle baseline (random topic assignment in journals)...")
    # Sample uniformly random topic IDs and check against observed mass
    obs_mass = mass_vector(jrn_docs, jrn_topic_ids)
    shuffle_w1 = np.empty(N_SPLITS)
    n_topics = len(jrn_topic_ids)
    for i in range(N_SPLITS):
        rand_assignments = rng.integers(0, n_topics, size=len(jrn_docs))
        counts = np.bincount(rand_assignments, minlength=n_topics).astype(float)
        rand_mass = counts / counts.sum()
        shuffle_w1[i] = w1_distance(obs_mass, rand_mass, cost_jrn)
    report["shuffle_baseline"] = {
        "n_iterations": N_SPLITS, "mean": float(shuffle_w1.mean()),
        "ci_lo": float(np.percentile(shuffle_w1, 2.5)), "ci_hi": float(np.percentile(shuffle_w1, 97.5)),
    }
    print(f"  mean W_1={shuffle_w1.mean():.4f}  CI [{np.percentile(shuffle_w1, 2.5):.4f}, {np.percentile(shuffle_w1, 97.5):.4f}]")

    # --- Observed cap→jrn (load from existing report) ---
    ot_report = json.loads((PHASE3 / "ot_alignment_report.json").read_text())
    observed_cap_jrn_w1 = float(ot_report["w1_distance"])
    report["observed_capstone_vs_journal"] = {"w1": observed_cap_jrn_w1}
    print(f"\nObserved capstone vs journal W_1 = {observed_cap_jrn_w1:.4f} (from method_a_bertopic_ot.py)")

    # --- Write report ---
    out_json = OUT_DIR / "null_baselines.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")

    # CSV summary
    out_csv = OUT_DIR / "null_baselines.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "W1_mean", "ci_lo", "ci_hi", "n"])
        w.writerow(["Observed capstone vs journal", round(observed_cap_jrn_w1, 4), "", "", "1 (point estimate)"])
        w.writerow(["Inter-journal pairs (28)", round(inter_w1.mean(), 4),
                    round(inter_w1.min(), 4), round(inter_w1.max(), 4), len(inter)])
        w.writerow(["Random-split within journals", round(splits.mean(), 4),
                    round(np.percentile(splits, 2.5), 4), round(np.percentile(splits, 97.5), 4), N_SPLITS])
        w.writerow(["Random-split within capstones", round(cap_splits.mean(), 4),
                    round(np.percentile(cap_splits, 2.5), 4), round(np.percentile(cap_splits, 97.5), 4), N_SPLITS])
        w.writerow(["Shuffle baseline (journals)", round(shuffle_w1.mean(), 4),
                    round(np.percentile(shuffle_w1, 2.5), 4), round(np.percentile(shuffle_w1, 97.5), 4), N_SPLITS])
    print(f"Wrote {out_csv}")

    # --- Figure: bar chart ---
    labels = [
        "Random-split\njournals",
        "Random-split\ncapstones",
        "Inter-journal\npairs",
        "Observed\ncap → journals",
        "Shuffle\nbaseline",
    ]
    means = [splits.mean(), cap_splits.mean(), inter_w1.mean(), observed_cap_jrn_w1, shuffle_w1.mean()]
    errs = [
        [splits.mean() - np.percentile(splits, 2.5)], [np.percentile(splits, 97.5) - splits.mean()],
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#74c476", "#74c476", "#fdae6b", "#c62828", "#9e9e9e"]
    bars = ax.bar(labels, means, color=colors)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Wasserstein-1 distance")
    ax.set_title("Method A — null and baseline W_1 distances (cosine ground cost)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "null_baselines.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'null_baselines.png'}")


if __name__ == "__main__":
    main()
