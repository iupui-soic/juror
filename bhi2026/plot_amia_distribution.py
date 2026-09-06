"""Generate the AMIA-domain distribution figures from phase1 results.

Two figures:
- amia_domain_distribution.png  — bar chart of mean sigmoid score and top1 share
- amia_by_year.png               — heatmap of top1 share per AMIA domain per year

Also writes a per-year breakdown CSV.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE1 = Path("bhi2026/phase1")
META_CSV = Path("bhi2026/metadata/corpus_metadata.csv")


def main() -> None:
    # Load corpus distribution
    with open(PHASE1 / "corpus_distribution.csv", encoding="utf-8") as f:
        dist = list(csv.DictReader(f))
    # Load per-document scores
    with open(PHASE1 / "per_document_scores.csv", encoding="utf-8") as f:
        per_doc = list(csv.DictReader(f))
    # Load metadata for year
    with open(META_CSV, encoding="utf-8") as f:
        meta = {r["capstone_id"]: r for r in csv.DictReader(f)}

    ids = [d["domain_id"] for d in dist]
    labels = [d["domain_label"] for d in dist]
    short_labels = [
        "Health Sci", "HIT", "Soc/Beh", "InfoSci/CS", "Data Analytics",
        "Leadership", "Trans Bioinf", "Clinical Inf", "Public Health",
        "Consumer Health",
    ]

    mean_scores = np.array([float(d["mean_sigmoid_score"]) for d in dist])
    top1_share = np.array([float(d["top1_share"]) for d in dist])

    # --- Figure 1: bar chart ---
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ids))
    width = 0.38
    ax.bar(x - width/2, mean_scores, width, label="Mean sigmoid score", color="#3273dc")
    ax.bar(x + width/2, top1_share, width, label="Top-1 share", color="#f57c00")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{s}" for i, s in zip(ids, short_labels)], fontsize=9)
    ax.set_ylabel("Score / Share")
    ax.set_title(f"AMIA foundational domain coverage — {len(per_doc)} capstones (Phase 1, BART-MNLI)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out1 = PHASE1 / "amia_domain_distribution.png"
    fig.savefig(out1, dpi=120)
    print(f"Wrote {out1}")
    plt.close(fig)

    # --- Per-year breakdown ---
    year_dom_count: dict[tuple[int, str], int] = defaultdict(int)
    year_total: dict[int, int] = defaultdict(int)
    for r in per_doc:
        if r["argmax_id"] == "NA":
            continue
        m = meta.get(r["capstone_id"])
        if not m:
            continue
        try:
            year = int(m["year"])
        except ValueError:
            continue
        year_dom_count[(year, r["argmax_id"])] += 1
        year_total[year] += 1

    years = sorted(year_total)
    matrix = np.zeros((len(ids), len(years)))
    for j, y in enumerate(years):
        for i, d_id in enumerate(ids):
            if year_total[y] > 0:
                matrix[i, j] = year_dom_count[(y, d_id)] / year_total[y]

    # Write csv
    out_csv = PHASE1 / "amia_by_year.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain_id", "domain_label"] + [f"{y} (n={year_total[y]})" for y in years])
        for i, d_id in enumerate(ids):
            w.writerow([d_id, labels[i]] + [round(matrix[i, j], 3) for j in range(len(years))])
    print(f"Wrote {out_csv}")

    # --- Figure 2: heatmap ---
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"{y}\n(n={year_total[y]})" for y in years])
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{ids[i]} — {short_labels[i]}" for i in range(len(ids))])
    ax.set_title("Top-1 AMIA domain share by year")
    for i in range(len(ids)):
        for j in range(len(years)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v < 0.4 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Top-1 share within year")
    fig.tight_layout()
    out2 = PHASE1 / "amia_by_year.png"
    fig.savefig(out2, dpi=120)
    print(f"Wrote {out2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
