"""Phase 2 first-look: Method B headline figure.

Compare per-AMIA-domain top-1 share between the capstone corpus (n=288) and
the journal corpus (n=10,374) and write:

- capstone_vs_journal_top1.png   — grouped bar chart
- capstone_vs_journal_delta.png  — Δ_domain bar chart (capstone − journal)
- capstone_vs_journal_table.csv  — numeric table with both shares + delta
- per_year_alignment.png         — per-year top-1 share in journals overlaid
                                    with capstone overall, by domain.

Produces the headline comparison figure and the per-year lead/lag plot.
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
CAPSTONE_DIST = PHASE1 / "corpus_distribution.csv"
JOURNAL_DIST = PHASE1 / "journals/corpus_distribution.csv"
JOURNAL_PER_ART = PHASE1 / "journals/per_article_scores.csv"
OUT_DIR = PHASE1 / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT = {
    "D1": "Health Sci", "D2": "HIT", "D3": "Soc/Beh", "D4": "InfoSci/CS",
    "D5": "Data Analytics", "D6": "Leadership", "D7": "Trans Bioinf",
    "D8": "Clinical Inf", "D9": "Public Health", "D10": "Consumer Health",
}


def load_dist(path: Path, top1_field: str = "top1_share") -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["domain_id"]] = {
                "label": r["domain_label"],
                "mean": float(r.get("mean_sigmoid_score", r.get("mean_sigmoid", 0))),
                "top1": float(r[top1_field]),
            }
    return out


def main() -> None:
    cap = load_dist(CAPSTONE_DIST, "top1_share")
    jrn = load_dist(JOURNAL_DIST, "top1_share_all")

    ids = sorted(cap.keys(), key=lambda x: int(x[1:]))
    cap_top1 = np.array([cap[d]["top1"] for d in ids])
    jrn_top1 = np.array([jrn[d]["top1"] for d in ids])

    # --- Headline grouped bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ids))
    width = 0.38
    b1 = ax.bar(x - width/2, cap_top1, width, label=f"Capstones (n=288)", color="#3273dc")
    b2 = ax.bar(x + width/2, jrn_top1, width, label=f"Journals (n=10,374)", color="#f57c00")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{SHORT[i]}" for i in ids], fontsize=9)
    ax.set_ylabel("Top-1 share of documents")
    ax.set_title("AMIA foundational domain coverage — Capstones vs Journals (Method B / BART-MNLI)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h >= 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.0%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capstone_vs_journal_top1.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'capstone_vs_journal_top1.png'}")

    # --- Delta chart ---
    delta = cap_top1 - jrn_top1
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#2e7d32" if d > 0 else "#c62828" for d in delta]
    ax.bar(x, delta, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{SHORT[i]}" for i in ids], fontsize=9)
    ax.set_ylabel("Δ share (capstone − journal)")
    ax.set_title("Curriculum lead (green) / lag (red) per AMIA domain")
    for i, d in enumerate(delta):
        ax.text(i, d + (0.005 if d > 0 else -0.012), f"{d:+.1%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "capstone_vs_journal_delta.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'capstone_vs_journal_delta.png'}")

    # --- Numeric table ---
    out_csv = OUT_DIR / "capstone_vs_journal_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain_id", "domain_label", "capstone_top1", "journal_top1",
                    "delta_capstone_minus_journal"])
        for i, d in enumerate(ids):
            w.writerow([d, cap[d]["label"], round(cap_top1[i], 4),
                        round(jrn_top1[i], 4), round(delta[i], 4)])
    print(f"Wrote {out_csv}")

    # --- Per-year journal share, with capstone overall as a horizontal ref ---
    jrn_by_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(JOURNAL_PER_ART, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            jrn_by_year[r["year"]][r["argmax_id"]] += 1
    years = sorted(jrn_by_year.keys())
    years = [y for y in years if y.isdigit()]
    # Convert to top-1 share per year
    jrn_year_share = {}
    for y in years:
        total = sum(jrn_by_year[y].values())
        jrn_year_share[y] = {d_id: jrn_by_year[y].get(d_id, 0) / total for d_id in ids}

    # Pick the 5 most-active domains by capstone+journal aggregate
    activity = cap_top1 + jrn_top1
    top_idx = np.argsort(-activity)[:5]
    selected = [ids[i] for i in top_idx]

    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), sharey=True)
    for ax, d_id in zip(axes, selected):
        ax.plot(years, [jrn_year_share[y][d_id] for y in years],
                marker="o", color="#f57c00", label="Journals")
        ax.axhline(cap[d_id]["top1"], color="#3273dc", linestyle="--", label="Capstones (overall)")
        ax.set_title(f"{d_id} {SHORT[d_id]}")
        ax.set_xlabel("Journal pub year")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 share")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Per-year journal top-1 share vs capstone overall — top 5 domains by joint activity")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_year_alignment.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'per_year_alignment.png'}")


if __name__ == "__main__":
    main()
