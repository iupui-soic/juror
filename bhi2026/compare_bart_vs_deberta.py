"""Compare BART-MNLI vs DeBERTa-v3-large-mnli AMIA classifications.

Generates the Method-B robustness section for the BHI paper. The headline
question: does the 65% D5 journal share and the −33 pt D5 lag survive
swapping the NLI backbone?

Comparisons reported:
1. Per-corpus per-domain top-1 share: BART vs DeBERTa, with Δ_model.
2. Spearman rank correlation across the 10 domains, per corpus.
3. Per-document argmax agreement rate (exact match) + Cohen's κ.
4. JS divergence (capstone vs journal) under each model — does the cross-corpus
   effect survive?
5. Per-domain Δ (capstone − journal) under each model side by side.

Outputs to bhi2026/phase1_deberta/comparison/.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PHASE1 = Path("bhi2026/phase1")
DEBERTA = Path("bhi2026/phase1_deberta")
OUT = DEBERTA / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
SHORT = {
    "D1": "Health Sci", "D2": "HIT", "D3": "Soc/Beh", "D4": "InfoSci/CS",
    "D5": "Data Analytics", "D6": "Leadership", "D7": "Trans Bioinf",
    "D8": "Clinical Inf", "D9": "Public Health", "D10": "Consumer Health",
}
EPS = 1e-12


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + EPS)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def load_argmax(path: Path, id_field: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            arg = r.get("argmax_id") or r.get("argmax_label")
            if arg and arg != "NA":
                out[r[id_field]] = arg
    return out


def cohen_kappa(labels_a: list[str], labels_b: list[str], all_labels: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    # Confusion matrix
    idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)
    cm = np.zeros((k, k))
    for a, b in zip(labels_a, labels_b):
        cm[idx[a], idx[b]] += 1
    po = np.trace(cm) / n
    pa = cm.sum(axis=1) / n
    pb = cm.sum(axis=0) / n
    pe = float(np.sum(pa * pb))
    if abs(1 - pe) < EPS:
        return 1.0
    return (po - pe) / (1 - pe)


def top1_share(labels: list[str]) -> dict[str, float]:
    c = Counter(labels)
    n = len(labels)
    return {d: c.get(d, 0) / n for d in DOMAINS}


def main() -> None:
    # --- Load argmax-per-document under both models, for both corpora ---
    bart_cap = load_argmax(PHASE1 / "per_document_scores.csv", "capstone_id")
    bart_jrn = load_argmax(PHASE1 / "journals/per_article_scores.csv", "pmid")
    deberta_cap = load_argmax(DEBERTA / "capstones/per_document_scores.csv", "id")
    deberta_jrn = load_argmax(DEBERTA / "journals/per_article_scores.csv", "id")

    print(f"BART capstones: {len(bart_cap)}, DeBERTa capstones: {len(deberta_cap)}")
    print(f"BART journals: {len(bart_jrn)}, DeBERTa journals: {len(deberta_jrn)}")

    # --- Per-corpus per-domain top-1 share ---
    common_cap = sorted(set(bart_cap) & set(deberta_cap))
    common_jrn = sorted(set(bart_jrn) & set(deberta_jrn))
    bart_cap_lab = [bart_cap[i] for i in common_cap]
    deb_cap_lab = [deberta_cap[i] for i in common_cap]
    bart_jrn_lab = [bart_jrn[i] for i in common_jrn]
    deb_jrn_lab = [deberta_jrn[i] for i in common_jrn]

    bart_cap_share = top1_share(bart_cap_lab)
    deb_cap_share = top1_share(deb_cap_lab)
    bart_jrn_share = top1_share(bart_jrn_lab)
    deb_jrn_share = top1_share(deb_jrn_lab)

    rows_per_corpus = []
    for corpus_name, bart_share, deb_share in [
        ("capstones", bart_cap_share, deb_cap_share),
        ("journals", bart_jrn_share, deb_jrn_share),
    ]:
        for d in DOMAINS:
            rows_per_corpus.append({
                "corpus": corpus_name,
                "domain": d,
                "short": SHORT[d],
                "bart_top1": round(bart_share[d], 4),
                "deberta_top1": round(deb_share[d], 4),
                "delta_model": round(deb_share[d] - bart_share[d], 4),
            })
    out_csv = OUT / "per_corpus_per_domain.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_per_corpus[0].keys()))
        w.writeheader()
        w.writerows(rows_per_corpus)
    print(f"Wrote {out_csv}")

    # --- Spearman + Cohen's κ ---
    spearman_cap = spearmanr([bart_cap_share[d] for d in DOMAINS],
                             [deb_cap_share[d] for d in DOMAINS])
    spearman_jrn = spearmanr([bart_jrn_share[d] for d in DOMAINS],
                             [deb_jrn_share[d] for d in DOMAINS])

    agree_cap = np.mean([a == b for a, b in zip(bart_cap_lab, deb_cap_lab)])
    agree_jrn = np.mean([a == b for a, b in zip(bart_jrn_lab, deb_jrn_lab)])
    kappa_cap = cohen_kappa(bart_cap_lab, deb_cap_lab, DOMAINS)
    kappa_jrn = cohen_kappa(bart_jrn_lab, deb_jrn_lab, DOMAINS)

    # --- JS divergences under each model ---
    def dist(share: dict[str, float]) -> np.ndarray:
        return np.array([share[d] for d in DOMAINS])

    js_bart = js_div(dist(bart_cap_share), dist(bart_jrn_share))
    js_deberta = js_div(dist(deb_cap_share), dist(deb_jrn_share))

    # --- Per-domain Δ side by side ---
    delta_rows = []
    for d in DOMAINS:
        delta_rows.append({
            "domain": d,
            "short": SHORT[d],
            "delta_bart": round(bart_cap_share[d] - bart_jrn_share[d], 4),
            "delta_deberta": round(deb_cap_share[d] - deb_jrn_share[d], 4),
        })

    # --- Reports ---
    report = {
        "n_capstones": len(common_cap),
        "n_journals": len(common_jrn),
        "spearman_capstones": {"rho": float(spearman_cap.statistic), "p": float(spearman_cap.pvalue)},
        "spearman_journals": {"rho": float(spearman_jrn.statistic), "p": float(spearman_jrn.pvalue)},
        "agreement_rate_capstones": float(agree_cap),
        "agreement_rate_journals": float(agree_jrn),
        "cohen_kappa_capstones": float(kappa_cap),
        "cohen_kappa_journals": float(kappa_jrn),
        "js_bart_capstone_vs_journal": float(js_bart),
        "js_deberta_capstone_vs_journal": float(js_deberta),
        "per_corpus_per_domain": rows_per_corpus,
        "per_domain_delta": delta_rows,
    }
    (OUT / "robustness_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT / 'robustness_report.json'}")

    print()
    print("=== Headline robustness numbers ===")
    print(f"Spearman ρ (per-domain share rank) — capstones: {spearman_cap.statistic:.3f}  journals: {spearman_jrn.statistic:.3f}")
    print(f"Per-doc argmax agreement — capstones: {agree_cap:.1%}  journals: {agree_jrn:.1%}")
    print(f"Cohen's κ — capstones: {kappa_cap:.3f}  journals: {kappa_jrn:.3f}")
    print(f"JS(capstone || journals) — BART: {js_bart:.4f}  DeBERTa: {js_deberta:.4f}")
    print()
    print("Per-domain top-1 share (BART → DeBERTa, capstones):")
    for d in DOMAINS:
        print(f"  {d} {SHORT[d]:<16}  BART {bart_cap_share[d]:>6.2%}  →  DeBERTa {deb_cap_share[d]:>6.2%}  (Δ {deb_cap_share[d]-bart_cap_share[d]:+.2%})")
    print()
    print("Per-domain top-1 share (BART → DeBERTa, journals):")
    for d in DOMAINS:
        print(f"  {d} {SHORT[d]:<16}  BART {bart_jrn_share[d]:>6.2%}  →  DeBERTa {deb_jrn_share[d]:>6.2%}  (Δ {deb_jrn_share[d]-bart_jrn_share[d]:+.2%})")

    # --- Figure: grouped per-domain bar for journals (where the load-bearing claim lives) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (corpus_name, bart_share, deb_share) in zip(
        axes,
        [("Capstones", bart_cap_share, deb_cap_share),
         ("Journals", bart_jrn_share, deb_jrn_share)],
    ):
        x = np.arange(len(DOMAINS))
        width = 0.38
        ax.bar(x - width/2, [bart_share[d] for d in DOMAINS], width, label="BART-MNLI", color="#3273dc")
        ax.bar(x + width/2, [deb_share[d] for d in DOMAINS], width, label="DeBERTa-v3-large", color="#7a3aff")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n{SHORT[d]}" for d in DOMAINS], fontsize=8)
        ax.set_title(corpus_name)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Top-1 share")
    fig.suptitle("AMIA top-1 share — BART vs DeBERTa robustness check")
    fig.tight_layout()
    fig.savefig(OUT / "bart_vs_deberta_per_domain.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT / 'bart_vs_deberta_per_domain.png'}")


if __name__ == "__main__":
    main()
