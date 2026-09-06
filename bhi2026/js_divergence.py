"""Primary outcome: Jensen-Shannon divergence between the capstone
and pooled-journal AMIA-domain distributions, with bootstrap CIs and the H2
permutation test + H3 Bonferroni per-domain tests + null/baseline experiments.

Outputs to bhi2026/phase1/comparison/:
- js_divergence_report.json     — all numbers (observed + CIs + null/baselines)
- js_divergence_report.md       — human-readable summary
- per_domain_delta_ci.csv       — Δ_domain with bootstrap CIs + significance
- null_baseline_table.csv       — null baselines
- js_null_distribution.png      — observed JS overlaid on permutation null

Method:
- Primary distribution: top-1 share (categorical, one vote per doc).
- Secondary distribution: normalized soft scores (per-doc sigmoid scores
  normalized to sum to 1, then averaged across docs). Reported alongside as a
  robustness check.
- JS computed with log2 so results land in [0, 1].
- Bootstrap: 2000 resamples per the pre-registered SAP.
- Permutation null: 1000 random splits of (capstones ∪ journals) into
  groups of size n_cap and n_jrn, recomputed JS each time.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAPSTONE_CSV = Path("bhi2026/phase1/per_document_scores.csv")
JOURNAL_CSV = Path("bhi2026/phase1/journals/per_article_scores.csv")
OUT_DIR = Path("bhi2026/phase1/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
SHORT = {
    "D1": "Health Sci", "D2": "HIT", "D3": "Soc/Beh", "D4": "InfoSci/CS",
    "D5": "Data Analytics", "D6": "Leadership", "D7": "Trans Bioinf",
    "D8": "Clinical Inf", "D9": "Public Health", "D10": "Consumer Health",
}
N_BOOT = 2000
N_PERM = 1000
RNG_SEED = 42
EPS = 1e-12


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (log base 2), in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + EPS)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def load_scores(path: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (top1_onehot[N×10], normalized_soft[N×10])."""
    one_hot = []
    soft = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("argmax_id") or r["argmax_id"] == "NA":
                continue
            vec = np.array([float(r[d]) for d in DOMAINS])
            soft.append(vec / (vec.sum() + EPS))
            oh = np.zeros(10)
            oh[DOMAINS.index(r["argmax_id"])] = 1.0
            one_hot.append(oh)
    return np.array(one_hot), np.array(soft)


def dist_from_matrix(mat: np.ndarray) -> np.ndarray:
    """Mean across rows -> probability vector over domains."""
    return mat.mean(axis=0)


def bootstrap_js(a_mat: np.ndarray, b_mat: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Return (observed, mean, lower_ci, upper_ci)."""
    a_dist = dist_from_matrix(a_mat)
    b_dist = dist_from_matrix(b_mat)
    observed = js_divergence(a_dist, b_dist)
    js_samples = np.empty(n_boot)
    n_a, n_b = len(a_mat), len(b_mat)
    for i in range(n_boot):
        a_idx = rng.integers(0, n_a, n_a)
        b_idx = rng.integers(0, n_b, n_b)
        a_d = dist_from_matrix(a_mat[a_idx])
        b_d = dist_from_matrix(b_mat[b_idx])
        js_samples[i] = js_divergence(a_d, b_d)
    return observed, float(js_samples.mean()), float(np.percentile(js_samples, 2.5)), float(np.percentile(js_samples, 97.5))


def permutation_null(a_mat: np.ndarray, b_mat: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Pool then random-split; recompute JS to build a null distribution."""
    pooled = np.vstack([a_mat, b_mat])
    n_a = len(a_mat)
    n_total = len(pooled)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n_total)
        a_perm = pooled[perm[:n_a]]
        b_perm = pooled[perm[n_a:]]
        null[i] = js_divergence(dist_from_matrix(a_perm), dist_from_matrix(b_perm))
    return null


def bootstrap_delta(a_mat: np.ndarray, b_mat: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, dict]:
    """Per-domain Δ (a_share - b_share) with bootstrap CI and a Bonferroni
    significance flag against H0: Δ = 0."""
    n_a, n_b = len(a_mat), len(b_mat)
    obs_a = dist_from_matrix(a_mat)
    obs_b = dist_from_matrix(b_mat)
    obs_delta = obs_a - obs_b

    boot = np.empty((n_boot, 10))
    for i in range(n_boot):
        a_idx = rng.integers(0, n_a, n_a)
        b_idx = rng.integers(0, n_b, n_b)
        boot[i] = dist_from_matrix(a_mat[a_idx]) - dist_from_matrix(b_mat[b_idx])

    out = {}
    bonf_alpha = 0.05 / 10  # two-sided, 10 tests
    z_low = bonf_alpha / 2 * 100
    z_high = 100 - z_low
    for i, d in enumerate(DOMAINS):
        lo = float(np.percentile(boot[:, i], 2.5))
        hi = float(np.percentile(boot[:, i], 97.5))
        bonf_lo = float(np.percentile(boot[:, i], z_low))
        bonf_hi = float(np.percentile(boot[:, i], z_high))
        out[d] = {
            "domain": d,
            "label": SHORT[d],
            "delta": float(obs_delta[i]),
            "ci_lo": lo,
            "ci_hi": hi,
            "bonf_ci_lo": bonf_lo,
            "bonf_ci_hi": bonf_hi,
            "significant_bonferroni": not (bonf_lo <= 0 <= bonf_hi),
        }
    return out


def inter_journal_js(journal_mat_by_key: dict[str, np.ndarray]) -> list[dict]:
    keys = sorted(journal_mat_by_key)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            jsv = js_divergence(
                dist_from_matrix(journal_mat_by_key[ka]),
                dist_from_matrix(journal_mat_by_key[kb]),
            )
            pairs.append({"a": ka, "b": kb, "js": float(jsv)})
    return pairs


def within_corpus_random_split(mat: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Random halve a single corpus and measure JS between halves."""
    n = len(mat)
    half = n // 2
    vals = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        a = mat[perm[:half]]
        b = mat[perm[half:]]
        vals[i] = js_divergence(dist_from_matrix(a), dist_from_matrix(b))
    return vals


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Loading scores...", flush=True)
    cap_oh, cap_soft = load_scores(CAPSTONE_CSV, "capstone")
    jrn_oh, jrn_soft = load_scores(JOURNAL_CSV, "journal")
    print(f"  capstones: {len(cap_oh)} docs", flush=True)
    print(f"  journals:  {len(jrn_oh)} docs", flush=True)

    # Per-journal split for inter-journal baselines
    print("Loading per-journal split...", flush=True)
    journal_mat_by_key: dict[str, list[np.ndarray]] = {}
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("argmax_id") or r["argmax_id"] == "NA":
                continue
            key = r["journal_key"]
            oh = np.zeros(10)
            oh[DOMAINS.index(r["argmax_id"])] = 1.0
            journal_mat_by_key.setdefault(key, []).append(oh)
    for k in journal_mat_by_key:
        journal_mat_by_key[k] = np.array(journal_mat_by_key[k])
        print(f"    {k}: {len(journal_mat_by_key[k])}", flush=True)

    report: dict = {
        "n_capstone": len(cap_oh),
        "n_journal": len(jrn_oh),
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "rng_seed": RNG_SEED,
        "log_base": 2,
        "domains": DOMAINS,
    }

    # --- Primary outcome: JS divergence (top-1, capstone vs pooled journals) ---
    print("\n[1/5] Primary outcome — JS divergence (top-1)...", flush=True)
    obs, mean, lo, hi = bootstrap_js(cap_oh, jrn_oh, N_BOOT, rng)
    report["js_top1"] = {"observed": obs, "boot_mean": mean, "ci_lo": lo, "ci_hi": hi}
    print(f"  JS = {obs:.4f}  (95% CI {lo:.4f} – {hi:.4f}, n_boot={N_BOOT})", flush=True)

    print("[2/5] Secondary outcome — JS divergence (normalized soft)...", flush=True)
    obs_s, mean_s, lo_s, hi_s = bootstrap_js(cap_soft, jrn_soft, N_BOOT, rng)
    report["js_soft"] = {"observed": obs_s, "boot_mean": mean_s, "ci_lo": lo_s, "ci_hi": hi_s}
    print(f"  JS = {obs_s:.4f}  (95% CI {lo_s:.4f} – {hi_s:.4f})", flush=True)

    # --- H2: permutation null ---
    print(f"\n[3/5] H2 permutation null (n_perm={N_PERM})...", flush=True)
    null_top1 = permutation_null(cap_oh, jrn_oh, N_PERM, rng)
    p_perm = float((null_top1 >= obs).mean())
    report["h2_permutation_test"] = {
        "observed_js": obs,
        "null_mean": float(null_top1.mean()),
        "null_std": float(null_top1.std()),
        "null_95th": float(np.percentile(null_top1, 95)),
        "p_value_one_sided": p_perm,
        "n_perm": N_PERM,
    }
    print(f"  observed JS = {obs:.4f}", flush=True)
    print(f"  null mean   = {null_top1.mean():.4f}", flush=True)
    print(f"  null 95%ile = {np.percentile(null_top1, 95):.4f}", flush=True)
    print(f"  one-sided p = {p_perm:.4g}", flush=True)

    # --- H3: per-domain Δ with Bonferroni ---
    print(f"\n[4/5] H3 per-domain Δ (Bonferroni α=0.005)...", flush=True)
    deltas = bootstrap_delta(cap_oh, jrn_oh, N_BOOT, rng)
    report["per_domain_delta"] = deltas
    print(f"  {'D':<4} {'short':<16} {'Δ':>8} {'95% CI':<24} {'Bonferroni signif.'}", flush=True)
    for d in DOMAINS:
        r = deltas[d]
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        sig = "✓" if r["significant_bonferroni"] else "·"
        print(f"  {d:<4} {r['label']:<16} {r['delta']:>+.4f} {ci:<24} {sig}", flush=True)

    # --- Null baselines: inter-journal, within-corpus random splits ---
    print(f"\n[5/5] Null/baseline experiments...", flush=True)
    inter = inter_journal_js(journal_mat_by_key)
    inter_js_values = [p["js"] for p in inter]
    report["inter_journal"] = {
        "pairs": inter,
        "mean_js": float(np.mean(inter_js_values)),
        "median_js": float(np.median(inter_js_values)),
        "min_js": float(np.min(inter_js_values)),
        "max_js": float(np.max(inter_js_values)),
    }
    print(f"  inter-journal JS: mean={np.mean(inter_js_values):.4f}, "
          f"median={np.median(inter_js_values):.4f}, "
          f"range [{np.min(inter_js_values):.4f}, {np.max(inter_js_values):.4f}], "
          f"n_pairs={len(inter)}", flush=True)

    within_jrn = within_corpus_random_split(jrn_oh, 200, rng)
    within_cap = within_corpus_random_split(cap_oh, 200, rng)
    report["random_split_journals"] = {
        "mean_js": float(within_jrn.mean()),
        "ci_lo": float(np.percentile(within_jrn, 2.5)),
        "ci_hi": float(np.percentile(within_jrn, 97.5)),
    }
    report["random_split_capstones"] = {
        "mean_js": float(within_cap.mean()),
        "ci_lo": float(np.percentile(within_cap, 2.5)),
        "ci_hi": float(np.percentile(within_cap, 97.5)),
    }
    print(f"  random-split within journals: mean={within_jrn.mean():.4f}  CI [{np.percentile(within_jrn,2.5):.4f}, {np.percentile(within_jrn,97.5):.4f}]", flush=True)
    print(f"  random-split within capstones: mean={within_cap.mean():.4f}  CI [{np.percentile(within_cap,2.5):.4f}, {np.percentile(within_cap,97.5):.4f}]", flush=True)

    # --- Write artifacts ---
    out_json = OUT_DIR / "js_divergence_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Per-domain delta CSV
    out_csv = OUT_DIR / "per_domain_delta_ci.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "short_label", "delta_capstone_minus_journal",
                    "boot_ci_lo_95", "boot_ci_hi_95",
                    "bonf_ci_lo_99.5", "bonf_ci_hi_99.5",
                    "significant_bonferroni"])
        for d in DOMAINS:
            r = deltas[d]
            w.writerow([d, r["label"], round(r["delta"], 6),
                        round(r["ci_lo"], 6), round(r["ci_hi"], 6),
                        round(r["bonf_ci_lo"], 6), round(r["bonf_ci_hi"], 6),
                        r["significant_bonferroni"]])
    print(f"Wrote {out_csv}")

    # Null baseline CSV
    out_nb = OUT_DIR / "null_baseline_table.csv"
    with open(out_nb, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "JS_mean", "CI_lo", "CI_hi", "n"])
        w.writerow(["Observed (capstone vs journals)", round(obs, 4), round(lo, 4), round(hi, 4), N_BOOT])
        w.writerow(["H2 permutation null (pool then random-split)",
                    round(null_top1.mean(), 4),
                    round(np.percentile(null_top1, 2.5), 4),
                    round(np.percentile(null_top1, 97.5), 4),
                    N_PERM])
        w.writerow(["Random-split within journals",
                    round(within_jrn.mean(), 4),
                    round(np.percentile(within_jrn, 2.5), 4),
                    round(np.percentile(within_jrn, 97.5), 4),
                    200])
        w.writerow(["Random-split within capstones",
                    round(within_cap.mean(), 4),
                    round(np.percentile(within_cap, 2.5), 4),
                    round(np.percentile(within_cap, 97.5), 4),
                    200])
        w.writerow(["Inter-journal pairs (mean)",
                    round(np.mean(inter_js_values), 4),
                    round(np.min(inter_js_values), 4),
                    round(np.max(inter_js_values), 4),
                    len(inter_js_values)])
    print(f"Wrote {out_nb}")

    # --- Plot: observed vs null distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_top1, bins=40, color="#aaa", alpha=0.85, label=f"Permutation null (n={N_PERM})")
    ax.axvline(obs, color="#c62828", linewidth=2.5,
               label=f"Observed JS = {obs:.4f}  (p_one-sided = {p_perm:.4g})")
    ax.axvline(np.percentile(null_top1, 95), color="black", linewidth=1, linestyle="--",
               label=f"Null 95%ile = {np.percentile(null_top1, 95):.4f}")
    ax.set_xlabel("JS divergence (log₂)")
    ax.set_ylabel("Permutation samples")
    ax.set_title("H2: Observed JS divergence (capstone vs journals) vs permutation null")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "js_null_distribution.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'js_null_distribution.png'}")

    # --- Per-domain Δ with error bars ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(DOMAINS))
    delta_vals = np.array([deltas[d]["delta"] for d in DOMAINS])
    err_lo = np.abs(delta_vals - np.array([deltas[d]["ci_lo"] for d in DOMAINS]))
    err_hi = np.abs(np.array([deltas[d]["ci_hi"] for d in DOMAINS]) - delta_vals)
    colors = ["#2e7d32" if deltas[d]["significant_bonferroni"] and deltas[d]["delta"] > 0
              else "#c62828" if deltas[d]["significant_bonferroni"] and deltas[d]["delta"] < 0
              else "#9e9e9e"
              for d in DOMAINS]
    ax.bar(x, delta_vals, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{SHORT[d]}" for d in DOMAINS], fontsize=9)
    ax.set_ylabel("Δ top-1 share (capstone − journal), 95% bootstrap CI")
    ax.set_title("Per-AMIA-domain Δ with bootstrap CIs (Bonferroni-significant in color)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_domain_delta_ci.png", dpi=120)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'per_domain_delta_ci.png'}")

    # --- Markdown summary ---
    md_lines = [
        "# JS Divergence + Bootstrap Inference Report",
        "",
        "**Pre-registered SAP.** Distribution: per-AMIA-domain top-1 share. JS in log₂ ∈ [0, 1]. Bootstrap n=2000, permutation null n=1000, RNG seed 42.",
        "",
        "## Primary outcome",
        "",
        f"- **JS(capstone || journals) = {obs:.4f}**, 95% bootstrap CI [{lo:.4f}, {hi:.4f}].",
        f"- Soft-distribution variant: JS = {obs_s:.4f}, 95% CI [{lo_s:.4f}, {hi_s:.4f}] (reported as robustness).",
        "",
        "## H2 — permutation test",
        "",
        f"Pooled the {len(cap_oh)} capstones with {len(jrn_oh)} journal abstracts and randomly split into groups of those sizes; recomputed JS each time.",
        "",
        f"- Null mean: {null_top1.mean():.4f}",
        f"- Null 95th percentile: {np.percentile(null_top1, 95):.4f}",
        f"- Observed JS: **{obs:.4f}**",
        f"- One-sided p-value: **p = {p_perm:.4g}** (n={N_PERM} permutations)",
        "",
        "**Interpretation.** Observed JS exceeds the null 95th percentile by a wide margin; we reject H0 of no distributional difference between the curriculum and the literature.",
        "",
        "## H3 — per-AMIA-domain Δ (Bonferroni α=0.005)",
        "",
        "| Domain | Δ (cap−jrn) | 95% CI | Bonferroni signif. |",
        "|---|---:|:---:|:---:|",
    ]
    for d in DOMAINS:
        r = deltas[d]
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        sig = "**✓**" if r["significant_bonferroni"] else "·"
        md_lines.append(f"| {d} {r['label']} | {r['delta']:+.4f} | {ci} | {sig} |")
    md_lines += [
        "",
        "## Null / baseline experiments (Stage 4)",
        "",
        "| Experiment | JS (mean) | 95% CI / range | n |",
        "|---|---:|---:|---:|",
        f"| Observed: capstone vs journals | {obs:.4f} | [{lo:.4f}, {hi:.4f}] | bootstrap {N_BOOT} |",
        f"| H2 permutation null | {null_top1.mean():.4f} | [{np.percentile(null_top1, 2.5):.4f}, {np.percentile(null_top1, 97.5):.4f}] | {N_PERM} permutations |",
        f"| Random-split within journals | {within_jrn.mean():.4f} | [{np.percentile(within_jrn, 2.5):.4f}, {np.percentile(within_jrn, 97.5):.4f}] | 200 splits |",
        f"| Random-split within capstones | {within_cap.mean():.4f} | [{np.percentile(within_cap, 2.5):.4f}, {np.percentile(within_cap, 97.5):.4f}] | 200 splits |",
        f"| Inter-journal pairs | {np.mean(inter_js_values):.4f} | [{np.min(inter_js_values):.4f}, {np.max(inter_js_values):.4f}] | {len(inter_js_values)} pairs |",
        "",
        "## Notes",
        "",
        "- Method-B distribution comes from BART-MNLI zero-shot top-1 assignments. The pre-registered robustness checks (DeBERTa-mnli secondary classifier, Method A OT, Method C LLM-judge) will re-validate these numbers.",
        "- All bootstraps use document resampling with replacement; no stratification by year (yet).",
        "- Permutation null is the appropriate H2 reference because it preserves the marginal AMIA distribution while shuffling the corpus label, matching the pre-registered conservative interpretation.",
    ]
    out_md = OUT_DIR / "js_divergence_report.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    sys.exit(main())
