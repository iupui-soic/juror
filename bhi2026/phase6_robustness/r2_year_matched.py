"""Is the capstone/journal year mismatch driving the result?

Capstones span 2020-2024; the journal corpus spans 2020-2025. Two questions:
  (1) Are the 3,259 discarded (HDBSCAN-noise) abstracts disproportionately 2025?
  (2) Does restricting the journal corpus to the capstone window (<=2024)
      change the OT alignment or the flow set?

Part 2 here holds the BERTopic fit fixed and re-derives centroids + masses from
the <=2024 members only, so the only thing that changes is the year window.
(r2b_refit.py re-fits BERTopic from scratch on the <=2024 subset.)
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    yrs = np.array([int(r["year"]) for r in jd])

    # ---- (1) outlier rate by year -------------------------------------------
    print("=== Journal-corpus outlier (HDBSCAN noise) rate by year ===")
    print(f"{'year':>6} {'n':>6} {'outliers':>9} {'rate':>7} {'%of corpus':>11} {'%of outliers':>13}")
    tbl, n_out_tot, n_tot = [], int((jt == -1).sum()), len(jt)
    for y in sorted(set(yrs)):
        m = yrs == y
        n, o = int(m.sum()), int(((jt == -1) & m).sum())
        tbl.append(dict(year=int(y), n=n, outliers=o, outlier_rate=o / n,
                        pct_of_corpus=n / n_tot, pct_of_outliers=o / n_out_tot))
        print(f"{y:>6} {n:>6} {o:>9} {o/n:>6.1%} {n/n_tot:>10.1%} {o/n_out_tot:>12.1%}")
    print(f"{'ALL':>6} {n_tot:>6} {n_out_tot:>9} {n_out_tot/n_tot:>6.1%}")

    # Is 2025 over-represented among outliers?  2x2 chi-square, and the
    # two-proportion test of outlier-rate(2025) vs outlier-rate(<=2024).
    m25 = yrs == 2025
    a, b = int(((jt == -1) & m25).sum()), int(((jt != -1) & m25).sum())
    c, d = int(((jt == -1) & ~m25).sum()), int(((jt != -1) & ~m25).sum())
    chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, d]])
    r25, rpre = a / (a + b), c / (c + d)
    se = np.sqrt(r25 * (1 - r25) / (a + b) + rpre * (1 - rpre) / (c + d))
    print(f"\n2025 outlier rate = {r25:.3%}  vs  <=2024 = {rpre:.3%}   "
          f"diff = {r25-rpre:+.3%}  95% CI [{r25-rpre-1.96*se:+.3%}, {r25-rpre+1.96*se:+.3%}]")
    print(f"chi2 = {chi2:.3f}, p = {p:.3f}  -> "
          f"{'2025 IS over-represented' if p < .05 and r25 > rpre else 'no excess of 2025 among the discards'}")
    print(f"2025 is {m25.sum()/n_tot:.1%} of the corpus and "
          f"{a/n_out_tot:.1%} of the outliers.")

    # ---- (2) year-matched OT (fit held fixed) --------------------------------
    print("\n=== Year-matched OT: journals restricted to <=2024 (fit held fixed) ===")
    base_ids_c, Ccap = common.centroids(ce, ct)
    a_mass = common.masses(ct, base_ids_c)
    base_ids_j, Cjrn = common.centroids(je, jt)
    base = common.solve_ot(Ccap, a_mass, Cjrn, common.masses(jt, base_ids_j),
                           cap_ids=base_ids_c, jrn_ids=base_ids_j)
    base_flows = common.flow_set(base["flows"])

    keep = yrs <= 2024
    jt_y, je_y = jt[keep], je[keep]
    # topics that survive with >=1 member in the window
    ids_y = [t for t in sorted(set(int(x) for x in jt_y)) if t != -1]
    Cj_y = np.stack([je_y[jt_y == t].mean(0) for t in ids_y])
    b_y = common.masses(jt_y, ids_y)
    r = common.solve_ot(Ccap, a_mass, Cj_y, b_y, cap_ids=base_ids_c, jrn_ids=ids_y)
    fs = common.flow_set(r["flows"])
    shared = fs & base_flows
    print(f"  journals kept: {int(keep.sum())}/{len(keep)} (dropped {int((~keep).sum())} 2025 abstracts)")
    print(f"  journal topics surviving: {len(ids_y)}/{len(base_ids_j)}")
    print(f"  W1  {base['w1']:.4f} (full)  ->  {r['w1']:.4f} (<=2024)   delta {r['w1']-base['w1']:+.4f}")
    print(f"  align {base['align']:.4f} (full) -> {r['align']:.4f} (<=2024)  delta {r['align']-base['align']:+.4f}")
    print(f"  flows shared with paper: {len(shared)}/24  jaccard {len(shared)/len(fs|base_flows):.3f}")
    if fs - base_flows:  print(f"  new flows:  {sorted(fs-base_flows)}")
    if base_flows - fs:  print(f"  lost flows: {sorted(base_flows-fs)}")

    res = dict(outlier_by_year=tbl,
               excess_2025=dict(rate_2025=r25, rate_pre2025=rpre, diff=r25 - rpre,
                                ci95=[r25 - rpre - 1.96 * se, r25 - rpre + 1.96 * se],
                                chi2=chi2, p=p,
                                pct_of_corpus_2025=float(m25.sum() / n_tot),
                                pct_of_outliers_2025=a / n_out_tot),
               year_matched=dict(n_kept=int(keep.sum()), n_dropped=int((~keep).sum()),
                                 n_topics=len(ids_y), w1_full=base["w1"], w1_matched=r["w1"],
                                 align_full=base["align"], align_matched=r["align"],
                                 flows_shared=len(shared),
                                 jaccard=len(shared) / len(fs | base_flows),
                                 new_flows=sorted(fs - base_flows),
                                 lost_flows=sorted(base_flows - fs)))
    (OUT / "r2_year_matched.json").write_text(json.dumps(res, indent=2, default=float))
    with open(OUT / "r2_outlier_by_year.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tbl[0].keys())); w.writeheader(); w.writerows(tbl)
    print(f"\nWrote {OUT/'r2_year_matched.json'}")


if __name__ == "__main__":
    main()
