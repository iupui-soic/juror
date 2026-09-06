"""Tier-2 calibration: Method C (pairwise LLM-judge) vs the adjudicated human
gold standard. Pre-specified calibration targets:
    Pearson r >= 0.6,  Cohen kappa >= 0.5,  within-1 agreement >= 80%.

Method D is flow-level (no per-pair rating), so it is validated against Method
C separately (rho=0.71); the per-pair Tier-2 anchor is Method C.

Also runs a sensitivity sweep: re-derives the verdict under alternative
adjudications of the 8 hard pairs, to show the conclusion does not hinge on
how those 8 were resolved.
Run: python3 tier2_calibration.py
"""
import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr
import agreement_panel as ap

M = {"no": 0, "partial": 1, "yes": 2}
HARD8 = ["pair_015", "pair_062", "pair_065", "pair_069",
         "pair_071", "pair_074", "pair_078", "pair_094"]


def load():
    with open("gold_standard.csv") as f:
        return list(csv.DictReader(f))


def metrics(adj, mc):
    """adj, mc: equal-length int lists (0-4). Returns dict of Tier-2 metrics."""
    adj = np.array(adj, float)
    mc = np.array(mc, float)
    n = len(adj)
    cm = ap.confusion([int(x) for x in adj], [int(y) for y in mc], 5)
    r, _ = pearsonr(adj, mc)
    rho, _ = spearmanr(adj, mc)
    return {
        "n": n,
        "pearson": r,
        "spearman": rho,
        "kappa_unw": ap.cohen(cm, ap.agree_weights(5, "identity")),
        "kappa_quad": ap.cohen(cm, ap.agree_weights(5, "quadratic")),
        "gwet_ac2": ap.gwet(cm, ap.agree_weights(5, "linear")),
        "exact": np.mean(adj == mc),
        "within1": np.mean(np.abs(adj - mc) <= 1),
    }


def verdict(m):
    p = "PASS" if m["pearson"] >= 0.6 else "FAIL"
    k = "PASS" if m["kappa_unw"] >= 0.5 else "FAIL"
    w = "PASS" if m["within1"] >= 0.80 else "FAIL"
    return p, k, w


def main():
    rows = [r for r in load() if r["method_c_rating"] not in (None, "")]
    print(f"Tier-2: Method C vs adjudicated gold standard   (n={len(rows)}; "
          f"pair_007 excluded — no Method C rating)\n")

    adj = [int(r["adj_rating"]) for r in rows]
    mc = [int(r["method_c_rating"]) for r in rows]
    m = metrics(adj, mc)
    p, k, w = verdict(m)

    print("=== RATING (0-4) ===")
    print(f"  Pearson r        : {m['pearson']:+.3f}   target >=0.60   [{p}]")
    print(f"  Spearman rho     : {m['spearman']:+.3f}")
    print(f"  Cohen kappa (unw): {m['kappa_unw']:+.3f}   target >=0.50   [{k}]")
    print(f"  Cohen kappa (quad): {m['kappa_quad']:+.3f}")
    print(f"  Gwet AC2 (linear): {m['gwet_ac2']:+.3f}")
    print(f"  Exact agreement  : {m['exact']*100:.1f}%")
    print(f"  Within-1 agreement: {m['within1']*100:.1f}%   target >=80%   [{w}]")

    overall = "PASS" if (p == k == w == "PASS") else "PARTIAL/FAIL"
    print(f"\n  TIER-2 VERDICT: {overall}  (Pearson {p}, kappa {k}, within-1 {w})")

    # confusion
    cm = ap.confusion(adj, mc, 5)
    print("\n  Confusion (rows=gold adjudicated, cols=Method C):")
    print("          " + " ".join(f"{l:>4}" for l in range(5)))
    for li in range(5):
        print(f"    G={li}  " + " ".join(f"{int(cm[li][j]):>4}" for j in range(5)))

    # methodology
    am = [M[r["adj_method"]] for r in rows if r["method_c_method"] in M]
    cmth = [M[r["method_c_method"]] for r in rows if r["method_c_method"] in M]
    mcm = ap.confusion(am, cmth, 3)
    print(f"\n=== METHODOLOGY (no/partial/yes)  n={len(am)} ===")
    print(f"  Cohen kappa      : {ap.cohen(mcm, ap.agree_weights(3,'identity')):+.3f}")
    print(f"  Gwet AC1         : {ap.gwet(mcm, ap.agree_weights(3,'identity')):+.3f}")
    print(f"  Exact agreement  : {np.trace(mcm)/len(am)*100:.1f}%")

    # ---- sensitivity over the 8 hard pairs ----
    print("\n=== SENSITIVITY: re-adjudicate the 8 hard pairs ===")
    by_id = {r["pair_id"]: r for r in rows}
    scenarios = {
        "frozen rule": lambda r: int(r["adj_rating"]),
        "R1 wins all 8": lambda r: int(r["r1_rating"]) if r["pair_id"] in HARD8 and r["r1_rating"] not in (None,"") else int(r["adj_rating"]),
        "R2 wins all 8": lambda r: int(r["r2_rating"]) if r["pair_id"] in HARD8 and r["r2_rating"] not in (None,"") else int(r["adj_rating"]),
        "drop the 8": None,
    }
    print(f"  {'scenario':<16} {'n':>3} {'Pearson':>8} {'kappa':>7} {'within1':>8}   verdict")
    for name, fn in scenarios.items():
        if name == "drop the 8":
            sub = [r for r in rows if r["pair_id"] not in HARD8]
            a2 = [int(r["adj_rating"]) for r in sub]
            c2 = [int(r["method_c_rating"]) for r in sub]
        else:
            a2 = [fn(r) for r in rows]
            c2 = mc
        mm = metrics(a2, c2)
        pp, kk, ww = verdict(mm)
        ov = "PASS" if pp==kk==ww=="PASS" else "partial"
        print(f"  {name:<16} {mm['n']:>3} {mm['pearson']:>+8.3f} "
              f"{mm['kappa_unw']:>+7.3f} {mm['within1']*100:>7.1f}%   "
              f"{ov} (r:{pp[0]} k:{kk[0]} w:{ww[0]})")


if __name__ == "__main__":
    main()
