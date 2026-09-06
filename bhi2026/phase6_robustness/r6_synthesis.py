"""R6 — Synthesis: (a) is JUROR's edge over BTM larger than noise? (b) does the
capstone/journal granularity asymmetry floor the rubric? (c) what does each
robustness scenario do to the rating distribution?
"""
import sys, csv, json
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

BH = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
RNG = np.random.default_rng(42)


def boot_delta(x1, x2, y, n=10000):
    """Bootstrap CI on Spearman(x1,y) - Spearman(x2,y), paired over flows."""
    d = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if len(set(y[idx])) < 3:
            continue
        d.append(stats.spearmanr(x1[idx], y[idx]).statistic -
                 stats.spearmanr(x2[idx], y[idx]).statistic)
    d = np.array([v for v in d if np.isfinite(v)])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
           float((d <= 0).mean())


def main():
    res = {}
    rows = list(csv.DictReader(open(OUT / "r4_btm_flows.csv")))
    y = np.array([float(r["method_c"] ) for r in rows])
    juror = np.array([float(r["juror"]) for r in rows])
    cos = np.array([float(r["cosine"]) for r in rows])

    print("=== (a) Is JUROR's advantage over BTM bigger than sampling noise? ===")
    print(f"  {'comparison':<42}{'d rho':>8}{'95% CI':>20}{'P(d<=0)':>10}")
    res["boot_delta"] = {}
    for lab, key in [("JUROR - BTM (max of both directions)", "btm_S_max"),
                     ("JUROR - BTM (mean of both directions)", "btm_S_mean"),
                     ("JUROR - BTM (capstone-native)", "btm_S_cap_native"),
                     ("JUROR - centroid cosine", "cosine")]:
        x = np.array([float(r[key]) for r in rows])
        m, lo, hi, p = boot_delta(juror, x, y)
        res["boot_delta"][lab] = dict(mean=m, lo=lo, hi=hi, p_le_0=p)
        print(f"  {lab:<42}{m:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{p:>10.3f}")
    print("  (CI straddling 0 = accuracy tie; the separation to argue is interpretive, not numeric)")

    # ---- (b) does capstone-topic breadth floor the rubric? -------------------
    print("\n=== (b) Granularity: do broader capstone topics rate lower? ===")
    ce, je, cd, jd, ct, jt = common.load_all()
    cap_ids, Ccap = common.centroids(ce, ct)
    jrn_ids, Cjrn = common.centroids(je, jt)

    def dispersion(emb, topics, ids, C):
        """mean cosine of members to their own centroid: HIGH = tight, LOW = broad."""
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        En = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return {t: float((En[topics == t] @ Cn[k]).mean()) for k, t in enumerate(ids)}

    cap_tight = dispersion(ce, ct, cap_ids, Ccap)
    jrn_tight = dispersion(je, jt, jrn_ids, Cjrn)
    print(f"  capstone topic tightness: median {np.median(list(cap_tight.values())):.3f} "
          f"(n={len(cap_tight)} topics, {int((ct!=-1).sum())} docs)")
    print(f"  journal  topic tightness: median {np.median(list(jrn_tight.values())):.3f} "
          f"(n={len(jrn_tight)} topics, {int((jt!=-1).sum())} docs)")

    fr = list(csv.DictReader(open(BH / "phase3_method_d/flow_ratings.csv")))
    per_cap = {}
    for r in fr:
        per_cap.setdefault(int(r["capstone_topic"]), []).append(int(r["thematic_alignment"]))
    ids = sorted(per_cap)
    tight = np.array([cap_tight[i] for i in ids])
    size = np.array([int((ct == i).sum()) for i in ids])
    mx = np.array([max(per_cap[i]) for i in ids])
    mn = np.array([np.mean(per_cap[i]) for i in ids])
    print(f"  {'cap topic':>10}{'n docs':>8}{'tightness':>11}{'max rating':>12}{'mean rating':>13}")
    for k, i in enumerate(ids):
        print(f"  {i:>10}{size[k]:>8}{tight[k]:>11.3f}{mx[k]:>12}{mn[k]:>13.2f}")
    out_b = {}
    for lab, xv in [("tightness", tight), ("topic size", size)]:
        for lab2, yv in [("max rating", mx), ("mean rating", mn)]:
            rho = stats.spearmanr(xv, yv)
            out_b[f"{lab} vs {lab2}"] = dict(spearman=float(rho.statistic), p=float(rho.pvalue))
            print(f"  Spearman({lab}, {lab2}) = {rho.statistic:+.3f}  p = {rho.pvalue:.3f}")
    print("  If breadth floored the rubric, tightness would correlate POSITIVELY and")
    print("  significantly with rating. It does not.")
    res["granularity"] = dict(cap_tightness=cap_tight, jrn_tightness_median=float(np.median(list(jrn_tight.values()))),
                              tests=out_b)

    # ---- (c) rating distributions per scenario ------------------------------
    RATER = sys.argv[1] if len(sys.argv) > 1 else "claude"
    print(f"\n=== (c) Rating distribution under each robustness scenario ({RATER}) ===")
    new = {(int(r["capstone_topic"]), int(r["journal_topic"])): r
           for r in csv.DictReader(open(OUT / f"r5_new_flow_ratings_{RATER}.csv"))
           if r["thematic_alignment"]}
    col = "gpt5_4_mini_rating" if RATER == "openai" else "claude_sonnet_4_6_rating"
    gpt_paper = {(int(r["capstone_topic"]), int(r["journal_topic"])): int(r[col])
                 for r in csv.DictReader(open(BH / "phase3_method_d_openai/cross_llm_comparison.csv"))}

    def dist(pairs):
        v = [gpt_paper[p] if p in gpt_paper else int(new[p]["thematic_alignment"])
             for p in pairs if p in gpt_paper or p in new]
        c = {k: int((np.array(v) == k).sum()) for k in range(5)}
        return v, c

    r2 = json.load(open(OUT / "r2_year_matched.json"))
    r3 = json.load(open(OUT / "r3_outlier_handling.json"))
    btm = json.load(open(OUT / "r4_btm.json"))
    paper = sorted(gpt_paper)
    scen = {"paper (OT top-3, 2020-25, outliers dropped)": paper}
    base = set(paper)
    for name in ["S1_nearest", "S2_nearest_p10"]:
        s = (base - {tuple(x) for x in r3[name]["lost_flows"]}) | {tuple(x) for x in r3[name]["new_flows"]}
        scen[f"outlier handling: {name}"] = sorted(s)
    s = (base - {tuple(x) for x in r2["year_matched"]["lost_flows"]}) | \
        {tuple(x) for x in r2["year_matched"]["new_flows"]}
    scen["year-matched journals <=2024"] = sorted(s)
    scen["BTM-selected flows (top-3 by pairing strength)"] = sorted(
        (int(ci), int(j)) for ci, js in btm["selection_overlap"]["btm_top3"].items() for j in js)

    res["scenarios"] = {}
    print(f"  {'scenario':<48}{'n':>4}{'  0  1  2  3  4':>16}{'mean':>7}{'>=3':>5}")
    for name, pairs in scen.items():
        v, c = dist(pairs)
        res["scenarios"][name] = dict(n=len(v), dist=c, mean=float(np.mean(v)),
                                      n_ge3=int(sum(1 for x in v if x >= 3)))
        print(f"  {name:<48}{len(v):>4}  " + " ".join(f"{c[k]:2d}" for k in range(5)) +
              f"{np.mean(v):>7.2f}{sum(1 for x in v if x>=3):>5}")

    json.dump(res, open(OUT / f"r6_synthesis_{RATER}.json", "w"), indent=2, default=float)
    print(f"\nWrote {OUT}/r6_synthesis_{RATER}.json")


if __name__ == "__main__":
    main()
