"""R10 — what the n=24 head-to-head against BTM does not show.

r4 compares JUROR and BTM on the paper's 24 flows and finds an accuracy tie.
Those 24 are OT's own top-3 edges, i.e. exactly the region where BTM has
document mass, so the comparison is favourable to BTM. This measures BTM's
coverage over the deeper flow set produced by r9 (ranks 1-10, 71 edges).

BTM pairing strength is S(t_i,t~_j) = n(D_ij)/n(D_i), a document co-occurrence
count, so a native topic of n documents can express at most n distinct links.
On a 288-document curriculum corpus that is a hard ceiling; the transport plan
has no such bound because it allocates mass over topics, not documents.

Requires: r4_btm.json and r9_topk_ratings.csv.
"""
import sys, csv, json, importlib.util
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
csv.field_size_limit(sys.maxsize)
OUT = HERE / "out"
spec = importlib.util.spec_from_file_location("r4", HERE / "r4_btm_baseline.py")
r4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r4)


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    cap_ids_o, Ccap_o = r4.centroids_with_outlier(ce, ct)
    jrn_ids_o, Cjrn_o = r4.centroids_with_outlier(je, jt)
    T21 = r4.cross_assign(ce, jrn_ids_o, Cjrn_o)
    cap = r4.btm_side(ct, T21, cap_ids_o)
    S = cap["S"]

    rows = [r for r in csv.DictReader(open(OUT / "r9_topk_ratings.csv")) if r["thematic_alignment"]]
    key = lambda r: (int(r["capstone_topic"]), int(r["journal_topic"]))
    btm = np.array([S.get(key(r), 0.0) for r in rows])
    rat = np.array([int(r["thematic_alignment"]) for r in rows])
    cos = np.array([float(r["cosine"]) for r in rows])
    n_zero = int((btm == 0).sum())

    print(f"=== BTM coverage over the {len(rows)} rated transport edges (ranks 1-10) ===")
    print(f"  pairing strength exactly 0 : {n_zero}/{len(rows)} ({n_zero/len(rows):.0%})")
    print(f"  median strength            : {np.median(btm):.4f}   max {btm.max():.4f}")

    print("\n=== expressible links per capstone topic ===")
    print(f"  {'topic':>6}{'docs':>6}{'BTM non-zero dests':>20}{'OT non-zero dests':>19}")
    ot_nz = {}
    cap_ids, Ccap = common.centroids(ce, ct); jrn_ids, Cjrn = common.centroids(je, jt)
    T = common.solve_ot(Ccap, common.masses(ct, cap_ids), Cjrn, common.masses(jt, jrn_ids),
                        topk=999, cap_ids=cap_ids, jrn_ids=jrn_ids)["T"]
    per = []
    for i, c in enumerate(cap_ids):
        docs = int((ct == c).sum())
        nb = sum(1 for (t, j), v in S.items() if t == c and j != -1 and v > 0)
        no = int((T[i] > 0).sum())
        ot_nz[c] = no
        per.append(dict(topic=c, docs=docs, btm_nonzero=nb, ot_nonzero=no))
        print(f"  {c:>6}{docs:>6}{nb:>20}{no:>19}")

    print("\n=== the one rating-3 flow, C5 -> J22 ===")
    cands = sorted(((j, s) for (t, j), s in S.items() if t == 5 and j != -1), key=lambda x: -x[1])
    print(f"  BTM destinations for capstone topic 5: {[(j, round(s,4)) for j, s in cands]}")
    print(f"  BTM strength of C5-J22 = {S.get((5,22), 0.0):.4f}"
          f"   ({'present' if S.get((5,22),0) else 'ABSENT — invisible to BTM'})")
    ot_rank = {key(r): int(r["rank"]) for r in rows}
    print(f"  OT rank of C5-J22      = {ot_rank.get((5,22))}")

    sp_b = stats.spearmanr(btm, rat); sp_c = stats.spearmanr(cos, rat)
    print(f"\n=== tracking the rubric across the {len(rows)} edges ===")
    print(f"  Spearman(BTM strength, JUROR rating)  = {sp_b.statistic:+.3f}  p={sp_b.pvalue:.4f}")
    print(f"  Spearman(centroid cosine, JUROR rating) = {sp_c.statistic:+.3f}  p={sp_c.pvalue:.4f}")
    print("  NOTE: the target here is JUROR's own rating, so this is indicative, not an")
    print("  independent validation. The BTM-vs-cosine contrast is unaffected by that,")
    print("  since neither predictor is JUROR.")

    json.dump(dict(n_edges=len(rows), n_btm_zero=n_zero, frac_btm_zero=n_zero / len(rows),
                   btm_median=float(np.median(btm)), btm_max=float(btm.max()),
                   per_capstone_topic=per,
                   c5_j22=dict(btm_strength=float(S.get((5, 22), 0.0)),
                               btm_destinations=[[int(j), float(s)] for j, s in cands],
                               ot_rank=ot_rank.get((5, 22))),
                   spearman_btm_vs_rating=float(sp_b.statistic),
                   spearman_cosine_vs_rating=float(sp_c.statistic),
                   caveat="target is JUROR's own rating; BTM-vs-cosine contrast unaffected"),
              open(OUT / "r10_btm_coverage.json", "w"), indent=2)
    print(f"\nWrote {OUT/'r10_btm_coverage.json'}")


if __name__ == "__main__":
    main()
