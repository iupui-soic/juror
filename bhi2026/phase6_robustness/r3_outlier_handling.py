"""Recompute results under alternative outlier-handling strategies.

Baseline (paper): HDBSCAN noise (-1) dropped before OT; topic mass renormalised
over retained documents only.

Alternatives compared here, all on the SAME BERTopic fit (so the only thing that
changes is what happens to the noise points):
  S0 drop            — the paper's choice.
  S1 nearest         — every outlier assigned to the retained topic with the
                       highest centroid cosine (= BERTopic reduce_outliers
                       strategy="embeddings", threshold 0).
  S2 nearest_thresh  — same, but only if cosine >= the 10th percentile of
                       in-topic member cosines; the rest stay dropped.
  S3 outlier_as_topic— noise kept as a legitimate topic (-1) on both sides, so
                       OT may route mass into it (this is what BTM does).

For each: recompute centroids + masses, re-solve OT, report W1 / alignment
score, and compare the resulting top-3 flow set against the paper's 24 flows.
"""
import sys, json, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def assign_nearest(emb, topics, ids, C, thresh_pct=None):
    """Return a copy of `topics` with -1 points moved to their nearest centroid."""
    t = topics.copy()
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    En = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = En @ Cn.T                      # [n_docs x n_topics]
    floor = -np.inf
    if thresh_pct is not None:
        member = [sims[i, ids.index(int(topics[i]))] for i in range(len(topics)) if topics[i] != -1]
        floor = float(np.percentile(member, thresh_pct))
    out_idx = np.where(topics == -1)[0]
    best = sims[out_idx].argmax(1)
    bestsim = sims[out_idx].max(1)
    keep = bestsim >= floor
    t[out_idx[keep]] = np.array(ids)[best[keep]]
    return t, int(keep.sum()), len(out_idx), floor


def run(cap_t, jrn_t, cap_emb, jrn_emb, include_outlier=False):
    cap_ids, Ccap = common.centroids(cap_emb, cap_t, include_outlier)
    jrn_ids, Cjrn = common.centroids(jrn_emb, jrn_t, include_outlier)
    if include_outlier:                    # mass over ALL docs, -1 included
        a = np.array([(cap_t == t).sum() for t in cap_ids], float); a /= a.sum()
        b = np.array([(jrn_t == t).sum() for t in jrn_ids], float); b /= b.sum()
    else:
        a = common.masses(cap_t, cap_ids); b = common.masses(jrn_t, jrn_ids)
    return common.solve_ot(Ccap, a, Cjrn, b, cap_ids=cap_ids, jrn_ids=jrn_ids)


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    base_ids_cap, Ccap0 = common.centroids(ce, ct)
    base_ids_jrn, Cjrn0 = common.centroids(je, jt)

    results, rows = {}, []
    base = run(ct, jt, ce, je)
    base_flows = common.flow_set(base["flows"])

    scenarios = {"S0_drop": (ct, jt, False, {})}

    c1, nc1, tc1, _ = assign_nearest(ce, ct, base_ids_cap, Ccap0)
    j1, nj1, tj1, _ = assign_nearest(je, jt, base_ids_jrn, Cjrn0)
    scenarios["S1_nearest"] = (c1, j1, False,
                               {"cap_reassigned": f"{nc1}/{tc1}", "jrn_reassigned": f"{nj1}/{tj1}"})

    c2, nc2, tc2, fc2 = assign_nearest(ce, ct, base_ids_cap, Ccap0, thresh_pct=10)
    j2, nj2, tj2, fj2 = assign_nearest(je, jt, base_ids_jrn, Cjrn0, thresh_pct=10)
    scenarios["S2_nearest_p10"] = (c2, j2, False,
                                   {"cap_reassigned": f"{nc2}/{tc2}", "jrn_reassigned": f"{nj2}/{tj2}",
                                    "cap_floor": round(fc2, 4), "jrn_floor": round(fj2, 4)})

    scenarios["S3_outlier_as_topic"] = (ct, jt, True, {})

    for name, (c, j, inc, extra) in scenarios.items():
        r = run(c, j, ce, je, include_outlier=inc)
        fs = common.flow_set(r["flows"])
        shared = fs & base_flows
        jac = len(shared) / len(fs | base_flows)
        n_out_c = int((c == -1).sum()); n_out_j = int((j == -1).sum())
        results[name] = dict(
            w1=r["w1"], alignment_score=r["align"], n_flows=len(fs),
            n_cap_outliers=n_out_c, n_jrn_outliers=n_out_j,
            jrn_outlier_rate=n_out_j / len(j),
            flows_shared_with_paper=len(shared), jaccard_vs_paper=jac,
            new_flows=sorted(fs - base_flows), lost_flows=sorted(base_flows - fs),
            **extra)
        rows.append(dict(scenario=name, w1=round(r["w1"], 4),
                         alignment=round(r["align"], 4),
                         jrn_outlier_pct=round(100 * n_out_j / len(j), 1),
                         n_flows=len(fs), shared_with_paper=len(shared),
                         jaccard=round(jac, 3)))
        print(f"{name:22s} W1={r['w1']:.4f} align={r['align']:.4f} "
              f"jrn_outliers={100*n_out_j/len(j):5.1f}%  flows={len(fs):2d} "
              f"shared={len(shared):2d}/24 jaccard={jac:.3f}")
        if fs - base_flows:
            print(f"   new flows: {sorted(fs - base_flows)}")
        if base_flows - fs:
            print(f"   lost flows: {sorted(base_flows - fs)}")

    (OUT / "r3_outlier_handling.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "r3_outlier_handling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT/'r3_outlier_handling.json'}")


if __name__ == "__main__":
    main()
