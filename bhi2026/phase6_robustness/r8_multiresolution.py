"""Matched-granularity / multi-resolution analysis.

The capstone side resolves to 8 topics, the journal side to 106. This asks
whether that asymmetry makes capstone topics too broad to ever reach the
"same research question" rating. R6 tested the mechanism (capstone topics are
not broader in embedding terms); this runs the analysis actually requested.

Journal topics are merged bottom-up to k in {8, 16, 32, 64, 106} by average-
linkage agglomerative clustering on centroid cosine. A merged topic's centroid
is the mass-weighted mean of its members and its mass is their sum, so the
journal marginal is preserved exactly. OT is re-solved against the unchanged 8
capstone topics at every level, and the k=8 (matched-granularity) flows are
re-rated with the unchanged JUROR prompt.
"""
import sys, csv, json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

BH = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
LEVELS = [8, 16, 32, 64, 106]


def merge(C, mass, ids, k):
    """Average-linkage merge of journal topics to k clusters; mass preserved."""
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    D = np.clip(1.0 - Cn @ Cn.T, 0, None); np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), k, criterion="maxclust")
    newC, newm, members = [], [], {}
    for c in sorted(set(lab)):
        sel = np.where(lab == c)[0]
        w = mass[sel]
        newC.append((C[sel] * w[:, None]).sum(0) / w.sum())
        newm.append(w.sum())
        members[c] = [ids[i] for i in sel[np.argsort(-w)]]
    return np.stack(newC), np.array(newm), sorted(set(lab)), members


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    cap_ids, Ccap = common.centroids(ce, ct); a = common.masses(ct, cap_ids)
    jrn_ids, Cjrn = common.centroids(je, jt); b = common.masses(jt, jrn_ids)

    base = common.solve_ot(Ccap, a, Cjrn, b, cap_ids=cap_ids, jrn_ids=jrn_ids)
    base_flows = common.flow_set(base["flows"])
    print(f"{'k (journal topics)':>20}{'W1':>9}{'align':>8}{'flows':>7}"
          f"{'cap topics used':>17}{'mean |flow| cos':>17}")
    rows, allmembers = [], {}
    for k in LEVELS:
        if k == len(jrn_ids):
            Cj, bj, ids_j, mem = Cjrn, b, jrn_ids, {t: [t] for t in jrn_ids}
        else:
            Cj, bj, ids_j, mem = merge(Cjrn, b, jrn_ids, k)
        assert abs(bj.sum() - b.sum()) < 1e-9, "mass not preserved"
        r = common.solve_ot(Ccap, a, Cj, bj, cap_ids=cap_ids, jrn_ids=ids_j)
        cos = np.mean([f["cosine_sim"] for f in r["flows"]])
        ncap = len({f["capstone_topic"] for f in r["flows"]})
        print(f"{k:>20}{r['w1']:>9.4f}{r['align']:>8.4f}{len(r['flows']):>7}{ncap:>17}{cos:>17.4f}")
        rows.append(dict(k=k, w1=r["w1"], align=r["align"], n_flows=len(r["flows"]),
                         mean_flow_cosine=float(cos)))
        allmembers[k] = {str(c): v for c, v in mem.items()}
        if k == 8:
            k8 = (r, ids_j, mem)

    print("\n=== Matched granularity (k=8 journal topics vs 8 capstone topics) ===")
    r8, ids8, mem8 = k8
    print(f"  W1 {base['w1']:.4f} (106 topics) -> {r8['w1']:.4f} (8 topics);"
          f"  alignment {base['align']:.4f} -> {r8['align']:.4f}")
    print(f"  mean flow cosine {np.mean([f['cosine_sim'] for f in base['flows']]):.4f}"
          f" -> {np.mean([f['cosine_sim'] for f in r8['flows']]):.4f}"
          "   (higher = coarser journal topics are a closer match, as expected)")
    print("\n  merged journal topics at k=8 (member topics by mass):")
    for c in sorted(mem8):
        print(f"    J*{c}: {len(mem8[c]):>3} source topics, e.g. {mem8[c][:6]}")

    with open(OUT / "r8_flows_k8.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["capstone_topic", "rank", "journal_topic",
                                          "transported_mass", "cost", "cosine_sim", "members"])
        w.writeheader()
        for fl in r8["flows"]:
            w.writerow({**fl, "members": " ".join(map(str, mem8[fl["journal_topic"]]))})
    json.dump(dict(levels=rows, k8_members=allmembers[8],
                   base=dict(w1=base["w1"], align=base["align"])),
              open(OUT / "r8_multiresolution.json", "w"), indent=2, default=float)
    print(f"\nWrote {OUT/'r8_multiresolution.json'}")


if __name__ == "__main__":
    main()
