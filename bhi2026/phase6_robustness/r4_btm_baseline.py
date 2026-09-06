"""Head-to-head against Bidirectional Topic Matching (BTM).

BTM (Adam & Kogler, arXiv:2412.18376) implemented per Sec. 2.2-3.1 of that paper,
on the SAME two BERTopic fits and the SAME bge-large embeddings used by JUROR:

  1. Train a topic model per corpus (done: 8 capstone topics, 106 journal topics).
  2. Apply each model reciprocally: assign every document of the other corpus to
     the topic (of this model) whose embedding is most similar -- with the
     outlier topic -1 kept as a legitimate category, as BTM specifies.
  3. Pairing strength  S(t_i, t~_j) = n(D_ij) / n(D_i)             (BTM Eq. 1)
       topic closeness  = sum over j >= 0 of S(t_i, t~_j)
       topic uniqueness = S(t_i, t~_-1);  >= 0.5 => "unique topic"
  4. Corpus closeness  C   = (1/T) sum_i topic_closeness(i)         (BTM Eq. 2)
     Weighted          C_w = sum_i n(D_i) cl(i) / sum_i n(D_i)      (BTM Eq. 3)
     theta = C_w - C ; uniqueness U = 1 - C, U_w = 1 - C_w          (Eqs. 4-6)

Head-to-head: BTM pairing strength and JUROR's rubric rating are scored against
the SAME target -- Method C's mean pair rating within each flow, the judge that
was calibrated against the adjudicated expert gold standard.
"""
import sys, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)


def centroids_with_outlier(emb, topics):
    ids = sorted(set(int(t) for t in topics))          # includes -1
    C = np.stack([emb[topics == t].mean(0) for t in ids])
    return ids, C / np.linalg.norm(C, axis=1, keepdims=True)


def cross_assign(emb, ids, C):
    """BTM step 2: each doc -> argmax-cosine topic of the OTHER corpus's model."""
    E = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return np.array(ids)[(E @ C.T).argmax(1)]


def btm_side(native, cross, native_ids):
    """BTM Eqs. 1-6 for one direction. Returns per-pair S and corpus measures."""
    S, closeness, sizes = {}, {}, {}
    for t in native_ids:
        m = native == t
        n_i = int(m.sum()); sizes[t] = n_i
        if n_i == 0:
            continue
        for u, c in zip(*np.unique(cross[m], return_counts=True)):
            S[(t, int(u))] = c / n_i
        closeness[t] = float(((cross[m] != -1).sum()) / n_i)
    keep = [t for t in native_ids if t != -1]           # Eq. 2 sums i = 0..T
    C_ = float(np.mean([closeness[t] for t in keep]))
    tot = sum(sizes[t] for t in keep)
    C_w = float(sum(sizes[t] * closeness[t] for t in keep) / tot)
    return dict(S=S, closeness=closeness, sizes=sizes, C=C_, C_w=C_w,
                theta=C_w - C_, U=1 - C_, U_w=1 - C_w,
                unique_topics=[t for t in keep if (1 - closeness[t]) >= 0.5])


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    cap_ids_o, Ccap_o = centroids_with_outlier(ce, ct)
    jrn_ids_o, Cjrn_o = centroids_with_outlier(je, jt)

    T21 = cross_assign(ce, jrn_ids_o, Cjrn_o)   # capstone docs -> journal topics
    T12 = cross_assign(je, cap_ids_o, Ccap_o)   # journal  docs -> capstone topics

    cap_side = btm_side(ct, T21, cap_ids_o)     # native = capstone
    jrn_side = btm_side(jt, T12, jrn_ids_o)     # native = journal

    print("=== BTM corpus measures (Eqs. 2-6) ===")
    for nm, s in (("capstone -> journal", cap_side), ("journal -> capstone", jrn_side)):
        print(f"  {nm:22s} C={s['C']:.4f}  C_w={s['C_w']:.4f}  theta={s['theta']:+.4f}  "
              f"U={s['U']:.4f}  U_w={s['U_w']:.4f}  unique topics={len(s['unique_topics'])}")
    print(f"  capstone topics classified 'unique' (uniqueness>=0.5): {cap_side['unique_topics']}")

    # ---- head-to-head on the paper's 24 flows --------------------------------
    mc = {}
    for r in csv.DictReader(open(ROOT / "phase3_method_d/method_d_vs_method_c.csv")):
        mc[(int(r["capstone_topic"]), int(r["journal_topic"]))] = dict(
            method_c=float(r["method_c_mean"]), juror=float(r["method_d_rating"]),
            mass=float(r["mass"]))
    cos = {}
    for r in csv.DictReader(open(ROOT / "phase3_method_a/capstone_to_journal_flows.csv")):
        cos[(int(r["capstone_topic"]), int(r["journal_topic"]))] = float(r["cosine_sim"])

    rows = []
    for (i, j), v in mc.items():
        s_cap = cap_side["S"].get((i, j), 0.0)     # frac of capstone-topic-i docs -> journal topic j
        s_jrn = jrn_side["S"].get((j, i), 0.0)     # frac of journal-topic-j docs -> capstone topic i
        rows.append(dict(capstone_topic=i, journal_topic=j,
                         btm_S_cap_native=s_cap, btm_S_jrn_native=s_jrn,
                         btm_S_mean=(s_cap + s_jrn) / 2, btm_S_max=max(s_cap, s_jrn),
                         cosine=cos[(i, j)], juror=v["juror"], method_c=v["method_c"]))
    rows.sort(key=lambda r: (r["capstone_topic"], r["journal_topic"]))

    y = np.array([r["method_c"] for r in rows])
    print(f"\n=== Head-to-head vs Method C pair means (n={len(rows)} flows) ===")
    print(f"  {'predictor':<34}{'Spearman':>10}{'p':>10}{'Pearson':>10}{'LLM calls':>11}")
    comp = {}
    for label, key, calls in [("JUROR (rubric rating)", "juror", 24),
                              ("BTM pairing strength (cap-native)", "btm_S_cap_native", 0),
                              ("BTM pairing strength (jrn-native)", "btm_S_jrn_native", 0),
                              ("BTM pairing strength (mean of both)", "btm_S_mean", 0),
                              ("BTM pairing strength (max of both)", "btm_S_max", 0),
                              ("centroid cosine (Ablation C)", "cosine", 0)]:
        x = np.array([r[key] for r in rows])
        rho = stats.spearmanr(x, y); pr = stats.pearsonr(x, y)
        comp[label] = dict(spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
                           pearson=float(pr.statistic), pearson_p=float(pr.pvalue),
                           llm_calls=calls)
        print(f"  {label:<34}{rho.statistic:>10.3f}{rho.pvalue:>10.4f}{pr.statistic:>10.3f}{calls:>11}")

    # Does BTM route to the same topic pairs OT does?
    ot_top = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "phase3_method_a/capstone_to_journal_flows.csv")):
        ot_top[int(r["capstone_topic"])].append(int(r["journal_topic"]))
    btm_top = {}
    for i in [t for t in cap_ids_o if t != -1]:
        cands = sorted(((j, s) for (t, j), s in cap_side["S"].items() if t == i and j != -1),
                       key=lambda x: -x[1])[:3]
        btm_top[i] = [j for j, _ in cands]
    inter = sum(len(set(ot_top[i]) & set(btm_top[i])) for i in ot_top)
    print(f"\n=== Flow selection: BTM top-3 vs OT top-3 per capstone topic ===")
    for i in sorted(ot_top):
        print(f"  capstone {i}: OT {ot_top[i]}   BTM {btm_top[i]}   shared {sorted(set(ot_top[i])&set(btm_top[i]))}")
    print(f"  total shared pairs: {inter}/24 ({inter/24:.0%})")

    # What the rubric says about the pairs BTM ranks highest
    juror_all = {}
    for r in csv.DictReader(open(ROOT / "phase3_method_d/flow_ratings.csv")):
        juror_all[(int(r["capstone_topic"]), int(r["journal_topic"]))] = int(r["thematic_alignment"])
    rated = [(r["btm_S_cap_native"], juror_all[(r["capstone_topic"], r["journal_topic"])])
             for r in rows]
    rated.sort(key=lambda x: -x[0])
    print("\n=== Top-5 flows by BTM pairing strength, with JUROR's rubric rating ===")
    for s, g in rated[:5]:
        print(f"  BTM S = {s:.3f}   JUROR rating = {g}")
    print(f"  max JUROR rating among all 24 flows = {max(juror_all.values())} "
          f"(threshold for 'shared research question' = 3)")

    with open(OUT / "r4_btm_flows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(dict(
        corpus_measures=dict(
            capstone_native={k: v for k, v in cap_side.items() if k not in ("S", "closeness", "sizes")},
            journal_native={k: v for k, v in jrn_side.items() if k not in ("S", "closeness", "sizes")}),
        capstone_topic_closeness=cap_side["closeness"],
        head_to_head=comp,
        selection_overlap=dict(shared=inter, total=24,
                               ot_top3={str(k): v for k, v in ot_top.items()},
                               btm_top3={str(k): v for k, v in btm_top.items()}),
    ), open(OUT / "r4_btm.json", "w"), indent=2, default=float)
    print(f"\nWrote {OUT/'r4_btm.json'}")


if __name__ == "__main__":
    main()
