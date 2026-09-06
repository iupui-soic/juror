"""Shared loaders + OT solve, replicating method_a_bertopic_ot.py exactly."""
from pathlib import Path
import csv, sys
import numpy as np
import ot as pot

csv.field_size_limit(sys.maxsize)
MA = Path(__file__).resolve().parent.parent / "phase3_method_a"


def load_docs(name, id_col):
    return list(csv.DictReader(open(MA / name, encoding="utf-8")))


def load_all():
    cap_emb = np.load(MA / "capstone_embeddings.npy")
    jrn_emb = np.load(MA / "journal_embeddings.npy")
    cap_docs = load_docs("capstone_docs.csv", "capstone_id")
    jrn_docs = load_docs("journal_docs.csv", "pmid")
    cap_t = np.array([int(r["topic"]) for r in cap_docs])
    jrn_t = np.array([int(r["topic"]) for r in jrn_docs])
    return cap_emb, jrn_emb, cap_docs, jrn_docs, cap_t, jrn_t


def centroids(emb, topics, include_outlier=False):
    """topic_id -> centroid, matching method_a's topic_centroid (excludes -1)."""
    ids = sorted(set(int(t) for t in topics))
    if not include_outlier:
        ids = [t for t in ids if t != -1]
    return ids, np.stack([emb[topics == t].mean(0) for t in ids])


def masses(topics, ids):
    """Empirical mass = members / total non-outlier docs (method_a's topic_size)."""
    n_in = int((topics != -1).sum())
    return np.array([(topics == t).sum() / n_in for t in ids], dtype=float)


def solve_ot(C_cap, a, C_jrn, b, topk=3, cap_ids=None, jrn_ids=None):
    """Returns dict with transport plan, W1, alignment score, and top-k flows."""
    cn = C_cap / np.linalg.norm(C_cap, axis=1, keepdims=True)
    jn = C_jrn / np.linalg.norm(C_jrn, axis=1, keepdims=True)
    sim = cn @ jn.T
    cost = 1.0 - sim
    T = pot.emd(a, b, cost)
    w1 = float((T * cost).sum())
    scale = float(cost.mean())
    align = float(np.exp(-w1 / scale)) if scale > 0 else 0.0
    cap_ids = cap_ids if cap_ids is not None else list(range(len(a)))
    jrn_ids = jrn_ids if jrn_ids is not None else list(range(len(b)))
    flows = []
    for i, ct in enumerate(cap_ids):
        row = T[i]
        if row.sum() == 0:
            continue
        for rank, j in enumerate(np.argsort(-row)[:topk], start=1):
            if row[j] <= 0:
                continue
            flows.append(dict(capstone_topic=ct, rank=rank, journal_topic=jrn_ids[j],
                              transported_mass=float(row[j]), cost=float(cost[i, j]),
                              cosine_sim=float(sim[i, j])))
    return dict(T=T, w1=w1, scale=scale, align=align, sim=sim, cost=cost, flows=flows)


def flow_set(flows):
    return {(f["capstone_topic"], f["journal_topic"]) for f in flows}
