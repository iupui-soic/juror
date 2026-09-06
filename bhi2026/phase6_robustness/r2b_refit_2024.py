"""Definitive year-matched analysis: re-fit BERTopic from
scratch on the <=2024 journal subset (capstone window) and re-solve OT.

Same hyperparameters and seed as method_a_bertopic_ot.py; the cached
journal_embeddings.npy is reused so nothing depends on re-embedding.
"""
import os, sys, csv, json, time
os.environ.setdefault("TRANSFORMERS_NO_TF", "1"); os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan, umap
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
SEED = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    yrs = np.array([int(r["year"]) for r in jd])
    keep = yrs <= 2024
    log(f"journals <=2024: {int(keep.sum())} / {len(keep)}")

    texts_by_pmid = {}
    with open(ROOT / "bhi2026/journals/journal_corpus.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            texts_by_pmid[r["pmid"]] = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
    texts = [texts_by_pmid.get(jd[i]["pmid"], "") or "[empty]" for i in np.where(keep)[0]]
    emb = je[keep]
    log(f"texts={len(texts)} emb={emb.shape}")

    um = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                   random_state=SEED, low_memory=True)
    hd = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10, metric="euclidean",
                         cluster_selection_method="eom", prediction_data=True)
    vec = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, stop_words="english")
    tm = BERTopic(embedding_model=None, umap_model=um, hdbscan_model=hd,
                  vectorizer_model=vec, calculate_probabilities=False, verbose=True)
    log("fitting BERTopic on <=2024 journals ...")
    topics, _ = tm.fit_transform(texts, emb)
    topics = np.array(topics)
    info = tm.get_topic_info()
    info.to_csv(OUT / "r2b_journal_topics_2024.csv", index=False)
    # Representative_Docs holds PubMed abstract text, which is not
    # redistributed; the *_public variant is the releasable artifact.
    info.drop(columns=["Representative_Docs"], errors="ignore").to_csv(
        OUT / "r2b_journal_topics_2024_public.csv", index=False)
    n_topics = int((info["Topic"] != -1).sum())
    n_out = int((topics == -1).sum())
    log(f"topics={n_topics}  outliers={n_out} ({n_out/len(topics):.1%})")

    ids_j = [t for t in sorted(set(int(x) for x in topics)) if t != -1]
    Cj = np.stack([emb[topics == t].mean(0) for t in ids_j])
    b = common.masses(topics, ids_j)
    ids_c, Cc = common.centroids(ce, ct)
    a = common.masses(ct, ids_c)
    r = common.solve_ot(Cc, a, Cj, b, cap_ids=ids_c, jrn_ids=ids_j)

    ref = json.load(open(ROOT / "bhi2026/phase3_method_a/ot_alignment_report.json"))
    log(f"W1    {ref['w1_distance']:.4f} (full 2020-25) -> {r['w1']:.4f} (refit <=2024)")
    log(f"align {ref['alignment_score_exp']:.4f}          -> {r['align']:.4f}")

    # topic-level correspondence: for each NEW journal topic, its nearest OLD topic
    ids_j0, Cj0 = common.centroids(je, jt)
    A = Cj / np.linalg.norm(Cj, axis=1, keepdims=True)
    B = Cj0 / np.linalg.norm(Cj0, axis=1, keepdims=True)
    S = A @ B.T
    nearest = S.argmax(1); nearsim = S.max(1)
    log(f"new->old topic nearest-centroid cosine: median {np.median(nearsim):.3f} "
        f"min {nearsim.min():.3f}  (>=0.90: {(nearsim>=0.90).mean():.0%})")

    # Do the refit flows land on journal topics that correspond to the paper's?
    with open(OUT / "r2b_flows_2024.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["capstone_topic", "rank", "journal_topic",
                                          "transported_mass", "cost", "cosine_sim",
                                          "nearest_old_topic", "nearest_old_cos"])
        w.writeheader()
        for fl in r["flows"]:
            j = ids_j.index(fl["journal_topic"])
            w.writerow({**fl, "nearest_old_topic": ids_j0[nearest[j]],
                        "nearest_old_cos": round(float(nearsim[j]), 4)})

    paper_flows = common.flow_set(common.solve_ot(
        Cc, a, Cj0, common.masses(jt, ids_j0), cap_ids=ids_c, jrn_ids=ids_j0)["flows"])
    mapped = {(fl["capstone_topic"], ids_j0[nearest[ids_j.index(fl["journal_topic"])]])
              for fl in r["flows"]}
    shared = mapped & paper_flows
    log(f"refit flows mapped onto paper topic ids: {len(shared)}/24 match the paper's flows")

    json.dump(dict(n_kept=int(keep.sum()), n_topics_2024=n_topics, n_topics_full=106,
                   outlier_rate_2024=n_out / len(topics), outlier_rate_full=0.3141,
                   w1_full=ref["w1_distance"], w1_2024=r["w1"],
                   align_full=ref["alignment_score_exp"], align_2024=r["align"],
                   new_to_old_cos_median=float(np.median(nearsim)),
                   new_to_old_cos_min=float(nearsim.min()),
                   flows_matching_paper=len(shared),
                   mapped_flows=sorted(mapped), paper_flows=sorted(paper_flows)),
              open(OUT / "r2b_refit_2024.json", "w"), indent=2, default=float)
    log("done")


if __name__ == "__main__":
    main()
