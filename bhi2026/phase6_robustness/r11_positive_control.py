"""R11 — Positive control for the flow-level judge.

Split the journal corpus into two random halves, fit BERTopic separately on
each (same hyperparameters and seed as method_a_bertopic_ot.py; cached
embeddings reused), solve OT between the halves, and rate the top-3 flows with
the unchanged JUROR prompt and rater. Two halves of one corpus share research
questions by construction, so the flows should reach rubric levels 3-4. If
they do not, the flow-level prompt is floored and the case-study zero cannot
be interpreted.

Variant "neutral" repeats the rank-1 flows with only the corpus labels in the
prompt changed ("Capstone topic" -> "Corpus A topic", "Journal topic" ->
"Corpus B topic"). Everything else is identical. This separates a genre-label
priming effect from a rubric floor.

Outputs (out/): r11_positive_control.json, r11_positive_control_ratings.csv,
r11_flows.csv, r11_half{A,B}_topics_public.csv, r11_half{A,B}_docs.csv
"""
import os, sys, csv, json, time
os.environ.setdefault("TRANSFORMERS_NO_TF", "1"); os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
from collections import defaultdict, Counter
import importlib.util
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan, umap
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from redact_surfaces import scrub_csv

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent.parent
BH = ROOT / "bhi2026"
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
SEED = 42
spec = importlib.util.spec_from_file_location("mdb", BH / "method_d_ot_llm_bridge.py")
mdb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdb)
N_REP_DOCS, DOC_CHARS = mdb.N_REP_DOCS, mdb.DOC_EXCERPT_CHARS


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_half(texts, emb, tag):
    um = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                   random_state=SEED, low_memory=True)
    hd = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10, metric="euclidean",
                         cluster_selection_method="eom", prediction_data=True)
    vec = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, stop_words="english")
    tm = BERTopic(embedding_model=None, umap_model=um, hdbscan_model=hd,
                  vectorizer_model=vec, calculate_probabilities=False, verbose=False)
    topics, _ = tm.fit_transform(texts, emb)
    topics = np.array(topics)
    info = tm.get_topic_info()
    info.drop(columns=["Representative_Docs"], errors="ignore").to_csv(
        OUT / f"r11_{tag}_topics_public.csv", index=False)
    kw = {int(r["Topic"]): mdb.parse_keywords(str(r["Name"])) for _, r in info.iterrows()}
    return topics, kw


def build_prompt(ci, ji, kwA, kwB, docsA, docsB, variant):
    tpl = mdb.PROMPT_TEMPLATE
    if variant == "neutral":
        tpl = tpl.replace("=== Topic A — Capstone topic {cap_topic_id} ===",
                          "=== Topic A — Corpus A topic {cap_topic_id} ===")
        tpl = tpl.replace("=== Topic B — Journal topic {jrn_topic_id} ===",
                          "=== Topic B — Corpus B topic {jrn_topic_id} ===")
        assert "Corpus A topic" in tpl and "Corpus B topic" in tpl
    fmt = lambda docs: "\n".join(f"  ({k}) {t[:DOC_CHARS].strip()}" for k, t in enumerate(docs, 1))
    return tpl.format(cap_topic_id=ci, jrn_topic_id=ji, cap_keywords=kwA, jrn_keywords=kwB,
                      cap_docs=fmt(docsA), jrn_docs=fmt(docsB))


def main():
    ce, je, cd, jd, ct, jt = common.load_all()
    texts_by_pmid = {}
    with open(BH / "journals/journal_corpus.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            texts_by_pmid[r["pmid"]] = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
    n = len(jd)
    perm = np.random.RandomState(SEED).permutation(n)
    idx = {"halfA": np.sort(perm[: n // 2]), "halfB": np.sort(perm[n // 2:])}
    halves = {}
    for tag, ix in idx.items():
        texts = [texts_by_pmid.get(jd[i]["pmid"], "") or "[empty]" for i in ix]
        emb = je[ix]
        log(f"fitting BERTopic on {tag}: {len(ix)} abstracts ...")
        topics, kw = fit_half(texts, emb, tag)
        ids = [t for t in sorted(set(int(x) for x in topics)) if t != -1]
        C = np.stack([emb[topics == t].mean(0) for t in ids])
        m = common.masses(topics, ids)
        docs_by = defaultdict(list)          # file order within the half, as method_a does
        with open(OUT / f"r11_{tag}_docs.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pmid", "journal_key", "year", "topic"])
            for i, t in zip(ix, topics):
                docs_by[int(t)].append(jd[i]["pmid"])
                w.writerow([jd[i]["pmid"], jd[i]["journal_key"], jd[i]["year"], int(t)])
        n_out = int((topics == -1).sum())
        log(f"  {tag}: topics={len(ids)} outliers={n_out} ({n_out/len(ix):.1%})")
        halves[tag] = dict(ids=ids, C=C, m=m, kw=kw, docs_by=docs_by, n=len(ix),
                           n_topics=len(ids), n_out=n_out, out_rate=n_out / len(ix))
    A, B = halves["halfA"], halves["halfB"]
    r = common.solve_ot(A["C"], A["m"], B["C"], B["m"], topk=3, cap_ids=A["ids"], jrn_ids=B["ids"])
    ref = json.load(open(BH / "phase3_method_a/ot_alignment_report.json"))
    log(f"W1 {r['w1']:.4f} (halves)  vs {ref['w1_distance']:.4f} (capstone vs journal); "
        f"align {r['align']:.4f} vs {ref['alignment_score_exp']:.4f}")
    # is the rank-1 OT destination also the nearest centroid?
    sim = r["sim"]; rank1_is_nearest = 0
    for fl in r["flows"]:
        if fl["rank"] == 1:
            i = A["ids"].index(fl["capstone_topic"]); j = B["ids"].index(fl["journal_topic"])
            rank1_is_nearest += int(sim[i].argmax() == j)
    with open(OUT / "r11_flows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r["flows"][0].keys())); w.writeheader(); w.writerows(r["flows"])

    targets = [("verbatim", fl) for fl in r["flows"]] + \
              [("neutral", fl) for fl in r["flows"] if fl["rank"] == 1]
    out_csv = OUT / "r11_positive_control_ratings.csv"
    done = {}
    if out_csv.exists():
        for row in csv.DictReader(open(out_csv)):
            if row.get("thematic_alignment") not in ("", None) and not row.get("error"):
                done[(row["variant"], int(row["capstone_topic"]), int(row["journal_topic"]))] = row
    fields = ["variant", "capstone_topic", "journal_topic", "rank", "transported_mass", "cosine_sim",
              "thematic_alignment", "shared_methodology", "justification", "cap_keywords", "jrn_keywords", "error"]
    log(f"rating {len(targets)} prompts ({len(done)} cached) with claude-sonnet-4-6 ...")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for k, (variant, fl) in enumerate(targets, 1):
            ci, ji = fl["capstone_topic"], fl["journal_topic"]
            key = (variant, ci, ji)
            if key in done:
                w.writerow(done[key]); continue
            docsA = [texts_by_pmid.get(p, "") for p in A["docs_by"][ci][:N_REP_DOCS]]
            docsB = [texts_by_pmid.get(p, "") for p in B["docs_by"][ji][:N_REP_DOCS]]
            prompt = build_prompt(ci, ji, A["kw"].get(ci, "?"), B["kw"].get(ji, "?"), docsA, docsB, variant)
            row = dict(variant=variant, capstone_topic=ci, journal_topic=ji, rank=fl["rank"],
                       transported_mass=round(fl["transported_mass"], 6), cosine_sim=round(fl["cosine_sim"], 4),
                       cap_keywords=A["kw"].get(ci, ""), jrn_keywords=B["kw"].get(ji, ""), error="",
                       thematic_alignment="", shared_methodology="", justification="")
            try:
                rt = mdb.call_claude(prompt)
                row.update(thematic_alignment=rt.get("thematic_alignment", ""),
                           shared_methodology=rt.get("shared_methodology", ""),
                           justification=rt.get("justification", ""))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            w.writerow(row); f.flush()
            log(f"  {k}/{len(targets)} {variant} A{ci}->B{ji} r{fl['rank']} -> {row['thematic_alignment']} {row['error']}")
            time.sleep(0.4)
    n_scrub = scrub_csv(out_csv, ["justification", "cap_keywords", "jrn_keywords"])

    rows = [x for x in csv.DictReader(open(out_csv)) if x["thematic_alignment"] not in ("", None)]
    mass = {(int(fl["capstone_topic"]), int(fl["journal_topic"])): fl["transported_mass"] for fl in r["flows"]}
    summary = dict(seed=SEED, n_journal=n, halves={t: {k: v for k, v in h.items() if k in ("n", "n_topics", "n_out", "out_rate")} for t, h in halves.items()},
                   w1_halves=r["w1"], align_halves=r["align"], w1_paper=ref["w1_distance"], align_paper=ref["alignment_score_exp"],
                   n_flows=len(r["flows"]), rank1_is_nearest_centroid=f"{rank1_is_nearest}/{A['n_topics']}",
                   n_redacted_fields=n_scrub, variants={})
    for variant in ("verbatim", "neutral"):
        vr = [x for x in rows if x["variant"] == variant]
        if not vr: continue
        rat = [int(float(x["thematic_alignment"])) for x in vr]
        byrank = {}
        for rk in (1, 2, 3):
            rr = [int(float(x["thematic_alignment"])) for x in vr if int(x["rank"]) == rk]
            if rr: byrank[rk] = dict(n=len(rr), dist=dict(sorted(Counter(rr).items())), mean=float(np.mean(rr)), n_ge3=int(sum(v >= 3 for v in rr)))
        tot_m = sum(mass[(int(x["capstone_topic"]), int(x["journal_topic"]))] for x in vr)
        mw = sum(mass[(int(x["capstone_topic"]), int(x["journal_topic"]))] * int(float(x["thematic_alignment"])) for x in vr) / tot_m
        summary["variants"][variant] = dict(n=len(rat), dist=dict(sorted(Counter(rat).items())), mean=float(np.mean(rat)),
                                            n_ge3=int(sum(v >= 3 for v in rat)), frac_ge3=float(np.mean([v >= 3 for v in rat])),
                                            mass_weighted_alignment=float(mw), by_rank=byrank,
                                            methodology=dict(Counter(x["shared_methodology"] for x in vr)))
    json.dump(summary, open(OUT / "r11_positive_control.json", "w"), indent=2, default=float)
    log(json.dumps(summary["variants"], indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
