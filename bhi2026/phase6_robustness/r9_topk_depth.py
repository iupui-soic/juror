"""R9 — how deep does the exception go?

The paper rates the top-3 transport edges per capstone topic (24 flows) and
reports one rating-3 flow that sits at rank 4. The obvious follow-up is whether
more exceptions appear further down the transport plan, i.e. whether "one
exception" is a property of the corpora or an artefact of stopping at K=3.

This rates every edge down to rank 10 (71 flows; the plan is sparse, so several
capstone topics have fewer than 10 non-zero destinations) with the unchanged
JUROR prompt, rubric and model, and reports the rating distribution and the
cumulative count of rating->=3 flows as a function of K.
"""
import os, sys, csv, json, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import importlib.util

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
from redact_surfaces import scrub_csv

ROOT = HERE.parent.parent
BH = ROOT / "bhi2026"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
csv.field_size_limit(sys.maxsize)
spec = importlib.util.spec_from_file_location("mdb", BH / "method_d_ot_llm_bridge.py")
mdb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdb)
MAX_K = 10


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)

    ce, je, cd, jd, ct, jt = common.load_all()
    cap_ids, Ccap = common.centroids(ce, ct)
    jrn_ids, Cjrn = common.centroids(je, jt)
    r = common.solve_ot(Ccap, common.masses(ct, cap_ids), Cjrn, common.masses(jt, jrn_ids),
                        topk=MAX_K, cap_ids=cap_ids, jrn_ids=jrn_ids)
    flows = r["flows"]
    log(f"{len(flows)} flows down to rank {MAX_K}")

    known = {}
    for row in csv.DictReader(open(BH / "phase3_method_d/flow_ratings.csv")):
        if row["thematic_alignment"]:
            known[(int(row["capstone_topic"]), int(row["journal_topic"]))] = row
    log(f"{len(known)} already rated in the paper (top-3)")

    out_csv = OUT / "r9_topk_ratings.csv"
    done = {}
    if out_csv.exists():
        for row in csv.DictReader(open(out_csv)):
            if row.get("thematic_alignment") and not row.get("error"):
                done[(int(row["capstone_topic"]), int(row["journal_topic"]))] = row
        log(f"resume: {len(done)} already in {out_csv.name}")

    cap_kw = mdb.load_topic_keywords(BH / "phase3_method_a/capstone_topics.csv")
    jrn_kw = mdb.load_topic_keywords(BH / "phase3_method_a/journal_topics.csv")
    cap_by = mdb.load_doc_topics(BH / "phase3_method_a/capstone_docs.csv", "capstone_id")
    jrn_by = mdb.load_doc_topics(BH / "phase3_method_a/journal_docs.csv", "pmid")
    os.chdir(ROOT)
    cap_tx, jrn_tx = mdb.load_capstone_texts(), mdb.load_journal_texts()

    fields = ["capstone_topic", "journal_topic", "rank", "mass", "cosine", "source",
              "thematic_alignment", "shared_methodology", "justification",
              "cap_keywords", "jrn_keywords", "error"]
    todo = [f for f in flows if (f["capstone_topic"], f["journal_topic"]) not in known
            and (f["capstone_topic"], f["journal_topic"]) not in done]
    log(f"{len(todo)} new LLM calls (~${0.008*len(todo):.2f})")

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for i, fl in enumerate(flows, 1):
            key = (fl["capstone_topic"], fl["journal_topic"])
            base = dict(capstone_topic=key[0], journal_topic=key[1], rank=fl["rank"],
                        mass=round(fl["transported_mass"], 6), cosine=round(fl["cosine_sim"], 4))
            if key in known:
                k = known[key]
                w.writerow({**base, "source": "paper_top3",
                            "thematic_alignment": k["thematic_alignment"],
                            "shared_methodology": k["shared_methodology"],
                            "justification": k["justification"],
                            "cap_keywords": k["cap_keywords"], "jrn_keywords": k["jrn_keywords"],
                            "error": ""})
                continue
            if key in done:
                w.writerow({**base, **{c: done[key].get(c, "") for c in
                                       ["thematic_alignment", "shared_methodology", "justification",
                                        "cap_keywords", "jrn_keywords", "error"]},
                            "source": "r9_depth"})
                continue
            prompt = mdb.PROMPT_TEMPLATE.format(
                cap_topic_id=key[0], jrn_topic_id=key[1],
                cap_keywords=cap_kw.get(key[0], "?"), jrn_keywords=jrn_kw.get(key[1], "?"),
                cap_docs="\n".join(f"  ({n}) {cap_tx.get(c,'')[:600].strip()}"
                                   for n, c in enumerate(cap_by[key[0]][:3], 1)),
                jrn_docs="\n".join(f"  ({n}) {jrn_tx.get(p,'')[:600].strip()}"
                                   for n, p in enumerate(jrn_by[key[1]][:3], 1)))
            try:
                rt = mdb.call_claude(prompt)
                row = {**base, "source": "r9_depth",
                       "thematic_alignment": rt.get("thematic_alignment", ""),
                       "shared_methodology": rt.get("shared_methodology", ""),
                       "justification": rt.get("justification", ""),
                       "cap_keywords": cap_kw.get(key[0], ""), "jrn_keywords": jrn_kw.get(key[1], ""),
                       "error": ""}
            except Exception as e:
                row = {**base, "source": "r9_depth", "thematic_alignment": "",
                       "shared_methodology": "", "justification": "",
                       "cap_keywords": "", "jrn_keywords": "", "error": f"{type(e).__name__}: {e}"}
            w.writerow(row); fh.flush()
            log(f"  {i}/{len(flows)} C{key[0]}-J{key[1]} rank {fl['rank']} -> "
                f"{row['thematic_alignment']} {row['error']}")
            time.sleep(0.4)

    n = scrub_csv(out_csv, ["justification", "cap_keywords", "jrn_keywords"])
    log(f"redacted {n} field(s) carrying an org surface")

    rows = [r for r in csv.DictReader(open(out_csv)) if r["thematic_alignment"]]
    log(f"\n=== rating distribution by transport rank (n={len(rows)}) ===")
    by_rank = defaultdict(list)
    for r_ in rows:
        by_rank[int(r_["rank"])].append(int(r_["thematic_alignment"]))
    print(f"{'rank':>5}{'n':>5}{'  0  1  2  3  4':>16}{'mean':>7}{'>=3':>5}")
    for k in sorted(by_rank):
        v = by_rank[k]; c = Counter(v)
        print(f"{k:>5}{len(v):>5}  " + " ".join(f"{c.get(i,0):2d}" for i in range(5)) +
              f"{np.mean(v):>7.2f}{sum(1 for x in v if x>=3):>5}")

    log("\n=== cumulative as a function of top-K ===")
    print(f"{'K':>4}{'flows':>7}{'mean':>7}{'flows >=3':>11}{'  which':>10}")
    cum = {}
    for K in range(1, MAX_K + 1):
        sel = [r_ for r_ in rows if int(r_["rank"]) <= K]
        v = [int(r_["thematic_alignment"]) for r_ in sel]
        ge3 = [f"C{r_['capstone_topic']}-J{r_['journal_topic']}" for r_ in sel
               if int(r_["thematic_alignment"]) >= 3]
        cum[K] = dict(n=len(v), mean=float(np.mean(v)), n_ge3=len(ge3), which=ge3)
        print(f"{K:>4}{len(v):>7}{np.mean(v):>7.2f}{len(ge3):>11}   {', '.join(ge3) or '-'}")

    json.dump(dict(max_k=MAX_K, n_flows=len(rows),
                   by_rank={str(k): dict(n=len(v), dist=dict(Counter(v)), mean=float(np.mean(v)),
                                         n_ge3=sum(1 for x in v if x >= 3))
                            for k, v in sorted(by_rank.items())},
                   cumulative=cum), open(OUT / "r9_topk_depth.json", "w"), indent=2, default=float)
    log(f"wrote {OUT/'r9_topk_depth.json'}")


if __name__ == "__main__":
    main()
