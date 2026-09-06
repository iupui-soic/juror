"""R8b — rate the matched-granularity (k=8) flows with the unchanged JUROR prompt.

A merged journal topic is described to the LLM the same way a native one is:
its 12 top keywords (pooled across member topics in descending mass order,
de-duplicated) and 3 representative documents drawn from its highest-mass
members. Everything else -- rubric, model, schema -- is untouched.
"""
import os, sys, csv, json, time, ast
from pathlib import Path
import importlib.util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact_surfaces import scrub_csv

ROOT = Path(__file__).resolve().parent.parent.parent
BH = ROOT / "bhi2026"
OUT = Path(__file__).resolve().parent / "out"
csv.field_size_limit(sys.maxsize)
spec = importlib.util.spec_from_file_location("mdb", BH / "method_d_ot_llm_bridge.py")
mdb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdb)


def topic_reps(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try: t = int(r["Topic"])
        except (KeyError, ValueError): continue
        try: kws = ast.literal_eval(r["Representation"])
        except Exception: kws = []
        out[t] = kws
    return out


def main():
    rater = sys.argv[1] if len(sys.argv) > 1 else "claude"
    if rater == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)
    flows = list(csv.DictReader(open(OUT / "r8_flows_k8.csv")))
    jrn_kw_full = topic_reps(BH / "phase3_method_a/journal_topics.csv")
    cap_kw = mdb.load_topic_keywords(BH / "phase3_method_a/capstone_topics.csv")
    cap_by = mdb.load_doc_topics(BH / "phase3_method_a/capstone_docs.csv", "capstone_id")
    jrn_by = mdb.load_doc_topics(BH / "phase3_method_a/journal_docs.csv", "pmid")
    os.chdir(ROOT)
    cap_tx, jrn_tx = mdb.load_capstone_texts(), mdb.load_journal_texts()

    fields = ["capstone_topic", "journal_topic_merged", "members", "mass", "cosine",
              "thematic_alignment", "shared_methodology", "justification",
              "cap_keywords", "jrn_keywords", "error"]
    with open(OUT / f"r8b_k8_ratings_{rater}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, fl in enumerate(flows, 1):
            ci = int(fl["capstone_topic"]); members = [int(x) for x in fl["members"].split()]
            kws, seen = [], set()
            for m in members:
                for kw in jrn_kw_full.get(m, []):
                    if kw not in seen:
                        seen.add(kw); kws.append(kw)
                    if len(kws) >= 12: break
                if len(kws) >= 12: break
            jrn_docs, k = [], 0
            for m in members:
                for pmid in jrn_by.get(m, [])[:1]:
                    k += 1; jrn_docs.append(f"  ({k}) {jrn_tx.get(pmid,'')[:600].strip()}")
                if k >= 3: break
            prompt = mdb.PROMPT_TEMPLATE.format(
                cap_topic_id=ci, jrn_topic_id=f"merged-{fl['journal_topic']}",
                cap_keywords=cap_kw.get(ci, "?"), jrn_keywords=", ".join(kws),
                cap_docs="\n".join(f"  ({j}) {cap_tx.get(c,'')[:600].strip()}"
                                   for j, c in enumerate(cap_by[ci][:3], 1)),
                jrn_docs="\n".join(jrn_docs))
            try:
                rt = mdb.call_claude(prompt)
                row = dict(capstone_topic=ci, journal_topic_merged=fl["journal_topic"],
                           members=fl["members"], mass=fl["transported_mass"], cosine=fl["cosine_sim"],
                           thematic_alignment=rt.get("thematic_alignment", ""),
                           shared_methodology=rt.get("shared_methodology", ""),
                           justification=rt.get("justification", ""),
                           cap_keywords=cap_kw.get(ci, ""), jrn_keywords=", ".join(kws), error="")
            except Exception as e:
                row = dict(capstone_topic=ci, journal_topic_merged=fl["journal_topic"],
                           members=fl["members"], mass=fl["transported_mass"], cosine=fl["cosine_sim"],
                           thematic_alignment="", shared_methodology="", justification="",
                           cap_keywords="", jrn_keywords="", error=f"{type(e).__name__}: {e}")
            w.writerow(row); f.flush()
            print(f"  {i}/{len(flows)} C{ci}-J*{fl['journal_topic']} ({len(members)} src topics) -> "
                  f"{row['thematic_alignment']} {row['error']}")
            time.sleep(0.4)
    n = scrub_csv(OUT / f"r8b_k8_ratings_{rater}.csv", ["justification", "cap_keywords", "jrn_keywords"])
    print(f"redacted {n} field(s) carrying an org surface")
    print(f"Wrote {OUT}/r8b_k8_ratings_{rater}.csv")


if __name__ == "__main__":
    main()
