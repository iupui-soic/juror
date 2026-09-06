"""Rate every new topic flow produced by the robustness analyses with the
unchanged JUROR prompt (same rubric, same 12 keywords + 3 representative docs,
same model), so each robustness check can be reported in rating units rather
than only as a change in transport mass.

Sources of new flows:
  * BTM's own top-3 selections per capstone topic  (R4 head-to-head)
  * year-matched OT, journals <=2024               (R2a / R2b)
  * alternative outlier handling S1/S2/S3          (R3)

Caches to out/r5_new_flow_ratings.csv; re-runs only what is missing.
"""
import os, sys, csv, json, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent.parent
BH = ROOT / "bhi2026"
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(BH))
import importlib.util
sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact_surfaces import scrub_csv
spec = importlib.util.spec_from_file_location("mdb", BH / "method_d_ot_llm_bridge.py")
mdb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdb)   # reuse prompt + helpers

N_REP_DOCS, DOC_CHARS = mdb.N_REP_DOCS, mdb.DOC_EXCERPT_CHARS


def collect_targets():
    have = {(int(r["capstone_topic"]), int(r["journal_topic"]))
            for r in csv.DictReader(open(BH / "phase3_method_d/flow_ratings.csv"))}
    want = defaultdict(list)
    btm = json.load(open(OUT / "r4_btm.json"))
    for ci, js in btm["selection_overlap"]["btm_top3"].items():
        for j in js:
            want[(int(ci), int(j))].append("BTM_top3")
    r2 = json.load(open(OUT / "r2_year_matched.json"))
    for ci, j in r2["year_matched"]["new_flows"]:
        want[(int(ci), int(j))].append("year_matched_<=2024")
    r2b = json.load(open(OUT / "r2b_refit_2024.json"))
    paper = {tuple(x) for x in r2b["paper_flows"]}
    for ci, j in r2b["mapped_flows"]:
        if (ci, j) not in paper:
            want[(int(ci), int(j))].append("refit_<=2024")
    r3 = json.load(open(OUT / "r3_outlier_handling.json"))
    for scen, d in r3.items():
        for ci, j in d.get("new_flows", []):
            if ci == -1 or j == -1:      # outlier "topic" is not a rateable theme
                continue
            want[(int(ci), int(j))].append(scen)
    return {k: sorted(set(v)) for k, v in want.items() if k not in have}, have


def call_openai(prompt, model="gpt-5.4-mini"):
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, max_completion_tokens=500)
    return json.loads(r.choices[0].message.content.strip())


def main():
    rater = sys.argv[1] if len(sys.argv) > 1 else "claude"
    rate = mdb.call_claude if rater == "claude" else call_openai
    print(f"rater = {rater}")
    targets, have = collect_targets()
    print(f"{len(have)} flows already rated in the paper; {len(targets)} new flows to rate")

    cap_kw = mdb.load_topic_keywords(BH / "phase3_method_a/capstone_topics.csv")
    jrn_kw = mdb.load_topic_keywords(BH / "phase3_method_a/journal_topics.csv")
    cap_by = mdb.load_doc_topics(BH / "phase3_method_a/capstone_docs.csv", "capstone_id")
    jrn_by = mdb.load_doc_topics(BH / "phase3_method_a/journal_docs.csv", "pmid")
    os.chdir(ROOT)
    cap_tx, jrn_tx = mdb.load_capstone_texts(), mdb.load_journal_texts()

    out_csv = OUT / f"r5_new_flow_ratings_{rater}.csv"
    done = {}
    if out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            if r.get("thematic_alignment") and not r.get("error"):
                done[(int(r["capstone_topic"]), int(r["journal_topic"]))] = r
    fields = ["capstone_topic", "journal_topic", "sources", "thematic_alignment",
              "shared_methodology", "justification", "cap_keywords", "jrn_keywords", "error"]
    print(f"writing {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for n, (key, srcs) in enumerate(sorted(targets.items()), 1):
            ci, ji = key
            if key in done:
                row = dict(done[key]); row["sources"] = ",".join(srcs); w.writerow(row); continue
            prompt = mdb.PROMPT_TEMPLATE.format(
                cap_topic_id=ci, jrn_topic_id=ji,
                cap_keywords=cap_kw.get(ci, "?"), jrn_keywords=jrn_kw.get(ji, "?"),
                cap_docs="\n".join(f"  ({k}) {cap_tx.get(c,'')[:DOC_CHARS].strip()}"
                                   for k, c in enumerate(cap_by[ci][:N_REP_DOCS], 1)),
                jrn_docs="\n".join(f"  ({k}) {jrn_tx.get(p,'')[:DOC_CHARS].strip()}"
                                   for k, p in enumerate(jrn_by[ji][:N_REP_DOCS], 1)))
            try:
                rt = rate(prompt)
                row = dict(capstone_topic=ci, journal_topic=ji, sources=",".join(srcs),
                           thematic_alignment=rt.get("thematic_alignment", ""),
                           shared_methodology=rt.get("shared_methodology", ""),
                           justification=rt.get("justification", ""),
                           cap_keywords=cap_kw.get(ci, ""), jrn_keywords=jrn_kw.get(ji, ""), error="")
            except Exception as e:
                row = dict(capstone_topic=ci, journal_topic=ji, sources=",".join(srcs),
                           thematic_alignment="", shared_methodology="", justification="",
                           cap_keywords=cap_kw.get(ci, ""), jrn_keywords=jrn_kw.get(ji, ""),
                           error=f"{type(e).__name__}: {e}")
            w.writerow(row); f.flush()
            print(f"  {n}/{len(targets)} C{ci}-J{ji} [{','.join(srcs)}] -> {row['thematic_alignment']} {row['error']}")
            time.sleep(0.4)
    n = scrub_csv(out_csv, ["justification", "cap_keywords", "jrn_keywords"])
    print(f"redacted {n} field(s) carrying an org surface")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
