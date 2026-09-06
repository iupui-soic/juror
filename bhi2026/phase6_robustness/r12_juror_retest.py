"""R12 — Test-retest reliability of the JUROR flow rating.

Re-rate the 24 main flows twice more with the unchanged prompt, rater
(claude-sonnet-4-6) and default sampling temperature. Reports agreement among
the three runs (exact, within-1, Spearman, one-way ICC) and each run's
agreement with the Method C target. Needed to interpret rho = 0.71 against a
target whose own reliability is ICC(1,k) = 0.72.

Outputs (out/): r12_retest_ratings.csv, r12_juror_retest.json
"""
import os, sys, csv, json, time
from pathlib import Path
from collections import Counter
import importlib.util
import numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact_surfaces import scrub_csv

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent.parent
BH = ROOT / "bhi2026"
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("mdb", BH / "method_d_ot_llm_bridge.py")
mdb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdb)
N_REP_DOCS, DOC_CHARS = mdb.N_REP_DOCS, mdb.DOC_EXCERPT_CHARS
RUNS = (2, 3)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def icc_oneway(groups):
    """groups: list of lists (one list of replicate ratings per flow)."""
    k = len(groups); N = sum(len(g) for g in groups)
    gm = np.mean([x for g in groups for x in g])
    msb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups) / (k - 1)
    msw = sum((x - np.mean(g)) ** 2 for g in groups for x in g) / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)), float((msb - msw) / msb)


def main():
    main_rows = list(csv.DictReader(open(BH / "phase3_method_d/flow_ratings.csv")))
    rcol = "thematic_alignment" if "thematic_alignment" in main_rows[0] else "method_d_rating"
    flows = [(int(r["capstone_topic"]), int(r["journal_topic"])) for r in main_rows]
    run1 = {(int(r["capstone_topic"]), int(r["journal_topic"])): int(float(r[rcol])) for r in main_rows}
    target = {(int(r["capstone_topic"]), int(r["journal_topic"])): float(r["method_c_mean"])
              for r in csv.DictReader(open(BH / "phase3_method_d/method_d_vs_method_c.csv"))}
    cap_kw = mdb.load_topic_keywords(BH / "phase3_method_a/capstone_topics.csv")
    jrn_kw = mdb.load_topic_keywords(BH / "phase3_method_a/journal_topics.csv")
    cap_by = mdb.load_doc_topics(BH / "phase3_method_a/capstone_docs.csv", "capstone_id")
    jrn_by = mdb.load_doc_topics(BH / "phase3_method_a/journal_docs.csv", "pmid")
    os.chdir(ROOT)
    cap_tx, jrn_tx = mdb.load_capstone_texts(), mdb.load_journal_texts()

    out_csv = OUT / "r12_retest_ratings.csv"
    done = {}
    if out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            if r.get("thematic_alignment") not in ("", None) and not r.get("error"):
                done[(int(r["run"]), int(r["capstone_topic"]), int(r["journal_topic"]))] = r
    fields = ["run", "capstone_topic", "journal_topic", "thematic_alignment", "shared_methodology", "justification", "error"]
    targets = [(run, ci, ji) for run in RUNS for ci, ji in flows]
    log(f"{len(targets)} calls ({len(done)} cached)")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for n, (run, ci, ji) in enumerate(targets, 1):
            if (run, ci, ji) in done:
                w.writerow(done[(run, ci, ji)]); continue
            prompt = mdb.PROMPT_TEMPLATE.format(
                cap_topic_id=ci, jrn_topic_id=ji,
                cap_keywords=cap_kw.get(ci, "?"), jrn_keywords=jrn_kw.get(ji, "?"),
                cap_docs="\n".join(f"  ({k}) {cap_tx.get(c,'')[:DOC_CHARS].strip()}" for k, c in enumerate(cap_by[ci][:N_REP_DOCS], 1)),
                jrn_docs="\n".join(f"  ({k}) {jrn_tx.get(p,'')[:DOC_CHARS].strip()}" for k, p in enumerate(jrn_by[ji][:N_REP_DOCS], 1)))
            row = dict(run=run, capstone_topic=ci, journal_topic=ji, thematic_alignment="", shared_methodology="", justification="", error="")
            try:
                rt = mdb.call_claude(prompt)
                row.update(thematic_alignment=rt.get("thematic_alignment", ""), shared_methodology=rt.get("shared_methodology", ""),
                           justification=rt.get("justification", ""))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            w.writerow(row); f.flush()
            log(f"  {n}/{len(targets)} run{run} C{ci}-J{ji} -> {row['thematic_alignment']} (run1={run1[(ci, ji)]}) {row['error']}")
            time.sleep(0.4)
    scrub_csv(out_csv, ["justification"])

    runs = {1: run1}
    for r in csv.DictReader(open(out_csv)):
        if r["thematic_alignment"] not in ("", None):
            runs.setdefault(int(r["run"]), {})[(int(r["capstone_topic"]), int(r["journal_topic"]))] = int(float(r["thematic_alignment"]))
    common = [fl for fl in flows if all(fl in runs[k] for k in runs)]
    res = dict(n_flows=len(common), runs={}, pairwise={})
    for k, d in runs.items():
        v = [d[fl] for fl in common]; t = [target[fl] for fl in common]
        res["runs"][k] = dict(dist=dict(sorted(Counter(v).items())), mean=float(np.mean(v)), n_ge3=int(sum(x >= 3 for x in v)),
                              spearman_vs_method_c=float(spearmanr(v, t).correlation), pearson_vs_method_c=float(pearsonr(v, t)[0]))
    ks = sorted(runs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a = [runs[ks[i]][fl] for fl in common]; b = [runs[ks[j]][fl] for fl in common]
            res["pairwise"][f"run{ks[i]}_vs_run{ks[j]}"] = dict(exact=float(np.mean([x == y for x, y in zip(a, b)])),
                                                              within1=float(np.mean([abs(x - y) <= 1 for x, y in zip(a, b)])),
                                                              spearman=float(spearmanr(a, b).correlation))
    icc1, icck = icc_oneway([[runs[k][fl] for k in ks] for fl in common])
    res["icc_single_call"] = icc1; res["icc_mean_of_runs"] = icck; res["n_runs"] = len(ks)
    mean3 = [float(np.mean([runs[k][fl] for k in ks])) for fl in common]
    res["mean_of_runs_vs_method_c_spearman"] = float(spearmanr(mean3, [target[fl] for fl in common]).correlation)
    json.dump(res, open(OUT / "r12_juror_retest.json", "w"), indent=2)
    log(json.dumps(res, indent=1))
    log("done")


if __name__ == "__main__":
    main()
