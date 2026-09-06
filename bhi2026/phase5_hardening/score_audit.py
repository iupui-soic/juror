"""
Score the filled faithfulness_audit_sheet.xlsx (one or more rater copies).

Produces the two numbers the paper needs:
  (A) DIRECT human<->JUROR agreement  (answers the 'two steps removed' objection)
      Spearman / Pearson / exact / within-1 of expert_rating vs JUROR rating, n=24.
  (B) EXPLANATION QUALITY  (answers 'interpretability asserted, never measured')
      per-axis faithfulness rate, methodology-flag accuracy, hallucination rate.
  If >1 rater file is given, also reports inter-rater agreement on both.

Usage:
  python3 score_audit.py rater1_filled.xlsx [rater2_filled.xlsx ...]
  (default: faithfulness_audit_sheet.xlsx)
  A rater's main workbook and supplement can be joined with '+', e.g.
  python3 score_audit.py "r1_main.xlsx+r1_supp.xlsx" "r2_main.xlsx+r2_supp.xlsx"
  Results are then reported for all flows and for the 24 paper flows alone.
"""
import sys, csv
from pathlib import Path
import numpy as np
from scipy import stats
import openpyxl

ROOT = Path("..").resolve()

# JUROR ratings keyed by flow_id "C{ct}-J{jt}": the 24 paper flows first, then
# the supplement flows (rank-4 rows of the depth sweep and the BTM-selected
# C7-J22), primary rater. Paper flows are never overridden.
juror, PAPER_FLOWS = {}, set()
for r in csv.DictReader(open(ROOT / "phase3_method_d/flow_ratings.csv")):
    fid = f"C{int(r['capstone_topic'])}-J{int(r['journal_topic'])}"
    juror[fid] = dict(rating=int(r["thematic_alignment"]), method=r["shared_methodology"].strip().lower())
    PAPER_FLOWS.add(fid)
for extra in ("phase6_robustness/out/r9_topk_ratings.csv", "phase6_robustness/out/r5_new_flow_ratings_claude.csv"):
    path = ROOT / extra
    if not path.exists(): continue
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("thematic_alignment") in ("", None): continue
        fid = f"C{int(r['capstone_topic'])}-J{int(r['journal_topic'])}"
        juror.setdefault(fid, dict(rating=int(float(r["thematic_alignment"])),
                                   method=r["shared_methodology"].strip().lower()))

def col_idx(ws, header):
    hdr = {ws.cell(1, c).value: c for c in range(1, ws.max_column + 1)}
    return hdr[header]

def read_independent(path):
    ws = openpyxl.load_workbook(path, data_only=True)["1_Rate_blind"]
    cf, cr, cm = col_idx(ws, "flow_id"), col_idx(ws, "your_rating_0to4"), col_idx(ws, "your_shared_methodology")
    out = {}
    for r in range(2, ws.max_row + 1):
        fid = ws.cell(r, cf).value
        if not fid: continue
        v = ws.cell(r, cr).value
        out[fid] = dict(rating=(None if v in (None, "") else float(v)),
                        method=(ws.cell(r, cm).value or "").strip().lower())
    return out

def read_faith(path):
    ws = openpyxl.load_workbook(path, data_only=True)["2_Audit_justification"]
    ci = {h: col_idx(ws, h) for h in
          ["flow_id", "faith_capstone_desc_210", "faith_journal_desc_210",
           "faith_relationship_210", "method_flag_correct_yn", "hallucinated_overlap_yn"]}
    out = {}
    for r in range(2, ws.max_row + 1):
        fid = ws.cell(r, ci["flow_id"]).value
        if not fid: continue
        g = lambda h: ws.cell(r, ci[h]).value
        out[fid] = dict(
            cap=g("faith_capstone_desc_210"), jrn=g("faith_journal_desc_210"),
            rel=g("faith_relationship_210"),
            flag=(g("method_flag_correct_yn") or "").strip().lower(),
            hall=(g("hallucinated_overlap_yn") or "").strip().lower())
    return out

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def fisher_ci(r, n, a=0.05):
    import math
    if abs(r) >= 1 or n <= 3: return (float("nan"), float("nan"))
    z = np.arctanh(r); se = 1/math.sqrt(n-3); zc = stats.norm.ppf(1-a/2)
    return tuple(round(v, 3) for v in np.tanh([z-zc*se, z+zc*se]))

def agreement_block(ind, fids, label):
    e = np.array([ind[f]["rating"] for f in fids])
    j = np.array([juror[f]["rating"] for f in fids])
    n = len(fids)
    exact = np.mean(e == j); w1 = np.mean(np.abs(e - j) <= 1)
    print(f"\n(A) DIRECT human<->JUROR rating agreement — {label} (n={n})")
    if n >= 4 and e.std() > 0 and j.std() > 0:
        sp = stats.spearmanr(e, j)[0]; pe = stats.pearsonr(e, j)[0]
        print(f"    Spearman rho = {sp:.3f}  95% CI {fisher_ci(sp,n)}")
        print(f"    Pearson  r   = {pe:.3f}  95% CI {fisher_ci(pe,n)}")
    print(f"    exact = {exact:.1%}   within-1 = {w1:.1%}   mean expert {e.mean():.2f} vs JUROR {j.mean():.2f}")
    print(f"    expert >=3 on {int((e>=3).sum())} flows; JUROR >=3 on {int((j>=3).sum())}; "
          f"both >=3 on {int(((e>=3)&(j>=3)).sum())}")
    mfl = [(ind[f]["method"], juror[f]["method"]) for f in fids if ind[f]["method"]]
    if mfl:
        print(f"    shared-methodology flag: expert vs JUROR exact = {np.mean([a == b for a, b in mfl]):.1%} (n={len(mfl)})")


def report_one(spec):
    paths = spec.split("+")
    print(f"\n{'='*64}\nRATER FILE(S): {', '.join(paths)}\n{'='*64}")
    ind, fa = {}, {}
    for path in paths:
        ind.update(read_independent(path)); fa.update(read_faith(path))
    unknown = [f for f in ind if f not in juror]
    if unknown:
        print(f"  [WARNING: no JUROR rating on file for {', '.join(unknown)} — excluded]")
        ind = {f: v for f, v in ind.items() if f in juror}
    fids = [f for f in ind if ind[f]["rating"] is not None]
    if not fids:
        print("  [Sheet 1 not yet filled — no expert ratings found]")
    else:
        agreement_block(ind, fids, "all rated flows")
        paper = [f for f in fids if f in PAPER_FLOWS]
        if len(paper) < len(fids) and paper:
            agreement_block(ind, paper, "24 paper flows only")
        supp = [f for f in fids if f not in PAPER_FLOWS]
        if supp:
            print("    supplement flows (expert vs JUROR): " +
                  ", ".join(f"{f}: {ind[f]['rating']:.0f} vs {juror[f]['rating']}" for f in supp))
    have = [f for f in fa if num(fa[f]["cap"]) is not None]
    if not have:
        print("\n  [Sheet 2 not yet filled — no faithfulness scores found]")
        return ind, fa
    print(f"\n(B) EXPLANATION QUALITY  (n={len(have)} justifications audited)")
    for axis, key in [("capstone description", "cap"), ("journal description", "jrn"),
                      ("relationship validity", "rel")]:
        vals = [num(fa[f][key]) for f in have if num(fa[f][key]) is not None]
        full = np.mean([v == 2 for v in vals]); mean = np.mean(vals)
        print(f"    {axis:<24}: fully faithful {full:.0%}  | mean {mean:.2f}/2")
    flags = [fa[f]["flag"] for f in have if fa[f]["flag"] in ("yes", "no")]
    if flags:
        print(f"    methodology-flag correct: {np.mean([x=='yes' for x in flags]):.0%}")
    halls = [fa[f]["hall"] for f in have if fa[f]["hall"] in ("yes", "no")]
    if halls:
        hr = np.mean([x == "yes" for x in halls])
        print(f"    HALLUCINATED overlap rate: {hr:.0%}  ({sum(x=='yes' for x in halls)}/{len(halls)})")
        bad = [f for f in have if fa[f]["hall"] == "yes"]
        if bad: print(f"      flows flagged: {', '.join(bad)}")
    return ind, fa

def inter_rater(reps):
    print(f"\n{'='*64}\nINTER-RATER AGREEMENT ({len(reps)} raters)\n{'='*64}")
    # ratings
    inds = [r[0] for r in reps]
    common = set.intersection(*[{f for f in d if d[f]['rating'] is not None} for d in inds])
    if len(common) >= 3:
        common = sorted(common)
        M = np.array([[d[f]['rating'] for f in common] for d in inds])
        if len(inds) == 2:
            print(f"  Expert rating: Spearman rho = {stats.spearmanr(M[0],M[1])[0]:.3f}, "
                  f"exact = {np.mean(M[0]==M[1]):.1%}, within-1 = {np.mean(np.abs(M[0]-M[1])<=1):.1%} (n={len(common)})")
    fas = [r[1] for r in reps]
    for axis in ["cap", "jrn", "rel"]:
        com = set.intersection(*[{f for f in d if num(d[f][axis]) is not None} for d in fas])
        if len(com) >= 3 and len(fas) == 2:
            com = sorted(com)
            a = np.array([num(fas[0][f][axis]) for f in com]); b = np.array([num(fas[1][f][axis]) for f in com])
            print(f"  Faithfulness[{axis}]: exact = {np.mean(a==b):.1%} (n={len(com)})")

if __name__ == "__main__":
    files = sys.argv[1:] or ["faithfulness_audit_sheet.xlsx"]
    reps = [report_one(p) for p in files]
    if len(files) > 1:
        inter_rater(reps)
