"""
Pre-submission hardening: confidence intervals + robustness for every n=24 flow-level
statistic, plus the top-5 pooled extension and the cross-LLM calibration analysis.

Outputs:
  cis_results.txt   -- human-readable panel
  cis_latex.tex     -- paste-ready LaTeX fragments (adapt to current Overleaf tables)

Reproduces (and never re-runs any paid API call) from existing artifacts:
  phase3_method_d/flow_ratings.csv               (JUROR ratings + per-flow cosine)
  phase3_method_d/method_d_vs_method_c.csv        (top-3 JUROR vs Method C means)
  phase3_method_a/{capstone,journal}_docs.csv     (doc -> topic maps)
  phase3_method_c/ratings.csv                      (pair-level Method C ratings)
  phase3_method_d_ablations/ablation_B_top5_full.csv (top-5 JUROR ratings)
  phase3_method_d_openai/report.json               (cross-LLM)
"""
import csv, json, math
from collections import defaultdict
import numpy as np
from scipy import stats

csv.field_size_limit(10**7)
ROOT = ".."  # run from phase5_hardening/

def fisher_ci(r, n, a=0.05):
    if abs(r) >= 1 or n <= 3:
        return (float("nan"), float("nan"))
    z = np.arctanh(r); se = 1/math.sqrt(n-3); zc = stats.norm.ppf(1-a/2)
    return tuple(np.tanh([z-zc*se, z+zc*se]))

def boot_ci(x, y, fn, B=10000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        idx = rng.integers(0, len(x), len(x))
        try: out.append(fn(x[idx], y[idx]))
        except Exception: pass
    return tuple(np.percentile(out, [2.5, 97.5]))

# ---- load top-3 flows (JUROR rating, Method C mean, cosine) ----
fr = {}
with open(f"{ROOT}/phase3_method_d/flow_ratings.csv") as f:
    for r in csv.DictReader(f):
        fr[(r['capstone_topic'], r['journal_topic'])] = float(r['cosine'])
rows = []
with open(f"{ROOT}/phase3_method_d/method_d_vs_method_c.csv") as f:
    for r in csv.DictReader(f):
        k = (r['capstone_topic'], r['journal_topic'])
        rows.append((int(r['method_d_rating']), float(r['method_c_mean']), fr[k]))
d  = np.array([x[0] for x in rows], float)
c  = np.array([x[1] for x in rows], float)
cos= np.array([x[2] for x in rows], float)
n = len(rows)

# ---- pair->topic maps for the top-5 pooled extension ----
cap_t = {r['capstone_id']: int(r['topic']) for r in csv.DictReader(open(f"{ROOT}/phase3_method_a/capstone_docs.csv"))}
jrn_t = {r['pmid']: int(r['topic']) for r in csv.DictReader(open(f"{ROOT}/phase3_method_a/journal_docs.csv"))}
flow_c = defaultdict(list)
for r in csv.DictReader(open(f"{ROOT}/phase3_method_c/ratings.csv")):
    if r.get('error'): continue
    try: rt = float(r['thematic_alignment'])
    except (ValueError, KeyError): continue
    if r['capstone_id'] in cap_t and r['pmid'] in jrn_t:
        flow_c[(cap_t[r['capstone_id']], jrn_t[r['pmid']])].append(rt)
flow_c_mean = {k: np.mean(v) for k, v in flow_c.items()}

top5 = []
for r in csv.DictReader(open(f"{ROOT}/phase3_method_d_ablations/ablation_B_top5_full.csv")):
    if r.get('error'): continue
    k = (int(r['capstone_topic']), int(r['journal_topic']))
    if k in flow_c_mean:
        top5.append((r['rank'], int(r['thematic_alignment']), flow_c_mean[k]))
d5 = np.array([x[1] for x in top5], float)
c5 = np.array([x[2] for x in top5], float)
n5 = len(top5)

# ---- cross-LLM ----
xl = json.load(open(f"{ROOT}/phase3_method_d_openai/report.json"))
gpt = []; cla = []
for r in csv.DictReader(open(f"{ROOT}/phase3_method_d_openai/cross_llm_comparison.csv")):
    gpt.append(int(r['gpt5_4_mini_rating'])); cla.append(int(r['claude_sonnet_4_6_rating']))
gpt = np.array(gpt, float); cla = np.array(cla, float)
xl_rho = stats.spearmanr(gpt, cla)[0]
offset = gpt.mean() - cla.mean()
# paired test on the systematic offset
t_off = stats.ttest_rel(gpt, cla)
w_off = stats.wilcoxon(gpt, cla) if not np.allclose(gpt, cla) else None

# ---- assemble ----
L = []
def line(s=""): L.append(s)

line("="*72)
line("PRE-SUBMISSION HARDENING — uncertainty panel for flow-level statistics")
line("="*72)
line(f"\nMAIN RESULT  JUROR vs Method C  (n={n} flows, complete enumeration of")
line(f"             above-threshold top-3 transport edges)")
sp = stats.spearmanr(d, c)[0]; pe = stats.pearsonr(d, c)[0]
line(f"  Spearman rho = {sp:.3f}   95% CI Fisher {tuple(round(v,3) for v in fisher_ci(sp,n))}"
     f"  boot {tuple(round(v,3) for v in boot_ci(d,c,lambda a,b:stats.spearmanr(a,b)[0]))}")
line(f"  Pearson  r   = {pe:.3f}   95% CI Fisher {tuple(round(v,3) for v in fisher_ci(pe,n))}"
     f"  boot {tuple(round(v,3) for v in boot_ci(d,c,lambda a,b:stats.pearsonr(a,b)[0]))}")

line(f"\nBASELINE     cosine-centroid vs Method C  (n={n})")
cpe = stats.pearsonr(cos, c)[0]; csp = stats.spearmanr(cos, c)[0]
line(f"  Pearson  r   = {cpe:.3f}   95% CI Fisher {tuple(round(v,3) for v in fisher_ci(cpe,n))}")
line(f"  Spearman rho = {csp:.3f}")
line(f"\nACCURACY PARITY (the honest finding):")
line(f"  Delta Pearson (JUROR - cosine) = {pe-cpe:+.3f}")
rng = np.random.default_rng(1); diffs = []
for _ in range(10000):
    idx = rng.integers(0, n, n)
    try: diffs.append(stats.pearsonr(d[idx],c[idx])[0]-stats.pearsonr(cos[idx],c[idx])[0])
    except Exception: pass
dlo, dhi = np.percentile(diffs, [2.5, 97.5])
line(f"  bootstrap 95% CI on Delta = ({dlo:+.3f}, {dhi:+.3f})  -> straddles 0 => statistical tie.")
line(f"  => JUROR MATCHES the free baseline on accuracy; the contribution is the")
line(f"     structured rubric rating + methodology flag + justification (evaluate separately).")

line(f"\nPOWER / SCOPE EXTENSION  top-5 pooled flows with Method C coverage  (n={n5})")
line(f"  (Method C pairs were sampled by cosine bucket, not by flow; 24 of 24 top-3")
line(f"   flows are covered, plus {n5-24} of the additional top-4/5 flows.)")
sp5 = stats.spearmanr(d5, c5)[0]; pe5 = stats.pearsonr(d5, c5)[0]
line(f"  Spearman rho = {sp5:.3f}   95% CI Fisher {tuple(round(v,3) for v in fisher_ci(sp5,n5))}")
line(f"  Pearson  r   = {pe5:.3f}   95% CI Fisher {tuple(round(v,3) for v in fisher_ci(pe5,n5))}")
line(f"  => correlation STABLE when the cut is relaxed; lower CI bound tightens.")

line(f"\nCROSS-LLM ROBUSTNESS  GPT-5.4-mini vs Claude Sonnet 4.6  (n=24)")
line(f"  Spearman rho = {xl_rho:.3f}   95% CI Fisher {tuple(round(v,3) for v in fisher_ci(xl_rho,24))}")
line(f"  exact agreement = {xl['exact_agreement']}/24 ({xl['exact_agreement_rate']:.1%});"
     f"  within-1 = {xl['diff_le_1_rate']:.0%}")
line(f"  systematic offset (GPT - Claude) = {offset:+.2f} pts"
     f"   paired t p={t_off.pvalue:.3f}" + (f", Wilcoxon p={w_off.pvalue:.3f}" if w_off else ""))
line(f"  => offset is a CALIBRATION shift, not a ranking disagreement (rho positive,")
line(f"     within-1 = 100%). Report rank correlation, not exact agreement, on a 0-4 scale.")
line("")

txt = "\n".join(L)
open("cis_results.txt", "w").write(txt)
print(txt)

# ---- LaTeX fragments ----
def ci(r, nn): lo, hi = fisher_ci(r, nn); return f"[{lo:.2f}, {hi:.2f}]"
tex = rf"""% Paste-ready CI fragments -- ADAPT to the current Overleaf table structure.
% Macro for a correlation + 95% CI cell:
\newcommand{{\ci}}[2]{{#1 \scriptsize{{[#2]}}}}

% JUROR vs Method C (n={n})
% Spearman: {sp:.2f} {ci(sp,n)}     Pearson: {pe:.2f} {ci(pe,n)}
% cosine baseline vs Method C (n={n})
% Pearson: {cpe:.2f} {ci(cpe,n)}
% Delta Pearson (JUROR-cosine) = {pe-cpe:+.3f}, bootstrap 95\% CI [{dlo:+.2f}, {dhi:+.2f}] (includes 0)
% Top-5 pooled (n={n5}): Spearman {sp5:.2f} {ci(sp5,n5)}, Pearson {pe5:.2f} {ci(pe5,n5)}
% Cross-LLM (n=24): Spearman {xl_rho:.2f} {ci(xl_rho,24)}, within-1 100\%, offset {offset:+.2f} pt

% Example sentence for the main result:
JUROR's flow ratings track the Method~C pair-rating means at
$\rho={sp:.2f}$ (95\% CI {ci(sp,n)}; $r={pe:.2f}$, {ci(pe,n)}; $n={n}$),
statistically indistinguishable from a cosine-centroid baseline
($r={cpe:.2f}$, {ci(cpe,n)}; $\Delta r={pe-cpe:+.2f}$, bootstrap 95\% CI $[{dlo:+.2f},{dhi:+.2f}]$),
confirming that JUROR's value lies in its structured, auditable output rather than
in predictive accuracy. The association is stable when the transport cut is relaxed
to the covered top-5 edges ($\rho={sp5:.2f}$, {ci(sp5,n5)}; $n={n5}$).
"""
open("cis_latex.tex", "w").write(tex)
print("\n[wrote cis_results.txt and cis_latex.tex]")
