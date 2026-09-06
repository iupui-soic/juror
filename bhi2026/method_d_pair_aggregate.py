"""Method D pair-aggregate variant — intermediate granularity ablation.

Instead of one LLM call rating the topic-pair holistically (with keywords +
representative document excerpts), rate 9 individual (capstone_doc, journal_doc)
pairs from top-3 × top-3 representative documents per flow, then aggregate to
a flow-level score by averaging.

This tests whether per-pair granularity recovers Method C's signal better than
the single holistic call, or whether the holistic call already saturates.

  24 flows × 9 pair calls = 216 calls × ~$0.008 = ~$1.73, ~10 min.

If pair-aggregate Method D agrees more with Method C means (ρ > 0.71), the
intermediate variant is the better operating point. If it agrees about the
same, the holistic call captures the same information at 1/9 the cost.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

csv.field_size_limit(sys.maxsize)

FLOWS_CSV = Path("bhi2026/phase3_method_a/capstone_to_journal_flows.csv")
CAP_DOCS = Path("bhi2026/phase3_method_a/capstone_docs.csv")
JRN_DOCS = Path("bhi2026/phase3_method_a/journal_docs.csv")
CAP_META = Path("bhi2026/metadata/corpus_metadata.csv")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
JOURNAL_CSV = Path("bhi2026/journals/journal_corpus.csv")
METHOD_C_CSV = Path("bhi2026/phase3_method_c/ratings.csv")
MAIN_D_REPORT = Path("bhi2026/phase3_method_d/validation_report.json")

OUT_DIR = Path("bhi2026/phase3_method_d_pair_aggregate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DOCS_PER_SIDE = 3
DOC_EXCERPT_CHARS = 3500

# Same per-pair rubric as Method C
PAIR_PROMPT_TEMPLATE = """You are an expert reviewer of biomedical informatics research. Rate the THEMATIC ALIGNMENT between the following two documents on a 0-4 scale.

0 — Unrelated. The documents cover completely different problems and methods.
1 — Tangentially related. They share a broad domain (e.g., both are healthcare) but tackle different research questions and use different methods.
2 — Adjacent. They share either the research question or the method, but not both.
3 — Substantially overlapping. They share the same research question and use overlapping methods, but the contributions are distinct.
4 — Same research question. They essentially address the same research question with comparable methods, even if the populations or settings differ.

Additionally indicate SHARED METHODOLOGY:
"yes" if the documents use overlapping methods.
"partial" if one technique is shared but the rest of the pipeline differs.
"no" if the methods are unrelated.

Provide a JUSTIFICATION of no more than 60 words.

Return ONLY a JSON object with this schema:
{{
  "thematic_alignment": <int 0-4>,
  "shared_methodology": "yes" | "partial" | "no",
  "justification": "<string>"
}}

=== Document A (capstone) ===
{doc_a}

=== Document B (journal) ===
{doc_b}
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_doc_topics(path: Path, id_col: str) -> dict[int, list[str]]:
    by_topic: dict[int, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["topic"])
            except (KeyError, ValueError):
                continue
            by_topic[t].append(r[id_col])
    return by_topic


def load_capstone_texts() -> dict[str, str]:
    out = {}
    with open(CAP_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = TXT_ROOT / (r["source_path"] + ".txt")
            out[r["capstone_id"]] = p.read_text(encoding="utf-8") if p.exists() else ""
    return out


def load_journal_texts() -> dict[str, str]:
    out = {}
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["pmid"]] = ((r.get("title") or "") + ". " + (r.get("abstract") or "")).strip()
    return out


def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)

    flows = []
    with open(FLOWS_CSV) as f:
        for r in csv.DictReader(f):
            flows.append({
                "capstone_topic": int(r["capstone_topic"]),
                "journal_topic": int(r["journal_topic"]),
                "rank": int(r["rank"]),
                "mass": float(r["transported_mass"]),
                "cosine": float(r["cosine_sim"]),
            })
    log(f"Flows: {len(flows)}")

    cap_docs_by_t = load_doc_topics(CAP_DOCS, "capstone_id")
    jrn_docs_by_t = load_doc_topics(JRN_DOCS, "pmid")
    cap_texts = load_capstone_texts()
    jrn_texts = load_journal_texts()

    # Build pair list: for each flow, all (cap_doc_i, jrn_doc_j) combinations
    # from top-3 × top-3 docs.
    pairs = []
    for fl in flows:
        ct = fl["capstone_topic"]; jt = fl["journal_topic"]
        cap_doc_ids = cap_docs_by_t[ct][:N_DOCS_PER_SIDE]
        jrn_doc_ids = jrn_docs_by_t[jt][:N_DOCS_PER_SIDE]
        for cid in cap_doc_ids:
            for pmid in jrn_doc_ids:
                pairs.append({
                    "capstone_topic": ct, "journal_topic": jt,
                    "capstone_id": cid, "pmid": pmid,
                    "mass": fl["mass"], "cosine": fl["cosine"], "rank": fl["rank"],
                })
    log(f"Total doc pairs to rate: {len(pairs)}")

    # Resume support
    out_csv = OUT_DIR / "pair_ratings.csv"
    done: set[tuple[str, str]] = set()
    existing = []
    if out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                if r.get("thematic_alignment") and not r.get("error"):
                    done.add((r["capstone_id"], r["pmid"]))
                    existing.append(r)
        log(f"Resume: {len(done)} pairs already rated")

    fieldnames = ["capstone_topic", "journal_topic", "capstone_id", "pmid",
                  "mass", "cosine", "rank",
                  "thematic_alignment", "shared_methodology", "justification", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        n_done = len(done)
        for i, p in enumerate(pairs, start=1):
            if (p["capstone_id"], p["pmid"]) in done:
                continue
            prompt = PAIR_PROMPT_TEMPLATE.format(
                doc_a=cap_texts.get(p["capstone_id"], "")[:DOC_EXCERPT_CHARS],
                doc_b=jrn_texts.get(p["pmid"], "")[:DOC_EXCERPT_CHARS],
            )
            try:
                rating = call_claude(prompt)
                row = {**p,
                       "thematic_alignment": rating.get("thematic_alignment", ""),
                       "shared_methodology": rating.get("shared_methodology", ""),
                       "justification": rating.get("justification", ""),
                       "error": ""}
            except Exception as e:
                row = {**p,
                       "thematic_alignment": "", "shared_methodology": "",
                       "justification": "",
                       "error": f"{type(e).__name__}: {e}"}
            w.writerow(row); f.flush()
            n_done += 1
            if n_done % 20 == 0 or i == len(pairs):
                log(f"  {n_done}/{len(pairs)} rated")

    log(f"Wrote {out_csv}")

    # === Aggregate per flow ===
    rows = list(csv.DictReader(open(out_csv)))
    rows = [r for r in rows if r.get("thematic_alignment") and not r.get("error")]
    log(f"Valid ratings: {len(rows)}")

    by_flow: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in rows:
        by_flow[(int(r["capstone_topic"]), int(r["journal_topic"]))].append(int(r["thematic_alignment"]))
    flow_means = {k: float(np.mean(v)) for k, v in by_flow.items()}
    flow_maxes = {k: int(np.max(v)) for k, v in by_flow.items()}

    # === Compare against Method C (per-flow mean of pair ratings in C) ===
    cap_topic_of: dict[str, int] = {}
    with open(CAP_DOCS) as f:
        for r in csv.DictReader(f):
            cap_topic_of[r["capstone_id"]] = int(r["topic"])
    jrn_topic_of: dict[str, int] = {}
    with open(JRN_DOCS) as f:
        for r in csv.DictReader(f):
            jrn_topic_of[r["pmid"]] = int(r["topic"])

    c_by_flow: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in csv.DictReader(open(METHOD_C_CSV)):
        if r["error"] or not r["thematic_alignment"]:
            continue
        ct = cap_topic_of.get(r["capstone_id"])
        jt = jrn_topic_of.get(r["pmid"])
        if ct is None or jt is None or ct == -1 or jt == -1:
            continue
        c_by_flow[(ct, jt)].append(int(r["thematic_alignment"]))
    c_means = {k: float(np.mean(v)) for k, v in c_by_flow.items() if v}

    # === Build comparison ===
    val_rows = []
    for k, d_mean in flow_means.items():
        cm = c_means.get(k)
        val_rows.append({
            "capstone_topic": k[0], "journal_topic": k[1],
            "method_d_pair_agg_mean": round(d_mean, 4),
            "method_d_pair_agg_max": flow_maxes[k],
            "method_c_pair_mean": round(cm, 4) if cm is not None else None,
            "n_c_pairs": len(c_by_flow.get(k, [])),
        })
    out_val = OUT_DIR / "vs_method_c.csv"
    with open(out_val, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        w.writeheader(); w.writerows(val_rows)
    log(f"Wrote {out_val}")

    # === Correlation vs Method C means ===
    matched = [r for r in val_rows if r["method_c_pair_mean"] is not None]
    if len(matched) >= 3:
        d = np.array([r["method_d_pair_agg_mean"] for r in matched])
        c = np.array([r["method_c_pair_mean"] for r in matched])
        sp = spearmanr(d, c); pe = pearsonr(d, c)
        log(f"\n=== Pair-aggregate Method D vs Method C (across {len(matched)} flows) ===")
        log(f"  Spearman ρ = {sp.statistic:.3f}  Pearson r = {pe.statistic:.3f}  (p={sp.pvalue:.3g})")
        log(f"  Pair-agg D mean: {d.mean():.3f}, Method C mean: {c.mean():.3f}")

        # Save report
        main_d = json.loads(MAIN_D_REPORT.read_text())
        report = {
            "n_pair_calls": len(rows),
            "n_flows_evaluated": len(by_flow),
            "n_flows_matched_with_method_c": len(matched),
            "spearman_vs_method_c": float(sp.statistic),
            "pearson_vs_method_c": float(pe.statistic),
            "method_d_pair_agg_mean": float(d.mean()),
            "method_c_mean_of_means": float(c.mean()),
            "comparison_with_main_method_d": {
                "main_spearman": main_d["spearman"],
                "main_pearson": main_d["pearson"],
                "pair_agg_spearman": float(sp.statistic),
                "pair_agg_pearson": float(pe.statistic),
                "spearman_delta": float(sp.statistic - main_d["spearman"]),
                "pearson_delta": float(pe.statistic - main_d["pearson"]),
            },
        }
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
        log(f"Wrote {OUT_DIR / 'report.json'}")

        log(f"\nComparison:")
        log(f"  Main Method D (1 holistic call per flow):  Spearman {main_d['spearman']:.3f}  Pearson {main_d['pearson']:.3f}")
        log(f"  Pair-aggregate Method D (9 calls per flow): Spearman {sp.statistic:.3f}  Pearson {pe.statistic:.3f}")
        log(f"  Δ Spearman = {sp.statistic - main_d['spearman']:+.3f}")
        log(f"  Δ Pearson  = {pe.statistic - main_d['pearson']:+.3f}")

        # Scatter plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        for r in matched:
            ax.scatter(r["method_c_pair_mean"], r["method_d_pair_agg_mean"], s=60, alpha=0.75)
            ax.annotate(f"T{r['capstone_topic']}->T{r['journal_topic']}",
                        (r["method_c_pair_mean"], r["method_d_pair_agg_mean"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        lo = min(d.min(), c.min()) - 0.2
        hi = max(d.max(), c.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
        ax.set_xlabel("Method C mean pair rating within flow")
        ax.set_ylabel("Method D pair-aggregate mean rating within flow")
        ax.set_title(f"Pair-aggregate Method D vs Method C — Spearman ρ = {sp.statistic:.3f}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "vs_method_c.png", dpi=120)
        plt.close(fig)
        log(f"Wrote {OUT_DIR / 'vs_method_c.png'}")


if __name__ == "__main__":
    main()
