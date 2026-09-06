"""Reproducible builder for the 100-pair expert alignment benchmark.

Deterministically (seed = 42) samples capstone-journal pairs from the Method C
rating pool into three buckets, shuffles them, and emits a blank rating
scaffold. Journal records with no PubMed abstract (errata, editorials, letters,
comments) are excluded, because the LLM-judge being calibrated operates on
abstract text.

  25 aligned    : highest Method-C-rated pairs (rating desc, then cosine desc)
  25 misaligned : Method C rating 0 with cosine < 0.55 (lowest cosine first)
  50 ambiguous  : remaining pairs, stratified across cosine quartiles

Row order is shuffled and the bucket label is written only to the internal
"Pair Summary" sheet, so a rater cannot infer the expected rating from position.

Outputs (under ./rebuild/, so the released benchmark_pairs.xlsx is never
overwritten):
  rebuild/benchmark_pairs.xlsx  -- Instructions / Pair Summary / Rater 1 / Rater 2 / Adjudication
  rebuild/pairs/pair_NNN_{capstone,journal}.txt

The benchmark actually rated for the paper is the curated benchmark_pairs.xlsx
in this folder; this script documents and regenerates the construction
procedure. The capstone corpus is IRB-restricted, so capstone excerpts populate
only when the source texts are present locally.

Run from anywhere: python3 build_benchmark.py
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 7)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # .../B584_B691_PPTs
METHOD_C = REPO / "bhi2026/phase3_method_c/ratings.csv"
JOURNAL_CSV = REPO / "bhi2026/journals/journal_corpus.csv"
CAP_META = REPO / "bhi2026/metadata/corpus_metadata.csv"
TXT_ROOT = REPO / "raw_data/OneDrive_extracted_text/texts"

OUT_DIR = HERE / "rebuild"
PAIRS_DIR = OUT_DIR / "pairs"

SEED = 42
N_ALIGNED, N_MISALIGNED, N_TOTAL = 25, 25, 100
CAP_CHARS, JRN_CHARS = 3500, 3000


def load_journal_meta():
    out = {}
    with open(JOURNAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[str(r["pmid"])] = {
                "title": r.get("title", ""), "abstract": r.get("abstract", ""),
                "journal_key": r.get("journal_key", ""), "year": r.get("year", ""),
                "doi": r.get("doi", ""),
            }
    return out


def load_capstone_texts():
    out = {}
    if not CAP_META.exists():
        return out
    with open(CAP_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = TXT_ROOT / (r["source_path"] + ".txt")
            out[r["capstone_id"]] = p.read_text(encoding="utf-8") if p.exists() else ""
    return out


def sample_pairs():
    jrn = load_journal_meta()
    has_abs = {p for p, m in jrn.items() if (m.get("abstract") or "").strip()}
    rows = []
    with open(METHOD_C, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["error"] or str(r["pmid"]) not in has_abs:
                continue
            r["rating_int"] = int(r["thematic_alignment"])
            r["cosine_f"] = float(r["cosine_sim"])
            rows.append(r)

    rng = random.Random(SEED)
    chosen, picks = set(), []

    def take(r, bucket):
        key = (r["capstone_id"], str(r["pmid"]))
        if key in chosen:
            return False
        chosen.add(key)
        picks.append({"bucket": bucket, "capstone_id": r["capstone_id"],
                      "pmid": str(r["pmid"]), "cosine_sim": r["cosine_f"],
                      "_method_c_rating": r["rating_int"]})
        return True

    # 25 aligned: highest rating, then highest cosine
    for r in sorted(rows, key=lambda r: (-r["rating_int"], -r["cosine_f"])):
        if sum(p["bucket"] == "aligned" for p in picks) >= N_ALIGNED:
            break
        take(r, "aligned")

    # 25 misaligned: rating 0, cosine < 0.55, lowest cosine first
    for r in sorted((r for r in rows if r["rating_int"] == 0 and r["cosine_f"] < 0.55),
                    key=lambda r: r["cosine_f"]):
        if sum(p["bucket"] == "misaligned" for p in picks) >= N_MISALIGNED:
            break
        take(r, "misaligned")

    # 50 ambiguous: remaining pairs stratified across cosine quartiles
    n_amb = N_TOTAL - len(picks)
    rest = [r for r in rows if (r["capstone_id"], str(r["pmid"])) not in chosen]
    rest.sort(key=lambda r: r["cosine_f"])
    q = max(1, len(rest) // 4)
    quartiles = [rest[0:q], rest[q:2 * q], rest[2 * q:3 * q], rest[3 * q:]]
    for ql in quartiles:
        rng.shuffle(ql)
    per = n_amb // 4
    for ql in quartiles:
        for r in ql[:per]:
            take(r, "ambiguous")
    leftover = [r for ql in quartiles for r in ql
                if (r["capstone_id"], str(r["pmid"])) not in chosen]
    rng.shuffle(leftover)
    for r in leftover:
        if len(picks) >= N_TOTAL:
            break
        take(r, "ambiguous")

    assert len(picks) == N_TOTAL, f"got {len(picks)} pairs"
    assert all(str(p["pmid"]) in has_abs for p in picks), "abstract-less pair sampled"

    rng.shuffle(picks)
    for i, p in enumerate(picks, start=1):
        p["pair_id"] = f"pair_{i:03d}"
    return picks, jrn


def write_workbook(picks, jrn, cap_texts):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    ws.append(["BHI 2026 expert alignment benchmark"])
    ws.append(["Rate THEMATIC ALIGNMENT 0-4 (col F) + methodology overlap "
               "yes/partial/no (col G). Full rubric in INSTRUCTIONS.md."])

    summ = wb.create_sheet("Pair Summary")
    summ.append(["pair_id", "_internal_bucket", "capstone_id", "pmid",
                 "journal_key", "year", "cosine_sim", "_method_c_rating"])
    for p in picks:
        m = jrn.get(p["pmid"], {})
        summ.append([p["pair_id"], p["bucket"], p["capstone_id"], p["pmid"],
                     m.get("journal_key", ""), m.get("year", ""),
                     round(p["cosine_sim"], 4), p["_method_c_rating"]])

    def rater_sheet(name):
        ws = wb.create_sheet(name)
        ws.append(["pair_id", "capstone_excerpt", "journal_title", "journal_abstract",
                   "cosine_sim_hint", "rating (0-4)", "methodology (yes/partial/no)", "notes"])
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="D9E1F2")
        dv_r = DataValidation(type="list", formula1='"0,1,2,3,4"', allow_blank=True)
        dv_m = DataValidation(type="list", formula1='"yes,partial,no"', allow_blank=True)
        ws.add_data_validation(dv_r)
        ws.add_data_validation(dv_m)
        for i, p in enumerate(picks, start=2):
            m = jrn.get(p["pmid"], {})
            ws.append([p["pair_id"], cap_texts.get(p["capstone_id"], "")[:CAP_CHARS],
                       m.get("title", ""), (m.get("abstract") or "")[:JRN_CHARS],
                       round(p["cosine_sim"], 3), None, None, None])
            dv_r.add(f"F{i}")
            dv_m.add(f"G{i}")
            for c in ws[i]:
                c.alignment = Alignment(vertical="top", wrap_text=True)
        ws.freeze_panes = "A2"

    rater_sheet("Rater 1")
    rater_sheet("Rater 2")

    adj = wb.create_sheet("Adjudication")
    adj.append(["pair_id", "_internal_bucket", "rater_1_rating", "rater_1_methodology",
                "rater_2_rating", "rater_2_methodology",
                "adjudicated_rating", "adjudicated_methodology", "notes"])
    for c in adj[1]:
        c.font = Font(bold=True)
    for p in picks:
        adj.append([p["pair_id"], p["bucket"], None, None, None, None, None, None, None])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_DIR / "benchmark_pairs.xlsx")


def write_pairs(picks, jrn, cap_texts):
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    for p in picks:
        pid = p["pair_id"]
        m = jrn.get(p["pmid"], {})
        (PAIRS_DIR / f"{pid}_capstone.txt").write_text(
            f"=== {pid} - Capstone ===\n\n"
            + cap_texts.get(p["capstone_id"], "")[:CAP_CHARS], encoding="utf-8")
        (PAIRS_DIR / f"{pid}_journal.txt").write_text(
            f"=== {pid} - Journal abstract ===\nPMID: {p['pmid']}\n"
            f"Journal: {m.get('journal_key')} ({m.get('year')})\n\n"
            f"Title: {m.get('title','')}\n\nAbstract:\n"
            f"{(m.get('abstract') or '')[:JRN_CHARS]}\n", encoding="utf-8")


def main():
    if not METHOD_C.exists():
        sys.exit(f"missing input: {METHOD_C}")
    picks, jrn = sample_pairs()
    cap_texts = load_capstone_texts()
    n_cap = sum(1 for p in picks if cap_texts.get(p["capstone_id"], "").strip())
    write_workbook(picks, jrn, cap_texts)
    write_pairs(picks, jrn, cap_texts)
    counts = defaultdict(int)
    for p in picks:
        counts[p["bucket"]] += 1
    print(f"Wrote {OUT_DIR/'benchmark_pairs.xlsx'}  ({dict(counts)})")
    print(f"Capstone excerpts populated for {n_cap}/100 pairs "
          f"({'IRB-restricted texts present' if n_cap else 'texts absent — ids only'}).")


if __name__ == "__main__":
    main()
