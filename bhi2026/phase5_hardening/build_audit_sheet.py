"""
Build the expert FAITHFULNESS + DIRECT-RATING audit sheet for the JUROR flows:
the 24 top-3 flows the paper rates, plus, for every capstone topic, the flow at
transport rank 4 (the first beyond the pre-specified cut) and the BTM-selected
C7-J22 — 33 rows, ordered by capstone topic then rank so no row stands out.
Pass --main-only to build the original 24-row sheet.

All evidence text and JUROR justifications pass through phase6_robustness/redact_surfaces
(curated PERSON/ORG markers) before they are written.

What this measures (two things, on the SAME topic-flows):
  (A) A DIRECT human rating of each flow on the 0-4 rubric, blind to JUROR ->
      a one-step human->JUROR anchor (the benchmark only anchored the pairwise
      judge, not JUROR).
  (B) Whether each JUROR justification FAITHFULLY describes the two topics, or
      hallucinates overlap -> measured explanation quality.

Evidence shown to the rater = the BEST available characterization of each topic:
  - full top-10 BERTopic keywords (the Representation column), and
  - BERTopic's 3 Representative_Documents (the most central docs of the topic),
    cleaned and truncated at a sentence boundary (no mid-word cut-offs).

Output: faithfulness_audit_sheet.xlsx  (0_README / 1_Rate_blind / 2_Audit_justification)
Run from phase5_hardening/.
"""
import csv, ast, sys
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

csv.field_size_limit(10**7)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6_robustness"))
from redact_surfaces import scrub
N_KW, N_DOCS, DOC_CHARS = 10, 3, 550
MAIN_ONLY = "--main-only" in sys.argv
EXTRA = [(7, 22)]                      # BTM-selected flow (Claude 2 / GPT 3)

# ---- topic representations: full keywords + BERTopic representative docs ----
def load_topics(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try: t = int(r["Topic"])
        except (KeyError, ValueError): continue
        try: kws = ", ".join(ast.literal_eval(r["Representation"])[:N_KW])
        except Exception: kws = r.get("Name", "")
        try: docs = [d for d in ast.literal_eval(r["Representative_Docs"]) if d.strip()]
        except Exception: docs = []
        out[t] = dict(kw=scrub(kws), docs=[scrub(d) for d in docs[:N_DOCS]])
    return out

capT = load_topics(ROOT / "phase3_method_a/capstone_topics.csv")
jrnT = load_topics(ROOT / "phase3_method_a/journal_topics.csv")

def clean(t, limit=DOC_CHARS):
    t = " ".join((t or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    for sep in (". ", "? ", "! "):
        i = cut.rfind(sep)
        if i > limit * 0.5:
            return cut[:i + 1] + " […]"
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut) + " […]"

def evidence_block(rep):
    lines = [f"KEYWORDS:  {rep['kw']}", "", "REPRESENTATIVE DOCUMENTS:"]
    if not rep["docs"]:
        lines.append("  (none available)")
    for i, d in enumerate(rep["docs"], 1):
        lines.append(f"[{i}]  {clean(d)}")
        lines.append("")
    return "\n".join(lines).rstrip()

# ---- flows + JUROR output (primary rater) ----
def make_flow(r, rank):
    ct, jt = int(r["capstone_topic"]), int(r["journal_topic"])
    return dict(
        ct=ct, jt=jt, rank=rank,
        rating=r["thematic_alignment"], method=r["shared_methodology"], just=scrub(r["justification"]),
        cap_kw=capT.get(ct, {}).get("kw", "?"), jrn_kw=jrnT.get(jt, {}).get("kw", "?"),
        cap_block=evidence_block(capT.get(ct, {"kw": "?", "docs": []})),
        jrn_block=evidence_block(jrnT.get(jt, {"kw": "?", "docs": []})))

flows = [make_flow(r, int(r["rank"]))
         for r in csv.DictReader(open(ROOT / "phase3_method_d/flow_ratings.csv"))]
if not MAIN_ONLY:
    have = {(f["ct"], f["jt"]) for f in flows}
    for r in csv.DictReader(open(ROOT / "phase6_robustness/out/r9_topk_ratings.csv", encoding="utf-8")):
        k = (int(r["capstone_topic"]), int(r["journal_topic"]))
        if r["rank"] == "4" and r["thematic_alignment"] not in ("", None) and k not in have:
            flows.append(make_flow(r, 4)); have.add(k)
    for r in csv.DictReader(open(ROOT / "phase6_robustness/out/r5_new_flow_ratings_claude.csv", encoding="utf-8")):
        k = (int(r["capstone_topic"]), int(r["journal_topic"]))
        if k in EXTRA and r["thematic_alignment"] not in ("", None) and k not in have:
            flows.append(make_flow(r, 5)); have.add(k)
    missing = [k for k in EXTRA if k not in have]
    assert not missing, f"extra flows not found in rating files: {missing}"
flows.sort(key=lambda f: (f["ct"], f["rank"], f["jt"]))

# =====================================================================
wb = openpyxl.Workbook()
H = Font(bold=True, size=11)
WRAP = Alignment(wrap_text=True, vertical="top")
WRAPC = Alignment(wrap_text=True, vertical="top", horizontal="center")
HEADERFILL = PatternFill("solid", fgColor="DDEBF7")
INPUTFILL = PatternFill("solid", fgColor="FFF2CC")
CAPFILL = PatternFill("solid", fgColor="EAF3EA")
JRNFILL = PatternFill("solid", fgColor="FBEFE6")
thin = Side(style="thin", color="BBBBBB")
BORD = Border(thin, thin, thin, thin)

def style_header(ws, ncol, h=34):
    for c in range(1, ncol + 1):
        cell = ws.cell(1, c); cell.font = H; cell.fill = HEADERFILL
        cell.alignment = WRAPC; cell.border = BORD
    ws.row_dimensions[1].height = h
    ws.freeze_panes = "B2"

# ---------- Sheet 0 — README / context ----------
rd = wb.active; rd.title = "0_README"
blocks = [
    ("JUROR faithfulness & direct-rating audit", "title"),
    ("", "n"),
    ("WHAT YOU ARE LOOKING AT.", "h"),
    ("JUROR is our method for comparing two text collections — here, the MS-HI capstone corpus and the HI", "n"),
    ("journal literature. It first clusters each corpus into TOPICS, then uses optimal transport to find the", "n"),
    ("strongest topic-to-topic links. Each such link is a FLOW. This sheet contains 33 flows: for each of the 8", "n"),
    ("capstone topics, its top-3 transport flows and the next-ranked one, plus one flow chosen by a comparison", "n"),
    ("method. Please treat every row the same way. This packet replaces the earlier 24-row version.", "n"),
    ("A flow is a pair of TOPICS (e.g. capstone-topic 'EMR/database' ↔ journal-topic 'large language models'),", "n"),
    ("NOT a pair of individual documents.", "n"),
    ("", "n"),
    ("WHY WE NEED YOU (two separate jobs).", "h"),
    ("1) Sheet '1_Rate_blind': read the two topics for each flow and give YOUR OWN 0–4 rating of how aligned", "n"),
    ("   they are. This gives JUROR a direct human comparison at the flow level.", "n"),
    ("2) Sheet '2_Audit_justification': we then show you JUROR's rating AND the short justification it wrote.", "n"),
    ("   You judge whether that justification TRUTHFULLY describes the two topics, or whether it overstates /", "n"),
    ("   invents overlap that isn't actually there ('hallucination').", "n"),
    ("Please complete Sheet 1 BEFORE looking at Sheet 2 — seeing JUROR's answer first would bias your rating.", "n"),
    ("", "n"),
    ("HOW THIS RELATES TO THE 100-PAIR REVIEW YOU DID BEFORE.", "h"),
    ("It is a DIFFERENT task. The earlier review scored 100 capstone–journal DOCUMENT PAIRS and was used to", "n"),
    ("validate our pairwise LLM judge. This audit scores 33 TOPIC-level FLOWS, and adds a second question the", "n"),
    ("first review did not ask: is JUROR's written explanation faithful? Same 0–4 rubric, same kind of domain", "n"),
    ("judgement — different unit (topics, not document pairs) and an added faithfulness check.", "n"),
    ("", "n"),
    ("WHO SHOULD FILL THIS IN.", "h"),
    ("Any HI domain expert comfortable with the 0–4 rubric. One rater is enough to produce the results; two", "n"),
    ("raters (ideally the same experts from the 100-pair review, since they already know the rubric) let us", "n"),
    ("also report agreement on the audit. It does NOT have to be both original raters. ~60–120 min for 33 flows.", "n"),
    ("", "n"),
    ("THE EVIDENCE SHOWN.", "h"),
    ("For each topic you see its top-10 keywords and its 3 most representative documents (lightly trimmed to", "n"),
    ("the first few sentences). That is the best available description of what the topic is about.", "n"),
    ("", "n"),
    ("RATING RUBRIC (0–4) — identical to the one you used before:", "h"),
    ("  0 — Unrelated. Completely different problems and methods.", "n"),
    ("  1 — Tangentially related. Share a broad domain (both healthcare) but different question AND method.", "n"),
    ("  2 — Adjacent. Share EITHER the research question OR the method, but not both.", "n"),
    ("  3 — Substantially overlapping. Same research question + overlapping methods; work-products differ.", "n"),
    ("  4 — Same research question with comparable methods.", "n"),
    ("  SHARED METHODOLOGY: yes / partial / no.", "n"),
    ("", "n"),
    ("FAITHFULNESS COLUMNS (Sheet 2) — score each 2 / 1 / 0:", "h"),
    ("  faith_capstone_desc — does JUROR's description of the CAPSTONE topic match its keywords/documents?", "n"),
    ("  faith_journal_desc  — same for the JOURNAL topic.", "n"),
    ("  faith_relationship  — is the claimed relationship (shared method? divergent question?) supported?", "n"),
    ("       2 = fully accurate · 1 = partly · 0 = wrong/unsupported", "n"),
    ("  method_flag_correct_yn — is JUROR's shared-methodology flag (yes/partial/no) correct? (yes/no)", "n"),
    ("  hallucinated_overlap_yn — does the justification assert a shared entity/method/finding that is NOT in", "n"),
    ("       either topic? (yes/no). THIS IS THE KEY FAILURE MODE — plausible-sounding overlap with no support.", "n"),
    ("  notes — anything notable, especially for any 'yes' hallucination or any 0/1 score.", "n"),
    ("", "n"),
    ("Yellow cells are for your input. When done, save and return the file.", "n"),
]
fonts = {"title": Font(bold=True, size=14), "h": Font(bold=True, size=11, color="1F4E79")}
for i, (txt, kind) in enumerate(blocks, 1):
    c = rd.cell(i, 1, txt)
    if kind in fonts: c.font = fonts[kind]
rd.column_dimensions["A"].width = 116

# ---------- Sheet 1 — blind rating ----------
s1 = wb.create_sheet("1_Rate_blind")
cols1 = ["flow_id", "CAPSTONE topic (keywords + representative docs)",
         "JOURNAL topic (keywords + representative docs)",
         "your_rating_0to4", "your_shared_methodology", "notes"]
s1.append(cols1)
for fl in flows:
    s1.append([f"C{fl['ct']}-J{fl['jt']}", fl["cap_block"], fl["jrn_block"], "", "", ""])
style_header(s1, len(cols1), h=46)
for i, w in enumerate([10, 64, 64, 13, 16, 26], 1):
    s1.column_dimensions[get_column_letter(i)].width = w
for r in range(2, len(flows) + 2):
    s1.row_dimensions[r].height = 230
    for c in range(1, len(cols1) + 1):
        s1.cell(r, c).alignment = WRAP; s1.cell(r, c).border = BORD
    s1.cell(r, 2).fill = CAPFILL; s1.cell(r, 3).fill = JRNFILL
    s1.cell(r, 1).alignment = WRAPC
    for c in (4, 5, 6): s1.cell(r, c).fill = INPUTFILL

# Data validation dropdowns
from openpyxl.worksheet.datavalidation import DataValidation
dv_rate = DataValidation(type="list", formula1='"0,1,2,3,4"', allow_blank=True)
dv_meth = DataValidation(type="list", formula1='"yes,partial,no"', allow_blank=True)
s1.add_data_validation(dv_rate); s1.add_data_validation(dv_meth)
dv_rate.add(f"D2:D{len(flows)+1}"); dv_meth.add(f"E2:E{len(flows)+1}")

# ---------- Sheet 2 — faithfulness audit ----------
s2 = wb.create_sheet("2_Audit_justification")
cols2 = ["flow_id", "capstone keywords", "journal keywords",
         "JUROR_rating", "JUROR_shared_methodology", "JUROR_justification",
         "faith_capstone_desc_210", "faith_journal_desc_210", "faith_relationship_210",
         "method_flag_correct_yn", "hallucinated_overlap_yn", "notes"]
s2.append(cols2)
for fl in flows:
    s2.append([f"C{fl['ct']}-J{fl['jt']}", fl["cap_kw"], fl["jrn_kw"],
               fl["rating"], fl["method"], fl["just"], "", "", "", "", "", ""])
style_header(s2, len(cols2), h=46)
for i, w in enumerate([10, 26, 26, 9, 13, 58, 12, 12, 12, 13, 14, 26], 1):
    s2.column_dimensions[get_column_letter(i)].width = w
for r in range(2, len(flows) + 2):
    s2.row_dimensions[r].height = 120
    for c in range(1, len(cols2) + 1):
        s2.cell(r, c).alignment = WRAP; s2.cell(r, c).border = BORD
    s2.cell(r, 1).alignment = WRAPC
    for c in range(7, 13): s2.cell(r, c).fill = INPUTFILL
dv210 = DataValidation(type="list", formula1='"2,1,0"', allow_blank=True)
dvyn = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
s2.add_data_validation(dv210); s2.add_data_validation(dvyn)
for col in ("G", "H", "I"): dv210.add(f"{col}2:{col}{len(flows)+1}")
for col in ("J", "K"): dvyn.add(f"{col}2:{col}{len(flows)+1}")

out = Path(__file__).resolve().parent / "faithfulness_audit_sheet.xlsx"
wb.save(out)
print(f"[wrote {out.name}] — {len(flows)} flows" + (" (main 24 only)" if MAIN_ONLY else " (24 paper + 9 supplement)"))
print("Sheets: 0_README / 1_Rate_blind (do first, blind) / 2_Audit_justification")
print("Evidence = full top-10 keywords + 3 BERTopic representative docs, sentence-trimmed.")
print("Dropdowns added for all rating/score cells.")
