"""Build corpus_metadata.csv for the 288-capstone corpus.

Columns: capstone_id, year, semester, source_path, ext, word_count,
has_speaker_notes, sections_present.

- capstone_id: zero-padded 4-digit ID (cap_0001 ... cap_0288), assigned in
  deterministic sort order of source_path so the IDs are stable across reruns.
- year, semester: parsed from the OneDrive folder name (e.g. "SP24 ..." -> 2024,
  "Spring"). Fall=20YY semester maps to Aug-Dec, Spring=Jan-May, Summer=May-Aug.
- ext: "pdf" or "pptx".
- word_count: from the already-extracted .txt file (whitespace split).
- has_speaker_notes: whether the source .pptx has any non-empty notes_slide
  (only meaningful for pptx; False for pdf).
- sections_present: pipe-separated list of section headings detected in the
  extracted text (see SECTION_PATTERNS). Empty string if none.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pptx import Presentation

SRC_ROOT = Path("raw_data/OneDrive_extracted/Practicum and capstone work")
TXT_ROOT = Path("raw_data/OneDrive_extracted_text/texts")
OUT_DIR = Path("bhi2026/metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEMESTER_MAP = {
    "FA": "Fall",
    "SP": "Spring",
    "SU": "Summer",
}

SECTION_PATTERNS = {
    "Introduction": re.compile(r"\bintroduction\b", re.I),
    "Practicum Scope": re.compile(r"\bpracticum\s+scope\b", re.I),
    "Learning Objectives": re.compile(r"\b(learning\s+objectives|practicum\s+outcomes\s+learning\s+objectives)\b", re.I),
    "Practicum Duties": re.compile(r"\bpracticum\s+duties\b", re.I),
    "Practicum Outcomes": re.compile(r"\bpracticum\s+outcomes\b", re.I),
    "Conclusion": re.compile(r"\bconclusion[s]?\b", re.I),
    "Methods": re.compile(r"\b(methods|methodology)\b", re.I),
    "Results": re.compile(r"\bresults\b", re.I),
    "References": re.compile(r"\breferences\b", re.I),
}


def parse_semester_folder(folder_name: str) -> tuple[int, str]:
    """Return (year, semester_long) parsed from e.g. 'SP24 584 &Capstone'."""
    m = re.match(r"^(FA|SP|SU)\s*(\d{2})", folder_name)
    if not m:
        return (0, "Unknown")
    short, yy = m.groups()
    year = 2000 + int(yy)
    return (year, SEMESTER_MAP[short])


def has_pptx_notes(path: Path) -> bool:
    try:
        prs = Presentation(path)
    except Exception:
        return False
    for slide in prs.slides:
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            return True
    return False


def detect_sections(text: str) -> list[str]:
    return [name for name, pat in SECTION_PATTERNS.items() if pat.search(text)]


def main() -> None:
    sources = sorted(
        p for p in SRC_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".pptx"}
    )
    assert len(sources) == 288, f"Expected 288 source files, found {len(sources)}"

    rows: list[dict] = []
    for i, src in enumerate(sources, start=1):
        rel = src.relative_to(SRC_ROOT)
        semester_folder = rel.parts[0]
        year, semester = parse_semester_folder(semester_folder)

        txt_path = TXT_ROOT / (rel.as_posix() + ".txt")
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        word_count = len(text.split())
        sections = detect_sections(text)

        ext = src.suffix.lower().lstrip(".")
        notes_flag = has_pptx_notes(src) if ext == "pptx" else False

        rows.append({
            "capstone_id": f"cap_{i:04d}",
            "year": year,
            "semester": semester,
            "semester_short": semester_folder.split()[0],
            "source_path": rel.as_posix(),
            "ext": ext,
            "word_count": word_count,
            "has_speaker_notes": notes_flag,
            "sections_present": "|".join(sections),
        })

    out_path = OUT_DIR / "corpus_metadata.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"Wrote {out_path} ({len(rows)} rows)")
    sem_counts = {}
    for r in rows:
        key = f"{r['year']} {r['semester']}"
        sem_counts[key] = sem_counts.get(key, 0) + 1
    print("\nPer-semester counts:")
    for key in sorted(sem_counts):
        print(f"  {key}: {sem_counts[key]}")

    ext_counts = {}
    for r in rows:
        ext_counts[r["ext"]] = ext_counts.get(r["ext"], 0) + 1
    print(f"\nFile-type counts: {ext_counts}")

    section_hits = sum(1 for r in rows if r["sections_present"])
    print(f"\nDocuments with at least one detected section: {section_hits}/{len(rows)}")

    notes_count = sum(1 for r in rows if r["has_speaker_notes"])
    pptx_count = sum(1 for r in rows if r["ext"] == "pptx")
    print(f"PPTX with speaker notes: {notes_count}/{pptx_count}")

    wc_sorted = sorted(r["word_count"] for r in rows)
    median = wc_sorted[len(wc_sorted) // 2]
    print(f"\nWord-count: min={wc_sorted[0]}, median={median}, max={wc_sorted[-1]}")
    print(f"Documents with < 50 words: {sum(1 for w in wc_sorted if w < 50)}")


if __name__ == "__main__":
    main()
