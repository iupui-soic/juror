"""Derive the releasable *_public topic tables from the full BERTopic outputs.

The full tables carry Representative_Docs (capstone text, IRB-restricted), and
their Name/Representation columns can carry PERSON or ORG surfaces that survived
Stage-0 redaction of the source documents: BERTopic's c-TF-IDF will happily
promote a supervisor's surname to a topic keyword, and it lower-cases it on the
way, which is how such a name can pass a case-sensitive release check. This
script is the committed derivation, so the public tables can be regenerated and
re-checked instead of being maintained by hand.

Both name lists are read from files that are never committed, so this script
names no person and no organisation:

    local/person_markers.txt   one PERSON surface per line
    local/org_markers.txt      one ORG surface per line

Lists are explicit and human-reviewed rather than inferred from the redaction
log: spaCy tagged phrases such as a database-deployment title as PERSON, and
auto-deriving from those mis-tags redacts domain vocabulary the tables need.
Matching is case-insensitive with letter boundaries rather than \\b, because
BERTopic joins Name tokens with underscores and \\b treats "_" as a word
character. ORG patterns are applied before PERSON so an institute name is not
mislabelled.

    python3 bhi2026/make_public_topic_tables.py           # rewrite both tables
    python3 bhi2026/make_public_topic_tables.py --check   # report only; exit 1 if stale
"""
import csv, io, re, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent
PERSONS = ROOT / "local/person_markers.txt"
ORGS = ROOT / "local/org_markers.txt"
TABLES = [("bhi2026/phase3_method_a/capstone_topics.csv",
           "bhi2026/phase3_method_a/capstone_topics_public.csv"),
          ("bhi2026/phase4_pubmedbert/capstone_topics.csv",
           "bhi2026/phase4_pubmedbert/capstone_topics_public.csv")]
KEEP_COLS = ["Topic", "Count", "Name", "Representation"]


def markers(path):
    if not path.exists():
        return []
    seen = {" ".join(l.split()) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    return sorted(seen, key=len, reverse=True)          # longest surface first


def build_patterns():
    def pat(surface):
        body = r"[\s_]*".join(map(re.escape, surface.split()))
        return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE)
    return ([(pat(m), "[ORG]") for m in markers(ORGS)] +
            [(pat(m), "[PERSON]") for m in markers(PERSONS)])


def scrub(text, pats, collapse=False):
    """Redact in place. Representation keeps one slot per original keyword, so a
    released top-10 list stays ten entries long; only Name collapses adjacent
    duplicates, where "x_[PERSON]_[PERSON]_y" would just be noise."""
    for p, repl in pats:
        text = p.sub(repl, text)
    if collapse:
        text = re.sub(r"(\[PERSON\]|\[ORG\])(_\1)+", r"\1", text)
    return text


def main():
    check = "--check" in sys.argv
    pats = build_patterns()
    if not pats:
        print("ERROR: no marker files found; run on the machine holding local/", file=sys.stderr)
        return 1
    stale = False
    for src_rel, dst_rel in TABLES:
        src, dst = ROOT / src_rel, ROOT / dst_rel
        if not src.exists():
            print(f"SKIP {dst_rel}: {src.name} absent (restricted corpus required)")
            continue
        rows, n = [], 0
        for r in csv.DictReader(open(src, encoding="utf-8")):
            row = {c: r.get(c, "") for c in KEEP_COLS}
            for c in ("Name", "Representation"):
                s = scrub(row[c], pats, collapse=(c == "Name"))
                n += s != row[c]
                row[c] = s
            rows.append(row)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=KEEP_COLS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
        new = buf.getvalue()
        old = dst.read_text(encoding="utf-8") if dst.exists() else ""
        if new == old:
            print(f"{dst_rel}: {len(rows)} topics, {n} field(s) redacted -> unchanged")
        else:
            stale = True
            if not check:
                dst.write_text(new, encoding="utf-8")
            print(f"{dst_rel}: {len(rows)} topics, {n} field(s) redacted -> "
                  f"{'would rewrite' if check else 'REWRITTEN'}")
    if check and stale:
        print("\n--check: public tables are stale with respect to the marker lists")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
